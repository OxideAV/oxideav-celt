//! The reference-exact §4.3.3 bit-allocation walk (RFC 6716) — the
//! codepoint search, the 1/64-step interpolation, the concurrent skip
//! decoding, the intensity / dual-stereo field placement, the
//! fine-energy vs. shape split, the final reallocation, and the
//! priority 0/1 stamp — plus the §4.3.4.1 bits↔pulses cache queries.
//!
//! ## Provenance
//!
//! Transcribed from the **normative RFC 6716 Appendix A reference
//! listing** (`rate.c` / `rate.h` and the static mode data), extracted
//! from the staged `docs/audio/opus/rfc6716-opus.txt` per §A.1 and
//! SHA-1-verified against the value §A.1 prints
//! (`86a927223e73d2476646a1b933fcd3fffb6ecc8c`). RFC 6716 §6 makes the
//! decoder side of that listing normative; §4.3.3 (lines 6111–6461)
//! is the prose narrative of the same walk. This module supersedes the
//! behavioral-chapter reconstruction in [`crate::realloc_walk`]: the
//! three constants that chapter's §10 left pinned only by
//! bit-exactness — the fine/shape split arithmetic, the skip
//! predicate, and the priority predicate — are all pinned by the
//! Appendix A listing and are implemented here exactly.
//!
//! Numeric tables cross-check against the staged extractions:
//! `docs/audio/opus/tables/log-n400.csv` ([`LOG_N400`]),
//! `band-allocation.csv` ([`crate::static_alloc::STATIC_ALLOC`]),
//! `cache-{index,bits}50.csv` ([`crate::pulse_cache`]), and
//! `log2-frac-table.csv` ([`crate::bit_allocation::LOG2_FRAC_TABLE`]).

use crate::band_layout::EBAND_EDGES_5MS;
use crate::bit_allocation::LOG2_FRAC_TABLE;
use crate::coarse_energy::NUM_BANDS;
use crate::pulse_cache::{CACHE_BITS50, CACHE_INDEX50};
use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::static_alloc::{NUM_Q, STATIC_ALLOC};
use crate::Error;

/// 1/8-bit resolution shift (RFC 6716 Appendix A `rate.h` `BITRES`).
pub const BITRES: u32 = 3;

/// Per-band fine-energy ceiling in whole bits per channel (Appendix A
/// `rate.h` `MAX_FINE_BITS`).
pub const MAX_FINE_BITS: i32 = 8;

/// Number of 1/64 interpolation bisection steps (Appendix A `rate.c`
/// `ALLOC_STEPS`).
pub const ALLOC_STEPS: u32 = 6;

/// `log2` of the maximum pseudo-pulse index (Appendix A `rate.h`
/// `LOG_MAX_PSEUDO`).
pub const LOG_MAX_PSEUDO: u32 = 6;

/// The per-band `logN` table of the 48 kHz mode (`logN400` in the
/// Appendix A static mode data; staged as
/// `docs/audio/opus/tables/log-n400.csv`): `log2` of each band's
/// LM = 0 width in 1/8-bit (Q3) units, used by the fine-split offset
/// and the split-resolution (`qtheta`) budget.
pub const LOG_N400: [i16; NUM_BANDS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36,
];

/// The §4.3.4.1 pseudo-pulse → pulse-count map (Appendix A `rate.h`
/// `get_pulses`): identity below 8, then `(8 + (i & 7)) << ((i >> 3) - 1)`.
#[inline]
pub fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// The pulse-cost cache row for `(band, lm)`, where `lm` is the
/// **split-adjusted** frame shift in `-1..=3` (the Appendix A callers
/// index row `lm + 1`). `None` on a sentinel row (never dereferenced
/// by legal traffic) or out-of-range arguments.
fn cache_row(band: usize, lm: i32) -> Option<&'static [u8]> {
    if band >= NUM_BANDS || !(-1..=3).contains(&lm) {
        return None;
    }
    let idx = CACHE_INDEX50[((lm + 1) as usize) * NUM_BANDS + band];
    if idx < 0 {
        return None;
    }
    let off = idx as usize;
    let max_pseudo = CACHE_BITS50[off] as usize;
    Some(&CACHE_BITS50[off..=off + max_pseudo])
}

/// The §4.3.4.1 bits → pseudo-pulses inversion (Appendix A `rate.h`
/// `bits2pulses`): a 6-step bisection over the cached cost curve
/// followed by a nearest-cost selection with ties **rounding down**.
/// `bits` is in 1/8-bit units. Returns the pseudo-pulse index
/// (`get_pulses` maps it to the actual pulse count `K`).
pub fn bits2pulses(band: usize, lm: i32, bits: i32) -> Option<i32> {
    let cache = cache_row(band, lm)?;
    let mut lo: i32 = 0;
    let mut hi: i32 = cache[0] as i32;
    let bits = bits - 1;
    for _ in 0..LOG_MAX_PSEUDO {
        let mid = (lo + hi + 1) >> 1;
        if cache[mid as usize] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_cost = if lo == 0 {
        -1
    } else {
        cache[lo as usize] as i32
    };
    if bits - lo_cost <= cache[hi as usize] as i32 - bits {
        Some(lo)
    } else {
        Some(hi)
    }
}

/// The §4.3.4.1 pseudo-pulses → bits cost (Appendix A `rate.h`
/// `pulses2bits`), in 1/8-bit units. Zero pseudo-pulses are free.
pub fn pulses2bits(band: usize, lm: i32, pulses: i32) -> Option<i32> {
    if pulses == 0 {
        return Some(0);
    }
    let cache = cache_row(band, lm)?;
    if pulses < 0 || pulses as usize >= cache.len() {
        return None;
    }
    Some(cache[pulses as usize] as i32 + 1)
}

/// Range-coder attachment for the walk. The decoder **reads** the
/// skip / intensity / dual-stereo symbols; the encoder **writes** its
/// own decisions at the identical bitstream positions.
#[allow(missing_debug_implementations)] // carries raw coder state
pub enum AllocIo<'a, 'b> {
    /// Decode side: symbols are read from the stream.
    Decode(&'a mut RangeDecoder<'b>),
    /// Encode side: symbols are written from the encoder's choices.
    Encode {
        /// The range encoder positioned right after the Table-56
        /// `alloc. trim` symbol.
        enc: &'a mut RangeEncoder,
        /// Desired first intensity-coded band (absolute index; clamped
        /// to the post-skip window). Only written when the intensity
        /// reservation was made.
        intensity: i32,
        /// Desired dual-stereo flag (only written when the dual
        /// reservation was made).
        dual_stereo: bool,
        /// The previous frame's `coded_bands` (the skip-choice
        /// hysteresis anchor; `0` on the first frame).
        prev_coded_bands: i32,
    },
}

/// The walk outputs — the four §4.3.3 results plus the decoded (or
/// echoed) stereo fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactAllocation {
    /// Per-band **shape** allocation in 1/8 bits (absolute band
    /// indexing, `[0; 21]` outside the coded window). This is the
    /// `pulses[]` output the §4.3.4 band loop prices against.
    pub shape_bits: [i32; NUM_BANDS],
    /// Per-band **fine-energy** allocation in whole bits per channel.
    pub fine_bits: [i32; NUM_BANDS],
    /// Per-band finalize priority (`false` = priority 0, `true` =
    /// priority 1; RFC 6716 §4.3.2.2).
    pub fine_priority: [bool; NUM_BANDS],
    /// One-past-last coded band (absolute index, `> start`).
    pub coded_bands: usize,
    /// Leftover 1/8 bits carried into the §4.3.4 band-loop
    /// rebalancing.
    pub balance: i32,
    /// First intensity-coded band (absolute index; `0` when the field
    /// was not coded).
    pub intensity: i32,
    /// Dual-stereo flag (`false` when the field was not coded).
    pub dual_stereo: bool,
}

/// The reference-exact `interp_bits2pulses` (Appendix A `rate.c`):
/// interpolation bisection between the two bracketing allocation
/// vectors, concurrent top-down skip decoding, intensity/dual field
/// coding, the remaining-bit spread, and the fine/shape split.
#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32; NUM_BANDS],
    bits2: &[i32; NUM_BANDS],
    thresh: &[i32; NUM_BANDS],
    cap: &[i32; NUM_BANDS],
    mut total: i32,
    skip_rsv: i32,
    mut intensity_rsv: i32,
    mut dual_stereo_rsv: i32,
    channels: i32,
    lm: u32,
    io: &mut AllocIo<'_, '_>,
) -> Result<ExactAllocation, Error> {
    let eb = |i: usize| EBAND_EDGES_5MS[i] as i32;
    let alloc_floor = channels << BITRES;
    let stereo = i32::from(channels > 1);
    let log_m = (lm << BITRES) as i32;

    // Interpolation bisection over the 1/64 fraction.
    let mut lo: i32 = 0;
    let mut hi: i32 = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum: i32 = 0;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                // Don't allocate more than we can actually use.
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let mut bits = [0i32; NUM_BANDS];
    let mut ebits = [0i32; NUM_BANDS];
    let mut fine_priority = [false; NUM_BANDS];
    let mut psum: i32 = 0;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            tmp = if tmp >= alloc_floor { alloc_floor } else { 0 };
        } else {
            done = true;
        }
        // Don't allocate more than we can actually use.
        tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // Decide which bands to skip, working backwards from the end.
    let mut coded_bands = end;
    loop {
        let j = coded_bands - 1;
        // Never skip the first band, nor a band boosted by dynalloc:
        // in the first case the skip bit would only signal wasted
        // bits, in the second it would undo a boost just signaled.
        if j <= skip_start {
            // Give the reserved skip-termination bit back to the pool.
            total += skip_rsv;
            break;
        }
        // How many leftover bits this band would inherit (including
        // bits stolen back from higher, skipped bands).
        let mut left = total - psum;
        let percoeff = left / (eb(coded_bands) - eb(start));
        left -= (eb(coded_bands) - eb(start)) * percoeff;
        let rem = (left - (eb(j) - eb(start))).max(0);
        let band_width = eb(coded_bands) - eb(j);
        let mut band_bits = bits[j] + percoeff * band_width + rem;
        // Only code a skip decision above the threshold for this
        // band; otherwise it is force-skipped (this also guarantees
        // the skip flag itself is affordable).
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            match io {
                AllocIo::Encode {
                    enc,
                    prev_coded_bands,
                    ..
                } => {
                    // Encoder skip choice (the only non-mandatory part
                    // of the walk): a hysteresis threshold anchored on
                    // the previous frame's coded_bands.
                    let anchor = if (j as i32) < *prev_coded_bands { 7 } else { 9 };
                    if band_bits > ((anchor * band_width) << lm << BITRES) >> 4 {
                        enc.enc_bit_logp(1, 1)
                            .map_err(|_| Error::InvalidParameter)?;
                        break;
                    }
                    enc.enc_bit_logp(0, 1)
                        .map_err(|_| Error::InvalidParameter)?;
                }
                AllocIo::Decode(dec) => {
                    if dec.dec_bit_logp(1) == 1 {
                        break;
                    }
                }
            }
            // We used a bit to code the skip decision for this band.
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        // Reclaim the bits originally allocated to this band.
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC_TABLE[j - start] as i32;
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            // Keep a fine-energy bit per channel if affordable.
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            // Otherwise this band gets nothing at all.
            bits[j] = 0;
        }
        coded_bands -= 1;
    }
    debug_assert!(coded_bands > start);

    // Code the intensity and dual-stereo parameters.
    let mut intensity_out: i32 = 0;
    if intensity_rsv > 0 {
        match io {
            AllocIo::Encode { enc, intensity, .. } => {
                let want = (*intensity).min(coded_bands as i32).max(start as i32);
                intensity_out = want;
                enc.enc_uint(
                    (want - start as i32) as u32,
                    (coded_bands + 1 - start) as u32,
                )
                .map_err(|_| Error::InvalidParameter)?;
            }
            AllocIo::Decode(dec) => {
                intensity_out = start as i32
                    + dec
                        .dec_uint((coded_bands + 1 - start) as u32)
                        .map_err(|_| Error::InvalidParameter)? as i32;
            }
        }
    }
    if intensity_out <= start as i32 {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    let mut dual_stereo_out = false;
    if dual_stereo_rsv > 0 {
        match io {
            AllocIo::Encode {
                enc, dual_stereo, ..
            } => {
                dual_stereo_out = *dual_stereo;
                enc.enc_bit_logp(u32::from(dual_stereo_out), 1)
                    .map_err(|_| Error::InvalidParameter)?;
            }
            AllocIo::Decode(dec) => {
                dual_stereo_out = dec.dec_bit_logp(1) == 1;
            }
        }
    }

    // Allocate the remaining bits.
    let mut left = total - psum;
    let percoeff = left / (eb(coded_bands) - eb(start));
    left -= (eb(coded_bands) - eb(start)) * percoeff;
    for (j, b) in bits.iter_mut().enumerate().take(coded_bands).skip(start) {
        *b += percoeff * (eb(j + 1) - eb(j));
    }
    for (j, b) in bits.iter_mut().enumerate().take(coded_bands).skip(start) {
        let tmp = left.min(eb(j + 1) - eb(j));
        *b += tmp;
        left -= tmp;
    }

    // The fine-energy vs. shape split with the running balance.
    let mut balance: i32 = 0;
    for j in start..coded_bands {
        let n0 = eb(j + 1) - eb(j);
        let n = n0 << lm;
        bits[j] += balance;
        let excess;
        if n > 1 {
            excess = (bits[j] - cap[j]).max(0);
            bits[j] -= excess;

            // Compensate for the extra degree of freedom in stereo.
            let den = channels * n
                + i32::from(
                    channels == 2 && n > 2 && !dual_stereo_out && (j as i32) < intensity_out,
                );
            let nclogn = den * (LOG_N400[j] as i32 + log_m);
            // Offset the fine bits by log2(N)/2 + FINE_OFFSET relative
            // to their fair share of total/N.
            let mut offset = (nclogn >> 1) - den * FINE_OFFSET;
            // N = 2 is the only point that doesn't match the curve.
            if n == 2 {
                offset += (den << BITRES) >> 2;
            }
            // Changing the offset for allocating the second and third
            // fine energy bit.
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nclogn >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nclogn >> 3;
            }
            // Divide with rounding.
            ebits[j] = 0.max((bits[j] + offset + (den << (BITRES - 1))) / (den << BITRES));
            // Make sure not to bust.
            if channels * ebits[j] > (bits[j] >> BITRES) {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }
            // More than MAX_FINE_BITS is useless.
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            // Rounded-down or capped bands are candidates for the
            // final fine-energy pass: priority 0.
            fine_priority[j] = ebits[j] * (den << BITRES) >= bits[j] + offset;
            // Remove the allocated fine bits; the rest go to PVQ.
            bits[j] -= (channels * ebits[j]) << BITRES;
        } else {
            // For N = 1, all bits go to fine energy except a sign bit.
            excess = (bits[j] - (channels << BITRES)).max(0);
            bits[j] -= excess;
            ebits[j] = 0;
            fine_priority[j] = true;
        }

        // Fine energy can't take advantage of the §4.3.4 band-loop
        // rebalancing, so rebalance the capped-off excess here.
        let mut excess = excess;
        if excess > 0 {
            let extra_fine = (excess >> (stereo as u32 + BITRES)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * channels) << BITRES;
            fine_priority[j] = extra_bits >= excess - balance;
            excess -= extra_bits;
        }
        balance = excess;

        debug_assert!(bits[j] >= 0);
        debug_assert!(ebits[j] >= 0);
    }

    // Skipped bands spend all their remaining bits on fine energy.
    for j in coded_bands..end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        debug_assert!((channels * ebits[j]) << BITRES == bits[j]);
        bits[j] = 0;
        fine_priority[j] = ebits[j] < 1;
    }

    let fine_bits = ebits;
    Ok(ExactAllocation {
        shape_bits: bits,
        fine_bits,
        fine_priority,
        coded_bands,
        balance,
        intensity: intensity_out,
        dual_stereo: dual_stereo_out,
    })
}

/// `FINE_OFFSET` (Appendix A `rate.h`): the fine-bit fair-share bias.
pub const FINE_OFFSET: i32 = 21;

/// `QTHETA_OFFSET` / `QTHETA_OFFSET_TWOPHASE` (Appendix A `rate.h`):
/// the split-resolution biases consumed by the §4.3.4.4 `compute_qn`.
pub const QTHETA_OFFSET: i32 = 4;
/// See [`QTHETA_OFFSET`].
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;

/// The reference-exact §4.3.3 allocation entry point (Appendix A
/// `rate.c` `compute_allocation`).
///
/// * `offsets` — decoded per-band dynalloc boosts, 1/8 bits (absolute
///   band indexing).
/// * `cap` — per-band maxima, 1/8 bits (from
///   [`crate::band_cap::compute_band_caps`], absolute indexing).
/// * `total` — the post-anti-collapse budget in 1/8 bits
///   (`frame_bytes * 64 - tell_frac - 1 - anti_collapse_rsv`).
/// * `io` — the live range coder; the walk reads (or writes) the
///   Table-56 `skip`, `intensity`, and `dual` symbols itself, and the
///   skip / intensity / dual **reservations** are carved out of
///   `total` here exactly as the listing does.
///
/// On return the coder sits immediately before the Table-56
/// `fine energy` section.
#[allow(clippy::too_many_arguments)]
pub fn compute_allocation_exact(
    start: usize,
    end: usize,
    offsets: &[i32; NUM_BANDS],
    cap: &[i32; NUM_BANDS],
    alloc_trim: i32,
    total: i32,
    channels: i32,
    lm: u32,
    mut io: AllocIo<'_, '_>,
) -> Result<ExactAllocation, Error> {
    if start >= end
        || end > NUM_BANDS
        || !(1..=2).contains(&channels)
        || lm > 3
        || !(0..=10).contains(&alloc_trim)
    {
        return Err(Error::InvalidParameter);
    }
    let eb = |i: usize| EBAND_EDGES_5MS[i] as i32;

    let mut total = total.max(0);
    let mut skip_start = start;
    // Reserve a bit to signal the end of manually skipped bands.
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;
    // Reserve bits for the intensity and dual-stereo parameters.
    let mut intensity_rsv = 0;
    let mut dual_stereo_rsv = 0;
    if channels == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[end - start] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }

    let mut thresh = [0i32; NUM_BANDS];
    let mut trim_offset = [0i32; NUM_BANDS];
    for j in start..end {
        let n = eb(j + 1) - eb(j);
        // Below this threshold no PVQ bits are allocated.
        thresh[j] = (channels << BITRES).max(((3 * n) << lm << BITRES) >> 4);
        // Tilt of the allocation curve.
        trim_offset[j] = (channels
            * n
            * (alloc_trim - 5 - lm as i32)
            * (end as i32 - j as i32 - 1)
            * (1 << (lm + BITRES)))
            >> 6;
        // Giving less resolution to single-coefficient bands because
        // they get more benefit from one coarse value per coefficient.
        if (n << lm) == 1 {
            trim_offset[j] -= channels << BITRES;
        }
    }

    // Codepoint search over the Table-57 quality columns.
    let mut lo: i32 = 1;
    let mut hi: i32 = NUM_Q as i32 - 1;
    loop {
        let mid = (lo + hi) >> 1;
        let mut psum: i32 = 0;
        let mut done = false;
        for j in (start..end).rev() {
            let n = eb(j + 1) - eb(j);
            let mut bitsj = (channels * n * STATIC_ALLOC[j][mid as usize] as i32) << lm >> 2;
            if bitsj > 0 {
                bitsj = 0.max(bitsj + trim_offset[j]);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                // Don't allocate more than we can actually use.
                psum += bitsj.min(cap[j]);
            } else if bitsj >= channels << BITRES {
                psum += channels << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    hi = lo;
    lo -= 1;

    let mut bits1 = [0i32; NUM_BANDS];
    let mut bits2 = [0i32; NUM_BANDS];
    for j in start..end {
        let n = eb(j + 1) - eb(j);
        let mut bits1j = (channels * n * STATIC_ALLOC[j][lo as usize] as i32) << lm >> 2;
        let mut bits2j = if hi >= NUM_Q as i32 {
            cap[j]
        } else {
            (channels * n * STATIC_ALLOC[j][hi as usize] as i32) << lm >> 2
        };
        if bits1j > 0 {
            bits1j = 0.max(bits1j + trim_offset[j]);
        }
        if bits2j > 0 {
            bits2j = 0.max(bits2j + trim_offset[j]);
        }
        if lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = 0.max(bits2j - bits1j);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    interp_bits2pulses(
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        skip_rsv,
        intensity_rsv,
        dual_stereo_rsv,
        channels,
        lm,
        &mut io,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band_cap::compute_band_caps;
    use crate::band_minimums::BAND_BINS_LM;

    fn caps_for(lm: u32, channels: i32) -> [i32; NUM_BANDS] {
        let bins: Vec<u32> = BAND_BINS_LM[lm as usize].to_vec();
        let mut caps16 = vec![0i16; NUM_BANDS];
        assert!(compute_band_caps(
            lm,
            channels == 2,
            channels as u32,
            &bins,
            &mut caps16
        ));
        let mut caps = [0i32; NUM_BANDS];
        for (c, &v) in caps.iter_mut().zip(caps16.iter()) {
            *c = v as i32;
        }
        caps
    }

    /// Encode-side walk followed by a decode-side walk over the
    /// produced bytes recovers the identical allocation — the §4.3.3
    /// lockstep requirement.
    #[test]
    fn encode_decode_lockstep() {
        for &(channels, lm, total) in &[
            (1i32, 0u32, 500i32),
            (1, 2, 1200),
            (1, 3, 2600),
            (2, 1, 1800),
            (2, 3, 6000),
            (1, 1, 90),
            (2, 0, 260),
        ] {
            let caps = caps_for(lm, channels);
            let mut offsets = [0i32; NUM_BANDS];
            offsets[3] = 48; // one boosted band pins skip_start
            let mut enc = RangeEncoder::new();
            let enc_alloc = compute_allocation_exact(
                0,
                NUM_BANDS,
                &offsets,
                &caps,
                5,
                total,
                channels,
                lm,
                AllocIo::Encode {
                    enc: &mut enc,
                    intensity: 17,
                    dual_stereo: true,
                    prev_coded_bands: 0,
                },
            )
            .expect("encode walk");
            let bytes = enc.finish();
            let mut dec = RangeDecoder::new(&bytes);
            let dec_alloc = compute_allocation_exact(
                0,
                NUM_BANDS,
                &offsets,
                &caps,
                5,
                total,
                channels,
                lm,
                AllocIo::Decode(&mut dec),
            )
            .expect("decode walk");
            assert_eq!(enc_alloc, dec_alloc, "C={channels} LM={lm} total={total}");
        }
    }

    /// Outputs respect the §4.3.3 invariants: non-negative shape and
    /// fine allocations, the fine ceiling, and a coded window that
    /// keeps at least the first band.
    #[test]
    fn output_invariants() {
        for &(channels, lm) in &[(1i32, 0u32), (1, 3), (2, 2)] {
            let caps = caps_for(lm, channels);
            let offsets = [0i32; NUM_BANDS];
            for total in [0, 8, 64, 300, 1000, 4000, 12000] {
                let mut enc = RangeEncoder::new();
                let alloc = compute_allocation_exact(
                    0,
                    NUM_BANDS,
                    &offsets,
                    &caps,
                    5,
                    total,
                    channels,
                    lm,
                    AllocIo::Encode {
                        enc: &mut enc,
                        intensity: 0,
                        dual_stereo: false,
                        prev_coded_bands: 0,
                    },
                )
                .expect("walk");
                assert!(alloc.coded_bands > 0 && alloc.coded_bands <= NUM_BANDS);
                for j in 0..NUM_BANDS {
                    assert!(alloc.shape_bits[j] >= 0);
                    assert!(alloc.fine_bits[j] >= 0 && alloc.fine_bits[j] <= MAX_FINE_BITS);
                }
                for j in alloc.coded_bands..NUM_BANDS {
                    assert_eq!(alloc.shape_bits[j], 0, "skipped bands carry no shape bits");
                }
            }
        }
    }

    /// The pseudo-pulse map: identity below 8, then the exponential
    /// ramp; `bits2pulses` inverts `pulses2bits` exactly on the grid.
    #[test]
    fn pulse_map_and_cache_roundtrip() {
        assert_eq!(get_pulses(0), 0);
        assert_eq!(get_pulses(7), 7);
        assert_eq!(get_pulses(8), 8);
        assert_eq!(get_pulses(15), 15);
        assert_eq!(get_pulses(16), 16);
        assert_eq!(get_pulses(17), 18);
        assert_eq!(get_pulses(24), 32);
        assert_eq!(get_pulses(32), 64);
        assert_eq!(get_pulses(40), 128);
        for band in 0..NUM_BANDS {
            for lm in -1i32..=3 {
                let Some(cache) = cache_row(band, lm) else {
                    continue;
                };
                let max_pseudo = cache[0] as i32;
                for q in 1..=max_pseudo {
                    let bits = pulses2bits(band, lm, q).unwrap();
                    let back = bits2pulses(band, lm, bits).unwrap();
                    // Cost-faithful inversion: on a plateau of the
                    // cost curve the nearest-with-ties-down selection
                    // returns the lowest pseudo index sharing the
                    // cost, so compare costs rather than indices.
                    assert_eq!(
                        pulses2bits(band, lm, back).unwrap(),
                        bits,
                        "band {band} lm {lm} pseudo {q}"
                    );
                    assert!(back <= q, "band {band} lm {lm} pseudo {q}");
                }
                // Monotone cost curve.
                for q in 1..max_pseudo {
                    assert!(cache[q as usize] <= cache[q as usize + 1]);
                }
            }
        }
    }

    /// A band boosted by dynalloc is never skipped: the walk stops at
    /// `skip_start` and returns the reserved skip bit to the pool.
    #[test]
    fn boosted_band_pins_skip_floor() {
        let lm = 2u32;
        let channels = 1i32;
        let caps = caps_for(lm, channels);
        let mut offsets = [0i32; NUM_BANDS];
        offsets[NUM_BANDS - 1] = 400;
        let mut enc = RangeEncoder::new();
        let alloc = compute_allocation_exact(
            0,
            NUM_BANDS,
            &offsets,
            &caps,
            5,
            700,
            channels,
            lm,
            AllocIo::Encode {
                enc: &mut enc,
                intensity: 0,
                dual_stereo: false,
                prev_coded_bands: 0,
            },
        )
        .expect("walk");
        assert_eq!(alloc.coded_bands, NUM_BANDS);
    }
}
