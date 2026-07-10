//! The §4.3.3 bit-allocation **reallocation walk** (RFC 6716) — the
//! interval-bisection search over the allocation quality parameter,
//! the 1/64-step linear interpolation, the concurrent skip decoding,
//! the intensity / dual-stereo field placement, the fine-energy vs.
//! shape split, the final rebalancing, and the priority 0/1
//! determination.
//!
//! ## Provenance
//!
//! This module transcribes the clean-room behavioral specification
//! `docs/audio/celt/spec/celt-reallocation-walk.md` (staged against
//! RFC 6716 §4.3.3 lines 6111–6461 and §4.3.2.2 lines 6079–6099),
//! which pins:
//!
//! * the static-allocation formula on **base-width** `N` with the
//!   `<< LM` restoring the actual bin count (§2 of the chapter),
//! * the per-band candidate `min(static + trim_offset + boost, cap)`
//!   (§3),
//! * the outer bisection over the 11 quality codepoints (§4),
//! * the 1/64 interpolation objective ("highest allocation that does
//!   not exceed the number of bits remaining", §5),
//! * the skip-walk **direction** (top band downward), the `{1,1}/2`
//!   skip PDF, the "one skip bit per candidate, stop on
//!   do-not-skip" contract, and the never-skipped lowest band (§6),
//! * the fine-energy whole-bits-per-channel unit and its
//!   value-independent cost (§7),
//! * the four outputs — shape, fine, priorities, balance (§8), and
//! * the final-bit distribution policy (priority 0 before 1, low band
//!   first, one extra fine bit per band per channel, §9).
//!
//! Three constants the chapter's §10 marks as pinned only by the
//! encoder/decoder bit-exactness requirement — the exact fine/shape
//! split divisor, the exact skip predicate composition, and the exact
//! priority 0/1 predicate — are in-crate deterministic decisions
//! documented at their definition sites below. Both codec sides run
//! this same walk from the bit-identical prefix, so the §4.3.3
//! lockstep requirement ("identical coding decisions ... in the
//! encoder and decoder") holds for this crate's own streams by
//! construction; bit-exactness against *other* encoders' streams is
//! validated black-box (see the crate README) and the residual gaps
//! remain flagged for a captured reference trace.
//!
//! No external library source was consulted.

use crate::band_minimums::SHORT_FRAME_BAND_BINS;
use crate::coarse_energy::NUM_BANDS;
use crate::fine_energy::{FinalizePriority, MAX_FINE_BITS};
use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::static_alloc::{NUM_Q, STATIC_ALLOC};
use crate::Error;

/// Cost of one coded skip bit in 1/8-bit units (`{1,1}/2` PDF — one
/// whole bit, RFC 6716 Table 56 / §4.3.3).
pub const SKIP_BIT_COST_8TH: i32 = 8;

/// Number of 1/64 interpolation steps between adjacent quality
/// codepoints (RFC 6716 §4.3.3 lines 6226–6229).
pub const FRAC_STEPS: u32 = 64;

/// The per-frame inputs of the reallocation walk — everything the
/// §4.3.3 setup (clean-room narrative §§2.1–2.6) produced ahead of the
/// walk. All slices are window-indexed (`[0]` is band `start`) and
/// must have length `end - start`.
#[derive(Debug, Clone, Copy)]
pub struct WalkBands<'a> {
    /// First coded band (0 for pure CELT, 17 in Hybrid mode).
    pub start: usize,
    /// One-past-last coded band (`<= 21`).
    pub end: usize,
    /// Per-band per-channel MDCT-bin count at the **actual** frame
    /// size (`BAND_BINS_LM[lm]` sliced to the window).
    pub bins: &'a [u32],
    /// Per-band maximum `cap[]` in 1/8 bits (§2.2).
    pub caps: &'a [i32],
    /// Decoded per-band boosts in 1/8 bits (§2.3).
    pub boost: &'a [i32],
    /// Per-band trim offsets in 1/8 bits (§2.6).
    pub trim_offsets: &'a [i32],
    /// Per-band hard shape minimums `thresh[]` in 1/8 bits (§2.6).
    pub thresh: &'a [i32],
    /// 1 (mono) or 2 (stereo).
    pub channels: u32,
    /// Frame-size shift `log2(frame_size / 120)`, `0..=3`.
    pub lm: u32,
}

/// The walk's budget inputs — the §2.5 post-reservation running total
/// plus the individual reservations the walk itself spends or returns.
#[derive(Debug, Clone, Copy)]
pub struct WalkBudget {
    /// Post-reservation budget in 1/8 bits (the §2.5 `total` after
    /// the anti-collapse / skip / intensity / dual reservations).
    pub total: i32,
    /// The skip reservation (8 iff reserved, else 0). Returned to the
    /// walk's pool — the reserved bit is spent by the first skip
    /// symbol, and further symbols are funded by the allocations the
    /// skipped bands hand back (chapter §6: the reserved bit "may be
    /// spent more than once").
    pub skip_rsv: i32,
    /// The intensity reservation (`LOG2_FRAC_TABLE[end-start]` iff
    /// reserved, else 0). Non-zero gates the intensity field decode.
    pub intensity_rsv: i32,
    /// The dual-stereo reservation (8 iff reserved, else 0). Non-zero
    /// gates the dual-stereo flag decode.
    pub dual_rsv: i32,
}

/// Range-coder attachment for the walk: the decoder **reads** the
/// skip / intensity / dual symbols; the encoder **writes** its chosen
/// decisions at the identical bitstream positions.
#[allow(missing_debug_implementations)] // carries raw coder state, not data
pub enum WalkIo<'a, 'b> {
    /// Decode side: symbols are read from the stream.
    Decode(&'a mut RangeDecoder<'b>),
    /// Encode side: symbols are written from the encoder's choices.
    Encode {
        /// The range encoder positioned right after the Table-56
        /// `alloc. trim` symbol.
        enc: &'a mut RangeEncoder,
        /// Desired one-past-last coded band (absolute index). Bands
        /// above it are skipped when the walk lets them be; the
        /// effective outcome is [`WalkAllocation::coded_bands`].
        coded_bands: usize,
        /// Desired intensity offset from `start` (clamped to the
        /// post-skip window; only written when `intensity_rsv > 0`).
        intensity: u32,
        /// Desired dual-stereo flag (only written when
        /// `dual_rsv == 8`).
        dual_stereo: bool,
    },
}

/// The §4.3.3 walk outputs — exactly the four results RFC 6716
/// enumerates (lines 6209–6214), plus the decoded stereo fields and
/// instrumentation for tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkAllocation {
    /// Per-band **shape** allocation in 1/8 bits, window-indexed.
    /// Zero on skipped bands.
    pub shape_bits: Vec<i32>,
    /// Per-band **fine-energy** allocation in whole bits per channel,
    /// window-indexed. Zero on skipped bands.
    pub fine_bits: Vec<u32>,
    /// Per-band finalize priority (0 before 1, RFC 6716 §4.3.2.2),
    /// window-indexed.
    pub fine_priority: Vec<FinalizePriority>,
    /// One-past-last coded band, **absolute** index
    /// (`start < coded_bands <= end` whenever the window is
    /// non-empty). Bands in `coded_bands..end` receive no shape bits.
    pub coded_bands: usize,
    /// Remaining unallocated 1/8 bits ("usually zero except at very
    /// high rates").
    pub balance: i32,
    /// Intensity-stereo offset from `start`: bands at
    /// `start + intensity_band` and above (within the coded window)
    /// are intensity-coded. Equal to `coded_bands - start` means
    /// intensity is never applied. 0 when the field was not coded.
    pub intensity_band: u32,
    /// Dual-stereo flag (`false` when the field was not coded).
    pub dual_stereo: bool,
    /// Number of skip symbols actually read/written (test
    /// instrumentation).
    pub skip_bits_used: u32,
    /// The selected lower quality codepoint (test instrumentation).
    pub qlo: usize,
    /// The selected 1/64 interpolation fraction (test
    /// instrumentation).
    pub frac: u32,
}

/// The §3 per-band candidate at an integer quality codepoint, in 1/8
/// bits: `clamp(static(band, q) + trim_offset + boost, 0, cap)`.
///
/// The static term evaluates on the 2.5 ms **base** width with the
/// `<< LM` restoring the actual bin count (chapter §2); the zero floor
/// is the in-crate reading of the chapter's `min(…, cap)` (a negative
/// allocation is meaningless and the §6 fold path handles the
/// sub-minimum bands).
fn candidate_at(bands: &WalkBands<'_>, q: usize, out: &mut [i32]) {
    for (j, out_j) in out.iter_mut().enumerate() {
        let band = bands.start + j;
        let base_n = SHORT_FRAME_BAND_BINS[band];
        let static8 = ((bands.channels * base_n * STATIC_ALLOC[band][q] as u32) << bands.lm) >> 2;
        let raw = static8 as i64 + bands.trim_offsets[j] as i64 + bands.boost[j] as i64;
        *out_j = raw.clamp(0, bands.caps[j].max(0) as i64) as i32;
    }
}

/// Window total of a candidate vector, in 1/8 bits.
fn total_of(v: &[i32]) -> i64 {
    v.iter().map(|&x| x as i64).sum()
}

/// The §5 interpolated per-band allocation:
/// `(bits1 * (64 - frac) + bits2 * frac) >> 6`, computed on the two
/// **cap-clamped** bracketing vectors (chapter §5 blends the capped
/// candidates, not the raw static terms).
fn interp_band(bits1: i32, bits2: i32, frac: u32) -> i32 {
    (((bits1 as i64) * ((FRAC_STEPS - frac) as i64) + (bits2 as i64) * (frac as i64)) >> 6) as i32
}

fn interp_total(bits1: &[i32], bits2: &[i32], frac: u32) -> i64 {
    bits1
        .iter()
        .zip(bits2)
        .map(|(&a, &b)| interp_band(a, b, frac) as i64)
        .sum()
}

/// Run the §4.3.3 reallocation walk.
///
/// The range coder in `io` must be positioned immediately after the
/// Table-56 `alloc. trim` symbol; on return it sits immediately before
/// the `fine energy` section. The walk reads (or writes) the skip
/// symbols, then the intensity field (iff `budget.intensity_rsv > 0`),
/// then the dual-stereo flag (iff `budget.dual_rsv == 8`) — the
/// Table-56 `skip` → `intensity` → `dual` order.
///
/// Returns [`Error::InvalidParameter`] on inconsistent inputs
/// (mismatched slice lengths, `channels ∉ {1,2}`, `lm > 3`, or a
/// window overflowing the 21-band mode).
pub fn realloc_walk(
    bands: &WalkBands<'_>,
    budget: &WalkBudget,
    io: WalkIo<'_, '_>,
) -> Result<WalkAllocation, Error> {
    let n = bands.end.saturating_sub(bands.start);
    if bands.start > bands.end
        || bands.end > NUM_BANDS
        || bands.bins.len() != n
        || bands.caps.len() != n
        || bands.boost.len() != n
        || bands.trim_offsets.len() != n
        || bands.thresh.len() != n
        || !(1..=2).contains(&bands.channels)
        || bands.lm > 3
    {
        return Err(Error::InvalidParameter);
    }
    let mut io = io;

    if n == 0 {
        return Ok(WalkAllocation {
            shape_bits: Vec::new(),
            fine_bits: Vec::new(),
            fine_priority: Vec::new(),
            coded_bands: bands.start,
            balance: budget.total.max(0),
            intensity_band: 0,
            dual_stereo: false,
            skip_bits_used: 0,
            qlo: 0,
            frac: 0,
        });
    }

    let total = budget.total.max(0) as i64;

    // ------------------------------------------------------------------
    // §4: outer bisection over the 11 quality codepoints — the highest
    // codepoint whose window total fits the budget. The chapter's
    // pseudocode verbatim: lo = 0, hi = 11 (exclusive), 4 iterations.
    // ------------------------------------------------------------------
    let mut scratch = vec![0i32; n];
    let mut lo: usize = 0;
    let mut hi: usize = NUM_Q;
    while hi - lo > 1 {
        let mid = (lo + hi) >> 1;
        candidate_at(bands, mid, &mut scratch);
        if total_of(&scratch) > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let qlo = lo;

    // ------------------------------------------------------------------
    // §5: inner 1/64 interpolation between the capped bracketing
    // vectors. At the top codepoint the grid degenerates (bits2 =
    // bits1); otherwise the largest frac whose blended total fits.
    // ------------------------------------------------------------------
    let mut bits1 = vec![0i32; n];
    candidate_at(bands, qlo, &mut bits1);
    let mut bits2 = vec![0i32; n];
    if qlo + 1 < NUM_Q {
        candidate_at(bands, qlo + 1, &mut bits2);
    } else {
        bits2.copy_from_slice(&bits1);
    }

    // `fhi` is exclusive. When even the qlo column overshoots (the
    // "very low rates" collapse: qlo == 0 and total_alloc(0) > total),
    // frac stays 0.
    let mut flo: u32 = 0;
    let mut fhi: u32 = FRAC_STEPS + 1;
    if total_of(&bits1) > total {
        fhi = 1;
    }
    while fhi - flo > 1 {
        let mid = (flo + fhi) >> 1;
        if interp_total(&bits1, &bits2, mid) > total {
            fhi = mid;
        } else {
            flo = mid;
        }
    }
    let frac = flo;

    let mut alloc: Vec<i32> = (0..n)
        .map(|j| interp_band(bits1[j], bits2[j], frac))
        .collect();

    // ------------------------------------------------------------------
    // §6: concurrent skip decoding, top band downward. The pool `left`
    // starts from the interpolation remainder plus the returned skip
    // reservation; every coded skip symbol costs one bit from the
    // pool, and every skipped band hands its allocation back.
    //
    // In-crate determinations inside the chapter's §10 residual gap
    // (documented, black-box-validated):
    //  * a band is a *viable* stop when `alloc[j] + left >= thresh[j]`
    //    (the band's allocation plus the pool share it would receive
    //    clears the §2.6 hard minimum);
    //  * a viable band with pool < one bit is coded without a symbol
    //    (nothing left to signal with);
    //  * a non-viable band folds silently (no symbol) — the chapter's
    //    `last_zero` "forced inactive, no skip bit spent" boundary;
    //  * skip-bit polarity: 1 = skip, 0 = stop (band coded).
    // ------------------------------------------------------------------
    let mut left: i64 = total + budget.skip_rsv as i64 - total_of(&alloc);
    let mut coded_w = n; // window units, exclusive
    let mut skip_bits_used = 0u32;
    for j in (1..n).rev() {
        let viable = alloc[j] as i64 + left >= bands.thresh[j] as i64;
        if viable {
            if left < SKIP_BIT_COST_8TH as i64 {
                break;
            }
            left -= SKIP_BIT_COST_8TH as i64;
            skip_bits_used += 1;
            let skip = match &mut io {
                WalkIo::Decode(dec) => dec.dec_bit_logp(1) == 1,
                WalkIo::Encode {
                    enc, coded_bands, ..
                } => {
                    let skip = bands.start + j >= *coded_bands;
                    enc.enc_bit_logp(u32::from(skip), 1)?;
                    skip
                }
            };
            if !skip {
                break;
            }
        }
        left += alloc[j] as i64;
        alloc[j] = 0;
        coded_w = j;
    }
    let coded_bands = bands.start + coded_w;

    // ------------------------------------------------------------------
    // Table 56: intensity, then dual. The intensity selector is
    // uniform over the post-skip coded window (`0..=coded_w`; the top
    // value means "never applied") — an in-crate determination inside
    // the §10 gap; the reservation is *not* refunded (the wire surplus
    // of a shrunken window lands in the finalize pool).
    // ------------------------------------------------------------------
    let intensity_band = if budget.intensity_rsv > 0 {
        let ft = coded_w as u32 + 1;
        match &mut io {
            WalkIo::Decode(dec) => dec.dec_uint(ft).unwrap_or(0),
            WalkIo::Encode { enc, intensity, .. } => {
                let v = (*intensity).min(coded_w as u32);
                enc.enc_uint(v, ft)?;
                v
            }
        }
    } else {
        0
    };
    let dual_stereo = if budget.dual_rsv == SKIP_BIT_COST_8TH {
        match &mut io {
            WalkIo::Decode(dec) => dec.dec_bit_logp(1) == 1,
            WalkIo::Encode {
                enc, dual_stereo, ..
            } => {
                enc.enc_bit_logp(u32::from(*dual_stereo), 1)?;
                *dual_stereo
            }
        }
    } else {
        false
    };

    // ------------------------------------------------------------------
    // §8: final reallocation — sweep the coded bands low to high and
    // hand the pool to bands below their cap; whatever cannot be
    // placed stays in the balance.
    // ------------------------------------------------------------------
    if left > 0 {
        for (j, alloc_j) in alloc.iter_mut().enumerate().take(coded_w) {
            let headroom = (bands.caps[j].max(0) - *alloc_j).max(0) as i64;
            let give = headroom.min(left);
            *alloc_j += give as i32;
            left -= give;
            if left == 0 {
                break;
            }
        }
    }
    let balance = left.clamp(i32::MIN as i64, i32::MAX as i64) as i32;

    // ------------------------------------------------------------------
    // §7 + §9: fine-energy vs. shape split and the priority stamp.
    //
    // In-crate determinations inside the §10 gap (documented,
    // black-box-validated): the split counts the band's gain as one
    // extra degree of freedom next to its N shape samples —
    // `fine = alloc / (8 * channels * (N + 1))` whole bits per
    // channel, capped at MAX_FINE_BITS — and a band is priority 0
    // when the division's remainder is at least half a fine-bit step
    // (the band most starved by the whole-bit rounding, chapter §9)
    // and it can still absorb another fine bit.
    // ------------------------------------------------------------------
    let mut shape_bits = vec![0i32; n];
    let mut fine_bits = vec![0u32; n];
    let mut fine_priority = vec![FinalizePriority::One; n];
    for j in 0..coded_w {
        let a = alloc[j].max(0) as u32;
        let den = 8 * bands.channels * (bands.bins[j] + 1);
        let fine = (a / den).min(MAX_FINE_BITS);
        let rem = a % den;
        fine_bits[j] = fine;
        shape_bits[j] = (a - 8 * bands.channels * fine) as i32;
        if 2 * rem >= den && fine < MAX_FINE_BITS {
            fine_priority[j] = FinalizePriority::Zero;
        }
    }

    Ok(WalkAllocation {
        shape_bits,
        fine_bits,
        fine_priority,
        coded_bands,
        balance,
        intensity_band,
        dual_stereo,
        skip_bits_used,
        qlo,
        frac,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band_cap::compute_band_caps;
    use crate::band_minimums::{compute_thresh, compute_trim_offsets, BAND_BINS_LM};

    /// Assemble a full 21-band mono/stereo input set at `lm`.
    struct Fixture {
        bins: Vec<u32>,
        caps: Vec<i32>,
        boost: Vec<i32>,
        trim_offsets: Vec<i32>,
        thresh: Vec<i32>,
        channels: u32,
        lm: u32,
    }

    impl Fixture {
        fn new(lm: u32, channels: u32, trim: i32, boost_pairs: &[(usize, i32)]) -> Self {
            let bins: Vec<u32> = BAND_BINS_LM[lm as usize].to_vec();
            let n = bins.len();
            let stereo = channels == 2;
            let mut caps16 = vec![0i16; n];
            assert!(compute_band_caps(lm, stereo, channels, &bins, &mut caps16));
            let caps: Vec<i32> = caps16.iter().map(|&c| c as i32).collect();
            let mut boost = vec![0i32; n];
            for &(b, v) in boost_pairs {
                boost[b] = v;
            }
            let mut trim_offsets = vec![0i32; n];
            assert!(compute_trim_offsets(
                trim,
                lm,
                channels,
                0,
                &bins,
                &mut trim_offsets
            ));
            let mut thresh = vec![0i32; n];
            assert!(compute_thresh(channels, &bins, &mut thresh));
            Self {
                bins,
                caps,
                boost,
                trim_offsets,
                thresh,
                channels,
                lm,
            }
        }

        fn bands(&self) -> WalkBands<'_> {
            WalkBands {
                start: 0,
                end: self.bins.len(),
                bins: &self.bins,
                caps: &self.caps,
                boost: &self.boost,
                trim_offsets: &self.trim_offsets,
                thresh: &self.thresh,
                channels: self.channels,
                lm: self.lm,
            }
        }
    }

    /// Encode-side walk into a fresh range encoder with the given
    /// choices; returns the finished bytes and the walk outcome.
    fn encode_walk(
        fx: &Fixture,
        budget: &WalkBudget,
        coded_bands: usize,
        intensity: u32,
        dual: bool,
    ) -> (Vec<u8>, WalkAllocation) {
        let mut enc = RangeEncoder::new();
        let out = realloc_walk(
            &fx.bands(),
            budget,
            WalkIo::Encode {
                enc: &mut enc,
                coded_bands,
                intensity,
                dual_stereo: dual,
            },
        )
        .expect("encode walk");
        (enc.finish(), out)
    }

    /// Decode-side walk over `bytes` with the same inputs.
    fn decode_walk(fx: &Fixture, budget: &WalkBudget, bytes: &[u8]) -> WalkAllocation {
        let mut dec = RangeDecoder::new(bytes);
        realloc_walk(&fx.bands(), budget, WalkIo::Decode(&mut dec)).expect("decode walk")
    }

    /// Brute-force §4 reference: the highest integer codepoint whose
    /// candidate total fits (linear scan; the chapter notes any
    /// monotone search order selects the same codepoint).
    fn brute_qlo(fx: &Fixture, total: i64) -> usize {
        let bands = fx.bands();
        let n = fx.bins.len();
        let mut best = 0usize;
        let mut v = vec![0i32; n];
        for q in 0..NUM_Q {
            candidate_at(&bands, q, &mut v);
            if total_of(&v) <= total {
                best = q;
            }
        }
        best
    }

    /// The bisection lands on the brute-force codepoint, and the
    /// chosen fraction is maximal: its total fits, `frac + 1` (or the
    /// next column) does not.
    #[test]
    fn bisection_matches_brute_force_and_frac_is_maximal() {
        for lm in 0..=3u32 {
            for channels in [1u32, 2] {
                let fx = Fixture::new(lm, channels, 5, &[(3, 24), (12, 48)]);
                let bands = fx.bands();
                let n = fx.bins.len();
                let mut top = vec![0i32; n];
                candidate_at(&bands, NUM_Q - 1, &mut top);
                let top_total = total_of(&top);
                for budget_total in [
                    0i64,
                    64,
                    top_total / 7,
                    top_total / 3,
                    top_total / 2,
                    top_total - 1,
                    top_total + 500,
                ] {
                    let budget = WalkBudget {
                        total: budget_total.max(0) as i32,
                        skip_rsv: 0,
                        intensity_rsv: 0,
                        dual_rsv: 0,
                    };
                    // Decode from an empty buffer: every skip read
                    // lands on the sticky-error 0 path ("coded"),
                    // which stops the walk immediately — fine for
                    // inspecting (qlo, frac).
                    let out = decode_walk(&fx, &budget, &[]);
                    let expect_q = brute_qlo(&fx, budget_total.max(0));
                    assert_eq!(out.qlo, expect_q, "lm={lm} ch={channels} b={budget_total}");

                    // frac maximality on the capped-blend grid.
                    let mut bits1 = vec![0i32; n];
                    candidate_at(&bands, out.qlo, &mut bits1);
                    let mut bits2 = vec![0i32; n];
                    if out.qlo + 1 < NUM_Q {
                        candidate_at(&bands, out.qlo + 1, &mut bits2);
                    } else {
                        bits2.copy_from_slice(&bits1);
                    }
                    let t = interp_total(&bits1, &bits2, out.frac);
                    if total_of(&bits1) <= budget_total.max(0) {
                        assert!(t <= budget_total.max(0), "chosen frac total over budget");
                        if out.frac < FRAC_STEPS {
                            assert!(
                                interp_total(&bits1, &bits2, out.frac + 1) > budget_total.max(0),
                                "frac not maximal (lm={lm} b={budget_total})"
                            );
                        }
                    } else {
                        assert_eq!(out.frac, 0, "collapse case must pin frac=0");
                    }
                }
            }
        }
    }

    /// Encode → decode lockstep: the decoder reconstructs the
    /// identical allocation from the bytes the encoder wrote, across
    /// frame sizes, channel counts, budgets, and skip targets.
    #[test]
    fn encode_decode_lockstep() {
        for lm in 0..=3u32 {
            for channels in [1u32, 2] {
                let stereo = channels == 2;
                let fx = Fixture::new(lm, channels, if lm == 2 { 7 } else { 4 }, &[(5, 16)]);
                let n = fx.bins.len();
                for budget_total in [50i32, 400, 1200, 4000, 20000] {
                    for coded_target in [n, n - 1, n / 2, 1] {
                        let budget = WalkBudget {
                            total: budget_total,
                            skip_rsv: if budget_total > 8 { 8 } else { 0 },
                            intensity_rsv: if stereo { 35 } else { 0 },
                            dual_rsv: if stereo { 8 } else { 0 },
                        };
                        let (bytes, enc_out) = encode_walk(&fx, &budget, coded_target, 7, stereo);
                        let dec_out = decode_walk(&fx, &budget, &bytes);
                        assert_eq!(
                            enc_out, dec_out,
                            "lockstep broke: lm={lm} ch={channels} b={budget_total} \
                             target={coded_target}"
                        );
                    }
                }
            }
        }
    }

    /// Budget conservation: shape + fine + balance + skip-symbol cost
    /// accounts for the full pool (`total + skip_rsv`) whenever the
    /// interpolation had room to fit (no collapse).
    #[test]
    fn budget_conservation_identity() {
        for lm in 0..=3u32 {
            for channels in [1u32, 2] {
                let fx = Fixture::new(lm, channels, 5, &[]);
                let n = fx.bins.len();
                for budget_total in [300i32, 900, 2500, 9000] {
                    let budget = WalkBudget {
                        total: budget_total,
                        skip_rsv: 8,
                        intensity_rsv: 0,
                        dual_rsv: 0,
                    };
                    let (bytes, _) = encode_walk(&fx, &budget, n, 0, false);
                    let out = decode_walk(&fx, &budget, &bytes);
                    let shape: i64 = out.shape_bits.iter().map(|&x| x as i64).sum();
                    let fine: i64 = out
                        .fine_bits
                        .iter()
                        .map(|&f| 8 * channels as i64 * f as i64)
                        .sum();
                    let spent = shape
                        + fine
                        + out.balance as i64
                        + SKIP_BIT_COST_8TH as i64 * out.skip_bits_used as i64;
                    assert_eq!(
                        spent,
                        budget_total as i64 + 8,
                        "conservation broke lm={lm} ch={channels} b={budget_total}"
                    );
                    assert!(out.balance >= 0, "negative balance");
                    for (j, &s) in out.shape_bits.iter().enumerate() {
                        assert!(s >= 0, "negative shape at {j}");
                    }
                }
            }
        }
    }

    /// The lowest coded band is never skipped, whatever the encoder
    /// asks for.
    #[test]
    fn lowest_band_never_skipped() {
        let fx = Fixture::new(2, 1, 5, &[]);
        let budget = WalkBudget {
            total: 2000,
            skip_rsv: 8,
            intensity_rsv: 0,
            dual_rsv: 0,
        };
        // Ask to skip everything (coded_bands = 0): the walk must
        // still code band 0.
        let (bytes, enc_out) = encode_walk(&fx, &budget, 0, 0, false);
        assert_eq!(enc_out.coded_bands, 1, "band 0 must stay coded");
        let dec_out = decode_walk(&fx, &budget, &bytes);
        assert_eq!(enc_out, dec_out);
        assert!(dec_out.shape_bits[0] > 0 || dec_out.fine_bits[0] > 0);
        for j in 1..fx.bins.len() {
            assert_eq!(dec_out.shape_bits[j], 0, "band {j} should be skipped");
            assert_eq!(dec_out.fine_bits[j], 0, "band {j} fine should be 0");
        }
    }

    /// Skipping hands the freed allocation to the surviving bands:
    /// with the same budget, a heavily skipped walk gives the coded
    /// window at least as many 1/8 bits as the unskipped walk gave
    /// the same bands (minus the extra skip symbols).
    #[test]
    fn skip_redistributes_to_lower_bands() {
        let fx = Fixture::new(1, 1, 5, &[]);
        let n = fx.bins.len();
        let budget = WalkBudget {
            total: 3000,
            skip_rsv: 8,
            intensity_rsv: 0,
            dual_rsv: 0,
        };
        let (bytes_all, _) = encode_walk(&fx, &budget, n, 0, false);
        let all = decode_walk(&fx, &budget, &bytes_all);
        let (bytes_half, _) = encode_walk(&fx, &budget, n / 2, 0, false);
        let half = decode_walk(&fx, &budget, &bytes_half);
        assert!(half.coded_bands < all.coded_bands);
        let low_total = |w: &WalkAllocation| -> i64 {
            (0..half.coded_bands)
                .map(|j| w.shape_bits[j] as i64 + 8 * w.fine_bits[j] as i64)
                .sum()
        };
        assert!(
            low_total(&half) >= low_total(&all),
            "skipping did not redistribute to the surviving bands"
        );
    }

    /// A zero-ish budget produces the collapse path: no skip symbols
    /// are affordable, every band above the first folds silently, and
    /// nothing goes negative.
    #[test]
    fn very_low_budget_collapses_without_symbols() {
        for channels in [1u32, 2] {
            let fx = Fixture::new(3, channels, 5, &[]);
            let budget = WalkBudget {
                total: 0,
                skip_rsv: 0,
                intensity_rsv: 0,
                dual_rsv: 0,
            };
            let (bytes, enc_out) = encode_walk(&fx, &budget, fx.bins.len(), 0, false);
            assert_eq!(enc_out.skip_bits_used, 0, "no skip symbol affordable");
            assert!(bytes.len() <= 1, "collapse walk should write nothing");
            let out = decode_walk(&fx, &budget, &bytes);
            assert_eq!(out, enc_out);
            assert_eq!(out.qlo, 0);
            assert_eq!(out.frac, 0);
            for &s in &out.shape_bits {
                assert!(s >= 0);
            }
        }
    }

    /// Fine bits respect the §7 contract: `fine <= MAX_FINE_BITS`,
    /// `shape = alloc - 8*channels*fine >= 0`, and the priority-0
    /// stamp implies a large whole-bit-rounding remainder.
    #[test]
    fn fine_split_and_priority_contract() {
        for channels in [1u32, 2] {
            let fx = Fixture::new(2, channels, 5, &[(0, 64)]);
            let n = fx.bins.len();
            let budget = WalkBudget {
                total: 6000,
                skip_rsv: 8,
                intensity_rsv: 0,
                dual_rsv: 0,
            };
            let (bytes, _) = encode_walk(&fx, &budget, n, 0, false);
            let out = decode_walk(&fx, &budget, &bytes);
            let mut saw_fine = false;
            for j in 0..n {
                assert!(out.fine_bits[j] <= MAX_FINE_BITS);
                assert!(out.shape_bits[j] >= 0);
                saw_fine |= out.fine_bits[j] > 0;
                if out.fine_priority[j] == FinalizePriority::Zero {
                    let a = out.shape_bits[j] as u32 + 8 * channels * out.fine_bits[j];
                    let den = 8 * channels * (fx.bins[j] + 1);
                    assert!(2 * (a % den) >= den, "priority-0 with small remainder");
                    assert!(out.fine_bits[j] < MAX_FINE_BITS);
                }
            }
            assert!(saw_fine, "a 6000-eighth-bit frame must fund fine bits");
        }
    }

    /// Stereo fields land on the wire in Table-56 order and round-trip;
    /// intensity is clamped to the post-skip window.
    #[test]
    fn stereo_fields_roundtrip_and_clamp() {
        let fx = Fixture::new(1, 2, 5, &[]);
        let n = fx.bins.len();
        let budget = WalkBudget {
            total: 2600,
            skip_rsv: 8,
            intensity_rsv: 37,
            dual_rsv: 8,
        };
        // Skip down to 6 coded bands but ask for intensity offset 15:
        // the written value must clamp to the surviving window.
        let (bytes, enc_out) = encode_walk(&fx, &budget, 6, 15, true);
        assert!(enc_out.coded_bands <= 6);
        let win = enc_out.coded_bands; // == coded_w for start = 0
        assert!(enc_out.intensity_band <= win as u32);
        assert!(enc_out.dual_stereo);
        let dec_out = decode_walk(&fx, &budget, &bytes);
        assert_eq!(enc_out, dec_out);
        // Without reservations the fields default silently.
        let budget0 = WalkBudget {
            total: 2600,
            skip_rsv: 8,
            intensity_rsv: 0,
            dual_rsv: 0,
        };
        let (bytes0, enc0) = encode_walk(&fx, &budget0, n, 15, true);
        let dec0 = decode_walk(&fx, &budget0, &bytes0);
        assert_eq!(dec0.intensity_band, 0);
        assert!(!dec0.dual_stereo);
        assert_eq!(enc0, dec0);
    }

    /// Boosted bands rise relative to the unboosted walk (until the
    /// cap binds) — the §3 candidate carries the boost through the
    /// whole walk.
    #[test]
    fn boost_biases_the_outcome() {
        let lm = 2u32;
        let base_fx = Fixture::new(lm, 1, 5, &[]);
        let boost_fx = Fixture::new(lm, 1, 5, &[(4, 160)]);
        let n = base_fx.bins.len();
        let budget = WalkBudget {
            total: 2000,
            skip_rsv: 8,
            intensity_rsv: 0,
            dual_rsv: 0,
        };
        let (b0, _) = encode_walk(&base_fx, &budget, n, 0, false);
        let base = decode_walk(&base_fx, &budget, &b0);
        let (b1, _) = encode_walk(&boost_fx, &budget, n, 0, false);
        let boosted = decode_walk(&boost_fx, &budget, &b1);
        let total_band = |w: &WalkAllocation, j: usize| w.shape_bits[j] + 8 * w.fine_bits[j] as i32;
        assert!(
            total_band(&boosted, 4) > total_band(&base, 4),
            "boost did not raise band 4"
        );
    }

    /// Input validation: mismatched slices and bad scalars are
    /// rejected.
    #[test]
    fn invalid_inputs_rejected() {
        let fx = Fixture::new(0, 1, 5, &[]);
        let budget = WalkBudget {
            total: 100,
            skip_rsv: 0,
            intensity_rsv: 0,
            dual_rsv: 0,
        };
        let mut bands = fx.bands();
        bands.channels = 3;
        let mut dec = RangeDecoder::new(&[]);
        assert!(matches!(
            realloc_walk(&bands, &budget, WalkIo::Decode(&mut dec)),
            Err(Error::InvalidParameter)
        ));
        let mut bands = fx.bands();
        bands.lm = 4;
        let mut dec = RangeDecoder::new(&[]);
        assert!(matches!(
            realloc_walk(&bands, &budget, WalkIo::Decode(&mut dec)),
            Err(Error::InvalidParameter)
        ));
        let short: Vec<i32> = fx.caps[..20].to_vec();
        let mut bands = fx.bands();
        bands.caps = &short;
        let mut dec = RangeDecoder::new(&[]);
        assert!(matches!(
            realloc_walk(&bands, &budget, WalkIo::Decode(&mut dec)),
            Err(Error::InvalidParameter)
        ));
    }

    /// An empty window degenerates cleanly.
    #[test]
    fn empty_window() {
        let budget = WalkBudget {
            total: 100,
            skip_rsv: 0,
            intensity_rsv: 0,
            dual_rsv: 0,
        };
        let bands = WalkBands {
            start: 17,
            end: 17,
            bins: &[],
            caps: &[],
            boost: &[],
            trim_offsets: &[],
            thresh: &[],
            channels: 1,
            lm: 3,
        };
        let mut dec = RangeDecoder::new(&[]);
        let out = realloc_walk(&bands, &budget, WalkIo::Decode(&mut dec)).unwrap();
        assert_eq!(out.coded_bands, 17);
        assert_eq!(out.balance, 100);
        assert!(out.shape_bits.is_empty());
    }

    /// Hybrid window (start = 17): the walk indexes the absolute band
    /// axis for the static table and produces a 4-band outcome.
    #[test]
    fn hybrid_window_walks() {
        let lm = 3u32;
        let full = Fixture::new(lm, 1, 5, &[]);
        let bins = &full.bins[17..];
        let caps = &full.caps[17..];
        let boost = &full.boost[17..];
        let mut trim_offsets = vec![0i32; 4];
        assert!(compute_trim_offsets(5, lm, 1, 17, bins, &mut trim_offsets));
        let mut thresh = vec![0i32; 4];
        assert!(compute_thresh(1, bins, &mut thresh));
        let bands = WalkBands {
            start: 17,
            end: 21,
            bins,
            caps,
            boost,
            trim_offsets: &trim_offsets,
            thresh: &thresh,
            channels: 1,
            lm,
        };
        let budget = WalkBudget {
            total: 1500,
            skip_rsv: 8,
            intensity_rsv: 0,
            dual_rsv: 0,
        };
        let mut enc = RangeEncoder::new();
        let enc_out = realloc_walk(
            &bands,
            &budget,
            WalkIo::Encode {
                enc: &mut enc,
                coded_bands: 21,
                intensity: 0,
                dual_stereo: false,
            },
        )
        .unwrap();
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        let dec_out = realloc_walk(&bands, &budget, WalkIo::Decode(&mut dec)).unwrap();
        assert_eq!(enc_out, dec_out);
        assert_eq!(dec_out.shape_bits.len(), 4);
        assert!(dec_out.coded_bands > 17 && dec_out.coded_bands <= 21);
    }
}
