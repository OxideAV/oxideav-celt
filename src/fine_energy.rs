//! Fine-energy decoding (RFC 6716 §4.3.2.2).
//!
//! ## Where this fits
//!
//! CELT quantises per-band log-energies in three steps (RFC 6716 §4.3.2):
//!
//! 1. **Coarse** — 6 dB integer steps in base-2 log, range-coded with a
//!    Laplace prior (§4.3.2.1; presently DOCS-GAP blocked, see
//!    [`crate::coarse_energy`]).
//! 2. **Fine** — per-band raw-bit refinement of the coarse value, with
//!    the bit count `B_i` for each band fixed by the §4.3.3 bit
//!    allocator. This module.
//! 3. **Finalize** — any leftover bits the allocator could not place
//!    are spent at the very end of the frame, ≤ 1 per band per
//!    channel, walked in `(priority 0 across bands, priority 1 across
//!    bands)` order. Also this module.
//!
//! ## What §4.3.2.2 actually specifies
//!
//! Quoting the relevant passage (RFC 6716 §4.3.2.2,
//! `docs/audio/opus/rfc6716-opus.txt` lines 6081–6099):
//!
//! > Let B_i be the number of fine energy bits for band i; the
//! > refinement is an integer f in the range [0,2**B_i-1]. The
//! > mapping between f and the correction applied to the coarse
//! > energy is equal to (f+1/2)/2**B_i - 1/2.
//! >
//! > When some bits are left "unused" after all other flags have been
//! > decoded, these bits are assigned to a "final" step of fine
//! > allocation. In effect, these bits are used to add one extra fine
//! > energy bit per band per channel. The allocation process
//! > determines two "priorities" for the final fine bits. Any
//! > remaining bits are first assigned only to bands of priority 0,
//! > starting from band 0 and going up. If all bands of priority 0
//! > have received one bit per channel, then bands of priority 1 are
//! > assigned an extra bit per channel, starting from band 0. If any
//! > bits are left after this, they are left unused.
//!
//! Everything in the above passage is implementable directly: `f` is
//! read with [`crate::range_decoder::RangeDecoder::dec_bits`] (RFC
//! §4.1.4 raw-bit packing, LSB-first from the end of the frame), and
//! the correction is a closed-form linear map on `(f, B_i)`. The
//! finalize loop is a straight nested walk over `priorities × bands ×
//! channels`. No Laplace decoder, no probability table, no
//! `e_prob_model` reference — fine energy is purely a raw-bit channel.
//!
//! ## Q14 representation
//!
//! The §4.3.2.1 coarse step is 6 dB = 1.0 in base-2 log. The §4.3.2.2
//! correction therefore lies in `[-1/2, +1/2)` base-2 log. We expose
//! the correction in **Q14** (1.0 base-2 log = `1 << 14 = 16384`),
//! which keeps every legal correction in `i16` range
//! (`[-8192, +8192)`) and avoids overflow when summed with a Q14
//! coarse-energy value across the 21-band envelope. Callers that work
//! in a different fixed-point scale can either rescale or use the
//! lower-level [`fine_correction_q14`] / [`fine_correction_qn`]
//! helpers.
//!
//! ### Closed-form derivation (transcribed from the spec)
//!
//! Starting from `correction = (f + 1/2)/2^B - 1/2`, multiply both
//! sides by `2^14` to land in Q14:
//!
//! ```text
//! correction_q14 = ((2f + 1) / 2^(B+1)) * 2^14 - 2^13
//!                = (2f + 1) * 2^(13 - B) - 2^13
//! ```
//!
//! The expression is exact when `B <= 13`; the CELT bit allocator
//! never assigns more than `MAX_FINE_BITS = 8` fine bits per band
//! (the alloc.trim / allocation interpolation tables top out well
//! below 13), so the shift count `13 - B` is always non-negative and
//! the multiplication never overflows `i32`.
//!
//! ## Clean-room provenance
//!
//! Every formula, every bit-count limit, every walk order in this
//! module is transcribed from RFC 6716 §4.3.2.2 in
//! `docs/audio/opus/rfc6716-opus.txt`. The libopus source file the
//! RFC names (`celt/quant_bands.c`, functions `quant_fine_energy` /
//! `unquant_energy_finalise`) was NOT consulted; the workspace
//! clean-room policy bars it.

use crate::coarse_energy::NUM_BANDS;
use crate::range_decoder::RangeDecoder;

/// Upper bound on the number of fine bits the §4.3.3 bit allocator can
/// assign to any single band.
///
/// The RFC does not state a single hard cap, but the band-allocator's
/// interpolation tables (§4.3.3) and the surrounding allocation
/// arithmetic constrain per-band fine bits to lie well below the Q14
/// scale's headroom. We pin the constant at 8 — generous against the
/// ~6-bit ceiling the allocator actually exercises, and comfortably
/// below the 13 above which [`fine_correction_q14`]'s shift-left
/// arithmetic would lose precision. Callers that pass `b_bits >
/// MAX_FINE_BITS` get the correction saturated, never a panic.
pub const MAX_FINE_BITS: u32 = 8;

/// Band-decode priority for the §4.3.2.2 finalize step.
///
/// The §4.3.3 bit-allocator tags each band with a 0-or-1 priority
/// indicating the order in which leftover bits should be spent. We
/// model the tag as an enum to keep call sites self-documenting; the
/// underlying invariant the spec asserts — "all priority-0 bands first,
/// then all priority-1 bands, both walked in ascending band order" — is
/// enforced inside [`finalize_extra_bits`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalizePriority {
    /// Spend leftover bits on this band first.
    Zero,
    /// Spend leftover bits on this band only after every priority-0
    /// band has received one extra bit per channel.
    One,
}

/// Outcome of one call to [`finalize_extra_bits`].
///
/// `extra_bits_q14[i]` is the additional Q14 correction (positive or
/// negative) the finalize step contributes to band `i` of one channel.
/// For stereo decodes the caller invokes [`finalize_extra_bits`] once
/// per channel (see the function docstring for the per-channel split
/// reasoning); the per-channel outputs are then summed band-by-band
/// onto the post-fine log-energy.
#[derive(Debug, Clone, Copy)]
pub struct FinalizeResult {
    /// Per-band Q14 correction contributed by the extra bit, summed
    /// over both channels. Bands that did not receive an extra bit
    /// carry zero.
    pub extra_correction_q14: [i32; NUM_BANDS],
    /// Number of bits actually consumed by this finalize call (≤ the
    /// caller-supplied budget).
    pub bits_consumed: u32,
    /// Number of bits left over because the budget exceeded the total
    /// possible 1-bit-per-band-per-channel slots. Per the RFC: "If
    /// any bits are left after this, they are left unused."
    pub bits_unused: u32,
}

/// Decode the §4.3.2.2 fine-energy refinement for one band.
///
/// `b_bits` is `B_i` from the spec: the number of fine bits the
/// §4.3.3 allocator assigned to this band. When `b_bits == 0` the
/// band gets no fine refinement; this function returns 0 and does not
/// consume any raw bits.
///
/// When `b_bits > 0` the function reads exactly `b_bits` raw bits
/// (via `dec_bits`, LSB-first from the back of the frame per
/// §4.1.4), forms the integer `f ∈ [0, 2^b_bits)`, and returns the
/// closed-form Q14 correction:
///
/// ```text
/// correction_q14 = (2f + 1) * 2^(13 - b_bits) - 2^13
/// ```
///
/// The return value lies in `[-8192, +8191]` for every legal
/// `(f, b_bits)` pair. The caller adds it (Q14) to the band's coarse
/// log-2 energy to produce the fine-quantised value.
///
/// For `b_bits > MAX_FINE_BITS` the function saturates the shift to
/// `13 - MAX_FINE_BITS` rather than panicking; in practice the
/// allocator never produces such values.
pub fn decode_fine_energy_band(dec: &mut RangeDecoder<'_>, b_bits: u32) -> i32 {
    if b_bits == 0 {
        // No refinement; the spec's correction (f+1/2)/2^0 - 1/2 is
        // ill-defined for B=0 (no raw bits to read), and the §4.3.3
        // allocator simply skips bands it could not buy any fine bits
        // for. Return zero correction, no decoder consumption.
        return 0;
    }
    let f = dec.dec_bits(b_bits);
    fine_correction_q14(f, b_bits)
}

/// Closed-form §4.3.2.2 correction in Q14, given the decoded `f` and
/// the band's fine-bit count `b_bits`.
///
/// Exposed standalone so encoders, test scaffolding, and callers
/// working at a different fixed-point scale can apply the same
/// arithmetic without re-deriving it.
pub fn fine_correction_q14(f: u32, b_bits: u32) -> i32 {
    // Saturate b_bits to MAX_FINE_BITS so the shift count never
    // overflows. The §4.3.3 allocator never crosses this boundary.
    let b = b_bits.min(MAX_FINE_BITS);
    // (2f + 1) is in [1, 2^(b+1) - 1]; for b <= MAX_FINE_BITS = 8 the
    // value fits easily in u32. We cast to i64 for the multiply +
    // subtract to keep the arithmetic obviously panic-free across the
    // full Q14 range.
    let shift = 13_i32 - b as i32;
    debug_assert!(
        (0..=13).contains(&shift),
        "fine_correction_q14: shift {shift} out of [0, 13]"
    );
    let two_f_plus_one = 2i64 * f as i64 + 1;
    let scaled = two_f_plus_one * (1i64 << shift);
    (scaled - (1i64 << 13)) as i32
}

/// Closed-form §4.3.2.2 correction in a caller-specified Q-scale,
/// given the decoded `f` and the band's fine-bit count `b_bits`.
///
/// Equivalent to `(f + 1/2)/2^b_bits - 1/2` evaluated in Qn:
///
/// ```text
/// correction_qn = (2f + 1) * 2^(n - 1 - b_bits) - 2^(n - 1)
/// ```
///
/// `n` must be at least `b_bits + 1` to keep the shift non-negative.
/// Callers that want Q8 (matching the coarse-energy scaffolding's
/// `prev_q8` convention) call this with `n = 8`; Q14 callers should
/// prefer [`fine_correction_q14`] which short-circuits the runtime
/// branch.
pub fn fine_correction_qn(f: u32, b_bits: u32, n: u32) -> i64 {
    // Real-valued spec formula: (f + 1/2) / 2^b - 1/2.
    // In Qn: ((2f + 1) << n) / 2^(b+1) - 2^(n-1).
    // We do the shift and divide in i64 with round-half-up so that
    // calling with n < b + 1 still produces the nearest-representable
    // Qn integer rather than degenerating to 0.
    let two_f_plus_one = 2i64 * f as i64 + 1;
    let half_step = 1i64 << (n - 1);
    let bp1 = b_bits as i64 + 1;
    let n_i = n as i64;
    let numerator = if bp1 > n_i {
        // Need a divide; round to nearest with the standard
        // add-half-divisor-and-floor trick.
        let shift = bp1 - n_i;
        let denom = 1i64 << shift;
        let half_denom = denom >> 1;
        (two_f_plus_one + half_denom) >> shift
    } else {
        two_f_plus_one << (n_i - bp1)
    };
    numerator - half_step
}

/// Decode the §4.3.2.2 fine refinement for every band whose allocator-
/// assigned bit count is non-zero, returning per-band Q14 corrections.
///
/// `bits_per_band[i] == B_i` for band `i`. Bands with `B_i == 0`
/// receive zero correction without consuming raw bits. The total raw
/// bits consumed is `sum(b_bits_per_band)`.
///
/// The §4.3.3 allocator guarantees that the sum fits within the
/// remaining post-coarse-energy raw-bit budget; this function does
/// not re-validate that budget (it would have to re-derive the §4.3.3
/// allocation, which is DOCS-GAP blocked separately). Callers that
/// want defence-in-depth should consult `tell()` before and after.
///
/// Caller's bit budget walks via the range decoder's `nbits_raw`
/// counter (visible through `tell()`/`tell_frac()`); each call to
/// [`decode_fine_energy_band`] advances it by exactly `B_i` bits.
pub fn decode_fine_energy(
    dec: &mut RangeDecoder<'_>,
    bits_per_band: &[u32; NUM_BANDS],
) -> [i32; NUM_BANDS] {
    let mut out = [0i32; NUM_BANDS];
    for b in 0..NUM_BANDS {
        out[b] = decode_fine_energy_band(dec, bits_per_band[b]);
    }
    out
}

/// Decode the §4.3.2.2 finalize step: spend up to `budget` leftover
/// raw bits on per-band extra fine refinement, walking in
/// `(priority 0 ascending, priority 1 ascending)` order with at most
/// one extra bit per band per channel.
///
/// ### Parameters
///
/// * `priorities[i]` — the §4.3.3-assigned priority of band `i`.
/// * `coded_bands` — number of bands the frame codes, in band-index
///   order. Pure CELT uses `0..21` (the full envelope); hybrid mode
///   uses `0..4` of the priorities slice, corresponding to bands
///   `17..21` (the caller is responsible for shifting the result
///   back into the absolute band slot if needed).
/// * `channels` — `1` for mono, `2` for stereo. The RFC explicitly
///   says "one extra fine energy bit per band per channel"; the
///   stereo case spends 2 bits per band, with both channels getting
///   the same correction sign at this stage (the per-channel sign
///   split happens during shape decoding, well beyond the scope of
///   this module).
/// * `budget` — number of raw bits the §4.3.3 allocator left
///   unspent. The RFC explicitly permits this value to overshoot the
///   total `coded_bands * channels` capacity, in which case the
///   surplus is "left unused".
///
/// ### Return
///
/// A [`FinalizeResult`] whose `extra_correction_q14[i]` field is the
/// per-band Q14 correction summed over both channels (so for stereo
/// it is twice the single-bit correction whenever both channels
/// received the extra bit). `bits_consumed + bits_unused == budget`.
///
/// ### Correction magnitude
///
/// An "extra fine bit" is exactly the `B_i = 1` case of the §4.3.2.2
/// formula:
///
/// ```text
/// correction(f, B=1) = (f + 1/2)/2 - 1/2 = (2f - 1)/4
/// ```
///
/// In Q14: `correction_q14(0, 1) = -4096`, `correction_q14(1, 1) =
/// +4096`. The function reuses [`fine_correction_q14`] so the
/// arithmetic stays in one place.
pub fn finalize_extra_bits(
    dec: &mut RangeDecoder<'_>,
    priorities: &[FinalizePriority],
    coded_bands: usize,
    channels: u32,
    budget: u32,
) -> FinalizeResult {
    assert!(
        coded_bands <= priorities.len(),
        "finalize_extra_bits: coded_bands={coded_bands} > priorities.len()={}",
        priorities.len()
    );
    assert!(
        coded_bands <= NUM_BANDS,
        "finalize_extra_bits: coded_bands={coded_bands} > NUM_BANDS={NUM_BANDS}"
    );
    assert!(
        channels == 1 || channels == 2,
        "finalize_extra_bits: channels={channels} must be 1 or 2"
    );

    let mut result = FinalizeResult {
        extra_correction_q14: [0i32; NUM_BANDS],
        bits_consumed: 0,
        bits_unused: 0,
    };

    if coded_bands == 0 || budget == 0 {
        result.bits_unused = budget;
        return result;
    }

    // Walk priorities 0 then 1, both in ascending band order, with one
    // extra bit per band per channel. The spec's phrasing — "If all
    // bands of priority 0 have received one bit per channel, then
    // bands of priority 1 are assigned an extra bit per channel" — is
    // strictly priority-major, then channel-major within a band-pair
    // step, then band-major. Concretely:
    //
    //   for prio in [Zero, One]:
    //       for band in 0..coded_bands where priorities[band] == prio:
    //           for ch in 0..channels:
    //               if budget == 0: break
    //               read 1 raw bit and accumulate the Q14 correction
    //
    // The RFC does not separately track per-channel corrections in the
    // log-energy envelope (the envelope is per-band, not per-band-per-
    // channel at the coarse/fine stage), so we sum the two channels'
    // corrections into the same `extra_correction_q14[band]` slot.
    let mut remaining = budget;
    'outer: for prio in [FinalizePriority::Zero, FinalizePriority::One] {
        for (band, &p) in priorities.iter().enumerate().take(coded_bands) {
            if p != prio {
                continue;
            }
            for _ch in 0..channels {
                if remaining == 0 {
                    break 'outer;
                }
                // One extra raw bit; B_i = 1 case of §4.3.2.2.
                let f = dec.dec_bits(1);
                let correction = fine_correction_q14(f, 1);
                result.extra_correction_q14[band] =
                    result.extra_correction_q14[band].saturating_add(correction);
                result.bits_consumed += 1;
                remaining -= 1;
            }
        }
    }

    result.bits_unused = remaining;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §4.3.2.2: with `b_bits == 0` the function must be a no-op on the
    /// decoder and return zero correction.
    #[test]
    fn fine_band_zero_bits_is_noop() {
        let buf = [0xFFu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let before_raw = dec.tell();
        let c = decode_fine_energy_band(&mut dec, 0);
        let after_raw = dec.tell();
        assert_eq!(c, 0);
        assert_eq!(before_raw, after_raw, "B=0 consumed bits");
        assert!(!dec.has_error());
    }

    /// §4.3.2.2 closed-form Q14 spot checks at every B in [1, 8] and
    /// every f in [0, 2^B - 1]. The correction must land in
    /// [-8192, +8191].
    #[test]
    fn fine_correction_q14_in_range_for_all_legal_b_f() {
        for b in 1..=MAX_FINE_BITS {
            let max_f = (1u32 << b) - 1;
            for f in 0..=max_f {
                let c = fine_correction_q14(f, b);
                assert!(
                    (-8192..=8191).contains(&c),
                    "B={b}, f={f}: q14 correction {c} out of [-8192, 8191]"
                );
            }
        }
    }

    /// §4.3.2.2 closed-form Q14 vs spec's real-valued formula:
    /// the correction is `(f + 1/2)/2^B - 1/2`. Scaling by 2^14 and
    /// rounding-to-nearest must match `fine_correction_q14` exactly.
    #[test]
    fn fine_correction_q14_matches_spec_formula() {
        for b in 1..=MAX_FINE_BITS {
            let max_f = (1u32 << b) - 1;
            for f in 0..=max_f {
                // (2f + 1) * 2^(13 - B) - 2^13 is the closed form. Compute
                // it again by the spec's decimal expression scaled to Q14:
                //   correction = (f + 1/2)/2^B - 1/2
                //   q14        = correction * 2^14
                // = ((2f + 1) * 2^14) / 2^(B+1) - 2^13
                // = (2f + 1) << (13 - B) - 2^13
                let expected = ((2 * f as i64 + 1) << (13 - b)) - (1 << 13);
                assert_eq!(
                    fine_correction_q14(f, b) as i64,
                    expected,
                    "B={b}, f={f}: closed-form mismatch"
                );
            }
        }
    }

    /// §4.3.2.2 midpoint property: for any B, the symbol `f = (2^B)/2`
    /// (when 2^B is even) lies just above zero, and `f = 2^B/2 - 1`
    /// lies just below zero. The Q14 corrections must be symmetric
    /// around zero modulo Q14's rounding step.
    #[test]
    fn fine_correction_q14_zero_symmetric() {
        for b in 1..=MAX_FINE_BITS {
            let half = 1u32 << (b - 1);
            // f = half - 1: just below zero
            let lo = fine_correction_q14(half - 1, b);
            // f = half: just above zero
            let hi = fine_correction_q14(half, b);
            assert!(lo < 0, "B={b}: f=half-1 gave {lo}, expected <0");
            assert!(hi > 0, "B={b}: f=half gave {hi}, expected >0");
            // Symmetric magnitudes: hi - 0 == 0 - lo, off-by-one in Q14
            // because the spec's "+1/2" bias makes (hi + lo) = 0.
            assert_eq!(
                hi + lo,
                0,
                "B={b}: asymmetric correction (lo={lo}, hi={hi})"
            );
        }
    }

    /// §4.3.2.2 endpoint property: f = 0 gives the most-negative
    /// correction; f = 2^B - 1 gives the most-positive. Both must be
    /// strict bounds of every other (f, B) pair.
    #[test]
    fn fine_correction_q14_endpoint_extrema() {
        for b in 1..=MAX_FINE_BITS {
            let max_f = (1u32 << b) - 1;
            let low = fine_correction_q14(0, b);
            let high = fine_correction_q14(max_f, b);
            for f in 0..=max_f {
                let c = fine_correction_q14(f, b);
                assert!(c >= low, "B={b}, f={f}: c={c} < low={low}");
                assert!(c <= high, "B={b}, f={f}: c={c} > high={high}");
            }
        }
    }

    /// `decode_fine_energy_band` must consume exactly `b_bits` raw
    /// bits, no more, no less. Verify via `tell()` before/after.
    #[test]
    fn fine_band_consumes_exact_bits() {
        let buf = [0xA3u8, 0x5C, 0xE8, 0x91, 0x42, 0xB7, 0x10, 0x7F];
        for b in 1..=MAX_FINE_BITS {
            let mut dec = RangeDecoder::new(&buf);
            let before = dec.tell();
            let _ = decode_fine_energy_band(&mut dec, b);
            let after = dec.tell();
            assert_eq!(
                after - before,
                b,
                "B={b}: consumed {} bits, expected {b}",
                after - before
            );
        }
    }

    /// `decode_fine_energy` walks 21 bands. Total bits consumed must
    /// equal `sum(bits_per_band)`.
    #[test]
    fn fine_energy_full_envelope_bit_count() {
        let buf = [0x5Au8; 64];
        let bits_per_band: [u32; NUM_BANDS] = [
            3, 3, 3, 3, 3, // bands 0..5
            2, 2, 2, 2, 2, // bands 5..10
            1, 1, 1, 1, 1, // bands 10..15
            1, 1, 1, 1, 1, // bands 15..20
            1, // band 20
        ];
        let expected_total: u32 = bits_per_band.iter().sum();
        let mut dec = RangeDecoder::new(&buf);
        let before = dec.tell();
        let _ = decode_fine_energy(&mut dec, &bits_per_band);
        let after = dec.tell();
        assert_eq!(after - before, expected_total);
        assert!(!dec.has_error());
    }

    /// `decode_fine_energy` with every B_i = 0 must be a no-op on the
    /// decoder and return all-zero corrections.
    #[test]
    fn fine_energy_all_zero_b_is_noop() {
        let buf = [0xFFu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let before = dec.tell();
        let out = decode_fine_energy(&mut dec, &[0; NUM_BANDS]);
        let after = dec.tell();
        assert_eq!(out, [0i32; NUM_BANDS]);
        assert_eq!(before, after, "All-zero B consumed bits");
    }

    /// `fine_correction_qn` at n = 14 must match `fine_correction_q14`
    /// for every legal (f, B) pair.
    #[test]
    fn fine_correction_qn_q14_agrees_with_specialised_path() {
        for b in 1..=MAX_FINE_BITS {
            let max_f = (1u32 << b) - 1;
            for f in 0..=max_f {
                let a = fine_correction_q14(f, b) as i64;
                let b_v = fine_correction_qn(f, b, 14);
                assert_eq!(a, b_v, "B={b}, f={f}: q14 vs qn(14) disagree");
            }
        }
    }

    /// `fine_correction_qn` at n = 8 (Q8 to match the coarse-energy
    /// scaffolding) must scale the Q14 correction down by 64. The
    /// Q8 representation's range is [-128, +128]: at B == 8 the
    /// most-positive correction f = 255 rounds up to +128, which
    /// nominally overflows the symmetric [-128, +127] Q8 range. We
    /// document the boundary rather than clamp.
    #[test]
    fn fine_correction_qn_q8_range_and_scale() {
        for b in 1..=MAX_FINE_BITS {
            let max_f = (1u32 << b) - 1;
            for f in 0..=max_f {
                let c8 = fine_correction_qn(f, b, 8);
                // Q8 correction must fit in [-128, +128] (the +128 endpoint
                // only happens at the rounded boundary of the spec's
                // half-open [-1/2, +1/2) range when B == 8 saturates).
                assert!(
                    (-128..=128).contains(&c8),
                    "B={b}, f={f}: q8 correction {c8} out of [-128, 128]"
                );
                // Q14 / 64 = Q8 (because the Q14 result is divisible
                // by 64 whenever B <= 7). At B == 8 the Q8 representation
                // loses one bit of precision so we accept off-by-one in
                // Q8 units (== off-by-64 in Q14 units).
                let c14 = fine_correction_q14(f, b) as i64;
                if b <= 7 {
                    assert_eq!(c14, c8 * 64, "B={b}, f={f}: q14 != q8 * 64");
                } else {
                    assert!(
                        (c14 - c8 * 64).abs() <= 64,
                        "B={b}, f={f}: q14 vs q8*64 differs by more than one Q8 step \
                         (c14={c14}, c8={c8})"
                    );
                }
            }
        }
    }

    /// `finalize_extra_bits` must be a complete no-op when `budget`
    /// is zero or `coded_bands` is zero.
    #[test]
    fn finalize_zero_budget_is_noop() {
        let buf = [0xFFu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let before = dec.tell();
        let r = finalize_extra_bits(
            &mut dec,
            &[FinalizePriority::Zero; NUM_BANDS],
            NUM_BANDS,
            1,
            0,
        );
        assert_eq!(before, dec.tell(), "zero budget consumed bits");
        assert_eq!(r.bits_consumed, 0);
        assert_eq!(r.bits_unused, 0);
        assert_eq!(r.extra_correction_q14, [0i32; NUM_BANDS]);
    }

    #[test]
    fn finalize_zero_coded_bands_is_noop() {
        let buf = [0xFFu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let before = dec.tell();
        let r = finalize_extra_bits(&mut dec, &[FinalizePriority::Zero; NUM_BANDS], 0, 2, 100);
        assert_eq!(before, dec.tell());
        assert_eq!(r.bits_consumed, 0);
        assert_eq!(r.bits_unused, 100);
    }

    /// `finalize_extra_bits` with budget exceeding capacity (mono):
    /// every band gets one extra bit; the surplus is `bits_unused`.
    #[test]
    fn finalize_surplus_left_unused_mono() {
        let buf = [0x00u8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let coded_bands = 5;
        // All bands priority 0 so the walk is the trivial ascending
        // band order.
        let prios = [FinalizePriority::Zero; NUM_BANDS];
        let r = finalize_extra_bits(&mut dec, &prios, coded_bands, 1, /*budget*/ 10);
        // 5 bands × 1 channel = 5 bits consumed; 5 left unused.
        assert_eq!(r.bits_consumed, 5);
        assert_eq!(r.bits_unused, 5);
        // All-zero buffer biases each raw bit to 0; correction(f=0, B=1) = -4096.
        for b in 0..coded_bands {
            assert_eq!(r.extra_correction_q14[b], -4096);
        }
        for b in coded_bands..NUM_BANDS {
            assert_eq!(r.extra_correction_q14[b], 0);
        }
    }

    /// `finalize_extra_bits` with budget exceeding capacity (stereo):
    /// every band gets two extra bits (one per channel), with the
    /// corrections summed.
    #[test]
    fn finalize_surplus_left_unused_stereo_corrections_summed() {
        let buf = [0x00u8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let coded_bands = 5;
        let prios = [FinalizePriority::Zero; NUM_BANDS];
        let r = finalize_extra_bits(&mut dec, &prios, coded_bands, 2, /*budget*/ 100);
        // 5 bands × 2 channels = 10 bits; 90 unused.
        assert_eq!(r.bits_consumed, 10);
        assert_eq!(r.bits_unused, 90);
        // All-zero bits: each channel contributes -4096; summed = -8192.
        for b in 0..coded_bands {
            assert_eq!(r.extra_correction_q14[b], -8192);
        }
    }

    /// `finalize_extra_bits` must spend bits in priority-major order:
    /// priority 0 across all bands first, only then priority 1. Test
    /// by giving a budget that exactly covers the priority-0 set
    /// (mono) and verifying priority-1 bands receive nothing.
    #[test]
    fn finalize_priority_zero_drains_before_priority_one() {
        let buf = [0xFFu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let mut prios = [FinalizePriority::One; NUM_BANDS];
        // Mark bands 1 and 3 as priority 0.
        prios[1] = FinalizePriority::Zero;
        prios[3] = FinalizePriority::Zero;
        let coded_bands = 5;
        let r = finalize_extra_bits(&mut dec, &prios, coded_bands, 1, /*budget*/ 2);
        // Exactly the two priority-0 bands should be filled.
        assert_eq!(r.bits_consumed, 2);
        assert_eq!(r.bits_unused, 0);
        assert_eq!(r.extra_correction_q14[0], 0);
        assert_ne!(r.extra_correction_q14[1], 0, "priority-0 band 1 missed");
        assert_eq!(r.extra_correction_q14[2], 0);
        assert_ne!(r.extra_correction_q14[3], 0, "priority-0 band 3 missed");
        assert_eq!(r.extra_correction_q14[4], 0);
    }

    /// After priority 0 is drained, priority 1 receives bits in
    /// ascending band order.
    #[test]
    fn finalize_priority_one_walks_ascending_after_priority_zero() {
        let buf = [0xFFu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let mut prios = [FinalizePriority::One; NUM_BANDS];
        prios[0] = FinalizePriority::Zero;
        let coded_bands = 5;
        // Budget = 4: priority 0 takes 1 (band 0), priority 1 takes
        // the next 3 (bands 1, 2, 3 in order).
        let r = finalize_extra_bits(&mut dec, &prios, coded_bands, 1, /*budget*/ 4);
        assert_eq!(r.bits_consumed, 4);
        assert_eq!(r.bits_unused, 0);
        assert_ne!(r.extra_correction_q14[0], 0, "priority-0 band 0 missed");
        assert_ne!(r.extra_correction_q14[1], 0, "priority-1 band 1 missed");
        assert_ne!(r.extra_correction_q14[2], 0, "priority-1 band 2 missed");
        assert_ne!(r.extra_correction_q14[3], 0, "priority-1 band 3 missed");
        // Band 4 is priority 1 but the budget ran out before reaching
        // it.
        assert_eq!(
            r.extra_correction_q14[4], 0,
            "band 4 should not have been touched"
        );
    }

    /// `finalize_extra_bits` total raw-bit consumption seen by the
    /// range decoder matches `bits_consumed`.
    #[test]
    fn finalize_bits_consumed_matches_tell_delta() {
        let buf = [0xA3u8, 0x5C, 0xE8, 0x91, 0x42, 0xB7, 0x10, 0x7F];
        let mut dec = RangeDecoder::new(&buf);
        let prios = [FinalizePriority::Zero; NUM_BANDS];
        let before = dec.tell();
        let r = finalize_extra_bits(&mut dec, &prios, 10, 2, /*budget*/ 25);
        let after = dec.tell();
        assert_eq!(after - before, r.bits_consumed);
        // 10 bands × 2 channels = 20 bits possible; budget 25 → 20
        // consumed, 5 unused.
        assert_eq!(r.bits_consumed, 20);
        assert_eq!(r.bits_unused, 5);
    }

    /// `finalize_extra_bits` budget-tight case: budget equals exactly
    /// the priority-0 capacity. Priority-1 bands must remain
    /// untouched.
    #[test]
    fn finalize_budget_exactly_priority_zero_capacity() {
        let buf = [0x00u8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let mut prios = [FinalizePriority::One; NUM_BANDS];
        // 3 bands of priority 0, all in the first 5 coded bands.
        prios[0] = FinalizePriority::Zero;
        prios[2] = FinalizePriority::Zero;
        prios[4] = FinalizePriority::Zero;
        // Mono budget = 3 == priority-0 capacity.
        let r = finalize_extra_bits(&mut dec, &prios, 5, 1, /*budget*/ 3);
        assert_eq!(r.bits_consumed, 3);
        assert_eq!(r.bits_unused, 0);
        for b in [0, 2, 4] {
            assert_ne!(r.extra_correction_q14[b], 0, "priority-0 band {b} missed");
        }
        for b in [1, 3] {
            assert_eq!(
                r.extra_correction_q14[b], 0,
                "priority-1 band {b} should be untouched"
            );
        }
    }

    /// Sanity round-trip: feed the §4.3.2.2 formula a known `f`, then
    /// verify by direct decimal arithmetic that the Q14 result is
    /// within one Q14 step of the true real value.
    #[test]
    fn fine_correction_q14_against_decimal() {
        // (f=2, B=3): correction = (2 + 0.5)/8 - 0.5 = 0.3125 - 0.5 = -0.1875
        // In Q14: -0.1875 * 16384 = -3072.
        assert_eq!(fine_correction_q14(2, 3), -3072);
        // (f=5, B=3): correction = (5 + 0.5)/8 - 0.5 = 0.6875 - 0.5 = +0.1875
        // In Q14: +0.1875 * 16384 = +3072.
        assert_eq!(fine_correction_q14(5, 3), 3072);
        // (f=0, B=1): correction = (0 + 0.5)/2 - 0.5 = 0.25 - 0.5 = -0.25
        // In Q14: -0.25 * 16384 = -4096.
        assert_eq!(fine_correction_q14(0, 1), -4096);
        // (f=1, B=1): correction = (1 + 0.5)/2 - 0.5 = 0.75 - 0.5 = +0.25
        // In Q14: +0.25 * 16384 = +4096.
        assert_eq!(fine_correction_q14(1, 1), 4096);
    }

    /// Smoke test stitching `decode_fine_energy` and
    /// `finalize_extra_bits` on the same range decoder: walks bits
    /// without panics, the two steps respect their separate budgets.
    #[test]
    fn fine_then_finalize_smoke() {
        let buf: Vec<u8> = (0u8..96).cycle().take(128).collect();
        let mut dec = RangeDecoder::new(&buf);
        let bits_per_band: [u32; NUM_BANDS] = [2; NUM_BANDS];
        let before = dec.tell();
        let _ = decode_fine_energy(&mut dec, &bits_per_band);
        let after_fine = dec.tell();
        assert_eq!(after_fine - before, 2 * NUM_BANDS as u32);
        // Now finalize: alternate priorities, mono, 5-bit budget.
        let mut prios = [FinalizePriority::One; NUM_BANDS];
        for b in (0..NUM_BANDS).step_by(2) {
            prios[b] = FinalizePriority::Zero;
        }
        let r = finalize_extra_bits(&mut dec, &prios, NUM_BANDS, 1, 5);
        assert!(!dec.has_error());
        assert_eq!(r.bits_consumed, 5);
        let after_final = dec.tell();
        assert_eq!(after_final - after_fine, 5);
    }
}
