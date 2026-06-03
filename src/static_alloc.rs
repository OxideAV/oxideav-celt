//! CELT static allocation table (RFC 6716 §4.3.3, Table 57).
//!
//! ## What this module covers
//!
//! After the §4.3.3 reservation walk
//! ([`crate::allocation_budget::compute_initial_reservations`]), the
//! band-boost loop ([`crate::band_cap::decode_band_boosts`]), the trim
//! decode ([`crate::bit_allocation::decode_alloc_trim`]), and the
//! §2.6 per-band minimums + trim offsets
//! ([`crate::band_minimums`]), the §4.3.3 allocator selects a
//! starting quality `q` from the **static allocation table** (RFC 6716
//! Table 57). The static allocation in 1/8 bits for a given band/q is:
//!
//! ```text
//!     channels * N * alloc[band][q] << LM >> 2
//! ```
//!
//! where `alloc[band][q]` is in units of 1/32 bit per MDCT bin
//! (RFC 6716 §4.3.3 lines 6223–6229), `N` is the per-channel MDCT-bin
//! count for the band, and `LM` is `log2(frame_size/120)` ∈ `{0,1,2,3}`.
//!
//! ## Table 57 transcription
//!
//! Table 57 is rendered as ASCII art directly inside RFC 6716 §4.3.3
//! (lines 6234–6286 of `docs/audio/opus/rfc6716-opus.txt`). Unlike
//! `e_prob_model` / `cache_caps50` / `LOG2_FRAC_TABLE`, which the RFC
//! delegates to external source files, Table 57 lives entirely inside
//! the RFC body — so we can transcribe its 11 × 21 = 231 cell values
//! directly with the docs/ allow-list satisfied. The CSV staging step
//! is therefore unnecessary; the table is reproduced here verbatim.
//!
//! ## Interpolation
//!
//! The actual per-band static allocation is obtained by linear
//! interpolation between two adjacent `q` columns in steps of 1/64
//! (RFC 6716 §4.3.3 lines 6227–6229). With the lower column `qlo` and
//! the upper column `qlo+1`, and a sub-column position `frac ∈ 0..=63`
//! representing the 1/64 step, the interpolated 1/32-bit-per-bin
//! coefficient is:
//!
//! ```text
//!   interp[band] = ((64 - frac) * alloc[band][qlo]
//!                 +       frac  * alloc[band][qlo+1] + 32) / 64
//! ```
//!
//! (rounding to nearest with a 32-half-bit step to mirror the RFC's
//! "highest allocation that does not exceed the bits remaining"
//! contract; the additive constant +32 is the half-LSB round-to-nearest
//! tweak that keeps the inner search deterministic across forward and
//! reverse evaluation orders).
//!
//! Together with [`band_static_alloc_1_8th`] (which folds in the
//! `channels * N << LM >> 2` scaling), the caller can search the
//! interpolation grid for the highest sub-column position whose total
//! over the coded-band window does not exceed the §4.3.3 remaining
//! budget. The search itself is the topic of a future round; this
//! module provides the table + the per-band evaluators it composes
//! with.

use crate::coarse_energy::NUM_BANDS;

/// CELT static allocation table (RFC 6716 §4.3.3, Table 57).
///
/// Indexed as `STATIC_ALLOC[band][q]`. Units are 1/32 bit per MDCT
/// bin. `band ∈ 0..NUM_BANDS` (21 bands per RFC Table 55), `q ∈ 0..11`
/// (the 11 quality columns).
///
/// Values transcribed verbatim from RFC 6716 §4.3.3 Table 57
/// (`docs/audio/opus/rfc6716-opus.txt` lines 6234–6286).
pub const STATIC_ALLOC: [[u8; NUM_Q]; NUM_BANDS] = [
    // band 0
    [0, 90, 110, 118, 126, 134, 144, 152, 162, 172, 200],
    // band 1
    [0, 80, 100, 110, 119, 127, 137, 145, 155, 165, 200],
    // band 2
    [0, 75, 90, 103, 112, 120, 130, 138, 148, 158, 200],
    // band 3
    [0, 69, 84, 93, 104, 114, 124, 132, 142, 152, 200],
    // band 4
    [0, 63, 78, 86, 95, 103, 113, 123, 133, 143, 200],
    // band 5
    [0, 56, 71, 80, 89, 97, 107, 117, 127, 137, 200],
    // band 6
    [0, 49, 65, 75, 83, 91, 101, 111, 121, 131, 200],
    // band 7
    [0, 40, 58, 70, 78, 85, 95, 105, 115, 125, 200],
    // band 8
    [0, 34, 51, 65, 72, 78, 88, 98, 108, 118, 198],
    // band 9
    [0, 29, 45, 59, 66, 72, 82, 92, 102, 112, 193],
    // band 10
    [0, 20, 39, 53, 60, 66, 76, 86, 96, 106, 188],
    // band 11
    [0, 18, 32, 47, 54, 60, 70, 80, 90, 100, 183],
    // band 12
    [0, 10, 26, 40, 47, 54, 64, 74, 84, 94, 178],
    // band 13
    [0, 0, 20, 31, 39, 47, 57, 67, 77, 87, 173],
    // band 14
    [0, 0, 12, 23, 32, 41, 51, 61, 71, 81, 168],
    // band 15
    [0, 0, 0, 15, 25, 35, 45, 55, 65, 75, 163],
    // band 16
    [0, 0, 0, 4, 17, 29, 39, 49, 59, 69, 158],
    // band 17
    [0, 0, 0, 0, 12, 23, 33, 43, 53, 63, 153],
    // band 18
    [0, 0, 0, 0, 1, 16, 26, 36, 46, 56, 148],
    // band 19
    [0, 0, 0, 0, 0, 10, 15, 20, 30, 45, 129],
    // band 20
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 20, 104],
];

/// Number of quality columns (`q` axis) in [`STATIC_ALLOC`].
///
/// RFC 6716 Table 57 enumerates eleven columns, `q ∈ 0..=10`.
pub const NUM_Q: usize = 11;

/// 1/64-step sub-column count used by the §4.3.3 linear interpolation
/// between adjacent quality columns (RFC 6716 lines 6227–6229).
pub const INTERP_STEPS: u32 = 64;

/// Half-step rounding offset for the interpolation (round-to-nearest
/// with a 1/2-unit additive constant, matching the §4.3.3 "highest
/// allocation that does not exceed the bits remaining" inner-search
/// contract).
const INTERP_ROUND: u32 = INTERP_STEPS / 2;

/// Return the per-band interpolated `alloc` coefficient in units of
/// 1/32 bit per MDCT bin, for a sub-column position `frac ∈ 0..=63`
/// between adjacent quality columns `qlo` and `qlo+1`.
///
/// `qlo + 1` MUST be `<= NUM_Q - 1` (i.e. `qlo ∈ 0..=NUM_Q-2`). The
/// boundary case `frac == 0` yields the exact `STATIC_ALLOC[band][qlo]`
/// entry; `frac == 63` yields a value `(63 * upper + 1 * lower + 32) / 64`
/// just below the upper-column entry; the integer endpoint is reached
/// by stepping `qlo` itself, not by extrapolating `frac == 64`.
///
/// Returns `None` if `band >= NUM_BANDS` or `qlo + 1 >= NUM_Q` or
/// `frac >= INTERP_STEPS`. All three error paths leave the caller in a
/// clean state (no clamping).
pub fn interp_alloc_1_32nd(band: usize, qlo: usize, frac: u32) -> Option<u32> {
    if band >= NUM_BANDS || qlo + 1 >= NUM_Q || frac >= INTERP_STEPS {
        return None;
    }
    let lower = STATIC_ALLOC[band][qlo] as u32;
    let upper = STATIC_ALLOC[band][qlo + 1] as u32;
    let weighted = (INTERP_STEPS - frac) * lower + frac * upper + INTERP_ROUND;
    Some(weighted / INTERP_STEPS)
}

/// Return the §4.3.3 static allocation for one band in **1/8 bits**.
///
/// Computes `channels * N * alloc << LM >> 2` per RFC 6716 §4.3.3
/// (line 6225), where `alloc` is the interpolated 1/32-bit-per-bin
/// coefficient produced by [`interp_alloc_1_32nd`] for the requested
/// `(band, qlo, frac)` position, `N = bins_per_channel`, and `LM` is
/// `log2(frame_size/120)`.
///
/// Returns `None` when [`interp_alloc_1_32nd`] would, or when
/// `channels ∉ {1, 2}`, or when `lm > 3`, or when the multiply would
/// exceed the `u32` range (defensively — the legitimate range is well
/// inside `u32`, but the saturation guard means a caller passing a
/// pathological `bins_per_channel` gets `None` instead of a wrap).
///
/// The shape of the multiply matches the RFC literally: the `<< LM`
/// scales the result by the per-frame-size factor, and the `>> 2`
/// reduces the 1/32-bit units to 1/8-bit units (factor of 4).
pub fn band_static_alloc_1_8th(
    band: usize,
    qlo: usize,
    frac: u32,
    channels: u32,
    bins_per_channel: u32,
    lm: u32,
) -> Option<u32> {
    if !(1..=2).contains(&channels) || lm > 3 {
        return None;
    }
    let alloc = interp_alloc_1_32nd(band, qlo, frac)?;
    // channels (∈ 1..=2) * N (≤ 176 per Table 55 at LM=3) * alloc
    // (≤ 200) ≤ 2 * 176 * 200 = 70_400 < 2^17 before the LM shift.
    // After `<< LM` (LM ≤ 3) the product is ≤ 70_400 * 8 = 563_200 <
    // 2^20, well inside `u32`.
    let raw = channels
        .checked_mul(bins_per_channel)?
        .checked_mul(alloc)?
        .checked_shl(lm)?;
    Some(raw >> 2)
}

/// Convenience: evaluate [`band_static_alloc_1_8th`] across a window of
/// coded bands and accumulate the total.
///
/// `coding_start` is the first band of the window (0 for pure CELT, 17
/// for Hybrid). `bins_per_band` is the per-channel MDCT-bin count for
/// each band in the window (`bins_per_band.len()` is the window
/// length). The total in 1/8 bits is summed across the window, with
/// each band evaluated at the same `(qlo, frac)` interpolation point.
///
/// Returns `None` on the same error paths as
/// [`band_static_alloc_1_8th`], including a window that overflows
/// `NUM_BANDS`.
pub fn window_static_alloc_1_8th(
    coding_start: usize,
    bins_per_band: &[u32],
    qlo: usize,
    frac: u32,
    channels: u32,
    lm: u32,
) -> Option<u32> {
    if coding_start + bins_per_band.len() > NUM_BANDS {
        return None;
    }
    let mut total: u32 = 0;
    for (i, &bins) in bins_per_band.iter().enumerate() {
        let band = coding_start + i;
        let cell = band_static_alloc_1_8th(band, qlo, frac, channels, bins, lm)?;
        total = total.checked_add(cell)?;
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spot-check shape: 21 bands × 11 quality columns per RFC Table 57.
    #[test]
    fn table_shape_matches_rfc() {
        assert_eq!(STATIC_ALLOC.len(), NUM_BANDS);
        for row in STATIC_ALLOC.iter() {
            assert_eq!(row.len(), NUM_Q);
        }
        assert_eq!(NUM_Q, 11);
    }

    /// `q = 0` column is identically zero per the RFC: no static
    /// allocation at the lowest quality level (all bits must come from
    /// boosts / reservations / minimums).
    #[test]
    fn q0_column_is_all_zero() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            assert_eq!(row[0], 0, "band {band} q=0 should be 0");
        }
    }

    /// `q = 10` column saturates at 200 for the first 8 bands (low
    /// frequencies) per RFC Table 57, and decreases for higher bands.
    /// Pin a few representative values to catch transcription errors.
    #[test]
    fn q10_endpoints_match_rfc() {
        // Bands 0..=7 saturate at 200.
        for (band, row) in STATIC_ALLOC.iter().enumerate().take(8) {
            assert_eq!(row[10], 200, "band {band} q=10");
        }
        // Bands 8..=20 fall off according to the RFC.
        assert_eq!(STATIC_ALLOC[8][10], 198);
        assert_eq!(STATIC_ALLOC[9][10], 193);
        assert_eq!(STATIC_ALLOC[10][10], 188);
        assert_eq!(STATIC_ALLOC[11][10], 183);
        assert_eq!(STATIC_ALLOC[12][10], 178);
        assert_eq!(STATIC_ALLOC[13][10], 173);
        assert_eq!(STATIC_ALLOC[14][10], 168);
        assert_eq!(STATIC_ALLOC[15][10], 163);
        assert_eq!(STATIC_ALLOC[16][10], 158);
        assert_eq!(STATIC_ALLOC[17][10], 153);
        assert_eq!(STATIC_ALLOC[18][10], 148);
        assert_eq!(STATIC_ALLOC[19][10], 129);
        assert_eq!(STATIC_ALLOC[20][10], 104);
    }

    /// Each row of Table 57 is monotonically non-decreasing in `q`:
    /// higher quality allocates more bits per bin. This is the spec's
    /// implicit invariant that drives the linear-interpolation search.
    #[test]
    fn rows_monotonic_in_q() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            for q in 1..NUM_Q {
                assert!(
                    row[q] >= row[q - 1],
                    "band {band} q={q} ({}) < q={} ({})",
                    row[q],
                    q - 1,
                    row[q - 1]
                );
            }
        }
    }

    /// `frac = 0` recovers the exact lower-column value.
    #[test]
    fn interp_frac_zero_is_lower_column() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            for (qlo, &lo) in row.iter().enumerate().take(NUM_Q - 1) {
                assert_eq!(
                    interp_alloc_1_32nd(band, qlo, 0),
                    Some(lo as u32),
                    "band {band} qlo {qlo}"
                );
            }
        }
    }

    /// Interpolation lands strictly between (or equal to) the two
    /// flanking columns for any sub-column position.
    #[test]
    fn interp_within_bracket() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            for (qlo, pair) in row.windows(2).enumerate() {
                let lo = pair[0] as u32;
                let hi = pair[1] as u32;
                let (min, max) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                for frac in 0..INTERP_STEPS {
                    let v = interp_alloc_1_32nd(band, qlo, frac).expect("valid (band, qlo, frac)");
                    assert!(
                        v >= min && v <= max + 1,
                        "band {band} qlo {qlo} frac {frac}: {v} not in [{min},{max}+1]"
                    );
                }
            }
        }
    }

    /// Interpolation is monotonically non-decreasing in `frac` whenever
    /// the upper column dominates the lower (i.e. the row is rising).
    #[test]
    fn interp_monotonic_when_row_rising() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            for (qlo, pair) in row.windows(2).enumerate() {
                let lo = pair[0] as u32;
                let hi = pair[1] as u32;
                if hi < lo {
                    continue;
                }
                let mut prev = interp_alloc_1_32nd(band, qlo, 0).unwrap();
                for frac in 1..INTERP_STEPS {
                    let v = interp_alloc_1_32nd(band, qlo, frac).unwrap();
                    assert!(v >= prev, "band {band} qlo {qlo} frac {frac}");
                    prev = v;
                }
            }
        }
    }

    /// Out-of-range inputs return `None` rather than panicking.
    #[test]
    fn interp_rejects_out_of_range() {
        assert_eq!(interp_alloc_1_32nd(NUM_BANDS, 0, 0), None);
        assert_eq!(interp_alloc_1_32nd(0, NUM_Q - 1, 0), None);
        assert_eq!(interp_alloc_1_32nd(0, 0, INTERP_STEPS), None);
    }

    /// `band_static_alloc_1_8th` evaluates the RFC formula directly:
    /// `channels * N * alloc << LM >> 2`. Pick a representative
    /// (band 0, q = 5, channels = 1, N = 4, LM = 0): alloc = 134;
    /// `1 * 4 * 134 << 0 >> 2 = 134`.
    #[test]
    fn band_static_alloc_hand_check() {
        // (band 0, qlo = 5, frac = 0) -> alloc = 134
        let v = band_static_alloc_1_8th(0, 5, 0, 1, 4, 0).unwrap();
        assert_eq!(v, 134);
        // Stereo doubles the channels factor.
        let v_stereo = band_static_alloc_1_8th(0, 5, 0, 2, 4, 0).unwrap();
        assert_eq!(v_stereo, 268);
        // LM = 1 doubles via the `<< LM` term.
        let v_lm1 = band_static_alloc_1_8th(0, 5, 0, 1, 4, 1).unwrap();
        assert_eq!(v_lm1, 268);
    }

    /// `band_static_alloc_1_8th` rejects out-of-range channels / LM.
    #[test]
    fn band_static_alloc_rejects_invalid() {
        assert_eq!(band_static_alloc_1_8th(0, 5, 0, 0, 4, 0), None);
        assert_eq!(band_static_alloc_1_8th(0, 5, 0, 3, 4, 0), None);
        assert_eq!(band_static_alloc_1_8th(0, 5, 0, 1, 4, 4), None);
    }

    /// Window sum matches the manual sum of two adjacent bands for a
    /// representative configuration.
    #[test]
    fn window_static_alloc_sums_window() {
        // Bands 0..2 (window of length 2), channels = 1, LM = 0.
        // bins_per_band = [4, 4]; qlo = 5, frac = 0.
        // alloc[0][5] = 134, alloc[1][5] = 127.
        // band 0: 1 * 4 * 134 << 0 >> 2 = 134
        // band 1: 1 * 4 * 127 << 0 >> 2 = 127
        // total: 261
        let bins = [4u32, 4];
        let total = window_static_alloc_1_8th(0, &bins, 5, 0, 1, 0).unwrap();
        assert_eq!(total, 134 + 127);
    }

    /// Hybrid-mode window starting at band 17 walks the right rows of
    /// Table 57. At q = 9 the cells are 63, 56, 45, 20.
    #[test]
    fn window_static_alloc_hybrid_q9() {
        // bins = 4 per band, channels = 1, LM = 0:
        //   band: channels * N * alloc << LM >> 2
        //       = 1 * 4 * alloc << 0 >> 2
        //       = alloc
        // (the `<< LM` cancels with the `>> 2` for this configuration)
        let bins = [4u32, 4, 4, 4];
        let total = window_static_alloc_1_8th(17, &bins, 9, 0, 1, 0).unwrap();
        // alloc cells (qlo = 9 ⇒ STATIC_ALLOC[band][9]):
        //   band 17 -> 63, band 18 -> 56, band 19 -> 45, band 20 -> 20
        // sum = 184
        assert_eq!(total, 63 + 56 + 45 + 20);
    }

    /// A window that overflows `NUM_BANDS` returns `None`.
    #[test]
    fn window_static_alloc_rejects_overflow() {
        let bins = [4u32; 5];
        // coding_start = 17 + 5 = 22 > 21.
        assert_eq!(window_static_alloc_1_8th(17, &bins, 5, 0, 1, 0), None);
    }

    /// The interpolation rounding constant matches the documented
    /// half-step (so the unit tests above can rely on it).
    #[test]
    fn interp_round_is_half_step() {
        assert_eq!(INTERP_ROUND, 32);
        assert_eq!(INTERP_STEPS, 64);
    }

    /// Cross-check the §4.3.3 prose against a representative 5 ms
    /// frame (LM = 1) stereo configuration. Use band 0 (where the
    /// table caps at 200 at q = 10) and bins_per_channel = 8.
    ///
    /// channels * N * alloc << LM >> 2 = 2 * 8 * 200 << 1 >> 2 = 1600.
    #[test]
    fn band_static_alloc_5ms_stereo() {
        let v = band_static_alloc_1_8th(0, 9, 63, 2, 8, 1).unwrap();
        // frac=63 ≈ upper column; with rounding +32 this rounds toward
        // 200 (the upper column for band 0):
        //   weighted = 1*172 + 63*200 + 32 = 172 + 12600 + 32 = 12804
        //   alloc = 12804 / 64 = 200
        // band_static = 2 * 8 * 200 << 1 >> 2 = 6400 >> 2 + shift…
        // 2 * 8 = 16; 16 * 200 = 3200; << 1 = 6400; >> 2 = 1600.
        assert_eq!(v, 1600);
    }
}
