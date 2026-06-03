//! Per-band shape-allocation minimums + trim offsets (RFC 6716 §4.3.3).
//!
//! ## What this module covers
//!
//! After the §4.3.3 reservation walk (anti-collapse / skip / intensity /
//! dual-stereo, exposed by
//! [`crate::allocation_budget::compute_initial_reservations`]) and the
//! band-boost loop ([`crate::band_cap::decode_band_boosts`]), the
//! §4.3.3 allocator builds two per-coded-band vectors before it walks
//! the Table-57 static-allocation rows:
//!
//! 1. `thresh[band]` — a **hard minimum** shape allocation in 1/8 bit
//!    units, designed to suppress very low-rate per-band allocations
//!    that would otherwise produce an excessively sparse spectrum.
//!    Per RFC 6716 §4.3.3 (lines 6431–6450):
//!
//!    > For each coded band, set thresh[band] to 24 times the number
//!    > of MDCT bins in the band and divide by 16. If 8 times the
//!    > number of channels is greater, use that instead. This sets the
//!    > minimum allocation to one bit per channel or 48 128th bits per
//!    > MDCT bin, whichever is greater. The band-size dependent part
//!    > of this value is not scaled by the channel count, because at
//!    > the very low rates where this limit is applicable there will
//!    > usually be no bits allocated to the side.
//!
//!    [`compute_thresh`] is the bit-exact implementation.
//!
//! 2. `trim_offsets[band]` — a per-coded-band tilt of the allocation
//!    derived from the previously-decoded `alloc.trim` scalar. Per RFC
//!    6716 §4.3.3 (lines 6452–6460):
//!
//!    > For each coded band take the alloc_trim and subtract 5 and LM.
//!    > Then, multiply the result by the number of channels, the
//!    > number of MDCT bins in the shortest frame size for this mode,
//!    > the number of remaining bands, 2**LM, and 8. Next, divide
//!    > this value by 64. Finally, if the number of MDCT bins in the
//!    > band per channel is only one, 8 times the number of channels
//!    > is subtracted in order to diminish the allocation by one bit,
//!    > because width 1 bands receive greater benefit from the coarse
//!    > energy coding.
//!
//!    [`compute_trim_offsets`] implements the prose exactly. The
//!    "number of MDCT bins in the shortest frame size" is the LM=0
//!    column of Table 55, which is exposed here as
//!    [`SHORT_FRAME_BAND_BINS`] for convenience. The "number of
//!    remaining bands" is the §4.3.3 prose phrase for the count of
//!    bands at index >= `band` within the coded-band window
//!    `[coding_start, coding_end)`.
//!
//! Both helpers operate on a caller-supplied `bins_per_band` slice
//! covering the **window of coded bands** for this frame
//! (`bins_per_band.len() == coding_end - coding_start`). The slice
//! holds the per-channel MDCT-bin count at the **actual** frame size
//! (i.e. `BAND_BINS_LM[lm][band]`); `compute_trim_offsets` separately
//! consults `SHORT_FRAME_BAND_BINS` for the LM=0 column the §4.3.3
//! prose explicitly cites.
//!
//! ## Table 55 — MDCT bins per channel per band (RFC 6716 sec 4.3)
//!
//! The §4.3 narrative lists the per-band, per-frame-size MDCT bin
//! counts in Table 55 (`docs/audio/opus/rfc6716-opus.txt` lines
//! 5813–5870). [`BAND_BINS_LM`] reproduces all 4×21 entries as a
//! `[[u32; NUM_BANDS]; 4]`, indexed by `[LM][band]`:
//!
//! * `LM = 0` is the 2.5 ms / 120-sample frame size — also the
//!   "shortest frame size" the §2.6 prose references.
//! * `LM = 1` is the 5 ms / 240-sample frame size.
//! * `LM = 2` is the 10 ms / 480-sample frame size.
//! * `LM = 3` is the 20 ms / 960-sample frame size.
//!
//! ## What is NOT in this module
//!
//! * The Table 57 static-allocation search and the linear interpolation
//!   step between adjacent `q` columns (RFC 6716 §4.3.3 lines 6362–
//!   6388 / clean-room narrative §2.1). The static-allocation table
//!   itself is inlined in RFC 6716; the interpolation pass that
//!   consumes `thresh[]` + `trim_offsets[]` together with the boost +
//!   reservation totals is a separate later round.
//! * The reallocation pass with concurrent skip decoding, the
//!   fine-energy / shape split, and the final balance / priority
//!   reallocation. These come after Table 57 and consume the residual
//!   budget at the end of §4.3.3.
//!
//! ## Clean-room provenance
//!
//! Every numeric value, every formula, and every field comment in
//! this file is transcribed from RFC 6716 §4.3 / §4.3.3
//! (`docs/audio/opus/rfc6716-opus.txt`) and the clean-room narrative
//! at `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.6.
//! No external library source was consulted.

use crate::coarse_energy::NUM_BANDS;

/// Number of `LM` rows covered by [`BAND_BINS_LM`] — the four CELT
/// frame sizes (RFC 6716 §4.3 Table 55).
pub const NUM_LM: usize = 4;

/// Per-frame-size, per-band MDCT bin counts (one channel) — RFC 6716
/// §4.3 Table 55.
///
/// Indexed as `BAND_BINS_LM[lm][band]`:
///
/// * `lm = 0` → 2.5 ms frame, 120 samples per channel.
/// * `lm = 1` → 5 ms frame, 240 samples per channel.
/// * `lm = 2` → 10 ms frame, 480 samples per channel.
/// * `lm = 3` → 20 ms frame, 960 samples per channel.
///
/// Each row sums to `100 << lm` (i.e. `100 / 200 / 400 / 800`), one
/// MDCT bin per 200 Hz step over the 0–20 kHz coded range. The
/// per-channel sample count of the underlying MDCT is `120 << lm`
/// (= `120 / 240 / 480 / 960`) — the 20-bin gap above 20 kHz at LM=0
/// is not band-coded.
///
/// Numeric values are transcribed from RFC 6716 §4.3 Table 55
/// (`docs/audio/opus/rfc6716-opus.txt` lines 5813–5870). Each subsequent
/// `lm` row equals twice the prior row, matching the table's published
/// doubling pattern across the four frame sizes.
pub const BAND_BINS_LM: [[u32; NUM_BANDS]; NUM_LM] = [
    // LM = 0 — 2.5 ms / 120 samples per channel.
    [
        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 6, 6, 8, 12, 18, 22,
    ],
    // LM = 1 — 5 ms / 240 samples per channel.
    [
        2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 12, 12, 16, 24, 36, 44,
    ],
    // LM = 2 — 10 ms / 480 samples per channel.
    [
        4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 24, 24, 32, 48, 72, 88,
    ],
    // LM = 3 — 20 ms / 960 samples per channel.
    [
        8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 48, 48, 64, 96, 144, 176,
    ],
];

/// The LM = 0 column of [`BAND_BINS_LM`] — the §4.3.3 §2.6 prose's
/// "number of MDCT bins in the shortest frame size for this mode".
///
/// Exposed as its own constant so [`compute_trim_offsets`] can index
/// it directly without indirecting through `BAND_BINS_LM[0]`.
pub const SHORT_FRAME_BAND_BINS: [u32; NUM_BANDS] = BAND_BINS_LM[0];

/// 1/8-bit units per whole bit. The §4.3.3 minimum / trim_offset
/// arithmetic emits its outputs in 1/8 bit units, matching the
/// running budget the allocator consumes.
pub const EIGHTH_BIT_QUANTUM: i32 = 8;

/// Compute the per-band hard-minimum shape allocation `thresh[]`
/// (RFC 6716 §4.3.3 lines 6431–6450, in 1/8 bit units).
///
/// For each coded band:
///
/// ```text
/// thresh[band] = max( (24 * N[band]) / 16,  8 * channels )
/// ```
///
/// where:
///
/// * `N[band]` is the per-channel MDCT-bin count for the band at the
///   actual frame size in use (i.e. `BAND_BINS_LM[lm][band]`).
/// * `channels` is `1` (mono) or `2` (stereo).
/// * The result is in 1/8 bit units: the `8 * channels` lower bound
///   is "one bit per channel" expressed as 8th bits, and the
///   `(24 * N) / 16` term is "48 128th bits per MDCT bin" expressed
///   as 8th bits (`48 / 128 = 3 / 8` per bin, scaled by `8 / 8`
///   yields `3 * N / 1 = 24 * N / 8 = 3 * N` 1/8 bits — equivalent to
///   `24 * N / 16` after multiplying numerator and denominator by 2 to
///   keep the integer division rounding match the RFC's stated form
///   exactly).
///
/// `bins_per_band` and `out` must have equal length (one entry per
/// coded band). `channels` must be 1 or 2.
///
/// Returns `false` (and does not modify `out`) if the input is
/// invalid; returns `true` on success.
pub fn compute_thresh(channels: u32, bins_per_band: &[u32], out: &mut [i32]) -> bool {
    if !(1..=2).contains(&channels) {
        return false;
    }
    if bins_per_band.len() != out.len() {
        return false;
    }
    let one_bit_per_channel = (EIGHTH_BIT_QUANTUM as u32).saturating_mul(channels) as i32;
    for (idx, &n) in bins_per_band.iter().enumerate() {
        // (24 * N) / 16 using integer division per the RFC prose.
        let bin_term = (24u32.saturating_mul(n) / 16) as i32;
        out[idx] = bin_term.max(one_bit_per_channel);
    }
    true
}

/// Compute the per-band `trim_offsets[]` vector (RFC 6716 §4.3.3 lines
/// 6452–6460, in 1/8 bit units).
///
/// For each coded band at index `i` into the coded-band window (the
/// window of bands the frame actually codes):
///
/// ```text
/// remaining = bins_per_band.len() - i        // RFC "number of remaining bands"
/// raw      = (alloc_trim - 5 - LM)
///          * channels
///          * SHORT_FRAME_BAND_BINS[absolute_band_index_of_i]
///          * remaining
///          * (1 << LM)
///          * 8
/// trim_offsets[i] = raw / 64
/// if bins_per_band[i] == 1 then trim_offsets[i] -= 8 * channels
/// ```
///
/// Inputs:
///
/// * `alloc_trim`: the §4.3.3 trim integer in `0..=10` (5 = no trim).
/// * `lm`: `log2(frame_size / 120)` in `0..=3`.
/// * `channels`: `1` (mono) or `2` (stereo).
/// * `coding_start`: the index into Table 55 / [`BAND_BINS_LM`] /
///   [`SHORT_FRAME_BAND_BINS`] at which the coded-band window starts.
///   Pure CELT starts at 0; Hybrid mode starts at 17.
/// * `bins_per_band`: per-channel MDCT bin counts for the coded-band
///   window at the **actual** frame size (i.e. `BAND_BINS_LM[lm]`
///   sliced to the coded-band window). Length equals
///   `coding_end - coding_start`.
/// * `out`: receives the per-coded-band trim offset in 1/8 bit units.
///   Length must equal `bins_per_band.len()`.
///
/// The "absolute band index of i" used to look up
/// `SHORT_FRAME_BAND_BINS` is `coding_start + i`.
///
/// Returns `false` and leaves `out` unchanged if any input is out of
/// range; returns `true` on success.
pub fn compute_trim_offsets(
    alloc_trim: i32,
    lm: u32,
    channels: u32,
    coding_start: usize,
    bins_per_band: &[u32],
    out: &mut [i32],
) -> bool {
    if !(0..=10).contains(&alloc_trim) {
        return false;
    }
    if lm > 3 {
        return false;
    }
    if !(1..=2).contains(&channels) {
        return false;
    }
    if bins_per_band.len() != out.len() {
        return false;
    }
    let coded_band_count = bins_per_band.len();
    if coding_start.saturating_add(coded_band_count) > NUM_BANDS {
        return false;
    }
    // (alloc_trim - 5 - LM) as a signed Q0 prefactor; can be negative.
    let trim_prefactor = alloc_trim - 5 - lm as i32;
    let two_pow_lm = 1i64 << lm;
    let one_bit_per_channel = 8i64 * channels as i64;
    for i in 0..coded_band_count {
        let absolute_band = coding_start + i;
        let short_n = SHORT_FRAME_BAND_BINS[absolute_band] as i64;
        let remaining = (coded_band_count - i) as i64;
        // Carry the running product in i64 so the multiplication chain
        // does not overflow at the extremes (LM=3, band 20, full window,
        // alloc_trim=10).
        let raw =
            (trim_prefactor as i64) * (channels as i64) * short_n * remaining * two_pow_lm * 8;
        let mut offset = raw / 64;
        if bins_per_band[i] == 1 {
            offset -= one_bit_per_channel;
        }
        // The product fits comfortably in i32 at every legal input
        // (worst case LM=3 + alloc_trim=10 + band 20 stays under 2^31)
        // but we cast through a saturating step to remain
        // defensive against future band-layout changes.
        out[i] = offset.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 6716 §4.3 Table 55: each frame-size row covers the
    /// 0–20 kHz coded range, summing to `100 << LM` MDCT bins. The
    /// underlying MDCT has `120 << LM` bins per channel; the
    /// remaining `20 << LM` bins above 20 kHz are not band-coded.
    #[test]
    fn band_bins_lm_row_sums_match_coded_range() {
        for (lm, row) in BAND_BINS_LM.iter().enumerate() {
            let row_sum: u32 = row.iter().sum();
            let expected = 100u32 << lm;
            assert_eq!(
                row_sum, expected,
                "BAND_BINS_LM[lm={lm}] row sum {row_sum} != 100 << lm = {expected}"
            );
        }
    }

    /// Each LM row is exactly twice the previous LM row — a direct
    /// consequence of the Table 55 doubling pattern.
    #[test]
    fn band_bins_lm_rows_double_across_frame_sizes() {
        for (lm, row) in BAND_BINS_LM.iter().enumerate().skip(1) {
            let prev = &BAND_BINS_LM[lm - 1];
            for (band, (&this, &prior)) in row.iter().zip(prev.iter()).enumerate() {
                assert_eq!(
                    this,
                    2 * prior,
                    "BAND_BINS_LM[{lm}][{band}] != 2 * BAND_BINS_LM[{}][{band}]",
                    lm - 1
                );
            }
        }
    }

    /// Spot-check a handful of explicit Table 55 cells. Picked from the
    /// RFC's published rows to anchor the constant against
    /// transcription drift.
    #[test]
    fn band_bins_lm_table_55_spot_checks() {
        // Band 0: 1/2/4/8 across LM=0..3.
        assert_eq!(BAND_BINS_LM[0][0], 1);
        assert_eq!(BAND_BINS_LM[1][0], 2);
        assert_eq!(BAND_BINS_LM[2][0], 4);
        assert_eq!(BAND_BINS_LM[3][0], 8);
        // Band 8: 2/4/8/16 across LM=0..3 (the first jump in the table).
        assert_eq!(BAND_BINS_LM[0][8], 2);
        assert_eq!(BAND_BINS_LM[1][8], 4);
        assert_eq!(BAND_BINS_LM[2][8], 8);
        assert_eq!(BAND_BINS_LM[3][8], 16);
        // Band 12: 4/8/16/32 (second jump).
        assert_eq!(BAND_BINS_LM[0][12], 4);
        assert_eq!(BAND_BINS_LM[3][12], 32);
        // Band 15: 6/12/24/48 (six-bin row).
        assert_eq!(BAND_BINS_LM[0][15], 6);
        assert_eq!(BAND_BINS_LM[3][15], 48);
        // Band 17: 8/16/32/64.
        assert_eq!(BAND_BINS_LM[0][17], 8);
        assert_eq!(BAND_BINS_LM[3][17], 64);
        // Band 18: 12/24/48/96.
        assert_eq!(BAND_BINS_LM[0][18], 12);
        assert_eq!(BAND_BINS_LM[3][18], 96);
        // Band 19: 18/36/72/144 (the largest interior band).
        assert_eq!(BAND_BINS_LM[0][19], 18);
        assert_eq!(BAND_BINS_LM[3][19], 144);
        // Band 20: 22/44/88/176 (the largest band, full bandwidth).
        assert_eq!(BAND_BINS_LM[0][20], 22);
        assert_eq!(BAND_BINS_LM[3][20], 176);
    }

    /// [`SHORT_FRAME_BAND_BINS`] must mirror `BAND_BINS_LM[0]` element-wise.
    #[test]
    fn short_frame_band_bins_matches_lm0_row() {
        assert_eq!(SHORT_FRAME_BAND_BINS, BAND_BINS_LM[0]);
    }

    /// At LM=0, mono, every band whose Table-55 bin count is at most
    /// 5 hits the `8 * channels = 8` lower bound; bands with 6 or more
    /// bins emerge from the `(24 * N) / 16` term. Verify both regimes.
    #[test]
    fn compute_thresh_mono_lm0_bounds() {
        let bins = &BAND_BINS_LM[0];
        let mut out = [0i32; NUM_BANDS];
        assert!(compute_thresh(1, bins, &mut out));
        // Bands 0..=14: N ∈ {1, 2, 4}. Each must hit the lower bound (8).
        for &b in &[0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] {
            assert_eq!(
                out[b], 8,
                "band {b} (N={}) expected lower-bound 8, got {}",
                bins[b], out[b]
            );
        }
        // Bands 15, 16: N = 6 → (24 * 6) / 16 = 9, above the lower bound.
        assert_eq!(out[15], 9);
        assert_eq!(out[16], 9);
        // Band 17: N = 8 → (24 * 8) / 16 = 12.
        assert_eq!(out[17], 12);
        // Band 18: N = 12 → (24 * 12) / 16 = 18.
        assert_eq!(out[18], 18);
        // Band 19: N = 18 → (24 * 18) / 16 = 27.
        assert_eq!(out[19], 27);
        // Band 20: N = 22 → (24 * 22) / 16 = 33.
        assert_eq!(out[20], 33);
    }

    /// At LM=0 stereo, the `8 * channels = 16` lower bound applies to
    /// every band where `(24 * N) / 16 < 16`, i.e. `N < 11`. Verify the
    /// regime split.
    #[test]
    fn compute_thresh_stereo_lm0_bounds() {
        let bins = &BAND_BINS_LM[0];
        let mut out = [0i32; NUM_BANDS];
        assert!(compute_thresh(2, bins, &mut out));
        // Bands 0..=17 have N ≤ 8 < 11 → lower bound 16.
        for b in 0..=17usize {
            assert_eq!(
                out[b], 16,
                "stereo LM=0 band {b} (N={}) expected lower-bound 16",
                bins[b]
            );
        }
        // Band 18: N = 12 → (24 * 12) / 16 = 18 > 16.
        assert_eq!(out[18], 18);
        // Band 19: N = 18 → (24 * 18) / 16 = 27.
        assert_eq!(out[19], 27);
        // Band 20: N = 22 → (24 * 22) / 16 = 33.
        assert_eq!(out[20], 33);
    }

    /// At LM=3 mono, every band emerges from the bin term — the lower
    /// bound (8) never applies because the smallest `N` is 8.
    #[test]
    fn compute_thresh_mono_lm3_uses_bin_term() {
        let bins = &BAND_BINS_LM[3];
        let mut out = [0i32; NUM_BANDS];
        assert!(compute_thresh(1, bins, &mut out));
        for (b, &n) in bins.iter().enumerate() {
            let expected = (24u32 * n / 16) as i32;
            assert_eq!(
                out[b], expected,
                "LM=3 band {b}: expected (24*{n})/16 = {expected}, got {}",
                out[b]
            );
            assert!(out[b] >= 8); // monotonicity sanity
        }
    }

    /// Invalid `channels` rejects the call without mutating `out`.
    #[test]
    fn compute_thresh_rejects_invalid_channels() {
        let bins = &BAND_BINS_LM[0];
        let mut out = [-1i32; NUM_BANDS];
        assert!(!compute_thresh(0, bins, &mut out));
        assert!(!compute_thresh(3, bins, &mut out));
        // out untouched (still all -1).
        assert!(out.iter().all(|&v| v == -1));
    }

    /// A length mismatch between `bins_per_band` and `out` rejects.
    #[test]
    fn compute_thresh_rejects_length_mismatch() {
        let bins = &BAND_BINS_LM[0];
        let mut out = [0i32; NUM_BANDS - 1];
        assert!(!compute_thresh(1, bins, &mut out));
    }

    /// At `alloc_trim = 5` and `LM = 0`, the trim prefactor is
    /// `5 - 5 - 0 = 0`. Every per-band offset (before the width-1
    /// adjustment) is zero; the width-1 step subtracts `8 * channels`
    /// from any band with `N = 1` per channel.
    #[test]
    fn compute_trim_offsets_default_trim_width1_adjustment_mono() {
        let bins = &BAND_BINS_LM[0]; // LM=0 has eight width-1 bands at the start.
        let mut out = [0i32; NUM_BANDS];
        assert!(compute_trim_offsets(5, 0, 1, 0, bins, &mut out));
        // Bands 0..=7 (N=1) → 0 - 8*1 = -8.
        for (b, &v) in out.iter().enumerate().take(8) {
            assert_eq!(v, -8, "band {b} expected -8, got {v}");
        }
        // Bands 8..=20 (N >= 2) → 0, no adjustment.
        for (b, &v) in out.iter().enumerate().skip(8) {
            assert_eq!(v, 0, "band {b} expected 0, got {v}");
        }
    }

    /// At `alloc_trim = 5`, `LM = 0`, stereo: trim prefactor 0 means
    /// every offset is 0 except width-1 bands which get `-16`
    /// (`8 * channels = 8 * 2`).
    #[test]
    fn compute_trim_offsets_default_trim_width1_adjustment_stereo() {
        let bins = &BAND_BINS_LM[0];
        let mut out = [0i32; NUM_BANDS];
        assert!(compute_trim_offsets(5, 0, 2, 0, bins, &mut out));
        for &v in out.iter().take(8) {
            assert_eq!(v, -16);
        }
        for &v in out.iter().skip(8) {
            assert_eq!(v, 0);
        }
    }

    /// Worked example: `alloc_trim = 10` (max upward tilt), `LM = 0`,
    /// mono, pure-CELT window. The raw RFC arithmetic for band 0:
    ///   (10 - 5 - 0) * 1 * 1 * 21 * 1 * 8 = 5 * 168 = 840
    ///   trim_offsets[0] = 840 / 64 = 13   (integer division)
    ///   N=1 → subtract 8*1 = 8 → 5.
    /// Band 8 (the first N=2 band):
    ///   5 * 1 * 2 * (21 - 8) * 1 * 8 = 5 * 2 * 13 * 8 = 1040
    ///   1040 / 64 = 16, no width-1 adjustment.
    /// Band 20 (last band, N=22):
    ///   5 * 1 * 22 * 1 * 1 * 8 = 880
    ///   880 / 64 = 13 (integer division), no width-1 adjustment.
    #[test]
    fn compute_trim_offsets_worked_example_trim10_mono_lm0() {
        let bins = &BAND_BINS_LM[0];
        let mut out = [0i32; NUM_BANDS];
        assert!(compute_trim_offsets(10, 0, 1, 0, bins, &mut out));
        // Band 0 — derivation above.
        assert_eq!(out[0], 5);
        // Band 1 — same N=1, one fewer remaining band:
        //   5 * 1 * 1 * 20 * 1 * 8 / 64 = 800 / 64 = 12; - 8 = 4.
        assert_eq!(out[1], 4);
        // Band 8 — first N=2 band.
        assert_eq!(out[8], 16);
        // Band 20 — last band, N=22.
        assert_eq!(out[20], 13);
    }

    /// `alloc_trim < 5` produces negative offsets (bias to lower bands);
    /// `alloc_trim > 5` produces positive offsets (bias to higher bands).
    /// At `alloc_trim = 0`, the offsets for `LM = 0` mono are the
    /// negatives of the `alloc_trim = 10` case, modulo the width-1
    /// adjustment which is independent of the trim value.
    #[test]
    fn compute_trim_offsets_symmetry_around_trim5() {
        let bins = &BAND_BINS_LM[0];
        let mut hi = [0i32; NUM_BANDS];
        let mut lo = [0i32; NUM_BANDS];
        assert!(compute_trim_offsets(10, 0, 1, 0, bins, &mut hi));
        assert!(compute_trim_offsets(0, 0, 1, 0, bins, &mut lo));
        // For each band, hi - width1_step == -(lo - width1_step), where
        // width1_step is the constant -8 for N=1 bands and 0 otherwise.
        for b in 0..NUM_BANDS {
            let width1_step = if bins[b] == 1 { -8 } else { 0 };
            let hi_pre = hi[b] - width1_step;
            let lo_pre = lo[b] - width1_step;
            assert_eq!(
                hi_pre, -lo_pre,
                "band {b}: hi {hi_pre} not symmetric with -lo {}",
                -lo_pre
            );
        }
    }

    /// Hybrid mode codes only bands 17..=20 (`coding_start = 17`). The
    /// "absolute band index" used to index `SHORT_FRAME_BAND_BINS` must
    /// be the absolute index, not the offset within the coded-band
    /// window. Verify by comparing band-17 / band-20 outputs in a
    /// hybrid window against a hand-computed expected value.
    #[test]
    fn compute_trim_offsets_hybrid_window_uses_absolute_band_index() {
        // LM=2 (10 ms), stereo, alloc_trim = 5 (so width-1 step
        // contribution survives but the trim prefactor is zero,
        // making the arithmetic easy to inspect).
        let coded = &BAND_BINS_LM[2][17..21]; // 4 hybrid bands
        let mut out = [0i32; 4];
        assert!(compute_trim_offsets(5, 2, 2, 17, coded, &mut out));
        // trim_prefactor = 5 - 5 - 2 = -2. Width-1 step never fires
        // (smallest N at LM=2 in band 17 is 8). Per band:
        //   raw = -2 * 2 * SHORT[b] * remaining * 4 * 8
        //   offset = raw / 64
        // Band 17: SHORT[17] = 8, remaining = 4 → -2 * 2 * 8 * 4 * 4 * 8 = -4096
        //   -4096 / 64 = -64.
        // Band 18: SHORT[18] = 12, remaining = 3 → -2 * 2 * 12 * 3 * 4 * 8 = -4608
        //   -4608 / 64 = -72.
        // Band 19: SHORT[19] = 18, remaining = 2 → -2 * 2 * 18 * 2 * 4 * 8 = -4608
        //   -4608 / 64 = -72.
        // Band 20: SHORT[20] = 22, remaining = 1 → -2 * 2 * 22 * 1 * 4 * 8 = -2816
        //   -2816 / 64 = -44.
        assert_eq!(out[0], -64);
        assert_eq!(out[1], -72);
        assert_eq!(out[2], -72);
        assert_eq!(out[3], -44);
    }

    /// Invalid inputs reject without mutating `out`.
    #[test]
    fn compute_trim_offsets_rejects_invalid_inputs() {
        let bins = &BAND_BINS_LM[0];
        let mut out = [-1i32; NUM_BANDS];
        // alloc_trim out of range
        assert!(!compute_trim_offsets(-1, 0, 1, 0, bins, &mut out));
        assert!(!compute_trim_offsets(11, 0, 1, 0, bins, &mut out));
        // LM out of range
        assert!(!compute_trim_offsets(5, 4, 1, 0, bins, &mut out));
        // channels out of range
        assert!(!compute_trim_offsets(5, 0, 0, 0, bins, &mut out));
        assert!(!compute_trim_offsets(5, 0, 3, 0, bins, &mut out));
        // Window over-runs Table 55
        let mut out_short = [-1i32; 5];
        let dummy = [1u32; 5];
        assert!(!compute_trim_offsets(5, 0, 1, 17, &dummy, &mut out_short));
        // out length mismatch
        let mut out_mismatch = [0i32; NUM_BANDS - 1];
        assert!(!compute_trim_offsets(5, 0, 1, 0, bins, &mut out_mismatch));
        // The original `out` is unchanged.
        assert!(out.iter().all(|&v| v == -1));
    }

    /// When the coded-band window is empty, both helpers succeed and
    /// produce empty outputs. This is the §4.3.3 degenerate-budget
    /// path where the allocator has no bands to consider.
    #[test]
    fn empty_window_succeeds() {
        let mut t_out: [i32; 0] = [];
        assert!(compute_thresh(1, &[], &mut t_out));
        let mut o_out: [i32; 0] = [];
        assert!(compute_trim_offsets(5, 0, 1, 0, &[], &mut o_out));
    }
}
