//! CELT post-filter (RFC 6716 §4.3.7.1) — tap coefficients, gain
//! reconstruction, and per-sample filter response.
//!
//! The post-filter is the final stage of the CELT decode pipeline; it
//! is applied after the inverse MDCT + weighted overlap-add. The
//! parameters (`octave`, `period`, `gain` index, `tapset`) are however
//! decoded near the beginning of the frame because they participate in
//! the bit budget — see [`crate::frame_header`].
//!
//! Per §4.3.7.1 the response is
//!
//! ```text
//! y(n) = x(n) + G * ( g0 * y(n - T)
//!                  + g1 * ( y(n - T + 1) + y(n - T - 1) )
//!                  + g2 * ( y(n - T + 2) + y(n - T - 2) ) )
//! ```
//!
//! (The RFC prose lists three tap *sets* whose coefficients are given
//! to ~10 fractional decimal digits and which sum to roughly 1.0 — the
//! tap shapes are symmetric three-point FIR lobes centered on the
//! pitch period `T`. This module reproduces the RFC's listed decimals
//! verbatim.)
//!
//! `T = (16 << octave) + fine_pitch - 1`, bounded in `[15, 1022]`.
//! `G = 3 * (gain + 1) / 32`, so `G ∈ {3/32, 6/32, …, 24/32}` —
//! i.e. `0.09375 … 0.75`.
//!
//! This module is decoder-side only. The encoder picks the parameters
//! and applies the same filter; we leave the choice to a future
//! encoder round.

/// Number of distinct post-filter tap shapes. The §4.3.7.1 tapset
/// selector is a 2-bit ICDF symbol drawn from `{0, 1, 2}` so a
/// well-formed bitstream never feeds a value outside `0..NUM_TAPSETS`.
pub const NUM_TAPSETS: usize = 3;

/// Number of taps in each post-filter shape (the centre tap at `T`
/// plus one tap on either side at `T ± 1` and `T ± 2`).
pub const TAPS_PER_SET: usize = 3;

/// Tap coefficients for the three §4.3.7.1 post-filter shapes,
/// transcribed verbatim from the RFC's decimal listings.
///
/// Rows are tapsets `0..3`, columns are taps `(g0, g1, g2)`:
///
/// * Tapset 0: `g0=0.3066406250, g1=0.2170410156, g2=0.1296386719`.
/// * Tapset 1: `g0=0.4638671875, g1=0.2680664062, g2=0`.
/// * Tapset 2: `g0=0.7998046875, g1=0.1000976562, g2=0`.
///
/// The §4.3.7.1 response weights `g1` and `g2` apply to the *pair*
/// `y(n - T ± 1)` and `y(n - T ± 2)` respectively, so the total
/// contribution of the lobe is `g0 + 2*g1 + 2*g2` ≈ unity:
///
/// * Tapset 0: 0.30664 + 2*0.21704 + 2*0.12964 ≈ 1.00000
/// * Tapset 1: 0.46387 + 2*0.26807 + 0          ≈ 1.00000
/// * Tapset 2: 0.79980 + 2*0.10010 + 0          ≈ 1.00000
///
/// The literals carry the RFC's full decimal precision verbatim;
/// they are rounded to the nearest f32 representation on parse,
/// which is why some Q15 entries differ from the "ideal" rational
/// reconstruction by ±1 ULP. Clippy's `excessive_precision` lint
/// is silenced here intentionally so the source reads the same as
/// the RFC text.
#[allow(clippy::excessive_precision)]
pub const POST_FILTER_TAPS_F32: [[f32; TAPS_PER_SET]; NUM_TAPSETS] = [
    [0.306_640_625, 0.217_041_015_6, 0.129_638_671_9],
    [0.463_867_187_5, 0.268_066_406_2, 0.0],
    [0.799_804_687_5, 0.100_097_656_2, 0.0],
];

/// Same tap coefficients quantized to Q15 (rounded half-up from the
/// f32 representation in [`POST_FILTER_TAPS_F32`]). Each entry equals
/// `round(POST_FILTER_TAPS_F32[set][k] * 32768)` exactly, so callers
/// can switch between the float and fixed-point representations
/// without an additional rounding step.
///
/// Note that the RFC's source decimals (e.g. `0.2170410156`) are
/// themselves 10-digit truncations of the underlying spec coefficient;
/// the nearest f32 representation differs from the unrounded
/// "ideal" Q15 by ±1 ULP in some entries (the f32 hits 7112 for
/// tapset 0 g1, whereas a hypothetical infinite-precision recovery
/// of the RFC's intended ratio `7113/32768` would land at 7113). This
/// is within the spec's 10-decimal precision budget and matches what
/// a pure-f32 implementation would compute on the same inputs.
pub const POST_FILTER_TAPS_Q15: [[i16; TAPS_PER_SET]; NUM_TAPSETS] =
    [[10048, 7112, 4248], [15200, 8784, 0], [26208, 3280, 0]];

/// Pitch-period lower / upper bounds per §4.3.7.1 ("bounded between 15
/// and 1022, inclusively").
pub const POST_FILTER_PERIOD_MIN: u16 = 15;
pub const POST_FILTER_PERIOD_MAX: u16 = 1022;

/// Return the three tap coefficients `(g0, g1, g2)` as f32 for the
/// given tapset. Out-of-range tapsets saturate to the last valid
/// tapset (index `NUM_TAPSETS - 1`) rather than panic; the §4.3.7.1
/// ICDF guarantees a well-formed bitstream never produces a value
/// outside `0..3`, so this clamp is only a defensive guard against
/// caller-side bitstream corruption.
#[inline]
pub fn tap_coefficients_f32(tapset: u8) -> (f32, f32, f32) {
    let idx = (tapset as usize).min(NUM_TAPSETS - 1);
    let row = POST_FILTER_TAPS_F32[idx];
    (row[0], row[1], row[2])
}

/// Return the three tap coefficients `(g0, g1, g2)` in Q15
/// fixed-point for the given tapset, with the same saturation as
/// [`tap_coefficients_f32`].
#[inline]
pub fn tap_coefficients_q15(tapset: u8) -> (i16, i16, i16) {
    let idx = (tapset as usize).min(NUM_TAPSETS - 1);
    let row = POST_FILTER_TAPS_Q15[idx];
    (row[0], row[1], row[2])
}

/// Reconstruct the §4.3.7.1 post-filter gain `G = 3*(gain+1)/32` as an
/// f32. `gain_index` is the 3-bit raw value from the bitstream
/// (`0..=7`); out-of-range values are clamped to 7.
///
/// Returns `0.09375` (`gain=0`) up to `0.75` (`gain=7`).
#[inline]
pub fn gain_f32(gain_index: u8) -> f32 {
    let g = (gain_index as u32).min(7);
    3.0 * (g as f32 + 1.0) / 32.0
}

/// Same as [`gain_f32`] but in Q15 fixed-point (`G_q15 = 3*(gain+1)*1024`).
/// Maximum value `gain=7 ⇒ 24576` (= 0.75 in linear scale). Out-of-range
/// `gain_index` is clamped to 7.
#[inline]
pub fn gain_q15(gain_index: u8) -> u32 {
    let g = (gain_index as u32).min(7);
    3 * (g + 1) * 1024
}

/// Apply a single sample of the §4.3.7.1 post-filter to `x` given the
/// past `history` of already-post-filtered output samples
/// `y(n - 1), y(n - 2), …`, the pitch period `T`, the gain `G`, and
/// the tapset.
///
/// `history[k]` is interpreted as `y(n - 1 - k)` (i.e. index 0 is the
/// most recent past sample). The function returns
///
/// ```text
/// y(n) = x + G * ( g0 * y(n - T)
///                + g1 * ( y(n - T + 1) + y(n - T - 1) )
///                + g2 * ( y(n - T + 2) + y(n - T - 2) ) )
/// ```
///
/// If any required past sample lies before the start of `history`
/// (i.e. `T + offset > history.len()`), that sample is treated as
/// zero. This matches the §4.3.7.1 startup-and-PLC behaviour: the
/// filter quietly degrades to passthrough when history is short
/// rather than reading out of bounds.
///
/// `period` is silently clamped to `POST_FILTER_PERIOD_MIN`; values
/// above `POST_FILTER_PERIOD_MAX` are accepted but unusual and
/// indicate caller-side bitstream corruption.
pub fn filter_sample_f32(x: f32, history: &[f32], period: u16, gain: f32, tapset: u8) -> f32 {
    let (g0, g1, g2) = tap_coefficients_f32(tapset);
    let t = period.max(POST_FILTER_PERIOD_MIN) as usize;

    // y(n - T - k) lives at history[T + k - 1]; y(n - T + k) lives
    // at history[T - k - 1] when T > k, else it's a future sample
    // that hasn't been produced yet (treat as zero per the
    // startup-conditions reading).
    let h = |k_back: usize| -> f32 {
        if k_back == 0 || k_back > history.len() {
            0.0
        } else {
            history[k_back - 1]
        }
    };

    let centre = h(t);
    let p1 = h(t.saturating_sub(1));
    let m1 = h(t + 1);
    let p2 = h(t.saturating_sub(2));
    let m2 = h(t + 2);

    x + gain * (g0 * centre + g1 * (p1 + m1) + g2 * (p2 + m2))
}

/// Apply the §4.3.7.1 post-filter to a contiguous output slice
/// `out`, in-place. `out[i]` enters as `x(i)` (the post-MDCT, pre-
/// post-filter sample) and leaves as `y(i)`. `prev_output` carries
/// the most-recent post-filtered samples from previous calls (with
/// `prev_output[k]` = `y(-1 - k)`), enabling seamless block-by-block
/// processing without re-running the §4.3.7.1 startup transient.
///
/// The function uses an O(T) scratch ring per call indexed off `out`
/// itself for samples produced within the current block and off
/// `prev_output` for samples preceding it. Both ends of the required
/// window (`T-2`, `T-1`, `T`, `T+1`, `T+2`) are looked up consistently
/// regardless of whether they fall before or after the block boundary.
///
/// Returns the number of samples written (= `out.len()`).
pub fn apply_post_filter_f32(
    out: &mut [f32],
    prev_output: &[f32],
    period: u16,
    gain_index: u8,
    tapset: u8,
) -> usize {
    let gain = gain_f32(gain_index);
    let t = period.max(POST_FILTER_PERIOD_MIN) as usize;
    let (g0, g1, g2) = tap_coefficients_f32(tapset);

    // y(n - k) at position `n` lives either at out[n - k] when k <= n
    // (already-filtered sample in this block) or at prev_output[k -
    // n - 1] when k > n (carry-over from a previous block); when both
    // run out we return 0.
    let n_total = out.len();
    for n in 0..n_total {
        let x = out[n];
        let lookup = |back: usize| -> f32 {
            if back == 0 {
                // y(n - 0) hasn't been produced yet; treat as 0 the
                // same way the per-sample form does for the centre
                // tap's k=0 corner case (which §4.3.7.1's formula
                // never actually evaluates).
                return 0.0;
            }
            if back <= n {
                out[n - back]
            } else {
                let pi = back - n - 1;
                if pi < prev_output.len() {
                    prev_output[pi]
                } else {
                    0.0
                }
            }
        };
        let centre = lookup(t);
        let p1 = lookup(t.saturating_sub(1));
        let m1 = lookup(t + 1);
        let p2 = lookup(t.saturating_sub(2));
        let m2 = lookup(t + 2);
        out[n] = x + gain * (g0 * centre + g1 * (p1 + m1) + g2 * (p2 + m2));
    }
    n_total
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    /// The three tap rows are transcribed from §4.3.7.1; the f32
    /// constants must equal the RFC's decimal listings to f32
    /// precision and the Q15 row must equal `round(x * 32768)` for
    /// each f32 entry.
    #[test]
    fn tap_rows_match_rfc_decimals() {
        // Tapset 0: g0=0.3066406250, g1=0.2170410156, g2=0.1296386719.
        assert!((POST_FILTER_TAPS_F32[0][0] - 0.306_640_625).abs() < 1e-9);
        assert!((POST_FILTER_TAPS_F32[0][1] - 0.217_041_015_6).abs() < 1e-9);
        assert!((POST_FILTER_TAPS_F32[0][2] - 0.129_638_671_9).abs() < 1e-9);
        // Tapset 1: g0=0.4638671875, g1=0.2680664062, g2=0.
        assert!((POST_FILTER_TAPS_F32[1][0] - 0.463_867_187_5).abs() < 1e-9);
        assert!((POST_FILTER_TAPS_F32[1][1] - 0.268_066_406_2).abs() < 1e-9);
        assert_eq!(POST_FILTER_TAPS_F32[1][2], 0.0);
        // Tapset 2: g0=0.7998046875, g1=0.1000976562, g2=0.
        assert!((POST_FILTER_TAPS_F32[2][0] - 0.799_804_687_5).abs() < 1e-9);
        assert!((POST_FILTER_TAPS_F32[2][1] - 0.100_097_656_2).abs() < 1e-9);
        assert_eq!(POST_FILTER_TAPS_F32[2][2], 0.0);
    }

    /// The Q15 row must equal `round(f32 * 32768)` for each entry.
    #[test]
    fn q15_taps_match_f32_quantization() {
        for set in 0..NUM_TAPSETS {
            for k in 0..TAPS_PER_SET {
                let f = POST_FILTER_TAPS_F32[set][k];
                let q = POST_FILTER_TAPS_Q15[set][k];
                let expected = (f * 32768.0).round() as i32;
                assert_eq!(
                    q as i32, expected,
                    "tapset {set} tap {k}: q15={q} vs round(f*32768)={expected}"
                );
            }
        }
    }

    /// The §4.3.7.1 prose describes each tap shape as a symmetric
    /// three-point FIR centered on the pitch period. The total weight
    /// `g0 + 2*g1 + 2*g2` should be ≈ 1.0 for every tapset (which is
    /// the unity-gain-at-the-pitch-period property the post-filter
    /// relies on to not change overall loudness).
    #[test]
    fn tap_shapes_sum_to_unity() {
        for (set, row) in POST_FILTER_TAPS_F32.iter().enumerate() {
            let (g0, g1, g2) = (row[0], row[1], row[2]);
            let s = g0 + 2.0 * g1 + 2.0 * g2;
            assert!(
                (s - 1.0).abs() < 5e-4,
                "tapset {set}: total weight {s} is not within tolerance of 1.0"
            );
        }
    }

    /// `tap_coefficients_f32` is a thin lookup; verify the three rows
    /// land at the correct tapset and that `tapset >= 3` saturates to
    /// the last valid tapset (the well-formed bitstream never feeds
    /// that, but the function still must not panic).
    #[test]
    fn tap_coefficients_f32_lookup() {
        assert_eq!(
            tap_coefficients_f32(0),
            (0.306_640_625, 0.217_041_015_6, 0.129_638_671_9)
        );
        assert_eq!(
            tap_coefficients_f32(1),
            (0.463_867_187_5, 0.268_066_406_2, 0.0)
        );
        assert_eq!(
            tap_coefficients_f32(2),
            (0.799_804_687_5, 0.100_097_656_2, 0.0)
        );
        // Saturation: anything >= 3 clamps to the last valid tapset (2).
        assert_eq!(tap_coefficients_f32(3), tap_coefficients_f32(2));
        assert_eq!(tap_coefficients_f32(255), tap_coefficients_f32(2));
    }

    /// Same shape as the f32 version, in Q15.
    #[test]
    fn tap_coefficients_q15_lookup() {
        assert_eq!(tap_coefficients_q15(0), (10048, 7112, 4248));
        assert_eq!(tap_coefficients_q15(1), (15200, 8784, 0));
        assert_eq!(tap_coefficients_q15(2), (26208, 3280, 0));
        assert_eq!(tap_coefficients_q15(3), tap_coefficients_q15(2));
    }

    /// `gain_f32` must satisfy `G = 3*(gain+1)/32` across all 8 raw
    /// indices, with `gain=0 ⇒ 3/32 = 0.09375` and `gain=7 ⇒ 24/32 =
    /// 0.75`.
    #[test]
    fn gain_f32_formula_across_full_range() {
        let expected = [
            3.0 / 32.0,
            6.0 / 32.0,
            9.0 / 32.0,
            12.0 / 32.0,
            15.0 / 32.0,
            18.0 / 32.0,
            21.0 / 32.0,
            24.0 / 32.0,
        ];
        for (i, want) in expected.iter().enumerate() {
            let got = gain_f32(i as u8);
            assert!(
                (got - want).abs() < 1e-9,
                "gain_f32({i})={got} vs expected {want}"
            );
        }
        // Saturation: any 0..=7 value above the legal range clamps to
        // the maximum gain.
        assert_eq!(gain_f32(8), gain_f32(7));
        assert_eq!(gain_f32(255), gain_f32(7));
    }

    /// `gain_q15` is the Q15 form of the same formula; check it
    /// matches `round(gain_f32(i) * 32768)` exactly. The decimal
    /// values land cleanly on integer Q15 representations because
    /// `2^15 / 32 = 1024` is exact.
    #[test]
    fn gain_q15_matches_f32_quantization() {
        for i in 0..=7u8 {
            let q = gain_q15(i);
            let f_to_q = (gain_f32(i) * 32768.0).round() as u32;
            assert_eq!(q, f_to_q, "gain_q15({i})={q} vs round(f*32768)={f_to_q}");
        }
        // Hand-checked corners:
        assert_eq!(gain_q15(0), 3072); // 3 * 1024
        assert_eq!(gain_q15(7), 24576); // 24 * 1024
    }

    /// `filter_sample_f32` with `G=0` returns the dry sample (the
    /// passthrough invariant); important because the encoder picks
    /// `G=0` to mean "post-filter disabled even when the flag is
    /// set".
    ///
    /// Note: the spec's `G = 3*(gain+1)/32` formula never actually
    /// hits 0 (smallest value is 3/32 ≈ 0.094), but the per-sample
    /// helper accepts an arbitrary `gain` argument for the case where
    /// the caller is interpolating the gain across a transition (per
    /// the §4.3.7.1 paragraph on smooth gain transitions during the
    /// MDCT-squared crossfade).
    #[test]
    fn filter_sample_zero_gain_passes_through() {
        let history = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for tapset in 0..3 {
            let y = filter_sample_f32(0.5, &history, 4, 0.0, tapset);
            assert_eq!(y, 0.5);
        }
    }

    /// `filter_sample_f32` with empty history (no past output) MUST
    /// degrade to passthrough — no taps can be applied because every
    /// y(n - k) is treated as zero.
    #[test]
    fn filter_sample_empty_history_passes_through() {
        let y = filter_sample_f32(0.7, &[], 16, 0.5, 1);
        assert_eq!(y, 0.7);
    }

    /// A single-impulse history at the centre tap position should
    /// contribute `gain * g0` to the output for tapset 0, scaled by
    /// the centre coefficient only (because `g1` / `g2` taps land on
    /// zero positions). Use `T = 20` which is comfortably above the
    /// §4.3.7.1 minimum pitch period of 15.
    #[test]
    fn filter_sample_single_centre_impulse() {
        // history index 0 = y(n-1); index k = y(n-1-k).
        // For T=20 the centre tap reads history[T-1] = history[19].
        let mut history = [0.0_f32; 48];
        history[19] = 1.0; // y(n-20) = 1.0
        let g = 0.5;
        let y = filter_sample_f32(0.0, &history, 20, g, 0);
        let (g0, _g1, _g2) = tap_coefficients_f32(0);
        assert!(
            (y - g * g0).abs() < 1e-7,
            "y={y} expected gain*g0={}",
            g * g0
        );
    }

    /// A symmetric impulse pair at `T-1` and `T+1` should pick up the
    /// `2*g1` lobe contribution. Use `T = 24` (above the spec
    /// minimum of 15).
    #[test]
    fn filter_sample_symmetric_g1_lobe() {
        // T=24: centre is history[23], +/-1 lobe is history[22] and history[24].
        let mut history = [0.0_f32; 64];
        history[22] = 1.0; // y(n - 23) = y(n - T + 1)
        history[24] = 1.0; // y(n - 25) = y(n - T - 1)
        let g = 0.25;
        // Tapset 1 has g0=0.4638…, g1=0.2680…, g2=0.
        let y = filter_sample_f32(0.0, &history, 24, g, 1);
        let (_g0, g1, _g2) = tap_coefficients_f32(1);
        let want = g * 2.0 * g1;
        assert!((y - want).abs() < 1e-7, "y={y} vs expected {want}");
    }

    /// Same shape for the `g2` lobe at `T-2` / `T+2`. Use tapset 0 so
    /// `g2` is non-zero, with `T = 30`.
    #[test]
    fn filter_sample_symmetric_g2_lobe() {
        // T=30: centre at history[29], g1 at history[28]/history[30],
        // g2 at history[27]/history[31].
        let mut history = [0.0_f32; 64];
        history[27] = 1.0;
        history[31] = 1.0;
        let g = 0.5;
        let y = filter_sample_f32(0.0, &history, 30, g, 0);
        let (_g0, _g1, g2) = tap_coefficients_f32(0);
        let want = g * 2.0 * g2;
        assert!((y - want).abs() < 1e-7, "y={y} vs expected {want}");
    }

    /// For an entirely zero past output with non-zero input, the
    /// filter is a pure passthrough regardless of tapset or gain
    /// (every tap multiplies a zero history sample).
    #[test]
    fn filter_sample_zero_history_passes_through() {
        for tapset in 0..3 {
            for gain_idx in 0..=7 {
                let g = gain_f32(gain_idx);
                let y = filter_sample_f32(0.42, &[0.0; 32], 100, g, tapset);
                assert!(
                    (y - 0.42).abs() < 1e-7,
                    "tapset={tapset} gain_idx={gain_idx} y={y} expected 0.42"
                );
            }
        }
    }

    /// `apply_post_filter_f32` over an all-zero input + all-zero
    /// previous output produces an all-zero result (nothing to feed
    /// the recursive part of the filter).
    #[test]
    fn apply_post_filter_all_zero() {
        let mut out = [0.0_f32; 64];
        let prev = [0.0_f32; 32];
        let n = apply_post_filter_f32(&mut out, &prev, 50, 3, 1);
        assert_eq!(n, 64);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    /// `apply_post_filter_f32` with `gain_index = 0` (G = 3/32 ≈
    /// 0.094) sees only a very small contribution from each pitch
    /// echo; the output therefore tracks the input within a bounded
    /// envelope. We don't assert a specific waveform here — just that
    /// the operation is finite, doesn't NaN, and that the dry input
    /// is preserved for the first `T-2` samples (no taps fire while
    /// the past window lies before position 0 with empty prev_output).
    #[test]
    fn apply_post_filter_startup_passthrough() {
        let mut out = [0.5_f32; 64];
        let prev: [f32; 0] = [];
        let period = 20;
        apply_post_filter_f32(&mut out, &prev, period, 0, 0);
        // For n < T - 2 = 18, every tap lookup is out-of-bounds on
        // the empty prev_output and out[n - k] hasn't been written
        // yet (it's the dry sample), so the response simplifies to
        // y(n) = x(n) for n < 18.
        let passthrough_len = period as usize - 2;
        for (n, &v) in out.iter().enumerate().take(passthrough_len) {
            assert!((v - 0.5).abs() < 1e-7, "sample {n}: y={v} not passthrough");
        }
        // All values are finite.
        for v in out {
            assert!(v.is_finite());
        }
    }

    /// `POST_FILTER_PERIOD_MIN` / `MAX` constants pin the §4.3.7.1
    /// "bounded between 15 and 1022, inclusively" prose.
    #[test]
    fn period_bounds_match_rfc() {
        assert_eq!(POST_FILTER_PERIOD_MIN, 15);
        assert_eq!(POST_FILTER_PERIOD_MAX, 1022);
    }

    /// `filter_sample_f32` silently clamps `period < 15` up to 15 to
    /// avoid an out-of-band read into the future. We check this by
    /// verifying `period=0` and `period=15` produce identical output
    /// on the same history.
    #[test]
    fn filter_sample_period_clamps_to_minimum() {
        let history: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let y0 = filter_sample_f32(0.1, &history, 0, 0.5, 1);
        let y15 = filter_sample_f32(0.1, &history, 15, 0.5, 1);
        assert!(
            (y0 - y15).abs() < 1e-7,
            "y(period=0)={y0} vs y(period=15)={y15}"
        );
    }
}
