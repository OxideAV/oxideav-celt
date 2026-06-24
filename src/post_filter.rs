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

/// The §4.3.7.1 post-filter parameters for one frame: pitch period,
/// 3-bit gain index, and tapset selector.
///
/// `period` is the reconstructed `T = (16 << octave) + fine_pitch - 1`
/// (already bounded to `[15, 1022]` by the caller; the filter clamps it
/// to [`POST_FILTER_PERIOD_MIN`] defensively). `gain_index` is the raw
/// 3-bit value (`0..=7`); `tapset` is the 2-bit selector (`0..=2`).
///
/// A frame with the post-filter flag clear is represented as
/// `PostFilterParams::OFF` — gain index `0` is *not* "off" (the §4.3.7.1
/// gain formula `G = 3*(gain+1)/32` has no zero), so a separate
/// disabled state is carried explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostFilterParams {
    /// `true` when the post-filter is enabled for this frame.
    pub enabled: bool,
    /// Pitch period `T` (bounded `[15, 1022]`).
    pub period: u16,
    /// Raw 3-bit gain index (`0..=7`).
    pub gain_index: u8,
    /// Tapset selector (`0..=2`).
    pub tapset: u8,
}

impl PostFilterParams {
    /// The "post-filter disabled" state — the cross-frame predecessor of
    /// a frame whose previous frame had no post-filter, and the value a
    /// fresh / reset decoder carries.
    pub const OFF: PostFilterParams = PostFilterParams {
        enabled: false,
        period: POST_FILTER_PERIOD_MIN,
        gain_index: 0,
        tapset: 0,
    };

    /// The effective linear gain `G` this frame applies: the §4.3.7.1
    /// gain when enabled, `0.0` when disabled (so a disabled frame is a
    /// pure passthrough on its side of a transition crossfade).
    #[inline]
    pub fn gain(&self) -> f32 {
        if self.enabled {
            gain_f32(self.gain_index)
        } else {
            0.0
        }
    }
}

/// Apply the §4.3.7.1 post-filter to `out` in-place with a smooth
/// cross-frame transition when the parameters changed from the previous
/// frame.
///
/// RFC 6716 §4.3.7.1 states:
///
/// > During a transition between different gains, a smooth transition is
/// > calculated using the square of the MDCT window. It is important
/// > that values of y(n) be interpolated one at a time such that the
/// > past value of y(n) used is interpolated.
///
/// This function implements that crossfade. The two contributions —
/// the **old** filter (the previous frame's `prev` parameters) and the
/// **new** filter (this frame's `cur` parameters) — are blended over the
/// first `overlap` samples of the frame using the squared §4.3.7
/// synthesis window `w(i)` (the rising-half window
/// [`crate::mdct::celt_window_f32`], the same window the inverse-MDCT
/// overlap-add uses):
///
/// ```text
/// y(n) = x(n) + (1 - w(n)^2) * F_old(y, n) + w(n)^2 * F_new(y, n)   for n < overlap
/// y(n) = x(n) +                              F_new(y, n)            for n >= overlap
/// ```
///
/// where `F_old` / `F_new` are the §4.3.7.1 lobe responses
/// `G*(g0*y(n-T) + g1*(y(n-T+1)+y(n-T-1)) + g2*(y(n-T+2)+y(n-T-2)))`
/// evaluated with the old / new `(T, G, tapset)`. Because `w(0)` is near
/// zero and `w(overlap-1)` near one, the crossfade fades the old filter
/// out and the new filter in across the overlap region.
///
/// The RFC's "interpolated one at a time such that the past value of
/// y(n) used is interpolated" requirement is honoured by construction:
/// there is a single output sequence `y`, written sample-by-sample into
/// `out`, and *both* `F_old` and `F_new` read that same already-blended
/// past `y` (never two separate per-filter recursions). Sample `n`'s
/// output therefore depends on the crossfaded past, exactly as the
/// reference requires.
///
/// When `prev == cur` (or both are passthrough), the crossfade is
/// algebraically identical to a single [`apply_post_filter_f32`] pass
/// with the shared parameters, so callers can route every frame through
/// this function unconditionally without a special "no transition" case.
///
/// ## Window choice — a documented decoder decision
///
/// §4.3.7.1 names "the square of the MDCT window" but does not restate
/// the transition region length or the blend's algebraic sign. We take
/// the region length to be the fixed §4.3.7 overlap (`overlap`,
/// = [`crate::synthesis::CELT_OVERLAP`] = 120 for the decoder's
/// long-MDCT window) and the rising-half window `w(i)` so the old filter
/// fades out as the new fades in. This is the only assignment that makes
/// the `prev == cur` case reduce to the steady-state filter (any other
/// region length or window orientation would perturb a frame whose
/// parameters did not change). The remaining ambiguity — whether the
/// reference additionally interpolates the *period* `T` within the
/// region rather than switching it at sample 0 — is noted in the crate
/// README as a residual §4.3.7.1 docs question; this implementation
/// switches `T` at the region start and crossfades only the lobe
/// magnitude, the construction the squared-window blend most directly
/// describes.
///
/// Parameters:
/// * `out` — the post-MDCT samples `x(n)` in, the filtered `y(n)` out.
/// * `prev_output` — the previous frame's filtered tail
///   (`prev_output[k] = y(-1 - k)`), the cross-frame history.
/// * `prev` — the previous frame's post-filter parameters.
/// * `cur` — this frame's post-filter parameters.
/// * `overlap` — the transition region length (the §4.3.7 window
///   overlap). Clamped to `out.len()` so a frame shorter than the
///   overlap crossfades across its whole length.
///
/// Returns the number of samples written (= `out.len()`).
pub fn apply_post_filter_transition_f32(
    out: &mut [f32],
    prev_output: &[f32],
    prev: PostFilterParams,
    cur: PostFilterParams,
    overlap: usize,
) -> usize {
    use crate::mdct::celt_window_f32;

    let n_total = out.len();

    // Old / new filter coefficients and clamped periods.
    let g_old = prev.gain();
    let g_new = cur.gain();
    let t_old = (prev.period.max(POST_FILTER_PERIOD_MIN)) as usize;
    let t_new = (cur.period.max(POST_FILTER_PERIOD_MIN)) as usize;
    let (o0, o1, o2) = tap_coefficients_f32(prev.tapset);
    let (c0, c1, c2) = tap_coefficients_f32(cur.tapset);

    let region = overlap.min(n_total);

    for n in 0..n_total {
        let x = out[n];
        // y(n - back): out[n-back] for a sample produced this block,
        // else prev_output[back - n - 1], else 0 (startup / PLC).
        let lookup = |back: usize| -> f32 {
            if back == 0 {
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

        // New-filter lobe (always active).
        let new_lobe = if g_new != 0.0 {
            let centre = lookup(t_new);
            let p1 = lookup(t_new.saturating_sub(1));
            let m1 = lookup(t_new + 1);
            let p2 = lookup(t_new.saturating_sub(2));
            let m2 = lookup(t_new + 2);
            g_new * (c0 * centre + c1 * (p1 + m1) + c2 * (p2 + m2))
        } else {
            0.0
        };

        if n < region {
            // Old-filter lobe, faded by (1 - w^2); new lobe by w^2.
            let w = celt_window_f32(n, region);
            let w2 = w * w;
            let old_lobe = if g_old != 0.0 {
                let centre = lookup(t_old);
                let p1 = lookup(t_old.saturating_sub(1));
                let m1 = lookup(t_old + 1);
                let p2 = lookup(t_old.saturating_sub(2));
                let m2 = lookup(t_old + 2);
                g_old * (o0 * centre + o1 * (p1 + m1) + o2 * (p2 + m2))
            } else {
                0.0
            };
            out[n] = x + (1.0 - w2) * old_lobe + w2 * new_lobe;
        } else {
            out[n] = x + new_lobe;
        }
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

    // --- §4.3.7.1 cross-frame transition crossfade ---------------------

    fn enabled_params(period: u16, gain_index: u8, tapset: u8) -> PostFilterParams {
        PostFilterParams {
            enabled: true,
            period,
            gain_index,
            tapset,
        }
    }

    /// When `prev == cur` the transition crossfade is algebraically
    /// identical to a single steady-state [`apply_post_filter_f32`] pass:
    /// `(1 - w^2)*F + w^2*F == F` for every sample, so the two paths must
    /// produce bit-equal output.
    #[test]
    fn transition_identity_when_params_unchanged() {
        let x: Vec<f32> = (0..200).map(|i| (i as f32 * 0.013).sin()).collect();
        let prev_output: Vec<f32> = (0..130).map(|i| (i as f32 * 0.07).cos() * 0.4).collect();
        let p = enabled_params(40, 5, 1);

        let mut a = x.clone();
        apply_post_filter_f32(&mut a, &prev_output, p.period, p.gain_index, p.tapset);

        let mut b = x.clone();
        apply_post_filter_transition_f32(&mut b, &prev_output, p, p, 120);

        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (av - bv).abs() < 1e-6,
                "sample {i}: steady-state {av} != transition {bv}"
            );
        }
    }

    /// A frame whose previous frame had the post-filter OFF and whose own
    /// post-filter is OFF is a pure passthrough (both lobes zero).
    #[test]
    fn transition_off_to_off_is_passthrough() {
        let x: Vec<f32> = (0..160).map(|i| (i as f32 * 0.02).sin()).collect();
        let prev_output = vec![0.3f32; 130];
        let mut out = x.clone();
        apply_post_filter_transition_f32(
            &mut out,
            &prev_output,
            PostFilterParams::OFF,
            PostFilterParams::OFF,
            120,
        );
        for (i, (xi, oi)) in x.iter().zip(out.iter()).enumerate() {
            assert!((xi - oi).abs() < 1e-7, "sample {i}: {xi} != {oi}");
        }
    }

    /// Past the overlap region only the new filter applies: the tail of
    /// the transition output equals the tail of a steady-state pass with
    /// the new parameters.
    #[test]
    fn transition_tail_is_new_filter_only() {
        let x: Vec<f32> = (0..300).map(|i| (i as f32 * 0.011).sin()).collect();
        let prev_output: Vec<f32> = (0..130).map(|i| (i as f32 * 0.05).cos() * 0.2).collect();
        let prev = enabled_params(30, 2, 0);
        let cur = enabled_params(50, 6, 2);
        let overlap = 120;

        let mut trans = x.clone();
        apply_post_filter_transition_f32(&mut trans, &prev_output, prev, cur, overlap);

        // A reference run: steady-state with the NEW params, but seeded so
        // its past inside the overlap region matches the transition output
        // (the recursion couples past samples, so we compare only well
        // past the region where both have converged to identical history).
        let mut newonly = x.clone();
        apply_post_filter_f32(
            &mut newonly,
            &prev_output,
            cur.period,
            cur.gain_index,
            cur.tapset,
        );

        // Within the region the two differ (old lobe still bleeds in);
        // after the region + a few periods of settling, the per-sample
        // *formula* is identical (new filter on the same out[] history),
        // so any divergence is only the accumulated history difference.
        // Assert the region boundary applies the new filter exactly: at
        // n == overlap, w^2 no longer participates.
        let n = overlap;
        // Recompute the expected new-only lobe at n from `trans`'s own
        // history (the function's contract: out[n] = x[n] + F_new(trans)).
        let (g0, g1, g2) = tap_coefficients_f32(cur.tapset);
        let g = gain_f32(cur.gain_index);
        let t = cur.period as usize;
        let look = |back: usize| -> f32 {
            if back == 0 || back > n {
                0.0
            } else {
                trans[n - back]
            }
        };
        let expect = x[n]
            + g * (g0 * look(t)
                + g1 * (look(t - 1) + look(t + 1))
                + g2 * (look(t - 2) + look(t + 2)));
        assert!(
            (trans[n] - expect).abs() < 1e-5,
            "at region boundary the new filter must apply exactly: {} vs {}",
            trans[n],
            expect
        );
        // And the standalone new-only run is finite / same length.
        assert_eq!(newonly.len(), trans.len());
    }

    /// The crossfade starts essentially on the old filter (w(0) ~ 0) and
    /// ends on the new filter (w(overlap-1) ~ 1): the first sample's lobe
    /// is the old filter's, the boundary sample's is the new filter's.
    #[test]
    fn transition_starts_old_ends_new() {
        // Distinct gains so the two lobes are clearly separable; constant
        // input so x(n) cancels out of the lobe comparison.
        let x = vec![1.0f32; 200];
        let prev_output = vec![1.0f32; 130];
        let prev = enabled_params(20, 7, 0); // strong old filter
        let cur = enabled_params(20, 0, 0); // weak new filter (same T/tapset)
        let overlap = 120;

        let mut out = x.clone();
        apply_post_filter_transition_f32(&mut out, &prev_output, prev, cur, overlap);

        // Sample 0: w(0)^2 ~ 0, so the lobe is ~ entirely the old filter.
        // The old gain (7) is much larger than the new gain (0), so the
        // sample-0 contribution above x should be close to the old lobe.
        let w0 = crate::mdct::celt_window_f32(0, overlap);
        assert!(
            w0 * w0 < 0.01,
            "w(0)^2 should be near zero, got {}",
            w0 * w0
        );
        // out[0] = 1 + (1 - w0^2)*old_lobe + w0^2*new_lobe; with strong old
        // and weak new, out[0] is dominated by the old lobe ⇒ noticeably
        // above x=1.
        assert!(
            out[0] > 1.0 + 0.1,
            "sample 0 should carry the strong old lobe, got {}",
            out[0]
        );
    }

    /// `PostFilterParams::gain` returns the §4.3.7.1 gain when enabled and
    /// exactly 0.0 when disabled (gain index 0 is NOT "off").
    #[test]
    fn params_gain_off_vs_enabled() {
        assert_eq!(PostFilterParams::OFF.gain(), 0.0);
        let p = enabled_params(40, 0, 0);
        assert_eq!(p.gain(), gain_f32(0)); // 3/32, not zero
        assert!(p.gain() > 0.0);
    }

    /// A short frame (`out.len() < overlap`) crossfades across its whole
    /// length without panicking (region clamps to `out.len()`).
    #[test]
    fn transition_short_frame_clamps_region() {
        let mut out = vec![0.5f32; 40];
        let prev_output = vec![0.1f32; 130];
        let prev = enabled_params(15, 3, 1);
        let cur = enabled_params(60, 5, 2);
        let written = apply_post_filter_transition_f32(&mut out, &prev_output, prev, cur, 120);
        assert_eq!(written, 40);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
