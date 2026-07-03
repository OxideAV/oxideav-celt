//! CELT de-emphasis / pre-emphasis filter pair (RFC 6716 §4.3.7.2).
//!
//! After the post-filter ([`crate::post_filter`]), CELT applies a
//! single-pole IIR de-emphasis filter that is the inverse of the
//! encoder's pre-emphasis filter. Per §4.3.7.2:
//!
//! ```text
//!  1     1            1
//! ---- = ---- = ---------------
//! A(z)   1/A(z)              -1
//!                1 - alpha_p*z
//! ```
//!
//! where `alpha_p = 0.8500061035`. The time-domain form is the
//! one-pole recursion
//!
//! ```text
//! y(n) = x(n) + alpha_p * y(n-1)
//! ```
//!
//! The filter is run continuously across frames; the `y(-1)` state
//! must persist from the last sample of the previous frame into the
//! first sample of the next.
//!
//! ## Encoder side
//!
//! §4.3.7.2 names the decoder filter "the inverse of the pre-emphasis
//! filter used in the encoder" and writes that inverse as `1/A(z)`
//! with `A(z) = 1 - alpha_p*z^-1` — so the encoder-side pre-emphasis
//! is the first-order FIR `A(z)` itself:
//!
//! ```text
//! y(n) = x(n) - alpha_p * x(n-1)
//! ```
//!
//! (§5.3 confirms the direction: "the filters and rotations in the
//! encoder are simply the inverse of the operation performed by the
//! decoder".) [`Preemphasis`] carries the `x(n-1)` memory across
//! frames the same way [`Deemphasis`] carries `y(n-1)`; composing the
//! two in either order is the identity, which the tests pin.

/// De-emphasis pole coefficient as listed in §4.3.7.2. The RFC's
/// decimal `0.8500061035` is rounded to the nearest f32 on parse;
/// the silenced `excessive_precision` lint keeps the source reading
/// like the RFC text rather than f32's lossy truncation.
#[allow(clippy::excessive_precision)]
pub const ALPHA_P_F32: f32 = 0.850_006_103_5;

/// Same coefficient quantized to Q15 fixed-point
/// (`round(alpha_p * 32768) = round(27852.8) = 27853`). The
/// quantization step is ~3e-5, comfortably below the spec's
/// 10-decimal-digit precision; downstream fixed-point implementations
/// pick whichever form matches their arithmetic budget.
pub const ALPHA_P_Q15: u16 = 27853;

/// Persistent state for the §4.3.7.2 de-emphasis filter — just the
/// previous output sample `y(n-1)`. Default-construct at decoder
/// open and after any §4.5.2 decoder reset.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Deemphasis {
    /// `y(n - 1)` carried across calls. Initialized to `0.0` at
    /// decoder open per §4.5.2.
    pub last_y: f32,
}

impl Deemphasis {
    /// Construct a freshly-initialized filter (`y(-1) = 0`).
    pub fn new() -> Self {
        Self { last_y: 0.0 }
    }

    /// Reset the filter state — equivalent to a §4.5.2 decoder reset.
    pub fn reset(&mut self) {
        self.last_y = 0.0;
    }

    /// Apply the filter to a single sample. Returns
    /// `y(n) = x(n) + alpha_p * y(n - 1)` and updates the internal
    /// state.
    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        let y = x + ALPHA_P_F32 * self.last_y;
        self.last_y = y;
        y
    }

    /// Apply the filter to a contiguous slice in-place. `out[i]` enters
    /// as `x(i)` and leaves as `y(i)`. The filter state is updated to
    /// the last `y` produced.
    pub fn apply_in_place(&mut self, out: &mut [f32]) {
        let mut y = self.last_y;
        for x in out.iter_mut() {
            y = *x + ALPHA_P_F32 * y;
            *x = y;
        }
        self.last_y = y;
    }

    /// Apply the filter to `xs` and write the result into `ys`. Panics
    /// if the slices have unequal lengths. Equivalent to copying `xs`
    /// into `ys` and calling [`Self::apply_in_place`] on `ys`.
    pub fn apply(&mut self, xs: &[f32], ys: &mut [f32]) {
        assert_eq!(
            xs.len(),
            ys.len(),
            "deemphasis::apply: xs / ys length mismatch ({} vs {})",
            xs.len(),
            ys.len()
        );
        let mut y = self.last_y;
        for (xi, yi) in xs.iter().zip(ys.iter_mut()) {
            y = *xi + ALPHA_P_F32 * y;
            *yi = y;
        }
        self.last_y = y;
    }
}

/// Convenience: one-shot de-emphasis of an in-place slice starting
/// from a known previous `y(n-1)`. Returns the new `y(n-1)` for the
/// next call.
pub fn deemphasize_in_place_f32(out: &mut [f32], y_prev: f32) -> f32 {
    let mut f = Deemphasis { last_y: y_prev };
    f.apply_in_place(out);
    f.last_y
}

/// Persistent state for the **encoder-side pre-emphasis** filter —
/// the first-order FIR `A(z) = 1 - alpha_p*z^-1` whose inverse
/// `1/A(z)` is the §4.3.7.2 de-emphasis pole:
///
/// ```text
/// y(n) = x(n) - alpha_p * x(n-1)
/// ```
///
/// The only state is the previous **input** sample `x(n-1)` (the FIR
/// tap), carried across frames so the filter runs continuously — the
/// encoder-side mirror of [`Deemphasis::last_y`]. Composing this
/// filter with [`Deemphasis`] in either order reproduces the input
/// exactly (up to f32 rounding), including across frame boundaries
/// with state carry.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Preemphasis {
    /// `x(n - 1)` carried across calls. Initialized to `0.0` at
    /// encoder open (the mirror of the §4.5.2 decoder-reset state).
    pub last_x: f32,
}

impl Preemphasis {
    /// Construct a freshly-initialized filter (`x(-1) = 0`).
    pub fn new() -> Self {
        Self { last_x: 0.0 }
    }

    /// Reset the filter state — the encoder-side mirror of a §4.5.2
    /// decoder reset.
    pub fn reset(&mut self) {
        self.last_x = 0.0;
    }

    /// Apply the filter to a single sample. Returns
    /// `y(n) = x(n) - alpha_p * x(n - 1)` and updates the internal
    /// state to the *input* sample.
    #[inline]
    pub fn step(&mut self, x: f32) -> f32 {
        let y = x - ALPHA_P_F32 * self.last_x;
        self.last_x = x;
        y
    }

    /// Apply the filter to a contiguous slice in-place. `buf[i]`
    /// enters as `x(i)` and leaves as `y(i)`. The filter state is
    /// updated to the last input consumed.
    pub fn apply_in_place(&mut self, buf: &mut [f32]) {
        let mut prev = self.last_x;
        for x in buf.iter_mut() {
            let cur = *x;
            *x = cur - ALPHA_P_F32 * prev;
            prev = cur;
        }
        self.last_x = prev;
    }

    /// Apply the filter to `xs` and write the result into `ys`. Panics
    /// if the slices have unequal lengths. Equivalent to copying `xs`
    /// into `ys` and calling [`Self::apply_in_place`] on `ys`.
    pub fn apply(&mut self, xs: &[f32], ys: &mut [f32]) {
        assert_eq!(
            xs.len(),
            ys.len(),
            "preemphasis::apply: xs / ys length mismatch ({} vs {})",
            xs.len(),
            ys.len()
        );
        let mut prev = self.last_x;
        for (xi, yi) in xs.iter().zip(ys.iter_mut()) {
            *yi = *xi - ALPHA_P_F32 * prev;
            prev = *xi;
        }
        self.last_x = prev;
    }
}

/// Convenience: one-shot pre-emphasis of an in-place slice starting
/// from a known previous input `x(n-1)`. Returns the new `x(n-1)` for
/// the next call.
pub fn preemphasize_in_place_f32(buf: &mut [f32], x_prev: f32) -> f32 {
    let mut f = Preemphasis { last_x: x_prev };
    f.apply_in_place(buf);
    f.last_x
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    /// The §4.3.7.2 coefficient is the RFC's `alpha_p = 0.8500061035`.
    #[test]
    fn alpha_p_matches_rfc() {
        // The constant is the literal RFC decimal.
        assert!((ALPHA_P_F32 - 0.850_006_103_5).abs() < 1e-9);
    }

    /// Q15 quantization must equal `round(alpha_p * 32768)`. The
    /// rounded value is 27853 (= round(27852.8)).
    #[test]
    fn alpha_p_q15_matches_quantization() {
        let q = (ALPHA_P_F32 * 32768.0).round() as u32;
        assert_eq!(q, ALPHA_P_Q15 as u32);
        assert_eq!(ALPHA_P_Q15, 27853);
    }

    /// A fresh filter has `last_y = 0.0` and is the default
    /// constructor's output too.
    #[test]
    fn new_and_default_zero_state() {
        let f1 = Deemphasis::new();
        let f2: Deemphasis = Deemphasis::default();
        assert_eq!(f1.last_y, 0.0);
        assert_eq!(f1, f2);
    }

    /// A single impulse `x = 1.0` at `n=0` with prior state 0 produces
    /// `y(0) = 1.0` and then `y(n) = alpha_p ^ n`. We check the first
    /// 8 samples of that geometric decay.
    #[test]
    fn impulse_response_is_geometric_decay() {
        let mut f = Deemphasis::new();
        let mut out = [0.0_f32; 8];
        out[0] = 1.0;
        f.apply_in_place(&mut out);

        let mut expected = 1.0_f32;
        for (n, y) in out.iter().enumerate() {
            assert!(
                (y - expected).abs() < 1e-5,
                "n={n}: y={y} expected {expected} (alpha^{n})"
            );
            expected *= ALPHA_P_F32;
        }
        // Final state is the last computed y.
        assert!((f.last_y - expected / ALPHA_P_F32).abs() < 1e-5);
    }

    /// Zero input + zero initial state stays zero forever.
    #[test]
    fn zero_input_zero_state_stays_zero() {
        let mut f = Deemphasis::new();
        let mut out = [0.0_f32; 32];
        f.apply_in_place(&mut out);
        for y in out {
            assert_eq!(y, 0.0);
        }
        assert_eq!(f.last_y, 0.0);
    }

    /// A DC input `x(n) = 1.0` converges geometrically toward
    /// `1 / (1 - alpha_p)` ≈ `6.667` (since the steady-state of
    /// `y = x + alpha * y` is `x / (1 - alpha)`). Verify the
    /// long-time response lands close to that limit.
    #[test]
    fn dc_input_converges_to_steady_state() {
        let mut f = Deemphasis::new();
        let mut out = [1.0_f32; 4096];
        f.apply_in_place(&mut out);
        let ss = 1.0 / (1.0 - ALPHA_P_F32);
        // The 4096th sample is essentially fully converged
        // (alpha^4096 is astronomically small).
        let last = *out.last().unwrap();
        assert!((last - ss).abs() < 1e-2, "last={last} vs steady-state {ss}");
    }

    /// The filter is **continuous across frames** — running it on two
    /// halves separately with state-carry must equal running it on
    /// the joined buffer in one call.
    #[test]
    fn frame_continuity_via_state_carry() {
        let xs: Vec<f32> = (0..256).map(|i| (i as f32 * 0.013).sin()).collect();

        // One-shot reference.
        let mut ref_buf = xs.clone();
        let mut f_ref = Deemphasis::new();
        f_ref.apply_in_place(&mut ref_buf);

        // Two-shot with carry.
        let mut split_a = xs[..128].to_vec();
        let mut split_b = xs[128..].to_vec();
        let mut f_split = Deemphasis::new();
        f_split.apply_in_place(&mut split_a);
        f_split.apply_in_place(&mut split_b);

        for (i, (r, s)) in ref_buf
            .iter()
            .zip(split_a.iter().chain(split_b.iter()))
            .enumerate()
        {
            assert!(
                (r - s).abs() < 1e-5,
                "sample {i}: one-shot={r} two-shot={s}"
            );
        }
        // Final state agrees too.
        assert!((f_ref.last_y - f_split.last_y).abs() < 1e-5);
    }

    /// `step()` and `apply_in_place()` are equivalent computations on
    /// the same input + state.
    #[test]
    fn step_matches_apply_in_place() {
        let mut f_step = Deemphasis::new();
        let mut f_slice = Deemphasis::new();
        let xs = [-0.7_f32, 0.3, 0.1, -0.4, 0.6, 0.0, 0.9, -0.2];

        let mut slice = xs;
        f_slice.apply_in_place(&mut slice);

        let mut step_out = [0.0_f32; 8];
        for (xi, yi) in xs.iter().zip(step_out.iter_mut()) {
            *yi = f_step.step(*xi);
        }
        for i in 0..8 {
            assert!(
                (step_out[i] - slice[i]).abs() < 1e-7,
                "i={i}: step={} apply={}",
                step_out[i],
                slice[i]
            );
        }
        assert!((f_step.last_y - f_slice.last_y).abs() < 1e-7);
    }

    /// `apply()` writes into `ys` without modifying `xs`.
    #[test]
    fn apply_writes_to_ys_keeps_xs() {
        let xs = [0.1_f32, 0.2, 0.3, 0.4];
        let xs_copy = xs;
        let mut ys = [0.0_f32; 4];
        let mut f = Deemphasis::new();
        f.apply(&xs, &mut ys);
        assert_eq!(xs, xs_copy);
        // First sample is just xs[0] (state was 0).
        assert!((ys[0] - 0.1).abs() < 1e-6);
        // Reference manually.
        let mut g = Deemphasis::new();
        let mut ref_buf = xs;
        g.apply_in_place(&mut ref_buf);
        for i in 0..4 {
            assert!((ys[i] - ref_buf[i]).abs() < 1e-7);
        }
    }

    /// `apply()` panics on length mismatch.
    #[test]
    #[should_panic(expected = "deemphasis::apply: xs / ys length mismatch")]
    fn apply_length_mismatch_panics() {
        let xs = [1.0_f32, 2.0, 3.0];
        let mut ys = [0.0_f32; 2];
        let mut f = Deemphasis::new();
        f.apply(&xs, &mut ys);
    }

    /// `reset()` zeros the state.
    #[test]
    fn reset_zeros_state() {
        let mut f = Deemphasis::new();
        let _ = f.step(1.0);
        let _ = f.step(0.5);
        assert!(f.last_y != 0.0);
        f.reset();
        assert_eq!(f.last_y, 0.0);
    }

    /// `deemphasize_in_place_f32` is a one-shot convenience wrapper —
    /// verify it matches the stateful filter when given the same
    /// `y_prev`.
    #[test]
    fn convenience_wrapper_matches_stateful() {
        let xs: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let mut a = xs.clone();
        let mut b = xs;
        let mut f = Deemphasis { last_y: 0.25 };
        f.apply_in_place(&mut a);
        let new_state = deemphasize_in_place_f32(&mut b, 0.25);
        for i in 0..64 {
            assert!((a[i] - b[i]).abs() < 1e-7, "sample {i} differs");
        }
        assert!((f.last_y - new_state).abs() < 1e-7);
    }

    /// All-finite invariant under bounded input: random-ish input
    /// must not produce NaN / inf at any sample.
    #[test]
    fn finite_output_for_bounded_input() {
        let mut f = Deemphasis::new();
        let mut out: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.137).sin()).collect();
        f.apply_in_place(&mut out);
        for y in out {
            assert!(y.is_finite());
        }
    }

    // ---- Pre-emphasis (encoder side of the §4.3.7.2 pair) ----

    /// The pre-emphasis impulse response is the FIR `[1, -alpha_p, 0,
    /// 0, ...]` — the literal `A(z) = 1 - alpha_p*z^-1` taps.
    #[test]
    fn preemphasis_impulse_response_is_fir_taps() {
        let mut f = Preemphasis::new();
        let mut out = [0.0_f32; 8];
        out[0] = 1.0;
        f.apply_in_place(&mut out);
        assert!((out[0] - 1.0).abs() < 1e-7);
        assert!((out[1] + ALPHA_P_F32).abs() < 1e-7);
        for (n, y) in out.iter().enumerate().skip(2) {
            assert_eq!(*y, 0.0, "FIR tail must be exactly zero at n={n}");
        }
        // State holds the last *input* (zero after the impulse).
        assert_eq!(f.last_x, 0.0);
    }

    /// Pre-emphasis then de-emphasis is the identity, continuously
    /// across frame boundaries with independent state carry on each
    /// side — the §4.3.7.2 "inverse of the pre-emphasis filter"
    /// relation, encoder → decoder direction.
    #[test]
    fn preemphasis_then_deemphasis_is_identity() {
        let xs: Vec<f32> = (0..512)
            .map(|i| (i as f32 * 0.0173).sin() * 0.8 + (i as f32 * 0.0041).cos() * 0.15)
            .collect();
        let mut pre = Preemphasis::new();
        let mut de = Deemphasis::new();
        let mut recovered = Vec::with_capacity(xs.len());
        // Uneven frame split to exercise the state carry.
        for chunk in xs.chunks(120) {
            let mut buf = chunk.to_vec();
            pre.apply_in_place(&mut buf);
            de.apply_in_place(&mut buf);
            recovered.extend_from_slice(&buf);
        }
        for (i, (x, r)) in xs.iter().zip(&recovered).enumerate() {
            assert!(
                (x - r).abs() < 1e-5,
                "identity broken at {i}: {x} vs {r} (pre→de)"
            );
        }
    }

    /// De-emphasis then pre-emphasis is also the identity (the filters
    /// commute as exact inverses), across frame boundaries.
    #[test]
    fn deemphasis_then_preemphasis_is_identity() {
        let xs: Vec<f32> = (0..512).map(|i| (i as f32 * 0.031).sin() * 0.5).collect();
        let mut de = Deemphasis::new();
        let mut pre = Preemphasis::new();
        let mut recovered = Vec::with_capacity(xs.len());
        for chunk in xs.chunks(97) {
            let mut buf = chunk.to_vec();
            de.apply_in_place(&mut buf);
            pre.apply_in_place(&mut buf);
            recovered.extend_from_slice(&buf);
        }
        for (i, (x, r)) in xs.iter().zip(&recovered).enumerate() {
            assert!(
                (x - r).abs() < 1e-4,
                "identity broken at {i}: {x} vs {r} (de→pre)"
            );
        }
    }

    /// A DC input settles to `(1 - alpha_p) * DC` after the first
    /// sample (the FIR high-pass response at DC).
    #[test]
    fn preemphasis_dc_response() {
        let mut f = Preemphasis::new();
        let mut out = [1.0_f32; 16];
        f.apply_in_place(&mut out);
        assert!((out[0] - 1.0).abs() < 1e-7, "first sample sees x(-1) = 0");
        let dc_gain = 1.0 - ALPHA_P_F32;
        for (n, y) in out.iter().enumerate().skip(1) {
            assert!(
                (y - dc_gain).abs() < 1e-6,
                "n={n}: y={y} expected DC gain {dc_gain}"
            );
        }
        assert_eq!(f.last_x, 1.0);
    }

    /// Frame continuity: two-shot with state carry equals one-shot.
    #[test]
    fn preemphasis_frame_continuity_via_state_carry() {
        let xs: Vec<f32> = (0..256).map(|i| (i as f32 * 0.017).sin()).collect();
        let mut ref_buf = xs.clone();
        let mut f_ref = Preemphasis::new();
        f_ref.apply_in_place(&mut ref_buf);

        let mut split_a = xs[..100].to_vec();
        let mut split_b = xs[100..].to_vec();
        let mut f_split = Preemphasis::new();
        f_split.apply_in_place(&mut split_a);
        f_split.apply_in_place(&mut split_b);

        for (i, (r, s)) in ref_buf
            .iter()
            .zip(split_a.iter().chain(split_b.iter()))
            .enumerate()
        {
            assert!(
                (r - s).abs() < 1e-7,
                "sample {i}: one-shot={r} two-shot={s}"
            );
        }
        assert_eq!(f_ref.last_x, f_split.last_x);
    }

    /// `step()`, `apply()`, `apply_in_place()`, and the one-shot
    /// wrapper are equivalent computations.
    #[test]
    fn preemphasis_entry_points_agree() {
        let xs = [-0.7_f32, 0.3, 0.1, -0.4, 0.6, 0.0, 0.9, -0.2];

        let mut slice = xs;
        let mut f_slice = Preemphasis { last_x: 0.2 };
        f_slice.apply_in_place(&mut slice);

        let mut f_step = Preemphasis { last_x: 0.2 };
        let step_out: Vec<f32> = xs.iter().map(|&x| f_step.step(x)).collect();

        let mut ys = [0.0_f32; 8];
        let mut f_pair = Preemphasis { last_x: 0.2 };
        f_pair.apply(&xs, &mut ys);

        let mut oneshot = xs;
        let new_state = preemphasize_in_place_f32(&mut oneshot, 0.2);

        for i in 0..8 {
            assert!((step_out[i] - slice[i]).abs() < 1e-7);
            assert!((ys[i] - slice[i]).abs() < 1e-7);
            assert!((oneshot[i] - slice[i]).abs() < 1e-7);
        }
        assert_eq!(f_step.last_x, f_slice.last_x);
        assert_eq!(f_pair.last_x, f_slice.last_x);
        assert_eq!(new_state, f_slice.last_x);
    }

    /// `apply()` panics on length mismatch (mirror of the de-emphasis
    /// contract).
    #[test]
    #[should_panic(expected = "preemphasis::apply: xs / ys length mismatch")]
    fn preemphasis_apply_length_mismatch_panics() {
        let xs = [1.0_f32, 2.0, 3.0];
        let mut ys = [0.0_f32; 2];
        let mut f = Preemphasis::new();
        f.apply(&xs, &mut ys);
    }

    /// `reset()` zeros the FIR memory; `new()` == `default()`.
    #[test]
    fn preemphasis_reset_and_default() {
        let mut f = Preemphasis::new();
        assert_eq!(f, Preemphasis::default());
        let _ = f.step(0.7);
        assert!(f.last_x != 0.0);
        f.reset();
        assert_eq!(f.last_x, 0.0);
    }
}
