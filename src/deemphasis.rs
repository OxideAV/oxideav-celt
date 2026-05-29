//! CELT de-emphasis filter (RFC 6716 §4.3.7.2).
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
//! This module is decoder-side only.

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
}
