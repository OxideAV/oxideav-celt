//! Spreading rotation (RFC 6716 §4.3.4.3, page 117 of
//! `docs/audio/opus/rfc6716-opus.txt`).
//!
//! ## What this module covers
//!
//! Once the §4.3.4.2 PVQ shape decoder produces a unit-norm vector
//! `X[0..N]` and the §4.3.4.3 [`Spread`](crate::spread::Spread) field
//! has been decoded, the decoder applies a rotation to that vector to
//! avoid tonal artifacts. The rotation has three components per
//! RFC 6716 §4.3.4.3 (page 117):
//!
//! 1. A rotation gain `g_r = N / (N + f_r * K)` and a rotation angle
//!    `theta = pi * g_r^2 / 4`.
//! 2. An N-D rotation expressed as a chain of 2-D rotations:
//!    `R(x_1, x_2), R(x_2, x_3), ..., R(x_{N-1}, x_N), R(x_{N-2}, x_{N-1}),
//!    ..., R(x_1, x_2)`. The chain walks the band forward, then walks
//!    it back the same way ("back and forth"). Each 2-D rotation is
//!    the standard `cos/sin` block.
//! 3. If the decoded vector spans more than one time block AND each
//!    block represents 8 samples or more, an "extra" rotation by
//!    `(pi/2 - theta)` is applied BEFORE the main chain. The extra
//!    rotation runs in an interleaved manner with stride
//!    `round(sqrt(N / nb_blocks))`, i.e. independently on each
//!    sub-sequence `S_k = { stride * n + k }`, `n = 0..N/stride - 1`.
//!
//! Per the §4.3.4.3 prose, when the band spans more than one time
//! block, "this spreading process is applied separately on each time
//! block" — the main rotation chain runs per block.
//!
//! ## What is NOT in this module
//!
//! * The §4.3.4.2 PVQ shape decode itself: this module consumes a
//!   `&mut [f32]` unit-norm vector produced by
//!   [`pvq::decode_unit_shape`](crate::pvq::decode_unit_shape).
//! * The §4.3.4.4 split decoder: that recurses on the spreading
//!   rotation per split; this module decorates a single band level.
//! * The §4.3.5 anti-collapse pass: it runs AFTER the spreading
//!   rotation and is a separate step.
//! * The §4.3.4.5 time-frequency change WHT, already in
//!   [`hadamard`](crate::hadamard).
//!
//! ## Clean-room provenance
//!
//! Every formula and every ordering decision in this module is
//! transcribed from RFC 6716 §4.3.4.3 (the four-paragraph block on
//! page 117 of `docs/audio/opus/rfc6716-opus.txt`). No external
//! implementation was consulted.

use crate::spread::{pre_rotation_stride, rotation_gain_squared_ratio, Spread};

/// Minimum per-block sample count that enables the §4.3.4.3 "extra
/// rotation" by `(pi/2 - theta)`.
///
/// Per RFC 6716 §4.3.4.3 page 117: "if each block represents 8
/// samples or more, then another N-D rotation, by (pi/2-theta), is
/// applied _before_ the rotation described above."
pub const EXTRA_ROTATION_MIN_BLOCK_SAMPLES: u32 = 8;

/// Compute the rotation angle `theta = pi * g_r^2 / 4` for a band of
/// `N` dimensions and `K` pulses under the supplied spreading mode.
///
/// Returns `0.0` for [`Spread::None`] (identity rotation) and for the
/// degenerate `N = 0` band. For non-degenerate inputs the rotation
/// gain is `g_r = N / (N + f_r * K)`, and the spec computes
/// `theta = pi * g_r^2 / 4`.
///
/// The angle is computed in `f64` to keep the squaring and the
/// division well below `f32` precision loss at the band sizes in
/// play; the rotation primitives themselves operate on `f32`.
pub fn rotation_angle_f64(spread: Spread, n: u32, k: u32) -> f64 {
    let (g2_num, g2_den) = rotation_gain_squared_ratio(spread, n, k);
    if g2_num == 0 || g2_den == 0 {
        return 0.0;
    }
    let g2 = (g2_num as f64) / (g2_den as f64);
    core::f64::consts::PI * g2 * 0.25
}

/// Apply a single 2-D rotation `R(i, j)` to a pair of samples.
///
/// Per RFC 6716 §4.3.4.3 page 117:
///
/// ```text
///     x_i' =  cos(theta) * x_i + sin(theta) * x_j
///     x_j' = -sin(theta) * x_i + cos(theta) * x_j
/// ```
///
/// `cos_theta` and `sin_theta` are precomputed once per band by the
/// orchestrator so the inner loop avoids redundant trig.
#[inline]
pub fn apply_2d_rotation(x_i: f32, x_j: f32, cos_theta: f32, sin_theta: f32) -> (f32, f32) {
    let xi = cos_theta * x_i + sin_theta * x_j;
    let xj = -sin_theta * x_i + cos_theta * x_j;
    (xi, xj)
}

/// Apply the §4.3.4.3 N-D rotation to a single time block of length
/// `len` (`samples.len()`).
///
/// The rotation chain is
/// `R(x_1, x_2), R(x_2, x_3), ..., R(x_{N-1}, x_N), R(x_{N-2}, x_{N-1}),
/// ..., R(x_1, x_2)` — forward pass over indices `0..len-1`, then a
/// reverse pass over indices `len-2..0`. The 2-D rotations on
/// adjacent index pairs run in sequence; each modifies the pair
/// in-place and the modifications feed forward into the next pair.
///
/// `theta == 0.0` is a no-op; the caller can also gate on
/// [`Spread::None`] before invoking this function.
///
/// `len < 2` is a no-op (a single-element vector cannot be rotated
/// pairwise).
pub fn apply_nd_rotation(samples: &mut [f32], theta: f64) {
    if theta == 0.0 || samples.len() < 2 {
        return;
    }
    let cos_theta = theta.cos() as f32;
    let sin_theta = theta.sin() as f32;
    let n = samples.len();

    // Forward pass: R(x_0, x_1), R(x_1, x_2), ..., R(x_{n-2}, x_{n-1}).
    // RFC indexing uses 1-based subscripts; we walk 0..n-1.
    for i in 0..n - 1 {
        let (xi, xj) = apply_2d_rotation(samples[i], samples[i + 1], cos_theta, sin_theta);
        samples[i] = xi;
        samples[i + 1] = xj;
    }
    // Reverse pass: R(x_{n-2}, x_{n-1}), ..., R(x_0, x_1).
    // RFC writes "R(x_N-2, X_N-1), ..., R(x_1, x_2)", i.e. mirrored
    // pairs of the forward pass with the last pair (R(x_{n-1}, x_n))
    // omitted from the reverse leg.
    for i in (0..n - 1).rev() {
        let (xi, xj) = apply_2d_rotation(samples[i], samples[i + 1], cos_theta, sin_theta);
        samples[i] = xi;
        samples[i + 1] = xj;
    }
}

/// Apply the §4.3.4.3 N-D rotation independently to each of
/// `nb_blocks` interleaved time blocks.
///
/// Per RFC 6716 §4.3.4.3 page 117: "If the decoded vector represents
/// more than one time block, then this spreading process is applied
/// separately on each time block."
///
/// The §4.3.4.3 prose does not pin the layout of the time blocks
/// within the per-band sample buffer. The other §4.3.4 routines that
/// handle multi-block bands (e.g. the §4.3.4.5 Hadamard transform's
/// `apply_tf_resolution_change`) interleave the blocks across the
/// band — index `i` in block `b` maps to `samples[b + nb_blocks * i]`.
/// We follow the same layout convention so the rotation composes with
/// the rest of the §4.3.4 path without extra reshuffling.
///
/// Returns `false` and leaves `samples` unchanged when the shape
/// constraint is violated (`nb_blocks == 0`, or `samples.len()` not
/// divisible by `nb_blocks`).
pub fn apply_nd_rotation_multi_block(samples: &mut [f32], nb_blocks: u32, theta: f64) -> bool {
    if nb_blocks == 0 || samples.is_empty() {
        return false;
    }
    let nb = nb_blocks as usize;
    if samples.len() % nb != 0 {
        return false;
    }
    if theta == 0.0 {
        return true;
    }
    let per_block = samples.len() / nb;
    if per_block < 2 {
        return true;
    }

    // Reuse the same precomputed cos/sin across blocks.
    let cos_theta = theta.cos() as f32;
    let sin_theta = theta.sin() as f32;

    let mut block = vec![0f32; per_block];
    for b in 0..nb {
        // Gather block `b` from the interleaved buffer.
        for i in 0..per_block {
            block[i] = samples[b + nb * i];
        }
        // Forward pass.
        for i in 0..per_block - 1 {
            let (xi, xj) = apply_2d_rotation(block[i], block[i + 1], cos_theta, sin_theta);
            block[i] = xi;
            block[i + 1] = xj;
        }
        // Reverse pass.
        for i in (0..per_block - 1).rev() {
            let (xi, xj) = apply_2d_rotation(block[i], block[i + 1], cos_theta, sin_theta);
            block[i] = xi;
            block[i + 1] = xj;
        }
        // Scatter back.
        for i in 0..per_block {
            samples[b + nb * i] = block[i];
        }
    }
    true
}

/// Apply the §4.3.4.3 "extra" pre-rotation by `(pi/2 - theta)`.
///
/// Per RFC 6716 §4.3.4.3 page 117: "if each block represents 8 samples
/// or more, then another N-D rotation, by (pi/2-theta), is applied
/// _before_ the rotation described above. This extra rotation is
/// applied in an interleaved manner with a stride equal to
/// round(sqrt(N/nb_blocks)), i.e., it is applied independently for
/// each set of sample S_k = {stride*n + k}, n=0..N/stride-1."
///
/// The pre-rotation is applied to the same interleaved layout that
/// the main rotation uses (interleaved across `nb_blocks` time blocks
/// within the band buffer); the §4.3.4.3 "stride" defines a SECOND
/// interleave on top of that, grouping samples within a block at
/// positions `0, stride, 2*stride, ...` into one sub-sequence.
///
/// Returns `false` and leaves `samples` unchanged when the pre-
/// rotation does not apply (`nb_blocks <= 1`, per-block sample count
/// below 8, divisibility failure, or empty band). A `false` return
/// signals the caller that only the main rotation runs.
///
/// The §4.3.4.3 prose pins the sub-sequence count to `N/stride`,
/// where `N` is the per-block sample count (the rotation runs per
/// block, so `N` here is the in-block sample count). Sub-sequences
/// at positions `k = 0..stride - 1` whose length is below 2 are
/// no-ops (a single-element sequence has no pair to rotate).
pub fn apply_pre_rotation(samples: &mut [f32], nb_blocks: u32, theta: f64) -> bool {
    if nb_blocks <= 1 || samples.is_empty() {
        return false;
    }
    let nb = nb_blocks as usize;
    if samples.len() % nb != 0 {
        return false;
    }
    let per_block_u32 = (samples.len() / nb) as u32;
    if per_block_u32 < EXTRA_ROTATION_MIN_BLOCK_SAMPLES {
        return false;
    }
    let Some(stride_u32) = pre_rotation_stride(samples.len() as u32, nb_blocks) else {
        return false;
    };
    let stride = stride_u32 as usize;
    if stride == 0 {
        return false;
    }
    let per_block = per_block_u32 as usize;

    // Pre-rotation angle.
    let pre_theta = core::f64::consts::FRAC_PI_2 - theta;
    if pre_theta == 0.0 {
        // Identity rotation; nothing to do.
        return true;
    }
    let cos_t = pre_theta.cos() as f32;
    let sin_t = pre_theta.sin() as f32;

    // Apply, per block, the stride-interleaved N-D rotation to each
    // sub-sequence S_k = { stride * n + k } for k in 0..stride.
    let mut sub = Vec::with_capacity(per_block);
    for b in 0..nb {
        for k in 0..stride {
            // Build the sub-sequence S_k from block `b`.
            sub.clear();
            let mut n_idx = 0;
            loop {
                let in_block_pos = stride * n_idx + k;
                if in_block_pos >= per_block {
                    break;
                }
                sub.push(samples[b + nb * in_block_pos]);
                n_idx += 1;
            }
            if sub.len() < 2 {
                continue;
            }
            // Forward + reverse N-D rotation on the sub-sequence.
            let m = sub.len();
            for i in 0..m - 1 {
                let (xi, xj) = apply_2d_rotation(sub[i], sub[i + 1], cos_t, sin_t);
                sub[i] = xi;
                sub[i + 1] = xj;
            }
            for i in (0..m - 1).rev() {
                let (xi, xj) = apply_2d_rotation(sub[i], sub[i + 1], cos_t, sin_t);
                sub[i] = xi;
                sub[i + 1] = xj;
            }
            // Scatter back.
            for (n_idx, &v) in sub.iter().enumerate() {
                let in_block_pos = stride * n_idx + k;
                samples[b + nb * in_block_pos] = v;
            }
        }
    }
    true
}

/// Apply the §4.3.4.3 spreading rotation to a unit-norm PVQ shape
/// vector.
///
/// This is the full §4.3.4.3 orchestrator:
///
/// 1. Compute `theta = pi * g_r^2 / 4` from the supplied `spread`,
///    band dimension `n` (= `samples.len() / nb_blocks` per block, or
///    `samples.len()` for `nb_blocks = 1`), and pulse count `k`.
/// 2. If the band spans more than one time block AND each block has
///    at least 8 samples, apply the `(pi/2 - theta)` pre-rotation
///    interleaved at stride `round(sqrt(N/nb_blocks))`.
/// 3. Apply the main N-D rotation by `theta`, independently per time
///    block.
///
/// `samples` is modified in place. Layout: interleaved across
/// `nb_blocks` time blocks within the band buffer (see
/// [`apply_nd_rotation_multi_block`]). For `nb_blocks == 1` the
/// buffer is contiguous and the rotation runs straight across.
///
/// Returns `false` and leaves the buffer untouched when the shape
/// constraint is violated (`nb_blocks == 0`, divisibility failure).
/// `Spread::None` yields a true return with `samples` unchanged.
pub fn apply_spread(spread: Spread, samples: &mut [f32], k: u32, nb_blocks: u32) -> bool {
    if nb_blocks == 0 || samples.is_empty() {
        return false;
    }
    if samples.len() % (nb_blocks as usize) != 0 {
        return false;
    }
    if matches!(spread, Spread::None) {
        return true;
    }

    let n_total = samples.len() as u32;
    let theta = rotation_angle_f64(spread, n_total, k);
    if theta == 0.0 {
        return true;
    }

    // Pre-rotation: only when nb_blocks > 1 AND per-block >= 8.
    let _ = apply_pre_rotation(samples, nb_blocks, theta);

    // Main per-block rotation chain.
    if nb_blocks == 1 {
        apply_nd_rotation(samples, theta);
        true
    } else {
        apply_nd_rotation_multi_block(samples, nb_blocks, theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_norm(samples: &[f32]) -> f64 {
        samples
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn rotation_angle_zero_for_spread_none() {
        // Spread::None ⇒ g_r = 0 ⇒ theta = 0.
        assert_eq!(rotation_angle_f64(Spread::None, 16, 4), 0.0);
        assert_eq!(rotation_angle_f64(Spread::None, 0, 0), 0.0);
    }

    #[test]
    fn rotation_angle_matches_closed_form() {
        // N=16, K=4, Light (f_r=15): g_r = 16/76, g_r^2 = 256/5776,
        // theta = pi * 256/5776 / 4 = pi * 64/5776 = pi * 4/361.
        let theta = rotation_angle_f64(Spread::Light, 16, 4);
        let expected = core::f64::consts::PI * 4.0 / 361.0;
        assert!(
            (theta - expected).abs() < 1e-12,
            "theta={theta} expected={expected}"
        );
    }

    #[test]
    fn rotation_angle_zero_n_is_zero() {
        // Empty band defends with zero gain ⇒ zero theta.
        assert_eq!(rotation_angle_f64(Spread::Aggressive, 0, 4), 0.0);
    }

    #[test]
    fn apply_2d_rotation_orthonormal() {
        // R(theta) = [[c, s], [-s, c]] is orthonormal ⇒ ||R x|| = ||x||.
        let theta = 0.7f32;
        let c = theta.cos();
        let s = theta.sin();
        let (xi, xj) = apply_2d_rotation(1.0, 0.0, c, s);
        // (cos(theta), -sin(theta)).
        assert!((xi - c).abs() < 1e-6);
        assert!((xj - (-s)).abs() < 1e-6);
        let n2 = (xi as f64).powi(2) + (xj as f64).powi(2);
        assert!((n2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_2d_rotation_identity_theta_zero() {
        let (xi, xj) = apply_2d_rotation(0.7, -0.3, 1.0, 0.0);
        assert_eq!(xi, 0.7);
        assert_eq!(xj, -0.3);
    }

    #[test]
    fn apply_nd_rotation_zero_theta_is_noop() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0];
        let before = samples.clone();
        apply_nd_rotation(&mut samples, 0.0);
        assert_eq!(samples, before);
    }

    #[test]
    fn apply_nd_rotation_single_sample_is_noop() {
        let mut samples = vec![0.7];
        apply_nd_rotation(&mut samples, 0.5);
        assert_eq!(samples, vec![0.7]);
    }

    #[test]
    fn apply_nd_rotation_preserves_l2_norm() {
        // Each 2-D rotation is orthonormal; composing them preserves
        // L2 norm. Spot check on a few unit vectors and a few thetas.
        for &theta in &[0.1f64, 0.3, 0.7, 1.2] {
            for n in [2usize, 4, 8, 16, 31] {
                let mut samples: Vec<f32> = (0..n).map(|i| ((i + 1) as f32).sin()).collect();
                let norm0 = l2_norm(&samples);
                if norm0 > 0.0 {
                    for x in samples.iter_mut() {
                        *x /= norm0 as f32;
                    }
                }
                let norm_before = l2_norm(&samples);
                apply_nd_rotation(&mut samples, theta);
                let norm_after = l2_norm(&samples);
                let diff = (norm_after - norm_before).abs();
                assert!(diff < 1e-4, "n={n} theta={theta} norm change {diff}");
            }
        }
    }

    #[test]
    fn apply_nd_rotation_nontrivial_on_canonical_basis() {
        // theta != 0 must alter at least one entry on a non-degenerate
        // input. Use canonical basis e_0 in N=4.
        let mut samples = vec![1.0, 0.0, 0.0, 0.0];
        apply_nd_rotation(&mut samples, 0.5);
        // Some energy must leak away from index 0.
        let diff: f64 = samples
            .iter()
            .enumerate()
            .map(|(i, &v)| if i == 0 { 0.0 } else { (v as f64).powi(2) })
            .sum();
        assert!(diff > 1e-6, "rotation produced no spread: {samples:?}");
    }

    #[test]
    fn apply_nd_rotation_multi_block_zero_theta_is_noop() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let before = samples.clone();
        let ok = apply_nd_rotation_multi_block(&mut samples, 2, 0.0);
        assert!(ok);
        assert_eq!(samples, before);
    }

    #[test]
    fn apply_nd_rotation_multi_block_rejects_misalignment() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 5 / 2 != int
        let before = samples.clone();
        let ok = apply_nd_rotation_multi_block(&mut samples, 2, 0.5);
        assert!(!ok);
        assert_eq!(samples, before);
    }

    #[test]
    fn apply_nd_rotation_multi_block_zero_blocks_rejected() {
        let mut samples = vec![1.0, 2.0];
        let ok = apply_nd_rotation_multi_block(&mut samples, 0, 0.5);
        assert!(!ok);
    }

    #[test]
    fn apply_nd_rotation_multi_block_preserves_l2_norm() {
        for &theta in &[0.1f64, 0.5, 1.0] {
            for (n, nb) in [(8usize, 2u32), (16, 2), (32, 4), (64, 4)] {
                let mut samples: Vec<f32> = (0..n).map(|i| ((i + 3) as f32).cos() * 0.5).collect();
                let norm_before = l2_norm(&samples);
                let ok = apply_nd_rotation_multi_block(&mut samples, nb, theta);
                assert!(ok);
                let norm_after = l2_norm(&samples);
                let diff = (norm_after - norm_before).abs();
                assert!(diff < 1e-4, "n={n} nb={nb} theta={theta} diff={diff}");
            }
        }
    }

    #[test]
    fn apply_nd_rotation_multi_block_matches_single_block_when_nb_one() {
        // nb_blocks=1 must produce the same output as apply_nd_rotation.
        let mut a = vec![1.0f32, 0.5, -0.25, 0.7, -0.3, 0.1, 0.2, -0.6];
        let mut b = a.clone();
        let theta = 0.4f64;
        apply_nd_rotation(&mut a, theta);
        let ok = apply_nd_rotation_multi_block(&mut b, 1, theta);
        assert!(ok);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "mismatch a={x} b={y}");
        }
    }

    #[test]
    fn apply_pre_rotation_gated_off_for_single_block() {
        let mut samples = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let before = samples.clone();
        let ok = apply_pre_rotation(&mut samples, 1, 0.5);
        assert!(!ok);
        assert_eq!(samples, before);
    }

    #[test]
    fn apply_pre_rotation_gated_off_for_short_blocks() {
        // 14 samples / 2 blocks = 7 samples/block < 8 ⇒ pre-rotation off.
        let mut samples: Vec<f32> = (0..14).map(|i| i as f32 * 0.1).collect();
        let before = samples.clone();
        let ok = apply_pre_rotation(&mut samples, 2, 0.3);
        assert!(!ok);
        assert_eq!(samples, before);
    }

    #[test]
    fn apply_pre_rotation_preserves_l2_norm() {
        // 64 dims / 2 blocks = 32/block ⇒ stride = round(sqrt(32)) = 6.
        let mut samples: Vec<f32> = (0..64).map(|i| ((i + 1) as f32).sin() * 0.05).collect();
        let norm_before = l2_norm(&samples);
        let ok = apply_pre_rotation(&mut samples, 2, 0.5);
        assert!(ok);
        let norm_after = l2_norm(&samples);
        let diff = (norm_after - norm_before).abs();
        assert!(diff < 1e-3, "L2 changed by {diff}");
    }

    #[test]
    fn apply_pre_rotation_rejects_misalignment() {
        let mut samples = vec![0.1f32; 17]; // 17 / 2 != int
        let before = samples.clone();
        let ok = apply_pre_rotation(&mut samples, 2, 0.5);
        assert!(!ok);
        assert_eq!(samples, before);
    }

    #[test]
    fn apply_spread_none_is_noop() {
        let mut samples = vec![1.0f32, 2.0, 3.0, 4.0];
        let before = samples.clone();
        let ok = apply_spread(Spread::None, &mut samples, 4, 1);
        assert!(ok);
        assert_eq!(samples, before);
    }

    #[test]
    fn apply_spread_rejects_zero_blocks() {
        let mut samples = vec![1.0f32];
        let ok = apply_spread(Spread::Light, &mut samples, 1, 0);
        assert!(!ok);
    }

    #[test]
    fn apply_spread_rejects_misalignment() {
        let mut samples = vec![1.0f32; 5];
        let ok = apply_spread(Spread::Light, &mut samples, 2, 2);
        assert!(!ok);
    }

    #[test]
    fn apply_spread_preserves_l2_norm_single_block() {
        // Per-block too-small triggers no pre-rotation; main rotation
        // alone should preserve norm.
        for n in [4usize, 8, 16, 32] {
            for spread in [Spread::Light, Spread::Normal, Spread::Aggressive] {
                let mut samples: Vec<f32> = (0..n).map(|i| ((i + 1) as f32).sin()).collect();
                let norm0 = l2_norm(&samples);
                for x in samples.iter_mut() {
                    *x /= norm0 as f32;
                }
                let before = l2_norm(&samples);
                let ok = apply_spread(spread, &mut samples, 4, 1);
                assert!(ok);
                let after = l2_norm(&samples);
                let diff = (after - before).abs();
                assert!(diff < 1e-3, "n={n} spread={spread:?} diff={diff}");
            }
        }
    }

    #[test]
    fn apply_spread_preserves_l2_norm_multi_block() {
        // Multi-block with pre-rotation engaged (per-block >= 8).
        let mut samples: Vec<f32> = (0..32).map(|i| ((i + 1) as f32).cos() * 0.1).collect();
        let norm_before = l2_norm(&samples);
        let ok = apply_spread(Spread::Normal, &mut samples, 8, 2);
        assert!(ok);
        let norm_after = l2_norm(&samples);
        let diff = (norm_after - norm_before).abs();
        assert!(diff < 1e-3, "multi-block diff={diff}");
    }

    #[test]
    fn apply_spread_consumes_spread_at_band_boundaries() {
        // Sanity: at high K with strong spread, the rotation actually
        // mixes the input. Energy must move off index 0.
        let mut samples = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ok = apply_spread(Spread::Aggressive, &mut samples, 4, 1);
        assert!(ok);
        let diff_other_indices: f64 = samples
            .iter()
            .enumerate()
            .map(|(i, &v)| if i == 0 { 0.0 } else { (v as f64).powi(2) })
            .sum();
        assert!(diff_other_indices > 1e-6, "no spread: {samples:?}");
    }

    #[test]
    fn apply_spread_single_sample_band_is_noop_after_check() {
        // N=1 means there is no pair to rotate; the forward/reverse
        // chain skips and the buffer is unchanged.
        let mut samples = vec![0.7f32];
        let ok = apply_spread(Spread::Normal, &mut samples, 1, 1);
        assert!(ok);
        assert_eq!(samples, vec![0.7]);
    }

    #[test]
    fn rotation_chain_forward_then_reverse_for_n2() {
        // For N=2 there is one forward pair (R) and one reverse pair (R),
        // so the composite is R^2. Verify against the closed form
        // R(2*theta).
        let theta = 0.5f64;
        let mut samples = vec![0.8f32, 0.6];
        apply_nd_rotation(&mut samples, theta);
        // R^2 = R(2*theta).
        let two = 2.0 * theta;
        let c = two.cos() as f32;
        let s = two.sin() as f32;
        let expected = (c * 0.8 + s * 0.6, -s * 0.8 + c * 0.6);
        assert!(
            (samples[0] - expected.0).abs() < 1e-5,
            "got {} expected {}",
            samples[0],
            expected.0
        );
        assert!(
            (samples[1] - expected.1).abs() < 1e-5,
            "got {} expected {}",
            samples[1],
            expected.1
        );
    }
}
