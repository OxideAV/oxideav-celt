//! Walsh–Hadamard transform primitives for the §4.3.4.5 TF-resolution
//! resolution-change step (RFC 6716).
//!
//! ## What this module provides
//!
//! RFC 6716 §4.3.4.5 states:
//!
//! > A negative TF adjustment means that the temporal resolution is
//! > increased, while a positive TF adjustment means that the
//! > frequency resolution is increased. Changes in TF resolution are
//! > implemented using the Hadamard transform. To increase the time
//! > resolution by N, N "levels" of the Hadamard transform are
//! > applied to the decoded vector for each interleaved MDCT vector.
//! > To increase the frequency resolution (assumes a transient
//! > frame), then N levels of the Hadamard transform are applied
//! > _across_ the interleaved MDCT vector. In the case of increased
//! > time resolution, the decoder uses the "sequency order" because
//! > the input vector is sorted in time.
//!
//! This module supplies three pure helpers:
//!
//! * [`walsh_hadamard_inplace`] — natural-order (Hadamard-ordered)
//!   forward WHT, the radix-2 butterfly cascade.
//! * [`walsh_hadamard_sequency_inplace`] — sequency-ordered (Walsh-
//!   ordered) forward WHT, used for the time-resolution-increase
//!   branch per §4.3.4.5 final sentence.
//! * [`apply_tf_resolution_change`] — orchestrator: given an
//!   interleaved `nb_blocks × subvec` band layout and a signed
//!   `tf_adjustment` from [`crate::tf_change::tf_adjustment`], it
//!   applies `|tf_adjustment|` Hadamard levels in the appropriate
//!   direction. `tf_adjustment == 0` is a no-op.
//!
//! ## Normalization choice
//!
//! RFC 6716 §4.3.4.5 does NOT specify the per-level scaling. The two
//! natural choices are:
//!
//! 1. **Unscaled** butterfly: `(a,b) -> (a+b, a-b)`. After `k`
//!    levels the energy is multiplied by `2^k`.
//! 2. **Orthonormal** butterfly: `(a,b) -> ((a+b)/√2, (a-b)/√2)`.
//!    Energy is preserved at every level (the Hadamard matrix is
//!    orthonormal after the `1/√(2^k)` scaling).
//!
//! Because the §4.3.4.5 paragraph is silent on this point and because
//! the band shape vectors fed into the TF resolution change are
//! unit-norm (the §4.3.4 prose describes the PVQ output as a
//! normalized vector on the unit hypersphere), this module uses the
//! **orthonormal** variant — the only choice that keeps the band
//! norm invariant across the TF resolution change so that the
//! §4.3.6 denormalization step (which multiplies by `sqrt(energy)`)
//! sees the same normalized shape regardless of `tf_adjustment`. This
//! is documented as an empirical decoder-side decision; if a follow-
//! up trace against `opusdec` shows the unscaled variant is what the
//! reference does, swap [`HADAMARD_LEVEL_SCALE`] to `1.0` and update
//! this rustdoc.
//!
//! ## Clean-room provenance
//!
//! The Walsh–Hadamard transform is textbook material that predates
//! CELT by decades (Hadamard 1893, Walsh 1923). The radix-2 butterfly
//! cascade implemented here is the standard fast WHT, written from
//! first principles against the matrix definition
//! `H_2 = [[1, 1], [1, -1]]`, `H_{2n} = H_2 ⊗ H_n` (Kronecker
//! product); no external library source was consulted. The
//! sequency-order permutation is the textbook `gray(bit_reverse(i))`
//! mapping that converts Hadamard order into Walsh / sequency order
//! — a standard result from any DSP textbook covering the Walsh
//! transform.

/// Per-level normalization scale: `1/sqrt(2)`. See module docs for the
/// rationale; the band shape vectors are unit-norm so we choose the
/// orthonormal butterfly so the norm is preserved across TF resolution
/// changes.
pub const HADAMARD_LEVEL_SCALE: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// In-place natural-order (Hadamard-ordered) forward Walsh–Hadamard
/// transform.
///
/// `samples.len()` must equal `1 << levels`. Each invocation applies
/// the full radix-2 butterfly cascade across all `levels`; the result
/// is multiplied by `(1/sqrt(2))^levels` so that the transform is
/// orthonormal.
///
/// Returns silently (no-op) for `levels == 0` regardless of buffer
/// length. Panics in debug builds (and silently no-ops in release)
/// when `samples.len() != 1 << levels`.
#[inline]
pub fn walsh_hadamard_inplace(samples: &mut [f32], levels: u32) {
    if levels == 0 {
        return;
    }
    let n = 1usize << levels;
    debug_assert_eq!(
        samples.len(),
        n,
        "walsh_hadamard_inplace: buffer length {} does not match 1<<{} = {}",
        samples.len(),
        levels,
        n
    );
    if samples.len() != n {
        return;
    }
    butterfly_cascade(samples, levels);
}

/// In-place sequency-ordered (Walsh-ordered) forward Walsh–Hadamard
/// transform.
///
/// Same shape contract as [`walsh_hadamard_inplace`]. The natural-
/// order butterfly cascade is applied first; the result is then
/// reordered in place via the textbook `gray(bit_reverse(i))`
/// permutation that converts Hadamard ordering into sequency
/// (Walsh) ordering. Per RFC 6716 §4.3.4.5 the time-resolution-
/// increase branch uses sequency order because the input vector is
/// sorted in time.
#[inline]
pub fn walsh_hadamard_sequency_inplace(samples: &mut [f32], levels: u32) {
    walsh_hadamard_inplace(samples, levels);
    sequency_reorder_inplace(samples, levels);
}

/// Apply the §4.3.4.5 TF resolution change to a band shape, in place.
///
/// * `band` — the band's normalized shape coefficients, interpreted as
///   `nb_blocks` interleaved sub-vectors. `band.len()` must equal
///   `nb_blocks * (1 << levels)` for some `levels >= |tf_adjustment|`.
/// * `tf_adjustment` — signed adjustment from
///   [`crate::tf_change::tf_adjustment`]. Negative ⇒ increase time
///   resolution (sequency-ordered WHT applied to each interleaved
///   sub-vector). Positive ⇒ increase frequency resolution (natural-
///   order WHT applied across the interleaved blocks). Zero ⇒ no-op.
/// * `nb_blocks` — number of interleaved short-MDCT blocks in the
///   band (= 1 << LM for transient frames per §4.3.1).
///
/// Returns `true` on success. Returns `false` and leaves `band`
/// unchanged when the shape constraints are violated (band length not
/// `nb_blocks * 2^k`, or requested `|tf_adjustment|` exceeds the
/// available levels in either direction).
pub fn apply_tf_resolution_change(band: &mut [f32], tf_adjustment: i8, nb_blocks: usize) -> bool {
    if tf_adjustment == 0 {
        return true;
    }
    if nb_blocks == 0 || band.is_empty() {
        return false;
    }
    if band.len() % nb_blocks != 0 {
        return false;
    }
    let sub_len = band.len() / nb_blocks;

    let levels_abs = (tf_adjustment.unsigned_abs()) as u32;

    if tf_adjustment < 0 {
        // Increase time resolution: apply |tf_adjustment| sequency-
        // ordered WHT levels to EACH interleaved sub-vector. The sub-
        // vector length must therefore be at least 2^|tf_adjustment|
        // and a power of two (else there is no consistent radix-2
        // butterfly to apply).
        if !is_power_of_two(sub_len) {
            return false;
        }
        let sub_levels = sub_len.trailing_zeros();
        if levels_abs > sub_levels {
            return false;
        }
        // The transform is over the LOW `1<<levels_abs` samples of
        // each sub-vector, repeated stride-`1` (the sub-vector is
        // contiguous in the band layout — the inter-block interleave
        // is the "across blocks" direction, not the within-block
        // direction).
        let sub_xform_len = 1usize << levels_abs;
        for block in 0..nb_blocks {
            let start = block * sub_len;
            walsh_hadamard_sequency_inplace(&mut band[start..start + sub_xform_len], levels_abs);
        }
        true
    } else {
        // Increase frequency resolution: apply |tf_adjustment|
        // natural-order WHT levels ACROSS the interleaved blocks.
        // nb_blocks must be a power of two and >= 2^|tf_adjustment|.
        if !is_power_of_two(nb_blocks) {
            return false;
        }
        let block_levels = nb_blocks.trailing_zeros();
        if levels_abs > block_levels {
            return false;
        }
        let xform_blocks = 1usize << levels_abs;
        // Walk every position WITHIN a sub-vector and apply the WHT
        // across the first `xform_blocks` blocks at that position.
        let mut scratch = vec![0.0f32; xform_blocks];
        for pos in 0..sub_len {
            for b in 0..xform_blocks {
                scratch[b] = band[b * sub_len + pos];
            }
            walsh_hadamard_inplace(&mut scratch, levels_abs);
            for b in 0..xform_blocks {
                band[b * sub_len + pos] = scratch[b];
            }
        }
        true
    }
}

// --- Private helpers -------------------------------------------------

#[inline]
fn is_power_of_two(n: usize) -> bool {
    n > 0 && n & (n - 1) == 0
}

/// Standard radix-2 in-place WHT butterfly cascade. After `levels`
/// stages the data is in natural (Hadamard) order; each butterfly is
/// scaled by `1/sqrt(2)` so the overall transform is orthonormal.
fn butterfly_cascade(samples: &mut [f32], levels: u32) {
    let n = 1usize << levels;
    let mut stride: usize = 1;
    while stride < n {
        let block = stride * 2;
        let mut base = 0usize;
        while base < n {
            for i in 0..stride {
                let a = samples[base + i];
                let b = samples[base + i + stride];
                samples[base + i] = (a + b) * HADAMARD_LEVEL_SCALE;
                samples[base + i + stride] = (a - b) * HADAMARD_LEVEL_SCALE;
            }
            base += block;
        }
        stride <<= 1;
    }
}

/// Convert natural (Hadamard) order into sequency (Walsh) order via
/// the `bit_reverse(gray(i))` permutation, in place.
///
/// The natural-Hadamard matrix rows are not arranged by sign-change
/// count. To put the sequency-`s` Walsh coefficient at output index
/// `s`, we read it from natural-Hadamard bin
/// `bit_reverse(gray(s))` — the standard textbook mapping between
/// the two orderings.
fn sequency_reorder_inplace(samples: &mut [f32], levels: u32) {
    if levels == 0 {
        return;
    }
    let n = 1usize << levels;
    debug_assert_eq!(samples.len(), n);
    let mut reordered = vec![0.0f32; n];
    for (i, slot) in reordered.iter_mut().enumerate() {
        let src = gray_then_bit_reverse(i, levels);
        *slot = samples[src];
    }
    samples.copy_from_slice(&reordered);
}

/// Compute `bit_reverse(gray_code(i), levels)` — the index in the
/// natural-order Hadamard output that contains the `i`-th sequency-
/// ordered Walsh coefficient.
#[inline]
fn gray_then_bit_reverse(i: usize, levels: u32) -> usize {
    let g = i ^ (i >> 1);
    let mut r = 0usize;
    let mut x = g;
    for _ in 0..levels {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed reference: natural-order WHT of [1, 0, 0, 0]
    /// (a single non-zero sample at index 0) is the all-ones row,
    /// scaled by the orthonormal factor `(1/sqrt(2))^levels = 1/2`
    /// for n=4. Every output bin is `0.5`.
    #[test]
    fn natural_order_impulse_at_zero_yields_constant_row() {
        let mut buf = [1.0f32, 0.0, 0.0, 0.0];
        walsh_hadamard_inplace(&mut buf, 2);
        for v in &buf {
            assert!((*v - 0.5).abs() < 1e-6, "expected 0.5, got {v}");
        }
    }

    /// Hand-computed reference: natural-order WHT of
    /// [1, 1, 1, 1] is `[2, 0, 0, 0]` (DC) with the orthonormal
    /// scaling.
    #[test]
    fn natural_order_dc_yields_impulse_at_zero() {
        let mut buf = [1.0f32, 1.0, 1.0, 1.0];
        walsh_hadamard_inplace(&mut buf, 2);
        let expected = [2.0f32, 0.0, 0.0, 0.0];
        for (got, want) in buf.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    /// Apply the WHT twice; with orthonormal scaling the result is
    /// the original vector (the orthonormal Hadamard matrix is its
    /// own inverse: `H_n · H_n = I`).
    #[test]
    fn involutivity_round_trip_n8() {
        let original = [0.5f32, -0.25, 0.75, -1.0, 0.125, 0.0, -0.5, 0.25];
        let mut buf = original;
        walsh_hadamard_inplace(&mut buf, 3);
        walsh_hadamard_inplace(&mut buf, 3);
        for (got, want) in buf.iter().zip(original.iter()) {
            assert!(
                (got - want).abs() < 1e-5,
                "round-trip mismatch: got {got}, want {want}"
            );
        }
    }

    /// The orthonormal WHT preserves the L2 norm.
    #[test]
    fn norm_preserved_n16() {
        let original: [f32; 16] = [
            0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 0.05, -0.15, 0.25, -0.35, 0.45,
            -0.55,
        ];
        let mut buf = original;
        walsh_hadamard_inplace(&mut buf, 4);
        let in_energy: f32 = original.iter().map(|x| x * x).sum();
        let out_energy: f32 = buf.iter().map(|x| x * x).sum();
        assert!(
            (in_energy - out_energy).abs() < 1e-5,
            "energy not preserved: in={in_energy}, out={out_energy}"
        );
    }

    /// Sequency-ordered WHT of [1, 1, 1, 1] (a constant signal) has
    /// all its energy in the zero-sequency (DC) bin.
    #[test]
    fn sequency_order_dc_to_zero_sequency() {
        let mut buf = [1.0f32, 1.0, 1.0, 1.0];
        walsh_hadamard_sequency_inplace(&mut buf, 2);
        assert!(
            (buf[0] - 2.0).abs() < 1e-6,
            "DC should map to sequency-0 bin = 2.0, got {}",
            buf[0]
        );
        for v in &buf[1..] {
            assert!(v.abs() < 1e-6, "non-DC sequency bin nonzero: {v}");
        }
    }

    /// Sequency-ordered WHT of [1, -1, 1, -1] (highest possible
    /// sign-alternation frequency on N=4) puts all its energy in the
    /// HIGHEST sequency bin (index N-1 = 3).
    #[test]
    fn sequency_order_alternating_to_highest_sequency() {
        let mut buf = [1.0f32, -1.0, 1.0, -1.0];
        walsh_hadamard_sequency_inplace(&mut buf, 2);
        for (i, v) in buf.iter().enumerate() {
            if i == 3 {
                assert!(
                    (v - 2.0).abs() < 1e-6,
                    "expected 2.0 at sequency bin 3, got {v}"
                );
            } else {
                assert!(v.abs() < 1e-6, "expected 0.0 at sequency bin {i}, got {v}");
            }
        }
    }

    /// `tf_adjustment == 0` is a no-op.
    #[test]
    fn apply_tf_zero_is_noop() {
        let original = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut buf = original;
        assert!(apply_tf_resolution_change(&mut buf, 0, 2));
        assert_eq!(buf, original);
    }

    /// Negative `tf_adjustment` applies a sequency-ordered WHT to
    /// each interleaved sub-vector. Build an 8-sample 2-block band
    /// where both sub-vectors are `[1, 1, 1, 1]`; after a -2 TF
    /// adjustment each sub-vector becomes `[2, 0, 0, 0]` (sequency-0
    /// DC) per `sequency_order_dc_to_zero_sequency`.
    #[test]
    fn apply_tf_negative_applies_per_block_sequency_wht() {
        let mut band = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert!(apply_tf_resolution_change(&mut band, -2, 2));
        let expected = [2.0f32, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        for (got, want) in band.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    /// Positive `tf_adjustment` applies a natural-order WHT across
    /// blocks. Build a 4-sample 4-block band where each block is a
    /// single sample, and the sequence across blocks is
    /// `[1, 1, 1, 1]`; after a +2 TF adjustment the across-block
    /// sample sequence becomes `[2, 0, 0, 0]` per
    /// `natural_order_dc_yields_impulse_at_zero`.
    #[test]
    fn apply_tf_positive_applies_across_block_natural_wht() {
        // band layout: 4 blocks × 1 sample per block = 4 samples total
        let mut band = [1.0f32, 1.0, 1.0, 1.0];
        assert!(apply_tf_resolution_change(&mut band, 2, 4));
        let expected = [2.0f32, 0.0, 0.0, 0.0];
        for (got, want) in band.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    /// `apply_tf_resolution_change` rejects shape mismatches.
    #[test]
    fn apply_tf_rejects_bad_shape() {
        let mut band = [0.0f32; 5]; // not divisible by nb_blocks=2
        assert!(!apply_tf_resolution_change(&mut band, -1, 2));

        let mut band = [0.0f32; 6]; // 2 blocks × 3 samples — 3 not a power of two
        assert!(!apply_tf_resolution_change(&mut band, -1, 2));

        let mut band = [0.0f32; 8]; // 2 blocks × 4 samples; -3 needs 8 levels per sub-vec
        assert!(!apply_tf_resolution_change(&mut band, -3, 2));

        let mut band = [0.0f32; 8]; // 3 blocks ≠ power-of-two
        assert!(!apply_tf_resolution_change(&mut band, 1, 3));
    }

    /// `gray_then_bit_reverse` against a hand-computed N=4 table:
    /// sequency index `s` → natural-Hadamard source bin
    /// `bit_reverse(gray(s), 2)`.
    ///   s=0  gray=0   rev2=00=0
    ///   s=1  gray=1   rev2=10=2
    ///   s=2  gray=3   rev2=11=3
    ///   s=3  gray=2   rev2=01=1
    #[test]
    fn gray_then_bit_reverse_n4_table() {
        assert_eq!(gray_then_bit_reverse(0, 2), 0);
        assert_eq!(gray_then_bit_reverse(1, 2), 2);
        assert_eq!(gray_then_bit_reverse(2, 2), 3);
        assert_eq!(gray_then_bit_reverse(3, 2), 1);
    }
}
