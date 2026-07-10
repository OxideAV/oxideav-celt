//! Band denormalization (RFC 6716 §4.3.6).
//!
//! After PVQ shape decoding ([`crate::pvq`]) produces a unit-L2-norm
//! shape vector for each band and the coarse + fine + finalize chain
//! ([`crate::coarse_energy`], [`crate::fine_energy`]) reconstructs the
//! per-band log-2 energy, the last step before the inverse MDCT is to
//! denormalize the bands. Per §4.3.6:
//!
//! > Each decoded normalized band is multiplied by the square root of
//! > the decoded energy.
//!
//! Working in the §4.3.2.1 Q8 base-2 log-energy domain, the linear
//! energy of band `b` is
//!
//! ```text
//! E_lin(b) = 2^(E_q8(b) / 256.0)
//! ```
//!
//! and the per-sample amplitude factor (the square root of the linear
//! energy) is
//!
//! ```text
//! A(b) = sqrt(E_lin(b)) = 2^(E_q8(b) / 512.0)
//! ```
//!
//! Each band's unit-norm shape vector is multiplied element-wise by
//! `A(b)` to yield the denormalized MDCT-domain samples the inverse
//! MDCT consumes.
//!
//! ## Scope
//!
//! This module is decoder-side only. It operates on caller-supplied
//! shape vectors + Q8 log-energy values, composing with the
//! [`crate::coarse_energy`] decoder (whose normative-form f32
//! log-energies land on this Q8 axis after a multiply by 256 and a
//! round). No range-decoder interaction; no allocation-budget
//! interaction.
//!
//! ## Provenance
//!
//! The §4.3.6 multiplicative step is stated in the RFC narrative as
//! plain prose (one sentence; no normative source-file delegation).
//! The Q8 fixed-point log-2 energy axis is a 256-per-log2-step
//! rendering of the §4.3.2.1 base-2 log domain (RFC 6716 §4.3.2.1
//! lines 6031–6037) that [`crate::coarse_energy`] reconstructs. The
//! `sqrt(2^E) = 2^(E/2)` identity is elementary arithmetic.

use crate::coarse_energy::NUM_BANDS;

/// Q8 fractional-bit count on the per-band log-2 energy
/// representation: `E_lin = 2^(E_q8 / Q8_DENOM)`.
pub const Q8_DENOM: f32 = 256.0;

/// Q8 sqrt-amplitude divisor: `A = sqrt(E_lin) = 2^(E_q8 / SQRT_Q8_DENOM)`.
/// Equals `2 * Q8_DENOM` because `sqrt(2^E) = 2^(E/2)` and Q8 carries
/// `256` fractional units per integer log-2 step.
pub const SQRT_Q8_DENOM: f32 = 512.0;

/// RFC 8251 sec 8 cap on the log-domain band energy, on the Q8 axis:
/// `32.0` base-2 log steps (`32 * 256 = 8192` Q8). Applied before the
/// linear conversion so extreme bitstreams cannot drive the amplitude
/// past single-precision range and poison the PCM with NaNs.
pub const MAX_LOG_ENERGY_Q8: i32 = 32 * 256;

/// Convert a Q8 base-2 log-energy value into the equivalent linear
/// amplitude (= sqrt of the linear energy). `A = 2^(E_q8 / 512)`.
///
/// `E_q8 = 0` returns `1.0` (the unit-energy band — denormalization is
/// a no-op for a band whose energy is exactly 1.0).
///
/// `E_q8 > 0` amplifies (linear energy > 1.0); `E_q8 < 0` attenuates
/// (linear energy < 1.0). The §4.3.2.1 prediction filter clamps the
/// reconstructed log-energy so the exponent stays in finite f32 range.
#[inline]
pub fn log_energy_q8_to_amplitude_f32(log_energy_q8: i32) -> f32 {
    // RFC 8251 (Opus update) sec 8 "Cap on Band Energy": on extreme
    // bitstreams the log-domain band energy can exceed what a
    // single-precision float represents once converted to a linear
    // scale, later producing NaNs; the update caps the log-domain
    // value at 32.0 (base-2 log steps) before the exp2 conversion.
    // Q8: 32.0 * 256 = 8192.
    let capped = log_energy_q8.min(MAX_LOG_ENERGY_Q8);
    // f32::exp2 takes a fractional exponent directly. Dividing in f32
    // up-front (rather than computing `log_energy_q8 as f32 / 512.0`
    // inside the call) keeps the inline expansion small.
    f32::exp2(capped as f32 / SQRT_Q8_DENOM)
}

/// Multiply each sample in `shape` by `amplitude` and write the
/// result into `out`. `shape` and `out` must have equal length;
/// returns `false` and leaves `out` untouched on a length mismatch.
///
/// This is the per-band §4.3.6 step factored out for callers that have
/// the amplitude precomputed (e.g. when the same band is denormalized
/// repeatedly across multiple MDCT blocks at the same energy).
pub fn scale_band_f32(shape: &[f32], amplitude: f32, out: &mut [f32]) -> bool {
    if shape.len() != out.len() {
        return false;
    }
    for (s, o) in shape.iter().zip(out.iter_mut()) {
        *o = *s * amplitude;
    }
    true
}

/// In-place variant of [`scale_band_f32`].
#[inline]
pub fn scale_band_in_place_f32(samples: &mut [f32], amplitude: f32) {
    for s in samples.iter_mut() {
        *s *= amplitude;
    }
}

/// Denormalize a single band: multiply the unit-norm `shape` vector by
/// `sqrt(2^(log_energy_q8 / 256))` and write the result into `out`.
///
/// `shape` and `out` must have equal length; returns `false` and
/// leaves `out` untouched on a length mismatch.
///
/// The function is total: any `log_energy_q8` that does not produce a
/// finite `f32` amplitude is the caller's problem to clamp (the
/// §4.3.2.1 prediction filter is the natural place to clamp upstream).
pub fn denormalize_band_f32(shape: &[f32], log_energy_q8: i32, out: &mut [f32]) -> bool {
    let amplitude = log_energy_q8_to_amplitude_f32(log_energy_q8);
    scale_band_f32(shape, amplitude, out)
}

/// In-place variant of [`denormalize_band_f32`].
#[inline]
pub fn denormalize_band_in_place_f32(samples: &mut [f32], log_energy_q8: i32) {
    let amplitude = log_energy_q8_to_amplitude_f32(log_energy_q8);
    scale_band_in_place_f32(samples, amplitude);
}

/// Walk a 21-band frame, denormalizing each band's shape vector
/// into a contiguous output buffer.
///
/// * `shapes` is a slice of 21 sub-slices (one per band; the
///   per-band MDCT-bin count comes from [`crate::BAND_BINS_LM`]).
///   Each sub-slice must be unit-L2-norm (the §4.3.4.2 PVQ
///   normalization step guarantees this; cf.
///   [`crate::normalize_to_unit_l2`]).
/// * `log_energies_q8` is the per-band reconstructed log-2 energy in
///   Q8 (i.e. the output of the §4.3.2.1 coarse + §4.3.2.2 fine +
///   §4.3.2.2 finalize chain).
/// * `out` receives the concatenated denormalized samples, in band
///   order (band 0 first). Its length must equal the sum of the
///   per-band shape lengths.
///
/// Returns `false` and leaves `out` untouched on any length mismatch
/// (shapes count vs `NUM_BANDS`; `out` length vs the band-shape total).
pub fn denormalize_bands_f32(
    shapes: &[&[f32]],
    log_energies_q8: &[i32; NUM_BANDS],
    out: &mut [f32],
) -> bool {
    if shapes.len() != NUM_BANDS {
        return false;
    }
    let total: usize = shapes.iter().map(|s| s.len()).sum();
    if out.len() != total {
        return false;
    }
    let mut cursor = 0;
    for (band, shape) in shapes.iter().enumerate() {
        let n = shape.len();
        let amplitude = log_energy_q8_to_amplitude_f32(log_energies_q8[band]);
        for (s, o) in shape.iter().zip(out[cursor..cursor + n].iter_mut()) {
            *o = *s * amplitude;
        }
        cursor += n;
    }
    true
}

/// In-place variant of [`denormalize_bands_f32`].
///
/// `samples` is a single contiguous buffer holding all 21 band-shape
/// vectors concatenated in band order; `bins_per_band[b]` gives the
/// length of band `b`'s shape sub-vector. Each sub-vector is scaled
/// by `sqrt(2^(log_energies_q8[b] / 256))`.
///
/// Returns `false` and leaves `samples` untouched on a length mismatch
/// (the sum of `bins_per_band` must equal `samples.len()`).
pub fn denormalize_bands_in_place_f32(
    samples: &mut [f32],
    bins_per_band: &[u32; NUM_BANDS],
    log_energies_q8: &[i32; NUM_BANDS],
) -> bool {
    let total: usize = bins_per_band.iter().map(|&n| n as usize).sum();
    if samples.len() != total {
        return false;
    }
    let mut cursor = 0;
    for band in 0..NUM_BANDS {
        let n = bins_per_band[band] as usize;
        let amplitude = log_energy_q8_to_amplitude_f32(log_energies_q8[band]);
        for s in samples[cursor..cursor + n].iter_mut() {
            *s *= amplitude;
        }
        cursor += n;
    }
    true
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    /// RFC 8251 sec 8: the log-domain energy is capped at 32.0 before
    /// the linear conversion, so an extreme envelope stays finite (and
    /// a capped value equals the amplitude at exactly 32.0).
    #[test]
    fn rfc8251_energy_cap_keeps_amplitude_finite() {
        use super::{log_energy_q8_to_amplitude_f32, MAX_LOG_ENERGY_Q8};
        let extreme = log_energy_q8_to_amplitude_f32(i32::MAX);
        assert!(extreme.is_finite());
        assert_eq!(extreme, log_energy_q8_to_amplitude_f32(MAX_LOG_ENERGY_Q8));
        // Below the cap the conversion is untouched.
        let below = log_energy_q8_to_amplitude_f32(MAX_LOG_ENERGY_Q8 - 256);
        assert!(below < extreme);
        // The amplitude is 2^(E_q8/512), so one 256-Q8 step below the
        // cap is a factor sqrt(2) in amplitude.
        assert!((below * core::f32::consts::SQRT_2 - extreme).abs() <= extreme * 1e-6);
    }

    use super::*;
    use crate::BAND_BINS_LM;

    /// `Q8_DENOM = 256` (the §4.3.2.1 Q8 fractional-bit count) and
    /// `SQRT_Q8_DENOM = 512` (= `2 * Q8_DENOM`). These pin the units
    /// the Q8 log-energy representation carries.
    #[test]
    fn q8_denom_constants() {
        assert_eq!(Q8_DENOM, 256.0);
        assert_eq!(SQRT_Q8_DENOM, 512.0);
        assert_eq!(SQRT_Q8_DENOM, 2.0 * Q8_DENOM);
    }

    /// `E_q8 = 0` ⇒ amplitude 1.0 (unit-energy band passes through).
    #[test]
    fn zero_log_energy_yields_unit_amplitude() {
        assert_eq!(log_energy_q8_to_amplitude_f32(0), 1.0);
    }

    /// Each integer log-2 step (= Q8 value 256) doubles the linear
    /// energy, so the amplitude (sqrt of linear) scales by sqrt(2).
    #[test]
    fn integer_log2_step_scales_by_sqrt_2() {
        let a0 = log_energy_q8_to_amplitude_f32(0);
        let a1 = log_energy_q8_to_amplitude_f32(256);
        let ratio = a1 / a0;
        let expected = (2.0_f32).sqrt();
        assert!(
            (ratio - expected).abs() < 1e-6,
            "ratio={ratio} expected {expected}"
        );
    }

    /// Two integer log-2 steps (Q8 value 512) double the amplitude
    /// (sqrt(2^2) = 2).
    #[test]
    fn two_log2_steps_double_amplitude() {
        let a = log_energy_q8_to_amplitude_f32(512);
        assert!((a - 2.0).abs() < 1e-6, "a={a}");
    }

    /// `E_q8 = -512` ⇒ amplitude 0.5 (the symmetric counterpart of the
    /// above).
    #[test]
    fn negative_log2_steps_halve_amplitude() {
        let a = log_energy_q8_to_amplitude_f32(-512);
        assert!((a - 0.5).abs() < 1e-6, "a={a}");
    }

    /// The amplitude function is multiplicatively additive on the
    /// log-energy axis: `A(E1 + E2) = A(E1) * A(E2)`.
    #[test]
    fn amplitude_is_multiplicative_on_log() {
        for e1 in [-512, -256, -64, 0, 64, 256, 512].iter().copied() {
            for e2 in [-256, -32, 0, 32, 256].iter().copied() {
                let a_sum = log_energy_q8_to_amplitude_f32(e1 + e2);
                let a_prod =
                    log_energy_q8_to_amplitude_f32(e1) * log_energy_q8_to_amplitude_f32(e2);
                assert!(
                    (a_sum - a_prod).abs() < 1e-5 * a_sum.abs().max(1.0),
                    "E1={e1} E2={e2}: A(sum)={a_sum} A1*A2={a_prod}"
                );
            }
        }
    }

    /// `scale_band_f32` multiplies each sample by the amplitude.
    #[test]
    fn scale_band_f32_scales_each_sample() {
        let shape = [1.0_f32, 0.5, -0.25, 0.0, 0.75];
        let mut out = [0.0_f32; 5];
        let ok = scale_band_f32(&shape, 2.0, &mut out);
        assert!(ok);
        assert_eq!(out, [2.0, 1.0, -0.5, 0.0, 1.5]);
    }

    /// `scale_band_f32` rejects length mismatch and leaves the output
    /// buffer untouched.
    #[test]
    fn scale_band_f32_rejects_length_mismatch() {
        let shape = [1.0_f32; 4];
        let mut out = [9.0_f32; 5];
        let ok = scale_band_f32(&shape, 2.0, &mut out);
        assert!(!ok);
        assert_eq!(out, [9.0, 9.0, 9.0, 9.0, 9.0]);
    }

    /// `scale_band_in_place_f32` is equivalent to `scale_band_f32` on
    /// the same input.
    #[test]
    fn scale_band_in_place_matches_scale_band() {
        let shape = [0.5_f32, -0.5, 0.25, -0.25, 1.0, -1.0, 0.0, 0.125];
        let mut a = [0.0_f32; 8];
        let mut b = shape;
        let ok = scale_band_f32(&shape, 1.5, &mut a);
        assert!(ok);
        scale_band_in_place_f32(&mut b, 1.5);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(*x, *y, "sample {i}: a={x} b={y}");
        }
    }

    /// Denormalizing a unit-norm shape with `E_q8 = 512` (= linear
    /// energy 4.0; amplitude 2.0) scales every sample by 2.0 and the
    /// output's squared L2 norm equals the linear energy.
    #[test]
    fn denormalize_band_preserves_energy() {
        // Unit-norm shape: 1/sqrt(4) on each of 4 samples.
        let s = 0.5_f32;
        let shape = [s, s, s, s];
        let mut out = [0.0_f32; 4];
        let ok = denormalize_band_f32(&shape, 512, &mut out);
        assert!(ok);
        // Amplitude factor = 2.0; each sample doubles.
        for o in out {
            assert!((o - 1.0).abs() < 1e-6, "o={o}");
        }
        // The denormalized squared L2 norm equals the linear energy =
        // 2^(512/256) = 4.0.
        let energy: f32 = out.iter().map(|x| x * x).sum();
        assert!((energy - 4.0).abs() < 1e-5, "energy={energy}");
    }

    /// Zero log-energy with a unit-norm shape leaves the shape
    /// unchanged.
    #[test]
    fn denormalize_band_zero_energy_is_identity() {
        let shape = [0.6_f32, -0.8, 0.0, 0.0];
        let mut out = [0.0_f32; 4];
        let ok = denormalize_band_f32(&shape, 0, &mut out);
        assert!(ok);
        for (a, b) in shape.iter().zip(out.iter()) {
            assert_eq!(*a, *b);
        }
    }

    /// `denormalize_band_in_place_f32` matches the copying variant on
    /// the same input.
    #[test]
    fn denormalize_band_in_place_matches_copying() {
        let shape = [0.3_f32, -0.4, 0.5, -0.6, 0.7, 0.0];
        let mut a = [0.0_f32; 6];
        let mut b = shape;
        denormalize_band_f32(&shape, 256, &mut a);
        denormalize_band_in_place_f32(&mut b, 256);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-7, "i={i}: a={x} b={y}");
        }
    }

    /// `denormalize_band_f32` rejects length mismatch and leaves the
    /// output buffer untouched.
    #[test]
    fn denormalize_band_rejects_length_mismatch() {
        let shape = [1.0_f32; 3];
        let mut out = [7.0_f32; 4];
        let ok = denormalize_band_f32(&shape, 0, &mut out);
        assert!(!ok);
        assert_eq!(out, [7.0, 7.0, 7.0, 7.0]);
    }

    /// Energy-preservation invariant per band: the squared L2 norm of
    /// the denormalized shape vector equals the linear energy
    /// `2^(E_q8 / 256)`. Sweep across a range of energies on the
    /// canonical 4-sample 0.5 unit-norm shape.
    #[test]
    fn energy_preservation_invariant_sweep() {
        let s = 0.5_f32;
        let shape = [s, s, s, s]; // unit L2 norm
        for e_q8 in [-1024, -512, -256, -64, 0, 64, 256, 512, 1024]
            .iter()
            .copied()
        {
            let mut out = [0.0_f32; 4];
            denormalize_band_f32(&shape, e_q8, &mut out);
            let observed: f32 = out.iter().map(|x| x * x).sum();
            let expected = f32::exp2(e_q8 as f32 / Q8_DENOM);
            let rel_err = (observed - expected).abs() / expected.max(1.0);
            assert!(
                rel_err < 1e-5,
                "E_q8={e_q8}: observed={observed} expected={expected}"
            );
        }
    }

    /// `denormalize_bands_f32` walks 21 bands and writes them into
    /// `out` in band order. The output's per-band squared-L2-sum
    /// equals the per-band linear energy.
    #[test]
    fn denormalize_bands_f32_walks_full_envelope() {
        // Build a 21-band 1-bin-per-band frame for ease of inspection.
        // Each "band" has a single sample of value 1.0 (trivially
        // unit-norm).
        let band_shape = [1.0_f32];
        let shapes_owned: Vec<&[f32]> = (0..NUM_BANDS).map(|_| &band_shape[..]).collect();

        // Per-band log-energy = band-index Q8 step (band 0 → 0, band
        // 1 → 256, band 2 → 512, ... — amplitudes 1.0, sqrt(2), 2.0, ...).
        let mut log_energies = [0_i32; NUM_BANDS];
        for (band, e) in log_energies.iter_mut().enumerate() {
            *e = (band as i32) * 256;
        }

        let mut out = vec![0.0_f32; NUM_BANDS];
        let ok = denormalize_bands_f32(&shapes_owned, &log_energies, &mut out);
        assert!(ok);

        for (band, x) in out.iter().enumerate() {
            let expected = f32::exp2((band as f32) * 0.5);
            let rel_err = (x - expected).abs() / expected.max(1.0);
            assert!(
                rel_err < 1e-5,
                "band {band}: observed={x} expected={expected}"
            );
        }
    }

    /// `denormalize_bands_f32` rejects wrong shapes count.
    #[test]
    fn denormalize_bands_f32_rejects_wrong_shape_count() {
        let shape = [1.0_f32];
        let too_few: Vec<&[f32]> = (0..NUM_BANDS - 1).map(|_| &shape[..]).collect();
        let log_energies = [0_i32; NUM_BANDS];
        let mut out = vec![0.0_f32; NUM_BANDS - 1];
        let ok = denormalize_bands_f32(&too_few, &log_energies, &mut out);
        assert!(!ok);
    }

    /// `denormalize_bands_f32` rejects `out` length mismatch.
    #[test]
    fn denormalize_bands_f32_rejects_out_length_mismatch() {
        let shape = [1.0_f32, 2.0];
        let shapes: Vec<&[f32]> = (0..NUM_BANDS).map(|_| &shape[..]).collect();
        let log_energies = [0_i32; NUM_BANDS];
        let mut out_short = vec![0.0_f32; NUM_BANDS]; // expects 42; got 21
        let ok = denormalize_bands_f32(&shapes, &log_energies, &mut out_short);
        assert!(!ok);
    }

    /// `denormalize_bands_in_place_f32` on a full LM=0 layout (sum of
    /// the per-band bin counts = `100 << 0 = 100`) succeeds and
    /// preserves the per-band energy invariant.
    #[test]
    fn denormalize_bands_in_place_lm0_layout() {
        // Set up a canonical LM=0 frame.
        let lm = 0;
        let bins_lm0 = BAND_BINS_LM[lm];
        let total_bins: usize = bins_lm0.iter().map(|&n| n as usize).sum();
        assert_eq!(total_bins, 100);

        // Fill the frame with unit-norm bands: each band b has
        // `bins_lm0[b]` samples, each of value `1 / sqrt(N)`.
        let mut samples = Vec::with_capacity(total_bins);
        let mut per_band_starts = Vec::with_capacity(NUM_BANDS);
        for &n in bins_lm0.iter() {
            let n_us = n as usize;
            per_band_starts.push(samples.len());
            let s = 1.0_f32 / (n_us as f32).sqrt();
            for _ in 0..n_us {
                samples.push(s);
            }
        }

        // Per-band log-energy = +256 Q8 (linear energy = 2.0).
        let log_energies = [256_i32; NUM_BANDS];

        let ok = denormalize_bands_in_place_f32(&mut samples, &bins_lm0, &log_energies);
        assert!(ok);

        // Verify per-band squared-L2-sum = 2.0.
        let mut cursor = 0;
        for (band, &n) in bins_lm0.iter().enumerate() {
            let n_us = n as usize;
            let band_slice = &samples[cursor..cursor + n_us];
            let observed: f32 = band_slice.iter().map(|x| x * x).sum();
            let rel_err = (observed - 2.0).abs() / 2.0;
            assert!(
                rel_err < 1e-5,
                "band {band}: observed={observed} expected=2.0 (rel_err={rel_err})"
            );
            cursor += n_us;
        }
    }

    /// `denormalize_bands_in_place_f32` rejects buffer-length mismatch
    /// and leaves the input untouched.
    #[test]
    fn denormalize_bands_in_place_rejects_length_mismatch() {
        let bins = BAND_BINS_LM[0];
        let log_energies = [0_i32; NUM_BANDS];
        let mut too_short = vec![3.5_f32; 99]; // expected 100
        let ok = denormalize_bands_in_place_f32(&mut too_short, &bins, &log_energies);
        assert!(!ok);
        // Buffer unchanged.
        for s in too_short {
            assert_eq!(s, 3.5);
        }
    }

    /// Per-band order is preserved: a non-monotone energy vector
    /// produces non-monotone per-band amplitudes in the output.
    #[test]
    fn band_ordering_preserved() {
        // 21 single-sample bands, each at unit shape.
        let shape = [1.0_f32];
        let shapes: Vec<&[f32]> = (0..NUM_BANDS).map(|_| &shape[..]).collect();
        // Energies alternate +/-256.
        let mut log_energies = [0_i32; NUM_BANDS];
        for (band, e) in log_energies.iter_mut().enumerate() {
            *e = if band % 2 == 0 { 256 } else { -256 };
        }
        let mut out = vec![0.0_f32; NUM_BANDS];
        let ok = denormalize_bands_f32(&shapes, &log_energies, &mut out);
        assert!(ok);

        let amp_pos = f32::exp2(256.0 / SQRT_Q8_DENOM);
        let amp_neg = f32::exp2(-256.0 / SQRT_Q8_DENOM);
        for (band, x) in out.iter().enumerate() {
            let expected = if band % 2 == 0 { amp_pos } else { amp_neg };
            assert!(
                (x - expected).abs() < 1e-6,
                "band {band}: observed={x} expected={expected}"
            );
        }
    }

    /// Empty bands (zero-bin shape) contribute nothing to the output
    /// and do not advance the output cursor. Useful for the §4.3.4.4
    /// band-split case where a leaf sub-band has been split out.
    #[test]
    fn empty_bands_pass_through() {
        let empty = [];
        let nonempty = [1.0_f32, 0.0];
        let mut shapes: Vec<&[f32]> = Vec::with_capacity(NUM_BANDS);
        for band in 0..NUM_BANDS {
            if band == 0 || band == NUM_BANDS - 1 {
                shapes.push(&nonempty[..]);
            } else {
                shapes.push(&empty[..]);
            }
        }
        let log_energies = [256_i32; NUM_BANDS];
        let mut out = vec![0.0_f32; 4]; // two bands × two samples each
        let ok = denormalize_bands_f32(&shapes, &log_energies, &mut out);
        assert!(ok);
        let amp = f32::exp2(256.0 / SQRT_Q8_DENOM);
        // Band 0 contributes [amp, 0]; band 20 contributes [amp, 0].
        assert!((out[0] - amp).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
        assert!((out[2] - amp).abs() < 1e-6);
        assert_eq!(out[3], 0.0);
    }

    /// A frame whose per-band log-energies are all zero is a no-op:
    /// the output equals the concatenation of the input shapes.
    #[test]
    fn all_zero_energies_is_identity() {
        let bins = BAND_BINS_LM[2]; // LM=2 = 10 ms
        let total_bins: usize = bins.iter().map(|&n| n as usize).sum();
        // Per-band shapes filled with the band index as the sample
        // value (NOT unit-norm; that's fine — we're testing the
        // identity, not energy preservation).
        let mut band_buffers: Vec<Vec<f32>> = Vec::with_capacity(NUM_BANDS);
        for (b, &n) in bins.iter().enumerate() {
            band_buffers.push(vec![(b as f32) * 0.1; n as usize]);
        }
        let shapes: Vec<&[f32]> = band_buffers.iter().map(|v| v.as_slice()).collect();
        let log_energies = [0_i32; NUM_BANDS];
        let mut out = vec![0.0_f32; total_bins];
        let ok = denormalize_bands_f32(&shapes, &log_energies, &mut out);
        assert!(ok);

        // Output equals concatenation of input shapes.
        let mut cursor = 0;
        for shape in &shapes {
            for (i, &s) in shape.iter().enumerate() {
                assert_eq!(out[cursor + i], s);
            }
            cursor += shape.len();
        }
    }
}
