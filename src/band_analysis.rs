//! Band energy analysis (RFC 6716 §4.3.6, encode direction).
//!
//! This module is the encoder-side inverse of the §4.3.6
//! denormalization ([`crate::denormalization`]). Where the decoder
//! multiplies a unit-L2-norm shape vector by the square root of the
//! decoded per-band energy to recover the MDCT-domain samples, the
//! encoder runs the reverse: it splits an MDCT-domain band into
//!
//! 1. its per-band log-2 **energy** (the §4.3.2.1 coarse-energy target
//!    that [`crate::coarse_energy::encode_coarse_energy`] consumes), and
//! 2. its unit-L2-norm **shape** (the §4.3.4.2 PVQ-search input that
//!    [`crate::pvq::encode_shape`] consumes).
//!
//! ## The identity
//!
//! §4.3.6 sets `band[i] = shape[i] * A` with the amplitude
//! `A = sqrt(2^E) = 2^(E/2)` and `shape` on the unit hypersphere
//! (`sum(shape[i]^2) = 1`). The band's linear energy is therefore
//!
//! ```text
//! sum(band[i]^2) = A^2 * sum(shape[i]^2) = A^2 = 2^E
//! ```
//!
//! so the base-2 log-energy the encoder must code is
//!
//! ```text
//! E = log2(sum(band[i]^2))
//! ```
//!
//! on the same axis [`crate::coarse_energy::CoarseEnergyState`] carries
//! (`1.0` = one integer log-2 step = 6 dB), and the shape is recovered
//! by dividing out the L2 norm:
//!
//! ```text
//! shape[i] = band[i] / sqrt(sum(band[i]^2)) = band[i] / 2^(E/2)
//! ```
//!
//! Both are elementary consequences of the one-sentence §4.3.6 rule;
//! this module performs no range-coder or allocation interaction.
//!
//! ## Silent bands
//!
//! A band whose samples are all zero has no defined log-energy
//! (`log2(0) = -inf`). CELT codes such a band with the frame-level
//! `silence` flag or a very negative coarse energy; rather than emit a
//! non-finite value, [`band_log_energy_f32`] floors the log-energy at
//! [`SILENCE_LOG_ENERGY`] — well below the §4.3.2.1 prediction clamp of
//! `-9.0` so the reconstruction treats the band as silent. The floor is
//! an analysis-side decision (§4.3.6 is silent on the degenerate case);
//! it never affects a band that carries any energy.
//!
//! ## Provenance
//!
//! The §4.3.6 multiplicative step is RFC 6716 §4.3.6 (one sentence); the
//! log-2 energy axis is the §4.3.2.1 base-2 log domain
//! [`crate::coarse_energy`] uses. The `E = log2(sum of squares)` and the
//! unit-norm division are elementary arithmetic — the exact inverse of
//! [`crate::denormalization::denormalize_band_f32`]. No external library
//! source consulted.

use crate::coarse_energy::NUM_BANDS;

/// Floor applied to a silent band's base-2 log-energy so
/// [`band_log_energy_f32`] never returns a non-finite value.
///
/// Chosen well below the §4.3.2.1 prediction clamp (`-9.0`) so a band
/// that hits the floor is unambiguously treated as silent by the
/// coarse-energy reconstruction. An analysis-side decision (§4.3.6 does
/// not define the degenerate all-zero case).
pub const SILENCE_LOG_ENERGY: f32 = -28.0;

/// The linear-energy floor corresponding to [`SILENCE_LOG_ENERGY`]
/// (`2^SILENCE_LOG_ENERGY`). A band whose sum of squares is at or below
/// this floors its log-energy.
#[inline]
fn energy_floor() -> f32 {
    f32::exp2(SILENCE_LOG_ENERGY)
}

/// The linear energy of a band: the sum of the squared samples
/// (`sum(band[i]^2)`), accumulated in `f64` to keep precision for long
/// bands before the final `f32` narrowing.
///
/// This equals `2^E` for a §4.3.6-denormalized band with base-2
/// log-energy `E`.
pub fn band_energy_f32(band: &[f32]) -> f32 {
    let sum: f64 = band.iter().map(|&x| (x as f64) * (x as f64)).sum();
    sum as f32
}

/// The base-2 log-energy of a band on the §4.3.2.1 coarse-energy axis
/// (`1.0` = one integer log-2 step = 6 dB): `E = log2(sum(band[i]^2))`.
///
/// This is the exact quantity
/// [`crate::coarse_energy::encode_coarse_energy`] takes as its per-band
/// target. A silent (all-zero, or near-zero) band floors at
/// [`SILENCE_LOG_ENERGY`] rather than returning `-inf`.
pub fn band_log_energy_f32(band: &[f32]) -> f32 {
    let energy = band_energy_f32(band).max(energy_floor());
    energy.log2()
}

/// The result of analyzing one MDCT-domain band: its coarse-energy
/// target and its unit-L2-norm shape.
#[derive(Debug, Clone, PartialEq)]
pub struct BandAnalysis {
    /// Base-2 log-energy on the §4.3.2.1 axis (`1.0` = 6 dB) — the
    /// [`crate::coarse_energy::encode_coarse_energy`] target.
    pub log_energy: f32,
    /// Unit-L2-norm shape vector — the
    /// [`crate::pvq::encode_shape`] / [`crate::pvq::pvq_search`] input.
    /// A silent band yields an all-zero shape.
    pub shape: Vec<f32>,
}

/// Analyze one MDCT-domain band into its coarse-energy target and its
/// unit-norm shape — the exact inverse of
/// [`crate::denormalization::denormalize_band_f32`].
///
/// The shape is `band / sqrt(sum(band[i]^2))`, so
/// `sum(shape[i]^2) == 1` (up to f32 rounding) for any band carrying
/// energy; a silent band (sum of squares at or below the
/// [`SILENCE_LOG_ENERGY`] floor) yields an all-zero shape and the
/// floored `log_energy`, since its direction is undefined.
pub fn analyze_band_f32(band: &[f32]) -> BandAnalysis {
    let energy = band_energy_f32(band);
    if energy <= energy_floor() {
        return BandAnalysis {
            log_energy: SILENCE_LOG_ENERGY,
            shape: vec![0.0; band.len()],
        };
    }
    // Divide by the true L2 norm (in f64) so the shape is unit-norm.
    let norm = (energy as f64).sqrt();
    let shape = band.iter().map(|&x| (x as f64 / norm) as f32).collect();
    BandAnalysis {
        log_energy: energy.log2(),
        shape,
    }
}

/// Analyze a full 21-band frame laid out as one contiguous MDCT
/// spectrum, returning the per-band coarse-energy targets and unit-norm
/// shapes.
///
/// * `spectrum` is the concatenated per-band samples in band order
///   (band 0 first), the layout
///   [`crate::denormalization::denormalize_bands_f32`] produces.
/// * `bins_per_band` gives each band's sample count; its length must be
///   [`NUM_BANDS`] and its sum must equal `spectrum.len()`.
///
/// Returns `None` on a length mismatch. On success the returned vector
/// has one [`BandAnalysis`] per band, in band order. The per-band
/// `log_energy` values compose directly into the `[[f32; NUM_BANDS]; _]`
/// target [`crate::coarse_energy::encode_coarse_energy`] consumes (for a
/// single channel).
pub fn analyze_bands_f32(spectrum: &[f32], bins_per_band: &[u32]) -> Option<Vec<BandAnalysis>> {
    if bins_per_band.len() != NUM_BANDS {
        return None;
    }
    let total: u64 = bins_per_band.iter().map(|&n| n as u64).sum();
    if total != spectrum.len() as u64 {
        return None;
    }
    let mut out = Vec::with_capacity(NUM_BANDS);
    let mut offset = 0usize;
    for &n in bins_per_band {
        let n = n as usize;
        out.push(analyze_band_f32(&spectrum[offset..offset + n]));
        offset += n;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::denormalization::{denormalize_band_f32, log_energy_q8_to_amplitude_f32};

    /// `analyze_band_f32` inverts `denormalize_band_f32`: denormalizing a
    /// unit-norm shape at a chosen Q8 energy then analyzing the result
    /// recovers the same log-energy (to Q8 resolution) and the same
    /// shape.
    #[test]
    fn analyze_inverts_denormalize() {
        // A unit-L2-norm shape.
        let raw = [0.5f32, -0.3, 0.2, -0.6, 0.1, 0.4, -0.25, 0.15];
        let norm: f32 = raw.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let shape: Vec<f32> = raw.iter().map(|&x| x / norm).collect();

        for &e_q8 in &[0i32, 256, 512, -256, 1024, -1024, 1536] {
            let mut band = vec![0.0f32; shape.len()];
            assert!(denormalize_band_f32(&shape, e_q8, &mut band));
            let analysis = analyze_band_f32(&band);
            // The recovered log-energy in Q8 must match the input Q8.
            let recovered_q8 = (analysis.log_energy * 256.0).round() as i32;
            assert!(
                (recovered_q8 - e_q8).abs() <= 1,
                "energy mismatch e_q8={e_q8} recovered={recovered_q8}"
            );
            // The recovered shape must match the input shape.
            for (a, b) in analysis.shape.iter().zip(shape.iter()) {
                assert!((a - b).abs() < 1e-5, "shape mismatch {a} vs {b}");
            }
        }
    }

    /// The analyzed shape is unit-L2-norm for any band carrying energy.
    #[test]
    fn analyzed_shape_is_unit_norm() {
        let band = [3.0f32, -4.0, 12.0, -1.0, 5.0];
        let analysis = analyze_band_f32(&band);
        let norm_sq: f32 = analysis.shape.iter().map(|&x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5, "shape norm^2 = {norm_sq}");
        // Energy = sum of squares = 9+16+144+1+25 = 195; log2(195).
        assert!((analysis.log_energy - 195.0f32.log2()).abs() < 1e-4);
    }

    /// A silent band floors the log-energy and yields an all-zero shape
    /// (no NaN from a zero-norm division).
    #[test]
    fn silent_band_floors_and_zeroes_shape() {
        let band = [0.0f32; 6];
        let analysis = analyze_band_f32(&band);
        assert_eq!(analysis.log_energy, SILENCE_LOG_ENERGY);
        assert_eq!(analysis.shape, vec![0.0; 6]);
        assert!(analysis.shape.iter().all(|x| x.is_finite()));
        // band_log_energy_f32 agrees.
        assert_eq!(band_log_energy_f32(&band), SILENCE_LOG_ENERGY);
    }

    /// `band_energy_f32` is the plain sum of squares and equals `2^E`
    /// for a denormalized band.
    #[test]
    fn band_energy_matches_amplitude_squared() {
        let shape = [0.6f32, 0.8]; // unit norm
        let e_q8 = 512; // E = 2.0 → linear energy 4.0
        let amp = log_energy_q8_to_amplitude_f32(e_q8);
        let band = [shape[0] * amp, shape[1] * amp];
        let energy = band_energy_f32(&band);
        assert!((energy - 4.0).abs() < 1e-4, "energy = {energy}");
    }

    /// `analyze_bands_f32` walks a contiguous spectrum and rejects a
    /// length mismatch.
    #[test]
    fn analyze_bands_walks_spectrum() {
        // Three bands of sizes 2, 3, 1.
        let mut bins = [0u32; NUM_BANDS];
        bins[0] = 2;
        bins[1] = 3;
        bins[2] = 1;
        // Remaining bands are zero-width.
        let spectrum = [1.0f32, 0.0, 0.0, 2.0, 0.0, -5.0];
        let bands = analyze_bands_f32(&spectrum, &bins).unwrap();
        assert_eq!(bands.len(), NUM_BANDS);
        // Band 0: [1, 0] → energy 1 → log2 = 0.
        assert!((bands[0].log_energy - 0.0).abs() < 1e-5);
        // Band 1: [0, 2, 0] → energy 4 → log2 = 2.
        assert!((bands[1].log_energy - 2.0).abs() < 1e-5);
        // Band 2: [-5] → energy 25 → log2(25).
        assert!((bands[2].log_energy - 25.0f32.log2()).abs() < 1e-4);
        // Zero-width bands are silent.
        assert_eq!(bands[3].log_energy, SILENCE_LOG_ENERGY);

        // Length mismatch rejected.
        assert!(analyze_bands_f32(&spectrum, &bins[..NUM_BANDS - 1]).is_none());
        let mut wrong = bins;
        wrong[0] = 3;
        assert!(analyze_bands_f32(&spectrum, &wrong).is_none());
    }
}
