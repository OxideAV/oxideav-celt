//! Final per-band log-energy assembly (RFC 6716 §4.3.2).
//!
//! ## Where this fits
//!
//! CELT reconstructs the per-band log-2 energy envelope in three
//! additive steps (RFC 6716 §4.3.2, `docs/audio/opus/rfc6716-opus.txt`
//! lines 6031–6099):
//!
//! 1. **Coarse** (§4.3.2.1) — 6 dB integer steps in base-2 log,
//!    range-coded with a Laplace prior and the 2-D inter/intra
//!    prediction filter. [`crate::coarse_energy::decode_coarse_energy`]
//!    lands the result in [`crate::coarse_energy::CoarseEnergyState`]
//!    as per-channel f32 log-2 energies (`1.0` = one integer log-2
//!    step = 6 dB).
//! 2. **Fine** (§4.3.2.2) — a per-band raw-bit refinement, `B_i` bits
//!    per band fixed by the §4.3.3 allocator, mapping to the additive
//!    correction `(f + 1/2)/2^B_i - 1/2` in the same base-2 log axis.
//!    [`crate::fine_energy::decode_fine_energy`] returns it in **Q14**.
//! 3. **Finalize** (§4.3.2.2) — leftover bits spent at the very end of
//!    the frame, ≤ 1 per band per channel, as an additional Q14
//!    correction. [`crate::fine_energy::finalize_extra_bits`] returns
//!    it in Q14, already summed across channels.
//!
//! Each step's correction is **added** to the running per-band log-2
//! energy (the §4.3.2.2 prose: "the correction applied to the coarse
//! energy"). This module performs that addition and converts the
//! result onto the **Q8** axis the §4.3.6 denormalization
//! ([`crate::denormalization`]) and the single-band orchestrator
//! ([`crate::band_decode::decode_band_shape`]) consume — the `multiply
//! by 256 and round` bridge those modules document as the caller's
//! responsibility.
//!
//! ## The three axes and why a bridge is needed
//!
//! The three producers each carry the base-2 log-energy at a different
//! fixed-point scale, all measuring the same physical quantity (`1.0`
//! integer step = 6 dB):
//!
//! | Producer                | Scale | One integer log-2 step |
//! | ----------------------- | ----- | ---------------------- |
//! | `CoarseEnergyState`     | f32   | `1.0`                  |
//! | `decode_fine_energy`    | Q14   | `16384`                |
//! | `finalize_extra_bits`   | Q14   | `16384`                |
//! | denormalization input   | Q8    | `256`                  |
//!
//! The fine/finalize Q14 corrections are therefore scaled by
//! `1 / 16384` to reach the f32 axis, or the f32 coarse energy is
//! scaled by `256` to reach Q8 and the Q14 corrections by
//! `256 / 16384 = 1 / 64` (a right-shift by 6). Both paths are exposed:
//! [`assemble_band_log_energy_f32`] returns the final envelope as f32,
//! and [`assemble_band_log_energy_q8`] returns it as the rounded Q8
//! integers the denormalization step indexes directly.
//!
//! ## Clean-room provenance
//!
//! The additive three-step structure and the §4.3.2.2 correction map
//! are transcribed from RFC 6716 §4.3.2 (`docs/audio/opus/
//! rfc6716-opus.txt`). The Q8 axis is the 256-per-log2-step rendering
//! [`crate::denormalization`] already establishes from §4.3.2.1. The
//! Q14↔Q8↔f32 rescalings are elementary arithmetic. No external
//! library source was consulted; this module introduces no new numeric
//! constant beyond the scale denominators its sibling modules already
//! export.

use crate::coarse_energy::{CoarseEnergyState, NUM_BANDS};
use crate::denormalization::Q8_DENOM;

/// Q14 fractional-bit count on the §4.3.2.2 fine/finalize correction
/// axis: a correction value `c` represents `c / 16384` integer log-2
/// steps. Matches the Q14 scale [`crate::fine_energy`] documents.
pub const FINE_Q14_DENOM: i32 = 1 << 14;

/// Right-shift count to rescale a Q14 fine/finalize correction onto the
/// Q8 axis (`16384 / 256 = 64 = 2^6`). Used by
/// [`assemble_band_log_energy_q8`] with round-to-nearest.
pub const Q14_TO_Q8_SHIFT: u32 = 6;

/// Rounding constant for the Q14→Q8 rescale (`1 << (Q14_TO_Q8_SHIFT - 1)`).
/// Added before the arithmetic right-shift to round to nearest rather
/// than toward negative infinity (corrections are signed).
const Q14_TO_Q8_ROUND: i32 = 1 << (Q14_TO_Q8_SHIFT - 1);

/// Combine the §4.3.2.1 coarse log-energy with the §4.3.2.2 fine and
/// finalize Q14 corrections into the final per-band base-2 log-energy
/// envelope, returned as f32 (`1.0` = one integer log-2 step = 6 dB).
///
/// Parameters:
/// * `coarse` — the per-channel coarse log-energies the §4.3.2.1
///   decoder reconstructed.
/// * `channel` — which channel's envelope to assemble (`0` mono, `0`/`1`
///   stereo). Out-of-range channels return `None`.
/// * `fine_q14` — the §4.3.2.2 per-band fine corrections
///   ([`crate::fine_energy::decode_fine_energy`]). `None` skips the
///   fine step (bands stay at the coarse value).
/// * `finalize_q14` — the §4.3.2.2 per-band finalize corrections
///   ([`crate::fine_energy::finalize_extra_bits`]'s
///   `extra_correction_q14`). `None` skips the finalize step.
///
/// The §4.3.2.2 finalize corrections are documented as already summed
/// across channels; in a mono decode that sum is the single channel's
/// contribution, and in a stereo decode the caller applies the same
/// summed correction to whichever channel envelope it is assembling
/// (the finalize step refines each channel's own coarse value).
///
/// Returns the 21-band f32 envelope, or `None` for an out-of-range
/// `channel`.
pub fn assemble_band_log_energy_f32(
    coarse: &CoarseEnergyState,
    channel: usize,
    fine_q14: Option<&[i32; NUM_BANDS]>,
    finalize_q14: Option<&[i32; NUM_BANDS]>,
) -> Option<[f32; NUM_BANDS]> {
    if channel >= coarse.energy.len() {
        return None;
    }
    let mut out = coarse.energy[channel];
    for (b, e) in out.iter_mut().enumerate() {
        let mut corr_q14 = 0i32;
        if let Some(fine) = fine_q14 {
            corr_q14 += fine[b];
        }
        if let Some(fin) = finalize_q14 {
            corr_q14 += fin[b];
        }
        if corr_q14 != 0 {
            *e += corr_q14 as f32 / FINE_Q14_DENOM as f32;
        }
    }
    Some(out)
}

/// Combine the §4.3.2.1 coarse log-energy with the §4.3.2.2 fine and
/// finalize Q14 corrections into the final per-band base-2 log-energy
/// envelope on the **Q8** axis (`256` per integer log-2 step), the
/// representation [`crate::denormalization::denormalize_bands_in_place_f32`]
/// and [`crate::band_decode::decode_band_shape`] index directly.
///
/// The coarse f32 value is scaled by `Q8_DENOM = 256` and rounded; the
/// Q14 corrections are rescaled by a round-to-nearest right-shift of
/// [`Q14_TO_Q8_SHIFT`] (= `÷64`). Performing the corrections' rescale
/// in integer Q-space (rather than rounding the f32 sum once) keeps the
/// raw-bit-exact fine/finalize values from accumulating an extra f32
/// rounding error before they reach the Q8 grid.
///
/// Parameters match [`assemble_band_log_energy_f32`]. Returns the
/// 21-band Q8 envelope, or `None` for an out-of-range `channel`.
pub fn assemble_band_log_energy_q8(
    coarse: &CoarseEnergyState,
    channel: usize,
    fine_q14: Option<&[i32; NUM_BANDS]>,
    finalize_q14: Option<&[i32; NUM_BANDS]>,
) -> Option<[i32; NUM_BANDS]> {
    if channel >= coarse.energy.len() {
        return None;
    }
    let coarse_ch = &coarse.energy[channel];
    let mut out = [0i32; NUM_BANDS];
    for (b, slot) in out.iter_mut().enumerate() {
        // Coarse: f32 log-2 step → Q8 (×256), round to nearest.
        let coarse_q8 = (coarse_ch[b] * Q8_DENOM).round() as i32;
        // Fine + finalize: Q14 → Q8 via a round-to-nearest ÷64.
        let mut corr_q14 = 0i32;
        if let Some(fine) = fine_q14 {
            corr_q14 += fine[b];
        }
        if let Some(fin) = finalize_q14 {
            corr_q14 += fin[b];
        }
        let corr_q8 = (corr_q14 + Q14_TO_Q8_ROUND) >> Q14_TO_Q8_SHIFT;
        *slot = coarse_q8 + corr_q8;
    }
    Some(out)
}

/// Convert a single f32 base-2 log-energy value onto the Q8 axis
/// (`E_q8 = round(E * 256)`), the per-band bridge the §4.3.6
/// denormalization documents as the caller's responsibility.
///
/// Exposed standalone so a caller that has already assembled an f32
/// envelope (e.g. via [`assemble_band_log_energy_f32`]) can feed
/// individual bands to [`crate::band_decode::decode_band_shape`]
/// without re-running the whole assembly.
#[inline]
pub fn log_energy_f32_to_q8(log_energy_f32: f32) -> i32 {
    (log_energy_f32 * Q8_DENOM).round() as i32
}

// ---------------------------------------------------------------------
// Interop energy convention (the wire's absolute energy scale).
// ---------------------------------------------------------------------

/// Per-band mean energy `eMeans`, quantized in Q4 (units of the coarse
/// base-2 log step). The coarse/fine energy on the wire is coded
/// **relative to this mean**: the encoder subtracts `eMeans[band]`
/// from the absolute band log-energy, and the decoder adds it back
/// before denormalization.
///
/// Values are the staged normative numeric table
/// `docs/audio/opus/tables/e-means.csv` (21 coded-band entries of the
/// 25-entry table; the 4 trailing entries pad unused bands). Numeric
/// data extraction per the staging area's Feist doctrine.
pub const E_MEANS_Q4: [i16; NUM_BANDS] = [
    103, 100, 92, 85, 81, 77, 72, 70, 78, 75, 73, 71, 78, 74, 69, 72, 70, 74, 76, 71, 60,
];

/// Per-`LM` spectral-scale bridge between this crate's MDCT-domain
/// band amplitudes and the wire's absolute convention, in Q8 base-2
/// log units.
///
/// RFC 6716 pins the inverse-MDCT scaling ("scaling by 1/2", §4.3.7)
/// but leaves the wire's absolute energy scale to the bit-exactness
/// requirement. The constant was **calibrated black-box**: per-band
/// wire energies decoded from reference-encoder streams (produced by
/// `opusenc` run as an opaque process over a known noise signal) were
/// regressed against this crate's analyzer for the same signal — the
/// offset `wire + eMeans - log2(band amplitude)` is flat across all
/// 21 bands and across the 5/10/20 ms frame sizes at `14.0 ± 0.2`
/// base-2 log units (Q8 `3584`), i.e. band- and LM-independent.
pub const SPECTRUM_SCALE_LOG2_Q8: [i32; 4] = [3584, 3584, 3584, 3584];

/// The Q8 bias between a band's **absolute** log2 amplitude in this
/// crate's spectrum scale and the **wire** energy value:
/// `wire_q8 = 512 * al2_ours_q8ish ... ` — concretely,
/// `wire = al2_ours + scale(lm) - eMeans[band]` in base-2 log
/// amplitude units, and this helper returns
/// `Q8(scale(lm) - eMeans[band])`.
#[inline]
pub fn interop_wire_bias_q8(band: usize, lm: u32) -> i32 {
    SPECTRUM_SCALE_LOG2_Q8[lm.min(3) as usize] - 16 * i32::from(E_MEANS_Q4[band.min(NUM_BANDS - 1)])
}

/// f32 variant of [`interop_wire_bias_q8`] for the encoder's coarse
/// targets (which live on the f32 log-2 axis, `1.0` = one wire step).
#[inline]
pub fn interop_wire_bias_f32(band: usize, lm: u32) -> f32 {
    interop_wire_bias_q8(band, lm) as f32 / Q8_DENOM
}

/// Convert a **wire-domain** Q8 envelope value (coarse + fine +
/// finalize as coded) into the **rendering** Q8 log-energy this
/// crate's §4.3.6 denormalization (`amplitude = 2^(q8/512)`) consumes:
/// remove the wire bias and double (one wire step = 6 dB = one full
/// base-2 log-amplitude step, while the rendering axis carries
/// base-2 log **energy**).
#[inline]
pub fn render_band_energy_q8(wire_env_q8: i32, band: usize, lm: u32) -> i32 {
    2 * (wire_env_q8 - interop_wire_bias_q8(band, lm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::denormalization::log_energy_q8_to_amplitude_f32;
    use crate::fine_energy::fine_correction_q14;

    fn state_with(channel: usize, values: [f32; NUM_BANDS]) -> CoarseEnergyState {
        let mut s = CoarseEnergyState::new();
        s.energy[channel] = values;
        s
    }

    #[test]
    fn no_corrections_returns_coarse_f32() {
        let mut coarse = [0.0f32; NUM_BANDS];
        coarse[0] = 1.0;
        coarse[5] = -2.5;
        coarse[20] = 3.25;
        let s = state_with(0, coarse);
        let out = assemble_band_log_energy_f32(&s, 0, None, None).unwrap();
        assert_eq!(out, coarse);
    }

    #[test]
    fn no_corrections_returns_coarse_q8_rounded() {
        let mut coarse = [0.0f32; NUM_BANDS];
        coarse[0] = 1.0; // → 256
        coarse[1] = -2.0; // → -512
        coarse[2] = 0.5; // → 128
        let s = state_with(0, coarse);
        let out = assemble_band_log_energy_q8(&s, 0, None, None).unwrap();
        assert_eq!(out[0], 256);
        assert_eq!(out[1], -512);
        assert_eq!(out[2], 128);
    }

    #[test]
    fn out_of_range_channel_is_none() {
        let s = CoarseEnergyState::new();
        assert!(assemble_band_log_energy_f32(&s, 2, None, None).is_none());
        assert!(assemble_band_log_energy_q8(&s, 2, None, None).is_none());
    }

    /// A fine correction adds onto the coarse value on the f32 axis.
    /// `fine_correction_q14(f, B)` at `B = 1` gives `±8192` Q14 =
    /// `±0.5` log-2 steps.
    #[test]
    fn fine_correction_adds_on_f32_axis() {
        let mut coarse = [0.0f32; NUM_BANDS];
        coarse[3] = 2.0;
        let s = state_with(0, coarse);
        let mut fine = [0i32; NUM_BANDS];
        // f = 1, B = 1 ⇒ (2*1+1)*2^(13-1) - 2^13 = 3*4096 - 8192 = 4096
        // = +0.25 log-2 steps (the upper half-bin midpoint of [0,1)).
        fine[3] = fine_correction_q14(1, 1);
        assert_eq!(fine[3], 4096);
        let out = assemble_band_log_energy_f32(&s, 0, Some(&fine), None).unwrap();
        assert!((out[3] - 2.25).abs() < 1e-6);
    }

    /// The same correction lands on the Q8 grid: `4096 Q14 / 64 = 64 Q8`
    /// = `0.25` log-2 steps, added to the `512 Q8` coarse value.
    #[test]
    fn fine_correction_adds_on_q8_axis() {
        let mut coarse = [0.0f32; NUM_BANDS];
        coarse[3] = 2.0; // → 512 Q8
        let s = state_with(0, coarse);
        let mut fine = [0i32; NUM_BANDS];
        fine[3] = fine_correction_q14(1, 1); // 4096 Q14
        let out = assemble_band_log_energy_q8(&s, 0, Some(&fine), None).unwrap();
        assert_eq!(out[3], 512 + 64);
    }

    /// Fine + finalize corrections both add.
    #[test]
    fn fine_and_finalize_both_add() {
        let s = state_with(0, [0.0f32; NUM_BANDS]);
        let mut fine = [0i32; NUM_BANDS];
        let mut fin = [0i32; NUM_BANDS];
        fine[7] = 4096; // +64 Q8
        fin[7] = -2048; // -32 Q8
        let out = assemble_band_log_energy_q8(&s, 0, Some(&fine), Some(&fin)).unwrap();
        assert_eq!(out[7], 64 - 32);
    }

    /// The Q14→Q8 rescale rounds to nearest (not toward -inf) for a
    /// negative correction. `-2080 Q14`: `(-2080 + 32) >> 6 = -2048 >> 6
    /// = -32`. A naive truncating `/64` would also give `-32` here, so
    /// pick a value that exposes the rounding: `-2079 Q14` →
    /// `(-2079 + 32) >> 6 = -2047 >> 6 = -32` (round-to-nearest), whereas
    /// `-2079 / 64` truncates toward zero to `-32` as well — use a
    /// half-step boundary instead.
    #[test]
    fn q14_to_q8_rounds_to_nearest() {
        let s = state_with(0, [0.0f32; NUM_BANDS]);
        let mut fine = [0i32; NUM_BANDS];
        // 96 Q14: exactly 1.5 Q8 → rounds to 2 (round-half-up via +32).
        fine[0] = 96;
        // -96 Q14: -1.5 Q8 → (-96 + 32) >> 6 = -64 >> 6 = -1.
        fine[1] = -96;
        // 32 Q14: 0.5 Q8 → (32 + 32) >> 6 = 64 >> 6 = 1.
        fine[2] = 32;
        let out = assemble_band_log_energy_q8(&s, 0, Some(&fine), None).unwrap();
        assert_eq!(out[0], 2);
        assert_eq!(out[1], -1);
        assert_eq!(out[2], 1);
    }

    /// The assembled Q8 envelope feeds the §4.3.6 amplitude directly:
    /// a band at `+1.0` log-2 step (512 Q8 after a +0.0 correction)
    /// denormalizes to amplitude `sqrt(2)`.
    #[test]
    fn assembled_q8_drives_denormalization_amplitude() {
        let mut coarse = [0.0f32; NUM_BANDS];
        coarse[4] = 1.0; // 256 Q8 ⇒ amplitude 2^(256/512) = sqrt(2)
        let s = state_with(0, coarse);
        let out = assemble_band_log_energy_q8(&s, 0, None, None).unwrap();
        let amp = log_energy_q8_to_amplitude_f32(out[4]);
        assert!((amp - std::f32::consts::SQRT_2).abs() < 1e-5);
    }

    /// Per-channel assembly: each channel's coarse envelope is combined
    /// independently; the stereo path assembles channel 1 from
    /// `energy[1]`.
    #[test]
    fn stereo_channel_one_uses_its_own_coarse() {
        let mut s = CoarseEnergyState::new();
        s.energy[0][2] = 1.0;
        s.energy[1][2] = -1.0;
        let ch0 = assemble_band_log_energy_q8(&s, 0, None, None).unwrap();
        let ch1 = assemble_band_log_energy_q8(&s, 1, None, None).unwrap();
        assert_eq!(ch0[2], 256);
        assert_eq!(ch1[2], -256);
    }

    #[test]
    fn log_energy_f32_to_q8_matches_assembly() {
        assert_eq!(log_energy_f32_to_q8(1.0), 256);
        assert_eq!(log_energy_f32_to_q8(-2.0), -512);
        assert_eq!(log_energy_f32_to_q8(0.5), 128);
        // round-to-nearest: 0.5/256 boundary
        assert_eq!(log_energy_f32_to_q8(0.5 / 256.0 * 1.0), 1);
    }
}
