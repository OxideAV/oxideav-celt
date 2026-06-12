//! Single-band shape-decode orchestrator (RFC 6716 §4.3.4 → §4.3.6).
//!
//! The per-band CELT decode chain for the **simplest case** — a band
//! whose codebook `V(N, K)` fits in 32 bits, so no §4.3.4.4 split is
//! required — is a fixed composition of the primitives the earlier
//! rounds already wired up. RFC 6716 §4.3.4 describes the chain
//! (lines 6462–6474):
//!
//! > In the simplest case, the number of bits allocated in
//! > Section 4.3.3 is converted to a number of pulses as described by
//! > Section 4.3.4.1. Knowing the number of pulses and the number of
//! > samples in the band, the decoder calculates the size of the
//! > codebook as detailed in Section 4.3.4.2. The size is used to
//! > decode an unsigned integer (uniform probability model), which is
//! > the codeword index. This index is converted into the
//! > corresponding vector as explained in Section 4.3.4.2. This vector
//! > is then scaled to unit norm.
//!
//! The unit-norm shape is then transformed, in §4.3 bitstream order,
//! by:
//!
//! 1. **§4.3.4.3 spreading rotation** — the `theta = pi * g_r^2 / 4`
//!    N-D rotation applied per time block ([`crate::spread_rotation::apply_spread`]).
//! 2. **§4.3.4.5 time-frequency resolution change** — `|tf|` Hadamard
//!    levels applied per the per-band TF adjustment
//!    ([`crate::hadamard::apply_tf_resolution_change`]). A zero
//!    adjustment is a no-op.
//! 3. **§4.3.6 denormalization** — the unit-norm shape is multiplied
//!    by `sqrt(2^(E_q8 / 256))`, the square root of the decoded band
//!    energy ([`crate::denormalization::denormalize_band_in_place_f32`]).
//!
//! Each of steps 1–2 is L2-norm-preserving (orthonormal composition),
//! so the unit-norm property the §4.3.6 step relies on survives the
//! spreading and TF passes; step 3 then scales the unit shape to the
//! decoded energy. The output is the MDCT-domain band the inverse
//! MDCT (§4.3.7, a future round) consumes.
//!
//! ## Scope
//!
//! This is the **non-split** single-band path. Bands whose codebook
//! exceeds the §4.3.4.4 32-bit budget ([`crate::band_split::band_needs_split`])
//! decode through the recursive halving tree
//! ([`crate::band_split::plan_band_split`]); the quantized split-gain
//! parameter that redistributes the L2 norm across the two halves is a
//! documented gap (the §4.3.4.4 prose defers its precision/PDF to the
//! reference), so the split path is not orchestrated here. Callers can
//! gate on `band_needs_split(n, k)` before invoking
//! [`decode_band_shape`].
//!
//! Stereo joint-coding (the `itheta` angle and the mid/side mixing the
//! §4.3.4.4 final sentence shares with the split mechanism) is also out
//! of scope for the same docs-gap reason; this path is the mono /
//! per-channel band decode.
//!
//! ## Provenance
//!
//! Every step is a composition of RFC 6716-grounded primitives from the
//! prior rounds. The chain ordering (§4.3.4 shape → §4.3.4.3 spread →
//! §4.3.4.5 TF → §4.3.6 denormalize) follows the §4.3 bitstream /
//! decode order. No external library source consulted; no new numeric
//! constant introduced.

use crate::denormalization::denormalize_band_in_place_f32;
use crate::hadamard::apply_tf_resolution_change;
use crate::pvq::decode_unit_shape;
use crate::range_decoder::RangeDecoder;
use crate::spread::Spread;
use crate::spread_rotation::apply_spread;

/// The decoded MDCT-domain samples for a single band, plus the pulse
/// count `K` the §4.3.4.1 allocation produced (carried through so the
/// caller can drive the §4.3.4.1 balance accumulator without re-running
/// the search).
#[derive(Debug, Clone, PartialEq)]
pub struct BandShape {
    /// The denormalized MDCT-domain band samples (length `N`, the band
    /// dimension). Interleaved across `nb_blocks` time blocks when the
    /// band spans multiple short-MDCT blocks (the layout
    /// [`apply_tf_resolution_change`] and
    /// [`crate::spread_rotation::apply_nd_rotation_multi_block`] use).
    pub samples: Vec<f32>,
    /// The pulse count `K` consumed for this band.
    pub k: u32,
}

/// Decode one non-split band's shape from the range decoder and return
/// the denormalized MDCT-domain samples (RFC 6716 §4.3.4 → §4.3.6).
///
/// Parameters:
/// * `dec` — the range decoder positioned at this band's PVQ index.
/// * `n` — the band dimension (number of MDCT bins for this band, the
///   §4.3 Table 55 `N[band]` value for the active `LM`, possibly times
///   `nb_blocks` for a multi-block short-MDCT band).
/// * `k` — the pulse count from the §4.3.4.1 bits-to-pulses search.
/// * `spread` — the §4.3.4.3 spreading parameter
///   ([`crate::spread::decode_spread`]).
/// * `nb_blocks` — the number of interleaved time blocks (1 for a
///   non-transient long-MDCT band).
/// * `tf_adjustment` — the §4.3.4.5 per-band TF adjustment
///   ([`crate::tf_change::tf_adjustment`]); `0` is a no-op.
/// * `log_energy_q8` — the §4.3.2.1/§4.3.2.2 reconstructed band energy
///   in the Q8 base-2 log domain
///   ([`crate::coarse_energy::CoarseEnergyState`] carries the
///   normative f32 form; multiply by 256 and round to get this value).
///
/// Returns `None` when the PVQ decode fails (saturated codebook —
/// caller must split per §4.3.4.4 — `N == 0` with `K > 0`, a sticky
/// range-decoder error, or an out-of-range index) or when the spreading
/// / TF shape constraints are violated (`nb_blocks == 0`, a band length
/// not divisible by `nb_blocks`, or a TF request exceeding the
/// available Hadamard levels).
///
/// The §4.3.4.1 "ties round down" pulse-count rule is the caller's
/// responsibility (the `k` argument is already the result of that
/// search); this function consumes exactly the PVQ index for `(N, K)`.
pub fn decode_band_shape(
    dec: &mut RangeDecoder<'_>,
    n: u32,
    k: u32,
    spread: Spread,
    nb_blocks: u32,
    tf_adjustment: i8,
    log_energy_q8: i32,
) -> Option<BandShape> {
    // §4.3.4 / §4.3.4.1 / §4.3.4.2: pulses → codebook index → vector →
    // unit norm.
    let mut samples = decode_unit_shape(dec, n, k)?;

    // §4.3.4.3: spreading rotation (per time block). A `false` return is
    // a shape-constraint violation the caller must not have produced; we
    // surface it as `None` so a malformed band parameterisation cannot
    // silently emit an un-rotated (wrong-energy-distribution) shape.
    if !apply_spread(spread, &mut samples, k, nb_blocks) {
        return None;
    }

    // §4.3.4.5: time-frequency resolution change. A zero adjustment is
    // an internal no-op; a non-zero one applies `|tf_adjustment|`
    // Hadamard levels in natural or sequency order per the §4.3.4.5
    // sign convention. `apply_tf_resolution_change` returns `false` on
    // a shape-constraint violation (non-power-of-two block count or
    // sub-vector length, or a request exceeding the available levels).
    if tf_adjustment != 0
        && !apply_tf_resolution_change(&mut samples, tf_adjustment, nb_blocks as usize)
    {
        return None;
    }

    // §4.3.6: denormalize by the square root of the decoded energy.
    denormalize_band_in_place_f32(&mut samples, log_energy_q8);

    Some(BandShape { samples, k })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::denormalization::log_energy_q8_to_amplitude_f32;

    fn l2_norm(samples: &[f32]) -> f64 {
        samples
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt()
    }

    /// A range decoder seeded with a `0xFF…` buffer decodes a
    /// well-formed band: the energy of the output equals the linear
    /// energy `2^(E_q8/256)` because the §4.3.4.3 / §4.3.4.5 passes
    /// preserve the unit norm and §4.3.6 scales it by the amplitude.
    #[test]
    fn energy_equals_linear_energy_no_spread_no_tf() {
        let buf = [0xFFu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let n = 4;
        let k = 3;
        let log_energy_q8 = 512; // two integer log-2 steps ⇒ amplitude 2.0
        let shape = decode_band_shape(&mut dec, n, k, Spread::None, 1, 0, log_energy_q8).unwrap();
        assert_eq!(shape.samples.len(), n as usize);
        assert_eq!(shape.k, k);
        // sum(|X|) == K for the pulse vector, so the unit-norm shape has
        // norm 1; after §4.3.6 the norm is the amplitude.
        let amp = log_energy_q8_to_amplitude_f32(log_energy_q8) as f64;
        assert!(
            (l2_norm(&shape.samples) - amp).abs() < 1e-4,
            "band energy {} != amplitude {}",
            l2_norm(&shape.samples),
            amp
        );
    }

    /// Zero energy (`E_q8 = 0`) yields amplitude 1.0, so the output is
    /// exactly the unit-norm shape (norm 1).
    #[test]
    fn zero_energy_is_unit_norm() {
        let buf = [0x5Au8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let shape = decode_band_shape(&mut dec, 6, 2, Spread::None, 1, 0, 0).unwrap();
        assert!((l2_norm(&shape.samples) - 1.0).abs() < 1e-4);
    }

    /// Spreading rotation preserves the L2 norm, so the post-§4.3.6
    /// energy still equals the amplitude when a non-identity spread is
    /// applied.
    #[test]
    fn spread_preserves_energy() {
        let buf = [0xA3u8; 24];
        let mut dec = RangeDecoder::new(&buf);
        let log_energy_q8 = 256; // one log-2 step ⇒ amplitude sqrt(2)
        let amp = log_energy_q8_to_amplitude_f32(log_energy_q8) as f64;
        let shape =
            decode_band_shape(&mut dec, 8, 4, Spread::Aggressive, 1, 0, log_energy_q8).unwrap();
        assert!(
            (l2_norm(&shape.samples) - amp).abs() < 1e-3,
            "spread changed band energy: {} vs {}",
            l2_norm(&shape.samples),
            amp
        );
    }

    /// A TF resolution change preserves the L2 norm (the Hadamard
    /// transform is orthonormal), so the post-§4.3.6 energy is
    /// unchanged from the no-TF case at the same energy.
    #[test]
    fn tf_change_preserves_energy() {
        // 2 blocks × 4 samples, interleaved. tf_adjustment = -1 applies
        // one sequency-ordered WHT level per sub-vector.
        let buf = [0x7Cu8; 32];
        let mut dec = RangeDecoder::new(&buf);
        let log_energy_q8 = 0;
        let shape = decode_band_shape(&mut dec, 8, 3, Spread::None, 2, -1, log_energy_q8).unwrap();
        assert!(
            (l2_norm(&shape.samples) - 1.0).abs() < 1e-4,
            "TF change altered band energy: {}",
            l2_norm(&shape.samples)
        );
    }

    /// The `+tf` (frequency-resolution-increase) branch also preserves
    /// the norm: 4 blocks × 2 samples with `tf_adjustment = +1`.
    #[test]
    fn tf_change_positive_preserves_energy() {
        let buf = [0x33u8; 32];
        let mut dec = RangeDecoder::new(&buf);
        let shape = decode_band_shape(&mut dec, 8, 2, Spread::None, 4, 1, 0).unwrap();
        assert!((l2_norm(&shape.samples) - 1.0).abs() < 1e-4);
    }

    /// `N == 0` with `K > 0` is rejected by the PVQ layer ⇒ `None`.
    #[test]
    fn zero_n_positive_k_is_none() {
        let buf = [0xFFu8; 8];
        let mut dec = RangeDecoder::new(&buf);
        assert!(decode_band_shape(&mut dec, 0, 1, Spread::None, 1, 0, 0).is_none());
    }

    /// A saturated codebook (`V(N, K)` overflows 32 bits) must be split
    /// per §4.3.4.4 before this path runs; the orchestrator returns
    /// `None` so the caller routes to the split walker.
    #[test]
    fn saturated_codebook_is_none() {
        let buf = [0xFFu8; 8];
        let mut dec = RangeDecoder::new(&buf);
        // V(180, 180) saturates per the pvq tests.
        assert!(decode_band_shape(&mut dec, 180, 180, Spread::None, 1, 0, 0).is_none());
    }

    /// `nb_blocks == 0` is a shape-constraint violation ⇒ `None` (after
    /// the unit-shape decode succeeds, the spreading guard rejects it).
    #[test]
    fn zero_blocks_is_none() {
        let buf = [0xFFu8; 8];
        let mut dec = RangeDecoder::new(&buf);
        assert!(decode_band_shape(&mut dec, 4, 2, Spread::Normal, 0, 0, 0).is_none());
    }

    /// A band length not divisible by `nb_blocks` is rejected by the
    /// spreading guard ⇒ `None`.
    #[test]
    fn indivisible_blocks_is_none() {
        let buf = [0xFFu8; 8];
        let mut dec = RangeDecoder::new(&buf);
        // N = 6 is not divisible by nb_blocks = 4.
        assert!(decode_band_shape(&mut dec, 6, 2, Spread::Normal, 4, 0, 0).is_none());
    }

    /// A TF request exceeding the available Hadamard levels is rejected
    /// ⇒ `None`. With 2 blocks × 2 samples, only one WHT level is
    /// available per 2-sample sub-vector; `tf_adjustment = -2` asks for
    /// two.
    #[test]
    fn oversized_tf_is_none() {
        let buf = [0xFFu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        assert!(decode_band_shape(&mut dec, 4, 2, Spread::None, 2, -2, 0).is_none());
    }

    /// Negative energy (`E_q8 < 0`) attenuates: the output norm is the
    /// sub-unity amplitude.
    #[test]
    fn negative_energy_attenuates() {
        let buf = [0x9Eu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let log_energy_q8 = -512; // amplitude 0.25
        let amp = log_energy_q8_to_amplitude_f32(log_energy_q8) as f64;
        let shape = decode_band_shape(&mut dec, 4, 2, Spread::None, 1, 0, log_energy_q8).unwrap();
        assert!((l2_norm(&shape.samples) - amp).abs() < 1e-4);
        assert!(amp < 1.0);
    }

    /// The decoded `BandShape::k` round-trips the supplied pulse count.
    #[test]
    fn band_shape_carries_k() {
        let buf = [0xC7u8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let shape = decode_band_shape(&mut dec, 5, 7, Spread::Light, 1, 0, 0).unwrap();
        assert_eq!(shape.k, 7);
        assert_eq!(shape.samples.len(), 5);
    }

    /// `K == 0` decodes the all-zero pulse vector ⇒ the unit-norm step
    /// degrades to all-zero, and §4.3.6 leaves it all-zero regardless of
    /// energy (no pseudo-random fill — that is the §4.3.5 anti-collapse
    /// step, a separate gap'd stage).
    #[test]
    fn zero_k_is_all_zero() {
        let buf = [0xFFu8; 8];
        let mut dec = RangeDecoder::new(&buf);
        let tell_before = dec.tell();
        let shape = decode_band_shape(&mut dec, 4, 0, Spread::None, 1, 0, 512).unwrap();
        assert!(shape.samples.iter().all(|&x| x == 0.0));
        // K = 0 does not consume a PVQ index (V(N, 0) = 1 ⇒ no bits).
        assert_eq!(dec.tell(), tell_before);
    }
}
