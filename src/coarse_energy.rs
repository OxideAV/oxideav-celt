//! Coarse-energy decoding (RFC 6716 §4.3.2.1).
//!
//! ## What this module covers
//!
//! CELT encodes the per-band energy envelope with a three-step
//! coarse/fine/finalize strategy (RFC 6716 §4.3.2). The coarse step,
//! described in §4.3.2.1, codes the integer part of the base-2 log of
//! the per-band energy with 6 dB resolution, after a prediction filter
//! that runs jointly across **time** (against the previous frame's
//! final fine-quantised energy) and **frequency** (across bands within
//! the current frame's coarse-quantised energies). The prediction
//! error is entropy-coded with a Laplace distribution whose
//! `(prob, decay)` parameters depend on the frame size and on the
//! intra-vs-inter mode ([`crate::e_prob_model::E_PROB_MODEL`]); the
//! per-symbol decode recurrence is [`crate::laplace::ec_laplace_decode`].
//!
//! Per RFC 6716 §4.3 (Table 55), the standard CELT mode operates on
//! **21 bands** (band index 0..=20). Hybrid mode reuses the same
//! 21-band layout but the first 17 bands (covering 0..8 kHz) are
//! coded by the SILK layer, leaving bands 17..=20 for CELT.
//!
//! ## The prediction filter
//!
//! The §4.3.2.1 filter's 2-D z-transform is
//!
//! ```text
//!                (1 - alpha*z_l^-1) * (1 - z_b^-1)
//! A(z_l, z_b) =  ---------------------------------
//!                         1 - beta*z_b^-1
//! ```
//!
//! where `l` is the frame index (the time arm) and `b` is the band
//! index (the frequency arm). Inverting the filter on the decode side
//! gives the per-band reconstruction recursion (the RFC names the
//! decode step `unquant_coarse_energy` in §4.3.2.1 prose; the recursion
//! below is the clean-room reading of the §4.3.2.1 prediction filter):
//!
//! ```text
//! E[b]  = coef * max(-9.0, E_prev_frame[b]) + prev + q[b]
//! prev += q[b] - beta * q[b]
//! ```
//!
//! with `q[b]` the decoded prediction error (in integer 6 dB steps)
//! and `prev` the running frequency-arm predictor, reset to zero at
//! the start of every frame. The `max(-9.0, ·)` floor is the
//! "prediction is clamped internally" sentence of §4.3.2.1 — it keeps
//! fixed-point and floating-point implementations in the same state.
//! The normative behaviour is the floating-point configuration
//! (RFC 6716 §4.3.2.1 / §4.3 designate floating point as normative), so
//! the recursion here runs in `f32`; energies are in base-2 log units
//! (1.0 = 6 dB).
//!
//! ## Prediction coefficients
//!
//! * Intra mode: `alpha = 0, beta = 4915/32768` — the only pair the
//!   §4.3.2.1 prose states directly ([`INTRA_ALPHA_Q15`],
//!   [`INTRA_BETA_Q15`]).
//! * Inter mode: per-frame-size `(alpha, beta)` Q15 pairs, the
//!   frame-size-dependent coefficients §4.3.2.1 says apply in the
//!   non-intra case (`celt-coarse-energy-and-allocation.md` §1.2),
//!   carried as the numeric `(pred_coef, beta_coef)` data
//!   [`PRED_COEF_Q15`] / [`BETA_COEF_Q15`].
//!
//! ## Budget-constrained fallbacks
//!
//! At very low rates the Laplace symbol may not fit in the remaining
//! frame budget. Per the §4.3.2.1 budget accounting
//! (`celt-laplace-decode.md` §1), each band/channel slot degrades
//! through three fallbacks keyed on `budget - tell()`:
//!
//! * `>= 15` bits left — full Laplace decode.
//! * `>= 2` — a 2-bit zig-zag symbol over [`SMALL_ENERGY_ICDF`]
//!   (`qi ∈ {-1, 0, +1}`).
//! * `>= 1` — a single `{1,1}/2` bit (`qi ∈ {-1, 0}`).
//! * otherwise — `qi = -1` without consuming any bits.
//!
//! ## Clean-room provenance
//!
//! The model and step order are RFC 6716 §4.3.2.1 (the prediction
//! filter, the intra coefficients `alpha=0, beta=4915/32768`, and the
//! internal clamp the prose mandates). The per-symbol Laplace
//! recurrence (named `ec_laplace_decode` by the RFC) is implemented in
//! [`crate::laplace`] from the clean-room narrative
//! `docs/audio/celt/spec/celt-laplace-decode.md`; the `{prob, decay}`
//! parameter table and the per-LM prediction/`beta` coefficients are
//! the data extractions `docs/audio/celt/tables/e_prob_model.csv` and
//! the narrative `celt-coarse-energy-and-allocation.md` §1. No external
//! library source — including the RFC's Appendix A reference listing —
//! was consulted.

use crate::e_prob_model::{E_PROB_MODEL, NUM_LM_FRAME_SIZES, PRED_INTER, PRED_INTRA};
use crate::laplace::ec_laplace_decode;
use crate::range_decoder::RangeDecoder;
use crate::Error;

/// Number of CELT bands per RFC 6716 Table 55 (band index 0..=20).
///
/// Hybrid-mode CELT reuses the same band layout but only codes bands
/// 17..=20 (the SILK layer covers bands 0..=16 below 8 kHz). Pure
/// CELT codes all 21 bands.
pub const NUM_BANDS: usize = 21;

/// Maximum number of channels a CELT frame codes (mono or stereo,
/// RFC 6716 §4.3).
pub const MAX_CHANNELS: usize = 2;

/// Intra-mode prediction coefficient α in Q15 fixed-point per
/// RFC 6716 §4.3.2.1.
///
/// The RFC states "alpha=0 ... when using intra energy": the time arm
/// of the prediction filter vanishes entirely on intra frames.
pub const INTRA_ALPHA_Q15: i32 = 0;

/// Intra-mode prediction coefficient β in Q15 fixed-point per
/// RFC 6716 §4.3.2.1.
///
/// The RFC writes the value as the fraction `4915/32768`. Since 32768
/// is `2^15`, the Q15 numerator is the literal 4915.
pub const INTRA_BETA_Q15: i32 = 4915;

/// Inter-mode time-arm prediction coefficients α in Q15, indexed by
/// `LM = log2(frame_size / 120) ∈ 0..=3` — the per-frame-size
/// `pred_coef` data. §4.3.2.1 prose: "The prediction coefficients
/// applied depend on the frame size in use when not using intra
/// energy" (`celt-coarse-energy-and-allocation.md` §1.2).
pub const PRED_COEF_Q15: [i32; NUM_LM_FRAME_SIZES] = [29440, 26112, 21248, 16384];

/// Inter-mode frequency-arm prediction coefficients β in Q15, indexed
/// by `LM` — the per-frame-size `beta_coef` data
/// (`celt-coarse-energy-and-allocation.md` §1.2).
pub const BETA_COEF_Q15: [i32; NUM_LM_FRAME_SIZES] = [30147, 22282, 12124, 6554];

/// Inverse-CDF table for the 2-bit low-budget fallback symbol — the
/// `small_energy_icdf` data: PDF `{2, 1, 1}/4` over the zig-zag-coded
/// `qi ∈ {0, -1, +1}` (§4.3.2.1 low-rate fallback).
pub const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

/// Q15 scale used by the prediction-filter coefficients above.
const Q15_ONE: i32 = 1 << 15;

/// The §4.3.2.1 internal prediction clamp: the previous frame's
/// log-energy is floored at -9.0 (base-2 log units) before the time
/// arm multiplies it — the internal clamp RFC 6716 §4.3.2.1 mandates so
/// that fixed- and floating-point decoders stay in the same state.
const ENERGY_FLOOR: f32 = -9.0;

/// Minimum whole-bit budget headroom required to decode a full
/// Laplace symbol for one band/channel slot. Below this the §4.3.2.1
/// decoder takes the low-rate fallbacks instead
/// (`celt-laplace-decode.md` §1: "the Laplace path is taken only when
/// at least 15 range-coder bits of budget remain").
const LAPLACE_MIN_BUDGET_BITS: u32 = 15;

/// Per-band, per-channel running log-energy state across CELT frames.
///
/// `energy[c][b]` is channel `c`'s band-`b` log-2 energy (1.0 = 6 dB)
/// as of the most recently decoded frame. The §4.3.2.1 inter-frame
/// time arm predicts against this state; the §4.3.2.2 fine and
/// finalize refinements further adjust it downstream before the next
/// frame's prediction runs.
///
/// Decoder lifecycle:
///
/// * [`CoarseEnergyState::new`] on stream open (all-zero history).
/// * [`decode_coarse_energy`] reads one frame's coarse envelope and
///   updates `energy` in place.
/// * A decoder reset (packet-loss recovery, mode switch) calls
///   [`CoarseEnergyState::reset`], matching the encoder's expected
///   state.
#[derive(Debug, Clone, Copy)]
pub struct CoarseEnergyState {
    /// Per-channel, per-band base-2 log-energy from the most recent
    /// frame. Zero on stream open and after any decoder reset.
    pub energy: [[f32; NUM_BANDS]; MAX_CHANNELS],
}

impl CoarseEnergyState {
    /// Construct a freshly-reset coarse-energy state. All bands start
    /// at zero log-energy.
    pub fn new() -> Self {
        Self {
            energy: [[0.0; NUM_BANDS]; MAX_CHANNELS],
        }
    }

    /// Zero the carried energies (§4.5.2 decoder reset).
    pub fn reset(&mut self) {
        self.energy = [[0.0; NUM_BANDS]; MAX_CHANNELS];
    }
}

impl Default for CoarseEnergyState {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode one CELT frame's coarse energy envelope
/// (RFC 6716 §4.3.2.1, the `unquant_coarse_energy` step the prose
/// names).
///
/// * `state` carries the previous frame's per-band log-energies and
///   is updated in place with this frame's coarse-quantised values
///   for bands `start..end` of each coded channel (other bands and
///   channels are left untouched).
/// * `intra` is the §4.3.2.1 intra flag decoded by
///   [`crate::frame_header::CeltFrameHeader::decode_prefix`]; it
///   selects `alpha = 0, beta = 4915/32768` and the intra column of
///   `E_PROB_MODEL`.
/// * `lm` is `log2(frame_size / 120) ∈ 0..=3` per RFC 6716 §4.3.3.
/// * `start..end` is the coded band window: `0..21` for pure CELT,
///   `17..21` for hybrid mode.
/// * `channels` is 1 (mono) or 2 (stereo). Channels interleave
///   within each band (band-major, channel-minor), matching the
///   bitstream order.
///
/// The per-slot decode degrades through the budget-keyed fallbacks
/// described in the module docs; the budget is the frame size in
/// bits ([`RangeDecoder::storage_bits`]).
///
/// Returns [`Error::InvalidParameter`] for an out-of-range `lm`,
/// band window, or channel count; the decoder and state are not
/// touched in that case.
pub fn decode_coarse_energy(
    dec: &mut RangeDecoder<'_>,
    state: &mut CoarseEnergyState,
    intra: bool,
    lm: u32,
    start: usize,
    end: usize,
    channels: usize,
) -> Result<(), Error> {
    if lm as usize >= NUM_LM_FRAME_SIZES
        || start > end
        || end > NUM_BANDS
        || channels == 0
        || channels > MAX_CHANNELS
    {
        return Err(Error::InvalidParameter);
    }
    let lm = lm as usize;
    let pred = if intra { PRED_INTRA } else { PRED_INTER };
    // Normative float configuration: the Q15 integer coefficients
    // divide out exactly (the numerators are < 2^15, so the f32
    // quotients are exact dyadic rationals).
    let (coef, beta) = if intra {
        (0.0_f32, INTRA_BETA_Q15 as f32 / Q15_ONE as f32)
    } else {
        (
            PRED_COEF_Q15[lm] as f32 / Q15_ONE as f32,
            BETA_COEF_Q15[lm] as f32 / Q15_ONE as f32,
        )
    };
    let budget = dec.storage_bits();
    // The frequency-arm predictor resets at every frame boundary,
    // independently per channel.
    let mut prev = [0.0_f32; MAX_CHANNELS];
    for band in start..end {
        for (c, prev_c) in prev.iter_mut().enumerate().take(channels) {
            let tell = dec.tell();
            let bits_left = budget.saturating_sub(tell);
            let qi: i32 = if bits_left >= LAPLACE_MIN_BUDGET_BITS {
                // Full Laplace decode. The table stores Q8 bytes; the
                // Laplace decoder wants the zero-probability in Q15
                // (<< 7) and the decay in Q14 (<< 6).
                let pd = E_PROB_MODEL[lm][pred][band.min(NUM_BANDS - 1)];
                ec_laplace_decode(dec, (pd.prob as u32) << 7, (pd.decay as u32) << 6)
            } else if bits_left >= 2 {
                // 2-bit zig-zag fallback: symbol s ∈ {0, 1, 2} maps to
                // qi ∈ {0, -1, +1} via (s >> 1) ^ -(s & 1).
                let s = dec.dec_icdf(&SMALL_ENERGY_ICDF, 2) as i32;
                (s >> 1) ^ -(s & 1)
            } else if bits_left >= 1 {
                // 1-bit fallback: qi ∈ {0, -1}.
                -(dec.dec_bit_logp(1) as i32)
            } else {
                // No bits left at all: the implicit prediction error.
                -1
            };
            let q = qi as f32;
            // §4.3.2.1 internal clamp on the time-arm input.
            let old = state.energy[c][band].max(ENERGY_FLOOR);
            state.energy[c][band] = coef * old + *prev_c + q;
            *prev_c += q - beta * q;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The band count comes from RFC 6716 Table 55, which enumerates
    /// 21 bands (index 0..=20). Pin the count so future refactors
    /// don't silently drift.
    #[test]
    fn num_bands_matches_rfc_table_55() {
        assert_eq!(NUM_BANDS, 21);
        assert_eq!(MAX_CHANNELS, 2);
    }

    /// The intra prediction coefficients are stated in RFC 6716
    /// §4.3.2.1: `alpha=0, beta=4915/32768`. The inter coefficient
    /// rows are the per-frame-size `pred_coef` / `beta_coef` Q15 data
    /// and must hold one Q15 value per LM.
    #[test]
    fn prediction_coefficients_match_spec() {
        assert_eq!(INTRA_ALPHA_Q15, 0);
        assert_eq!(INTRA_BETA_Q15, 4915);
        assert_eq!(Q15_ONE, 32_768);
        assert_eq!(PRED_COEF_Q15, [29440, 26112, 21248, 16384]);
        assert_eq!(BETA_COEF_Q15, [30147, 22282, 12124, 6554]);
        // Every coefficient is a proper Q15 fraction (< 1.0), so the
        // recursion is stable in both arms.
        for lm in 0..NUM_LM_FRAME_SIZES {
            assert!(PRED_COEF_Q15[lm] > 0 && PRED_COEF_Q15[lm] < Q15_ONE);
            assert!(BETA_COEF_Q15[lm] > 0 && BETA_COEF_Q15[lm] < Q15_ONE);
        }
        // The time-arm coefficient weakens as frames get longer
        // (more time between predictions), monotonically.
        for lm in 1..NUM_LM_FRAME_SIZES {
            assert!(PRED_COEF_Q15[lm] < PRED_COEF_Q15[lm - 1]);
            assert!(BETA_COEF_Q15[lm] < BETA_COEF_Q15[lm - 1]);
        }
    }

    /// The 2-bit fallback table is the `{2, 1, 1}/4` PDF in §4.1.3.3
    /// ICDF form, with the mandatory terminating 0.
    #[test]
    fn small_energy_icdf_shape() {
        assert_eq!(SMALL_ENERGY_ICDF, [2, 1, 0]);
        // Strictly decreasing with terminating zero per §4.1.3.3.
        assert!(SMALL_ENERGY_ICDF[0] > SMALL_ENERGY_ICDF[1]);
        assert!(SMALL_ENERGY_ICDF[1] > SMALL_ENERGY_ICDF[2]);
        assert_eq!(*SMALL_ENERGY_ICDF.last().unwrap(), 0);
    }

    /// A freshly-constructed state has all bands of both channels at
    /// zero log-energy; `reset()` restores that after mutation.
    #[test]
    fn new_state_is_all_zero_and_resets() {
        let mut state = CoarseEnergyState::new();
        assert_eq!(state.energy, [[0.0; NUM_BANDS]; MAX_CHANNELS]);
        assert_eq!(CoarseEnergyState::default().energy, state.energy);
        state.energy[1][20] = -3.5;
        state.reset();
        assert_eq!(state.energy, [[0.0; NUM_BANDS]; MAX_CHANNELS]);
    }

    /// Out-of-range parameters are rejected without touching the
    /// decoder or the carried state.
    #[test]
    fn invalid_parameters_rejected() {
        let buf = [0u8; 8];
        let mut dec = RangeDecoder::new(&buf);
        let mut state = CoarseEnergyState::new();
        let tell_before = dec.tell();
        // lm out of range.
        assert_eq!(
            decode_coarse_energy(&mut dec, &mut state, true, 4, 0, NUM_BANDS, 1),
            Err(Error::InvalidParameter)
        );
        // band window out of range.
        assert_eq!(
            decode_coarse_energy(&mut dec, &mut state, true, 0, 0, NUM_BANDS + 1, 1),
            Err(Error::InvalidParameter)
        );
        // inverted band window.
        assert_eq!(
            decode_coarse_energy(&mut dec, &mut state, true, 0, 5, 4, 1),
            Err(Error::InvalidParameter)
        );
        // channel counts.
        assert_eq!(
            decode_coarse_energy(&mut dec, &mut state, true, 0, 0, NUM_BANDS, 0),
            Err(Error::InvalidParameter)
        );
        assert_eq!(
            decode_coarse_energy(&mut dec, &mut state, true, 0, 0, NUM_BANDS, 3),
            Err(Error::InvalidParameter)
        );
        assert_eq!(dec.tell(), tell_before);
        assert_eq!(state.energy, [[0.0; NUM_BANDS]; MAX_CHANNELS]);
    }

    /// An empty band window decodes nothing and leaves both the
    /// decoder and the state untouched.
    #[test]
    fn empty_band_window_is_noop() {
        let buf = [0xA5u8; 8];
        let mut dec = RangeDecoder::new(&buf);
        let mut state = CoarseEnergyState::new();
        let tell_before = dec.tell();
        decode_coarse_energy(&mut dec, &mut state, false, 2, 7, 7, 2).unwrap();
        assert_eq!(dec.tell(), tell_before);
        assert_eq!(state.energy, [[0.0; NUM_BANDS]; MAX_CHANNELS]);
    }

    /// With an empty frame the budget is zero, so every band/channel
    /// slot takes the no-bits fallback `qi = -1` and the decoder is
    /// never consulted. The reconstruction must then follow the
    /// §4.3.2.1 prediction recursion exactly; verify against an
    /// independent evaluation with the intra coefficients.
    #[test]
    fn zero_budget_intra_matches_hand_recursion() {
        let mut dec = RangeDecoder::new(&[]);
        let mut state = CoarseEnergyState::new();
        let tell_before = dec.tell();
        decode_coarse_energy(&mut dec, &mut state, true, 0, 0, NUM_BANDS, 1).unwrap();
        // No bits were available, so the decoder was never touched.
        assert_eq!(dec.tell(), tell_before);

        // Independent recursion: intra => coef = 0, beta = 4915/32768,
        // qi = -1 everywhere.
        let beta = 4915.0_f32 / 32768.0;
        let mut prev = 0.0_f32;
        for band in 0..NUM_BANDS {
            let q = -1.0_f32;
            let expected = prev + q; // coef = 0 kills the time arm
            assert_eq!(
                state.energy[0][band], expected,
                "band {band} mismatch vs hand recursion"
            );
            prev += q - beta * q;
        }
        // Channel 1 was not coded (mono) and stays untouched.
        assert_eq!(state.energy[1], [0.0; NUM_BANDS]);
    }

    /// Zero-budget inter mode exercises the time arm: pre-seed the
    /// previous frame's energies (including one below the -9.0 floor)
    /// and verify the clamp + `coef * old + prev + q` recursion.
    #[test]
    fn zero_budget_inter_applies_time_arm_and_floor() {
        let lm = 2usize;
        let mut dec = RangeDecoder::new(&[]);
        let mut state = CoarseEnergyState::new();
        // Seed history: band 0 sits below the floor, others above.
        state.energy[0] = std::array::from_fn(|band| band as f32 * 0.25 - 2.0);
        state.energy[0][0] = -50.0;
        let seeded = state.energy[0];

        decode_coarse_energy(&mut dec, &mut state, false, lm as u32, 0, NUM_BANDS, 1).unwrap();

        let coef = PRED_COEF_Q15[lm] as f32 / 32768.0;
        let beta = BETA_COEF_Q15[lm] as f32 / 32768.0;
        let mut prev = 0.0_f32;
        for (band, &seed) in seeded.iter().enumerate() {
            let q = -1.0_f32;
            let old = seed.max(-9.0);
            let expected = coef * old + prev + q;
            assert_eq!(
                state.energy[0][band], expected,
                "band {band} mismatch vs hand recursion"
            );
            prev += q - beta * q;
        }
        // The floor engaged on band 0: the time arm saw -9.0, not -50.
        assert_eq!(state.energy[0][0], coef * -9.0 - 1.0);
    }

    /// The hybrid window (bands 17..21) leaves bands 0..17 untouched.
    #[test]
    fn hybrid_window_only_touches_coded_bands() {
        let mut dec = RangeDecoder::new(&[]);
        let mut state = CoarseEnergyState::new();
        state.energy[0] = std::array::from_fn(|band| 1.0 + band as f32);
        let seeded = state.energy[0];
        decode_coarse_energy(&mut dec, &mut state, true, 3, 17, NUM_BANDS, 1).unwrap();
        for (band, &seed) in seeded.iter().enumerate() {
            if band < 17 {
                assert_eq!(state.energy[0][band], seed, "band {band} clobbered");
            } else {
                assert_ne!(state.energy[0][band], seed, "band {band} unchanged");
            }
        }
    }

    /// Stereo interleaves channels within each band and carries an
    /// independent frequency-arm predictor per channel: with a zero
    /// budget both channels of a band receive identical treatment, so
    /// their reconstructions from identical histories must agree.
    #[test]
    fn stereo_channels_track_independently_but_identically() {
        let mut dec = RangeDecoder::new(&[]);
        let mut state = CoarseEnergyState::new();
        state.energy[0][3] = 2.5;
        state.energy[1][3] = 2.5;
        decode_coarse_energy(&mut dec, &mut state, false, 1, 0, NUM_BANDS, 2).unwrap();
        assert_eq!(state.energy[0], state.energy[1]);
    }

    /// With a real (non-empty) buffer the full Laplace path runs for
    /// every slot and consumes range-coder bits; the decode must stay
    /// in sync (no sticky error) for arbitrary stream bytes.
    #[test]
    fn laplace_path_consumes_bits_and_stays_in_sync() {
        let buf: Vec<u8> = (0..64u32).map(|i| (i * 89 + 3) as u8).collect();
        for intra in [false, true] {
            for lm in 0..NUM_LM_FRAME_SIZES as u32 {
                let mut dec = RangeDecoder::new(&buf);
                let mut state = CoarseEnergyState::new();
                let tell_before = dec.tell();
                decode_coarse_energy(&mut dec, &mut state, intra, lm, 0, NUM_BANDS, 2).unwrap();
                assert!(dec.tell() > tell_before, "no bits consumed");
                assert!(!dec.has_error(), "sticky error lm={lm} intra={intra}");
                for c in 0..2 {
                    for band in 0..NUM_BANDS {
                        assert!(
                            state.energy[c][band].is_finite(),
                            "non-finite energy at c={c} band={band}"
                        );
                    }
                }
            }
        }
    }
}
