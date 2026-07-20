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
//! the narrative `celt-coarse-energy-and-allocation.md` §1. The
//! two-pass rate-distortion encoder ([`quant_coarse_energy_rd`]) is
//! transcribed from the RFC's own Appendix A reference listing
//! (`quant_coarse_energy` / `quant_coarse_energy_impl` /
//! `loss_distortion`), extracted from the staged RFC text per §A.1
//! and SHA-1-verified; no source outside the staged docs was
//! consulted.

use crate::e_prob_model::{E_PROB_MODEL, NUM_LM_FRAME_SIZES, PRED_INTER, PRED_INTRA};
use crate::laplace::{ec_laplace_decode, ec_laplace_encode};
use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
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

/// Encode one CELT frame's coarse energy envelope
/// (RFC 6716 §4.3.2.1) — the exact inverse of [`decode_coarse_energy`].
///
/// Given the encoder's chosen per-band target log-2 energies
/// (`target[c][band]`, 1.0 = 6 dB) this walks the same
/// band-major/channel-minor order the decoder reads, quantizes each
/// band's prediction error to the nearest integer 6 dB step
/// `qi = round(target - prediction)` — the natural inverse of the
/// decoder reconstruction `E = prediction + qi` — and writes it through
/// the same budget-keyed dispatch the decoder uses:
///
/// * `>= 15` bits left — full Laplace encode ([`ec_laplace_encode`]).
/// * `>= 2` — the 2-bit zig-zag [`SMALL_ENERGY_ICDF`] symbol
///   (`qi` clamped to `{-1, 0, +1}`).
/// * `>= 1` — a single `{1,1}/2` bit (`qi` clamped to `{-1, 0}`).
/// * otherwise — the implicit `qi = -1`, no bits written.
///
/// The prediction state is updated with the **actually coded** `qi`
/// (post-clamp, post-Laplace-clamp), so a subsequent
/// [`decode_coarse_energy`] over the produced frame reconstructs the
/// identical per-band `state.energy` and the two sides stay in the
/// exact range-coder lockstep RFC 6716 §4.3.3 requires.
///
/// * `budget` is the intended total frame size in bits — the value the
///   decoder will report from [`RangeDecoder::storage_bits`]. The
///   encoder must therefore know the target frame size so its fallback
///   dispatch matches the decoder's. `enc.tell()` locksteps with the
///   decoder's `dec.tell()` after the same preceding symbols (§5.1.6 /
///   §4.1.6), so the caller encodes the Table-56 prefix into `enc`
///   before this call, exactly as a real frame is laid out.
///
/// Returns [`Error::InvalidParameter`] for an out-of-range `lm`, band
/// window, or channel count (the encoder and state are untouched), or
/// propagates a [`RangeEncoder`] rejection (a finalized stream).
#[allow(clippy::too_many_arguments)]
pub fn encode_coarse_energy(
    enc: &mut RangeEncoder,
    state: &mut CoarseEnergyState,
    target: &[[f32; NUM_BANDS]; MAX_CHANNELS],
    intra: bool,
    lm: u32,
    start: usize,
    end: usize,
    channels: usize,
    budget: u32,
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
    let (coef, beta) = if intra {
        (0.0_f32, INTRA_BETA_Q15 as f32 / Q15_ONE as f32)
    } else {
        (
            PRED_COEF_Q15[lm] as f32 / Q15_ONE as f32,
            BETA_COEF_Q15[lm] as f32 / Q15_ONE as f32,
        )
    };
    let mut prev = [0.0_f32; MAX_CHANNELS];
    for band in start..end {
        for (c, prev_c) in prev.iter_mut().enumerate().take(channels) {
            let tell = enc.tell();
            let bits_left = budget.saturating_sub(tell);
            let old = state.energy[c][band].max(ENERGY_FLOOR);
            // The decoder reconstructs E = coef*old + prev + qi, so the
            // encoder picks qi to make that land nearest the target.
            let prediction = coef * old + *prev_c;
            let qi_ideal = (target[c][band] - prediction).round() as i32;
            let qi: i32 = if bits_left >= LAPLACE_MIN_BUDGET_BITS {
                let pd = E_PROB_MODEL[lm][pred][band.min(NUM_BANDS - 1)];
                ec_laplace_encode(enc, qi_ideal, (pd.prob as u32) << 7, (pd.decay as u32) << 6)?
            } else if bits_left >= 2 {
                // 2-bit zig-zag fallback: qi ∈ {0, -1, +1} ⇒ s ∈ {0,1,2}
                // via the inverse of (s >> 1) ^ -(s & 1).
                let clamped = qi_ideal.clamp(-1, 1);
                let s = match clamped {
                    0 => 0usize,
                    -1 => 1,
                    _ => 2,
                };
                enc.enc_icdf(s, &SMALL_ENERGY_ICDF, 2)?;
                clamped
            } else if bits_left >= 1 {
                // 1-bit fallback: qi ∈ {0, -1}, bit = -qi.
                let clamped = qi_ideal.clamp(-1, 0);
                enc.enc_bit_logp((-clamped) as u32, 1)?;
                clamped
            } else {
                // No bits left: the implicit prediction error.
                -1
            };
            let q = qi as f32;
            state.energy[c][band] = coef * old + *prev_c + q;
            *prev_c += q - beta * q;
        }
    }
    Ok(())
}

/// One pass of the §A.1 listing's coarse-energy encode
/// (`quant_coarse_energy_impl`): the fixed-resolution walk of
/// [`encode_coarse_energy`] extended with the listing's
/// rate-distortion guards —
///
/// * the **decay bound**: a band may not drop more than `max_decay`
///   below its previous quantized energy (floored at −28), so one
///   low-energy frame cannot wreck the inter prediction;
/// * the **end-of-budget clamps**: once fewer than 30 bits remain
///   after reserving 3 bits for every not-yet-coded slot, `qi` is
///   clamped to `<= 1` (under 24) and `>= -1` (under 16), keeping the
///   tail of the envelope cheap instead of starving the shape bits;
/// * the intra flag itself is written here (Table 56 position) when
///   `tell + 3 <= budget`.
///
/// Returns the pass's **badness** — the summed distance between the
/// ideal and actually-coded `qi` — the distortion measure the
/// two-pass selection in [`quant_coarse_energy_rd`] compares.
#[allow(clippy::too_many_arguments)]
fn quant_coarse_energy_impl(
    enc: &mut RangeEncoder,
    state: &mut CoarseEnergyState,
    target: &[[f32; NUM_BANDS]; MAX_CHANNELS],
    intra: bool,
    lm: usize,
    start: usize,
    end: usize,
    channels: usize,
    budget: u32,
    tell0: u32,
    max_decay: f32,
) -> Result<i32, Error> {
    let mut badness = 0i32;
    let pred = if intra { PRED_INTRA } else { PRED_INTER };
    let (coef, beta) = if intra {
        (0.0_f32, INTRA_BETA_Q15 as f32 / Q15_ONE as f32)
    } else {
        (
            PRED_COEF_Q15[lm] as f32 / Q15_ONE as f32,
            BETA_COEF_Q15[lm] as f32 / Q15_ONE as f32,
        )
    };
    if tell0 + 3 <= budget {
        enc.enc_bit_logp(u32::from(intra), 3)?;
    }
    let mut prev = [0.0_f32; MAX_CHANNELS];
    for band in start..end {
        for (c, prev_c) in prev.iter_mut().enumerate().take(channels) {
            let x = target[c][band];
            let old = state.energy[c][band].max(ENERGY_FLOOR);
            let f = x - coef * old - *prev_c;
            let mut qi = (0.5 + f).floor() as i32;
            let decay_bound = state.energy[c][band].max(-28.0) - max_decay;
            // Prevent the energy from dropping faster than the decay
            // bound (e.g. one-bin bands on a silence transition).
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x) as i32;
                if qi > 0 {
                    qi = 0;
                }
            }
            let qi0 = qi;
            // If the budget cannot cover the rest of the envelope,
            // clamp toward the implicit cheap symbols.
            let tell = enc.tell();
            let bits_left = budget as i32 - tell as i32 - 3 * (channels * (end - band)) as i32;
            if band != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
            }
            let bits_avail = budget.saturating_sub(tell);
            let qi = if bits_avail >= LAPLACE_MIN_BUDGET_BITS {
                let pd = E_PROB_MODEL[lm][pred][band.min(NUM_BANDS - 1)];
                ec_laplace_encode(enc, qi, (pd.prob as u32) << 7, (pd.decay as u32) << 6)?
            } else if bits_avail >= 2 {
                let clamped = qi.clamp(-1, 1);
                let s = match clamped {
                    0 => 0usize,
                    -1 => 1,
                    _ => 2,
                };
                enc.enc_icdf(s, &SMALL_ENERGY_ICDF, 2)?;
                clamped
            } else if bits_avail >= 1 {
                let clamped = qi.clamp(-1, 0);
                enc.enc_bit_logp((-clamped) as u32, 1)?;
                clamped
            } else {
                -1
            };
            badness += (qi0 - qi).abs();
            let q = qi as f32;
            state.energy[c][band] = coef * old + *prev_c + q;
            *prev_c += q - beta * q;
        }
    }
    Ok(badness)
}

/// The §A.1 listing's loss-robustness distortion measure
/// (`loss_distortion`): the squared distance between this frame's
/// energy targets and the previous frame's quantized envelope, summed
/// over the coded window and capped at 200 — the "how much would
/// inter prediction hurt on packet loss" statistic that drives the
/// delayed-intra state.
fn loss_distortion(
    target: &[[f32; NUM_BANDS]; MAX_CHANNELS],
    state: &CoarseEnergyState,
    start: usize,
    eff_end: usize,
    channels: usize,
) -> f32 {
    let mut dist = 0.0f32;
    for c in 0..channels {
        for i in start..eff_end {
            let d = target[c][i] - state.energy[c][i];
            dist += d * d;
        }
    }
    dist.min(200.0)
}

/// The two-pass rate-distortion coarse-energy encode of the §A.1
/// reference listing (`quant_coarse_energy`) — the encoder-side
/// §4.3.2.1 walk with the listing's exact mode selection:
///
/// 1. the **intra pass** runs first (on scratch copies of the range
///    coder and prediction state) whenever two-pass selection is on
///    or intra is forced;
/// 2. the **inter pass** then re-encodes from the saved starting
///    state, and the frame keeps whichever pass has the lower
///    *badness* (summed `|ideal - coded|` clamping distortion), ties
///    broken by the cheaper `tell_frac` (plus the loss-rate intra
///    bias); the losing pass's bytes are discarded by restoring the
///    saved encoder;
/// 3. `max_decay = min(16, nb_available_bytes/8)` bounds the per-band
///    energy drop, and the `delayed_intra` state accumulates the
///    [`loss_distortion`] statistic (decayed by `pred_coef^2` on
///    inter frames) so a non-two-pass caller can force intra after
///    heavy envelope motion.
///
/// Writes the Table-56 intra flag and the whole coarse envelope into
/// `enc`; `state` holds the quantized energies both passes and the
/// decoder agree on. Returns the chosen intra mode.
#[allow(clippy::too_many_arguments)]
pub fn quant_coarse_energy_rd(
    enc: &mut RangeEncoder,
    state: &mut CoarseEnergyState,
    target: &[[f32; NUM_BANDS]; MAX_CHANNELS],
    lm: u32,
    start: usize,
    end: usize,
    eff_end: usize,
    channels: usize,
    budget: u32,
    nb_available_bytes: i32,
    force_intra: bool,
    delayed_intra: &mut f32,
    mut two_pass: bool,
    loss_rate: u32,
) -> Result<bool, Error> {
    if lm as usize >= NUM_LM_FRAME_SIZES
        || start > end
        || end > NUM_BANDS
        || eff_end > end
        || channels == 0
        || channels > MAX_CHANNELS
    {
        return Err(Error::InvalidParameter);
    }
    let lm = lm as usize;
    let mut intra = force_intra
        || (!two_pass
            && *delayed_intra > (2 * channels * (end - start)) as f32
            && nb_available_bytes > (channels * (end - start)) as i32);
    let intra_bias =
        ((budget as f32 * *delayed_intra * loss_rate as f32) / (channels as f32 * 512.0)) as i64;
    let new_distortion = loss_distortion(target, state, start, eff_end, channels);
    let tell0 = enc.tell();
    if tell0 + 3 > budget {
        two_pass = false;
        intra = false;
    }
    let max_decay = (16.0f32).min(0.125 * nb_available_bytes as f32);

    let enc_start = enc.clone();
    let mut state_intra = *state;
    let mut badness1 = 0i32;
    if two_pass || intra {
        badness1 = quant_coarse_energy_impl(
            enc,
            &mut state_intra,
            target,
            true,
            lm,
            start,
            end,
            channels,
            budget,
            tell0,
            max_decay,
        )?;
    }
    if !intra {
        let enc_intra = enc.clone();
        let tell_intra = enc.tell_frac();
        *enc = enc_start;
        let badness2 = quant_coarse_energy_impl(
            enc, state, target, false, lm, start, end, channels, budget, tell0, max_decay,
        )?;
        if two_pass
            && (badness1 < badness2
                || (badness1 == badness2
                    && enc.tell_frac() as i64 + intra_bias > tell_intra as i64))
        {
            *enc = enc_intra;
            *state = state_intra;
            intra = true;
        }
    } else {
        *state = state_intra;
    }
    if intra {
        *delayed_intra = new_distortion;
    } else {
        let pc = PRED_COEF_Q15[lm] as f32 / Q15_ONE as f32;
        *delayed_intra = pc * pc * *delayed_intra + new_distortion;
    }
    Ok(intra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_encoder::RangeEncoder;

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

    /// The two-pass RD encode stays in exact decoder lockstep: for a
    /// spread of targets, states, and budgets, the frame decodes to
    /// the identical intra flag and the identical quantized energies
    /// (both channels), through the plain §4.3.2.1 decode walk.
    #[test]
    fn quant_coarse_energy_rd_decoder_lockstep() {
        for case in 0..8u32 {
            let lm = case % 4;
            let channels = 1 + (case as usize / 4);
            let budget_bytes = [12u32, 96, 40, 200, 24, 160, 64, 250][case as usize];
            let budget = budget_bytes * 8;
            let mut state = CoarseEnergyState::new();
            let mut target = [[0.0f32; NUM_BANDS]; MAX_CHANNELS];
            for c in 0..channels {
                for b in 0..NUM_BANDS {
                    state.energy[c][b] = ((b * 7 + c * 3 + case as usize) % 11) as f32 - 5.0;
                    target[c][b] = ((b * 5 + c * 2 + 2 * case as usize) % 13) as f32 * 0.7 - 4.0;
                }
            }
            let mut delayed = 1.5f32;
            let mut enc = RangeEncoder::new();
            let mut enc_state = state;
            let intra = quant_coarse_energy_rd(
                &mut enc,
                &mut enc_state,
                &target,
                lm,
                0,
                NUM_BANDS,
                NUM_BANDS,
                channels,
                budget,
                budget_bytes as i32,
                false,
                &mut delayed,
                true,
                0,
            )
            .unwrap();
            let bytes = enc.finish_to_size(budget_bytes as usize).unwrap();
            let mut dec = crate::range_decoder::RangeDecoder::new(&bytes);
            let dec_intra = dec.dec_bit_logp(3) == 1;
            assert_eq!(dec_intra, intra, "case {case}: intra flag mismatch");
            let mut dec_state = state;
            decode_coarse_energy(&mut dec, &mut dec_state, intra, lm, 0, NUM_BANDS, channels)
                .unwrap();
            for c in 0..channels {
                for b in 0..NUM_BANDS {
                    assert!(
                        (dec_state.energy[c][b] - enc_state.energy[c][b]).abs() < 1e-4,
                        "case {case} c={c} b={b}: {} vs {}",
                        dec_state.energy[c][b],
                        enc_state.energy[c][b]
                    );
                }
            }
        }
    }

    /// The decay bound: with a small byte budget, a hard drop to
    /// silence may not pull any band more than
    /// `max_decay = min(16, bytes/8)` below its previous quantized
    /// energy.
    #[test]
    fn quant_coarse_energy_rd_decay_bound() {
        let bytes = 40u32; // max_decay = 5.0
        let mut state = CoarseEnergyState::new();
        for b in 0..NUM_BANDS {
            state.energy[0][b] = 6.0;
        }
        let start_energy = state.energy;
        let target = [[-28.0f32; NUM_BANDS]; MAX_CHANNELS];
        let mut delayed = 0.0f32;
        let mut enc = RangeEncoder::new();
        quant_coarse_energy_rd(
            &mut enc,
            &mut state,
            &target,
            2,
            0,
            NUM_BANDS,
            NUM_BANDS,
            1,
            bytes * 8,
            bytes as i32,
            false,
            &mut delayed,
            true,
            0,
        )
        .unwrap();
        for b in 0..NUM_BANDS {
            assert!(
                state.energy[0][b] >= start_energy[0][b] - 5.0 - 1.0,
                "band {b} dropped past the decay bound: {} -> {}",
                start_energy[0][b],
                state.energy[0][b]
            );
        }
    }

    /// Badness beats rate in the two-pass selection: when the inter
    /// pass must clamp hard (large prediction errors against a stale
    /// state) and the intra pass codes the envelope faithfully, the
    /// intra pass wins even though its flag costs ~3 bits.
    #[test]
    fn quant_coarse_energy_rd_prefers_lower_badness() {
        let bytes = 21u32; // tight: end-of-budget clamps bite inter
        let mut state = CoarseEnergyState::new();
        for b in 0..NUM_BANDS {
            state.energy[0][b] = if b % 2 == 0 { 8.0 } else { -8.0 };
        }
        let mut target = [[0.0f32; NUM_BANDS]; MAX_CHANNELS];
        for (b, t) in target[0].iter_mut().enumerate() {
            *t = if b % 2 == 0 { -6.0 } else { 6.0 };
        }
        let mut delayed = 0.0f32;
        let mut enc = RangeEncoder::new();
        let intra = quant_coarse_energy_rd(
            &mut enc,
            &mut state,
            &target,
            0,
            0,
            NUM_BANDS,
            NUM_BANDS,
            1,
            bytes * 8,
            bytes as i32,
            false,
            &mut delayed,
            true,
            0,
        )
        .unwrap();
        assert!(intra, "high-motion envelope should pick the intra pass");
        // And the delayed-intra statistic reflects the distortion cap.
        assert!(delayed > 0.0 && delayed <= 200.0);
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

    /// The coarse-energy encode is the exact inverse of the decode:
    /// encoding a set of target energies then decoding the produced
    /// frame reconstructs the identical per-band `state.energy`, for
    /// mono and stereo, intra and inter, over every frame size. The
    /// budget is large enough that every slot takes the full Laplace
    /// path.
    #[test]
    fn encode_decode_roundtrip_full_laplace() {
        // A representative target envelope (base-2 log energies).
        let target: [[f32; NUM_BANDS]; MAX_CHANNELS] = [
            std::array::from_fn(|b| 4.0 - 0.12 * b as f32),
            std::array::from_fn(|b| 3.5 - 0.10 * b as f32),
        ];
        for intra in [false, true] {
            for lm in 0..NUM_LM_FRAME_SIZES as u32 {
                for channels in 1..=2usize {
                    // Ample budget: a 128-byte frame (1024 bits) keeps
                    // every slot on the full Laplace path.
                    let budget = 1024u32;
                    let mut enc_state = CoarseEnergyState::new();
                    let mut enc = RangeEncoder::new();
                    encode_coarse_energy(
                        &mut enc,
                        &mut enc_state,
                        &target,
                        intra,
                        lm,
                        0,
                        NUM_BANDS,
                        channels,
                        budget,
                    )
                    .unwrap();
                    let out = enc.finish();
                    // The decoder needs storage_bits() == budget; pad the
                    // frame to exactly 128 bytes so the fallback dispatch
                    // matches the encoder's.
                    let mut framed = out;
                    framed.resize((budget / 8) as usize, 0);
                    let mut dec = RangeDecoder::new(&framed);
                    let mut dec_state = CoarseEnergyState::new();
                    decode_coarse_energy(
                        &mut dec,
                        &mut dec_state,
                        intra,
                        lm,
                        0,
                        NUM_BANDS,
                        channels,
                    )
                    .unwrap();
                    assert!(!dec.has_error(), "sticky error lm={lm} intra={intra}");
                    for c in 0..channels {
                        assert_eq!(
                            enc_state.energy[c], dec_state.energy[c],
                            "state mismatch c={c} lm={lm} intra={intra} ch={channels}"
                        );
                    }
                }
            }
        }
    }

    /// The reconstructed coarse energy is within a half 6 dB step of the
    /// target on the full-Laplace path (coarse quantization is
    /// round-to-nearest of the prediction error).
    #[test]
    fn encode_reconstruction_is_nearest_step() {
        let target: [[f32; NUM_BANDS]; MAX_CHANNELS] = [
            std::array::from_fn(|b| 2.0 + 0.37 * (b as f32).sin()),
            [0.0; NUM_BANDS],
        ];
        let mut enc_state = CoarseEnergyState::new();
        let mut enc = RangeEncoder::new();
        encode_coarse_energy(
            &mut enc,
            &mut enc_state,
            &target,
            true,
            2,
            0,
            NUM_BANDS,
            1,
            2048,
        )
        .unwrap();
        // Reconstruction error per band must not exceed one integer 6 dB
        // step (the coarse resolution); the prediction chain can spread
        // the residual, but each qi is the nearest-integer choice.
        for (b, (&got, &want)) in enc_state.energy[0].iter().zip(target[0].iter()).enumerate() {
            let err = (got - want).abs();
            assert!(
                err <= 1.0,
                "band {b} coarse error {err} exceeds a full step"
            );
        }
    }

    /// Encoding then decoding under a tiny budget exercises the low-rate
    /// fallbacks (2-bit / 1-bit / no-bit) and still round-trips the
    /// reconstructed state exactly.
    #[test]
    fn encode_decode_roundtrip_low_budget_fallbacks() {
        let target: [[f32; NUM_BANDS]; MAX_CHANNELS] = [
            std::array::from_fn(|b| 1.0 - 0.2 * b as f32),
            [0.0; NUM_BANDS],
        ];
        // A 4-byte frame (32 bits): after a few Laplace symbols the
        // budget drops through the 2-bit, 1-bit and no-bit fallbacks.
        let budget = 32u32;
        let mut enc_state = CoarseEnergyState::new();
        let mut enc = RangeEncoder::new();
        encode_coarse_energy(
            &mut enc,
            &mut enc_state,
            &target,
            false,
            1,
            0,
            NUM_BANDS,
            1,
            budget,
        )
        .unwrap();
        let mut framed = enc.finish();
        framed.resize((budget / 8) as usize, 0);
        let mut dec = RangeDecoder::new(&framed);
        let mut dec_state = CoarseEnergyState::new();
        decode_coarse_energy(&mut dec, &mut dec_state, false, 1, 0, NUM_BANDS, 1).unwrap();
        assert_eq!(enc_state.energy[0], dec_state.energy[0]);
    }

    /// Out-of-range parameters are rejected by the encoder without
    /// touching the encoder or the state.
    #[test]
    fn encode_invalid_parameters_rejected() {
        let target = [[0.0f32; NUM_BANDS]; MAX_CHANNELS];
        let mut state = CoarseEnergyState::new();
        let mut enc = RangeEncoder::new();
        let tell_before = enc.tell();
        assert_eq!(
            encode_coarse_energy(&mut enc, &mut state, &target, true, 4, 0, NUM_BANDS, 1, 1024),
            Err(Error::InvalidParameter)
        );
        assert_eq!(
            encode_coarse_energy(&mut enc, &mut state, &target, true, 0, 5, 4, 1, 1024),
            Err(Error::InvalidParameter)
        );
        assert_eq!(
            encode_coarse_energy(&mut enc, &mut state, &target, true, 0, 0, NUM_BANDS, 3, 1024),
            Err(Error::InvalidParameter)
        );
        assert_eq!(enc.tell(), tell_before);
        assert_eq!(state.energy, [[0.0; NUM_BANDS]; MAX_CHANNELS]);
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
