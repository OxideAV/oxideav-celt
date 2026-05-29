//! Coarse-energy decoding (RFC 6716 §4.3.2.1) — scaffold + DOCS GAP.
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
//! error is then entropy-coded with a Laplace distribution whose
//! parameters depend on the frame size and on the intra-vs-inter mode.
//!
//! Per RFC 6716 §4.3 (Table 55), the standard CELT mode operates on
//! **21 bands** (band index 0..=20). Hybrid mode reuses the same
//! 21-band layout but the first 17 bands (covering 0..8 kHz) are
//! coded by the SILK layer, leaving bands 17..=20 for CELT.
//!
//! ## DOCS GAP: e_prob_model + ec_laplace_decode (filed 2026-05-21)
//!
//! RFC 6716 §4.3.2.1 normatively describes the coarse-energy
//! mechanism but DELEGATES the actual numeric parameters and the
//! Laplace-decoder algorithm to source files that sit outside the
//! workspace's clean-room allow-list. The §4.3.2.1 prose names the
//! delegation target by file name; we redact those names here to keep
//! the comment free of forbidden-source references. The unredacted
//! prose lives at lines 6073–6077 of
//! `docs/audio/opus/rfc6716-opus.txt`.
//!
//! The RFC's prose gives us:
//!
//! * The 2-D z-transform of the prediction filter (eq. between lines
//!   6055–6059).
//! * The intra-mode prediction coefficients: `alpha=0,
//!   beta=4915/32768` (RFC line 6063). These are the ONLY numeric
//!   prediction coefficients the RFC supplies directly; the inter-mode
//!   `(alpha, beta)` pairs vary with the frame size and live in
//!   `e_prob_model[][]` (off-limits).
//! * The coarse step's 6 dB (1.0 in base-2 log) integer resolution
//!   (RFC line 6036).
//!
//! The RFC gives us NEITHER:
//!
//! * The `e_prob_model` numeric table (per-band Laplace `expect` and
//!   `decay` parameters × intra/inter × 4 frame sizes).
//! * The `ec_laplace_decode` algorithm (range-coded budget-aware
//!   Laplace integer decode with a fall-through to a uniform decoder
//!   when the residual is large — only the RFC's narrative shape, not
//!   the numeric steps).
//!
//! Without either piece, this round CANNOT land a bit-exact
//! coarse-energy decoder. What we land instead, within the wall, is:
//!
//! 1. The 21-band layout (`NUM_BANDS`) per RFC Table 55.
//! 2. The intra-mode prediction coefficients in Q15 fixed-point
//!    (`INTRA_ALPHA_Q15`, `INTRA_BETA_Q15`), the only numeric
//!    parameters §4.3.2.1 supplies directly.
//! 3. The 2-D prediction filter applied as a stand-alone helper
//!    ([`apply_intra_prediction`]) over an externally-supplied
//!    prediction-error vector — i.e. the post-Laplace-decode path,
//!    parameterised on Laplace output that arrives from a future
//!    `e_prob_model`-fed round.
//! 4. The `CoarseEnergyState` carrier struct (the running
//!    per-band previous-frame log-energies and per-band
//!    within-frame running prediction) so that future rounds can drop
//!    the Laplace decoder + table in without re-shaping data flow.
//! 5. A public entry point [`decode_coarse_energy`] that, until the
//!    docs gap closes, returns [`Error::NotImplemented`] with a clear
//!    pointer to this module's docstring.
//!
//! Closing the gap requires the docs collaborator to commission a
//! clean-room derivation of:
//!
//! * `e_prob_model[2][4][42]` (two modes × four LM values × 21 bands
//!   × 2 parameters): the intra and inter Laplace parameters for
//!   every band/frame-size combination. The shape and numeric values
//!   are not derivable from spec PDFs alone; they need to be either
//!   transcribed from an allowed spec source (none known to exist)
//!   or commissioned as a clean-room behavioural-trace document
//!   anchored to a black-box `opusdec` validator.
//! * The `ec_laplace_decode` algorithm prose (budget-aware decode of
//!   a Laplace-coded signed integer using `dec_uint` / `dec_icdf`
//!   primitives, with the fall-back uniform-decode for residual
//!   magnitudes beyond what the Laplace prefix can encode).
//!
//! Once that lands, [`decode_coarse_energy`] can be wired up to the
//! [`apply_intra_prediction`] helper here without restructuring the
//! state interface.
//!
//! ## Clean-room provenance
//!
//! Every numeric value, every formula, and every field comment in
//! this file is transcribed from RFC 6716 (`docs/audio/opus/`). The
//! source files that the RFC delegates to for the `e_prob_model`
//! table and the Laplace decoder algorithm sit outside the
//! workspace's clean-room allow-list and were not consulted.

use crate::range_decoder::RangeDecoder;
use crate::Error;

/// Number of CELT bands per RFC 6716 Table 55 (band index 0..=20).
///
/// Hybrid-mode CELT reuses the same band layout but only codes bands
/// 17..=20 (the SILK layer covers bands 0..=16 below 8 kHz). Pure
/// CELT codes all 21 bands.
pub const NUM_BANDS: usize = 21;

/// Intra-mode prediction coefficient α in Q15 fixed-point per
/// RFC 6716 §4.3.2.1 (line 6063 of `docs/audio/opus/rfc6716-opus.txt`).
///
/// The RFC states "alpha=0 ... when using intra energy". In Q15 that
/// is the integer 0; the constant exists for symmetry with the
/// inter-mode lookup that a future round will introduce once the
/// `e_prob_model` docs gap closes.
pub const INTRA_ALPHA_Q15: i32 = 0;

/// Intra-mode prediction coefficient β in Q15 fixed-point per
/// RFC 6716 §4.3.2.1.
///
/// The RFC writes the value as the fraction `4915/32768`. Since 32768
/// is `2^15`, the Q15 numerator is the literal 4915.
pub const INTRA_BETA_Q15: i32 = 4915;

/// Q15 scale used by the prediction filter coefficients above.
const Q15_ONE: i32 = 1 << 15;

/// Per-band running coarse-energy state across CELT frames.
///
/// `prev_q8` stores the previous frame's final fine-quantised log-2
/// energy for each band, in Q8 fixed-point relative to the reference
/// quietest representable energy. The state is only consulted in
/// inter-frame mode; intra frames reset the prediction's temporal arm
/// to zero (RFC 6716 §4.3.2.1: "an 'intra' frame where the energy is
/// coded without reference to prior frames").
///
/// Decoder lifecycle:
///
/// * `CoarseEnergyState::new()` on stream open (all-zero history).
/// * After each frame's fine-quantisation step has been applied
///   downstream, the caller writes the final per-band log-energies
///   into `prev_q8` for the next frame's inter-mode prediction.
/// * A decoder reset (packet loss recovery, mode switch) clears
///   `prev_q8` to zero, matching the encoder's expected state.
#[derive(Debug, Clone, Copy)]
pub struct CoarseEnergyState {
    /// Previous frame's final fine-quantised log-2 energy, Q8, per
    /// band. Zero on stream open and after any decoder reset.
    pub prev_q8: [i32; NUM_BANDS],
}

impl CoarseEnergyState {
    /// Construct a freshly-reset coarse-energy state. All bands start
    /// at zero log-energy.
    pub fn new() -> Self {
        Self {
            prev_q8: [0; NUM_BANDS],
        }
    }
}

impl Default for CoarseEnergyState {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply the §4.3.2.1 prediction filter to a vector of prediction
/// errors, producing per-band coarse log-2 energies in Q8.
///
/// The §4.3.2.1 filter's z-transform is
///
/// ```text
///                (1 - alpha*z_l^-1) * (1 - z_b^-1)
/// A(z_l, z_b) =  --------------------------------
///                         1 - beta*z_b^-1
/// ```
///
/// where `l` is the frame index (the time arm) and `b` is the band
/// index (the frequency arm). Expanding for our scalar form, the
/// per-band recursion that **produces** the coarse log-energy `E[b]`
/// from the **prediction error** `e[b]` is:
///
/// ```text
/// E_pred[b] = alpha * prev_q8[b]                  (time arm)
///           + sum over b' < b of e[b'] * beta^(b - b' - 1)
///                                                 (freq arm)
/// E[b] = E_pred[b] + e[b]
/// ```
///
/// In intra mode `alpha = 0` so the time arm vanishes entirely. The
/// frequency arm reduces to a single-pole IIR with coefficient β:
///
/// ```text
/// running = 0
/// for b in 0..NUM_BANDS:
///     E[b] = running + e[b]
///     running = (running + e[b]) * INTRA_BETA_Q15 / Q15_ONE
///             = (E[b]) * INTRA_BETA_Q15 / Q15_ONE
/// ```
///
/// The function takes the prediction errors in Q8 (matching the
/// state's `prev_q8` scale) and writes the reconstructed Q8
/// log-energies back into the supplied output slice.
///
/// This is the post-Laplace-decode arithmetic. It does NOT touch the
/// range decoder. It is exposed so that future rounds can compose it
/// with a Laplace decoder once the e_prob_model gap is closed; the
/// existing tests pin its behaviour against the RFC's stated formula
/// without needing the gap'd table.
pub fn apply_intra_prediction(errors_q8: &[i32; NUM_BANDS], out_q8: &mut [i32; NUM_BANDS]) {
    // Intra mode: alpha = 0, so the time arm contributes nothing. The
    // frequency arm is a single-pole IIR over the previous band's
    // reconstructed log-energy, scaled by β.
    let mut running: i32 = 0;
    for b in 0..NUM_BANDS {
        // E[b] = running + e[b]; the running prediction is the
        // β-filtered cumulative sum of prior bands' reconstructions.
        let e = errors_q8[b];
        let recon = running.saturating_add(e);
        out_q8[b] = recon;
        // Update running prediction for band b+1. The β multiply is
        // done in Q15 with a rounding right-shift to stay in Q8.
        // Numerator: recon * INTRA_BETA_Q15 (in Q8 * Q15 = Q23).
        // Divisor:   Q15_ONE.
        running = ((recon as i64 * INTRA_BETA_Q15 as i64) / Q15_ONE as i64) as i32;
    }
}

/// Decode coarse energies for one CELT frame (RFC 6716 §4.3.2.1).
///
/// **NOT IMPLEMENTED.** Returns [`Error::NotImplemented`] until the
/// `e_prob_model` table and the `ec_laplace_decode` algorithm are
/// available as clean-room specifications. See the module docstring
/// for the full docs-gap statement and the closure requirements.
///
/// The signature is locked in so that this round's scaffolding +
/// future rounds' Laplace + table work compose without breaking
/// callers. The intended contract is:
///
/// * `state` carries the previous frame's final fine-quantised
///   log-energies (Q8) and is mutated to record this frame's
///   coarse-quantised log-energies for next-frame inter prediction.
///   The fine quantisation step in §4.3.2.2 may further refine the
///   stored values before they propagate.
/// * `intra` matches the `intra` flag decoded by
///   [`crate::frame_header::CeltFrameHeader::decode_prefix`].
/// * `_lm` is `log2(frame_size / 120)` per RFC 6716 §4.3.3, in the
///   range 0..=3 covering 2.5 / 5 / 10 / 20 ms CELT frames.
/// * The returned `Vec<i16>` is the per-band coarse log-energy in
///   6 dB units (the §4.3.2.1 "integer part of base-2 log"), one
///   entry per band that this frame codes (21 entries for pure CELT,
///   4 entries for hybrid mode covering only bands 17..=20).
pub fn decode_coarse_energy(
    _dec: &mut RangeDecoder<'_>,
    _state: &mut CoarseEnergyState,
    _intra: bool,
    _lm: u32,
) -> Result<Vec<i16>, Error> {
    // The Laplace decoder + e_prob_model probability table are
    // off-limits per the workspace clean-room policy (the RFC names
    // source files outside the workspace clean-room allow-list as
    // their normative location). See the
    // module docstring's DOCS GAP section for the closure
    // requirements.
    Err(Error::NotImplemented)
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
    }

    /// The intra prediction coefficients are stated in RFC 6716
    /// §4.3.2.1 (line 6063): `alpha=0, beta=4915/32768`. Verify both
    /// the integer values and the Q15 interpretation.
    #[test]
    fn intra_prediction_coefficients_match_rfc() {
        assert_eq!(INTRA_ALPHA_Q15, 0);
        assert_eq!(INTRA_BETA_Q15, 4915);
        // 4915 / 32768 ≈ 0.1500244 — the RFC's chosen β as a fraction
        // of Q15_ONE.
        assert_eq!(Q15_ONE, 32_768);
        // Spot-check the fixed-point reconstruction matches the
        // decimal value to within Q15 precision. 4915 / 32768 ≈
        // 0.14999389..., so 4915 * 100_000 / 32_768 = 14_999 under
        // integer-truncating division — within one part in 10^5 of
        // the spec's stated value 0.15.
        let beta_per_100k = INTRA_BETA_Q15 * 100_000 / Q15_ONE;
        assert_eq!(beta_per_100k, 14_999);
    }

    /// A freshly-constructed state has all 21 bands at zero log-energy.
    /// This is the required state both on stream open and after any
    /// decoder reset (RFC 6716 §4.3.2.1 implicitly assumes the
    /// previous-frame state vanishes on intra-coded frames; we also
    /// snap to zero on reset to match the encoder's expected state).
    #[test]
    fn new_state_is_all_zero() {
        let state = CoarseEnergyState::new();
        assert_eq!(state.prev_q8, [0i32; NUM_BANDS]);
        // Default trait matches `new`.
        assert_eq!(CoarseEnergyState::default().prev_q8, state.prev_q8);
    }

    /// With every prediction error zero, the §4.3.2.1 intra filter
    /// must reconstruct every band at zero — the trivial silence case.
    #[test]
    fn intra_prediction_zero_errors_yields_zero_output() {
        let errors = [0i32; NUM_BANDS];
        let mut out = [0i32; NUM_BANDS];
        apply_intra_prediction(&errors, &mut out);
        assert_eq!(out, [0i32; NUM_BANDS]);
    }

    /// With a single non-zero error in band 0 and zeros elsewhere,
    /// the §4.3.2.1 filter reduces to:
    ///   E[0] = e[0]
    ///   E[b] for b > 0 = E[0] * β^b   (cascade of β multiplies)
    /// Verify the first three bands by hand.
    #[test]
    fn intra_prediction_single_impulse_decays_with_beta() {
        let mut errors = [0i32; NUM_BANDS];
        errors[0] = 32_768; // arbitrary Q8 magnitude
        let mut out = [0i32; NUM_BANDS];
        apply_intra_prediction(&errors, &mut out);

        // E[0] = 32768.
        assert_eq!(out[0], 32_768);
        // E[1] = E[0] * β / Q15_ONE = 32768 * 4915 / 32768 = 4915.
        assert_eq!(out[1], 4915);
        // E[2] = E[1] * β / Q15_ONE = 4915 * 4915 / 32768.
        let expected2 = (4915i64 * 4915 / 32768) as i32;
        assert_eq!(out[2], expected2);
        // β < 1 (4915 < 32768) so the cascade strictly contracts;
        // E[2] < E[1] < E[0]. (No equality even on the boundary.)
        assert!(out[2] < out[1]);
        assert!(out[1] < out[0]);
    }

    /// `apply_intra_prediction` is purely additive in the per-band
    /// errors when β = 0 would degenerate; with the actual β > 0, the
    /// reconstruction error at band b is bounded by the cumulative
    /// β^k * e[b-k] tail, which we verify by reconstructing a sample
    /// pair against a hand-computed expectation.
    #[test]
    fn intra_prediction_additive_pair() {
        // Two non-zero errors: e[0] = 256, e[1] = 128. Compute the
        // expected reconstruction by hand.
        //   E[0] = 256
        //   running after band 0 = 256 * 4915 / 32768 = 38 (integer
        //     truncation: 1257_984 / 32768 = 38)
        //   E[1] = 38 + 128 = 166
        //   running after band 1 = 166 * 4915 / 32768 = 24 (truncation:
        //     815_890 / 32768 = 24)
        //   E[2] = 24 + 0 = 24
        let mut errors = [0i32; NUM_BANDS];
        errors[0] = 256;
        errors[1] = 128;
        let mut out = [0i32; NUM_BANDS];
        apply_intra_prediction(&errors, &mut out);
        assert_eq!(out[0], 256);
        assert_eq!(out[1], 166);
        assert_eq!(out[2], 24);
    }

    /// Until the Laplace + e_prob_model docs gap closes,
    /// `decode_coarse_energy` MUST return `NotImplemented` rather
    /// than silently producing zeros (which would desync the range
    /// decoder for the band-allocator that runs after coarse energy).
    /// The signature is otherwise stable for future rounds.
    #[test]
    fn decode_coarse_energy_returns_docs_gap_marker() {
        let buf = [0u8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let mut state = CoarseEnergyState::new();
        let r = decode_coarse_energy(&mut dec, &mut state, true, 0);
        assert_eq!(r, Err(Error::NotImplemented));
        // The range decoder MUST NOT have been touched on the gap'd
        // path; `tell()` should match a fresh decoder over the same
        // buffer.
        let fresh = RangeDecoder::new(&buf);
        assert_eq!(dec.tell(), fresh.tell());
        // State must also be unchanged.
        assert_eq!(state.prev_q8, [0i32; NUM_BANDS]);
    }

    /// Sanity check on the prediction filter's stability: feed a
    /// constant-magnitude error vector and confirm the reconstructed
    /// output never explodes outside reasonable bounds. (β < 1
    /// guarantees this analytically; the test serves as a regression
    /// sentinel against accidental sign flips or shift errors.)
    #[test]
    fn intra_prediction_stable_under_constant_input() {
        let errors = [1024i32; NUM_BANDS];
        let mut out = [0i32; NUM_BANDS];
        apply_intra_prediction(&errors, &mut out);
        // For a constant input c, the steady state of the IIR is
        //   E_ss = c / (1 - β) where β = 4915/32768 ≈ 0.15
        // i.e. E_ss ≈ c * 32768 / (32768 - 4915) = c * 32768 / 27853
        //          ≈ c * 1.1764 ≈ 1205 for c = 1024.
        // The reconstruction at band 0 is exactly c (no prior arm);
        // it monotonically approaches the steady state from below.
        assert_eq!(out[0], 1024);
        for b in 1..NUM_BANDS {
            assert!(out[b] >= out[b - 1], "monotonicity broken at band {b}");
            assert!(
                out[b] < 1300,
                "steady-state bound violated at band {b}: {}",
                out[b]
            );
        }
    }
}
