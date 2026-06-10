//! # oxideav-celt
//!
//! Pure-Rust CELT layer of the Opus codec (RFC 6716).
//!
//! **Status (2026-06-09):** round-22. The bit-exact CELT/SILK range
//! decoder (RFC 6716 §4.1) is complete; the CELT frame-header prefix
//! (silence / post-filter / transient / intra per §4.3, plus the
//! deferred anti-collapse bit per §4.3.5) is wired up. The §4.3.2.1
//! coarse-energy scaffolding (21-band layout from Table 55 + intra
//! prediction filter with `α=0, β=4915/32768`) is in place. The
//! §4.3.2.1 `e_prob_model` Laplace-parameter table is now transcribed
//! verbatim (`E_PROB_MODEL[lm][intra][band] -> ProbDecay { prob,
//! decay }`, 4 × 2 × 21 = 168 Q8 pairs from
//! `docs/audio/celt/tables/e_prob_model.csv`) with the
//! `prob_decay(lm, intra, band)` accessor that folds the `bool intra`
//! flag onto the staged CSV's `0 = inter / 1 = intra` middle axis;
//! the `ec_laplace_decode` algorithm itself remains queued for a
//! future round (the RFC narrative does not state the per-symbol
//! decode recurrence). The §4.3.2.2
//! fine-energy refinement decoder + finalize step is bit-exact. The
//! §4.3.3 bit-allocation field decoders (alloc.trim, skip,
//! intensity-band, dual-stereo) are exposed standalone, gated on
//! caller-supplied reservation booleans, plus the §4.3.3 stereo
//! reservation helpers (`LOG2_FRAC_TABLE` lookup + `intensity_rsv` +
//! `reserve_stereo`) that compute the intensity + dual gates from
//! the running budget. The §4.3.3 per-band cap[] machinery
//! (`CACHE_CAPS50` table + `compute_band_caps`) and the §4.3.3
//! band-boost dynalloc-logp loop (`decode_band_boosts`) are now wired
//! up, closing the `cache_caps50` docs-gap blocker. The §4.3.3
//! initial-reservations budget walk
//! (`compute_initial_reservations` → `InitialReservations`) chains
//! the `total_initial` init + anti-collapse + skip + intensity +
//! dual-stereo reservations into one call and synthesises the
//! `BandAllocationGates` for the existing band-allocation decoder.
//! The §4.3.3 §2.6 per-band hard-minimum shape allocation
//! (`compute_thresh`) and the per-band `trim_offsets[]` derivation
//! (`compute_trim_offsets`) are now wired up, alongside the
//! §4.3 Table 55 MDCT-bin layout (`BAND_BINS_LM` for all four LM
//! values + `SHORT_FRAME_BAND_BINS` for the LM=0 column the §2.6 prose
//! cites).
//! The §4.3.4.5
//! time-frequency change parameters (per-band `tf_change` + the gated
//! global `tf_select` + the four TF-adjustment tables 60–63) plus the
//! §4.3.4.5 Hadamard transform primitives (orthonormal radix-2 WHT in
//! both natural and sequency order + the `apply_tf_resolution_change`
//! orchestrator) are wired up. The §4.3.4.3 spreading parameter (PDF
//! `{7, 2, 21, 2}/32`) + Table 59 `f_r` lookup + closed-form
//! rotation-gain helpers are wired up. The §4.3.7.1 post-filter tap
//! shapes (three §4.3.7.1 tapsets in f32 + Q15) plus gain
//! reconstruction and per-sample / slice filter response, and the
//! §4.3.7.2 single-pole de-emphasis filter (`α_p = 0.8500061035`),
//! are now wired up. The §4.3.3 Table 57 CELT Static Allocation
//! Table (`STATIC_ALLOC[21][11]`) is transcribed verbatim from the
//! RFC body, alongside the per-band evaluator
//! `band_static_alloc_1_8th` (the `channels * N * alloc << LM >> 2`
//! formula folded with the 1/64-step linear interpolation between
//! adjacent quality columns) and the `window_static_alloc_1_8th`
//! window sum the static-allocation search driver composes with. The
//! §4.3.3 inner static-allocation search (`find_static_alloc` →
//! `StaticAllocSearch { qlo, frac, total_1_8th }`) bisects the 1/64-
//! step interpolation grid for the highest `(qlo, frac)` whose window
//! total in 1/8 bits does not exceed the supplied "remaining" budget.
//! The §4.3.4.2 PVQ codebook size `V(N, K)` and per-band shape
//! decoder (`v_count`, `decode_index_to_pulses`, `decode_pulses`,
//! `normalize_to_unit_l2`, `decode_unit_shape`) reproduce the codebook
//! recurrence `V(N, K) = V(N-1, K) + V(N, K-1) + V(N-1, K-1)` and the
//! §4.3.4.2 per-position reconstruction loop; the decoded integer
//! pulse vector is normalised to unit L2 norm so the §4.3.4.3
//! spreading rotation can consume it directly. The §4.3.4.1
//! bits-to-pulses search + balance accumulator
//! (`bits_to_pulses_search`, `BalanceAccumulator`,
//! `bits_to_pulses_band_loop`) are now wired up: the per-band `K`
//! search picks the largest pulse count whose `V(N, K)` codebook
//! cost does not exceed the supplied 1/8-bit target ("nearest to
//! target, not exceeding it; ties round down"), and the running
//! balance carries the truncation residue forward with the §4.3.4.1
//! share divisors (`3` general, `2` second-to-last, `1` last). The
//! cost-of-(N, K) function is decoupled and supplied by the caller;
//! a closed-form `cost_log2_v_count_8th` estimator based on
//! `ceil(log2(V(N, K)))` ships as the default. The §4.3.4.3 spreading
//! rotation chain (`apply_spread`, `apply_nd_rotation`,
//! `apply_nd_rotation_multi_block`, `apply_pre_rotation`,
//! `apply_2d_rotation`, `rotation_angle_f64`) applies the
//! `theta = pi * g_r^2 / 4` N-D forward+reverse 2-D rotation chain
//! to a unit-norm PVQ shape vector, with per-time-block independence
//! and the `(pi/2 - theta)` interleaved pre-rotation at stride
//! `round(sqrt(N/nb_blocks))` when blocks span at least 8 samples.
//! The §4.3.4.4 PVQ band-split gating + recursion geometry
//! (`band_needs_split`, `split_dimensions`, `max_split_levels`,
//! `plan_band_split` → `BandSplitNode`) detects when `V(N, K)` would
//! exceed the 32-bit codebook budget and synthesises the recursive
//! halving tree (capped at `LM + 1` levels per the §4.3.4.4 prose)
//! the band-decode walker traverses to reach leaf-PVQ sub-bands.
//! The quantized split-gain parameter that redistributes the L2 norm
//! across the two halves is queued as a docs gap (the §4.3.4.4 prose
//! defers the precise precision/PDF to the reference). The §4.3.6
//! band denormalization (the per-sample multiplicative pass that
//! scales each PVQ-decoded unit-norm shape by `sqrt(2^(E_q8 / 256))`
//! before the inverse MDCT) is wired up as pure arithmetic against
//! caller-supplied Q8 log-energies, so it composes cleanly once the
//! `ec_laplace_decode` docs gap on the coarse-energy path closes. The
//! §4.3.4 → §4.3.6 single non-split band-decode orchestrator
//! (`decode_band_shape` → `BandShape`) chains the simplest-case decode
//! chain in §4.3 bitstream order: PVQ unit-shape decode
//! (§4.3.4.1/§4.3.4.2) → spreading rotation (§4.3.4.3) → time-frequency
//! resolution change (§4.3.4.5) → denormalization (§4.3.6). The
//! reallocation loop (concurrent skip decoding), the fine-energy /
//! shape split, the §4.3.4.4 split-gain band-split path, and the MDCT
//! machinery still come later.
//!
//! Every other public API path returns [`Error::NotImplemented`].
//!
//! ## Clean-room provenance
//!
//! All routines in this crate are transcribed from RFC 6716 (the IETF
//! standards-track definition of Opus) and RFC 8251 (the Opus update).
//! Source files the RFC delegates to for normative numeric tables and
//! algorithms are off-limits under the workspace clean-room policy
//! and were not consulted. Black-box invocations of `opusdec` /
//! `opusenc` are permitted under the workspace clean-room policy as
//! opaque validators.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod allocation_budget;
pub mod band_cap;
pub mod band_decode;
pub mod band_minimums;
pub mod band_split;
pub mod bit_allocation;
pub mod bits_to_pulses;
pub mod coarse_energy;
pub mod deemphasis;
pub mod denormalization;
pub mod e_prob_model;
pub mod fine_energy;
pub mod frame_header;
pub mod hadamard;
pub mod post_filter;
pub mod pvq;
pub mod range_decoder;
pub mod spread;
pub mod spread_rotation;
pub mod static_alloc;
pub mod tf_change;

pub use allocation_budget::{
    compute_initial_reservations, InitialReservations, RSV_BIT_8TH, RSV_INITIAL_SLACK_8TH,
};
pub use band_cap::{compute_band_caps, decode_band_boosts, BoostResult, CACHE_CAPS50};
pub use band_decode::{decode_band_shape, BandShape};
pub use band_minimums::{
    compute_thresh, compute_trim_offsets, BAND_BINS_LM, EIGHTH_BIT_QUANTUM, NUM_LM,
    SHORT_FRAME_BAND_BINS,
};
pub use band_split::{
    band_needs_split, max_split_levels, plan_band_split, split_dimensions, BandSplitNode, MAX_LM,
};
pub use bit_allocation::{
    decode_alloc_trim, decode_band_allocation, decode_dual_stereo, decode_intensity_band,
    decode_skip_flag, intensity_rsv, reserve_stereo, BandAllocation, BandAllocationGates,
    DEFAULT_ALLOC_TRIM, LOG2_FRAC_TABLE,
};
pub use bits_to_pulses::{
    bits_to_pulses_band_loop, bits_to_pulses_search, cost_log2_v_count_8th, BalanceAccumulator,
    BitsToPulses, DEFAULT_BALANCE_DIVISOR, EIGHTH_BITS_PER_BIT, K_SEARCH_CAP, LAST_BALANCE_DIVISOR,
    SECOND_TO_LAST_BALANCE_DIVISOR,
};
pub use coarse_energy::{
    apply_intra_prediction, decode_coarse_energy, CoarseEnergyState, INTRA_ALPHA_Q15,
    INTRA_BETA_Q15, NUM_BANDS,
};
pub use deemphasis::{deemphasize_in_place_f32, Deemphasis, ALPHA_P_F32, ALPHA_P_Q15};
pub use denormalization::{
    denormalize_band_f32, denormalize_band_in_place_f32, denormalize_bands_f32,
    denormalize_bands_in_place_f32, log_energy_q8_to_amplitude_f32, scale_band_f32,
    scale_band_in_place_f32, Q8_DENOM, SQRT_Q8_DENOM,
};
pub use e_prob_model::{
    prob_decay, ProbDecay, E_PROB_MODEL, NUM_LM_FRAME_SIZES, NUM_PREDICTION_TYPES, PRED_INTER,
    PRED_INTRA,
};
pub use fine_energy::{
    decode_fine_energy, decode_fine_energy_band, finalize_extra_bits, fine_correction_q14,
    fine_correction_qn, FinalizePriority, FinalizeResult, MAX_FINE_BITS,
};
pub use frame_header::{decode_anti_collapse_flag, CeltFrameHeader, PostFilter};
pub use hadamard::{
    apply_tf_resolution_change, walsh_hadamard_inplace, walsh_hadamard_sequency_inplace,
    HADAMARD_LEVEL_SCALE,
};
pub use post_filter::{
    apply_post_filter_f32, filter_sample_f32, gain_f32, gain_q15, tap_coefficients_f32,
    tap_coefficients_q15, NUM_TAPSETS, POST_FILTER_PERIOD_MAX, POST_FILTER_PERIOD_MIN,
    POST_FILTER_TAPS_F32, POST_FILTER_TAPS_Q15, TAPS_PER_SET,
};
pub use pvq::{
    decode_index_to_pulses, decode_pulses, decode_unit_shape, normalize_to_unit_l2, v_count,
    V_COUNT_SATURATION,
};
pub use range_decoder::RangeDecoder;
pub use spread::{
    decode_spread, pre_rotation_stride, rotation_gain_ratio, rotation_gain_squared_ratio, Spread,
    DEFAULT_SPREAD,
};
pub use spread_rotation::{
    apply_2d_rotation, apply_nd_rotation, apply_nd_rotation_multi_block, apply_pre_rotation,
    apply_spread, rotation_angle_f64, EXTRA_ROTATION_MIN_BLOCK_SAMPLES,
};
pub use static_alloc::{
    band_static_alloc_1_8th, find_static_alloc, interp_alloc_1_32nd, window_static_alloc_1_8th,
    StaticAllocSearch, INTERP_STEPS, NUM_Q, STATIC_ALLOC,
};
pub use tf_change::{
    decode_tf_changes, decode_tf_parameters, decode_tf_select, tf_adjustment, tf_select_matters,
    TfParameters, LM_VALUES, TABLE_60_NON_TRANSIENT_SEL0, TABLE_61_NON_TRANSIENT_SEL1,
    TABLE_62_TRANSIENT_SEL0, TABLE_63_TRANSIENT_SEL1, TF_CHANGE_VALUES,
};

/// Crate-local error type. The encoder, frame-level decoder, and
/// higher-level codec entry points are not yet wired up; calling them
/// returns [`Error::NotImplemented`]. The range decoder primitives in
/// [`range_decoder`] do not use this type for their hot paths — they
/// latch a sticky error flag instead, mirroring the behaviour
/// recommended by RFC 6716 §4.1.5 for corrupt frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A higher-level CELT entry point that has not been implemented
    /// yet (everything beyond the range decoder, today).
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => write!(
                f,
                "oxideav-celt: requested entry point is not yet implemented"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// No-op codec registration. The range decoder is a leaf primitive
/// and does not need to advertise a codec ID; once the band/MDCT
/// path lands, this hook will register the `celt` decoder with the
/// runtime.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("celt", register);
