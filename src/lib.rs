//! # oxideav-celt
//!
//! Pure-Rust CELT layer of the Opus codec (RFC 6716) — a
//! **reference-exact decoder**, a **reference-compatible encoder
//! that measures ahead of the reference listing**, and the
//! oxideav-core registry wiring, transcribed from RFC 6716
//! (including the Appendix A reference listing embedded in the
//! staged RFC text, extracted per §A.1 and SHA-1-verified) and
//! validated black-box against oracle binaries built from that
//! listing.
//!
//! The load-bearing modules:
//!
//! * [`ref_decode`] ([`ref_decode::CeltRefDecoder`]) — the complete
//!   Table-56 decode walk: every budget gate at its exact position,
//!   the exact §4.3.3 allocation ([`alloc_exact`]) and §4.3.4 band
//!   loop ([`band_quant`]), the absolute `eMeans` energy scale
//!   (under the RFC 8251 sec 8 cap on band energy), the exact
//!   §4.3.5 anti-collapse, the two-stage §4.3.7.1 comb filter, the
//!   §4.3.7.2 de-emphasis, and the §4.3 packet-loss concealment
//!   ([`ref_decode::CeltRefDecoder::decode_lost`]). Reference
//!   streams decode at the decoder pair\'s float-rounding floor.
//! * [`ref_encode`] ([`ref_encode::CeltRefEncoder`]) — the exact
//!   encode-direction mirror plus the §5.3 decision layer (TF
//!   Viterbi, two-pass coarse RD, trim/spread/dual analyses,
//!   Table-66 intensity, transient detection, the §5.3.1 pitch
//!   prefilter chain) and the §A.1 VBR/constrained-VBR controller.
//!   Every emitted stream decodes identically through this crate\'s
//!   decoder and the listing decoder; decoded quality measures
//!   above the listing encoder at every measured CBR point.
//! * [`custom_mode`] ([`custom_mode::CeltCustomMode`]) — the
//!   Appendix-A custom-mode construction for any legal
//!   `(rate, frame size)` geometry (8–96 kHz, 40–1024 even
//!   samples); fed `(48000, 960)` it reproduces every staged
//!   48 kHz table bit-exactly.
//! * [`codec`] — `register` plus the direct `make_decoder` /
//!   `make_encoder` factories (the workspace dual-API convention):
//!   one raw CELT frame per packet, interleaved-f32 frames, options
//!   `frame_size` / `start_band` / `end_band` / `resample` / `vbr`
//!   / `vbr_constrained`; empty packets conceal.
//!
//! Both directions cover the §3.1 CELT-mode bandwidths (end band
//! 13/17/19/21), the Hybrid-mode CELT layer (`start = 17`),
//! reduced-rate PCM I/O (8/12/16/24 kHz), and custom modes. The
//! remaining modules are the earlier §4.3-prose building blocks
//! (range coder, frame prefix, coarse/fine energy, PVQ, MDCT,
//! synthesis spines, the `pcm_encode` auto codec with its own
//! documented in-crate wire) — retained as documented, tested
//! primitives; see the README for the full status and measurement
//! record, and `fuzz/` for the coverage-guided harnesses.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod alloc_combine;
#[doc(hidden)]
pub mod alloc_exact;
#[doc(hidden)]
pub mod allocation_budget;
#[doc(hidden)]
pub mod analysis;
#[doc(hidden)]
pub mod anti_collapse;
#[doc(hidden)]
pub mod band_analysis;
#[doc(hidden)]
pub mod band_cap;
#[doc(hidden)]
pub mod band_decode;
#[doc(hidden)]
pub mod band_energy;
#[doc(hidden)]
pub mod band_layout;
#[doc(hidden)]
pub mod band_minimums;
#[doc(hidden)]
pub mod band_quant;
#[doc(hidden)]
pub mod band_split;
#[doc(hidden)]
pub mod bit_allocation;
#[doc(hidden)]
pub mod bits_to_pulses;
#[doc(hidden)]
pub mod coarse_energy;
pub mod codec;
pub mod custom_mode;
#[doc(hidden)]
pub mod deemphasis;
#[doc(hidden)]
pub mod denormalization;
#[doc(hidden)]
pub mod derive_pulses;
#[doc(hidden)]
pub mod e_prob_model;
#[doc(hidden)]
pub mod encoder_decisions;
#[doc(hidden)]
pub mod fine_energy;
#[doc(hidden)]
pub mod frame_decode;
#[doc(hidden)]
pub mod frame_encode;
#[doc(hidden)]
pub mod frame_header;
#[doc(hidden)]
pub mod frame_synthesis;
#[doc(hidden)]
pub mod hadamard;
#[doc(hidden)]
pub mod laplace;
#[doc(hidden)]
pub mod mdct;
#[doc(hidden)]
pub mod pcm_encode;
#[doc(hidden)]
pub mod pitch;
mod plc;
#[doc(hidden)]
pub mod post_filter;
#[doc(hidden)]
pub mod pulse_cache;
#[doc(hidden)]
pub mod pvq;
#[doc(hidden)]
pub mod range_decoder;
#[doc(hidden)]
pub mod range_encoder;
#[doc(hidden)]
pub mod realloc_walk;
pub mod ref_decode;
pub mod ref_encode;
#[doc(hidden)]
pub mod residual;
#[doc(hidden)]
pub mod spread;
#[doc(hidden)]
pub mod spread_rotation;
#[doc(hidden)]
pub mod static_alloc;
#[doc(hidden)]
pub mod synthesis;
#[doc(hidden)]
pub mod tf_change;

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub use alloc_combine::{
    combine_band_allocation, find_combined_alloc, CombinedAllocSearch, CombinedAllocation,
};
#[doc(hidden)]
pub use allocation_budget::{
    compute_initial_reservations, InitialReservations, RSV_BIT_8TH, RSV_INITIAL_SLACK_8TH,
};
#[doc(hidden)]
pub use analysis::{extract_coded_spectrum, LongMdctAnalysis, StereoPcmAnalysis};
#[doc(hidden)]
pub use anti_collapse::{apply_anti_collapse, ENERGY_HISTORY_FLOOR_LOG2};
#[doc(hidden)]
pub use band_analysis::{
    analyze_band_f32, analyze_bands_f32, band_energy_f32, band_log_energy_f32, BandAnalysis,
    SILENCE_LOG_ENERGY,
};
#[doc(hidden)]
pub use band_cap::{
    compute_band_caps, decode_band_boosts, encode_band_boosts, BoostResult, CACHE_CAPS50,
};
#[doc(hidden)]
pub use band_decode::{decode_band_shape, BandShape};
#[doc(hidden)]
pub use band_energy::{
    assemble_band_log_energy_f32, assemble_band_log_energy_q8, interop_wire_bias_f32,
    interop_wire_bias_q8, log_energy_f32_to_q8, render_band_energy_q8, E_MEANS_Q4, FINE_Q14_DENOM,
    Q14_TO_Q8_SHIFT, SPECTRUM_SCALE_LOG2_Q8,
};
#[doc(hidden)]
pub use band_layout::{
    band_bin_range, band_bins, band_edge, coded_total_bins, EBAND_EDGES_5MS, NUM_BAND_EDGES,
};
#[doc(hidden)]
pub use band_minimums::{
    compute_thresh, compute_trim_offsets, BAND_BINS_LM, EIGHTH_BIT_QUANTUM, NUM_LM,
    SHORT_FRAME_BAND_BINS,
};
#[doc(hidden)]
pub use band_split::{
    band_needs_split, max_split_levels, plan_band_split, split_dimensions, BandSplitNode, MAX_LM,
};
#[doc(hidden)]
pub use bit_allocation::{
    decode_alloc_trim, decode_band_allocation, decode_dual_stereo, decode_intensity_band,
    decode_skip_flag, encode_alloc_trim, encode_band_allocation, encode_dual_stereo,
    encode_intensity_band, encode_skip_flag, intensity_rsv, reserve_stereo, BandAllocation,
    BandAllocationGates, DEFAULT_ALLOC_TRIM, LOG2_FRAC_TABLE,
};
#[doc(hidden)]
pub use bits_to_pulses::{
    bits_to_pulses_band_loop, bits_to_pulses_band_loop_cached,
    bits_to_pulses_band_loop_cached_thresh, bits_to_pulses_search, cost_log2_v_count_8th,
    BalanceAccumulator, BitsToPulses, DEFAULT_BALANCE_DIVISOR, EIGHTH_BITS_PER_BIT, K_SEARCH_CAP,
    LAST_BALANCE_DIVISOR, SECOND_TO_LAST_BALANCE_DIVISOR,
};
#[doc(hidden)]
pub use coarse_energy::{
    decode_coarse_energy, encode_coarse_energy, quant_coarse_energy_rd, CoarseEnergyState,
    BETA_COEF_Q15, INTRA_ALPHA_Q15, INTRA_BETA_Q15, MAX_CHANNELS, NUM_BANDS, PRED_COEF_Q15,
    SMALL_ENERGY_ICDF,
};
pub use custom_mode::{CeltCustomMode, MAX_BANDS};
#[doc(hidden)]
pub use deemphasis::{
    deemphasize_in_place_f32, preemphasize_in_place_f32, Deemphasis, Preemphasis, ALPHA_P_F32,
    ALPHA_P_Q15,
};
#[doc(hidden)]
pub use denormalization::{
    denormalize_band_f32, denormalize_band_in_place_f32, denormalize_bands_f32,
    denormalize_bands_in_place_f32, log_energy_q8_to_amplitude_f32, scale_band_f32,
    scale_band_in_place_f32, MAX_LOG_ENERGY_Q8, Q8_DENOM, SQRT_Q8_DENOM,
};
#[doc(hidden)]
pub use derive_pulses::{
    decode_celt_frame_auto, derive_band_allocation, derive_band_allocation_dual,
    derive_band_pulses, derive_band_pulses_dual, pulses_from_walk, run_prefix_walk,
    DerivedAllocation,
};
#[doc(hidden)]
pub use e_prob_model::{
    prob_decay, ProbDecay, E_PROB_MODEL, NUM_LM_FRAME_SIZES, NUM_PREDICTION_TYPES, PRED_INTER,
    PRED_INTRA,
};
#[doc(hidden)]
pub use encoder_decisions::{
    boost_quanta_8th, boost_thresholds, choose_alloc_trim, choose_band_boosts, choose_intra_mode,
    choose_mid_side_stereo, intensity_start_band, low_band_stereo_correlation, mid_side_extra_dof,
    spectral_tilt_slope, MID_SIDE_DECISION_BANDS, TRIM_TILT_GAIN,
};
#[doc(hidden)]
pub use fine_energy::{
    apply_finalize_scale_f32, decode_fine_energy, decode_fine_energy_band,
    encode_finalize_extra_bits_depth, encode_fine_energy, encode_fine_energy_band,
    finalize_correction_q14, finalize_extra_bits, finalize_extra_bits_depth,
    finalize_priorities_from_k, fine_correction_q14, fine_correction_qn, quantize_fine_energy_band,
    quantize_fine_energy_f32, FinalizeDepthResult, FinalizePriority, FinalizeResult, FINE_Q14_ONE,
    MAX_FINE_BITS,
};
#[doc(hidden)]
pub use frame_decode::{decode_frame_prefix, FramePrefix};
#[doc(hidden)]
pub use frame_encode::{
    encode_celt_frame, encode_celt_frame_auto, encode_celt_frame_auto_boosted, encode_frame_prefix,
    encode_stereo_celt_frame, encode_stereo_celt_frame_auto, EncodedFrame, FramePrefixSpec,
    StereoEncodedFrame,
};
#[doc(hidden)]
pub use frame_header::{
    decode_anti_collapse_flag, encode_anti_collapse_flag, CeltFrameHeader, PostFilter,
};
#[doc(hidden)]
pub use frame_synthesis::{
    decode_celt_frame, CeltDecodeState, DecodedFrame, PostFilterParams, StereoCeltDecodeState,
    StereoDecodedFrame,
};
#[doc(hidden)]
pub use hadamard::{
    apply_tf_resolution_change, apply_tf_resolution_change_inverse, walsh_hadamard_inplace,
    walsh_hadamard_sequency_inplace, walsh_hadamard_sequency_inverse_inplace, HADAMARD_LEVEL_SCALE,
};
#[doc(hidden)]
pub use laplace::{
    ec_laplace_decode, ec_laplace_encode, LAPLACE_LOG_MINP, LAPLACE_MINP, LAPLACE_NMIN,
};
#[doc(hidden)]
pub use mdct::{
    build_low_overlap_window_f32, build_window_half_f32, celt_window_f32, imdct_naive_f32,
    mdct_naive_f32, short_block_geometry, MdctAnalysis, MdctSynthesis, BASIC_WINDOW_HALF,
    BASIC_WINDOW_LEN,
};
#[doc(hidden)]
pub use pcm_encode::{
    encode_celt_frame_pcm, encode_celt_frame_pcm_auto, encode_stereo_celt_frame_pcm,
    encode_stereo_celt_frame_pcm_auto, CeltEncodeState, StereoCeltEncodeState,
};
#[doc(hidden)]
pub use pitch::{
    choose_post_filter_params, pitch_search, PitchEstimate, CONTINUITY_HALF_WIDTH,
    CONTINUITY_RATIO, MIN_PITCH_CORRELATION, MULTIPLE_AVOIDANCE_RATIO,
};
#[doc(hidden)]
pub use post_filter::{
    apply_pitch_prefilter_transition_f32, apply_post_filter_f32, filter_sample_f32, gain_f32,
    gain_q15, tap_coefficients_f32, tap_coefficients_q15, NUM_TAPSETS, POST_FILTER_PERIOD_MAX,
    POST_FILTER_PERIOD_MIN, POST_FILTER_TAPS_F32, POST_FILTER_TAPS_Q15, TAPS_PER_SET,
};
#[doc(hidden)]
pub use pulse_cache::{
    cache_cost_8th, cache_max_k, cache_offset, cache_offset_half_block, cache_stored_qbits,
    cached_bits_to_pulses, cached_bits_to_pulses_extended, cost_exact_8th, CachedPulses,
    CACHE_BITS50, CACHE_INDEX50, EXTENDED_K_CAP, NUM_FRAME_LM, NUM_LM_ROWS,
};
#[doc(hidden)]
pub use pvq::{
    decode_index_to_pulses, decode_pulses, decode_unit_shape, encode_pulses,
    encode_pulses_to_index, encode_shape, encode_unit_shape, normalize_to_unit_l2, pvq_search,
    v_count, V_COUNT_SATURATION,
};
#[doc(hidden)]
pub use range_decoder::RangeDecoder;
#[doc(hidden)]
pub use range_encoder::{RangeEncoder, REM_EMPTY};
#[doc(hidden)]
pub use realloc_walk::{
    realloc_walk, WalkAllocation, WalkBands, WalkBudget, WalkIo, FRAC_STEPS, SKIP_BIT_COST_8TH,
};
#[doc(hidden)]
pub use residual::{
    decode_residual_bands, decode_stereo_residual_bands, ResidualSpectrum, StereoResidualSpectrum,
};
#[doc(hidden)]
pub use spread::{
    decode_spread, encode_spread, pre_rotation_stride, rotation_gain_ratio,
    rotation_gain_squared_ratio, Spread, DEFAULT_SPREAD,
};
#[doc(hidden)]
pub use spread_rotation::{
    apply_2d_rotation, apply_nd_rotation, apply_nd_rotation_multi_block, apply_pre_rotation,
    apply_spread, rotation_angle_f64, EXTRA_ROTATION_MIN_BLOCK_SAMPLES,
};
#[doc(hidden)]
pub use static_alloc::{
    band_static_alloc_1_8th, find_static_alloc, interp_alloc_1_32nd, window_static_alloc_1_8th,
    window_static_alloc_per_band_1_8th, StaticAllocSearch, INTERP_STEPS, NUM_Q, STATIC_ALLOC,
};
#[doc(hidden)]
pub use synthesis::{
    mdct_size, place_residual_spectrum, LongMdctSynthesis, StereoChannel, StereoLongMdctSynthesis,
    CELT_OVERLAP,
};
#[doc(hidden)]
pub use tf_change::{
    decode_tf_changes, decode_tf_parameters, decode_tf_select, encode_tf_changes,
    encode_tf_parameters, encode_tf_select, tf_adjustment, tf_select_matters, TfParameters,
    LM_VALUES, TABLE_60_NON_TRANSIENT_SEL0, TABLE_61_NON_TRANSIENT_SEL1, TABLE_62_TRANSIENT_SEL0,
    TABLE_63_TRANSIENT_SEL1, TF_CHANGE_VALUES,
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
    /// A caller-supplied parameter was out of its documented range
    /// (e.g. `lm > 3`, a band window beyond [`NUM_BANDS`], or a
    /// channel count outside `1..=2`). The decoder state is never
    /// touched when this is returned.
    InvalidParameter,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => write!(
                f,
                "oxideav-celt: requested entry point is not yet implemented"
            ),
            Error::InvalidParameter => write!(
                f,
                "oxideav-celt: a caller-supplied parameter is out of range"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub use codec::{make_decoder, make_encoder, CeltCodecOptions, CeltDecoder, CeltEncoder};

/// Install the `celt` codec (decoder + encoder) into the runtime
/// context — see [`codec::register`].
pub fn register(ctx: &mut RuntimeContext) {
    codec::register(ctx);
}

oxideav_core::register!("celt", register);
