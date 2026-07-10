//! # oxideav-celt
//!
//! Pure-Rust CELT layer of the Opus codec (RFC 6716).
//!
//! **Status (2026-07-04):** round-389. The **stereo PCM codec loop**
//! closes over the uncoupled (Table 56 `dual`) path:
//! [`encode_stereo_celt_frame_pcm_auto`] →
//! [`StereoCeltDecodeState::decode_stereo_frame_auto`] is a fully
//! self-contained interleaved stereo PCM → bytes → PCM codec with no
//! out-of-band data — stereo Table-56 prefix (both channels' coarse
//! energy + the dual/intensity selectors), band-major channel-minor
//! fine energy, one PVQ index per channel per band at a shared `K`
//! derived on both sides from the bit-identical prefix
//! ([`derive_band_pulses_dual`]), per-channel §4.3.6 → §4.3.7.2
//! synthesis. Two latent §4.3.4.1/§4.3.3 defects fixed on the way:
//! the balance accumulator no longer double-counts the granted share
//! (aggregate spend now conserves under the raw-target budget), and
//! the pulse derivations cap the arithmetic budget by the re-measured
//! wire remainder + run on the §4.1.5 worst-case cost estimator (the
//! staged `cache_bits50` mapping is inconsistent — docs question
//! filed). §5.3 encoder decisions grew the §5.3.4.2 allocation trim
//! (wired through the auto encoders) and the §5.3.5 mid/side-vs-dual
//! and Table-66 intensity threshold helpers (decision-only until the
//! §4.3.4.4 `itheta` gap closes). The joint (`itheta`) stereo
//! coupling, transient short blocks, anti-collapse, and the §5.3.1
//! pitch search stay documented docs gaps.
//!
//! **Status (2026-06-21):** round-356. The frame-decode → PCM
//! orchestrator now covers the **stereo** dimension:
//! [`StereoCeltDecodeState::decode_stereo_frame`] decodes the stereo
//! Table 56 control prefix and **both channels' §4.3.2.1 coarse energy**
//! from the range-coded bitstream (the stereo coarse channel interleave
//! is specified, `channels = 2`), composes each channel's §4.3.2 Q8
//! envelope (bitstream coarse + caller-supplied fine), and runs the
//! per-channel §4.3.6 → §4.3.7 synthesis on the two denormalized residual
//! spectra to emit interleaved L/R/L/R PCM — the same docs-gap boundary
//! the mono [`decode_celt_frame`] draws for `band_k`. The shared
//! [`CoarseEnergyState`] carries both channels' inter-frame coarse-energy
//! prediction across frames; `reset()` zeroes it with the per-channel
//! overlap / de-emphasis / post-filter memory. The §4.3.2.1 Laplace
//! recurrence + constants are now sourced from the clean-room narrative
//! `docs/audio/celt/spec/celt-laplace-decode.md` and the data extraction
//! `docs/audio/celt/tables/laplace_constants.csv` (provenance re-anchored
//! off the RFC's Appendix A reference source, which the clean-room policy
//! bars). Transient short-block reassembly + the §4.3.4.4 `itheta` joint
//! coupling + the main §4.3.2.2 fine-energy channel interleave stay
//! documented docs gaps.
//!
//! **Status (2026-06-19):** round-341. The end-to-end frame-decode → PCM
//! orchestrator (`frame_synthesis`) chains the whole documented decode
//! pipeline into one call: `decode_celt_frame` walks Table 56 prefix
//! (`decode_frame_prefix`) → §4.3.2.2 fine energy (`decode_fine_energy`)
//! → §4.3.2 per-band Q8 envelope (`assemble_band_log_energy_q8`, sliced
//! to the coded window) → §4.3.4 residual shape (`decode_residual_bands`)
//! → §4.3.6/§4.3.7 long-MDCT synthesis (`LongMdctSynthesis`) → §4.3.7.1
//! post-filter (`apply_post_filter_f32`) → §4.3.7.2 de-emphasis
//! (`Deemphasis`), turning a CELT range-coded frame into `120 << lm`
//! time-domain PCM samples. `CeltDecodeState` carries the cross-frame
//! overlap tail, post-filter history, de-emphasis memory, and §4.3.2.1
//! coarse-energy prediction for gapless playback (§4.5.2 reset zeroes
//! all four). The per-band pulse counts (`band_k`) and fine-bit counts
//! (`fine_bits`) are inputs — the same RFC-deferred §4.3.3 reallocation
//! boundary the residual loop keeps — so the driver stays inside
//! fully-specified §4.3 territory. Handles the mono, non-transient
//! (single long MDCT) case; a transient/stereo frame is rejected with
//! `NotImplemented`. The §4.3.5 anti-collapse injection stays a
//! documented docs gap (the RFC describes the intent but defers the
//! collapse-detection threshold + pseudo-random generator + injection
//! magnitude to its own reference code); it does not block the non-transient path,
//! where the anti-collapse bit is not decoded.
//!
//! **Status (2026-06-18):** round-336. The §4.3.6 → §4.3.7 long-MDCT
//! synthesis spine (`synthesis`) closes the seam between the residual
//! band-loop and the inverse MDCT: `place_residual_spectrum` maps a
//! coded-window residual spectrum (the `decode_residual_bands` output)
//! into the full `120 << lm`-bin MDCT spectrum at its absolute
//! band-edge offset (zeroing the uncoded low bins and the `20 << lm`
//! high-frequency gap above the `100 << lm` coding top), and
//! `LongMdctSynthesis` runs `MdctSynthesis::frame` with the fixed
//! 120-sample-overlap §4.3.7 window (`CELT_OVERLAP` / `mdct_size`) to
//! emit `120 << lm` time-domain samples — the §4.3.7.1 post-filter's
//! input. This is the residual-spectrum → PCM counterpart of the
//! `decode_residual_bands` band-loop spine, for the non-transient
//! (single long MDCT) case; the transient short-block reassembly stays
//! a documented docs gap (§4.3.1 / §4.3.7 defer the per-short-block
//! layout + inter-block overlap-add to the reference).
//!
//! **Status (2026-06-17):** round-326. The §4.3.3 per-band
//! shape-allocation assembly (`alloc_combine`) combines the interpolated
//! Table-57 static allocation, the decoded band boosts, and the
//! `alloc.trim`-derived per-band tilt into a per-band candidate clamped
//! to the per-band `cap[]` and floored at zero
//! (`combine_band_allocation` → `CombinedAllocation`). This is the
//! "minimums / cap / boost composition in the next stage" the
//! `StaticAllocSearch` result hands off to; the §2.7 hard-minimum skip
//! decision (thresh-floor + concurrent skip decoding) stays deferred
//! to the reference per the RFC / narrative §2.7.
//!
//! **Status (2026-06-15):** round-29. The §4.3 MDCT band-layout module
//! (`band_layout`) exposes the canonical CELT band-edge layout
//! (`EBAND_EDGES_5MS`, the 22 LM=0 cumulative MDCT-bin offsets `0..=100`
//! whose consecutive differences are the RFC 6716 §4.3 Table 55
//! "2.5 ms / Bins" column) plus the band→bin-range accessors the
//! band-walking steps need — `band_edge` / `band_bins` /
//! `band_bin_range` / `coded_total_bins` — all scaled by `1 << lm`. The
//! bin counts are bit-identical to `BAND_BINS_LM`; a test pins the
//! edge-form and count-form transcriptions of Table 55 against each
//! other.
//!
//! **Status (2026-06-14):** round-28. The §4.3 frame-prefix decode
//! driver (`decode_frame_prefix` → `FramePrefix`) chains every CELT
//! control-symbol decoder in RFC 6716 Table 56 bitstream order
//! (silence / post-filter / transient / intra → coarse energy →
//! tf_change / tf_select → spread → caps → band boosts → initial
//! reservations → band allocation), threading the reservation/boost
//! budget between steps so the trim and stereo gates fire against the
//! correct intermediate `ec_tell_frac()`. It stops at the Table 56
//! `fine energy` symbol — the docs-gap boundary where the §4.3.3
//! reallocation pass (deferred to the reference by §4.3.3 / narrative
//! §2.7) begins.
//!
//! **Status (2026-06-13):** round-26. The §4.3.2 final per-band
//! log-energy assembly (`band_energy`) combines the §4.3.2.1 coarse f32
//! log-energies, the §4.3.2.2 fine Q14 corrections, and the §4.3.2.2
//! finalize Q14 corrections into one final per-band log-energy and
//! bridges it onto the Q8 axis the §4.3.6 denormalization and
//! `decode_band_shape` consume (`assemble_band_log_energy_f32` /
//! `assemble_band_log_energy_q8` / `log_energy_f32_to_q8`), closing the
//! `multiply by 256 and round` seam those modules previously left to
//! the caller.
//!
//! **Status (2026-06-12):** round-25. The bit-exact CELT/SILK range
//! decoder (RFC 6716 §4.1) is complete; the CELT frame-header prefix
//! (silence / post-filter / transient / intra per §4.3, plus the
//! deferred anti-collapse bit per §4.3.5) is wired up. The §4.3.2.1
//! coarse-energy decoder is complete: the `ec_laplace_decode`
//! per-symbol recurrence (from the clean-room narrative
//! `docs/audio/celt/spec/celt-laplace-decode.md`, which recovers the
//! range-coder interval narrowing as wire-format facts) drives
//! `decode_coarse_energy`, which runs the §4.3.2.1
//! `unquant_coarse_energy` walk: per-band,
//! per-channel Laplace decode against the `E_PROB_MODEL[lm][intra]
//! [band] -> ProbDecay { prob, decay }` table (4 × 2 × 21 = 168 Q8
//! pairs from `docs/audio/celt/tables/e_prob_model.csv`), the
//! budget-keyed low-rate fallbacks (2-bit zig-zag over
//! `SMALL_ENERGY_ICDF`, 1-bit `{1,1}/2`, implicit `qi = -1`), and the
//! 2-D prediction reconstruction `E[b] = coef*max(-9, E_prev[b]) +
//! prev + q` / `prev += (1 - beta)*q` in the normative
//! floating-point configuration, with the per-LM inter coefficient
//! rows `PRED_COEF_Q15` / `BETA_COEF_Q15` and the intra pair
//! `α=0, β=4915/32768`. The §4.3.2.2
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
//! The §4.3.3 static-allocation search additionally exposes the per-band
//! interpolated allocation vector (`window_static_alloc_per_band_1_8th`):
//! the per-band 1/8-bit breakdown of the window total at a chosen
//! `(qlo, frac)` grid position (`channels * N * interp_alloc << LM >> 2`
//! per band, with the top-column saturated exit reachable), which the
//! §4.3.3 reallocation pass (§2.7 outcome) consumes alongside the
//! minimums, trim offsets, and caps.
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
//! caller-supplied Q8 log-energies, composing directly with the
//! coarse-energy decoder's reconstructed envelope. The
//! §4.3.4 → §4.3.6 single non-split band-decode orchestrator
//! (`decode_band_shape` → `BandShape`) chains the simplest-case decode
//! chain in §4.3 bitstream order: PVQ unit-shape decode
//! (§4.3.4.1/§4.3.4.2) → spreading rotation (§4.3.4.3) → time-frequency
//! resolution change (§4.3.4.5) → denormalization (§4.3.6). The
//! §4.3.4 multi-band residual loop (`decode_residual_bands` →
//! `ResidualSpectrum`) is the residual-section integration spine: given
//! the per-band pulse counts the §4.3.3 allocation produced, it walks
//! the coded-band window `[start, end)` in bitstream order, computing
//! each band's `N` (Table 55) and short-block count (`2^lm` transient /
//! `1` long), evaluating the §4.3.4.5 TF adjustment, invoking the
//! single-band chain, and assembling the per-band shapes into the
//! contiguous MDCT-domain spectrum the §4.3.7 inverse MDCT consumes. It
//! is the residual-section counterpart of `decode_frame_prefix` (the
//! prefix-section spine) and stays inside fully-specified §4.3.4
//! territory by taking `K[]` as input rather than computing it via the
//! gap'd reallocation pass. The
//! §4.3.7 inverse MDCT machinery (the Vorbis-derived power-of-sine
//! window in closed form, validated against the staged
//! `window120.csv` / `window240.csv` data extractions; the low-overlap
//! window construction with its hop-`N` power-complementarity
//! invariant; the direct-form `1/2`-scaled IMDCT + its `4/N` forward
//! companion; and the streaming weighted-overlap-add synthesis state
//! `MdctSynthesis`) is wired up. The reallocation loop (concurrent
//! skip decoding), the fine-energy / shape split, and the §4.3.4.4
//! split-gain band-split path still come later.
//!
//! Every other public API path returns [`Error::NotImplemented`].
//!
//! ## Clean-room provenance
//!
//! All routines in this crate are derived from the **normative prose**
//! of RFC 6716 §1–§8 (the IETF standards-track definition of Opus) and
//! RFC 8251 (the Opus update), together with the project's own
//! clean-room behavioural-narrative and numeric-table material under
//! `docs/audio/celt/` (`spec/celt-*.md` narratives; `tables/*.csv`
//! data extractions with `.meta` provenance). Where the RFC prose
//! names a function but defers the procedure to its implementation
//! (e.g. `ec_laplace_decode`, the §4.3.3 reallocation pass, the
//! transient short-block layout), the procedure is taken from the
//! clean-room narrative that recovers it as wire-format facts, not from
//! the reference source. RFC 6716's **Appendix A reference
//! implementation** (the C source extractable per §A.1) is off-limits
//! under the workspace clean-room policy and was **not** consulted.
//! Black-box invocations of `opusdec` / `opusenc` are permitted as
//! opaque validators.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod alloc_combine;
pub mod allocation_budget;
pub mod analysis;
pub mod anti_collapse;
pub mod band_analysis;
pub mod band_cap;
pub mod band_decode;
pub mod band_energy;
pub mod band_layout;
pub mod band_minimums;
pub mod band_split;
pub mod bit_allocation;
pub mod bits_to_pulses;
pub mod coarse_energy;
pub mod deemphasis;
pub mod denormalization;
pub mod derive_pulses;
pub mod e_prob_model;
pub mod encoder_decisions;
pub mod fine_energy;
pub mod frame_decode;
pub mod frame_encode;
pub mod frame_header;
pub mod frame_synthesis;
pub mod hadamard;
pub mod laplace;
pub mod mdct;
pub mod pcm_encode;
pub mod pitch;
pub mod post_filter;
pub mod pulse_cache;
pub mod pvq;
pub mod range_decoder;
pub mod range_encoder;
pub mod residual;
pub mod spread;
pub mod spread_rotation;
pub mod static_alloc;
pub mod synthesis;
pub mod tf_change;

pub use alloc_combine::{
    combine_band_allocation, find_combined_alloc, CombinedAllocSearch, CombinedAllocation,
};
pub use allocation_budget::{
    compute_initial_reservations, InitialReservations, RSV_BIT_8TH, RSV_INITIAL_SLACK_8TH,
};
pub use analysis::{extract_coded_spectrum, LongMdctAnalysis, StereoPcmAnalysis};
pub use anti_collapse::{apply_anti_collapse, ENERGY_HISTORY_FLOOR_LOG2};
pub use band_analysis::{
    analyze_band_f32, analyze_bands_f32, band_energy_f32, band_log_energy_f32, BandAnalysis,
    SILENCE_LOG_ENERGY,
};
pub use band_cap::{
    compute_band_caps, decode_band_boosts, encode_band_boosts, BoostResult, CACHE_CAPS50,
};
pub use band_decode::{decode_band_shape, BandShape};
pub use band_energy::{
    assemble_band_log_energy_f32, assemble_band_log_energy_q8, log_energy_f32_to_q8,
    FINE_Q14_DENOM, Q14_TO_Q8_SHIFT,
};
pub use band_layout::{
    band_bin_range, band_bins, band_edge, coded_total_bins, EBAND_EDGES_5MS, NUM_BAND_EDGES,
};
pub use band_minimums::{
    compute_thresh, compute_trim_offsets, BAND_BINS_LM, EIGHTH_BIT_QUANTUM, NUM_LM,
    SHORT_FRAME_BAND_BINS,
};
pub use band_split::{
    band_needs_split, max_split_levels, plan_band_split, split_dimensions, BandSplitNode, MAX_LM,
};
pub use bit_allocation::{
    decode_alloc_trim, decode_band_allocation, decode_dual_stereo, decode_intensity_band,
    decode_skip_flag, encode_alloc_trim, encode_band_allocation, encode_dual_stereo,
    encode_intensity_band, encode_skip_flag, intensity_rsv, reserve_stereo, BandAllocation,
    BandAllocationGates, DEFAULT_ALLOC_TRIM, LOG2_FRAC_TABLE,
};
pub use bits_to_pulses::{
    bits_to_pulses_band_loop, bits_to_pulses_band_loop_cached,
    bits_to_pulses_band_loop_cached_thresh, bits_to_pulses_search, cost_log2_v_count_8th,
    BalanceAccumulator, BitsToPulses, DEFAULT_BALANCE_DIVISOR, EIGHTH_BITS_PER_BIT, K_SEARCH_CAP,
    LAST_BALANCE_DIVISOR, SECOND_TO_LAST_BALANCE_DIVISOR,
};
pub use coarse_energy::{
    decode_coarse_energy, encode_coarse_energy, CoarseEnergyState, BETA_COEF_Q15, INTRA_ALPHA_Q15,
    INTRA_BETA_Q15, MAX_CHANNELS, NUM_BANDS, PRED_COEF_Q15, SMALL_ENERGY_ICDF,
};
pub use deemphasis::{
    deemphasize_in_place_f32, preemphasize_in_place_f32, Deemphasis, Preemphasis, ALPHA_P_F32,
    ALPHA_P_Q15,
};
pub use denormalization::{
    denormalize_band_f32, denormalize_band_in_place_f32, denormalize_bands_f32,
    denormalize_bands_in_place_f32, log_energy_q8_to_amplitude_f32, scale_band_f32,
    scale_band_in_place_f32, Q8_DENOM, SQRT_Q8_DENOM,
};
pub use derive_pulses::{
    decode_celt_frame_auto, derive_band_allocation, derive_band_allocation_dual,
    derive_band_pulses, derive_band_pulses_dual, DerivedAllocation,
};
pub use e_prob_model::{
    prob_decay, ProbDecay, E_PROB_MODEL, NUM_LM_FRAME_SIZES, NUM_PREDICTION_TYPES, PRED_INTER,
    PRED_INTRA,
};
pub use encoder_decisions::{
    boost_quanta_8th, boost_thresholds, choose_alloc_trim, choose_band_boosts, choose_intra_mode,
    choose_mid_side_stereo, intensity_start_band, low_band_stereo_correlation, mid_side_extra_dof,
    spectral_tilt_slope, MID_SIDE_DECISION_BANDS, TRIM_TILT_GAIN,
};
pub use fine_energy::{
    apply_finalize_scale_f32, decode_fine_energy, decode_fine_energy_band,
    encode_finalize_extra_bits_depth, encode_fine_energy, encode_fine_energy_band,
    finalize_correction_q14, finalize_extra_bits, finalize_extra_bits_depth,
    finalize_priorities_from_k, fine_correction_q14, fine_correction_qn, quantize_fine_energy_band,
    quantize_fine_energy_f32, FinalizeDepthResult, FinalizePriority, FinalizeResult, FINE_Q14_ONE,
    MAX_FINE_BITS,
};
pub use frame_decode::{decode_frame_prefix, FramePrefix};
pub use frame_encode::{
    encode_celt_frame, encode_celt_frame_auto, encode_celt_frame_auto_boosted, encode_frame_prefix,
    encode_stereo_celt_frame, encode_stereo_celt_frame_auto, EncodedFrame, FramePrefixSpec,
    StereoEncodedFrame,
};
pub use frame_header::{
    decode_anti_collapse_flag, encode_anti_collapse_flag, CeltFrameHeader, PostFilter,
};
pub use frame_synthesis::{
    decode_celt_frame, CeltDecodeState, DecodedFrame, PostFilterParams, StereoCeltDecodeState,
    StereoDecodedFrame,
};
pub use hadamard::{
    apply_tf_resolution_change, apply_tf_resolution_change_inverse, walsh_hadamard_inplace,
    walsh_hadamard_sequency_inplace, walsh_hadamard_sequency_inverse_inplace, HADAMARD_LEVEL_SCALE,
};
pub use laplace::{
    ec_laplace_decode, ec_laplace_encode, LAPLACE_LOG_MINP, LAPLACE_MINP, LAPLACE_NMIN,
};
pub use mdct::{
    build_low_overlap_window_f32, build_window_half_f32, celt_window_f32, imdct_naive_f32,
    mdct_naive_f32, short_block_geometry, MdctAnalysis, MdctSynthesis, BASIC_WINDOW_HALF,
    BASIC_WINDOW_LEN,
};
pub use pcm_encode::{
    encode_celt_frame_pcm, encode_celt_frame_pcm_auto, encode_stereo_celt_frame_pcm,
    encode_stereo_celt_frame_pcm_auto, CeltEncodeState, StereoCeltEncodeState,
};
pub use pitch::{
    choose_post_filter_params, pitch_search, PitchEstimate, CONTINUITY_HALF_WIDTH,
    CONTINUITY_RATIO, MIN_PITCH_CORRELATION, MULTIPLE_AVOIDANCE_RATIO,
};
pub use post_filter::{
    apply_pitch_prefilter_transition_f32, apply_post_filter_f32, filter_sample_f32, gain_f32,
    gain_q15, tap_coefficients_f32, tap_coefficients_q15, NUM_TAPSETS, POST_FILTER_PERIOD_MAX,
    POST_FILTER_PERIOD_MIN, POST_FILTER_TAPS_F32, POST_FILTER_TAPS_Q15, TAPS_PER_SET,
};
pub use pulse_cache::{
    cache_cost_8th, cache_max_k, cache_offset, cache_offset_half_block, cache_stored_qbits,
    cached_bits_to_pulses, cached_bits_to_pulses_extended, cost_exact_8th, CachedPulses,
    CACHE_BITS50, CACHE_INDEX50, EXTENDED_K_CAP, NUM_FRAME_LM, NUM_LM_ROWS,
};
pub use pvq::{
    decode_index_to_pulses, decode_pulses, decode_unit_shape, encode_pulses,
    encode_pulses_to_index, encode_shape, encode_unit_shape, normalize_to_unit_l2, pvq_search,
    v_count, V_COUNT_SATURATION,
};
pub use range_decoder::RangeDecoder;
pub use range_encoder::{RangeEncoder, REM_EMPTY};
pub use residual::{
    decode_residual_bands, decode_stereo_residual_bands, ResidualSpectrum, StereoResidualSpectrum,
};
pub use spread::{
    decode_spread, encode_spread, pre_rotation_stride, rotation_gain_ratio,
    rotation_gain_squared_ratio, Spread, DEFAULT_SPREAD,
};
pub use spread_rotation::{
    apply_2d_rotation, apply_nd_rotation, apply_nd_rotation_multi_block, apply_pre_rotation,
    apply_spread, rotation_angle_f64, EXTRA_ROTATION_MIN_BLOCK_SAMPLES,
};
pub use static_alloc::{
    band_static_alloc_1_8th, find_static_alloc, interp_alloc_1_32nd, window_static_alloc_1_8th,
    window_static_alloc_per_band_1_8th, StaticAllocSearch, INTERP_STEPS, NUM_Q, STATIC_ALLOC,
};
pub use synthesis::{
    mdct_size, place_residual_spectrum, LongMdctSynthesis, StereoChannel, StereoLongMdctSynthesis,
    CELT_OVERLAP,
};
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

/// No-op codec registration. The range decoder is a leaf primitive
/// and does not need to advertise a codec ID; once the band/MDCT
/// path lands, this hook will register the `celt` decoder with the
/// runtime.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("celt", register);
