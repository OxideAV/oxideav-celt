# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2](https://github.com/OxideAV/oxideav-celt/compare/v0.1.1...v0.1.2) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- add comb pitch pre-filter (RFC 6716 §4.3.7.1)
- parametric encode_hybrid_body_{mono,stereo} for LM=2 + LM=3
- encoder + decoder: add LM=2 (10 ms / 480-sample) frame-size path
- per-band TF resolution analyser (RFC §4.3.4.5 + §5.3.6)
- add encode_hybrid_body_stereo for Hybrid stereo high-band
- harden transient detector + quantitative SNR tests
- adopt slim VideoFrame/AudioFrame shape

### Other

- pitch_analysis: add encoder-side comb pre-filter analyser (RFC 6716 §4.3.7.1) — Hann-windowed normalised autocorrelation pitch search over τ ∈ [15, 1022] with first-significant-local-max picking + sub-multiple guard, 3-bit gain mapping from peak NCC, and best-of-three tapset selection by post-pre-filter residual energy. Returns the encoded `(octave, fine_pitch, gain_idx)` syntax fields directly.
- encoder: wire the pre-filter into `encode_frame` (mono) and `encode_frame_stereo` — analysis runs on the pre-emphasized PCM, the in-place `comb_filter` is applied with negated gains so the decoder's matching post-filter cancels exactly (modulo MDCT quantisation), per-channel history + parameter rotation track the decoder's `pf_history` / `pf_*_old` state, silence frames wipe the pre-filter state. Stereo picks one shared parameter set from the channel with the higher peak NCC.
- expose `set_enable_prefilter` test hook + add public-API A/B PSNR test on a 220 Hz + 4 harmonic fixture; verifies the analyser flips the post-filter flag on tonal content and pre-filter-on does not regress reconstruction SNR vs pre-filter-off
- encoder: extend `encode_hybrid_body_mono` + `encode_hybrid_body_stereo` to honour the per-instance `self.frame_samples` / `self.coded_n` / `self.lm` so the LM=2 (480-sample / 10 ms) hybrid path now lights up alongside the LM=3 (960-sample / 20 ms) one — same band selection, MDCT plumbing, and bit-budget logic, just driven off the per-instance frame size. Required for Opus 10 ms Hybrid (configs 12 / 14) wiring in `oxideav-opus`.
- encoder + decoder: add LM=2 (10 ms / 480-sample) frame-size path alongside the existing LM=3 (20 ms / 960-sample) default — same per-band quantisation, PVQ, TF analyser and silence/transient/post-filter machinery, just a smaller MDCT (`coded_n = 100 * M = 400`) and a smaller default packet budget (75% of LM=3)
- expose `CeltEncoder::new_with_frame_samples` + `CeltDecoder::new_with_frame_samples` constructors and per-instance `frame_samples()` accessors; `new()` continues to default to LM=3 for back-compat
- public-API LM=2 round-trip tests: silence + 1 kHz sine (mono) + click-vs-silence transient + stereo dual-tone + constructor rejects unsupported frame sizes (240 / 120 / 720)
- tf_analysis: add per-band TF resolution analyser (RFC 6716 §4.3.4.5 + §5.3.6) — Viterbi-style L1-norm masking model picks per-band `tf_change` and the matching raw delta + `tf_select` bits the decoder reconstructs via `TF_SELECT_TABLE`
- encoder_bands: wire haar1 recombine + time-divide wrapping through `encode_all_bands_mono` so non-zero `tf_change` from the analyser actually applies to the shape vector before / after `quant_partition_enc` — encoder + decoder stay in sync
- encoder: replace hard-coded all-zero TF emission in mono `encode_frame` with the analyser output (constrained to non-transient long-block frames in this round; transient + stereo paths still emit the no-op decision)
- expose `set_force_tf_off` test hook + add A/B PSNR round-trip test on multi-tone + envelope-noise fixtures
- detect_transient: switch peak/min ratio to peak/median for slow-envelope robustness (fade-in no longer false-positives)
- expose `detect_transient` + `detect_transient_with_threshold` + `DEFAULT_TRANSIENT_THRESHOLD_DB` for callers that want to pin the operating point
- add encoder unit tests pinning detector behaviour on silence / sine / fade-in / click / white noise / threshold parameter
- add encoder→header tests pinning `transient` flag on click vs steady-sine end-to-end + `force_long_only` override
- add public-API quantitative SNR tests: pre-echo RMS short vs long and burst SNR short vs long + stationary-signal regression check (detector-on equals force-long on a pure sine)
- refresh stale module-level docstrings on encoder.rs / decoder.rs that still claimed transient was unimplemented

## [0.2.0](https://github.com/OxideAV/oxideav-celt/compare/v0.1.0...v0.2.0) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- clamp post-filter period to avoid OOB on short subframes
- guard post-filter tapset on remaining bit budget
- use live range-coder rng for per-band PVQ + anti-collapse seed
- add post-filter smoke test + preemph/deemph inverse test
- wire post-filter + pre/de-emphasis through decoder & encoder
- add BSD-3-Clause attribution for libopus-derived code
- release v0.1.0

## [0.1.0](https://github.com/OxideAV/oxideav-celt/compare/v0.0.4...v0.1.0) - 2026-04-19

### Other

- promote to 0.1.0

## [0.0.4](https://github.com/OxideAV/oxideav-celt/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- transient test A/Bs short-block vs long-only pre-echo
- README + lib-doc: transient/short-blocks are implemented
- emit transient / short-block packets from encoder
- wire short-block IMDCT into decoder (transient path)
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- rewrite README + update Cargo description
- add public-API encode -> decode roundtrip tests
- emit silence flag (§4.3 Table 56) for near-silent input
- add standalone CeltDecoder on the CodecRegistry path
