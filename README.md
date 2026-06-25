# oxideav-celt

Pure-Rust CELT (the MDCT path of Opus, RFC 6716).

## Status

**Decoder building blocks, in progress.** The range decoder and the
full CELT control-symbol decode path (frame-prefix → coarse energy →
fine energy → time-frequency parameters → spreading → bit allocation)
are implemented bit-exactly per RFC 6716, along with the PVQ shape
decoder, spreading rotation, Hadamard TF transforms, band
denormalization, post-filter, de-emphasis, and the inverse MDCT /
overlap-add synthesis. The `decode_frame_prefix` driver chains every
control symbol in RFC 6716 Table 56 order up to the `fine energy`
symbol; the `LongMdctSynthesis` spine places a decoded residual
spectrum into the full `120 << lm`-bin MDCT spectrum and runs the
§4.3.7 inverse MDCT + weighted overlap-add.

**End-to-end PCM (mono, long MDCT).** `decode_celt_frame` now chains
the whole documented decode pipeline into a single call that turns a
CELT range-coded frame into time-domain PCM: Table 56 prefix → §4.3.2.2
fine energy → §4.3.2 per-band Q8 envelope assembly → §4.3.4 residual
(shape) decode → §4.3.6/§4.3.7 long-MDCT synthesis → §4.3.7.1
post-filter → §4.3.7.2 de-emphasis. A streaming `CeltDecodeState`
carries the cross-frame overlap tail, post-filter history,
de-emphasis memory, and §4.3.2.1 coarse-energy prediction for gapless
playback. The per-band pulse counts (`band_k`) and fine-bit counts
(`fine_bits`) are inputs — the same RFC-deferred boundary the residual
loop already draws — so the driver stays inside fully-specified §4.3
territory.

**Stereo synthesis (per-channel spectra → interleaved PCM).** The
§4.3.6 denormalization and the §4.3.7 inverse MDCT + weighted
overlap-add are per-channel and fully specified by RFC 6716, as is the
§4.3.2 per-channel energy envelope (`assemble_band_log_energy_q8`
already accepts channel `0`/`1`). `StereoLongMdctSynthesis` and the
streaming `StereoCeltDecodeState` / `synthesize_stereo_frame` run that
per-channel chain for two channels — two independent IMDCT + WOLA spines
with their own overlap tails, per-channel §4.3.7.1 post-filter history
plus previous-frame parameters (for the §4.3.7.1 gain-transition
crossfade), and per-channel §4.3.7.2 de-emphasis memory — and interleave the result
into one L/R/L/R PCM buffer. They take the two channels' decoded
denormalized spectra as input, drawing the boundary at the §4.3.4.4
`itheta` mid/side coupling docs gap (the same input-boundary the mono
spine draws for `band_k`). `tests/stereo_synthesis.rs` drives the whole
documented per-channel stereo chain (energy → denormalize → synthesize)
end-to-end.

**Stereo frame decode (bitstream prefix + coarse energy → interleaved
PCM).** `StereoCeltDecodeState::decode_stereo_frame` is the stereo
counterpart of `decode_celt_frame`: it decodes the stereo Table 56
control prefix and **both channels' §4.3.2.1 coarse energy** from the
range-coded bitstream (the stereo coarse channel interleave *is*
specified — `decode_coarse_energy` with `channels = 2`), composes each
channel's §4.3.2 Q8 envelope (bitstream coarse + caller-supplied fine
corrections), and runs the per-channel §4.3.6 → §4.3.7 synthesis on the
two denormalized residual spectra to emit interleaved PCM. The shared
`CoarseEnergyState` carries both channels' inter-frame coarse-energy
prediction; `reset()` zeroes it alongside the per-channel overlap /
de-emphasis / post-filter memory. The per-channel residual spectra and
the main fine-energy corrections are inputs — the §4.3.4.4 `itheta`
coupling and the main §4.3.2.2 fine-energy channel interleave are the
docs-gap boundaries, the same shape the mono path keeps for `band_k`.

Not yet implemented: the §4.3.3 reallocation loop that produces
`band_k` / `fine_bits` (concurrent skip decoding + fine/shape split),
the §4.3.4.4 split-gain band-split path, the §4.3.4.4 stereo `itheta`
mid/side bitstream coupling, the main §4.3.2.2 fine-energy per-channel
interleave, the transient short-block time-domain reassembly
(§4.3.1 / §4.3.7), and the §4.3.5 anti-collapse injection — each blocked
on detail that RFC 6716 §4.3.3 / §4.3.4.4 / §4.3.1 / §4.3.7 / §4.3.5 and
the clean-room narrative §2.7 defer to the reference implementation.
Both `decode_celt_frame` and `decode_stereo_frame` reject a transient
frame with `Error::NotImplemented` rather than mis-decode it.

**Encode building blocks, in progress.** The encode-direction primitives
are growing: the §4.3.4.2 PVQ codeword index encode
(`encode_pulses_to_index`, the exact inverse of the decode loop), the
§5.3.8.1 PVQ codeword search (`pvq_search` → `encode_unit_shape`) giving
the full input-vector → integer-codeword → bitstream-index PVQ encode
chain, the **§5.1 range encoder** (`RangeEncoder`, new in r371) — the
bit-exact inverse of the §4.1 range decoder that serialises every
encode-side range symbol into a frame — and the **§4.3.2.2 fine-energy
encode** (`quantize_fine_energy_band` → `encode_fine_energy[_band]`, also
r371), the inverse of `decode_fine_energy` that quantises a correction to
the band's index `f` and writes it as `B_i` raw bits. The coarse
§4.3.2.1 Laplace energy encode (DOCS-GAP, same boundary as the coarse
decode) and the codec-registration entry point still return
`Error::NotImplemented`.

**Range encoder (RFC 6716 §5.1).** `RangeEncoder` is the bit-packer for
the CELT/SILK encode path, the exact inverse of `RangeDecoder`. It keeps
the §5.1 four-tuple state `(val, rng, rem, ext)` and exposes `encode`
(§5.1.1 generic `(fl, fh, ft)` symbol), `encode_bin` (§5.1.2.1),
`enc_bit_logp` (§5.1.2.2), `enc_icdf` (§5.1.2.3, reusing the decoder's
icdf tables), `enc_bits` (§5.1.3 raw bits from the end), `enc_uint`
(§5.1.4, with the `ftb > 8` range-coded-top-8-bits + raw-remainder
split), and `finish` (§5.1.5 `ec_enc_done`: the maximal-trailing-zeros
`end` selection, carry flush, and range/raw byte merge). Renormalization
(§5.1.1.1) drives the §5.1.1.2 carry-propagation / output-buffering
scheme (the deferred `rem`/`ext` byte accounting). `tell()` / `tell_frac()`
(§5.1.6) report the **same** budget the decoder reports after the same
symbols — the §5.1 / §4.1.6 lockstep hook that CELT bit allocation
depends on. Verified by full round-trips through `RangeDecoder`
(`tests/range_codec_roundtrip.rs`): every symbol type, the large-`ft`
split path, mixed interleaved streams, and 1000-op deterministic random
streams recover bit-exactly with matching `rng`/`tell`/`tell_frac` at
every step.

**Documented allocation→pulses seam.** `tests/allocation_to_pulses.rs`
composes the fully-specified §4.3.3 modules on *both sides* of the one
remaining allocation docs gap (`interp_bits2pulses`: the reallocation +
concurrent skip + fine/shape split): `decode_frame_prefix` →
`find_combined_alloc` (the §4.3.3 column search "nearest but not
exceeding … subject to tilt, boosts, [and] band maximums") →
`bits_to_pulses_band_loop_cached` (§4.3.4.1 bit-exact `cache_bits50`
pulse counts). For pure-CELT mono the derived `band_k` drives the full
`decode_celt_frame` pipeline to finite PCM, deterministically, with the
total pulse count monotone in the frame budget — demonstrating the
`band_k` the residual loop takes as input *can* be produced from the
documented modules wherever §4.3.3 is not deferred. Only the fine/shape
split + concurrent-skip reallocation remains a genuine docs gap.

**Caller-input-free mono decode (`decode_celt_frame_auto`).** The
documented allocation→pulses seam is now a public API, not just a test
composition. `derive_band_pulses(prefix, lm, channels, stereo)` runs the
§4.3.3 column search over a decoded `FramePrefix`'s post-boost budget,
then the §4.3.4.1 bits-to-pulses loop (threading the balance accumulator
across the coded-band window in spec order), and returns the per-band
pulse counts — clamping each `K` to the largest value whose PVQ codebook
`V(N, K)` stays representable in the single-block decode (a larger `K` is
the deferred §4.3.4.4 split regime). `decode_celt_frame_auto(state,
frame_bytes, start, end)` chains the whole thing: decode the prefix,
derive the pulse counts, and run `decode_celt_frame` with no fine
refinement — a mono, non-transient CELT frame → PCM with **no
caller-supplied `band_k` / `fine_bits`**. It is the deepest
caller-input-free decode the wall permits: every step is RFC-specified
except the deferred fine/shape split, approximated by treating the whole
combined allocation as shape. A transient frame, an out-of-range window,
or a band that hits the §4.3.4.4 split gap is surfaced as
`Error::NotImplemented` / `InvalidParameter`, never mis-decoded. The
derivation is deterministic (identical bytes → identical pulse counts →
identical PCM) and matches the manual `decode_frame_prefix` →
`derive_band_pulses` → `decode_celt_frame` compose exactly.

The module-by-module API surface is documented below.

Range decoder (RFC 6716 §4.1):

* `RangeDecoder::new(buf)` — initialization per §4.1.1.
* `decode_bin(ftb)` — power-of-two-`ft` decode (§4.1.3.1).
* `dec_bit_logp(logp)` — binary symbol with probability `2^-logp` of
  a "1" (§4.1.3.2).
* `dec_icdf(icdf, ftb)` — table-driven inverse-CDF symbol decode,
  the primary SILK interface (§4.1.3.3).
* `dec_bits(n)` — raw bits, packed LSB-first from the end of the
  frame (§4.1.4).
* `dec_uint(ft)` — uniformly-distributed integer in `0..ft`,
  including the `ftb > 8` split-decode branch (§4.1.5).
* `tell()` — whole-bit budget accounting (§4.1.6.1).
* `tell_frac()` — 1/8th-bit-precision budget accounting (§4.1.6.2),
  satisfying `tell() == ceil(tell_frac()/8)` everywhere.
* `storage_bits()` — the total frame size in bits, the budget the
  §4.3.2.1 coarse-energy fallback dispatch compares `tell()` against.
* Sticky `has_error()` for the corrupt-frame path documented in
  §4.1.5.

CELT frame header (RFC 6716 §4.3 prefix + §4.3.5 anti-collapse):

* `CeltFrameHeader::decode_prefix(dec)` walks Table 56 in order
  through silence (§4.3 / `{32767,1}/32768`), the post-filter flag
  and its four parameters (§4.3.7.1: `octave` uniform 0..6,
  `period` = `4+octave` raw bits, `gain` 3 raw bits, `tapset`
  `{2,1,1}/4`), the transient flag (§4.3.1, `{7,1}/8`), and the
  intra flag (§4.3.2.1, `{7,1}/8`).
* `decode_anti_collapse_flag(dec, transient)` reads the §4.3.5
  `{1,1}/2` bit on transient frames and is a no-op otherwise. This
  is exposed separately because §4.3.5 places anti-collapse AFTER
  the band shape vectors in the bitstream.
* `CeltFrameHeader::post_filter_gain_q15()` rebuilds the §4.3.7.1
  gain `G = 3*(gain+1)/32` in Q15 fixed-point.
* `CeltFrameHeader::encode_prefix(enc)` is the exact inverse of
  `decode_prefix`: it serialises the Table-56 prefix (silence →
  post-filter flag + its four §4.3.7.1 parameters, with `fine_pitch`
  reconstructed from `period` → transient → intra) into a `RangeEncoder`.
  `encode_anti_collapse_flag(enc, transient, on)` writes the §4.3.5 bit
  at the post-band-shape position (no-op on non-transient frames). Both
  reject out-of-range fields with `Error::InvalidParameter`.

Coarse-energy decoding (RFC 6716 §4.3.2.1 + Appendix A `laplace.c`
/ `quant_bands.c`):

* `NUM_BANDS = 21` (RFC Table 55). Pure CELT codes all 21 bands;
  Hybrid mode codes only bands 17..=20 (the SILK layer covers
  0..=16 below 8 kHz). `MAX_CHANNELS = 2`.
* `ec_laplace_decode(dec, fs0, decay)` — the §4.3.2.1 per-symbol
  Laplace decode recurrence, transcribed from RFC 6716 Appendix A
  `laplace.c` (the reference listing embedded in the RFC's own text,
  extracted per §A.1 and SHA-1-verified against the value §A.1
  prints). One 15-bit `ec_decode_bin` probe, a geometric walk down
  the decaying tail (`fs * decay >> 15` per magnitude step with the
  `LAPLACE_MINP` floor), the uniform `LAPLACE_MINP` far-tail
  (`LAPLACE_NMIN = 16` deltas guaranteed per direction), sign from
  the lower/upper half of the magnitude span, and the `ec_dec_update`
  interval commit. Validated by inverting a test-only Appendix A
  `entenc.c`/`laplace.c` encoder transcription bit-for-bit across
  every `E_PROB_MODEL` cell (`tests/laplace_roundtrip.rs`), including
  the far-tail clamp and a pinned anchor stream.
* `decode_coarse_energy(dec, state, intra, lm, start, end, channels)`
  — the full §4.3.2.1 envelope walk (Appendix A
  `unquant_coarse_energy`): per band × channel, budget-keyed dispatch
  on `storage_bits() - tell()` (≥ 15 bits → Laplace with the
  `E_PROB_MODEL[lm][intra][band]` pair scaled `prob << 7` / `decay
  << 6`; ≥ 2 → 2-bit zig-zag over `SMALL_ENERGY_ICDF = {2, 1, 0}`;
  ≥ 1 → one `{1,1}/2` bit; else implicit `qi = -1`), then the
  normative-float reconstruction `E[b] = coef * max(-9.0,
  E_prev[b]) + prev + q` with `prev += (1 - beta) * q` per channel.
* `CoarseEnergyState { energy: [[f32; 21]; 2] }` carries the
  per-channel base-2 log-energies (1.0 = 6 dB) across frames;
  `reset()` zeroes it for the §4.5.2 decoder reset.
* Prediction coefficients: intra `INTRA_ALPHA_Q15 = 0` /
  `INTRA_BETA_Q15 = 4915` (the literal §4.3.2.1 `4915/32768`);
  inter per-LM rows `PRED_COEF_Q15 = [29440, 26112, 21248, 16384]`
  and `BETA_COEF_Q15 = [30147, 22282, 12124, 6554]` (Appendix A
  `quant_bands.c`).
* `E_PROB_MODEL[lm][intra][band] -> ProbDecay { prob, decay }` —
  the §4.3.2.1 Laplace-parameter table, 4 × 2 × 21 = 168 Q8 pairs,
  transcribed verbatim from `docs/audio/celt/tables/e_prob_model.csv`.
  `prob_decay(lm, intra, band) -> Option<ProbDecay>` folds the
  `bool intra` flag onto the staged CSV's `0 = inter / 1 = intra`
  middle axis. Constants `NUM_LM_FRAME_SIZES = 4`,
  `NUM_PREDICTION_TYPES = 2`, `PRED_INTER = 0`, `PRED_INTRA = 1`
  formalise the table's three axes.

Stereo reservation helpers (RFC 6716 §4.3.3 + clean-room narrative
§2.5):

* `LOG2_FRAC_TABLE: [u8; 24]` — the conservative `log2(n)` table in
  1/8-bit units, indexed by `coded_bands ∈ 0..=23`. Numeric values
  reproduced from `docs/audio/celt/tables/log2_frac_table.csv`.
* `intensity_rsv(coded_bands, stereo, total_8th_bits) -> u32` —
  returns the intensity-stereo selector reservation in 1/8 bits
  (`LOG2_FRAC_TABLE[coded_bands]` when stereo + budget covers it,
  zero otherwise per §4.3.3).
* `reserve_stereo(coded_bands, stereo, total_8th_bits) -> (u32, i32, u32)`
  — runs both stereo reservation steps (intensity_rsv then
  dual_stereo_rsv) in §4.3.3 order. Returns
  `(intensity_rsv, total_after, dual_stereo_rsv)` in 1/8 bits. Mono
  frames pass through unchanged.

Bit-allocation field decoders (RFC 6716 §4.3.3 + Table 58):

* `decode_alloc_trim(dec, gated)` decodes the trim parameter via
  Table 58 PDF `{2,2,5,10,22,46,22,10,5,2,2}/128`; returns `Some(0..=10)`
  when gated on, `None` when the §4.3.3 budget gate
  `ec_tell_frac()+48 <= total - total_boost` was missed (caller
  treats absent as the default `5`).
* `decode_skip_flag(dec, gated)` decodes the §4.3.3 `{1,1}/2` skip
  bit (`logp=1`); `None` when no skip reservation was made.
* `decode_intensity_band(dec, gated, coded_bands)` decodes the
  uniform intensity-band offset in `0..=coded_bands` via
  `dec_uint(coded_bands+1)`; `None` for mono frames or when the
  stereo intensity reservation didn't fit.
* `decode_dual_stereo(dec, gated)` decodes the §4.3.3 `{1,1}/2`
  dual-stereo flag; `None` when no dual reservation was made.
* `decode_band_allocation(dec, gates)` orchestrates the four
  fields in Table 56 order (trim → skip → intensity → dual),
  returning a `BandAllocation` with §4.3.3 defaults filled in for
  every gated-off field. The orchestrator does not touch the
  range decoder for gated-off fields, so caller-side
  `ec_tell_frac()` accounting stays accurate.
* `encode_alloc_trim` / `encode_skip_flag` / `encode_intensity_band` /
  `encode_dual_stereo` and the `encode_band_allocation(enc, gates,
  alloc)` orchestrator are the exact inverses: each field is written
  only when its gate is open, in Table-56 order, leaving the range
  encoder untouched for gated-off fields. Out-of-range fields
  (`alloc_trim > 10`, `intensity_band_offset > coded_bands`) are
  rejected with `Error::InvalidParameter`.

Initial reservations budget walk (RFC 6716 §4.3.3 + clean-room
narrative §2.5):

* `RSV_BIT_8TH = 8` (1 bit = 8 1/8-bits) — the quantum for each of
  the three binary reservations (`anti_collapse_rsv`, `skip_rsv`,
  `dual_stereo_rsv`).
* `RSV_INITIAL_SLACK_8TH = 1` — the conservative slack subtracted
  from the raw frame-size budget at the start of the §2.5 walk.
* `compute_initial_reservations(frame_bytes, ec_tell_frac,
  is_transient, lm, stereo, coded_bands) -> InitialReservations`
  chains the §2.5 four-step reservation walk in one call:
  1. `total_initial = frame_bytes * 64 - ec_tell_frac - 1` (frame
     size in 1/8 bits, less the bits decoded so far, less the
     conservative slack).
  2. `anti_collapse_rsv = 8` iff transient AND `LM > 1` AND
     `total >= (LM+2)*8`; clamp `total >= 0` afterwards.
  3. `skip_rsv = 8` iff `total > 8` after anti-collapse.
  4. Stereo intensity_rsv (via `LOG2_FRAC_TABLE`) + dual_stereo_rsv
     (8 iff stereo and budget > 8 after intensity); mono frames
     skip both.
* `InitialReservations { total_initial, anti_collapse_rsv,
  skip_rsv, intensity_rsv, dual_stereo_rsv, total, coded_bands,
  stereo }` packages the running budget at the point the band-boost
  loop should start (`total`) alongside every individual reservation
  for caller cross-checks. `total_reserved()` returns the sum and
  satisfies the identity `total = total_initial - total_reserved()`.
* `InitialReservations::gates_for_band_allocation(ec_tell_frac_now,
  total_boost) -> BandAllocationGates` synthesises the four-field
  gate booleans `decode_band_allocation` needs:
  - `trim_gated = ec_tell_frac_now + 48 <= total_initial - total_boost`.
  - `skip_gated = (skip_rsv == 8)`.
  - `intensity_gated = stereo AND intensity_rsv > 0`.
  - `dual_gated = stereo AND dual_stereo_rsv == 8`.
  The `total_boost` argument is supplied by the band-boost loop's
  `BoostResult::total_boost` once it has run.

Per-band caps and band boosts (RFC 6716 §4.3.3 + clean-room
narrative §§2.2–2.3):

* `CACHE_CAPS50: [[u8; 21]; 8]` — the §4.3.3 per-band maximum-
  allocation cache (`cache.caps`), 168 entries arranged as
  `CACHE_CAPS50[2*LM + stereo][band]` for `LM ∈ {0,1,2,3}` ×
  `stereo ∈ {0,1}`. Numeric values reproduced from
  `docs/audio/celt/tables/cache_caps50.csv`.
* `compute_band_caps(lm, stereo, channels, bins_per_band, caps) -> bool`
  applies the §4.3.3 conversion `cap[band] = (cache.caps[i] + 64) *
  channels * N[band] / 4`. Output saturates to `i16::MAX` defensively
  in the unlikely overflow corner; returns `false` for `lm > 3` /
  `channels ∉ {1,2}` / length mismatches.
* `decode_band_boosts(dec, start, end, bins_per_band, caps, total_bits)
  -> Option<BoostResult>` runs the §4.3.3 dynalloc-logp loop (RFC 6716
  lines 6339–6360) literally. Per-band quanta
  `min(8*N, max(48, N))`, starting `dynalloc_logp = 6` falling to
  `1` mid-band on the first accepted boost and stepping down one
  notch per boosted band (floored at `2`). The §4.3.3 "at very low
  rates ... inner loop may not run even once" path is exercised
  explicitly — `total_bits` below the initial guard ⇒ zero boosts
  emitted, the range decoder is not consulted.
* `BoostResult { boost: Vec<i32>, total_boost: i32,
  total_bits_remaining: i32 }` — per-band 1/8-bit boosts plus the
  `total_boost` accumulator the §4.3.3 alloc-trim gate consumes
  (`ec_tell_frac() + 48 <= total_bits - total_boost`).

Fine-energy refinement (RFC 6716 §4.3.2.2):

* `decode_fine_energy_band(dec, b_bits)` reads exactly `b_bits` raw
  bits via `dec_bits` (§4.1.4) to form `f ∈ [0, 2^b_bits)`, and
  returns the closed-form Q14 correction
  `(2f+1) * 2^(13-b_bits) - 2^13`. `b_bits = 0` is a no-op (bands
  the allocator could not buy fine bits for stay at the coarse
  value); the function does not consume raw bits in that case.
* `decode_fine_energy(dec, bits_per_band)` walks all 21 bands and
  returns per-band Q14 corrections in one call. Total bits consumed
  equals `sum(bits_per_band)`.
* `fine_correction_q14(f, b_bits)` / `fine_correction_qn(f, b_bits, n)`
  expose the closed-form §4.3.2.2 correction at arbitrary fixed-point
  scales (Q14 specialised path + rounding Qn variant) for encoders
  and test scaffolding.
* `finalize_extra_bits(dec, priorities, coded_bands, channels, budget)`
  walks the §4.3.2.2 finalize step: leftover raw bits are spent ≤ 1
  per band per channel in `(priority 0 ascending, priority 1
  ascending)` order. Returns per-band Q14 corrections summed across
  channels plus `(bits_consumed, bits_unused)`. Excess budget is left
  unused per the spec ("If any bits are left after this, they are
  left unused").
* `MAX_FINE_BITS = 8` caps the per-band fine-bit count (the §4.3.3
  allocator never exceeds this in practice; the Q14 closed-form is
  exact for `b_bits <= 13`).
* `quantize_fine_energy_band(correction_q14, b_bits)` inverts the
  §4.3.2.2 decode map, returning the quantizer index
  `f = clamp(floor((correction + 1/2)*2^B), 0, 2^B - 1)`. Round-trips
  exactly with `fine_correction_q14` on the grid (`quantize ∘ decode ==
  id` over every legal `(f, B)`).
* `encode_fine_energy_band(enc, f, b_bits)` /
  `encode_fine_energy(enc, f_per_band, bits_per_band)` write the chosen
  `f` as `B_i` raw bits through the §5.1 `RangeEncoder` — the exact
  inverse of `decode_fine_energy_band` / `decode_fine_energy`. `B_i == 0`
  is a no-op; an `f` exceeding `b_bits` is rejected with
  `Error::InvalidParameter`.

Time-frequency change parameters (RFC 6716 §4.3.4.5 + §4.3.1):

* `decode_tf_changes(dec, start, end, is_transient)` reads one bit
  per coded band, with PDF `{3,1}/4` (transient) or `{15,1}/16`
  (non-transient) on the first band and `{15,1}/16` (transient) or
  `{31,1}/32` (non-transient) on subsequent bands. The rare "1"
  symbol toggles the running TF choice per the §4.3.4.5 differential
  encoding.
* `decode_tf_select(dec, is_transient, lm, tf_changes)` decodes the
  global `tf_select` flag with PDF `{1,1}/2`, but only when at least
  one decoded `tf_change` would produce a different TF adjustment
  under tf_select=0 vs tf_select=1. Returns `(0, false)` when the
  bit was gated off; the range decoder is untouched in that case.
* `tf_select_matters(is_transient, lm, tf_changes)` is the pure
  predicate the §4.3.4.5 prose calls "can it have an impact on the
  result".
* `tf_adjustment(is_transient, tf_select, lm, tf_change)` indexes
  the four published Tables 60–63 in one function. Returns a signed
  i8 per band (negative ⇒ +temporal resolution, positive ⇒
  +frequency resolution, both via Hadamard levels per §4.3.4.5).
* `decode_tf_parameters(dec, start, end, is_transient, lm)`
  orchestrates the full §4.3.4.5 walk in Table 56 order and returns
  a `TfParameters { tf_changes, tf_select, tf_select_decoded }`.
* `encode_tf_changes` / `encode_tf_select` / `encode_tf_parameters` are
  the exact inverses: `encode_tf_changes` recovers each band's toggle bit
  from the cumulative per-band TF choice and writes it with the
  first/subsequent-band logp; `encode_tf_select` re-evaluates the
  "can it have an impact" gate and emits the `{1,1}/2` bit only when
  open; `encode_tf_parameters` chains both in Table-56 order. A
  closed-gate non-zero `tf_select` is rejected.
* The four tables themselves are exposed as `pub const`s
  (`TABLE_60_NON_TRANSIENT_SEL0`, `TABLE_61_NON_TRANSIENT_SEL1`,
  `TABLE_62_TRANSIENT_SEL0`, `TABLE_63_TRANSIENT_SEL1`),
  `[i8; 2] × 4` rows indexed by LM ∈ `{0,1,2,3}`
  (= 2.5 / 5 / 10 / 20 ms) and `tf_change` ∈ `{0, 1}`.

Hadamard transform primitives (RFC 6716 §4.3.4.5 final paragraph):

* `walsh_hadamard_inplace(samples, levels)` — in-place orthonormal
  forward Walsh–Hadamard transform in natural (Hadamard) order. The
  radix-2 butterfly cascade with per-level `1/√2` scaling, so the
  matrix is orthonormal and the L2 norm is preserved.
* `walsh_hadamard_sequency_inplace(samples, levels)` — same transform
  in sequency (Walsh) order. Natural-order butterfly cascade followed
  by the textbook `bit_reverse(gray(s))` permutation so that the
  zero-sequency (DC) bin lands at index 0 and the maximum-sequency
  bin lands at index `2^levels - 1`. Used by the time-resolution-
  increase branch per §4.3.4.5 ("the decoder uses the 'sequency
  order' because the input vector is sorted in time").
* `apply_tf_resolution_change(band, tf_adjustment, nb_blocks)` —
  orchestrator over the interleaved `nb_blocks × subvec` band
  layout. `tf_adjustment < 0` applies `|tf_adjustment|` sequency-
  ordered WHT levels to each sub-vector (increase time resolution);
  `tf_adjustment > 0` applies `|tf_adjustment|` natural-order WHT
  levels across the interleaved blocks (increase frequency
  resolution); `tf_adjustment == 0` is a no-op. Returns `false` and
  leaves `band` unchanged when the shape constraints are violated
  (non-power-of-two block count or sub-vector length, or a request
  exceeding the available WHT levels in either direction).
* `HADAMARD_LEVEL_SCALE = 1/√2` — the per-level normalization that
  makes the radix-2 butterfly orthonormal. Documented as a
  decoder-side decision because §4.3.4.5 is silent on the scaling
  convention; chosen to keep the unit-norm band shape vectors
  invariant across TF resolution changes so the §4.3.6
  denormalization sees the same shape regardless of `tf_adjustment`.

Spreading parameter (RFC 6716 §4.3.4.3 + Table 56 + Table 59):

* `decode_spread(dec)` reads the §4.3.4.3 spread field with PDF
  `{7, 2, 21, 2}/32` via the §4.1.3.3 ICDF path. Returns one of
  the four `Spread` variants `{None, Light, Normal, Aggressive}`
  in raw-value order (`spread = 0..=3`). `encode_spread(enc, spread)`
  is the exact inverse, writing the same ICDF symbol.
* `Spread::f_r()` is the Table 59 lookup: `None` (`spread=0`),
  `Some(15)` (`spread=1`), `Some(10)` (`spread=2`), `Some(5)`
  (`spread=3`). `f_r = None` means the spreading rotation is the
  identity.
* `rotation_gain_ratio(spread, n, k)` returns the `g_r = N/(N+f_r*K)`
  rotation gain as a `(num, den)` unsigned-integer pair so the PVQ
  caller can pick its own fixed-point representation. `Spread::None`
  collapses to `(0, 1)`. `N = 0` returns `(0, 1)` defensively so a
  caller looping over all band indices sees a well-defined zero.
* `rotation_gain_squared_ratio(spread, n, k)` returns the same ratio
  squared (`u64` per term to avoid overflow), feeding the
  `theta = pi * g_r^2 / 4` rotation-angle computation that the PVQ
  caller will run.
* `pre_rotation_stride(n, nb_blocks)` returns the
  `round(sqrt(N/nb_blocks))` interleave stride that the §4.3.4.3
  "extra rotation" rule applies BEFORE the main rotation when each
  time block represents at least 8 samples. Returns `None` for the
  gated-off cases (`nb_blocks <= 1`, per-block sample count below 8,
  or empty band).
* `DEFAULT_SPREAD = Spread::Normal` is the bulk-probability fallback
  for callers that skip the CELT side of the bitstream entirely.

Spreading rotation (RFC 6716 §4.3.4.3 page 117):

* `rotation_angle_f64(spread, n, k) -> f64` — the §4.3.4.3 rotation
  angle `theta = pi * g_r^2 / 4` computed from the closed-form
  `g_r = N / (N + f_r * K)`. Returns `0.0` for `Spread::None` and for
  the degenerate `N = 0` band. The squaring and the division run in
  f64 to keep the result well below f32 precision loss.
* `apply_2d_rotation(x_i, x_j, cos_theta, sin_theta) -> (f32, f32)`
  — the single 2-D rotation primitive (`cos / sin / -sin / cos`)
  the §4.3.4.3 prose names `R(i, j)`. `cos_theta` / `sin_theta` are
  precomputed once per band.
* `apply_nd_rotation(samples, theta)` — the §4.3.4.3 N-D rotation
  chain on a single time block. Forward pass `R(x_0, x_1),
  R(x_1, x_2), ..., R(x_{n-2}, x_{n-1})`, then reverse pass
  `R(x_{n-2}, x_{n-1}), ..., R(x_0, x_1)` per the "back and forth"
  ordering. `theta == 0.0` or `len < 2` is a no-op.
* `apply_nd_rotation_multi_block(samples, nb_blocks, theta)` —
  applies the chain independently per time block on a buffer
  interleaved as `samples[b + nb_blocks * i]` for block `b`,
  in-block index `i` (the layout the §4.3.4.5 Hadamard transform
  uses). Returns `false` on shape-constraint violations
  (`nb_blocks == 0`, divisibility failure).
* `apply_pre_rotation(samples, nb_blocks, theta)` — the §4.3.4.3
  "extra rotation" by `(pi/2 - theta)`, applied BEFORE the main
  rotation on each stride-interleaved sub-sequence
  `S_k = { stride * n + k }, n = 0..N/stride - 1`, with stride
  `round(sqrt(N/nb_blocks))`. Gated off when `nb_blocks <= 1` or
  the per-block sample count drops below
  `EXTRA_ROTATION_MIN_BLOCK_SAMPLES = 8`. Returns `false` when the
  pre-rotation does not apply so the caller can skip straight to
  the main rotation.
* `apply_spread(spread, samples, k, nb_blocks)` — full §4.3.4.3
  orchestrator. Computes `theta`, runs the gated pre-rotation, and
  applies the main per-block rotation chain. `Spread::None` is a
  no-op; the L2 norm of a unit-norm input is preserved across the
  pass (orthonormal composition).
* `EXTRA_ROTATION_MIN_BLOCK_SAMPLES = 8` pins the §4.3.4.3
  "8 samples or more" guard from page 117.

Post-filter (RFC 6716 §4.3.7.1):

* `POST_FILTER_TAPS_F32` / `POST_FILTER_TAPS_Q15` — the three §4.3.7.1
  tap shapes in both f32 and Q15 fixed-point. Each row is
  `[g0, g1, g2]`; the §4.3.7.1 response weights `g1` / `g2` apply to
  the symmetric pair `y(n - T ± k)`, so the total lobe contribution
  is `g0 + 2*g1 + 2*g2 ≈ 1.0` for every tapset.
* `tap_coefficients_f32(tapset)` / `tap_coefficients_q15(tapset)` —
  thin lookup wrappers returning the three taps as a tuple; out-of-
  range tapsets saturate defensively to the last valid entry.
* `gain_f32(gain_index)` / `gain_q15(gain_index)` — reconstruct the
  post-filter gain `G = 3*(gain+1)/32` from the 3-bit raw index.
  Q15 is exact (`3*(gain+1)*1024`); the f32 form covers
  `0.09375..=0.75`.
* `filter_sample_f32(x, history, period, gain, tapset)` — single-
  sample evaluation of the §4.3.7.1 recursion
  `y(n) = x + G * (g0*y(n-T) + g1*(y(n-T+1)+y(n-T-1)) + g2*(y(n-T+2)+y(n-T-2)))`.
  Past samples that lie before `history` are treated as zero (the
  startup-condition reading); the pitch period is silently clamped
  to the §4.3.7.1 minimum 15.
* `apply_post_filter_f32(out, prev_output, period, gain_index, tapset)`
  — in-place slice variant. `prev_output` carries the most-recent
  filtered samples across frame boundaries.
* `PostFilterParams { enabled, period, gain_index, tapset }` +
  `PostFilterParams::OFF` package one frame's §4.3.7.1 parameters (gain
  index `0` is *not* "off" — the `G = 3*(gain+1)/32` formula has no
  zero — so the disabled state is carried explicitly).
* `apply_post_filter_transition_f32(out, prev_output, prev, cur, overlap)`
  — the §4.3.7.1 cross-frame **gain-transition crossfade**. RFC 6716
  §4.3.7.1: "During a transition between different gains, a smooth
  transition is calculated using the square of the MDCT window. It is
  important that values of y(n) be interpolated one at a time such that
  the past value of y(n) used is interpolated." The old (`prev`) and new
  (`cur`) filter lobes are blended `(1 - w²)` / `w²` over the first
  `overlap` samples using the squared §4.3.7 synthesis window
  (`mdct::celt_window_f32`); both lobes read the single already-blended
  past output `y`, so the recursion is interpolated one sample at a
  time. `prev == cur` reduces algebraically to a steady-state
  `apply_post_filter_f32` pass and `OFF→OFF` to passthrough, so
  `decode_celt_frame` routes every mono frame through it
  unconditionally. The transition region length (`= CELT_OVERLAP`) and
  the rising-half window orientation are a documented decoder decision —
  the only assignment under which the unchanged-parameter case stays
  steady-state. Whether the reference additionally interpolates the
  *period* `T` within the region (rather than switching it at the region
  start) is a residual §4.3.7.1 docs question; this implementation
  switches `T` at sample 0 and crossfades only the lobe magnitude.
* `POST_FILTER_PERIOD_MIN = 15`, `POST_FILTER_PERIOD_MAX = 1022`
  pin the §4.3.7.1 "bounded between 15 and 1022, inclusively"
  prose.

De-emphasis (RFC 6716 §4.3.7.2):

* `ALPHA_P_F32 = 0.8500061035` (the spec coefficient) and the Q15
  form `ALPHA_P_Q15 = 27853 = round(α_p · 32768)`.
* `Deemphasis { last_y }` — single-pole IIR state carrying the
  `y(n-1)` sample across calls so the filter runs continuously
  through frame boundaries.
* `Deemphasis::step(x)` per-sample evaluation; `apply_in_place`
  / `apply` for slice processing; `reset()` for a §4.5.2 decoder
  reset.
* `deemphasize_in_place_f32(out, y_prev) -> y_prev_next` —
  one-shot convenience wrapper.

Per-band minimum + trim-offset helpers (RFC 6716 §4.3.3 §2.6, lines
6431–6460):

* `BAND_BINS_LM: [[u32; 21]; 4]` — RFC 6716 §4.3 Table 55 — the
  per-channel MDCT-bin count for every band at every CELT frame
  size (`LM = 0..=3` for 2.5/5/10/20 ms). Each row sums to
  `100 << LM` (the 0–20 kHz coded range); the underlying MDCT
  has `120 << LM` bins per channel.
* `SHORT_FRAME_BAND_BINS: [u32; 21]` — the LM = 0 column of
  `BAND_BINS_LM`, exposed as a stand-alone constant because the
  §2.6 prose explicitly cites "the number of MDCT bins in the
  shortest frame size for this mode" in the `trim_offsets`
  derivation.
* `compute_thresh(channels, bins_per_band, out) -> bool` — the
  §4.3.3 hard-minimum per-band shape allocation, in 1/8 bit units:
  `max((24 * N[band]) / 16, 8 * channels)`. Works over the
  caller-supplied coded-band window so Hybrid mode (bands 17..=20)
  composes the same way as pure CELT (bands 0..=20).
* `compute_trim_offsets(alloc_trim, lm, channels, coding_start,
  bins_per_band, out) -> bool` — the §4.3.3 per-band trim-derived
  offset, in 1/8 bit units, including the width-1 (single-bin-per-
  channel) downward adjustment that diminishes the allocation by
  one bit. The "absolute band index" used to index
  `SHORT_FRAME_BAND_BINS` is `coding_start + i` so the Hybrid
  window picks up the correct LM=0 reference cell.
* `EIGHTH_BIT_QUANTUM = 8`, `NUM_LM = 4` constants pin the unit
  and row count.

MDCT band layout (RFC 6716 §4.3, Table 55):

* `EBAND_EDGES_5MS: [u32; 22]` — the canonical CELT band edges at the
  2.5 ms (LM=0) frame size, in per-channel MDCT bins: the cumulative
  offsets `[0, 1, 2, …, 78, 100]`. `EBAND_EDGES_5MS[band]` is the start
  bin of `band`; `EBAND_EDGES_5MS[band+1]` its exclusive end; the final
  entry `100` is the total LM=0 coded bin count (the 0–20 kHz range; the
  underlying MDCT has `120 << lm` bins/channel, so the 20-bin gap above
  20 kHz is not band-coded). The consecutive differences reproduce the
  Table 55 "2.5 ms / Bins" column
  (`[1,1,1,1,1,1,1,1,2,2,2,2,4,4,4,6,6,8,12,18,22]`).
* `band_edge(band, lm) -> Option<u32>` — the per-channel bin offset of
  `band`'s start at frame-size shift `lm`, `EBAND_EDGES_5MS[band] << lm`.
  Accepts `band == NUM_BANDS` (returns the terminal edge `100 << lm`).
* `band_bins(band, lm) -> Option<u32>` — the Table 55 per-band bin count
  `N[band]` (the edge difference scaled by `1 << lm`); bit-identical to
  `BAND_BINS_LM[lm][band]`.
* `band_bin_range(band, lm) -> Option<(u32, u32)>` — the half-open
  `[start, end)` MDCT-bin range of `band`, with
  `end - start == band_bins(band, lm)`.
* `coded_total_bins(start, end, lm) -> Option<u32>` — the total
  per-channel bin span over the coded-band window `[start, end)`
  (`band_edge(end, lm) - band_edge(start, lm)`). Pure CELT uses
  `start = 0`; Hybrid mode uses `start = 17` (a 60-bin LM=0 span over
  bands `17..21`). Rejects `start > end`, `end > NUM_BANDS`, `lm > 3`.
* `NUM_BAND_EDGES = 22` (= `NUM_BANDS + 1`). The edge-form layout is the
  companion of the count-form `BAND_BINS_LM`; the band-decode walkers
  (§4.3.6 denormalization, §4.3.7 IMDCT input layout, §4.3.4 PVQ band
  walk, §4.3.5 anti-collapse) index a band to its bin range through
  these accessors instead of recomputing the prefix sum per call site.

Static allocation table (RFC 6716 §4.3.3, Table 57):

* `STATIC_ALLOC: [[u8; 11]; 21]` — the CELT Static Allocation Table
  transcribed verbatim from RFC 6716 §4.3.3 (`docs/audio/opus/
  rfc6716-opus.txt` lines 6234–6286). Indexed `STATIC_ALLOC[band][q]`;
  units are 1/32 bit per MDCT bin. `band ∈ 0..21`, `q ∈ 0..11`.
* `NUM_Q = 11`, `INTERP_STEPS = 64` pin the quality-column count and
  the 1/64-step interpolation grid.
* `interp_alloc_1_32nd(band, qlo, frac) -> Option<u32>` returns the
  per-band interpolated 1/32-bit-per-bin coefficient at sub-column
  position `frac ∈ 0..=63` between adjacent quality columns `qlo`
  and `qlo + 1` (RFC 6716 §4.3.3 lines 6227–6229; round-to-nearest
  with a 32-half-bit additive constant).
* `band_static_alloc_1_8th(band, qlo, frac, channels, bins_per_channel,
  lm) -> Option<u32>` folds in the §4.3.3 `channels * N * alloc <<
  LM >> 2` scaling, returning the per-band static allocation in 1/8
  bits. Rejects `channels ∉ {1, 2}` / `lm > 3` and saturates
  defensively against `u32` overflow on the multiply (the legitimate
  range is well inside `u32`).
* `window_static_alloc_1_8th(coding_start, bins_per_band, qlo, frac,
  channels, lm) -> Option<u32>` sums the per-band static allocation
  across a coded-band window for the static-allocation search driver
  (the §4.3.3 "highest allocation that does not exceed the bits
  remaining" inner search composes by calling this for increasing
  `(qlo, frac)` until the budget is exceeded).
* `find_static_alloc(coding_start, bins_per_band, channels, lm,
  budget_1_8th) -> Option<StaticAllocSearch>` runs the §4.3.3 inner
  search. Bisects in two phases — a coarse column scan over
  `q ∈ 0..=NUM_Q-1` for the largest `qlo` whose integer-column
  window total fits the budget, then a fine fractional scan over
  `frac ∈ 0..=INTERP_STEPS` for the highest sub-column position
  whose interpolated window total fits. Returns
  `StaticAllocSearch { qlo, frac, total_1_8th }` with `total_1_8th
  <= budget_1_8th` and (away from saturation) stepping by one grid
  position strictly exceeds the budget. The §4.3.3 "very low rates"
  zero-allocation exit returns
  `StaticAllocSearch { qlo: 0, frac: 0, total_1_8th: 0 }`. The
  canonical upper-column-edge pin (`frac == INTERP_STEPS`) is
  evaluated as `qlo+1` at frac=0 so the interpolated evaluator's
  `frac ∈ 0..INTERP_STEPS` contract stays intact and the top column
  (`qlo == NUM_Q - 1`) is reachable.
* `window_static_alloc_per_band_1_8th(coding_start, bins_per_band,
  qlo, frac, channels, lm, out) -> bool` emits the **per-band**
  breakdown of the static-allocation window at a grid position —
  `channels * N * interp_alloc(band, qlo, frac) << LM >> 2` per band,
  the same cell `band_static_alloc_1_8th` computes. Where
  `find_static_alloc` returns only the scalar window total, this fills
  `out` with the individual per-band 1/8-bit allocations the §4.3.3
  reallocation pass (§2.7 outcome) consumes alongside the minimums
  (`thresh[]`), trim offsets, and caps (`cap[]`). The top-column
  saturated exit (`qlo == NUM_Q-1`, `frac == 0`) is reachable via
  direct integer-column evaluation; a non-zero `frac` there is
  rejected. The emitted vector decomposes the window total exactly
  (`out.iter().sum()` equals `window_static_alloc_1_8th`). Computed
  through a scratch buffer so `out` is untouched on any rejection.

Per-band shape-allocation assembly at a quality column (RFC 6716
§4.3.3 §§2.1–2.3/2.6):

* `combine_band_allocation(coding_start, bins_per_band, qlo, frac,
  boost, alloc_trim, channels, stereo, lm) -> Option<CombinedAllocation>`
  assembles the per-band shape-allocation candidate the §2.7 search
  evaluates at one quality column. For each coded band it computes
  `bits[b] = clamp(static[b] + boost[b] + trim_offset[b], 0, cap[b])`
  in 1/8-bit units: the interpolated Table-57 static allocation (§2.1,
  via `window_static_alloc_per_band_1_8th`) plus the decoded band boost
  (§2.3) plus the `alloc.trim`-derived tilt (§2.6, via
  `compute_trim_offsets`), clamped above by the per-band `cap[]` (§2.2,
  via `compute_band_caps`) and floored at zero. This is the "minimums /
  cap / boost composition in the next stage" the `StaticAllocSearch`
  doc hands off to. The additive chain accumulates in `i64` so it
  cannot overflow before the clamp re-narrows it.
* `CombinedAllocation { bits, caps, total }` carries the per-coded-band
  candidate (`bits`), the `cap[]` used by the clamp (`caps`, kept so the
  deferred §2.7 bisection reuses it), and the window sum (`total`, in
  1/8 bits). `bits[b]` is always in `0..=caps[b]`; `total` equals the
  sum of `bits`.
* Returns `None` on any input-validation failure (window overflowing
  `NUM_BANDS`, `channels ∉ {1,2}`, `lm > 3`, `alloc_trim ∉ 0..=10`,
  `qlo`/`frac` outside the interpolation grid, or a `boost` slice whose
  length does not match the window).
* The §2.7 hard-minimum **skip** decision (comparing each candidate
  against `thresh[]` and zeroing bands that fall short) stays out of
  scope: it is coupled to the reallocation bisection / concurrent skip
  decoding that RFC 6716 §4.3.3 and the clean-room narrative §2.7 defer
  to the reference. `compute_thresh`'s `thresh[]` is carried forward
  into that deferred step unchanged.
* `find_combined_alloc(coding_start, bins_per_band, boost, alloc_trim,
  channels, stereo, lm, budget_1_8th) -> Option<CombinedAllocSearch>`
  takes `find_static_alloc` one step further into the §4.3.3 search:
  where that one finds the highest grid column whose *static-only*
  window total fits, this searches the grid against the **combined**
  (cap-clamped, boost- and trim-inclusive) total — "the entry nearest
  but not exceeding the available space, subject to the tilt, boosts,
  [and] band maximums," before the linear interpolation. Boosts and
  trim offsets are column-independent and `clamp(static + const, 0,
  cap)` is non-decreasing in `static`, so the combined total stays
  monotone along the grid and the same two-phase bisection
  `find_static_alloc` uses applies. `CombinedAllocSearch { qlo, frac,
  alloc }` returns the chosen position and the `CombinedAllocation`
  assembled there, with `alloc.total <= budget_1_8th` whenever the
  budget admits the `(0,0)` cell (otherwise it returns that floor cell,
  whose total is the minimum achievable). Input validation propagates
  from `combine_band_allocation`. The §2.7 hard-minimum **skip**
  decision and the fine-energy/shape split remain deferred (same
  docs-gap boundary).

Pyramid Vector Quantizer (RFC 6716 §4.3.4.2 decode + §5.3.8.1 encode):

* `v_count(n, k)` — the §4.3.4.2 codebook size `V(N, K)` computed
  from the recurrence `V(N, K) = V(N-1, K) + V(N, K-1) +
  V(N-1, K-1)` with base cases `V(N, 0) = 1` and `V(0, K) = 0` for
  `K > 0`. Returns `u32`; saturates to `V_COUNT_SATURATION = u32::MAX`
  when the recurrence would overflow (§4.3.4.4 splits large bands
  before this happens in legal traffic).
* `decode_index_to_pulses(index, n, k)` — runs the §4.3.4.2
  per-position reconstruction loop on a caller-supplied index in
  `[0, V(N, K))`. Returns the signed integer pulse vector
  `X[0..N]` whose absolute-value sum is exactly `K`. Returns `None`
  for out-of-range index, `N = 0` paired with `K > 0`, or a
  saturated codebook size.
* `decode_pulses(dec, n, k)` — composes `dec_uint(V(N, K))`
  (§4.1.5) with `decode_index_to_pulses`. Returns `None` on a
  sticky range-decoder error.
* `normalize_to_unit_l2(pulses)` — divides by the f64 L2 norm so
  the output `f32` vector lies on the unit hypersphere per
  §4.3.4.2 final paragraph; the all-zero input degrades to the
  all-zero output rather than producing a NaN.
* `decode_unit_shape(dec, n, k)` — convenience composition of the
  two above, returning the unit-norm `Vec<f32>` ready to feed the
  §4.3.4.3 spreading rotation.
* `encode_pulses_to_index(pulses, n, k)` — the exact arithmetic
  inverse of `decode_index_to_pulses`: maps a signed integer pulse
  vector `X` with `sum(|X|) = K` back to its unique codeword index
  `i ∈ [0, V(N, K))`, by replaying the decoder's per-position
  half-selection + magnitude walk forward and re-accumulating the
  residual the decoder subtracted from `i`. The §4.3.4.2 codeword↔index
  map is a bijection, so `encode_pulses_to_index ∘ decode_index_to_pulses
  == id` over the whole codeword space. Returns `None` on a length
  mismatch, a magnitude sum other than `K`, an entry exceeding the
  remaining budget, or a saturated `V(N, K)`.
* `pvq_search(x, n, k)` — the encoder-side §5.3.8.1 codeword search:
  quantizes an input vector onto the §4.3.4.2 codebook (every integer
  vector with `sum(|y|) = K`) by projecting `x` onto the `K-1`-pulse
  pyramid with truncate-toward-zero
  (`y0[j] = trunc((K-1)·x[j]/Σ|x|)`), then greedily adding the
  remaining pulses one at a time to maximize the signed normalized
  correlation `xᵀy/||y||` (minimizing §5.3.8.1's `J = -xᵀy/||y||`).
  Each greedy step is constrained to add a pulse (matching-sign /
  zero-entry candidates), never cancel one, so `sum(|y|)` reaches `K`
  exactly. `K = 0` returns the zero vector; `N = 0` with `K > 0` is
  `None`. The RFC names this method and permits any search yielding a
  valid codebook vector, so the implementation is clean-room by
  construction.
* `encode_unit_shape(x, n, k)` — composes `pvq_search` with
  `encode_pulses_to_index`, returning `(index, pulses)` so the caller
  both transmits the codeword index and keeps the quantized integer
  pulse vector. The full encode chain: input vector → integer codeword
  → bitstream index. `decode_index_to_pulses(index) == pulses` always.
* `encode_pulses(enc, pulses, n, k)` — serialises a pulse vector into
  the §5.1 range coder (`encode_pulses_to_index` → `enc_uint(index,
  V(N,K))`), the exact inverse of `decode_pulses`. `encode_shape(enc, x,
  n, k)` chains `pvq_search` with it, returning the quantised pulse
  vector — the full input-vector → range-coded-bitstream → pulse-vector
  PVQ shape encode (encode-side counterpart of `decode_unit_shape`).
* `V_COUNT_SATURATION = u32::MAX` — sentinel for the over-budget
  codebook case (callers that hit it must split per §4.3.4.4
  before retrying).

Bits-to-pulses search and balance accumulator (RFC 6716 §4.3.4.1):

* `cost_log2_v_count_8th(n, k)` — the §4.1.5 `dec_uint(V(N, K))`
  worst-case cost in 1/8-bit units (`ceil(log2(V(N, K))) * 8`).
  `K = 0` is free; saturated codebook sizes return `u32::MAX` so the
  search short-circuits cleanly. This is the closed-form estimator
  the search composes with by default and on the cache-sentinel
  small-band path; the bit-exact cache below supersedes it for every
  band the §4.3.3 allocator reaches.

The bit-exact §4.3.4.1 pulse-cost cache (`pulse_cache` module),
embedding the `cache_index50` / `cache_bits50` tables the RFC
§4.3.4.1 search runs against:

* `CACHE_INDEX50` — 105 `i16` per-(band, LM) offsets in band-major
  order (`CACHE_INDEX50[band*5 + LM]`); `-1` is the closed-form
  sentinel (band 0 at every LM, band 1 at LM 0..2).
* `CACHE_BITS50` — 392 `u8` packed cost runs; each run is a `maxK`
  byte followed by `qbits[1..=maxK]` in 1/8-bit units, monotone in
  `K`. The 23 distinct runs tile the 392 bytes exactly.
* `cache_offset(band, lm)` / `cache_max_k(band, lm)` /
  `cache_cost_8th(band, lm, k)` — the bit-exact lookups (`None` for a
  sentinel or an out-of-range `K`).
* `cached_bits_to_pulses(band, lm, budget_8th) -> CachedPulses` — the
  §4.3.4.1 inner loop returning the largest cached `K` whose cost
  fits the budget.
* `bits_to_pulses_band_loop_cached(lm, band_start, band_n,
  band_target_8th)` — the band walk driving the bit-exact cache for
  cached tuples and the estimator only on sentinels.
  Values from `docs/audio/opus/pulse-cache-format-trace.md` (#118) +
  `tables/cache-{index,bits}50.csv`.
* `bits_to_pulses_search(n, target_8th, cost_fn) -> BitsToPulses` —
  for a single band, picks the largest `K ∈ [0, K_SEARCH_CAP]` whose
  reported cost does not exceed `target_8th`. The §4.3.4.1
  "rounding down if exactly halfway" tie-breaker is implicit: the
  search never advances past the lower of a tied pair because the
  cost-monotone-in-K property keeps it within budget. `n = 0` or
  `target_8th = 0` returns `K = 0` directly.
* `BalanceAccumulator { balance_8th }` — the running 1/8-bit
  balance the §4.3.4.1 band loop maintains. `adjusted_target(raw,
  divisor) = raw + balance / divisor` folds the share into a
  candidate target (round-toward-zero division, the natural i32 `/`
  semantics; saturates non-negatively at zero). `update(raw,
  bits_used)` accumulates `raw - bits_used` into the running
  balance. `reset()` zeros it for a §4.5.2 decoder reset.
* `bits_to_pulses_band_loop(band_n, band_target_8th, cost_fn)`
  orchestrates the §4.3.4.1 walk across a sequence of bands,
  applying the share divisors `DEFAULT_BALANCE_DIVISOR = 3` to
  bands `0..nbands - 2`, `SECOND_TO_LAST_BALANCE_DIVISOR = 2` to
  band `nbands - 2`, and `LAST_BALANCE_DIVISOR = 1` to band
  `nbands - 1` per the §4.3.4.1 prose ("one third / half / whole
  balance"). Returns the per-band `(K, bits_used_8th)` plus the
  final balance.
* `K_SEARCH_CAP = 128` pins a deterministic worst-case on the inner
  loop. The §4.3.4.4 band-splitting machinery keeps the legitimate
  per-band `K` well inside this bound.

PVQ band-split gating and recursion geometry (RFC 6716 §4.3.4.4,
page 118):

* `band_needs_split(n, k) -> bool` returns `true` iff `V(N, K)`
  would not fit in 32 bits — the §4.3.4.4 trigger ("the maximum
  size allowed for codebooks is 32 bits"). `K = 0` and `N = 0` are
  never-split base cases.
* `split_dimensions(n) -> (N_lo, N_hi)` returns the §4.3.4.4
  "two sub-vectors of size N/2" split: `(N/2, N/2)` for even `N`,
  `(N/2, N/2 + 1)` for odd `N`, `(0, 0)` for `N <= 1`. The lower
  index gets the smaller half so the leaf-PVQ walk sees a
  deterministic ordering.
* `max_split_levels(lm) -> u32` returns the §4.3.4.4 recursion
  cap `LM + 1`. `lm` is clamped to `MAX_LM = 3` defensively.
* `BandSplitNode { Leaf { n }, Split { lo, hi } }` is the
  recursive descriptor of how a band decomposes into a tree of
  leaf-PVQ sub-bands. Helpers `total_n` / `leaf_count` / `depth`
  / `for_each_leaf` / `leaf_dims` walk the tree without exposing
  the recursion.
* `plan_band_split(n, k, lm) -> BandSplitNode` descends the
  recursion from the given `(N, K, LM)` halving the band on each
  step until `band_needs_split` is `false`, the depth reaches
  `LM + 1`, or `N` drops to `<= 1`. The quantized split-gain
  parameter the §4.3.4.4 prose mentions to redistribute the L2
  norm across the two halves is a docs gap (the RFC narrative
  defers it to the reference implementation); the geometry tree
  emitted here is wired up to a gain-aware leaf walker without
  re-shaping when the gain decode lands.
* `MAX_LM = 3` pins the canonical CELT frame-size range
  (`LM ∈ {0, 1, 2, 3}` ↔ 2.5/5/10/20 ms frame durations).

Band denormalization (RFC 6716 §4.3.6):

* `log_energy_q8_to_amplitude_f32(log_energy_q8) -> f32` returns the
  per-sample amplitude factor `A = sqrt(2^(E_q8 / 256)) = 2^(E_q8 / 512)`
  in f32, working on the Q8 rendering of the §4.3.2.1 base-2
  log-energy axis (`CoarseEnergyState::energy` carries the normative
  f32 form; multiply by 256 and round to land on this axis).
* `scale_band_f32(shape, amplitude, out)` and
  `scale_band_in_place_f32(samples, amplitude)` multiply each sample by
  a precomputed amplitude; useful when one band is denormalized
  repeatedly across multiple MDCT blocks at the same energy.
* `denormalize_band_f32(shape, log_energy_q8, out)` and
  `denormalize_band_in_place_f32(samples, log_energy_q8)` apply the
  §4.3.6 per-band multiplicative step (`output[i] = shape[i] *
  sqrt(2^(E_q8 / 256))`).
* `denormalize_bands_f32(shapes, log_energies_q8, out)` walks 21 bands
  and concatenates the denormalized shapes into `out`;
  `denormalize_bands_in_place_f32(samples, bins_per_band,
  log_energies_q8)` operates on a single contiguous buffer instead.
* `Q8_DENOM = 256.0` and `SQRT_Q8_DENOM = 512.0` pin the Q8 unit on
  the log-energy axis.

The §4.3.6 prose ("each decoded normalized band is multiplied by the
square root of the decoded energy") is one sentence; the §4.3.2.1 Q8
log-2 representation and the `sqrt(2^E) = 2^(E/2)` identity supply the
arithmetic with no normative source-file delegation.

Final per-band log-energy assembly (RFC 6716 §4.3.2):

* CELT reconstructs the per-band log-2 energy in three additive steps:
  §4.3.2.1 coarse (f32, `1.0` = one integer log-2 step = 6 dB, carried
  in `CoarseEnergyState`), §4.3.2.2 fine (Q14 corrections from
  `decode_fine_energy`), and §4.3.2.2 finalize (Q14 corrections from
  `finalize_extra_bits`). Each correction is *added* to the running
  log-2 energy per the §4.3.2.2 prose ("the correction applied to the
  coarse energy"). The three producers carry the same physical quantity
  at three fixed-point scales (f32 `1.0` / Q14 `16384` / Q8 `256` per
  integer log-2 step); this module performs the additions and the
  rescale onto the Q8 axis the §4.3.6 denormalization indexes.
* `assemble_band_log_energy_f32(coarse, channel, fine_q14,
  finalize_q14) -> Option<[f32; 21]>` returns the final envelope as
  f32. `fine_q14` / `finalize_q14` are `Option` so a caller can skip a
  step (bands stay at the coarse value). Out-of-range `channel` ⇒
  `None`.
* `assemble_band_log_energy_q8(...) -> Option<[i32; 21]>` returns the
  envelope on the Q8 axis: the coarse f32 value is scaled `×256` and
  rounded; the summed fine + finalize Q14 correction is rescaled by a
  round-to-nearest `÷64` (`(corr + 32) >> 6`). Rescaling the
  corrections in integer Q-space keeps the raw-bit-exact fine/finalize
  values off an extra f32 rounding before they reach the Q8 grid.
* `log_energy_f32_to_q8(e) -> i32` exposes the per-value `round(e*256)`
  bridge standalone, for callers that already hold an f32 envelope and
  want to feed individual bands to `decode_band_shape`.
* `FINE_Q14_DENOM = 16384`, `Q14_TO_Q8_SHIFT = 6` pin the Q14 scale and
  the Q14→Q8 rescale shift. The module is pure spec-grounded
  arithmetic: no range-decoder interaction and no new numeric table.

Single-band shape-decode orchestrator (RFC 6716 §4.3.4 → §4.3.6):

* `decode_band_shape(dec, n, k, spread, nb_blocks, tf_adjustment,
  log_energy_q8) -> Option<BandShape>` runs the simplest-case per-band
  decode chain for a band whose `V(N, K)` codebook fits in 32 bits (no
  §4.3.4.4 split required), in §4.3 bitstream order: `decode_unit_shape`
  (§4.3.4.1 bits-to-pulses + §4.3.4.2 codebook index → vector → unit
  norm) → `apply_spread` (§4.3.4.3 spreading rotation) →
  `apply_tf_resolution_change` (§4.3.4.5 TF change, skipped when
  `tf_adjustment == 0`) → `denormalize_band_in_place_f32` (§4.3.6).
  Each of the §4.3.4.3 / §4.3.4.5 passes is L2-norm-preserving, so the
  §4.3.6 step sees the unit-norm shape and scales it to the decoded
  energy.
* `BandShape { samples, k }` carries the denormalized MDCT-domain band
  samples (length `N`, interleaved across `nb_blocks` time blocks) plus
  the consumed pulse count `K`.
* Returns `None` when the PVQ decode fails (saturated codebook ⇒ caller
  must split per §4.3.4.4, `N == 0` with `K > 0`, a sticky range-decoder
  error, or an out-of-range index) or when the spreading / TF shape
  constraints are violated (`nb_blocks == 0`, a band length not
  divisible by `nb_blocks`, or a TF request exceeding the available
  Hadamard levels).
* The §4.3.4.4 split-gain band-split path and the stereo joint-coding
  path remain out of scope for the same docs-gap reason (the precise
  split-gain precision/PDF is deferred to the reference). Callers gate
  on `band_needs_split(n, k)` before invoking this orchestrator.

Multi-band residual decode loop (RFC 6716 §4.3.4 → §4.3.6):

* `decode_residual_bands(dec, lm, start, end, is_transient, tf_select,
  tf_changes, band_k, spread, log_energy_q8) -> Result<ResidualSpectrum,
  Error>` is the residual-section integration spine — the band-loop
  counterpart of `decode_frame_prefix`. It walks the coded-band window
  `[start, end)` in bitstream order, and for each band computes `N`
  (§4.3 Table 55 via `band_bins`), the short-block count (`2^lm` on a
  transient frame, `1` on a long MDCT, per §4.3.1), and the §4.3.4.5 TF
  adjustment (`tf_adjustment`), invokes `decode_band_shape`, and lays
  the band's samples into its `band_bin_range` slot — offset to the
  window origin so a Hybrid (`start = 17`) window indexes its spectrum
  from 0.
* `ResidualSpectrum { samples, band_k }` carries the contiguous
  denormalized MDCT-domain spectrum (length `coded_total_bins(start,
  end, lm)`) and the per-band pulse counts consumed.
* The per-band pulse counts `band_k[]` are an **input** (the §4.3.4.1
  bits-to-pulses output), so the loop stays inside the fully-specified
  §4.3.4 territory and does not depend on the RFC-deferred §4.3.3
  reallocation pass — the same boundary `bits_to_pulses_band_loop`
  already draws.
* A saturated codebook (the §4.3.4.4 split gap), an indivisible block
  count, or an over-large TF request surfaces as
  `Error::NotImplemented` rather than a silent mis-decode;
  length-mismatched per-band slices or an out-of-range window/`lm` are
  `Error::InvalidParameter`. This is the mono / per-channel, non-split
  loop; stereo joint coding and the §4.3.5 anti-collapse pass (which
  follows it) stay out of scope for the same docs-gap reasons.

Frame-prefix decode driver (RFC 6716 §4.3, Table 56):

* `decode_frame_prefix(dec, coarse_state, lm, frame_bytes, stereo,
  start, end) -> Result<FramePrefix, Error>` walks the CELT bitstream
  in Table 56 order from `silence` through the §4.3.3 band-allocation
  fields, chaining the per-symbol decoders (`CeltFrameHeader::
  decode_prefix` → `decode_coarse_energy` → `decode_tf_parameters` →
  `decode_spread` → `compute_band_caps` → `decode_band_boosts` →
  `compute_initial_reservations` → `decode_band_allocation`). The
  reservation/boost budget is threaded between steps so every gate is
  evaluated against the correct intermediate `ec_tell_frac()`: the
  initial reservation walk reads the post-spread `tell_frac`, the
  band-boost loop advances the decoder, and the trim/skip/intensity/
  dual gates are evaluated against the post-boost `tell_frac` and the
  accumulated `total_boost`, exactly as §4.3.3 specifies.
* `FramePrefix { header, tf, spread, caps, reservations, boosts,
  allocation, start, end }` carries every decoded control parameter
  plus the running budget state the §4.3.3 reallocation pass consumes.
  The coarse-energy `state` is mutated in place so the §4.3.2.1
  inter-frame prediction carries across frames.
* The driver leaves the range decoder positioned at the Table 56
  `fine energy` symbol. Everything from there onward — the §4.3.3
  reallocation pass (concurrent skip decoding), the fine-energy vs.
  shape split, the §4.3.4 residual (per-band PVQ) loop, the §4.3.5
  anti-collapse processing, and the finalize step — depends on the
  reallocation bisection and the fine/shape split formula, both of
  which RFC 6716 §4.3.3 and the clean-room narrative §2.7 defer to the
  reference implementation. They remain documented docs gaps; the
  driver stops at that boundary.
* Out-of-range `lm` (`> 3`) or band window (`start > end` /
  `end > 21`) is rejected with `Error::InvalidParameter` before any
  decode. Callers check `RangeDecoder::has_error()` after the call for
  the §4.1.5 corrupt-frame path.

Inverse MDCT and low-overlap window (RFC 6716 §4.3.7):

* `celt_window_f32(n, overlap)` — the §4.3.7 Vorbis-derived window
  coefficient `W(n) = sin((pi/2) * sin((pi/2) * (n + 1/2) / L)^2)`
  (the square applies to the inner sine; the staged
  `docs/audio/opus/tables/window120.csv` / `window240.csv` data
  extractions match this reading to full printed precision and rule
  out the outer-square reading). `build_window_half_f32(overlap)`
  emits the full rising half (the staged tables' layout).
* `build_low_overlap_window_f32(n, overlap)` — the §4.3.7 low-overlap
  construction: zero-pad the basic window and insert ones in the
  middle (`[0 × (N-L)/2 | rise | 1 × (N-L) | fall | 0 × (N-L)/2]`),
  preserving the [PRINCEN86] power complementarity
  `w(i)^2 + w(i+N)^2 = 1` at hop `N`. `N = L` degenerates to the
  basic full-overlap 240-sample window (`BASIC_WINDOW_LEN = 240`,
  `BASIC_WINDOW_HALF = 120`).
* `imdct_naive_f32(spectrum, out)` — the §4.3.7 inverse MDCT, direct
  form: `N` frequency samples in, `2*N` time samples out, scaling by
  the literal §4.3.7 `1/2`. Accumulates in f64.
* `mdct_naive_f32(time, out)` — the forward companion with the `4/N`
  analysis normalization, the unique choice making the `1/2`-scaled
  inverse a unit-gain weighted-overlap-add round trip (§4.3.7 leaves
  the encoder side unconstrained; this exists for the future encoder
  and round-trip validation).
* `MdctSynthesis` — streaming weighted-overlap-add state: per frame,
  IMDCT → synthesis window → overlap-add against the previous frame's
  saved tail → emit `N` samples (`frame`), with `reset()` for the
  §4.5.2 decoder reset. Feeds the §4.3.7.1 post-filter ("output of
  the inverse MDCT (after weighted overlap-add)").

Long-MDCT synthesis spine (RFC 6716 §4.3.6 → §4.3.7):

* `mdct_size(lm)` — the full per-channel MDCT size `120 << lm`, the
  span the §4.3.7 inverse MDCT transforms. The band-coded range tops
  out at `100 << lm` (§4.3 Table 55), so the upper `20 << lm` bins are
  the uncoded high-frequency gap.
* `CELT_OVERLAP` — the fixed 120-sample §4.3.7 overlap (the basic
  240-sample window's rising/falling half).
* `place_residual_spectrum(residual, lm, start, end)` — maps the
  coded-window residual spectrum (the `decode_residual_bands` output,
  length `coded_total_bins(start, end, lm)`) into the full
  `120 << lm`-bin MDCT spectrum at its absolute band-edge offset
  `[band_edge(start), band_edge(end))`, leaving the uncoded low bins
  (below a Hybrid `start`) and the high-frequency gap at zero. This is
  the §4.3.7 inverse-MDCT input.
* `LongMdctSynthesis` — streaming long-MDCT synthesis state for a fixed
  `lm`: `synthesize(residual, start, end)` places the residual spectrum
  and runs `MdctSynthesis::frame` with the fixed-overlap §4.3.7 window,
  emitting `mdct_size(lm)` time-domain samples (the §4.3.7.1
  post-filter's input), with `reset()` for the §4.5.2 decoder reset.
  This is the residual-spectrum → time-domain PCM counterpart of the
  `decode_residual_bands` band-loop spine. It handles the non-transient
  (single long MDCT) case only — the transient short-block reassembly
  (the per-short-block frequency-vector layout and inter-block
  overlap-add) is delegated to the reference by §4.3.1 / §4.3.7 and
  remains a documented docs gap, the same boundary the residual loop
  keeps for the short-block geometry.
* `StereoLongMdctSynthesis` — the two-channel counterpart: two
  independent per-channel `LongMdctSynthesis` spines, each with its own
  §4.3.7 overlap tail (there is no cross-channel state in the synthesis
  stage). `synthesize(left_residual, right_residual, start, end)` places
  both channels' denormalized spectra, runs the per-channel inverse MDCT
  + weighted overlap-add, and interleaves the time-domain outputs into a
  single L/R/L/R buffer of `2 * mdct_size(lm)` samples; both overlap
  tails advance atomically (a length mismatch on either channel rejects
  the frame without advancing either tail), with `reset()` zeroing both.
  `StereoChannel { Left, Right }` names the two interleave slots
  (`offset()` = 0/1). The §4.3.6 denormalization and §4.3.7 inverse MDCT
  are per-channel and fully specified; only the §4.3.4.4 `itheta` mid/
  side band coupling that produces the two channel spectra from the
  bitstream is the documented docs gap, so this spine takes the two
  channel spectra as input (the same boundary the mono spine draws).

End-to-end frame decode → PCM (RFC 6716 §4.3, Table 56 → §4.3.7):

* `decode_celt_frame(state, frame_bytes, start, end, fine_bits,
  band_k) -> Result<DecodedFrame, Error>` is the top-level driver for a
  **mono, non-transient (single long MDCT)** CELT frame. It walks the
  documented decode chain end-to-end: `decode_frame_prefix` (Table 56
  prefix) → `decode_fine_energy` (§4.3.2.2) → `assemble_band_log_energy_q8`
  (§4.3.2 envelope, sliced to the coded window) → `decode_residual_bands`
  (§4.3.4 shape) → `LongMdctSynthesis::synthesize` (§4.3.6/§4.3.7) →
  `apply_post_filter_transition_f32` (§4.3.7.1, with the squared-window
  gain-transition crossfade against the previous frame's parameters) →
  `Deemphasis::apply_in_place` (§4.3.7.2), returning the `120 << lm`
  PCM samples plus the decoded `FramePrefix`. The streaming
  `CeltDecodeState` carries the previous frame's `PostFilterParams` (for
  the §4.3.7.1 transition) alongside the post-filter output history.
* `CeltDecodeState::new(lm) -> Option<Self>` builds the streaming state
  for a fixed frame-size shift; it carries the §4.3.2.1 coarse-energy
  prediction, the §4.3.7 synthesis overlap tail, the §4.3.7.1
  post-filter history, and the §4.3.7.2 de-emphasis memory across
  frames. `reset()` zeroes every carried memory for a §4.5.2 decoder
  reset; `frame_size()` / `lm()` / `coarse_energy()` expose the
  geometry and state.
* `DecodedFrame { pcm, prefix }` carries the final PCM and the decoded
  control prefix.
* The per-band pulse counts (`band_k`, the §4.3.4.1 bits-to-pulses
  output) and the per-band fine-bit counts (`fine_bits`, the §4.3.2.2
  allocation) are **inputs**: the §4.3.3 reallocation pass that
  produces them is RFC-deferred (the same boundary `decode_residual_bands`
  and `bits_to_pulses_band_loop` keep). When that pass lands, the only
  change here is to compute these two vectors from the `FramePrefix`.
* A `transient` or stereo frame is rejected with
  `Error::NotImplemented` (the §4.3.1/§4.3.7 short-block reassembly and
  the §4.3.4.4 stereo-angle gaps); a saturated codebook surfaces the
  §4.3.4.4 split gap the same way. A `silence`-flagged frame decodes to
  all-zero PCM.

The §4.3.5 anti-collapse processing is a documented docs gap: the RFC
§4.3.5 narrative describes the intent ("a pseudo-random signal is
inserted with an energy corresponding to the minimum energy over the
two previous frames; a renormalization step is then required") but
gives no collapse-detection threshold, pseudo-random generator, or
injection magnitude — those live in `bands.c::anti_collapse()`, outside
the staged docs. Since `decode_celt_frame` only handles non-transient
frames (where §4.3.5 does not even decode the anti-collapse bit), the
gap does not block the mono long-MDCT path.

The encoder and codec-registration entry points with the runtime still
return `Error::NotImplemented`.

## Clean-room provenance

The implementation references only the IETF specifications under
`docs/audio/opus/`:

* RFC 6716 — Definition of the Opus Audio Codec (CELT layer + range
  coder + MDCT path).
* RFC 8251 — Opus Update.
* RFC 7845 — Ogg Encapsulation for Opus (consulted for framing).

RFC 6716's Appendix A reference listing is embedded in the RFC's own
text (a base64 tarball extracted per §A.1, SHA-1-verified against the
value §A.1 prints) and is therefore part of the staged spec; the
§4.3.2.1 coarse-energy path cites it as "RFC 6716 Appendix A
`<file>`". Any source outside the staged RFC text — including any
external distribution of the same reference code — sits outside the
workspace clean-room allow-list and was not consulted. Black-box
invocations of a reference command-line encoder/decoder are allowed as
opaque validators only.

## License

MIT. See `LICENSE`.
