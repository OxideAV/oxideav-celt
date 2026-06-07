# oxideav-celt

Pure-Rust CELT (the MDCT path of Opus, RFC 6716).

## Status — 2026-06-07

**Round-19.** The bit-exact CELT/SILK range decoder (RFC 6716 §4.1),
the CELT frame-header prefix (RFC 6716 §4.3, the scalar fields that
precede band-decode), the §4.3.2.1 coarse-energy scaffolding (21-band
layout + intra prediction filter), the §4.3.2.2 fine-energy
refinement decoder + finalize step, the §4.3.3 bit-allocation scalar
fields (alloc.trim, skip, intensity-band, dual-stereo), the §4.3.3
stereo reservation helpers (`LOG2_FRAC_TABLE` lookup + `intensity_rsv`
+ `reserve_stereo`), the §4.3.3 per-band `cap[]` machinery (the full
168-entry `CACHE_CAPS50` table + `compute_band_caps`) and the §4.3.3
band-boost dynalloc-logp loop (`decode_band_boosts` →
`BoostResult { boost, total_boost, total_bits_remaining }`), the
§4.3.3 initial-reservations budget walk
(`compute_initial_reservations` chains `total_initial` init,
anti-collapse, skip, intensity, and dual-stereo reservations into
one call and emits an `InitialReservations` carrier ready to feed
the band-boost loop and the band-allocation orchestrator), the
§4.3.4.5 time-frequency change parameters (per-band `tf_change` +
the gated global `tf_select` + the four tabulated TF-adjustment
tables 60–63), the §4.3.4.5 Hadamard transform primitives
(orthonormal radix-2 WHT in natural and sequency order + the
`apply_tf_resolution_change` orchestrator), the §4.3.4.3 spreading
parameter (PDF `{7, 2, 21, 2}/32` + Table 59 `f_r` lookup + closed-
form rotation-gain helpers), the §4.3.7.1 post-filter (three tap
shapes in f32 + Q15 + gain reconstruction + per-sample / slice
filter response), and the §4.3.7.2 single-pole de-emphasis filter
(`α_p = 0.8500061035`, in both f32 and Q15) are now wired up. The
Laplace decoder + `e_prob_model` table remain queued for a future
round (numeric CSV is now staged at
`docs/audio/celt/tables/e_prob_model.csv` so the work is no longer
docs-gap-blocked). The §4.3.3 §2.6 per-band hard-minimum shape
allocation (`compute_thresh`) and the per-band `trim_offsets[]`
derivation (`compute_trim_offsets`) are now wired up, alongside the
full §4.3 Table 55 MDCT-bin layout (`BAND_BINS_LM` for all four LM
values + `SHORT_FRAME_BAND_BINS` for the LM=0 column the §2.6 prose
cites), and the §4.3.3 Table 57 static-allocation matrix
(`STATIC_ALLOC[band][q]`, 21 × 11) with the §4.3.3 per-band evaluator
`band_static_alloc_1_8th` (the `channels * N * alloc << LM >> 2`
formula folded with the 1/64-step linear interpolation between
adjacent quality columns) and the `window_static_alloc_1_8th` window
sum used by the static-allocation search driver, and the §4.3.3
inner static-allocation search (`find_static_alloc` →
`StaticAllocSearch { qlo, frac, total_1_8th }` — bisects the 1/64-
step interpolation grid for the highest grid position whose window
total fits the remaining 1/8-bit budget) are now wired up.
The §4.3.3 inner static-allocation search (`find_static_alloc` →
`StaticAllocSearch { qlo, frac, total_1_8th }`) bisects the 1/64-
step interpolation grid for the highest `(qlo, frac)` whose window
total in 1/8 bits does not exceed the supplied "remaining" budget.
The §4.3.4.2 Pyramid Vector Quantizer codebook size `V(N, K)` and
per-band shape decoder (`v_count` + `decode_index_to_pulses` +
`decode_pulses` + `normalize_to_unit_l2` + `decode_unit_shape`)
reproduce the §4.3.4.2 recurrence `V(N, K) = V(N-1, K) + V(N, K-1)
+ V(N-1, K-1)` and the per-position reconstruction loop, then
normalise the decoded integer pulse vector to unit L2 norm so the
§4.3.4.3 spreading rotation can consume it directly. The §4.3.4.1
bits-to-pulses search + balance accumulator
(`bits_to_pulses_search` + `BalanceAccumulator` +
`bits_to_pulses_band_loop`) reproduce the §4.3.4.1 K-search rule
("nearest to target, not exceeding it; ties round down") with a
caller-supplied cost-of-(N, K) closure, and chain it across a band
sequence with the §4.3.4.1 share divisors 3 / 2 / 1 for the
general / second-to-last / last band; the running 1/8-bit balance
absorbs the residue. The §4.3.4.3 spreading rotation chain
(`apply_spread`, `apply_nd_rotation`, `apply_nd_rotation_multi_block`,
`apply_pre_rotation`, `apply_2d_rotation`, `rotation_angle_f64`)
applies the `theta = pi * g_r^2 / 4` N-D forward + reverse 2-D
rotation chain to a unit-norm PVQ shape vector, with per-time-block
independence and the `(pi/2 - theta)` interleaved pre-rotation at
stride `round(sqrt(N/nb_blocks))` when each time block spans at
least 8 samples. The reallocation loop (concurrent skip decoding),
the fine-energy / shape split, and the MDCT machinery still come
later.

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

Coarse-energy scaffold (RFC 6716 §4.3.2.1):

* `NUM_BANDS = 21` (RFC Table 55). Pure CELT codes all 21 bands;
  Hybrid mode codes only bands 17..=20 (the SILK layer covers
  0..=16 below 8 kHz).
* `INTRA_ALPHA_Q15 = 0`, `INTRA_BETA_Q15 = 4915` (the literal Q15
  coefficients RFC line 6063 supplies for intra-mode prediction;
  `4915 / 32768 ≈ 0.1500`).
* `CoarseEnergyState::new()` holds the previous frame's per-band
  Q8 log-energies, zero on stream open and after any decoder reset.
* `apply_intra_prediction(errors_q8, out_q8)` runs the §4.3.2.1
  prediction filter in its intra reduction — the β-IIR over bands
  with the temporal arm vanishing.
* `decode_coarse_energy(dec, state, intra, lm)` is the locked-in
  public entry point; it currently returns `Error::NotImplemented`
  because the `e_prob_model` table and the `ec_laplace_decode`
  algorithm are docs-gap-blocked (the RFC delegates them to a
  source file outside the workspace clean-room allow-list). The
  gap'd path is asserted not to disturb the range decoder or the
  carried state so that future rounds compose cleanly.

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
  in raw-value order (`spread = 0..=3`).
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

Pyramid Vector Quantizer (RFC 6716 §4.3.4.2):

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
* `V_COUNT_SATURATION = u32::MAX` — sentinel for the over-budget
  codebook case (callers that hit it must split per §4.3.4.4
  before retrying).

Bits-to-pulses search and balance accumulator (RFC 6716 §4.3.4.1):

* `cost_log2_v_count_8th(n, k)` — the §4.1.5 `dec_uint(V(N, K))`
  worst-case cost in 1/8-bit units (`ceil(log2(V(N, K))) * 8`).
  `K = 0` is free; saturated codebook sizes return `u32::MAX` so the
  search short-circuits cleanly. This is the closed-form estimator
  the search composes with by default; the bit-exact per-(N, K) cost
  cache the §4.3.4.1 reference uses is delegated to a source file
  outside the workspace clean-room allow-list, so this round ships
  the spec-grounded shape against the worst-case estimator.
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
  loop. The §4.3.4.4 band-splitting machinery (a future round) keeps
  the legitimate per-band `K` well inside this bound.

Higher-level entry points (frame decoder, encoder, codec
registration with the runtime) still return `Error::NotImplemented`.

## Clean-room provenance

The implementation references only the IETF specifications under
`docs/audio/opus/`:

* RFC 6716 — Definition of the Opus Audio Codec (CELT layer + range
  coder + MDCT path).
* RFC 8251 — Opus Update.
* RFC 7845 — Ogg Encapsulation for Opus (consulted for framing).

Source files the RFC delegates to for normative numeric tables and
algorithms sit outside the workspace clean-room allow-list and were
not consulted. Black-box invocations of `opusdec` / `opusenc` are
allowed as opaque validators only.

## License

MIT. See `LICENSE`.
