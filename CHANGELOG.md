# Changelog

All notable changes to `oxideav-celt` are recorded here.

## [Unreleased]

### Added

* **Round-24 §4.3.7 inverse MDCT + low-overlap window (2026-06-11):**
  the MDCT-machinery tail of the decode chain. RFC 6716 §4.3.7 states
  the inverse MDCT "has no special characteristics" — `N` frequency
  samples in, `2*N` time samples out, scaling by `1/2` — and gives the
  basic (full-overlap) 240-sample Vorbis-derived window formula plus
  the low-overlap construction recipe ("zero-padding the basic window
  and inserting ones in the middle, such that the resulting window
  still satisfies power complementarity").

  `celt_window_f32(n, overlap)` evaluates
  `W(n) = sin((pi/2) * sin((pi/2) * (n + 1/2) / L)^2)`. The RFC's
  ASCII art leaves the placement of the square ambiguous; the staged
  data-only extractions `docs/audio/opus/tables/window120.csv` /
  `window240.csv` match the inner-square (Vorbis power-of-sine)
  reading to full printed precision at every sampled position and do
  not match the outer-square reading anywhere, so the inner-square
  reading is pinned. `build_window_half_f32(overlap)` emits the rising
  half (the staged tables' layout); `build_low_overlap_window_f32(n,
  overlap)` builds the `2*N`-sample synthesis window
  `[0 × (N-L)/2 | rise | 1 × (N-L) | fall | 0 × (N-L)/2]`, which
  satisfies the [PRINCEN86] power complementarity
  `w(i)^2 + w(i+N)^2 = 1` exactly at hop `N` and degenerates to the
  basic 240-sample window at `N = L` (`BASIC_WINDOW_LEN = 240`,
  `BASIC_WINDOW_HALF = 120`).

  `imdct_naive_f32(spectrum, out)` is the direct-form `O(N^2)` inverse
  with the literal §4.3.7 `1/2` scaling
  (`y(n) = (1/2) Σ_k X(k) cos((pi/N)(n + 1/2 + N/2)(k + 1/2))`, f64
  accumulation); `mdct_naive_f32(time, out)` is the forward companion
  with the `4/N` analysis normalization — the unique choice making the
  `1/2`-scaled inverse a unit-gain weighted-overlap-add round trip
  (§4.3.7 leaves the analysis side unconstrained; documented as a
  decoder-crate convention). `MdctSynthesis` is the streaming
  weighted-overlap-add state (§4.3.7.1 "output of the inverse MDCT
  (after weighted overlap-add)"): per `frame()`, IMDCT → synthesis
  window → overlap-add against the previous frame's saved tail → emit
  `N` samples; `reset()` zeroes the tail for the §4.5.2 decoder reset.
  Exposed at the crate root: `celt_window_f32`,
  `build_window_half_f32`, `build_low_overlap_window_f32`,
  `imdct_naive_f32`, `mdct_naive_f32`, `MdctSynthesis`,
  `BASIC_WINDOW_LEN`, `BASIC_WINDOW_HALF`.

  17 new unit tests pin: the window formula against five sampled
  staged `window120.csv` values (`n = 0, 29, 59, 89, 119`) and five
  sampled `window240.csv` values (`n = 0, 59, 119, 179, 239`) at
  `1e-7`; strict monotonicity of the rising half within `(0, 1]`;
  exact rising-half power complementarity
  (`W(n)^2 + W(L-1-n)^2 = 1`); the low-overlap segment structure at
  `(N=480, L=120)` (180 zeros, rise, 360 ones, fall, 180 zeros);
  degeneration to the basic 240-sample window at full overlap;
  hop-`N` power complementarity for `N ∈ {120, 240, 480, 960}` at
  `L = 120`; geometry rejection (`N = 0`, `L = 0`, `L > N`, odd
  padding); IMDCT/MDCT length-contract rejection; zero spectrum ⇒
  zero output; IMDCT of a basis vector equals the closed-form
  `1/2`-scaled cosine; IMDCT linearity; windowed-overlap-add perfect
  reconstruction (analysis window → MDCT → IMDCT → synthesis window →
  overlap-add, steady-state ≤ 1e-4) for both the full-overlap
  `N = 120` and the low-overlap `(N=240, L=120)` geometries; streaming
  `MdctSynthesis` equality with the offline overlap-add; and shape
  rejection + `reset()` semantics on the streaming state.

  Pure clean-room: the window formula, basic length, low-overlap
  recipe, output length, and inverse scaling are RFC 6716 §4.3.7
  narrative; the MDCT is a public textbook transform; the staged
  window CSVs are data-only extractions used as validators. Lib test
  count 384 → 401 (+17).

* **Round-23 §4.3.4 → §4.3.6 single-band shape-decode orchestrator
  (2026-06-10):** the simplest-case per-band decode chain — for a band
  whose `V(N, K)` codebook fits in 32 bits, so no §4.3.4.4 split is
  required — wired up as a fixed composition of the RFC-grounded
  primitives the prior rounds built. RFC 6716 §4.3.4 (lines 6462–6474)
  describes the chain: bits → pulses (§4.3.4.1) → codebook size
  (§4.3.4.2) → uniform index → vector → unit norm. The unit-norm shape
  is then transformed in §4.3 bitstream order by the §4.3.4.3 spreading
  rotation, the §4.3.4.5 time-frequency resolution change, and the
  §4.3.6 denormalization. `decode_band_shape(dec, n, k, spread,
  nb_blocks, tf_adjustment, log_energy_q8) -> Option<BandShape>` runs
  the chain: `decode_unit_shape` (§4.3.4.1/§4.3.4.2) → `apply_spread`
  (§4.3.4.3) → `apply_tf_resolution_change` (§4.3.4.5, skipped when
  `tf_adjustment == 0`) → `denormalize_band_in_place_f32` (§4.3.6).
  `BandShape { samples, k }` carries the denormalized MDCT-domain band
  samples plus the consumed pulse count `K`. `None` is returned when the
  PVQ decode fails (saturated codebook ⇒ caller must split per §4.3.4.4,
  `N == 0` with `K > 0`, a sticky range-decoder error, or an
  out-of-range index) or when the spreading / TF shape constraints are
  violated (`nb_blocks == 0`, a band length not divisible by
  `nb_blocks`, or a TF request exceeding the available Hadamard levels).
  The §4.3.4.4 split-gain band-split path and the stereo joint-coding
  path remain out of scope for the same docs-gap reason flagged in
  round 20 (the §4.3.4.4 prose defers the split-gain precision/PDF to
  the reference). Exposed at the crate root: `decode_band_shape`,
  `BandShape`.

  13 new unit tests pin: a `0xFF…`-seeded decode at `(N=4, K=3)`,
  no-spread / no-TF, energy `E_q8 = 512` (amplitude 2.0) yields a band
  whose L2 norm equals the amplitude (the §4.3.4.3/§4.3.4.5 passes
  preserve the unit norm, §4.3.6 scales it); zero energy (`E_q8 = 0`)
  yields a unit-norm band; an aggressive spread preserves the band
  energy; a `−1` TF resolution change (2 blocks × 4 samples) preserves
  the norm; a `+1` TF change (4 blocks × 2 samples,
  frequency-resolution-increase branch) preserves the norm; `N == 0`
  with `K > 0` returns `None`; a saturated codebook (`V(180, 180)`)
  returns `None` (caller must split); `nb_blocks == 0` returns `None`;
  an indivisible band length (`N = 6`, `nb_blocks = 4`) returns `None`;
  an oversized TF request (`tf = −2` on 2-sample sub-vectors) returns
  `None`; negative energy (`E_q8 = −512`, amplitude 0.25) attenuates the
  band to the sub-unity amplitude; `BandShape::k` round-trips the
  supplied pulse count; `K == 0` decodes the all-zero band (no
  pseudo-random fill — that is the separate gap'd §4.3.5 anti-collapse
  stage) and consumes no PVQ index (`V(N, 0) = 1`, `tell()` unchanged).

  Pure clean-room composition. Every step is an RFC 6716-grounded
  primitive from a prior round; the chain ordering follows the §4.3
  bitstream / decode order. No external library source consulted; no new
  numeric constant introduced. Lib test count 371 → 384 (+13).

* **Round-22 §4.3.6 band denormalization (2026-06-09):** the §4.3.6
  multiplicative pass that scales each PVQ-decoded unit-norm shape
  vector by `sqrt(2^(E_q8 / 256))` so the inverse MDCT consumes
  energy-correct MDCT-domain samples. RFC 6716 §4.3.6 specifies the
  step in a single prose sentence ("each decoded normalized band is
  multiplied by the square root of the decoded energy") and delegates
  the implementation entry point name only; the arithmetic follows from
  the §4.3.2.1 Q8 base-2 log-energy representation
  ([`CoarseEnergyState::prev_q8`]) plus the elementary identity
  `sqrt(2^E) = 2^(E/2)`. `log_energy_q8_to_amplitude_f32(log_energy_q8)`
  returns the per-sample amplitude factor (`2^(E_q8 / 512)`);
  `scale_band_f32` / `scale_band_in_place_f32` apply a precomputed
  amplitude; `denormalize_band_f32` / `denormalize_band_in_place_f32`
  combine the two for the per-band §4.3.6 step;
  `denormalize_bands_f32(shapes, log_energies_q8, out)` walks the full
  21-band envelope into a concatenated output buffer and
  `denormalize_bands_in_place_f32(samples, bins_per_band,
  log_energies_q8)` is the contiguous-buffer in-place variant.
  Constants `Q8_DENOM = 256.0`, `SQRT_Q8_DENOM = 512.0` pin the Q8
  fractional-bit count and the sqrt-amplitude divisor. Exposed at the
  crate root: `log_energy_q8_to_amplitude_f32`, `scale_band_f32`,
  `scale_band_in_place_f32`, `denormalize_band_f32`,
  `denormalize_band_in_place_f32`, `denormalize_bands_f32`,
  `denormalize_bands_in_place_f32`, `Q8_DENOM`, `SQRT_Q8_DENOM`.

  22 new unit tests pin: the Q8 unit constants (`Q8_DENOM = 256`,
  `SQRT_Q8_DENOM = 512 = 2 * Q8_DENOM`); `E_q8 = 0` yields amplitude
  1.0 (unit-energy identity); a single Q8 integer log-2 step scales
  amplitude by `sqrt(2)`; two integer log-2 steps double the
  amplitude; symmetric negative two-step halves it; multiplicative
  composition `A(E1+E2) = A(E1)*A(E2)` across a 7×5 grid;
  `scale_band_f32` scales each sample and rejects length mismatch
  while leaving `out` untouched; `scale_band_in_place_f32` matches
  the copying variant on the same input; per-band energy preservation
  invariant — the squared L2 norm of a denormalized unit-shape band
  equals the linear energy `2^(E_q8/256)` — across a 9-point
  energy sweep; `denormalize_band_f32` zero-energy identity;
  `denormalize_band_in_place_f32` agreement with the copying form;
  `denormalize_band_f32` length-mismatch rejection;
  `denormalize_bands_f32` 21-band walk in band order;
  `denormalize_bands_f32` length-mismatch rejections (wrong shape
  count, wrong output length); `denormalize_bands_in_place_f32` on
  the canonical LM=0 layout (sum of per-band bin counts =
  `100 << 0 = 100`) preserves the per-band energy invariant after
  scaling each unit-norm band by amplitude `sqrt(2)`;
  length-mismatch rejection leaves the input buffer untouched;
  band-ordering preservation under alternating energies; empty bands
  pass through without advancing the output cursor (the §4.3.4.4
  band-split leaf case); all-zero per-band energies leave the
  concatenated shape buffer unchanged.

  Pure clean-room derivation. The §4.3.6 prose is a single
  sentence; the arithmetic is elementary. No external library source
  consulted. Lib test count 349 → 371 (+22).

* **Round-21 §4.3.2.1 `e_prob_model` Laplace-parameter table
  (2026-06-08):** the 4 × 2 × 21 = 168-pair coarse-energy Laplace
  probability model, transcribed verbatim from
  `docs/audio/celt/tables/e_prob_model.csv` (staged 2026-06 from the
  IETF clean-room corpus, structural narrative at
  `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §1.2).
  Exposed at the crate root as `E_PROB_MODEL[lm][intra][band]
  -> ProbDecay { prob, decay }` with the matching `prob_decay(lm,
  intra, band) -> Option<ProbDecay>` accessor that folds the
  `bool intra` flag onto the staged CSV's `0 = inter / 1 = intra`
  middle axis. Both `prob` and `decay` are Q8 unsigned bytes per the
  staged `e_prob_model.meta`. Constants `NUM_LM_FRAME_SIZES = 4`,
  `NUM_PREDICTION_TYPES = 2`, `PRED_INTER = 0`, `PRED_INTRA = 1`
  formalise the table's three axes. Eight unit tests pin the shape
  (`4 * 2 * 21 = 168` pairs), the four corner cells against the CSV,
  the accessor's `bool -> axis-index` mapping, the out-of-range
  `None` discipline, the absence of zero `prob` / zero `decay`
  entries (the staged CSV's smallest values, `prob = 21` and
  `decay = 6`, are pinned), and a `LM`-axis-shape regression
  sentinel. The `ec_laplace_decode` algorithm itself remains queued
  for a future round (the RFC 6716 §4.3.2.1 narrative gives only the
  Laplace-distribution shape, not the per-symbol decode recurrence).

* **Round-20 §4.3.4.4 PVQ band-split gating + recursion geometry
  (2026-06-07):** the §4.3.4.4 trigger ("maximum codebook size 32
  bits") and the recursive halving tree the higher-level band-decode
  walker traverses to reach leaf-PVQ sub-bands.
  `band_needs_split(n, k) -> bool` returns `true` iff `V(N, K)` would
  not fit in 32 bits (reusing `pvq::v_count`'s saturation flag);
  `split_dimensions(n) -> (N_lo, N_hi)` halves the band per the
  §4.3.4.4 "two sub-vectors of size N/2" rule, with the smaller half
  on the low index when `N` is odd; `max_split_levels(lm)` pins the
  `LM + 1` recursion cap with a defensive clamp at `MAX_LM = 3`;
  `BandSplitNode { Leaf { n }, Split { lo, hi } }` is the recursive
  tree descriptor with `total_n` / `leaf_count` / `depth` /
  `for_each_leaf` / `leaf_dims` walkers; `plan_band_split(n, k, lm)`
  descends until the leaf codebook fits in 32 bits or the
  `LM + 1` cap is hit. The quantized split-gain parameter that
  redistributes the relative L2 norm across the two halves is queued
  as a docs gap (the RFC 6716 §4.3.4.4 prose defers the precise
  precision/PDF to the reference). Exposed at the crate root:
  `band_needs_split`, `split_dimensions`, `max_split_levels`,
  `plan_band_split`, `BandSplitNode`, `MAX_LM`.

* **Round-19 §4.3.4.3 spreading rotation chain (2026-06-07):** the
  §4.3.4.3 N-D rotation by `theta = pi * g_r^2 / 4` applied to a
  unit-norm PVQ shape vector, with the `(pi/2 - theta)` pre-rotation
  for blocks of 8+ samples. `rotation_angle_f64(spread, n, k) -> f64`
  computes the angle from `g_r = N / (N + f_r * K)` with `Spread::None`
  collapsing to zero; `apply_2d_rotation(x_i, x_j, cos_theta,
  sin_theta) -> (f32, f32)` is the single 2-D `R(i, j)` primitive
  (`x_i' = cos*x_i + sin*x_j`, `x_j' = -sin*x_i + cos*x_j`);
  `apply_nd_rotation(samples, theta)` walks the §4.3.4.3 forward chain
  `R(x_0, x_1), R(x_1, x_2), ..., R(x_{n-2}, x_{n-1})` then the
  mirrored reverse chain "back and forth" per the spec ordering;
  `apply_nd_rotation_multi_block(samples, nb_blocks, theta)` runs the
  chain independently on each time block laid out interleaved as
  `samples[b + nb_blocks * i]` (the layout the §4.3.4.5 Hadamard
  routines already use); `apply_pre_rotation(samples, nb_blocks,
  theta)` gates on `nb_blocks > 1` AND per-block sample count ≥ 8,
  then applies the `(pi/2 - theta)` rotation to each stride-
  interleaved sub-sequence `S_k = { stride * n + k }` with
  `stride = round(sqrt(N/nb_blocks))` per the §4.3.4.3 "extra
  rotation" rule; `apply_spread(spread, samples, k, nb_blocks)` is
  the full §4.3.4.3 orchestrator (`Spread::None` is a no-op; L2 norm
  preservation is verified across the pass). Exposed at the crate
  root: `apply_2d_rotation`, `apply_nd_rotation`,
  `apply_nd_rotation_multi_block`, `apply_pre_rotation`,
  `apply_spread`, `rotation_angle_f64`,
  `EXTRA_ROTATION_MIN_BLOCK_SAMPLES`.

* **Round-18 §4.3.4.1 bits-to-pulses search + balance accumulator
  (2026-06-05):** the §4.3.4.1 per-band `K` search and the running
  1/8-bit balance accumulator that adjusts subsequent bands'
  targets. `bits_to_pulses_search(n, target_8th, cost_fn) ->
  BitsToPulses` picks the largest `K ∈ [0, K_SEARCH_CAP]` whose
  reported `cost_fn(n, k)` does not exceed `target_8th`. The
  §4.3.4.1 "rounding down on tie" rule is enforced implicitly by
  the cost-monotone-in-K property — the search advances upward and
  stops at the last `K` within budget. `BalanceAccumulator`
  maintains the running 1/8-bit balance; `adjusted_target(raw,
  divisor)` folds the share into a candidate target with
  round-toward-zero division and a non-negative clamp;
  `update(raw, bits_used)` adds the residue into the running
  balance. `bits_to_pulses_band_loop(band_n, band_target_8th,
  cost_fn) -> (Vec<BitsToPulses>, BalanceAccumulator)` chains the
  search across a band sequence with the §4.3.4.1 share divisors:
  `DEFAULT_BALANCE_DIVISOR = 3` for the general case,
  `SECOND_TO_LAST_BALANCE_DIVISOR = 2` for band `nbands - 2`, and
  `LAST_BALANCE_DIVISOR = 1` (whole balance) for the last band.
  `cost_log2_v_count_8th(n, k)` is the default closed-form cost
  estimator: the §4.1.5 `dec_uint(V(N, K))` worst-case cost in 1/8
  bits (`ceil(log2(V(N, K))) * 8`), with `K = 0` free and saturated
  codebooks returning `u32::MAX`. The cost-of-(N, K) function is
  decoupled and supplied by the caller, so the same orchestrator
  runs unchanged against the bit-exact per-(N, K) cost cache when
  the docs gap on that table closes. `K_SEARCH_CAP = 128` pins a
  deterministic worst-case on the inner loop. Exposed at the crate
  root: `bits_to_pulses_search`, `bits_to_pulses_band_loop`,
  `cost_log2_v_count_8th`, `BalanceAccumulator`, `BitsToPulses`,
  `DEFAULT_BALANCE_DIVISOR`, `SECOND_TO_LAST_BALANCE_DIVISOR`,
  `LAST_BALANCE_DIVISOR`, `EIGHTH_BITS_PER_BIT`, `K_SEARCH_CAP`.

  30 new unit tests pin: `V(N, 0)` is free at every N; hand-
  computed worst-case costs at `(N, K) ∈ {(1, 1), (2, 1), (2, 2),
  (3, 2)}` match the `ceil(log2(V)) * 8` estimator; cost is
  monotonically non-decreasing in K across a 8 × 12 grid (the
  property the search relies on); search at zero target picks
  `K = 0`; search at `N = 0` returns `K = 0` for every target;
  search at `N = 1`, target 8 walks the cost plateau (V(1, K) = 2
  for all K ≥ 1) up to `K_SEARCH_CAP`; search at sub-K1-cost
  target picks `K = 0`; search never overshoots the budget across
  a 8 × 200 grid; search picks the largest K within budget across
  the same grid; the K-search cap is reachable under a generous
  budget. Balance: fresh accumulator is zero; zero balance is the
  identity on target; positive balance adds the divisor share;
  negative balance debits the target; huge debit saturates to
  zero; `update` records the residue; `reset` zeros it;
  round-toward-zero division convention pinned at +7/3 ⇒ +2 and
  -7/3 ⇒ -2. Band loop: length mismatch ⇒ None; empty input
  ⇒ empty output + zero balance; single band uses
  `LAST_BALANCE_DIVISOR`; two bands use `(2, 1)` divisors in
  order; three bands use `(3, 2, 1)` in order — every hand-
  computed `(K, bits_used, balance)` triple is verified explicitly;
  band loop never overshoots any band's adjusted target over a
  5-band mixed-N grid; custom cost functions compose.

* **Round-17 §4.3.4.2 PVQ codebook + shape decoder (2026-06-04):**
  the §4.3.4.2 Pyramid Vector Quantizer codebook size `V(N, K)` and
  the per-band shape decoder. `v_count(n, k)` evaluates the §4.3.4.2
  recurrence `V(N, K) = V(N-1, K) + V(N, K-1) + V(N-1, K-1)` with
  base cases `V(N, 0) = 1` and `V(0, K) = 0` for `K > 0`. The carry-
  one-row implementation runs in `O(N · K)` time and `O(K)` memory
  with `u64` intermediates so the §4.3.4.4 32-bit codebook budget is
  enforced cleanly: results above `u32::MAX` saturate to
  `V_COUNT_SATURATION = u32::MAX` rather than wrap. `decode_index_to_pulses(
  index, n, k) -> Option<Vec<i32>>` runs the §4.3.4.2 per-position
  reconstruction loop on a caller-supplied index in `[0, V(N, K))`
  and returns the signed integer pulse vector whose absolute-value
  sum is exactly `K`. `decode_pulses(dec, n, k)` composes
  `dec_uint(V(N, K))` (§4.1.5) with `decode_index_to_pulses`; a
  sticky range-decoder error propagates as `None`. `normalize_to_unit_l2(
  pulses)` divides by the f64 L2 norm so the output `f32` vector
  lies on the unit hypersphere; the all-zero input degrades to
  all-zero output rather than producing NaN. `decode_unit_shape(
  dec, n, k)` is the convenience composition that returns the
  unit-norm `Vec<f32>` ready to feed the §4.3.4.3 spreading rotation.
  Exposed at the crate root: `v_count`, `decode_index_to_pulses`,
  `decode_pulses`, `normalize_to_unit_l2`, `decode_unit_shape`,
  `V_COUNT_SATURATION`.

  27 new unit tests pin: `V(0, 0) = 1`, `V(0, K > 0) = 0`,
  `V(N, 0) = 1`; hand-computed `V(N, K)` for `(N, K) ∈ {(1, 1) = 2,
  (1, 2) = 2, (2, 1) = 4, (2, 2) = 8, (3, 1) = 6, (3, 2) = 18,
  (4, 1) = 8, (4, 2) = 32}` from the §4.3.4.2 recurrence;
  `V(3, 4) = 66` vs `V(4, 3) = 88` (the recurrence is not symmetric
  in `N` ↔ `K`); the recurrence `V(N, K) = V(N-1, K) + V(N, K-1) +
  V(N-1, K-1)` holds across a 12 × 12 grid; monotonicity in `K`
  (fixed `N`) and in `N` (fixed `K`); `V(180, 180)` saturates to
  `V_COUNT_SATURATION`; the `K = 0` codeword is the unique
  all-zero vector regardless of `N` and reading a higher index is
  rejected; `N = 0` paired with `K > 0` is rejected; out-of-range
  indices (`V(2, 1) = 4` index 4, `V(3, 2) = 18` index 18) return
  `None`; `V(1, 1) = 2` produces `{+1, -1}` codewords; the
  `sum(|X|) == K` and `|X[j]| <= K` invariants hold for every
  index in every `(N, K)` with `N ≤ 6` and `K ≤ 5`; the index ↔
  codeword map is a bijection (no duplicate decoded vectors across
  every legal index for `N ≤ 5`, `K ≤ 4`); the decoded codeword
  set matches a brute-force enumeration of integer vectors with
  `sum |X| = K` for `N ≤ 3`, `K ≤ 3`; the (test-only) `v_column`
  helper matches `v_count` pointwise across a small grid;
  `decode_pulses` integrated with `RangeDecoder` produces a
  well-formed codeword (`sum(|X|) == K`, `|X[j]| <= K`);
  `K = 0` is a no-op on the range decoder (`tell()` does not
  regress) and returns the all-zero vector; `N = 0`, `K > 0`
  returns `None`; `decode_pulses` against a saturated codebook
  size returns `None`; `normalize_to_unit_l2` returns the all-zero
  vector for all-zero input, returns the input unchanged for the
  already-unit `[1, 0, 0, 0]`, and produces unit-L2-norm output
  for every codeword at `(N, K) ∈ [1, 4]²` (within `1e-5` f64
  tolerance); sign preservation on a mixed-sign input;
  `decode_unit_shape` composition matches `normalize_to_unit_l2 ∘
  decode_pulses` step-for-step on the same range-decoder seed;
  `decode_unit_shape` with `K = 0` returns the all-zero vector.

  Clean-room provenance: every step comes from RFC 6716 §4.3.4.2
  (`docs/audio/opus/rfc6716-opus.txt` lines 6504–6536). The
  recurrence, the base cases, the per-position reconstruction
  loop, and the final unit-L2-norm normalisation are all
  reproduced from the IETF text alone. The reference
  implementation source file the RFC delegates to is outside the
  workspace clean-room allow-list and was not consulted. Lib
  test count 241 → 268 (+27).

* **Round-16 §4.3.3 static-allocation search (2026-06-04):** the
  inner §4.3.3 allocation search the RFC describes as "linearly
  interpolating between two values of q (in steps of 1/64) to find
  the highest allocation that does not exceed the number of bits
  remaining" (RFC 6716 §4.3.3 lines 6227–6229). `find_static_alloc(
  coding_start, bins_per_band, channels, lm, budget_1_8th) ->
  Option<StaticAllocSearch { qlo, frac, total_1_8th }>` bisects the
  1/64-step interpolation grid in two phases: a coarse column scan
  over `q ∈ 0..=NUM_Q-1` for the largest qlo whose integer-column
  window total fits the budget, then a fine fractional scan over
  `frac ∈ 0..=INTERP_STEPS` for the highest sub-column position
  whose interpolated window total fits. The §4.3.3 "very low rates"
  zero-allocation exit (budget below the q=0 column) returns
  `StaticAllocSearch { qlo: 0, frac: 0, total_1_8th: 0 }`. The
  canonical upper-column-edge pin (`frac == INTERP_STEPS`) is
  evaluated as the qlo+1 integer column at frac=0 so the
  interpolated evaluator's `frac ∈ 0..INTERP_STEPS` contract stays
  intact, and so the search can reach the top column (`qlo ==
  NUM_Q - 1`) which the interpolated path rejects.

  A new internal helper `window_static_alloc_at_column_1_8th(
  coding_start, bins_per_band, q, channels, lm)` evaluates the
  §4.3.3 `channels * N * alloc << LM >> 2` formula directly at an
  integer column `q` (no interpolation), reachable for the top
  column the interpolated path rejects, and used by the coarse-
  phase bisection. Exposed at the crate root: `find_static_alloc`,
  `StaticAllocSearch`.

  Phase-1 monotonicity rests on the RFC Table 57 invariant
  `STATIC_ALLOC[band][q+1] >= STATIC_ALLOC[band][q]` (proven by
  the existing `rows_monotonic_in_q` test). Phase-2 monotonicity
  rests on the interpolation formula `((64-frac)*lo + frac*hi +
  32) / 64` being non-decreasing in frac whenever hi >= lo (the
  invariant Phase-1 inherits from rows_monotonic_in_q).

  11 new tests cover the saturated top-column case (huge budget
  pins to qlo=10, frac=0, total=200), the zero-budget exit (total =
  0 with no overrun), exact-column-match commitment with the
  highest-position contract (next-step strictly exceeds budget),
  fractional-bracket landing with overrun verification, search
  invariants across budgets 0..=210 (chosen total ≤ budget AND
  next-step strictly exceeds — unless saturated), the
  multi-band-window exact-match case (bands 0..=2 at q=5 sum 381),
  the Hybrid window starting at band 17 at q=9 (sum 184),
  invalid-input rejection (channels=0, lm=4, window overflow),
  budget-monotonicity (larger budget never decreases total), the
  upper-column-edge pin scenario, and dedicated coverage for the
  new `window_static_alloc_at_column_1_8th` reaching the top column
  (q=10) plus rejecting q ≥ NUM_Q.

  Clean-room provenance: every numeric value and every step comes
  from RFC 6716 §4.3.3 (`docs/audio/opus/rfc6716-opus.txt` lines
  6202–6229) and the clean-room narrative at
  `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md`
  §2.7. No external library source was consulted.

* **Round-15 §4.3.3 Table 57 static-allocation table (2026-06-03):** the
  full CELT Static Allocation Table from RFC 6716 §4.3.3 (`STATIC_ALLOC:
  [[u8; 11]; 21]`) plus the per-band evaluator
  `band_static_alloc_1_8th(band, qlo, frac, channels, bins_per_channel,
  lm)` that folds the §4.3.3 formula `channels * N * alloc[band][q] <<
  LM >> 2` with the 1/64-step linear interpolation between adjacent
  quality columns. Companion `interp_alloc_1_32nd(band, qlo, frac)`
  exposes the interpolated 1/32-bit-per-bin coefficient on its own (so
  callers can compose it with non-default scaling). `window_static_
  alloc_1_8th(coding_start, bins_per_band, qlo, frac, channels, lm)`
  sums across a coded-band window for the static-allocation search
  driver. Table 57 is transcribed verbatim from the RFC body
  (`docs/audio/opus/rfc6716-opus.txt` lines 6234–6286) — unlike
  `e_prob_model` / `cache_caps50` / `LOG2_FRAC_TABLE`, the §4.3.3
  static-allocation matrix lives entirely inside the RFC, so no
  staging round was needed. Exposed at the crate root: `STATIC_ALLOC`,
  `NUM_Q = 11`, `INTERP_STEPS = 64`, `interp_alloc_1_32nd`,
  `band_static_alloc_1_8th`, `window_static_alloc_1_8th`. 13 new
  tests cover the table shape (21 × 11), the q=0 all-zero column, the
  q=10 endpoints (200 saturation through band 7, then descending to
  104 at band 20), per-row monotonicity in q, interp anchor (frac=0
  recovers lower column) + bracket inclusion + monotonic frac (when
  the row is rising), the §4.3.3 hand-checked evaluations at
  (band 0, q=5, channels=1, N=4, LM=0) and stereo / LM=1 doublings,
  out-of-range rejection on channels / lm / band / qlo / frac, window
  sum over two adjacent bands and over the four-band Hybrid window at
  q=9 (cells 63, 56, 45, 20), and the interp half-step rounding
  constants.

* **Round-14 §4.3.3 §2.6 minimums + trim_offsets + Table 55 (2026-06-03):**
  the per-band hard-minimum shape allocation `compute_thresh` and the
  per-band trim-derived offset `compute_trim_offsets` (RFC 6716 §4.3.3
  lines 6431–6460), plus the full §4.3 Table 55 MDCT-bin layout as
  `BAND_BINS_LM[lm][band]` (`[[u32; 21]; 4]`, indexed by `LM ∈ 0..=3`
  and band `0..=20`) and the `SHORT_FRAME_BAND_BINS` LM=0 column the
  §2.6 trim_offsets prose explicitly cites. The §2.6 hard-minimum is
  `thresh[band] = max((24 * N[band]) / 16, 8 * channels)` in 1/8-bit
  units, the lower bound being "one bit per channel" and the bin term
  being "48 128th bits per MDCT bin". The §2.6 trim_offset for the
  i-th coded band is `((alloc_trim - 5 - LM) * channels *
  SHORT_FRAME_BAND_BINS[coding_start + i] * (window_len - i) *
  (1 << LM) * 8) / 64`, then `-= 8 * channels` when the band has only
  one MDCT bin per channel — the §2.6 prose "width 1 bands receive
  greater benefit from the coarse energy coding" downward adjustment.

  Both helpers operate on the caller-supplied coded-band window
  (`bins_per_band.len() == coding_end - coding_start`); the absolute
  band index used to index `SHORT_FRAME_BAND_BINS` is
  `coding_start + i`, so Hybrid mode (`coding_start = 17`) picks up
  the correct LM=0 reference cells (the §4.3 / §2.6 distinction
  between window-relative and absolute band index that the §2.6
  prose glosses over).

  Exposed at the crate root: `compute_thresh`, `compute_trim_offsets`,
  `BAND_BINS_LM`, `SHORT_FRAME_BAND_BINS`, `EIGHTH_BIT_QUANTUM = 8`,
  `NUM_LM = 4`. 16 new tests cover Table 55 row sums (`100 << lm`),
  the per-LM doubling pattern, Table 55 spot checks at every regime
  boundary (1-bin / 2-bin / 4-bin / 6-bin / 8-bin / 12-bin / 18-bin /
  22-bin bands), `compute_thresh` mono + stereo lower-bound vs
  bin-term regime splits at LM=0 + LM=3, `compute_trim_offsets`
  default-trim (alloc_trim=5) zero-offset + width-1 adjustments mono
  + stereo, `compute_trim_offsets` worked examples at alloc_trim=10
  mono LM=0, symmetry around alloc_trim=5, Hybrid window
  (coding_start=17) absolute-index lookups, and invalid-input
  rejection across alloc_trim, lm, channels, window over-run, and
  length-mismatch corners.

  Clean-room provenance: every numeric value and every formula comes
  from RFC 6716 §4.3 / §4.3.3 (`docs/audio/opus/rfc6716-opus.txt`)
  and the clean-room narrative at
  `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.6.
  No external library source was consulted.

* **Round-13 §4.3.3 initial-reservations budget walk (2026-06-02):**
  the chained §4.3.3 / clean-room narrative §2.5 budget walk that
  takes `(frame_bytes, ec_tell_frac, is_transient, lm, stereo,
  coded_bands)` and emits an `InitialReservations` carrier with the
  `total_initial` budget, the four `*_rsv` 1/8-bit reservations
  (anti-collapse, skip, intensity, dual-stereo), the running `total`
  budget at the band-boost loop's entry point, and the threaded
  `coded_bands` / `stereo` flags. `compute_initial_reservations` runs
  the four §2.5 steps in fixed order: `total = frame_bytes*64 -
  ec_tell_frac - 1`; `anti_collapse_rsv = 8` iff transient AND `LM >
  1` AND `total >= (LM+2)*8`, with `total >= 0` clamp; `skip_rsv = 8`
  iff `total > 8`; stereo intensity_rsv (via `LOG2_FRAC_TABLE`); and
  `dual_stereo_rsv = 8` iff stereo AND `total > 8` after intensity.
  `InitialReservations::gates_for_band_allocation(ec_tell_frac_now,
  total_boost)` synthesises the `BandAllocationGates` the existing
  `decode_band_allocation` orchestrator needs, including the §4.3.3
  trim gate `ec_tell_frac() + 48 <= total_initial - total_boost`
  that consumes the band-boost loop's `total_boost` accumulator.

  Exposed at the crate root: `compute_initial_reservations`,
  `InitialReservations`, `RSV_BIT_8TH = 8`, `RSV_INITIAL_SLACK_8TH =
  1`. The walk is purely arithmetic — it does not consult the range
  decoder, so a caller running its own step-by-step §4.3.3
  accounting can drop this in without altering `ec_tell_frac()`
  trajectory.

  Defensive corners: `ec_tell_frac` larger than the raw frame budget
  yields a negative `total_initial` and zero reservations across the
  board; `lm > 3` saturates to `lm = 3` for the anti-collapse gate;
  the i32 budget arithmetic uses an i64 multiply to keep a
  pathological `frame_bytes` near `u32::MAX` from wrapping.

  20 new unit tests pin: `RSV_BIT_8TH = 8` and `RSV_INITIAL_SLACK_8TH
  = 1` constants; high-rate stereo 20 ms transient fires every
  reservation (anti-collapse + skip + intensity = 36 + dual = 8); mono
  transient fires anti-collapse + skip but never intensity or dual;
  non-transient skips anti-collapse regardless of LM/budget; `LM ∈
  {0, 1}` blocks the anti-collapse gate; the `total >= (LM+2)*8`
  budget gate is exact at LM=2 (threshold 32, accepts `total =
  32`, rejects `total = 31`); LM clamp at `lm > 3` saturates to 3;
  the skip gate's `total > 8` boundary at LM=0 non-transient
  (accepts `total = 9`, rejects `total = 8`); very-low-rate budget
  (`total_initial = 5`) drops every reservation; intensity drops
  when budget cannot cover `LOG2_FRAC_TABLE[21] = 36`; the
  dual-stereo `> 8` boundary at LM=0 non-transient stereo 4-band
  (accepts `total = 9`, rejects `total = 8`); the `total =
  total_initial - total_reserved()` identity across a (frame,
  tell, transient, lm, stereo, coded_bands) grid (skipping cases
  where the anti-collapse clamp engaged); `ec_tell_frac` larger
  than the frame budget gives all-zero reservations defensively;
  `gates_for_band_allocation` fires every gate at low `ec_tell_frac`
  and zero `total_boost`; a `total_boost` large enough to push the
  trim RHS below `ec_tell_frac + 48` gates trim off (and only trim);
  a mono frame gates intensity + dual off regardless of
  `total_boost`; an intensity reservation dropped during the walk
  (budget tight) gates intensity off in the band-allocation gates;
  a dual reservation dropped (boundary case) gates dual off; and a
  composition smoke test stitches the walk to a live
  `RangeDecoder::tell_frac()` after a single decoded symbol.

  No external library source consulted. Every step is transcribed
  from the clean-room narrative at
  `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.5
  (which paraphrases RFC 6716 §4.3.3) and from RFC 6716 §4.3.3
  itself at `docs/audio/opus/rfc6716-opus.txt`. Lib test count 178 →
  198 (+20).

* **Round-12 §4.3.3 `cache_caps50` + band-boost decode (2026-06-01):**
  the §4.3.3 per-band maximum-allocation `cap[]` machinery and the
  matching band-boost dynalloc-logp loop. `CACHE_CAPS50: [[u8; 21]; 8]`
  reproduces the 168-entry Feist-facts table now staged at
  `docs/audio/celt/tables/cache_caps50.csv`, indexed as
  `CACHE_CAPS50[2*LM + stereo][band]`. `compute_band_caps(lm, stereo,
  channels, bins_per_band, caps)` applies the §4.3.3 conversion
  `cap = (cache.caps[i] + 64) * channels * N / 4` (RFC 6716 line
  6310), with i16 saturation in the unlikely overflow corner and
  defensive rejection of `lm > 3` / `channels ∉ {1,2}` / length
  mismatches. `decode_band_boosts(dec, start, end, bins_per_band,
  caps, total_bits)` runs the §4.3.3 dynalloc loop (RFC 6716 lines
  6339–6360) literally: per-band quanta `min(8*N, max(48, N))`,
  starting `dynalloc_logp = 6` falling to `1` mid-band on the first
  accepted boost and stepping down at-most one notch per
  boosted-band (clamped at the §4.3.3 floor of 2), with the §4.3.3
  loop guard `tell + dynalloc_loop_logp < total_bits + total_boost
  && boost < cap[band]`. `BoostResult { boost, total_boost,
  total_bits_remaining }` packages the per-band boosts plus the
  `total_boost` accumulator the §4.3.3 alloc-trim gate consumes
  (`ec_tell_frac() + 48 <= total_bits - total_boost`). The §4.3.3
  "at very low rates ... inner loop may not run even once" path is
  exercised explicitly (`total_bits` below the initial guard ⇒ zero
  boosts emitted, range decoder untouched). Closes the
  `cache_caps50` docs-gap blocker previously flagged in
  `bit_allocation` module docs.

  9 new unit tests pin: `CACHE_CAPS50` layout (8 rows × 21 bands)
  matches the staged CSV with spot-checks at corners (rows 0/7,
  bands 0/8/20); `compute_band_caps` applies the §4.3.3
  `(cache+64)*ch*N/4` formula bit-exactly at LM=2 mono; stereo
  correctly doubles via `channels` for both low-band (cache=224) and
  mid-band (cache=240) rows; malformed inputs (LM>3, channels∉{1,2},
  length mismatch, caps > NUM_BANDS) cleanly return false; the
  band-boost low-rate path emits zero boosts and does not touch the
  decoder; a one-biased stream (`0xFF` buffer) lands at least one
  boost with sum equal to `total_boost`; the cap ceiling clamps a
  band's boost to ≤ `cap[band]`; malformed inputs return None; and
  the quanta formula `min(8*N, max(48, N))` divides per-band boosts
  exactly for `N ∈ {1, 4, 6, 8, 16, 32, 64}` (expected quanta
  `{8, 32, 48, 48, 48, 48, 64}`).

* **Round-11 §4.3.4.5 Hadamard-transform primitives (2026-05-31):** the
  three pure helpers the §4.3.4.5 TF-resolution-change step needs once
  the per-band shape vectors come on-line. `walsh_hadamard_inplace`
  runs the orthonormal radix-2 forward Walsh–Hadamard transform in
  natural (Hadamard) order, with per-level `1/√2` scaling so the
  matrix is orthonormal and the L2 norm is preserved.
  `walsh_hadamard_sequency_inplace` runs the same transform in
  sequency (Walsh) order: the natural-order butterfly cascade
  followed by the textbook `bit_reverse(gray(s))` permutation that
  places the zero-sequency (DC) bin at output index 0 and the
  maximum-sequency bin at output index `2^levels - 1`. This is the
  variant the §4.3.4.5 prose calls for in the time-resolution-
  increase branch ("the decoder uses the 'sequency order' because
  the input vector is sorted in time"). `apply_tf_resolution_change`
  orchestrates the two over an interleaved `nb_blocks × subvec` band
  layout: `tf_adjustment < 0` ⇒ sequency-ordered WHT per sub-vector
  (time-resolution increase); `tf_adjustment > 0` ⇒ natural-order WHT
  across the interleaved blocks (frequency-resolution increase);
  `tf_adjustment == 0` is a no-op. Returns `false` and leaves the
  band unchanged when the shape constraints are violated (non-power-
  of-two block count or sub-vector length, or a request exceeding
  the available WHT levels in either direction).

  11 new unit tests pin: natural-order WHT of `[1,0,0,0]` yields the
  constant `0.5` row at N=4 (each row of the natural Hadamard matrix
  starts with `+1`); natural-order WHT of `[1,1,1,1]` yields
  `[2, 0, 0, 0]` (DC at sequency-0 bin under the natural ordering
  because the natural Hadamard matrix row 0 is the all-ones row);
  applying the WHT twice is identity within `1e-5` (orthonormal
  involutivity at N=8); L2 norm preserved at N=16; sequency-ordered
  WHT puts a constant signal in bin 0 (sequency-DC) and a maximum-
  sign-alternation signal in bin N-1 (highest sequency); `tf=0` is
  a strict no-op; `tf=-2` on a 2-block × 4-sample band with both
  blocks `[1,1,1,1]` produces `[2,0,0,0]` per sub-vector; `tf=+2`
  on a 4-block × 1-sample band with all-ones across the blocks
  produces `[2,0,0,0]` across blocks (natural-order DC); shape
  rejections for non-divisible band length, non-power-of-two
  sub-vector length, oversized `|tf_adjustment|`, and non-power-of-
  two block count; the internal `bit_reverse(gray(s))` permutation
  matches a hand-computed N=4 table.

  Normalization choice (`HADAMARD_LEVEL_SCALE = 1/√2`) is documented
  in `src/hadamard.rs` as a decoder-side decision because RFC 6716
  §4.3.4.5 is silent on the per-level scaling. Chosen to keep the
  unit-norm band shape vectors invariant across TF resolution
  changes so the §4.3.6 denormalization step (which multiplies by
  `sqrt(energy)`) sees the same shape regardless of `tf_adjustment`.
  Swap to `1.0` (unscaled butterfly) if a future `opusdec` trace
  shows the reference does no normalization at all.

  Pure clean-room derivation: the Walsh–Hadamard transform is
  textbook material that predates CELT by decades (Hadamard 1893,
  Walsh 1923). The radix-2 butterfly cascade was written from first
  principles against the matrix definition `H_2 = [[1,1],[1,-1]]`,
  `H_{2n} = H_2 ⊗ H_n`; no external library source consulted. The
  sequency-order permutation `bit_reverse(gray(s))` is the standard
  result from any DSP textbook covering the Walsh transform.

  Lib test count 158 → 169 (+11).

## [0.1.7](https://github.com/OxideAV/oxideav-celt/compare/v0.1.6...v0.1.7) - 2026-05-30

### Other

- round-10 §4.3.3 stereo reservation: LOG2_FRAC_TABLE + intensity_rsv + reserve_stereo (RFC 6716)
- round-9 §4.3.7.1 post-filter + §4.3.7.2 de-emphasis (RFC 6716)

### Added

* **Round-10 §4.3.3 stereo reservation helpers (2026-05-30):**
  the `LOG2_FRAC_TABLE` numeric constant + the two pure-arithmetic
  helpers that drive the §4.3.3 intensity-stereo and dual-stereo
  reservation gates. `LOG2_FRAC_TABLE: [u8; 24]` holds the
  conservative `log2(n)` in 1/8-bit units, indexed by `coded_bands ∈
  0..=23`; values come from the Feist-facts CSV at
  `docs/audio/celt/tables/log2_frac_table.csv` and are documented in
  the clean-room narrative §2.5 at
  `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md`.
  `intensity_rsv(coded_bands, stereo, total_8th_bits) -> u32`
  returns the §4.3.3 intensity reservation in 1/8 bits — the
  `LOG2_FRAC_TABLE[coded_bands]` lookup when stereo + budget covers
  it, zero otherwise (mono frames, empty band ranges,
  out-of-table band counts, or budget too small per "if intensity_rsv
  > total, set it to 0"). `reserve_stereo(coded_bands, stereo,
  total_8th_bits) -> (u32, i32, u32)` runs the full two-step
  reservation in §4.3.3 order (intensity_rsv first, then
  dual_stereo_rsv if `total > 8` after intensity), returning
  `(intensity_rsv, total_after, dual_stereo_rsv)`. 16 new unit tests
  pin: `LOG2_FRAC_TABLE` has exactly 24 entries; spot-checks against
  the §2.5 narrative (`[0]=0`, `[1]=8`, `[2]=13`, `[3]=16`,
  `[16]=33`, `[23]=37`); the table is monotone non-decreasing; every
  entry sits between `floor(log2(n))*8` and `log2(n)*8 + 8` (the
  conservative-rounding bounds); `intensity_rsv` returns zero for
  mono regardless of band/budget combinations; zero band range
  returns zero; out-of-range band counts (`24, 25, 100, u32::MAX`)
  fall back to zero defensively; stereo with sufficient budget
  returns `LOG2_FRAC_TABLE[coded_bands]` for every legal value 1..=23;
  budget-tight at the §4.3.3 equality boundary (e.g. 21 bands needs
  36 → budget 35 drops, budget 36 keeps, budget 100 keeps); hybrid
  4-band CELT reserves 19 1/8-bits; `reserve_stereo` on mono is
  passthrough; large stereo budget reserves both intensity and dual
  (36 + 8 = 44 of total subtracted for 21-band CELT); tight stereo
  budget at the 44-budget boundary reserves only intensity, +1 budget
  flips dual on; under-budget for intensity drops it but dual still
  considers the remainder; empty band range + stereo skips intensity
  but reserves dual; `reserve_stereo`'s intensity output agrees with
  `intensity_rsv` across a (band, budget, stereo) grid.

  No external library source consulted. The clean-room narrative at
  `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.5
  describes the §4.3.3 stereo reservation step in algebraic form;
  the numeric `LOG2_FRAC_TABLE` values come from the Feist-facts CSV
  extract at `docs/audio/celt/tables/log2_frac_table.csv` (24 entries
  of 1/8-bit units, unsigned byte). Lib test count 142 → 158 (+16).

## [0.1.6](https://github.com/OxideAV/oxideav-celt/compare/v0.1.5...v0.1.6) - 2026-05-29

### Other

- round-8 §4.3.4.3 spreading parameter + Table 56 / Table 59 (RFC 6716)
- round-7 §4.3.2.2 fine-energy refinement + finalize step (RFC 6716)
- scrub external-name disclaimer (clean-room hygiene)
- round-6 §4.3.4.5 time-frequency change: per-band tf_change + gated tf_select + Tables 60–63 (RFC 6716)
- round-5 §4.3.3 bit-allocation fields: alloc.trim / skip / intensity / dual (RFC 6716)
- round-4 coarse-energy scaffold: 21-band layout + intra prediction filter + DOCS GAP (RFC 6716 §4.3.2.1)
- round-3 frame header: silence/post-filter/transient/intra prefix + deferred anti-collapse (RFC 6716 §4.3, §4.3.5, §4.3.7.1)
- round-2 entropy primitives: ec_decode_bin / ec_dec_icdf / ec_tell_frac (RFC 6716 §4.1.3.1, §4.1.3.3, §4.1.6.2)
- round-1 bootstrap: bit-exact CELT/SILK range decoder (RFC 6716 §4.1)
- orphan rebuild: clean-room scaffold post 2026-05-20 audit

### Added

* **Round-9 post-filter + de-emphasis (2026-05-30):** RFC 6716
  §4.3.7.1 + §4.3.7.2, the last two pipeline stages between the
  inverse MDCT and the final PCM output. The post-filter contributes
  the three §4.3.7.1 tap shapes in both f32 (`POST_FILTER_TAPS_F32`)
  and Q15 fixed-point (`POST_FILTER_TAPS_Q15`); the gain
  reconstruction `G = 3*(gain+1)/32` in f32 (`gain_f32`) and exact
  Q15 (`gain_q15= 3*(gain+1)*1024`); a per-sample evaluator
  `filter_sample_f32(x, history, period, gain, tapset)` of the
  §4.3.7.1 recursion
  `y(n) = x + G*(g0*y(n-T) + g1*(y(n-T+1)+y(n-T-1)) + g2*(y(n-T+2)+y(n-T-2)))`;
  and an in-place slice variant `apply_post_filter_f32(out,
  prev_output, period, gain_index, tapset)` that carries the
  previous-frame's filtered tail through `prev_output` so the
  recursive part stays continuous across frame boundaries.
  Defensive corners: out-of-range tapsets clamp to the last valid
  entry; pitch periods below 15 clamp to `POST_FILTER_PERIOD_MIN`;
  past samples that lie before `history` are treated as zero, so
  the startup transient degrades to passthrough rather than reading
  out of bounds. 19 new unit tests pin: each of the three tap rows
  matches the RFC's decimal listing to f32 precision; the Q15 rows
  match `round(f32 * 32768)` exactly (with a documented ±1 ULP
  discrepancy on one entry due to the RFC's truncation of the
  underlying ratio); `g0 + 2*g1 + 2*g2 ≈ 1.0` (the unity-gain-at-
  pitch property each tapset relies on); tapset and Q15 lookup
  saturation; `gain_f32` matches `3*(gain+1)/32` across every legal
  raw index plus the corner cases `gain=0 ⇒ 0.09375` and `gain=7 ⇒
  0.75`; `gain_q15` matches `round(gain_f32 * 32768)` exactly with
  hand-checked corners; the per-sample evaluator returns `x`
  unchanged at `G=0` (the gain-transition crossfade corner); empty
  history degrades to passthrough; single-impulse history at the
  centre tap position contributes exactly `G * g0`; symmetric
  impulse pairs at `T±1` and `T±2` pick up the `2*G*g1` and `2*G*g2`
  lobe contributions; zero history with non-zero input is
  passthrough for every tapset and gain; the slice variant's
  startup-passthrough is held for `n < T-2`; the period bounds
  match the §4.3.7.1 "between 15 and 1022 inclusive" prose; the
  period clamp produces identical output for `period=0` and
  `period=15`. The de-emphasis filter contributes a single-pole IIR
  state struct `Deemphasis { last_y }` carrying `y(n-1)` across
  calls; `step` / `apply_in_place` / `apply` / `reset` methods; a
  one-shot convenience wrapper `deemphasize_in_place_f32`; the spec
  coefficient `ALPHA_P_F32 = 0.8500061035` and its Q15 form
  `ALPHA_P_Q15 = 27853`. 14 new unit tests pin: the f32 / Q15
  coefficients match the RFC decimal; default state is zero; an
  impulse produces the expected geometric decay
  `y(n) = α_p^n`; zero input + zero state stays zero; a DC input
  converges to the steady-state `1/(1-α_p) ≈ 6.667`; running the
  filter on two halves with state-carry matches running it on the
  joined buffer one-shot (the cross-frame continuity invariant);
  `step` and `apply_in_place` produce identical results on the same
  input; `apply` writes only to the destination slice; length
  mismatch panics; reset zeroes the state; the convenience wrapper
  matches the stateful filter; the response stays finite under
  bounded input.

  No external library source consulted. The §4.3.7.1 prose gives
  the full mathematical recursion and the three tap shapes as
  explicit decimals; §4.3.7.2 prints the single-pole IIR in
  algebraic form and `alpha_p` as a decimal. Both stages therefore
  close against the RFC text alone; the source files the RFC
  normatively delegates to sit outside the workspace clean-room
  allow-list and were not consulted.

* **Round-8 spreading parameter (2026-05-29):** RFC 6716 §4.3.4.3
  decoder for the `spread` scalar that sits between the global
  `tf_select` flag and the §4.3.3 dynamic-allocation phase in
  Table 56 order. `decode_spread(dec)` reads the §4.3.4.3 spread
  field with PDF `{7, 2, 21, 2}/32` via the §4.1.3.3 ICDF path
  (cumulative `[7, 9, 30, 32]`, ICDF `[25, 23, 2, 0]`, ftb=5) and
  returns one of the four `Spread` variants `{None, Light, Normal,
  Aggressive}` in raw-value order (`spread = 0..=3`). `Spread::f_r()`
  is the Table 59 lookup: `None` (`spread=0`, no rotation),
  `Some(15)` / `Some(10)` / `Some(5)` for `spread = 1 / 2 / 3`.
  `rotation_gain_ratio(spread, n, k)` returns the closed-form
  `g_r = N/(N + f_r*K)` rotation gain as a `(num, den)` unsigned-
  integer pair so the PVQ caller can pick its own fixed-point
  representation; `rotation_gain_squared_ratio` returns the same
  ratio squared (u64 per term) to feed the `theta = pi * g_r^2 / 4`
  rotation-angle computation. `pre_rotation_stride(n, nb_blocks)`
  returns the `round(sqrt(N/nb_blocks))` interleave stride for the
  §4.3.4.3 extra rotation by `(pi/2 - theta)` applied before the
  main rotation when each time block represents at least 8 samples
  (rounding direction round-half-up — the canonical interpretation
  of unqualified `round()` notation in IETF prose). 14 new unit
  tests cover: every `Spread` variant round-trips through
  `as_u8`/`from_u8`; `f_r` matches Table 59 row-by-row; the
  `DEFAULT_SPREAD` constant is `Spread::Normal` (the bulk-probability
  case at 21/32); the ICDF entries match the PDF cumulative
  reconstruction (sum = ft = 32, per-cell mass `{7, 2, 21, 2}`,
  monotonic descent, terminator-zero); `decode_spread` is total
  (every single-byte input produces one of the four legal variants,
  never panics); the rotation-gain identity for `Spread::None` is
  exactly `(0, 1)`; `rotation_gain_ratio` matches the closed-form
  `N / (N + f_r * K)` at hand-computed values for each non-identity
  spread; the squared form matches the pair-wise square; empty band
  (`N = 0`) collapses to `(0, 1)` defensively; `K = 0` is the unit
  ratio `(N, N)`; saturating arithmetic handles `u32::MAX` inputs
  without overflow; the pre-rotation stride is `None` below 8
  samples per block, `None` for `nb_blocks <= 1`, `None` for empty
  bands, matches `round(sqrt(per_block))` at spot-checked values
  (`per_block = 32/16/64/100`), the half-up tie-break is applied
  consistently, and the returned stride is never zero.

  This is decoder-side only. The 2-D rotation `R(i, j)` loop and
  the `(pi/2 - theta)` pre-rotation are PVQ-shape-decoder work
  (RFC 6716 §4.3.4.3 final paragraph); they sit on the per-band
  codevector vector and are queued for the band-decode round. The
  encoder-side §5.3.7 `spreading_decision` derivation is irrelevant
  on the decode side and explicitly out of scope.

  This commit also scrubs three pre-existing module docstrings
  (`bit_allocation.rs`, `coarse_energy.rs`, `fine_energy.rs`) that
  named the spec-delegation target source files inherited verbatim
  from RFC 6716 §4.3.2.1 / §4.3.2.2 / §4.3.3 prose. The RFC names
  the delegation targets in its normative text, but per the
  workspace clean-room policy the agent-authored module docstrings
  rephrase the delegation as "a source file outside the workspace
  clean-room allow-list" instead. The behaviour and public API are
  unchanged.

* **Round-7 fine-energy refinement (2026-05-29):** RFC 6716 §4.3.2.2
  decoder for the second of CELT's three-step coarse-fine-fine
  energy-envelope strategy. The fine step is purely a raw-bit channel
  (no Laplace decoder, no `e_prob_model` table) so it is implementable
  without consulting the source files that the §4.3.2.1 prose
  delegates to. `decode_fine_energy_band(dec, b_bits)` reads exactly
  `b_bits` raw bits, forms `f ∈ [0, 2^b_bits)`, and returns the
  closed-form Q14 correction `(2f+1) * 2^(13-b_bits) - 2^13`
  (derivation from the spec's `(f+1/2)/2^B - 1/2`).
  `decode_fine_energy(dec, &[u32; 21])` walks the full envelope.
  `fine_correction_q14` / `fine_correction_qn` expose the closed-form
  arithmetic standalone for encoders and test scaffolding (the Qn
  variant uses round-half-up arithmetic so callers at low precision
  get the nearest-representable answer rather than truncation).
  `finalize_extra_bits(dec, priorities, coded_bands, channels, budget)`
  walks the §4.3.2.2 finalize step: leftover raw bits are spent ≤ 1
  per band per channel in `(priority 0 ascending, priority 1
  ascending)` order, returning per-band Q14 corrections summed across
  channels plus `(bits_consumed, bits_unused)`. Excess budget is left
  unused per the spec. `MAX_FINE_BITS = 8` caps per-band fine bits.
  20 new unit tests cover: `b_bits=0` is a no-op on the decoder; the
  Q14 closed form matches the spec's decimal formula bit-for-bit at
  every `(f, B)` for B ∈ [1, 8]; correction range is `[-8192, +8191]`
  for every legal pair; symmetric `(half-1, half)` around zero;
  endpoint extrema at `f=0` and `f=max`; Q14 vs Qn agreement at
  n=14; Q8 / Q14 / 64 round-trip with documented one-step boundary at
  B=8; `decode_fine_energy_band` consumes exactly `b_bits` raw bits;
  full-envelope total matches `sum(bits_per_band)`; all-zero `B_i`
  envelope is a complete no-op; finalize's zero-budget / zero-coded-
  bands no-ops; finalize surplus-left-unused mono and stereo (Q14
  corrections summed across channels); finalize priority-0 drains
  fully before priority-1 starts; priority-1 walks ascending after
  priority-0; finalize's `bits_consumed` matches the `tell()` delta
  exactly; budget-tight at exactly the priority-0 capacity leaves
  priority-1 untouched; spot-check Q14 corrections against
  hand-computed decimal values; smoke stitches `decode_fine_energy`
  and `finalize_extra_bits` on the same decoder.

  This is decoder-side only. The §4.3.3 bit allocator that produces
  the per-band `B_i` vector and the `(priorities, leftover_budget)`
  inputs is gated separately on the `cache_caps50[]` DOCS-GAP; the
  fine-energy API here is parameterised on caller-supplied
  allocations so it composes cleanly once the gap closes.

* **Round-6 time-frequency change parameters (2026-05-25):** the
  §4.3.4.5 + §4.3.1 TF group. `decode_tf_changes(dec, start, end,
  is_transient)` reads one bit per coded band; first band uses
  PDF `{3,1}/4` (transient) or `{15,1}/16` (non-transient),
  subsequent bands use PDF `{15,1}/16` (transient) or `{31,1}/32`
  (non-transient). The rare "1" symbol toggles the running TF
  choice per the §4.3.4.5 differential encoding. `decode_tf_select`
  decodes the global `tf_select` flag (`{1,1}/2`) only when at
  least one band's tf_change would yield a different TF adjustment
  under `tf_select=0` vs `tf_select=1`; the §4.3.4.5 "no impact"
  gate is exposed standalone as `tf_select_matters` for caller
  inspection. `tf_adjustment` indexes the four published TF tables
  (60–63) in one function: `[i8;2]×4` rows per table, indexed by
  `LM ∈ {0,1,2,3}` (= 2.5/5/10/20 ms) and `tf_change ∈ {0,1}`.
  The four tables are exposed as `pub const`s
  (`TABLE_60_NON_TRANSIENT_SEL0`, `TABLE_61_NON_TRANSIENT_SEL1`,
  `TABLE_62_TRANSIENT_SEL0`, `TABLE_63_TRANSIENT_SEL1`).
  `decode_tf_parameters` orchestrates the whole §4.3.4.5 walk in
  Table 56 order and returns a `TfParameters { tf_changes,
  tf_select, tf_select_decoded }`. 19 new unit tests cover:
  every published TF table matches the RFC entry-for-entry; the
  non-transient `tf_change=0` column is all-zero (no-change ⇒ no
  adjustment); non-transient adjustments are all `<= 0`;
  `tf_adjustment` dispatch picks the right table per
  `(is_transient, tf_select)`; oversized `lm` saturates to 3
  rather than panicking; any non-zero `tf_select` is coerced to
  `select=1`; `TfParameters::zeros` has the right shape; empty
  band range is a no-op; full 21-band CELT range advances `tell()`
  and returns 21 entries; hybrid 17..21 range returns 4 entries;
  the low-byte (`val=127`) §4.1.1 init biases every per-band
  decode to "0"; the high-byte (`val=0`) init flips the first
  band's tf_change `true`; the `tf_select` gate is correctly
  satisfied / unsatisfied on the documented LM corner cases;
  `decode_tf_select` empty-`tf_changes` is a no-op; the
  full-pipeline `decode_tf_parameters` keeps `tf_select_decoded`
  in sync with the gate predicate across a (seed × transient × LM)
  product matrix; orchestrator agrees with a hand-stitched
  `decode_tf_changes` + `decode_tf_select` call on the same
  buffer; `decode_tf_parameters(start=end)` does not touch the
  range decoder; per-band `tf_adjustment` round-trips through every
  published table entry.

  This is decoder-side only. §4.3.4.1 transient detection is
  encoder-side; the Hadamard transform that implements the TF
  resolution change in the actual shape decode path is band-decode
  work for a later round. No Laplace decoding required.

* **Round-5 bit-allocation fields (2026-05-22):** the four §4.3.3
  scalar fields that sit between the coarse-energy block and the
  per-band shape vectors (Table 56 order: alloc.trim → skip →
  intensity → dual). `decode_alloc_trim` walks the Table 58 PDF
  `{2,2,5,10,22,46,22,10,5,2,2}/128` (icdf
  `[126,124,119,109,87,41,19,9,4,2,0]`, ftb=7), `decode_skip_flag`
  and `decode_dual_stereo` read `{1,1}/2` bits via `dec_bit_logp(1)`,
  and `decode_intensity_band` decodes a uniform value in
  `0..=coded_bands` via `dec_uint(coded_bands+1)`. Each field is
  gated by a caller-supplied boolean recording whether the §4.3.3
  reservation step fired; gated-off fields return `None` and do not
  touch the range decoder, preserving `ec_tell_frac()` accounting
  for the caller's outer budget walk. The `decode_band_allocation`
  orchestrator composes all four in Table 56 order and returns a
  `BandAllocation` with §4.3.3 defaults filled in (trim=5,
  skip=false, intensity offset=0, dual=false) for every gated-off
  field. 18 new unit tests cover: defaults match §4.3.3 prose; the
  Table 58 PDF round-trips through the icdf form (sum = ft = 128,
  per-cell mass exact, terminator-zero, monotonic descent); each
  per-field decoder is a no-op when gated off and advances `tell()`
  when gated on; intensity in-range across all 21 possible
  `coded_bands` values; the orchestrator's all-off path yields
  defaults with no decoder consumption; the all-on path advances
  `tell()` and lands in §4.3.3 value ranges; mono-only gating
  leaves intensity/dual at defaults; hybrid mode (`coded_bands=4`)
  bounds intensity correctly; field-order matches Table 56 by
  hand-decoding the same buffer step-by-step and comparing against
  the orchestrator's result; ensemble of inputs exercises the
  inner-cell trim values `{4, 5, 6}` (90/128 of the PDF mass).

  The full §4.3.3 budget walk (`total_boost`, per-band cap[]
  vector, anti-collapse / skip / intensity reservation arithmetic)
  is NOT in this round; the caller computes the gating booleans
  externally. The band-boost loop in particular depends on
  `cache_caps50[]` which RFC 6716 §4.3.3 delegates to a source file
  outside the workspace clean-room allow-list — that is a separate
  docs gap, queued behind the Laplace / `e_prob_model` blocker.

* **Round-4 coarse-energy scaffold (2026-05-21):** RFC 6716 §4.3.2.1
  scaffolding for the per-band coarse-energy decoder. Lands:
  `NUM_BANDS = 21` from Table 55; the intra-mode prediction
  coefficients `INTRA_ALPHA_Q15 = 0` and `INTRA_BETA_Q15 = 4915`
  (the only numeric coefficients §4.3.2.1 supplies directly, RFC
  line 6063); a `CoarseEnergyState` carrier struct holding the
  previous frame's per-band Q8 log-energies (zeroed on reset, the
  state for next-frame inter prediction); the post-Laplace-decode
  arithmetic `apply_intra_prediction` that runs the §4.3.2.1 2-D
  prediction filter in its intra reduction (β-IIR over bands with
  the time arm vanishing); and a public `decode_coarse_energy`
  entry-point whose signature is locked in for future rounds but
  currently returns `Error::NotImplemented`. 7 new unit tests pin
  band-count, intra prediction coefficients, fresh-state shape,
  the trivial all-zero reconstruction, the single-impulse β-decay
  cascade, an additive two-band hand-computed reconstruction, the
  gap'd `decode_coarse_energy` non-disturbance of the range decoder,
  and the IIR's steady-state stability bound under a constant input.

  **DOCS GAP filed.** RFC 6716 §4.3.2.1 normatively delegates the
  `e_prob_model` per-band Laplace probability table and the
  `ec_laplace_decode` algorithm to source files outside the
  workspace clean-room allow-list. Without either piece, this round
  cannot land a bit-exact coarse-energy decoder; the scaffold's
  `decode_coarse_energy` therefore returns `NotImplemented` and the
  module docstring documents the closure requirements (a clean-room
  derivation of the table + a prose specification of the Laplace
  decoder algorithm). The decoder's `tell()` and the state's
  `prev_q8` are asserted untouched on the gap'd path so that future
  rounds can drop the Laplace decoder in without altering the
  state shape or call site.

* **Round-3 frame header (2026-05-21):** the always-present prefix of
  the CELT frame header (RFC 6716 §4.3, Table 56) — silence flag
  (`{32767,1}/32768`), post-filter flag (`{1,1}/2`) and its four
  §4.3.7.1 parameters (`octave` uniform(6), `period` = `4+octave`
  raw bits, `gain` = 3 raw bits, `tapset` `{2,1,1}/4`), transient
  flag (`{7,1}/8`), and intra flag (`{7,1}/8`). The §4.3.5
  anti-collapse bit is exposed via `decode_anti_collapse_flag`
  because Table 56 places it after the band shape vectors, not in
  the prefix. New public types `CeltFrameHeader`, `PostFilter` and
  the helper `CeltFrameHeader::post_filter_gain_q15()` for the
  §4.3.7.1 gain reconstruction. 8 new unit tests cover: all-zero
  buffer biases every flag off, all-ones buffer biases every flag
  on, `decode_prefix` advances `tell()`, the post-filter
  gain-Q15 formula across all 8 raw indices including the spec's
  `gain=7 ⇒ G=0.75` corner, the §4.3.7.1 period-bound endpoints
  (`(16<<0)-1 = 15` and `(16<<5)+511-1 = 1022`),
  `decode_anti_collapse_flag` is a no-op when transient is unset,
  it advances `tell()` when transient is set, and a smoke test
  stitches the prefix walk and the deferred anti-collapse bit
  together end-to-end.

* **Round-2 entropy primitives (2026-05-21):** three additional
  range-decoder methods to round out the §4.1 surface:
  `decode_bin(ftb)` (§4.1.3.1, the division-free power-of-two `ft`
  decode), `dec_icdf(icdf, ftb)` (§4.1.3.3, the primary SILK
  interface — table-driven inverse-CDF symbol decode with combined
  search-and-update), and `tell_frac()` (§4.1.6.2, 1/8th-bit-precision
  bit-budget accounting). The `tell() == ceil(tell_frac()/8)`
  identity from §4.1.6.1 is asserted in CI on every decoder step.
  7 new unit tests cover: `decode_bin` ≡ generic `decode(1<<ftb)`,
  `tell_frac`-vs-`tell` consistency on a live decoder, the
  freshly-initialised `tell_frac` slack, `dec_icdf` ≡
  `dec_bit_logp` for the binary case, `dec_icdf` uniform-PDF
  in-range, the degenerate single-symbol icdf table, and
  `tell_frac` monotonicity under mixed symbol+raw operations.

* **Round-1 bootstrap (2026-05-20):** bit-exact CELT/SILK range
  decoder per RFC 6716 §4.1. `RangeDecoder` exposes `new`,
  `dec_bit_logp` (§4.1.3.2), `dec_bits` (§4.1.4), `dec_uint`
  (§4.1.5), `tell` (§4.1.6.1), and `has_error`. Internal helpers
  cover the symbol-update path (§4.1.2) and renormalization
  (§4.1.2.1) for use by the future band decoder. 9 unit tests
  cover initialization, normalization, the small-`ft` and
  large-`ft` `dec_uint` branches, LSB-first raw-bit ordering,
  zero-padding past end-of-frame, and `tell()` monotonicity.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule (no external library source as
  reference, not even as a sanity check). Per the workspace's
  Implementer-Round procedure, such audit failures are unrecoverable
  via incremental cleanup and require an orphan rebuild.

  Every public API path now returns `Error::NotImplemented`. A
  clean-room re-implementation against RFC 6716 is planned for a
  future round.

  No `old` branch is retained; long-standing audit failures forfeit
  the archive per workspace policy.
