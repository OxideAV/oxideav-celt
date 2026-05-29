# Changelog

All notable changes to `oxideav-celt` are recorded here.

## [Unreleased]

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
