# Changelog

All notable changes to `oxideav-celt` are recorded here.

## [Unreleased]

### Added

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
  `ec_laplace_decode` algorithm to libopus source files
  (`quant_bands.c`, `laplace.c`) which the workspace clean-room
  policy bars us from reading. Without either piece, this round
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
