# Changelog

All notable changes to `oxideav-celt` are recorded here.

## [Unreleased]

### Added

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
