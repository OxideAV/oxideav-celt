# oxideav-celt

Pure-Rust CELT (the MDCT path of Opus, RFC 6716).

## Status — 2026-05-20

**Round-1 bootstrap.** The bit-exact CELT/SILK range decoder
(RFC 6716 §4.1) is implemented and unit-tested. This is the leaf
entropy-coding primitive that every CELT and SILK symbol passes
through; the band-decode, PVQ, and MDCT paths will layer on top of
it in later rounds.

What is wired up today:

* `RangeDecoder::new(buf)` — initialization per §4.1.1.
* `dec_bit_logp(logp)` — binary symbol with probability `2^-logp` of
  a "1" (§4.1.3.2).
* `dec_bits(n)` — raw bits, packed LSB-first from the end of the
  frame (§4.1.4).
* `dec_uint(ft)` — uniformly-distributed integer in `0..ft`,
  including the `ftb > 8` split-decode branch (§4.1.5).
* `tell()` — whole-bit budget accounting (§4.1.6.1).
* Sticky `has_error()` for the corrupt-frame path documented in
  §4.1.5.

Higher-level entry points (frame decoder, encoder, codec
registration with the runtime) still return `Error::NotImplemented`.

## Clean-room provenance

The implementation references only the IETF specifications under
`docs/audio/opus/`:

* RFC 6716 — Definition of the Opus Audio Codec (CELT layer + range
  coder + MDCT path).
* RFC 8251 — Opus Update.
* RFC 7845 — Ogg Encapsulation for Opus (consulted for framing).

No external library source — libopus, the Opus reference encoder /
decoder, etc. — is permitted as a reference under the workspace
clean-room policy. Black-box invocations of `opusdec` / `opusenc`
are allowed as opaque validators only.

## License

MIT. See `LICENSE`.
