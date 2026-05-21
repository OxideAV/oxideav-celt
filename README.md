# oxideav-celt

Pure-Rust CELT (the MDCT path of Opus, RFC 6716).

## Status — 2026-05-21

**Round-3.** The bit-exact CELT/SILK range decoder (RFC 6716 §4.1)
plus the CELT frame-header prefix (RFC 6716 §4.3, the scalar
fields that precede band-decode) are now wired up. The band
decode, PVQ, energy envelope, and MDCT paths still come later.

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
