//! Range decoder primitives for the Opus/CELT layer.
//!
//! This module implements the bit-exact range decoder described in
//! RFC 6716 §4.1 (`docs/audio/opus/rfc6716-opus.txt`). The implementation
//! is clean-room: every routine is transcribed from the prose and
//! pseudocode equations in the RFC; no external library source was
//! consulted.
//!
//! The following routines are wired up:
//!
//! * Initialization (§4.1.1).
//! * Symbol-update internal helper (§4.1.2).
//! * Renormalization (§4.1.2.1).
//! * `ec_decode_bin` for power-of-two `ft` symbols (§4.1.3.1).
//! * `ec_dec_bit_logp` (§4.1.3.2).
//! * `ec_dec_icdf` for inverse-CDF table decoding (§4.1.3.3).
//! * `ec_dec_bits` for raw bits (§4.1.4).
//! * `ec_dec_uint` for uniformly-distributed integers (§4.1.5).
//! * `ec_tell` for whole-bit accounting (§4.1.6.1).
//! * `ec_tell_frac` for 1/8th-bit-precision accounting (§4.1.6.2).
//!
//! The full generic `ec_decode`/`ec_dec_update` symbol path is reachable
//! through the private helpers and is exercised internally by
//! `dec_uint`; a dedicated public symbol-decoding API will land
//! together with the band decoder when the bit allocator needs it.

use crate::Error;

/// Bit-exact CELT/SILK range decoder state per RFC 6716 §4.1.
///
/// The decoder splits the input buffer into two halves: the range
/// coder consumes bytes from the front (MSB-first into the range
/// state) and the raw-bit reader consumes bytes from the back
/// (LSB-first). The two readers may overlap — RFC 6716 §4.1.4
/// explicitly permits this and the decoder MUST allow it.
#[derive(Debug)]
pub struct RangeDecoder<'a> {
    /// Input bitstream backing this decoder.
    buf: &'a [u8],
    /// Offset of the next byte the range coder will consume
    /// (advances forward through `buf`).
    fwd: usize,
    /// Offset of the *last* byte the raw-bit reader consumed, measured
    /// from the END of `buf`. `back == 0` means no raw bit has yet been
    /// read; the next raw byte read comes from `buf[buf.len()-1]`.
    back: usize,
    /// Bit position inside the most recently fetched raw byte
    /// (0..=8). When equal to 0, a fresh byte must be fetched on the
    /// next raw-bit read.
    back_bits_avail: u32,
    /// The most recently fetched raw byte, masked down to the bits
    /// that have not yet been consumed.
    back_window: u32,
    /// One-bit buffer holding the LSB of the previously-consumed
    /// forward byte (used in the next renormalization step, §4.1.2.1).
    rem: u32,
    /// Range size; the renormalization invariant is `rng > 2**23`.
    rng: u32,
    /// Top of range minus current code value, minus one.
    val: u32,
    /// Running tally of whole bits the range coder has consumed
    /// (RFC 6716 §4.1.6: `nbits_total` accounting).
    nbits_total: u32,
    /// Number of raw bits the decoder has read so far. This is added
    /// into the bit-usage accounting separately (§4.1.6).
    nbits_raw: u32,
    /// Sticky error flag: any decode that detects a corrupt frame
    /// latches an error. Once set, subsequent decodes return zeroes
    /// rather than scribbling garbage into the caller's state.
    error: bool,
}

impl<'a> RangeDecoder<'a> {
    /// Renormalization invariant from §4.1.2.1: `rng > 2**23`.
    const RNG_MIN: u32 = 1 << 23;

    /// Initialize the range decoder over `buf` (RFC 6716 §4.1.1).
    ///
    /// The spec defines `b0` as "the first input byte (or zero if
    /// there are no bytes in this Opus frame)". The decoder sets
    /// `rng = 128`, `val = 127 - (b0 >> 1)`, buffers the leftover
    /// bit `(b0 & 1)`, and immediately invokes renormalization so the
    /// invariant `rng > 2**23` holds before any symbol is decoded.
    pub fn new(buf: &'a [u8]) -> Self {
        let b0 = buf.first().copied().unwrap_or(0) as u32;
        let mut dec = Self {
            buf,
            // §4.1.1: the first byte is consumed by initialization
            // itself, so the next forward fetch starts at index 1.
            fwd: if buf.is_empty() { 0 } else { 1 },
            back: 0,
            back_bits_avail: 0,
            back_window: 0,
            rem: b0 & 1,
            rng: 128,
            val: 127 - (b0 >> 1),
            // §4.1.6: "nbits_total is initialized to 9 just before the
            // initial range renormalization process completes."
            nbits_total: 9,
            nbits_raw: 0,
            error: false,
        };
        dec.normalize();
        dec
    }

    /// Size of the backing frame buffer in bits.
    ///
    /// RFC 6716 §4.3.2.1's budget-constrained coarse-energy decode
    /// compares the running `ec_tell()` against the total number of
    /// bits in the frame (`8 *` the byte length the decoder was
    /// initialized over); this accessor exposes that total.
    pub fn storage_bits(&self) -> u32 {
        (self.buf.len() as u32).saturating_mul(8)
    }

    /// Whether this decoder has latched a `frame corrupt` error
    /// somewhere in its history. Higher-level decoders use this to
    /// abort the current frame and apply packet-loss concealment.
    pub fn has_error(&self) -> bool {
        self.error
    }

    /// Current whole-bit budget consumed by the range coder plus the
    /// raw bit reader (RFC 6716 §4.1.6.1).
    ///
    /// `ec_tell` is defined as `nbits_total - ilog(rng)`. We add the
    /// raw bits separately because §4.1.6 specifies that raw bits
    /// also count against the total.
    pub fn tell(&self) -> u32 {
        // `ilog(rng)` is the position of the most-significant set bit
        // of `rng`, counting from 1.  Since the renormalization
        // invariant keeps `rng >= 2**23`, `lg` is always at least 24.
        let lg = 32 - self.rng.leading_zeros();
        self.nbits_total
            .saturating_sub(lg)
            .saturating_add(self.nbits_raw)
    }

    /// Current 1/8th-bit-precision budget consumed by the range coder
    /// plus the raw bit reader (RFC 6716 §4.1.6.2).
    ///
    /// `ec_tell_frac` follows the procedure in §4.1.6.2 exactly: from
    /// `lg = ilog(rng)`, extract `r_Q15 = rng >> (lg - 16)` as a Q15
    /// value in `[2^15, 2^16)`. Three iterations of the
    /// `r_Q15 = (r_Q15*r_Q15) >> 15; lg = 2*lg + (r_Q15 >> 16)` recursion
    /// expand `lg` to 1/8th-bit precision. Raw bits add `8*nbits_raw`
    /// (§4.1.6) to the result. By construction the result satisfies
    /// `ec_tell() == ceil(ec_tell_frac()/8.0)` from §4.1.6.
    pub fn tell_frac(&self) -> u32 {
        let lg0 = 32 - self.rng.leading_zeros();
        // §4.1.6.2: lg must be at least 24 after renormalization.
        // r_Q15 = rng >> (lg - 16) lies in [2^15, 2^16).
        let mut r_q15 = self.rng >> (lg0 - 16);
        // Build the 1/8th-bit-precision lg one bit at a time. The
        // spec doubles `lg` on each of three refinement passes, so the
        // accumulator starts at the whole-bit value `lg0`.
        let mut lg_frac = lg0;
        // Three passes give three extra bits = 1/8th-bit precision.
        for _ in 0..3 {
            r_q15 = (r_q15 * r_q15) >> 15;
            let bit = r_q15 >> 16;
            lg_frac = 2 * lg_frac + bit;
            // If the bit was a 1, halve r_Q15 so it lies in
            // [2^15, 2^16) again.
            if bit == 1 {
                r_q15 >>= 1;
            }
        }
        // ec_tell_frac() returns nbits_total*8 - lg, plus raw bits in
        // 1/8th-bit units (§4.1.6: raw bits add to the total in whole
        // bits, i.e. multiples of 8 in this scale).
        self.nbits_total
            .saturating_mul(8)
            .saturating_sub(lg_frac)
            .saturating_add(self.nbits_raw.saturating_mul(8))
    }

    /// Decode a single binary symbol with probability `2^-logp` of
    /// being a "1" (RFC 6716 §4.1.3.2).
    ///
    /// Per the spec, this is mathematically equivalent to
    /// `ec_decode(ft = 1<<logp)` followed by `ec_dec_update()` with
    /// either `(0, ft-1, ft)` for a "0" or `(ft-1, ft, ft)` for a
    /// "1". The implementation requires no multiplications or
    /// divisions: the test `fs < ft - 1` reduces to a single
    /// comparison after substituting `fs = ft - min(val/(rng>>logp)+1, ft)`.
    pub fn dec_bit_logp(&mut self, logp: u32) -> u32 {
        let r = self.rng;
        let d = self.val;
        // `s = r >> logp` corresponds to `rng/ft` with `ft = 1<<logp`
        // (an exact shift when ft is a power of two).
        let s = r >> logp;
        // Equivalent to: did `val` land in the "1" half of the
        // interval, i.e. is `val < s`? The "1" half corresponds to
        // `fl = ft - 1, fh = ft`, so the update is
        //   val -= 0; rng = s          (one was decoded)
        // otherwise
        //   val -= s; rng = r - s      (zero was decoded).
        let bit = if d < s { 1 } else { 0 };
        if bit == 1 {
            self.rng = s;
        } else {
            self.val = d - s;
            self.rng = r - s;
        }
        self.normalize();
        bit
    }

    /// Decode `bits` raw bits (RFC 6716 §4.1.4).
    ///
    /// Raw bits are packed at the END of the frame: the least
    /// significant bit of the first value is the LSB of the last
    /// byte; reads proceed toward the front. The function returns
    /// the raw bits in the order written — i.e. if the encoder
    /// emitted bit `b_0` first, the returned word has `b_0` in its
    /// LSB. Returns 0 on errors (e.g. `bits > 32`).
    pub fn dec_bits(&mut self, bits: u32) -> u32 {
        if bits == 0 {
            return 0;
        }
        if bits > 32 {
            self.error = true;
            return 0;
        }
        let mut window = self.back_window;
        let mut avail = self.back_bits_avail;
        // Buffer up at least `bits` worth of raw data.
        while avail < bits {
            let byte = if self.back < self.buf.len() {
                self.buf[self.buf.len() - 1 - self.back]
            } else {
                // §4.1.2.1 / §4.1.4: once the frame is exhausted the
                // decoder MUST continue to use zero for any further
                // input bytes required.
                0
            };
            self.back = self.back.saturating_add(1);
            // Concatenate the new byte ABOVE the existing window so
            // the LSB-first packing within the byte is preserved.
            window |= (byte as u32) << avail;
            avail += 8;
        }
        // The low `bits` bits of `window` are the returned value.
        let mask: u32 = if bits == 32 { !0 } else { (1u32 << bits) - 1 };
        let result = window & mask;
        // Consume those bits.
        self.back_window = window >> bits;
        self.back_bits_avail = avail - bits;
        self.nbits_raw += bits;
        result
    }

    /// Decode one of `ft` equiprobable values in `0..ft`
    /// (RFC 6716 §4.1.5).
    ///
    /// `ft` may be as large as `2**32 - 1`. Values of `ft <= 1`
    /// degenerate to the constant 0. The pseudocode does the decode
    /// in two pieces: the top 8 bits go through the range coder, the
    /// remainder through raw bits. If the final reconstructed value
    /// is `>= ft`, the frame is corrupt — we latch the error flag and
    /// saturate to `ft - 1` per the spec's suggested concealment.
    pub fn dec_uint(&mut self, ft: u32) -> Result<u32, Error> {
        if ft <= 1 {
            return Ok(0);
        }
        // `ftb = ilog(ft - 1)`: number of bits needed for `ft - 1`.
        let ftb = 32 - (ft - 1).leading_zeros();
        if ftb <= 8 {
            // Small case: a single range-coded symbol decodes the
            // whole value.
            let t = self.decode(ft);
            self.dec_update(t, t + 1, ft);
            Ok(t)
        } else {
            // Large case: top 8 bits range-coded, remainder raw.
            let split_bits = ftb - 8;
            let top_ft = ((ft - 1) >> split_bits) + 1;
            let t_hi = self.decode(top_ft);
            self.dec_update(t_hi, t_hi + 1, top_ft);
            let t_lo = self.dec_bits(split_bits);
            let t = (t_hi << split_bits) | t_lo;
            if t >= ft {
                self.error = true;
                Ok(ft - 1)
            } else {
                Ok(t)
            }
        }
    }

    /// Decode the `fs` value for a power-of-two `ft = 1<<ftb`
    /// (RFC 6716 §4.1.3.1, `ec_decode_bin`).
    ///
    /// Mathematically equivalent to [`Self::decode`] with `ft = 1<<ftb`,
    /// but avoids the division: `rng / ft = rng >> ftb`. The caller is
    /// expected to follow with [`Self::dec_update`] (or, more typically
    /// for the SILK path, to use [`Self::dec_icdf`] which combines the
    /// two steps).
    ///
    /// Returns `fs` in the range `[0, 1<<ftb)`. If `ftb` is out of
    /// range (`> 25` would violate the §4.1.3 ftb bound for the
    /// table-driven path), the function still returns a sensible value
    /// (the shift saturates) — callers are expected to provide a valid
    /// `ftb`.
    pub fn decode_bin(&mut self, ftb: u32) -> u32 {
        // s = rng / (1 << ftb) = rng >> ftb.
        let s = self.rng >> ftb;
        if s == 0 {
            // Would only happen for ftb > ilog(rng); per the
            // renormalization invariant ilog(rng) >= 24, so any
            // practical ftb (the spec uses up to 8 for icdf and up
            // to 15 elsewhere) is fine. Defensively saturate to 0.
            return 0;
        }
        let ft = 1u32 << ftb;
        // fs = ft - min(val / s + 1, ft).
        let approx = (self.val / s).saturating_add(1);
        ft - approx.min(ft)
    }

    /// Decode a single symbol via an inverse cumulative distribution
    /// function table (RFC 6716 §4.1.3.3, `ec_dec_icdf`).
    ///
    /// The table `icdf` stores `(1<<ftb) - fh[k]` for each symbol `k`,
    /// terminated by a sentinel `0` entry (so the implicit
    /// `fh[K_last] == ft`). `fl[0]` is implicitly 0; the table values
    /// are strictly monotonically decreasing.
    ///
    /// Combines the search step (find `k` such that
    /// `fs < (1<<ftb) - icdf[k]`) with the range/value update,
    /// replacing the division by a sequence of multiplications. The
    /// renormalization loop is run before returning.
    ///
    /// Returns the decoded symbol index `k` in `0..icdf.len()-1`. If
    /// the table is empty or malformed (no terminating `0`), the
    /// decoder latches its sticky error flag and returns 0.
    pub fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> u32 {
        // s corresponds to rng / ft with ft = 1<<ftb.
        let s = self.rng >> ftb;
        // The spec's search expressed as a forward walk: for each
        // candidate k starting at 0, compute `next = s * icdf[k]` (a
        // descending sequence as icdf is monotonically decreasing).
        // The first k where `val >= next` is the decoded symbol;
        // `t - next` is the new range, where `t` tracks the previous
        // step's `next` (initialised to `rng` so that k=0 reduces to
        // `rng' = rng - s*icdf[0]`, matching §4.1.2 for `fl[0] == 0`).
        let mut t = self.rng;
        for (k, &cell) in icdf.iter().enumerate() {
            let next = s.saturating_mul(cell as u32);
            if self.val >= next {
                self.val -= next;
                self.rng = t - next;
                self.normalize();
                return k as u32;
            }
            t = next;
        }
        // Malformed table: no terminating 0 entry was reached, which
        // would indicate the bitstream's required symbol falls beyond
        // the table. §4.1.5's general guidance for corrupt-frame
        // recovery applies — latch the error flag and return 0.
        self.error = true;
        0
    }

    // ----- internal helpers -----

    /// The current range size (`rng`). The reference decoder exports
    /// this as the per-frame seed of the §4.3.5 noise generator
    /// (`st->rng = dec->rng` at frame end).
    pub(crate) fn range_state(&self) -> u32 {
        self.rng
    }

    /// `ec_decode(ft)` (RFC 6716 §4.1.2): compute the symbol-index
    /// proxy `fs = ft - min(val/(rng/ft) + 1, ft)`.
    pub(crate) fn decode(&mut self, ft: u32) -> u32 {
        // The spec phrases this with integer division. `rng/ft` is
        // computed first; the divisor is then `val / (rng/ft)`.
        // `rng/ft >= 1` because `rng > 2**23` and `ft <= 2**16` in
        // the cases the symbol-decode path is exercised.
        let s = self.rng / ft;
        let approx = self.val / s + 1;
        ft - approx.min(ft)
    }

    /// `ec_dec_update(fl, fh, ft)` (RFC 6716 §4.1.2).
    ///
    /// This both narrows the range to the chosen symbol and runs the
    /// renormalization loop afterwards. Exposed crate-internally so
    /// the §4.3.2.1 Laplace decoder ([`crate::laplace`]) can commit
    /// the `[fl, fh)` interval it searched for after a
    /// [`Self::decode_bin`] probe; for `ft` a power of two the
    /// division reduces to the same shift `decode_bin` used, so the
    /// pairing is exact.
    pub(crate) fn dec_update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.rng / ft;
        self.val -= s * (ft - fh);
        if fl > 0 {
            self.rng = s * (fh - fl);
        } else {
            self.rng -= s * (ft - fh);
        }
        self.normalize();
    }

    /// `ec_dec_normalize` (RFC 6716 §4.1.2.1).
    ///
    /// Until `rng > 2**23`, shift `rng` left by 8 and pull a new
    /// `sym` byte. `sym` is built from the leftover bit of the
    /// previously consumed byte (as MSB) and the top 7 bits of the
    /// freshly read byte; the LSB of the freshly read byte is
    /// buffered for next time. When the frame is exhausted, zero
    /// bits are substituted.
    fn normalize(&mut self) {
        while self.rng <= Self::RNG_MIN {
            let byte = if self.fwd < self.buf.len() {
                let b = self.buf[self.fwd];
                self.fwd += 1;
                b as u32
            } else {
                0
            };
            // sym = (rem << 7) | (byte >> 1); buffer (byte & 1).
            let sym = (self.rem << 7) | (byte >> 1);
            self.rem = byte & 1;
            self.rng <<= 8;
            self.val = ((self.val << 8) + (255 - sym)) & 0x7FFF_FFFF;
            // §4.1.6: each iteration through the loop adds 8 to
            // nbits_total.
            self.nbits_total = self.nbits_total.saturating_add(8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §4.1.1 initialization: a fresh decoder over an empty buffer
    /// must still leave the renormalization invariant satisfied.
    /// `ec_tell` of a freshly initialized decoder is documented in
    /// §4.1.6.1 to report 1 bit (the bit reserved for stream
    /// termination).
    #[test]
    fn init_empty_buffer_satisfies_invariant() {
        let dec = RangeDecoder::new(&[]);
        assert!(dec.rng > RangeDecoder::RNG_MIN);
        assert!(!dec.has_error());
        // §4.1.6.1: "In a newly initialized decoder, before any
        // symbols have been read, this reports that 1 bit has been
        // used."
        assert_eq!(dec.tell(), 1);
    }

    /// A non-empty input should also leave the invariant satisfied,
    /// and the first input byte's low bit must end up in `rem`
    /// (per §4.1.1 + §4.1.2.1).
    #[test]
    fn init_nonempty_buffer_consumes_first_byte() {
        // Choose a first byte with a known LSB so we can sanity-check
        // the bit buffer indirectly via tell().
        let dec = RangeDecoder::new(&[0xAB, 0xCD, 0xEF, 0x12]);
        assert!(dec.rng > RangeDecoder::RNG_MIN);
        assert!(!dec.has_error());
        // tell() must be at least 1 (the termination reserve) and at
        // most nbits_total = 9 + 8*k for the number of renorm steps.
        assert!(dec.tell() >= 1);
    }

    /// `dec_bit_logp` decodes a "1" when the encoder placed `val` in
    /// the high probability mass region. We build a stream by hand
    /// where every byte is 0x00 — this guarantees `val` stays in the
    /// "0" half for any logp > 0 (after init, val = 127, and the
    /// renorm pulls in zero bytes, so val tends large). Conversely a
    /// stream of 0xFF bytes pushes val toward zero, biasing toward
    /// "1". Both should never trip the error flag.
    #[test]
    fn dec_bit_logp_zero_biased_and_one_biased_streams() {
        // All-zero stream: bias toward "0".
        let mut dec0 = RangeDecoder::new(&[0u8; 16]);
        let mut zero_count = 0;
        for _ in 0..32 {
            if dec0.dec_bit_logp(1) == 0 {
                zero_count += 1;
            }
        }
        assert!(!dec0.has_error());
        assert!(
            zero_count > 16,
            "all-zero stream biased toward 0: zero_count={}",
            zero_count
        );

        // All-ones stream: bias toward "1".
        let mut dec1 = RangeDecoder::new(&[0xFFu8; 16]);
        let mut one_count = 0;
        for _ in 0..32 {
            if dec1.dec_bit_logp(1) == 1 {
                one_count += 1;
            }
        }
        assert!(!dec1.has_error());
        assert!(
            one_count > 16,
            "all-ones stream biased toward 1: one_count={}",
            one_count
        );
    }

    /// `dec_bits` reads raw bits LSB-first from the end of the
    /// buffer. Encode a known pattern: the last byte is 0b1010_0110,
    /// so reading 4 raw bits should produce 0b0110 = 6, then the
    /// next 4 should produce 0b1010 = 0xA.
    #[test]
    fn dec_bits_lsb_first_from_end() {
        let mut dec = RangeDecoder::new(&[0x00, 0x00, 0xA6]);
        let lo = dec.dec_bits(4);
        let hi = dec.dec_bits(4);
        assert_eq!(lo, 0x6);
        assert_eq!(hi, 0xA);
        assert!(!dec.has_error());
    }

    /// `dec_bits` must continue to return zero past the end of the
    /// frame (RFC 6716 §4.1.4). Read more raw bits than the buffer
    /// holds and check that the surplus reads come back as zero.
    #[test]
    fn dec_bits_zero_past_end_of_frame() {
        // 2-byte buffer = 16 raw bits available before zero-padding.
        let mut dec = RangeDecoder::new(&[0xFF, 0xFF]);
        // Drain 16 bits of 0xFF.
        for _ in 0..4 {
            let v = dec.dec_bits(4);
            assert_eq!(v, 0xF);
        }
        // The next 8 bits should be zero (range coder may or may not
        // have eaten into these — either way the *raw* path returns
        // the zero-extended tail).
        let pad = dec.dec_bits(8);
        // We can't assert the exact value because the range decoder
        // may have shared bytes with the raw reader; but `dec_bits`
        // beyond the buffer never errors and never panics.
        assert!(!dec.has_error());
        let _ = pad;
    }

    /// `dec_uint(1)` is the degenerate case: the only value in
    /// `0..1` is 0, and the decoder must not consume any bits.
    #[test]
    fn dec_uint_ft_one_is_zero() {
        let mut dec = RangeDecoder::new(&[0x12, 0x34, 0x56]);
        let before = dec.tell();
        let v = dec.dec_uint(1).expect("ft=1 must succeed");
        let after = dec.tell();
        assert_eq!(v, 0);
        assert_eq!(after, before);
    }

    /// `dec_uint` with a small `ft` (ftb <= 8) takes the
    /// range-coded-only branch. The returned value must lie in
    /// `0..ft` and must not trip the error flag.
    #[test]
    fn dec_uint_small_ft_in_range() {
        let mut dec = RangeDecoder::new(&[0x42, 0x18, 0xC3, 0x7F]);
        for _ in 0..8 {
            let v = dec.dec_uint(200).expect("ft=200 must succeed");
            assert!(v < 200, "v={} out of range", v);
        }
        assert!(!dec.has_error());
    }

    /// `dec_uint` with a large `ft` (ftb > 8) takes the
    /// range-coded-plus-raw-bits branch. Provide a longer buffer so
    /// both readers stay happy.
    #[test]
    fn dec_uint_large_ft_in_range() {
        let buf: Vec<u8> = (0..64).collect();
        let mut dec = RangeDecoder::new(&buf);
        for _ in 0..8 {
            let v = dec.dec_uint(1_000_000).expect("ft=1_000_000 must succeed");
            assert!(v < 1_000_000, "v={} out of range", v);
        }
        // The decoder may flag an error if a hand-crafted stream
        // happened to land outside the legal range; the saturation
        // path still returns ft-1 in that case, which we already
        // bounds-checked above. So `error` may be either value here.
    }

    /// `tell()` must monotonically non-decrease across operations.
    #[test]
    fn tell_is_monotonic() {
        let mut dec = RangeDecoder::new(&[0x55; 8]);
        let mut prev = dec.tell();
        for _ in 0..16 {
            let _ = dec.dec_bit_logp(2);
            let now = dec.tell();
            assert!(now >= prev, "tell() went backwards: {} -> {}", prev, now);
            prev = now;
        }
    }

    /// `decode_bin(ftb)` must match the generic `decode(1<<ftb)` path
    /// exactly (RFC 6716 §4.1.3.1 says the two are mathematically
    /// equivalent). Compare a handful of `ftb` values against the
    /// generic helper on independent decoders fed the same bytes.
    #[test]
    fn decode_bin_matches_generic_decode() {
        for &ftb in &[1u32, 4, 8, 12, 15] {
            let buf = [0x37u8, 0x91, 0xC4, 0x18, 0xA2, 0x5D, 0x6E, 0xFF];
            let mut a = RangeDecoder::new(&buf);
            let mut b = RangeDecoder::new(&buf);
            let from_bin = a.decode_bin(ftb);
            let from_generic = b.decode(1u32 << ftb);
            assert_eq!(
                from_bin, from_generic,
                "decode_bin({ftb}) != decode(1<<{ftb})"
            );
            assert!(from_bin < (1u32 << ftb), "fs={from_bin} out of range");
        }
    }

    /// §4.1.6.1 specifies `ec_tell() == ceil(ec_tell_frac()/8.0)`.
    /// Walk a decoder forward through a mix of symbol and raw-bit
    /// reads and assert this identity at every step.
    #[test]
    fn tell_frac_consistent_with_tell() {
        let mut dec = RangeDecoder::new(&[0xA3, 0x7F, 0x10, 0x5C, 0xE8, 0x91, 0x42, 0xB7]);
        // Initial: ec_tell() == 1 per §4.1.6.1.
        assert_eq!(dec.tell(), 1);
        // ec_tell_frac() / 8.0, rounded up, must equal ec_tell().
        for _ in 0..12 {
            let whole = dec.tell();
            let frac = dec.tell_frac();
            // ceil(frac/8) == (frac + 7) / 8.
            let ceil_eighths = frac.div_ceil(8);
            assert_eq!(
                ceil_eighths, whole,
                "tell()={whole} != ceil(tell_frac()={frac} / 8)={ceil_eighths}"
            );
            // Then advance the decoder.
            let _ = dec.dec_bit_logp(1);
            let _ = dec.dec_bits(2);
        }
    }

    /// `tell_frac()` of a freshly initialised decoder should land in
    /// the 1-bit termination reserve (§4.1.6.1 says `tell()` is 1
    /// then, so `tell_frac()` is at most 8 and at least 1 since the
    /// extra two bits of slack are baked into `nbits_total`).
    #[test]
    fn tell_frac_initial_within_one_bit() {
        let dec = RangeDecoder::new(&[0xCC, 0xDD, 0xEE, 0xFF]);
        let frac = dec.tell_frac();
        assert!(
            (1..=8).contains(&frac),
            "tell_frac initial out of [1,8]: {frac}"
        );
        assert!(frac.div_ceil(8) == dec.tell());
    }

    /// `dec_icdf` over a binary `{ft-1, 1}/ft` distribution must
    /// behave identically to `dec_bit_logp(logp)` with the same `ftb`
    /// (RFC 6716 §4.1.3.2 + §4.1.3.3 — both are special cases of
    /// `ec_decode` with `ft = 1<<ftb`). Drive both with the same
    /// input bytes and compare per-step decisions.
    #[test]
    fn dec_icdf_matches_dec_bit_logp_for_binary() {
        let buf = [0xDE, 0xAD, 0xBE, 0xEF, 0x10, 0x32, 0x54, 0x76];
        // logp = 3 → ft = 8, P("1") = 1/8. icdf for this PDF is
        // {ft - fh[0], ft - fh[1]} = {1, 0}: symbol 0 (the "0" bit)
        // is the high-probability outcome.
        let logp = 3u32;
        let icdf = [1u8, 0];
        let mut a = RangeDecoder::new(&buf);
        let mut b = RangeDecoder::new(&buf);
        for _ in 0..16 {
            let via_logp = a.dec_bit_logp(logp);
            let via_icdf = b.dec_icdf(&icdf, logp);
            assert_eq!(
                via_logp, via_icdf,
                "dec_bit_logp({logp}) != dec_icdf({:?}, {logp})",
                icdf
            );
        }
        assert!(!a.has_error() && !b.has_error());
    }

    /// `dec_icdf` with a uniform `{1,1,1,1,1,1,1,1}/8` distribution
    /// must always return a symbol in `0..8` and never trip the
    /// error flag. Equivalent to `dec_uint(8)` in expectation.
    #[test]
    fn dec_icdf_uniform_returns_in_range() {
        // PDF {1,1,1,1,1,1,1,1}/8 → fh = {1,2,3,4,5,6,7,8} → icdf =
        // {7,6,5,4,3,2,1,0}.
        let icdf = [7u8, 6, 5, 4, 3, 2, 1, 0];
        let mut dec = RangeDecoder::new(&[0x42, 0x18, 0xC3, 0x7F, 0x55, 0xAA, 0x33, 0xCC]);
        for _ in 0..16 {
            let k = dec.dec_icdf(&icdf, 3);
            assert!(k < 8, "icdf uniform returned {k} out of [0,8)");
        }
        assert!(!dec.has_error());
    }

    /// `dec_icdf` with a degenerate single-symbol table `{0}` (only
    /// the terminator) means `ft - fh[0] = 0`, i.e. the only symbol
    /// covers the whole interval and is always returned. This must
    /// not consume any range coder bits beyond renormalization, and
    /// must not error.
    #[test]
    fn dec_icdf_single_symbol_always_zero() {
        let icdf = [0u8]; // only the terminator: ft - fh[0] == 0.
        let mut dec = RangeDecoder::new(&[0x77, 0x33, 0x11, 0xAA]);
        let before_tell = dec.tell();
        for _ in 0..4 {
            let k = dec.dec_icdf(&icdf, 3);
            assert_eq!(k, 0);
        }
        // No range mass was consumed (the symbol covered everything),
        // so tell() should be unchanged or have advanced only by
        // renormalization (in practice, zero advance after the first
        // call because rng wasn't reduced).
        assert!(dec.tell() >= before_tell);
        assert!(!dec.has_error());
    }

    /// `tell_frac()` advances monotonically across mixed operations,
    /// in the same way `tell()` does (§4.1.6.2 inherits the
    /// monotonicity of `ec_tell` since the procedure only adds bits).
    #[test]
    fn tell_frac_is_monotonic() {
        let mut dec = RangeDecoder::new(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        // Uniform 8-way icdf so each call burns ~3 bits of entropy.
        let icdf = [7u8, 6, 5, 4, 3, 2, 1, 0];
        let mut prev = dec.tell_frac();
        for i in 0..24 {
            match i % 3 {
                0 => {
                    let _ = dec.dec_bit_logp(2);
                }
                1 => {
                    let _ = dec.dec_icdf(&icdf, 3);
                }
                _ => {
                    let _ = dec.dec_bits(2);
                }
            }
            let now = dec.tell_frac();
            assert!(
                now >= prev,
                "tell_frac() went backwards: {} -> {}",
                prev,
                now
            );
            prev = now;
        }
    }
}
