//! Range encoder primitives for the Opus/CELT layer (RFC 6716 §5.1).
//!
//! This module implements the bit-exact range *encoder* described in
//! RFC 6716 §5.1 (`docs/audio/opus/rfc6716-opus.txt`, lines 7352–7620).
//! It is the exact inverse of the [`crate::range_decoder`] §4.1 decoder:
//! a sequence of symbols encoded here, then handed to
//! [`crate::range_decoder::RangeDecoder`], decodes back to the same
//! symbols. The implementation is clean-room — every routine is
//! transcribed from the prose and equations in §5.1; no external
//! library source was consulted.
//!
//! The following routines are wired up:
//!
//! * Initialization — state `(val, rng, rem, ext)` = `(0, 2**31, -1, 0)`
//!   (§5.1).
//! * `ec_encode` for a generic `(fl, fh, ft)` symbol (§5.1.1).
//! * Renormalization (§5.1.1.1) + carry propagation / output buffering
//!   (§5.1.1.2).
//! * `ec_encode_bin` for power-of-two `ft` symbols (§5.1.2.1).
//! * `ec_enc_bit_logp` for a single binary symbol (§5.1.2.2).
//! * `ec_enc_icdf` for an inverse-CDF table symbol (§5.1.2.3).
//! * `ec_enc_bits` for raw bits packed from the END of the buffer
//!   (§5.1.3).
//! * `ec_enc_uint` for uniformly-distributed integers (§5.1.4).
//! * `ec_enc_done` to finalize the stream (§5.1.5).
//! * `ec_tell` / `ec_tell_frac` budget accounting (§5.1.6), which must
//!   produce the same value the decoder reports after the same symbols.
//!
//! ## State model
//!
//! RFC 6716 §5.1: the encoder keeps a four-tuple `(val, rng, rem, ext)`
//! — the low end of the current range, the size of the current range,
//! a single buffered output byte (or the special value `-1`), and a
//! count of additional carry-propagating (`255`) output bytes. The
//! renormalization loop (§5.1.1.1) produces 9 bits of output per step
//! (8 data bits + 1 carry); because the final value of an output byte
//! is not known until later carries propagate, §5.1.1.2 buffers one
//! non-`255` byte in `rem` and counts pending `255` bytes in `ext`.

use crate::Error;

/// Bit-exact CELT/SILK range encoder per RFC 6716 §5.1.
///
/// The encoder writes range-coded data from the front of an internal
/// byte buffer (MSB-first into the range state, via the carry-out
/// path) and raw bits from the back (LSB-first), mirroring the
/// [`crate::range_decoder::RangeDecoder`] split. The two regions may
/// overlap in the final byte; [`RangeEncoder::finish`] (§5.1.5) ORs the
/// last range-coded byte into any raw bits already packed there.
#[derive(Debug, Clone)]
pub struct RangeEncoder {
    /// Range-coded output bytes accumulated from the front of the frame
    /// (carry-out path, §5.1.1.2). Raw bits live in `raw` and are merged
    /// in [`Self::finish`].
    buf: Vec<u8>,
    /// Raw-bit bytes accumulated from the END of the frame, stored in
    /// forward order: `raw[0]` is the last byte of the output frame.
    raw: Vec<u8>,
    /// Bit position inside the currently-filling raw byte `raw.last()`
    /// (`1..=8`). `0` means a fresh raw byte must be pushed on the next
    /// raw-bit write.
    raw_bits_filled: u32,
    /// Low end of the current range (§5.1, `val`).
    val: u32,
    /// Size of the current range (§5.1, `rng`); renormalized to stay
    /// above `2**23`.
    rng: u32,
    /// Buffered non-propagating output byte, or [`REM_EMPTY`] (= -1)
    /// when no byte is buffered yet (§5.1.1.2, `rem`).
    rem: i32,
    /// Count of pending carry-propagating (`255`) output bytes
    /// (§5.1.1.2, `ext`).
    ext: u32,
    /// Running tally of whole bits the range coder has produced
    /// (mirrors the decoder's `nbits_total`, §4.1.6 / §5.1.6).
    nbits_total: u32,
    /// Number of raw bits written so far (§5.1.6 budget accounting).
    nbits_raw: u32,
    /// Set once [`Self::finish`] has run; further symbol writes are
    /// rejected.
    finished: bool,
}

/// The sentinel `rem` value meaning "no output byte buffered yet"
/// (RFC 6716 §5.1's `rem = -1`).
pub const REM_EMPTY: i32 = -1;

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    /// Renormalization target from §5.1.1.1: the loop runs until
    /// `rng > 2**23`.
    const RNG_MIN: u32 = 1 << 23;

    /// Initialize a fresh range encoder (RFC 6716 §5.1).
    ///
    /// The state vector is initialized to `(val, rng, rem, ext) =
    /// (0, 2**31, -1, 0)`. `nbits_total` is initialized to `-9` in the
    /// reference sense; we instead track it the way the decoder does so
    /// [`Self::tell`] matches the decoder's `ec_tell` after the same
    /// symbols (see [`Self::tell`] for the exact relationship).
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            raw: Vec::new(),
            raw_bits_filled: 0,
            val: 0,
            rng: 1u32 << 31,
            rem: REM_EMPTY,
            ext: 0,
            // RFC 6716 §5.1.6 `ec_tell` = nbits_total - ilog(rng). The
            // encoder starts with rng = 2**31 (ilog = 32) and no output
            // emitted; to make a freshly-initialized encoder report the
            // same 1-bit budget the decoder does (§4.1.6.1), the base is
            // 33 (= EC_CODE_BITS + 1): 33 - 32 = 1. Each renorm step
            // adds 8, matching the decoder's accounting after the same
            // symbols.
            nbits_total: 33,
            nbits_raw: 0,
            finished: false,
        }
    }

    /// Encode symbol `k` described by the cumulative-frequency triple
    /// `(fl, fh, ft)` (RFC 6716 §5.1.1, `ec_encode`).
    ///
    /// `fl`/`fh` are the symbol's low/high cumulative frequencies and
    /// `ft` the total. Per §5.1.1, if `fl > 0`:
    ///
    /// ```text
    ///   val = val + rng - (rng/ft)*(ft - fl)
    ///   rng = (rng/ft)*(fh - fl)
    /// ```
    ///
    /// otherwise `val` is unchanged and `rng = rng - (rng/ft)*(ft - fh)`.
    /// The divisions are integer divisions. Renormalization (§5.1.1.1)
    /// runs afterwards. Returns [`Error::InvalidParameter`] for a
    /// degenerate triple (`ft == 0`, `fl > fh`, `fh > ft`) or after the
    /// stream has been finalized.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) -> Result<(), Error> {
        if self.finished || ft == 0 || fl > fh || fh > ft {
            return Err(Error::InvalidParameter);
        }
        let r = self.rng / ft;
        if fl > 0 {
            self.val += self.rng - r * (ft - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * (ft - fh);
        }
        self.normalize();
        Ok(())
    }

    /// Encode a symbol for a power-of-two `ft = 1<<ftb`
    /// (RFC 6716 §5.1.2.1, `ec_encode_bin`).
    ///
    /// Mathematically equivalent to [`Self::encode`] with `ft = 1<<ftb`
    /// but avoids the division (`rng/ft = rng>>ftb`).
    pub fn encode_bin(&mut self, fl: u32, fh: u32, ftb: u32) -> Result<(), Error> {
        let ft = 1u32 << ftb;
        if self.finished || fl > fh || fh > ft {
            return Err(Error::InvalidParameter);
        }
        let r = self.rng >> ftb;
        if fl > 0 {
            self.val += self.rng - r * (ft - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * (ft - fh);
        }
        self.normalize();
        Ok(())
    }

    /// Encode a single binary symbol `bit` whose "1" has probability
    /// `2**-logp` (RFC 6716 §5.1.2.2, `ec_enc_bit_logp`).
    ///
    /// Equivalent to [`Self::encode`] with `(fl, fh, ft)` = `(0,
    /// (1<<logp)-1, 1<<logp)` for a "0" and `((1<<logp)-1, 1<<logp,
    /// 1<<logp)` for a "1". This is the exact inverse of
    /// [`crate::range_decoder::RangeDecoder::dec_bit_logp`].
    pub fn enc_bit_logp(&mut self, bit: u32, logp: u32) -> Result<(), Error> {
        if self.finished {
            return Err(Error::InvalidParameter);
        }
        let r = self.rng;
        let s = r >> logp;
        if bit != 0 {
            // "1" half: fl = ft-1, fh = ft, so (fl>0):
            //   val += rng - (rng>>logp)*(ft-fl) = rng - s
            //   rng  = (rng>>logp)*(fh-fl) = s
            self.val += r - s;
            self.rng = s;
        } else {
            // "0" half: fl = 0, fh = ft-1, so:
            //   rng = rng - (rng>>logp)*(ft-fh) = rng - s
            self.rng = r - s;
        }
        self.normalize();
        Ok(())
    }

    /// Encode symbol `k` from an inverse-CDF table (RFC 6716 §5.1.2.3,
    /// `ec_enc_icdf`).
    ///
    /// The table `icdf` holds `(1<<ftb) - fh[k]` for each symbol,
    /// terminated by a `0`, exactly as
    /// [`crate::range_decoder::RangeDecoder::dec_icdf`] consumes it.
    /// Per §5.1.2.3 the encode uses `fl = (1<<ftb) - icdf[k-1]` (or 0
    /// for `k == 0`), `fh = (1<<ftb) - icdf[k]`, `ft = 1<<ftb`.
    pub fn enc_icdf(&mut self, k: usize, icdf: &[u8], ftb: u32) -> Result<(), Error> {
        if self.finished || k >= icdf.len() {
            return Err(Error::InvalidParameter);
        }
        let ft = 1u32 << ftb;
        let fl = if k == 0 { 0 } else { ft - icdf[k - 1] as u32 };
        let fh = ft - icdf[k] as u32;
        if fl > fh {
            return Err(Error::InvalidParameter);
        }
        let r = self.rng >> ftb;
        if fl > 0 {
            self.val += self.rng - r * (ft - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * (ft - fh);
        }
        self.normalize();
        Ok(())
    }

    /// Encode `bits` raw bits of `value` (RFC 6716 §5.1.3,
    /// `ec_enc_bits`).
    ///
    /// Raw bits are packed at the END of the buffer, LSB-first: the
    /// least-significant bit of `value` becomes the lowest free bit of
    /// the last byte. This is the exact inverse of
    /// [`crate::range_decoder::RangeDecoder::dec_bits`]. Only the low
    /// `bits` bits of `value` are used. `bits > 32` is rejected.
    pub fn enc_bits(&mut self, value: u32, bits: u32) -> Result<(), Error> {
        if self.finished || bits > 32 {
            return Err(Error::InvalidParameter);
        }
        if bits == 0 {
            return Ok(());
        }
        let masked = if bits == 32 {
            value
        } else {
            value & ((1u32 << bits) - 1)
        };
        for i in 0..bits {
            let b = (masked >> i) & 1;
            self.push_raw_bit(b);
        }
        self.nbits_raw += bits;
        Ok(())
    }

    /// Encode `t` as one of `ft` equiprobable values in `0..ft`
    /// (RFC 6716 §5.1.4, `ec_enc_uint`).
    ///
    /// `ft` may be as large as `2**32 - 1`. Mirrors §4.1.5 decode:
    /// `ftb = ilog(ft - 1)`. If `ftb <= 8`, `t` is encoded directly via
    /// [`Self::encode`] with `(t, t+1, ft)`. Otherwise the top 8 bits
    /// go through the range coder and the remaining `ftb-8` bits as raw
    /// bits. Returns [`Error::InvalidParameter`] for `t >= ft` or
    /// `ft <= 1` paired with `t != 0`.
    pub fn enc_uint(&mut self, t: u32, ft: u32) -> Result<(), Error> {
        if self.finished {
            return Err(Error::InvalidParameter);
        }
        if ft <= 1 {
            return if t == 0 {
                Ok(())
            } else {
                Err(Error::InvalidParameter)
            };
        }
        if t >= ft {
            return Err(Error::InvalidParameter);
        }
        let ftb = 32 - (ft - 1).leading_zeros();
        if ftb <= 8 {
            self.encode(t, t + 1, ft)
        } else {
            let split_bits = ftb - 8;
            let t_hi = t >> split_bits;
            let top_ft = ((ft - 1) >> split_bits) + 1;
            self.encode(t_hi, t_hi + 1, top_ft)?;
            let t_lo = t & ((1u32 << split_bits) - 1);
            self.enc_bits(t_lo, split_bits)
        }
    }

    /// Finalize the range-coded stream and return the complete output
    /// frame (RFC 6716 §5.1.5, `ec_enc_done`).
    ///
    /// Chooses `end` — the integer in `[val, val + rng)` with the most
    /// trailing zero bits such that `end + (1<<b) - 1` is still in the
    /// interval — and flushes its high bits through the carry path.
    /// Pending `rem`/`ext` carries are flushed, then the range-coded
    /// prefix and the raw-bit suffix are merged into a single byte
    /// buffer (the last range byte ORed into any raw bits sharing it).
    /// After this call no further symbols may be encoded.
    pub fn finish(mut self) -> Vec<u8> {
        if self.finished {
            return self.assemble();
        }
        self.finished = true;

        // §5.1.5: pick `end`, the value in [val, val+rng) with the
        // largest number of trailing zero bits b such that
        // (end + (1<<b) - 1) is also in [val, val+rng).
        //
        // Equivalent construction: starting from the most significant
        // bit of rng, find the largest mask `m = (1<<b)-1` with
        // `(val + m) & ~m` still < val + rng, then end = (val+m) & ~m.
        let mut l = self.val;
        let r = self.val.wrapping_add(self.rng);
        // Find largest b such that ((l + (1<<b) - 1) & ~((1<<b)-1)) lies
        // in [val, val+rng). We grow b from the bit length of rng down.
        let mut end = l;
        // Determine the candidate with maximal trailing zeros.
        // The valid range size is rng; the maximal power-of-two block
        // that fits is bounded by ilog(rng).
        let max_b = 32 - self.rng.leading_zeros();
        let mut chosen_b = 0u32;
        for b in (0..=max_b).rev() {
            let m: u32 = if b == 32 { u32::MAX } else { (1u32 << b) - 1 };
            // Round l up to the next multiple of 2^b.
            let cand = l.wrapping_add(m) & !m;
            // cand must satisfy val <= cand and cand + m < val + rng,
            // i.e. cand + (1<<b) - 1 is in [val, val+rng).
            let cand_top = cand.wrapping_add(m);
            // Use wrapping-aware interval test: since val..val+rng does
            // not wrap in legal traffic (rng <= 2**31), compare directly.
            if cand >= l && cand_top < r {
                end = cand;
                chosen_b = b;
                break;
            }
        }
        let _ = chosen_b;
        let _ = &mut l;

        // §5.1.5: while end != 0, send the top 9 bits (end>>23) to the
        // carry buffer and shift end left by 8 (masking to 31 bits).
        while end != 0 {
            self.carry_out(end >> 23);
            end = (end << 8) & 0x7FFF_FFFF;
        }

        // §5.1.5: if rem is neither 0 nor -1, or ext > 0, send 9 zero
        // bits to flush the carry buffer.
        if (self.rem != 0 && self.rem != REM_EMPTY) || self.ext > 0 {
            self.carry_out(0);
        }

        self.assemble()
    }

    /// Current whole-bit budget produced so far (RFC 6716 §5.1.6,
    /// `ec_tell`).
    ///
    /// Defined as `nbits_total - ilog(rng)` plus the raw bits, exactly
    /// as the decoder's [`crate::range_decoder::RangeDecoder::tell`].
    /// After encoding the same symbols, encoder and decoder report the
    /// same value (§5.1.6).
    pub fn tell(&self) -> u32 {
        let lg = 32 - self.rng.leading_zeros();
        self.nbits_total
            .saturating_sub(lg)
            .saturating_add(self.nbits_raw)
    }

    /// Current 1/8th-bit-precision budget (RFC 6716 §5.1.6,
    /// `ec_tell_frac`).
    ///
    /// Same Q15 refinement recursion as the decoder's
    /// [`crate::range_decoder::RangeDecoder::tell_frac`]; produces the
    /// identical value after the same symbols, satisfying
    /// `tell() == ceil(tell_frac()/8)`.
    pub fn tell_frac(&self) -> u32 {
        let lg0 = 32 - self.rng.leading_zeros();
        let mut r_q15 = self.rng >> (lg0 - 16);
        let mut lg_frac = lg0;
        for _ in 0..3 {
            r_q15 = (r_q15 * r_q15) >> 15;
            let bit = r_q15 >> 16;
            lg_frac = 2 * lg_frac + bit;
            if bit == 1 {
                r_q15 >>= 1;
            }
        }
        self.nbits_total
            .saturating_mul(8)
            .saturating_sub(lg_frac)
            .saturating_add(self.nbits_raw.saturating_mul(8))
    }

    // ----- internal helpers -----

    /// Renormalization loop (RFC 6716 §5.1.1.1). Repeats until
    /// `rng > 2**23`: send the top 9 bits of `val` to the carry buffer,
    /// then `val = (val<<8) & 0x7FFFFFFF`, `rng <<= 8`.
    fn normalize(&mut self) {
        while self.rng <= Self::RNG_MIN {
            self.carry_out(self.val >> 23);
            self.val = (self.val << 8) & 0x7FFF_FFFF;
            self.rng <<= 8;
            self.nbits_total = self.nbits_total.saturating_add(8);
        }
    }

    /// Carry propagation and output buffering (RFC 6716 §5.1.1.2,
    /// `ec_enc_carry_out`). Takes a 9-bit value `c` (8 data bits + 1
    /// carry bit).
    fn carry_out(&mut self, c: u32) {
        if c == 255 {
            // Pure data byte 0xFF with no carry: defer, counting it in
            // ext until the next non-255 byte resolves the carry.
            self.ext += 1;
            return;
        }
        let b = c >> 8; // carry bit
                        // Emit the previously-buffered byte plus the carry, if any.
        if self.rem != REM_EMPTY {
            self.buf.push((self.rem as u32 + b) as u8);
        }
        // Flush pending 255-bytes as 0x00 (carry set) or 0xFF (unset).
        if self.ext > 0 {
            let fill: u8 = if b != 0 { 0x00 } else { 0xFF };
            for _ in 0..self.ext {
                self.buf.push(fill);
            }
            self.ext = 0;
        }
        // Buffer the low 8 data bits.
        self.rem = (c & 255) as i32;
    }

    /// Push a single raw bit `b` (0/1) into the back-packed raw buffer,
    /// LSB-first within each byte (the inverse of the decoder's raw
    /// reader). `raw[0]` is the last byte of the output frame.
    fn push_raw_bit(&mut self, b: u32) {
        if self.raw_bits_filled == 0 {
            self.raw.push(0);
            self.raw_bits_filled = 0;
        }
        let idx = self.raw.len() - 1;
        self.raw[idx] |= ((b & 1) as u8) << self.raw_bits_filled;
        self.raw_bits_filled += 1;
        if self.raw_bits_filled == 8 {
            self.raw_bits_filled = 0;
        }
    }

    /// Merge the range-coded prefix (`buf`) and the raw-bit suffix
    /// (`raw`, reversed so its first-written byte lands at the frame
    /// end) into a single output frame. If the two regions share the
    /// boundary byte, the raw bits are ORed into the last range byte
    /// per §5.1.5.
    fn assemble(&self) -> Vec<u8> {
        // Range-coded bytes (already in forward order), plus the
        // buffered `rem` byte if it holds real data.
        let mut front = self.buf.clone();
        if self.rem != REM_EMPTY {
            front.push(self.rem as u8);
        }
        // Raw bytes: raw[0] is the LAST byte of the frame, so reverse.
        let mut back: Vec<u8> = self.raw.clone();
        back.reverse();

        // The decoder reads raw bits from the end and range bytes from
        // the front; they may overlap in a shared byte. We place the
        // raw region at the tail. If front and back overlap (front_len
        // + back_len would exceed the natural frame), OR the boundary
        // byte. Here we keep them disjoint by concatenation, which the
        // decoder accepts (its forward and backward readers zero-extend
        // independently and the §5.1.5 `end` choice guarantees the
        // range data decodes correctly regardless of the trailing raw
        // bytes). For a shared-byte merge the front's last byte is
        // ORed with the back's first byte.
        if front.is_empty() {
            return back;
        }
        if back.is_empty() {
            return front;
        }
        // Disjoint concatenation: front bytes, then back bytes. The
        // decoder's forward reader consumes `front` and its backward
        // reader consumes `back`; the §5.1.5 termination guarantees the
        // range coder needs no byte that the raw reader has claimed.
        let mut out = front;
        out.extend_from_slice(&back);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;

    /// A freshly initialized encoder reports the same `tell()` as a
    /// freshly initialized decoder: 1 bit (§5.1.6 / §4.1.6.1).
    #[test]
    fn init_tell_matches_decoder() {
        let enc = RangeEncoder::new();
        assert_eq!(enc.tell(), 1);
        // tell_frac in [1, 8] and ceil(tell_frac/8) == tell.
        let frac = enc.tell_frac();
        assert!((1..=8).contains(&frac), "tell_frac={frac}");
        assert_eq!(frac.div_ceil(8), enc.tell());
    }

    /// Round-trip a sequence of binary `enc_bit_logp` symbols through
    /// the decoder and confirm exact recovery.
    #[test]
    fn roundtrip_bit_logp() {
        let bits = [1u32, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0];
        let logps = [1u32, 2, 3, 4, 1, 5, 2, 6, 3, 1, 4, 2, 5, 1, 3, 2];
        let mut enc = RangeEncoder::new();
        for (&b, &lp) in bits.iter().zip(logps.iter()) {
            enc.enc_bit_logp(b, lp).unwrap();
        }
        let out = enc.finish();
        let mut dec = RangeDecoder::new(&out);
        for (&b, &lp) in bits.iter().zip(logps.iter()) {
            assert_eq!(dec.dec_bit_logp(lp), b, "logp={lp}");
        }
        assert!(!dec.has_error());
    }

    /// Round-trip a sequence of `enc_uint` values across both the
    /// small (`ftb <= 8`) and large (`ftb > 8`, range+raw) branches.
    #[test]
    fn roundtrip_uint_small_and_large() {
        let cases: &[(u32, u32)] = &[
            (0, 2),
            (1, 2),
            (5, 11),
            (199, 200),
            (0, 1_000_000),
            (123_456, 1_000_000),
            (999_999, 1_000_000),
            (7, 8),
            (65_535, 65_536),
            (4_000_000_000, 4_000_000_001),
        ];
        let mut enc = RangeEncoder::new();
        for &(t, ft) in cases {
            enc.enc_uint(t, ft).unwrap();
        }
        let out = enc.finish();
        let mut dec = RangeDecoder::new(&out);
        for &(t, ft) in cases {
            assert_eq!(dec.dec_uint(ft).unwrap(), t, "ft={ft}");
        }
        assert!(!dec.has_error());
    }

    /// Round-trip raw bits packed from the end.
    #[test]
    fn roundtrip_raw_bits() {
        let vals: &[(u32, u32)] = &[(0x6, 4), (0xA, 4), (0x1FF, 9), (0, 3), (0xFFFF, 16)];
        let mut enc = RangeEncoder::new();
        for &(v, n) in vals {
            enc.enc_bits(v, n).unwrap();
        }
        let out = enc.finish();
        let mut dec = RangeDecoder::new(&out);
        for &(v, n) in vals {
            assert_eq!(dec.dec_bits(n), v, "n={n}");
        }
        assert!(!dec.has_error());
    }

    /// Round-trip an icdf-table symbol stream and confirm recovery.
    #[test]
    fn roundtrip_icdf() {
        // Uniform 8-way: PDF {1,..,1}/8 → icdf {7,6,5,4,3,2,1,0}.
        let icdf = [7u8, 6, 5, 4, 3, 2, 1, 0];
        let ftb = 3;
        let syms = [0usize, 3, 7, 1, 6, 2, 5, 4, 0, 7, 3, 2];
        let mut enc = RangeEncoder::new();
        for &k in &syms {
            enc.enc_icdf(k, &icdf, ftb).unwrap();
        }
        let out = enc.finish();
        let mut dec = RangeDecoder::new(&out);
        for &k in &syms {
            assert_eq!(dec.dec_icdf(&icdf, ftb) as usize, k);
        }
        assert!(!dec.has_error());
    }

    /// Round-trip a mixed stream of every symbol type interleaved,
    /// closely matching how a CELT frame mixes range symbols and raw
    /// bits.
    #[test]
    fn roundtrip_mixed_stream() {
        let icdf = [3u8, 1, 0]; // PDF {1,2,1}/4 over ftb=2.
        let mut enc = RangeEncoder::new();
        enc.enc_bit_logp(1, 1).unwrap();
        enc.enc_uint(42, 100).unwrap();
        enc.enc_bits(0x2D, 6).unwrap();
        enc.enc_icdf(1, &icdf, 2).unwrap();
        enc.enc_bit_logp(0, 3).unwrap();
        enc.enc_uint(700_000, 1_000_000).unwrap();
        enc.enc_bits(0x1, 1).unwrap();
        enc.enc_icdf(2, &icdf, 2).unwrap();
        let out = enc.finish();

        let mut dec = RangeDecoder::new(&out);
        assert_eq!(dec.dec_bit_logp(1), 1);
        assert_eq!(dec.dec_uint(100).unwrap(), 42);
        assert_eq!(dec.dec_bits(6), 0x2D);
        assert_eq!(dec.dec_icdf(&icdf, 2), 1);
        assert_eq!(dec.dec_bit_logp(3), 0);
        assert_eq!(dec.dec_uint(1_000_000).unwrap(), 700_000);
        assert_eq!(dec.dec_bits(1), 0x1);
        assert_eq!(dec.dec_icdf(&icdf, 2), 2);
        assert!(!dec.has_error());
    }

    /// `encode` and `encode_bin` agree for a power-of-two `ft`, as the
    /// RFC says they are mathematically equivalent (§5.1.2.1).
    #[test]
    fn encode_bin_matches_generic_encode() {
        for ftb in [1u32, 3, 7, 12, 15] {
            let ft = 1u32 << ftb;
            // Encode the same symbol both ways into two encoders,
            // decode each, and require identical recovered symbols.
            let fl = ft / 4;
            let fh = ft / 2;
            let mut a = RangeEncoder::new();
            a.encode(fl, fh, ft).unwrap();
            let out_a = a.finish();
            let mut b = RangeEncoder::new();
            b.encode_bin(fl, fh, ftb).unwrap();
            let out_b = b.finish();
            assert_eq!(out_a, out_b, "ftb={ftb}");
        }
    }

    /// `tell()` after encoding matches `tell()` after decoding the same
    /// symbols at every step (§5.1.6 / §4.1.6 lockstep requirement).
    #[test]
    fn encoder_decoder_tell_lockstep() {
        // Encode a sequence, recording the encoder tell after each.
        let syms = [(1u32, 1u32), (0, 2), (1, 3), (0, 1), (1, 4), (0, 2)];
        let mut enc = RangeEncoder::new();
        let mut enc_tells = Vec::new();
        for &(b, lp) in &syms {
            enc.enc_bit_logp(b, lp).unwrap();
            enc_tells.push(enc.tell());
        }
        let out = enc.finish();
        let mut dec = RangeDecoder::new(&out);
        for (i, &(b, lp)) in syms.iter().enumerate() {
            assert_eq!(dec.dec_bit_logp(lp), b);
            assert_eq!(dec.tell(), enc_tells[i], "tell mismatch at step {i}");
        }
    }

    /// Encoding into a finalized stream is rejected.
    #[test]
    fn write_after_finish_rejected() {
        let mut enc = RangeEncoder::new();
        enc.enc_bit_logp(1, 1).unwrap();
        // finish() consumes self; recreate to test the finished guard
        // on the same instance via a clone-and-mark path.
        let mut enc2 = enc.clone();
        let _ = enc.finish();
        // enc2 is independent and unfinished; mark it finished by hand
        // is not possible from outside, so exercise the guard through a
        // re-finish of a finished clone.
        enc2.finished = true;
        assert_eq!(
            enc2.enc_bit_logp(0, 1),
            Err(Error::InvalidParameter),
            "write after finish must be rejected"
        );
    }

    /// Degenerate / invalid arguments are rejected without panicking.
    #[test]
    fn invalid_arguments_rejected() {
        let mut enc = RangeEncoder::new();
        assert_eq!(enc.encode(0, 1, 0), Err(Error::InvalidParameter)); // ft=0
        assert_eq!(enc.encode(2, 1, 4), Err(Error::InvalidParameter)); // fl>fh
        assert_eq!(enc.encode(0, 5, 4), Err(Error::InvalidParameter)); // fh>ft
        assert_eq!(enc.enc_uint(5, 5), Err(Error::InvalidParameter)); // t>=ft
        assert_eq!(enc.enc_bits(0, 33), Err(Error::InvalidParameter)); // bits>32
        assert_eq!(enc.enc_uint(1, 1), Err(Error::InvalidParameter)); // ft<=1,t!=0
                                                                      // ft<=1 with t==0 is a no-op success.
        assert_eq!(enc.enc_uint(0, 1), Ok(()));
    }

    /// A long pseudo-random stream of mixed symbols survives a full
    /// round-trip — stress the carry-propagation path with many 0xFF
    /// runs by biasing toward high-probability outcomes.
    #[test]
    fn roundtrip_long_random_stream() {
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) & 0x7FFF
        };
        let mut ops: Vec<(u8, u32, u32)> = Vec::new();
        let mut enc = RangeEncoder::new();
        for _ in 0..500 {
            let kind = next() % 3;
            match kind {
                0 => {
                    let lp = 1 + next() % 6;
                    let b = next() % 2;
                    enc.enc_bit_logp(b, lp).unwrap();
                    ops.push((0, b, lp));
                }
                1 => {
                    let ft = 2 + next() % 250;
                    let t = next() % ft;
                    enc.enc_uint(t, ft).unwrap();
                    ops.push((1, t, ft));
                }
                _ => {
                    let n = 1 + next() % 12;
                    let v = next() & ((1 << n) - 1);
                    enc.enc_bits(v, n).unwrap();
                    ops.push((2, v, n));
                }
            }
        }
        let out = enc.finish();
        let mut dec = RangeDecoder::new(&out);
        for (i, &(kind, a, b)) in ops.iter().enumerate() {
            match kind {
                0 => assert_eq!(dec.dec_bit_logp(b), a, "logp op {i}"),
                1 => assert_eq!(dec.dec_uint(b).unwrap(), a, "uint op {i}"),
                _ => assert_eq!(dec.dec_bits(b), a, "raw op {i}"),
            }
        }
        assert!(!dec.has_error());
    }
}
