//! CELT frame-header decoding (RFC 6716 §4.3).
//!
//! The CELT layer of an Opus frame opens with a small prefix of
//! scalar fields that the band-decode bit allocator needs *before*
//! the energy envelope, PVQ shape, and MDCT machinery run. This
//! module decodes exactly that prefix.
//!
//! Per Table 56 (RFC 6716 §4.3) the CELT bitstream begins with:
//!
//! 1. `silence`            — PDF `{32767, 1}/32768`.
//! 2. `post-filter` flag   — PDF `{1, 1}/2` (logp=1).
//! 3. If post-filter set:
//!    * `octave`           — uniform in `[0, 6)`.
//!    * `period`           — `4 + octave` raw bits.
//!    * `gain`             — 3 raw bits.
//!    * `tapset`           — PDF `{2, 1, 1}/4`.
//! 4. `transient`          — PDF `{7, 1}/8`.
//! 5. `intra`              — PDF `{7, 1}/8`.
//!
//! Several later fields (coarse energy, tf_change, allocation,
//! spread, …, anti-collapse, finalize) still need to be wired up
//! in subsequent rounds. Anti-collapse in particular is decoded
//! *after* the band shape vectors per Table 56 + §4.3.5, so it
//! cannot be folded into the prefix walk and is exposed here as a
//! standalone helper to be called once the band-decode loop lands.
//!
//! No external library source was consulted; every PDF, every raw
//! bit count, and every parameter expression is transcribed from
//! RFC 6716 (`docs/audio/opus/rfc6716-opus.txt`).

use crate::range_decoder::RangeDecoder;

/// Post-filter parameters carried in the CELT frame header
/// (RFC 6716 §4.3.7.1).
///
/// The post-filter is applied at the very END of the decode pipeline
/// (after the inverse MDCT and overlap-add), but its parameters are
/// encoded near the BEGINNING of the frame, immediately after the
/// silence flag, so that the bit allocator can account for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostFilter {
    /// Octave index in `0..=5` (PDF `uniform(6)`).
    pub octave: u8,
    /// Reconstructed pitch period, `T = (16<<octave) + fine_pitch - 1`,
    /// bounded between 15 and 1022 inclusive per §4.3.7.1.
    pub period: u16,
    /// Raw-bit gain index in `0..=7`. The post-filter gain is
    /// `G = 3 * (gain + 1) / 32` per §4.3.7.1; we keep the raw index
    /// here so that downstream code can do its own fixed-point
    /// arithmetic.
    pub gain: u8,
    /// Tapset selector in `0..=2` (PDF `{2, 1, 1}/4`). Picks one of
    /// three sets of post-filter coefficients listed in §4.3.7.1.
    pub tapset: u8,
}

/// CELT frame-header scalar fields (RFC 6716 §4.3, prefix portion).
///
/// `anti_collapse_on` is `None` until [`decode_anti_collapse_flag`]
/// fills it in after the band-decode loop has run. It is also `None`
/// when the frame is non-transient (per §4.3.5 the anti-collapse bit
/// is only emitted on transient frames).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CeltFrameHeader {
    /// "All output samples are zero for this frame" flag.
    pub silence: bool,
    /// Post-filter parameters, `Some(_)` when the post-filter is
    /// enabled, `None` otherwise.
    pub post_filter: Option<PostFilter>,
    /// `true` if the frame uses multiple short MDCTs, `false` for a
    /// single long MDCT (RFC 6716 §4.3.1).
    pub transient: bool,
    /// `true` if coarse energy is coded without reference to prior
    /// frames (RFC 6716 §4.3.2.1).
    pub intra: bool,
    /// Anti-collapse bit, only set on transient frames and only after
    /// the band-decode loop has run. `None` until then.
    pub anti_collapse_on: Option<bool>,
}

/// ICDF table for the silence flag.
///
/// PDF `{32767, 1}/32768`; cumsum `[32767, 32768]`; icdf is
/// `[32768-32767, 32768-32768] = [1, 0]`. ftb=15.
const SILENCE_ICDF: &[u8] = &[1, 0];

/// ICDF table for the tapset selector.
///
/// PDF `{2, 1, 1}/4`; cumsum `[2, 3, 4]`; icdf is
/// `[4-2, 4-3, 4-4] = [2, 1, 0]`. ftb=2.
const TAPSET_ICDF: &[u8] = &[2, 1, 0];

impl CeltFrameHeader {
    /// Decode the always-present prefix of the CELT frame header
    /// (silence + post-filter + transient + intra).
    ///
    /// Returns a `CeltFrameHeader` with `anti_collapse_on = None`;
    /// callers fill that in via [`decode_anti_collapse_flag`] once the
    /// band-decode loop has placed the range decoder at the right
    /// position in the bitstream.
    pub fn decode_prefix(dec: &mut RangeDecoder<'_>) -> Self {
        // §4.3 Table 56: silence flag, PDF {32767, 1}/32768.
        // dec_icdf returns the symbol index; symbol 1 = silence set.
        let silence = dec.dec_icdf(SILENCE_ICDF, 15) == 1;

        // §4.3 Table 56: post-filter flag, PDF {1, 1}/2 = logp=1.
        // dec_bit_logp returns the bit value directly.
        let post_filter_on = dec.dec_bit_logp(1) == 1;

        // §4.3.7.1: if the post-filter is enabled, decode its four
        // parameters in order. Otherwise leave the slot empty.
        let post_filter = if post_filter_on {
            // Octave: uniform(6) → ec_dec_uint(6).
            // dec_uint returns Ok(0) on the ft=1 case; for ft=6 the
            // value is bounded to [0, 6), so the cast to u8 is safe.
            let octave = dec.dec_uint(6).unwrap_or(0) as u8;

            // Period: 4 + octave raw bits, expanded to the actual
            // pitch period via T = (16 << octave) + fine_pitch - 1.
            // §4.3.7.1 states "bounded between 15 and 1022,
            // inclusively"; the formula's worst case is
            // octave=5, fine_pitch=2^9-1=511 →
            //   (16 << 5) + 511 - 1 = 512 + 510 = 1022. ✓
            let raw_period_bits = 4 + octave as u32;
            let fine_pitch = dec.dec_bits(raw_period_bits);
            let period = ((16u32 << octave) + fine_pitch).saturating_sub(1);
            // Clamp into u16; the spec's upper bound 1022 fits in 10
            // bits, but downstream code may want u16's wider range
            // for arithmetic headroom.
            let period = period.min(u16::MAX as u32) as u16;

            // Gain: 3 raw bits, stored as the raw index. The actual
            // gain G = 3*(gain+1)/32 is computed downstream.
            let gain = dec.dec_bits(3) as u8;

            // Tapset: PDF {2, 1, 1}/4 → icdf [2, 1, 0], ftb=2.
            let tapset = dec.dec_icdf(TAPSET_ICDF, 2) as u8;

            Some(PostFilter {
                octave,
                period,
                gain,
                tapset,
            })
        } else {
            None
        };

        // §4.3.1: transient flag, PDF {7, 1}/8 = logp=3.
        let transient = dec.dec_bit_logp(3) == 1;

        // §4.3.2.1: intra flag, PDF {7, 1}/8 = logp=3.
        let intra = dec.dec_bit_logp(3) == 1;

        Self {
            silence,
            post_filter,
            transient,
            intra,
            anti_collapse_on: None,
        }
    }

    /// Reconstruct the post-filter gain `G = 3*(gain+1)/32` as a Q15
    /// fixed-point value (RFC 6716 §4.3.7.1). Returns 0 when the
    /// post-filter is disabled.
    ///
    /// In Q15: `G_q15 = 3*(gain+1)*1024` since `2^15 / 32 == 1024`.
    /// Maximum `gain=7` ⇒ `G_q15 = 3*8*1024 = 24576` (i.e. 0.75 in
    /// linear scale).
    pub fn post_filter_gain_q15(&self) -> u32 {
        match self.post_filter {
            Some(pf) => 3 * (pf.gain as u32 + 1) * 1024,
            None => 0,
        }
    }
}

/// Decode the anti-collapse bit (RFC 6716 §4.3.5), or skip it.
///
/// Per §4.3.5: "When the frame has the transient bit set, an
/// anti-collapse bit is decoded." For non-transient frames the bit
/// is not emitted; this function returns `false` in that case
/// without touching the decoder state.
///
/// Important: this MUST be called at the right point in the
/// bitstream — after the band shape vectors have been decoded, per
/// the order in Table 56. Calling it earlier or later will desync
/// the range decoder.
pub fn decode_anti_collapse_flag(dec: &mut RangeDecoder<'_>, transient: bool) -> bool {
    if !transient {
        return false;
    }
    // PDF {1, 1}/2 → logp=1.
    dec.dec_bit_logp(1) == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh header decoded from an all-zero buffer must satisfy:
    /// the silence flag is the high-probability outcome (symbol 0,
    /// i.e. `silence = false`); the post-filter is similarly off in
    /// the all-zero common case (a few bits of bias don't matter to
    /// the structural assertion); transient and intra are both the
    /// high-probability "0" outcomes. Most importantly the decoder
    /// must not latch an error walking through the prefix.
    #[test]
    fn decode_prefix_all_zero_no_error() {
        // The all-zero buffer keeps `val` large after each renorm, so
        // the icdf walks pick symbol 0 every time. That is the
        // common-case "no flags set" frame header.
        let mut dec = RangeDecoder::new(&[0u8; 32]);
        let hdr = CeltFrameHeader::decode_prefix(&mut dec);
        assert!(!dec.has_error(), "decode_prefix tripped error flag");
        // With the bias toward symbol 0 across every PDF, all flags
        // should be the high-probability "off" choice.
        assert!(!hdr.silence, "silence biased on for all-zero stream");
        assert!(
            hdr.post_filter.is_none(),
            "post-filter biased on for all-zero stream"
        );
        assert!(!hdr.transient, "transient biased on for all-zero stream");
        assert!(!hdr.intra, "intra biased on for all-zero stream");
        assert_eq!(hdr.anti_collapse_on, None);
    }

    /// `decode_prefix` MUST consume at least the no-flags lower bound
    /// of bits: silence (≤1 bit, near-deterministic), post-filter
    /// (1 bit), transient (~1/8 bit on the low-probability side), and
    /// intra (~1/8 bit). The exact count depends on the input bytes,
    /// but `tell()` must strictly advance compared to a freshly
    /// initialized decoder.
    #[test]
    fn decode_prefix_advances_tell() {
        let buf = [0xA3u8, 0x7F, 0x10, 0x5C, 0xE8, 0x91, 0x42, 0xB7];
        let mut dec = RangeDecoder::new(&buf);
        let before = dec.tell();
        let _hdr = CeltFrameHeader::decode_prefix(&mut dec);
        let after = dec.tell();
        assert!(
            after > before,
            "tell() did not advance through frame header: {before} -> {after}"
        );
    }

    /// Drive a hand-crafted stream that biases every flag the wrong
    /// way: all-ones bytes push `val` toward zero, which (in the icdf
    /// walk) lands on the *last* symbol of every table — so silence
    /// is set, transient is set, intra is set. Post-filter being a
    /// 1/2 PDF will also fire. The whole prefix walks without error.
    #[test]
    fn decode_prefix_all_ones_walks_without_error() {
        let mut dec = RangeDecoder::new(&[0xFFu8; 64]);
        let hdr = CeltFrameHeader::decode_prefix(&mut dec);
        assert!(!dec.has_error());
        // With val pushed all the way to the "1" side, the binary
        // PDFs all decode to 1.
        assert!(hdr.silence, "all-ones stream should bias silence on");
        assert!(
            hdr.post_filter.is_some(),
            "all-ones stream should bias post-filter on"
        );
        assert!(hdr.transient, "all-ones stream should bias transient on");
        assert!(hdr.intra, "all-ones stream should bias intra on");

        let pf = hdr.post_filter.unwrap();
        // octave must lie in [0, 6) per §4.3.7.1 / uniform(6).
        assert!(pf.octave < 6, "octave={} out of range [0,6)", pf.octave);
        // period must satisfy the §4.3.7.1 bound 15..=1022.
        assert!(
            (15..=1022).contains(&pf.period),
            "period={} outside [15, 1022]",
            pf.period
        );
        // gain index ∈ [0, 8).
        assert!(pf.gain < 8, "gain={} >= 8", pf.gain);
        // tapset ∈ [0, 3).
        assert!(pf.tapset < 3, "tapset={} >= 3", pf.tapset);
    }

    /// The post-filter gain helper must mirror the §4.3.7.1 formula
    /// `G = 3*(gain+1)/32`, expressed in Q15.
    #[test]
    fn post_filter_gain_q15_matches_spec_formula() {
        // Header without post-filter → gain 0.
        let hdr_off = CeltFrameHeader {
            silence: false,
            post_filter: None,
            transient: false,
            intra: false,
            anti_collapse_on: None,
        };
        assert_eq!(hdr_off.post_filter_gain_q15(), 0);

        // Sweep every gain index and confirm the Q15 reconstruction.
        for g in 0u8..8 {
            let hdr = CeltFrameHeader {
                silence: false,
                post_filter: Some(PostFilter {
                    octave: 0,
                    period: 15,
                    gain: g,
                    tapset: 0,
                }),
                transient: false,
                intra: false,
                anti_collapse_on: None,
            };
            // 2^15 / 32 == 1024; so G_q15 = 3*(g+1)*1024.
            let expected = 3 * (g as u32 + 1) * 1024;
            assert_eq!(
                hdr.post_filter_gain_q15(),
                expected,
                "gain={g}: q15 mismatch"
            );
        }
        // Spot check: gain=7 ⇒ G = 24/32 = 0.75 ⇒ Q15 = 24576.
        let hdr_max = CeltFrameHeader {
            silence: false,
            post_filter: Some(PostFilter {
                octave: 0,
                period: 15,
                gain: 7,
                tapset: 0,
            }),
            transient: false,
            intra: false,
            anti_collapse_on: None,
        };
        assert_eq!(hdr_max.post_filter_gain_q15(), 24_576);
    }

    /// The pitch-period reconstruction formula
    /// `T = (16<<octave) + fine_pitch - 1` must produce a value in
    /// `[15, 1022]` for every legal `(octave, fine_pitch)` pair
    /// (RFC 6716 §4.3.7.1). Walk the boundaries by hand.
    #[test]
    fn period_formula_boundaries() {
        // Recompute the formula stand-alone to confirm the
        // documented bounds; this is a spec-fidelity assertion rather
        // than a decoder test.
        // Lowest: octave=0, fine_pitch=0 ⇒ (16<<0) + 0 - 1 = 15.
        // (Substituting the literal 16 for the no-op `<< 0` keeps
        // clippy::identity_op happy without losing the trace.)
        let lo = 16u32 - 1;
        assert_eq!(lo, 15);
        // Highest: octave=5, fine_pitch=(1<<(4+5))-1=511 ⇒
        // (16<<5) + 511 - 1 = 512 + 510 = 1022.
        let hi = (16u32 << 5) + ((1u32 << 9) - 1) - 1;
        assert_eq!(hi, 1022);
        // Also sanity-check a mid-range pair: octave=3, fine_pitch=0 ⇒
        // (16<<3) - 1 = 128 - 1 = 127, which is inside [15, 1022].
        let mid = (16u32 << 3) - 1;
        assert_eq!(mid, 127);
    }

    /// `decode_anti_collapse_flag` MUST be a no-op (no decoder
    /// consumption, return `false`) when the transient flag is unset.
    #[test]
    fn anti_collapse_skipped_when_not_transient() {
        let mut dec = RangeDecoder::new(&[0xFFu8; 8]);
        let before = dec.tell();
        let v = decode_anti_collapse_flag(&mut dec, false);
        let after = dec.tell();
        assert!(!v);
        assert_eq!(
            before,
            after,
            "decode_anti_collapse_flag(transient=false) consumed {} bits",
            after - before
        );
        assert!(!dec.has_error());
    }

    /// On a transient frame, `decode_anti_collapse_flag` reads a
    /// single {1, 1}/2 bit. The exact decision depends on the input,
    /// but `tell()` must advance.
    #[test]
    fn anti_collapse_consumes_a_bit_when_transient() {
        let mut dec = RangeDecoder::new(&[0xA5u8, 0x33, 0x77, 0xCC]);
        let before = dec.tell();
        let _v = decode_anti_collapse_flag(&mut dec, true);
        let after = dec.tell();
        assert!(
            after > before,
            "decode_anti_collapse_flag(transient=true) did not advance tell()"
        );
        assert!(!dec.has_error());
    }

    /// End-to-end smoke test: walk a non-trivial buffer through the
    /// full prefix and then (assuming transient was set) through the
    /// anti-collapse helper, and confirm no panics or sticky errors.
    /// This stitches together the round-1+2 primitives with the
    /// round-3 frame-header walker.
    #[test]
    fn smoke_walk_through_prefix_and_anti_collapse() {
        let buf: Vec<u8> = (0u8..64).cycle().take(96).collect();
        let mut dec = RangeDecoder::new(&buf);
        let hdr = CeltFrameHeader::decode_prefix(&mut dec);
        let ac = decode_anti_collapse_flag(&mut dec, hdr.transient);
        // hdr.anti_collapse_on was None at decode_prefix exit; the
        // caller is responsible for stamping the bit back into the
        // header once the band-decode position is reached.
        let final_hdr = CeltFrameHeader {
            anti_collapse_on: if hdr.transient { Some(ac) } else { None },
            ..hdr
        };
        assert!(!dec.has_error());
        if hdr.transient {
            assert!(final_hdr.anti_collapse_on.is_some());
        } else {
            assert!(final_hdr.anti_collapse_on.is_none());
        }
    }
}
