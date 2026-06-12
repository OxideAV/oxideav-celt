//! Laplace-distributed symbol decoding (RFC 6716 §4.3.2.1).
//!
//! ## Where this fits
//!
//! The §4.3.2.1 coarse-energy step codes each band's prediction error
//! as a signed integer drawn from a Laplace distribution whose
//! `(prob, decay)` parameters come from the
//! [`crate::e_prob_model::E_PROB_MODEL`] table. RFC 6716 §4.3.2.1
//! states: "The decoding of the Laplace-distributed values is
//! implemented in ec_laplace_decode() (laplace.c)" — i.e. the RFC
//! delegates the per-symbol recurrence to its own Appendix A
//! reference listing, which is part of the staged spec
//! (`docs/audio/opus/rfc6716-opus.txt`, extracted per §A.1 with the
//! SHA-1 stated there). This module transcribes that recurrence from
//! RFC 6716 Appendix A `laplace.c`.
//!
//! ## The model
//!
//! The distribution is laid out over a 15-bit total frequency
//! (`ft = 32768`):
//!
//! * `fs` — the probability of the value 0, in units of 1/32768.
//! * `decay` — the geometric decay rate of the tail, in units of
//!   1/16384: the probability of `±(k+1)` is roughly `decay/16384`
//!   times the probability of `±k`.
//! * A guaranteed floor of [`LAPLACE_MINP`] per symbol keeps at least
//!   [`LAPLACE_NMIN`] deltas representable in each direction even
//!   when the geometric part has decayed to nothing; values beyond
//!   the decaying part fall into a uniform `LAPLACE_MINP`-wide tail.
//!
//! Each magnitude `k >= 1` occupies a probability span split evenly
//! between `-k` (lower half) and `+k` (upper half).
//!
//! ## Clean-room provenance
//!
//! The recurrence is transcribed from RFC 6716 Appendix A `laplace.c`
//! (the reference listing embedded in the RFC's own text and
//! extracted per §A.1); the surrounding narrative is RFC 6716
//! §4.3.2.1. No source outside the staged RFC was consulted.

use crate::range_decoder::RangeDecoder;

/// Base-2 log of the minimum probability floor of an energy delta
/// (out of 32768). RFC 6716 Appendix A `laplace.c` sets this to 0,
/// i.e. the floor is a single 1/32768 slot per symbol.
pub const LAPLACE_LOG_MINP: u32 = 0;

/// The minimum probability of an energy delta, `1 << LAPLACE_LOG_MINP`
/// out of 32768 (RFC 6716 Appendix A `laplace.c`).
pub const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;

/// The minimum number of guaranteed representable energy deltas in
/// one direction (RFC 6716 Appendix A `laplace.c`).
pub const LAPLACE_NMIN: u32 = 16;

/// Total frequency of the Laplace model: the symbol lives in a 15-bit
/// probability space, `ft = 32768`.
const FT_TOTAL: u32 = 1 << 15;

/// Frequency span of the magnitude-1 pair given the zero-probability
/// `fs0` and the `decay` rate (RFC 6716 Appendix A `laplace.c`,
/// `ec_laplace_get_freq1`).
///
/// `ft` is the probability mass left over after the zero symbol and
/// the `2 * LAPLACE_NMIN` reserved floor slots; the magnitude-1 span
/// takes a `(16384 - decay) / 32768` share of it
/// (`ft * (16384 - decay) >> 15`).
#[inline]
fn laplace_get_freq1(fs0: u32, decay: u32) -> u32 {
    let ft = FT_TOTAL - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    (ft * (16384 - decay)) >> 15
}

/// Decode one signed value drawn from the §4.3.2.1 Laplace
/// distribution (RFC 6716 Appendix A `laplace.c`,
/// `ec_laplace_decode`).
///
/// * `fs0` — probability of the value 0, multiplied by 32768. The
///   coarse-energy caller derives it from the
///   [`crate::e_prob_model::E_PROB_MODEL`] `prob` byte as
///   `prob << 7` (Q8 → Q15).
/// * `decay` — tail decay rate, multiplied by 16384. Derived from the
///   table's `decay` byte as `decay << 6` (Q8 → Q14). Must be below
///   16384.
///
/// The walk mirrors the encoder's interval layout exactly:
///
/// 1. Decode the 15-bit cumulative-frequency proxy `fm`
///    (`ec_decode_bin(15)`, §4.1.3.1).
/// 2. If `fm` lands inside the zero span (`fm < fs0`), the value is 0.
/// 3. Otherwise walk the geometrically decaying magnitude spans
///    (`fs * decay >> 15` per step, with the `LAPLACE_MINP` floor
///    folded in) until the span containing `fm` is found.
/// 4. If the decaying part has bottomed out at the floor, the
///    remaining distance is read directly from `fm` as a uniform
///    offset (`di`).
/// 5. The lower half of each magnitude span is the negative value,
///    the upper half the positive one.
/// 6. Commit the `[fl, fl + fs)` interval via `ec_dec_update`
///    (§4.1.2) with `ft = 32768`.
///
/// The function never reads more than one range-coded symbol and is
/// total for any 15-bit `fm`, including corrupt streams.
pub fn ec_laplace_decode(dec: &mut RangeDecoder<'_>, fs0: u32, decay: u32) -> i32 {
    debug_assert!(fs0 > 0 && fs0 < FT_TOTAL, "fs0 out of (0, 32768): {fs0}");
    debug_assert!(decay < 16384, "decay out of [0, 16384): {decay}");
    let mut val: i32 = 0;
    let fm = dec.decode_bin(15);
    let mut fl: u32 = 0;
    let mut fs = fs0;
    if fm >= fs {
        val += 1;
        fl = fs;
        fs = laplace_get_freq1(fs0, decay) + LAPLACE_MINP;
        // Search the decaying part of the PDF.
        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            fs = ((fs - 2 * LAPLACE_MINP) * decay) >> 15;
            fs += LAPLACE_MINP;
            val += 1;
        }
        // Everything beyond that has probability LAPLACE_MINP.
        if fs <= LAPLACE_MINP {
            let di = ((fm - fl) >> (LAPLACE_LOG_MINP + 1)) as i32;
            val += di;
            fl += 2 * (di as u32) * LAPLACE_MINP;
        }
        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }
    debug_assert!(fl < FT_TOTAL);
    debug_assert!(fs > 0);
    debug_assert!(fl <= fm);
    debug_assert!(fm < (fl + fs).min(FT_TOTAL));
    dec.dec_update(fl, (fl + fs).min(FT_TOTAL), FT_TOTAL);
    val
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::e_prob_model::E_PROB_MODEL;

    /// The constants pin the Appendix A `laplace.c` parameterisation:
    /// a 1/32768 floor slot and 16 guaranteed deltas per direction.
    #[test]
    fn constants_match_appendix() {
        assert_eq!(LAPLACE_LOG_MINP, 0);
        assert_eq!(LAPLACE_MINP, 1);
        assert_eq!(LAPLACE_NMIN, 16);
        assert_eq!(FT_TOTAL, 32_768);
    }

    /// `laplace_get_freq1` at decay = 0 hands the whole leftover mass
    /// to the magnitude-1 span scaled by 16384/32768 = 1/2; at the
    /// maximum legal decay (just below 16384) it approaches zero.
    #[test]
    fn freq1_boundary_behaviour() {
        // fs0 = 32 << 7 = 4096: leftover ft = 32768 - 32 - 4096 = 28640.
        let fs0 = 4096u32;
        // decay = 0 → freq1 = 28640 * 16384 >> 15 = 14320.
        assert_eq!(laplace_get_freq1(fs0, 0), 14_320);
        // decay = 16383 → freq1 = 28640 * 1 >> 15 = 0.
        assert_eq!(laplace_get_freq1(fs0, 16_383), 0);
        // A mid value, hand-computed: decay = 8192 →
        // 28640 * 8192 >> 15 = 7160.
        assert_eq!(laplace_get_freq1(fs0, 8_192), 7_160);
    }

    /// With a very high zero probability, the zero span covers nearly
    /// the whole interval, so a generic stream decodes to 0. fs0 =
    /// 32000 leaves only 768 slots for every non-zero magnitude.
    #[test]
    fn high_zero_probability_decodes_zero() {
        let buf = [0x5Au8, 0xC3, 0x99, 0x10, 0x42, 0xF0, 0x07, 0x6E];
        let mut dec = RangeDecoder::new(&buf);
        let v = ec_laplace_decode(&mut dec, 32_000, 64);
        assert_eq!(v, 0);
        assert!(!dec.has_error());
    }

    /// Decoding from identical buffers with identical parameters is
    /// deterministic.
    #[test]
    fn deterministic_across_decoders() {
        let buf: Vec<u8> = (0..32).map(|i| (i * 37 + 11) as u8).collect();
        for &(fs0, decay) in &[(9216u32, 8128u32), (128, 11_392), (24_576, 640)] {
            let mut a = RangeDecoder::new(&buf);
            let mut b = RangeDecoder::new(&buf);
            for _ in 0..8 {
                assert_eq!(
                    ec_laplace_decode(&mut a, fs0, decay),
                    ec_laplace_decode(&mut b, fs0, decay)
                );
            }
        }
    }

    /// Every `(prob, decay)` pair in the staged `E_PROB_MODEL` table
    /// must drive the decoder without tripping the sticky error flag
    /// or violating the value bound implied by the 15-bit space
    /// (|val| can never exceed 32768/2 distinct magnitudes), on both
    /// a zero-biased and a one-biased stream.
    #[test]
    fn all_prob_model_cells_decode_within_bounds() {
        let zero_biased = [0u8; 24];
        let one_biased = [0xFFu8; 24];
        for lm_row in E_PROB_MODEL.iter() {
            for intra_row in lm_row.iter() {
                for cell in intra_row.iter() {
                    let fs0 = (cell.prob as u32) << 7;
                    let decay = (cell.decay as u32) << 6;
                    for buf in [&zero_biased[..], &one_biased[..]] {
                        let mut dec = RangeDecoder::new(buf);
                        for _ in 0..4 {
                            let v = ec_laplace_decode(&mut dec, fs0, decay);
                            assert!(
                                v.unsigned_abs() < FT_TOTAL / 2,
                                "decoded |{v}| out of bound for prob={} decay={}",
                                cell.prob,
                                cell.decay
                            );
                        }
                        assert!(!dec.has_error());
                    }
                }
            }
        }
    }

    /// `tell()` advances by at most 16 bits per Laplace symbol (the
    /// symbol lives in a 15-bit space, plus renormalization slack)
    /// and never regresses.
    #[test]
    fn tell_progresses_sanely() {
        let buf: Vec<u8> = (0..48).map(|i| (i * 73 + 5) as u8).collect();
        let mut dec = RangeDecoder::new(&buf);
        let mut prev = dec.tell();
        for _ in 0..16 {
            let _ = ec_laplace_decode(&mut dec, 5376, 9088);
            let now = dec.tell();
            assert!(now >= prev, "tell went backwards: {prev} -> {now}");
            assert!(now - prev <= 16, "one symbol consumed {} bits", now - prev);
            prev = now;
        }
        assert!(!dec.has_error());
    }

    /// An exhausted decoder (empty buffer) keeps producing values
    /// without panicking — the §4.1.2.1 zero-fill rule applies to the
    /// Laplace path like any other symbol decode.
    #[test]
    fn empty_buffer_decodes_total_function() {
        let mut dec = RangeDecoder::new(&[]);
        for _ in 0..8 {
            let v = ec_laplace_decode(&mut dec, 2816, 11_392);
            assert!(v.unsigned_abs() < FT_TOTAL / 2);
        }
    }
}
