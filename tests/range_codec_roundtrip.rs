//! Integration test: the §5.1 range *encoder* is the exact inverse of
//! the §4.1 range *decoder*.
//!
//! These tests drive the public `RangeEncoder` API (RFC 6716 §5.1) and
//! decode the result back with the public `RangeDecoder` (§4.1),
//! asserting bit-exact recovery of every symbol. Encoder/decoder
//! lockstep on the range size (`rng`) and the bit budget (`tell` /
//! `tell_frac`) is the §5.1 / §4.1.6 conformance hook ("the value of
//! rng in the encoder should exactly match the value of rng in the
//! decoder after decoding the same sequence of symbols").

use oxideav_celt::{RangeDecoder, RangeEncoder};

/// Encode an interleaved stream of every symbol type the CELT control
/// path uses, then decode it back symbol-for-symbol.
#[test]
fn encode_decode_full_symbol_mix() {
    let icdf = [3u8, 1, 0]; // PDF {1,2,1}/4 over ftb=2.
    let mut enc = RangeEncoder::new();
    enc.enc_bit_logp(1, 1).unwrap(); // silence-style flag
    enc.enc_uint(3, 7).unwrap(); // post-filter octave
    enc.enc_bits(0b1011, 4).unwrap(); // raw fine-pitch bits
    enc.enc_icdf(2, &icdf, 2).unwrap(); // tapset-style icdf
    enc.enc_bit_logp(0, 3).unwrap(); // transient flag
    enc.enc_uint(250_000, 1_000_000).unwrap(); // large-ft split path
    enc.enc_bits(0b101, 3).unwrap();
    enc.enc_icdf(0, &icdf, 2).unwrap();
    let frame = enc.finish();

    let mut dec = RangeDecoder::new(&frame);
    assert_eq!(dec.dec_bit_logp(1), 1);
    assert_eq!(dec.dec_uint(7).unwrap(), 3);
    assert_eq!(dec.dec_bits(4), 0b1011);
    assert_eq!(dec.dec_icdf(&icdf, 2), 2);
    assert_eq!(dec.dec_bit_logp(3), 0);
    assert_eq!(dec.dec_uint(1_000_000).unwrap(), 250_000);
    assert_eq!(dec.dec_bits(3), 0b101);
    assert_eq!(dec.dec_icdf(&icdf, 2), 0);
    assert!(!dec.has_error());
}

/// A fresh encoder and a fresh decoder agree on the initial bit budget
/// (§4.1.6.1: a newly initialized coder reports 1 bit).
#[test]
fn fresh_encoder_decoder_agree_on_initial_tell() {
    let enc = RangeEncoder::new();
    let dec = RangeDecoder::new(&[]);
    assert_eq!(enc.tell(), dec.tell());
    assert_eq!(enc.tell(), 1);
}

/// The encoder's `tell()` after writing a sequence equals the decoder's
/// `tell()` after reading it back, at every step — the §5.1.6 / §4.1.6
/// lockstep requirement that drives CELT's bit allocation.
#[test]
fn encoder_decoder_tell_lockstep_over_uint_stream() {
    let cases: &[(u32, u32)] = &[(0, 4), (2, 4), (5, 6), (40, 41), (1, 2), (255, 256)];
    let mut enc = RangeEncoder::new();
    let mut enc_tells = Vec::new();
    let mut enc_fracs = Vec::new();
    for &(t, ft) in cases {
        enc.enc_uint(t, ft).unwrap();
        enc_tells.push(enc.tell());
        enc_fracs.push(enc.tell_frac());
    }
    let frame = enc.finish();

    let mut dec = RangeDecoder::new(&frame);
    for (i, &(t, ft)) in cases.iter().enumerate() {
        assert_eq!(dec.dec_uint(ft).unwrap(), t, "value at step {i}");
        assert_eq!(dec.tell(), enc_tells[i], "tell at step {i}");
        assert_eq!(dec.tell_frac(), enc_fracs[i], "tell_frac at step {i}");
    }
    assert!(!dec.has_error());
}

/// A long deterministic pseudo-random stream survives a full
/// round-trip across all three branches (range symbols, large-ft
/// split, raw bits) — exercising the carry-propagation path heavily.
#[test]
fn long_stream_roundtrip() {
    let mut state: u32 = 0xC0FF_EE42;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 13) & 0xFFFF
    };
    let mut ops: Vec<(u8, u32, u32)> = Vec::new();
    let mut enc = RangeEncoder::new();
    for _ in 0..1000 {
        match next() % 3 {
            0 => {
                let lp = 1 + next() % 8;
                let b = next() % 2;
                enc.enc_bit_logp(b, lp).unwrap();
                ops.push((0, b, lp));
            }
            1 => {
                let ft = 2 + next() % 60_000;
                let t = next() % ft;
                enc.enc_uint(t, ft).unwrap();
                ops.push((1, t, ft));
            }
            _ => {
                let n = 1 + next() % 14;
                let v = next() & ((1u32 << n) - 1);
                enc.enc_bits(v, n).unwrap();
                ops.push((2, v, n));
            }
        }
    }
    let frame = enc.finish();
    let mut dec = RangeDecoder::new(&frame);
    for (i, &(kind, a, b)) in ops.iter().enumerate() {
        match kind {
            0 => assert_eq!(dec.dec_bit_logp(b), a, "bit_logp op {i}"),
            1 => assert_eq!(dec.dec_uint(b).unwrap(), a, "uint op {i}"),
            _ => assert_eq!(dec.dec_bits(b), a, "raw op {i}"),
        }
    }
    assert!(!dec.has_error());
}
