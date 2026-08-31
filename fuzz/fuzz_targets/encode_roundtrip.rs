#![no_main]

//! Coverage-guided harness for the reference-compatible encoder's
//! contract envelope: a structure-aware, contract-VALID configuration
//! (LM × channels × band window × reduced input rate × rate-control
//! mode) over arbitrary PCM.
//!
//! Every `expect` here fires only on a real defect: arbitrary
//! finite PCM at any legal payload size MUST encode (to exactly the
//! requested size in CBR, to 2..=1275 bytes in VBR) and every
//! emitted frame MUST decode cleanly through the crate's own
//! decoder at the matching configuration, with finite output.

use libfuzzer_sys::fuzz_target;
use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;

/// Reduced-rate PCM I/O points (the RFC 6716 §4.2.9 ladder).
const RATES: [u32; 5] = [48_000, 24_000, 16_000, 12_000, 8_000];
/// §3.1 CELT-mode bandwidths (`end` band).
const ENDS: [usize; 4] = [13, 17, 19, 21];

fuzz_target!(|data: &[u8]| {
    let (Some(&b0), Some(&b1), Some(&b2), Some(&b3)) =
        (data.first(), data.get(1), data.get(2), data.get(3))
    else {
        return;
    };
    let lm = u32::from(b0 & 3);
    let channels = 1 + usize::from((b0 >> 2) & 1);
    let hybrid = (b0 >> 3) & 1 == 1;
    let rate = RATES[usize::from(b0 >> 4) % RATES.len()];
    let start = if hybrid { 17 } else { 0 };
    let end = if hybrid {
        [19, 21][usize::from(b1) % 2]
    } else {
        ENDS[usize::from(b1) % ENDS.len()]
    };
    let vbr_mode = (b1 >> 4) & 3; // 0/1 = CBR, 2 = VBR, 3 = constrained VBR
    let v = u16::from_le_bytes([b2, b3]);

    let mut enc =
        CeltRefEncoder::new_with_config(lm, channels, start, end, rate).expect("legal config");
    let mut dec =
        CeltRefDecoder::new_with_config(lm, channels, start, end, rate).expect("legal config");
    let spf = enc.input_frame_size() * channels;
    let rest = &data[4..];
    let mut off = 0usize;
    for _ in 0..3 {
        // PCM from 16-bit LE pairs, zero-extended: always finite, in
        // the reference float-API input scale.
        let mut pcm = vec![0f32; spf];
        for s in pcm.iter_mut() {
            let lo = rest.get(off).copied().unwrap_or(0);
            let hi = rest.get(off + 1).copied().unwrap_or(0);
            *s = f32::from(i16::from_le_bytes([lo, hi])) / 32768.0;
            off += 2;
        }
        let bytes = if vbr_mode < 2 {
            let frame_bytes = 2 + usize::from(v) % 1274;
            let out = enc
                .encode_frame(&pcm, frame_bytes)
                .expect("contract-valid CBR encode");
            assert_eq!(out.len(), frame_bytes, "CBR frames are exactly-sized");
            out
        } else {
            let bitrate = 500 + u32::from(v) * 8;
            let out = enc
                .encode_frame_vbr(&pcm, 1275, bitrate, vbr_mode == 3)
                .expect("contract-valid VBR encode");
            assert!(
                (2..=1275).contains(&out.len()),
                "VBR frames stay on the wire limits"
            );
            out
        };
        let out = dec.decode_frame(&bytes).expect("own stream must decode");
        assert_eq!(out.len(), dec.output_frame_size() * channels);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "own-stream decode stays finite"
        );
        if off >= rest.len() {
            break;
        }
    }
});
