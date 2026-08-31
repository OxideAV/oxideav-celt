#![no_main]

//! Coverage-guided harness for the Appendix-A custom-mode
//! construction: `CeltCustomMode::new` over arbitrary
//! `(rate, frame_size)` pairs.
//!
//! An illegal geometry must be rejected with `Err(..)` — never a
//! panic, an overflow, or an unbounded allocation — and every
//! ACCEPTED mode must be fully usable: encoder and decoder
//! construction at a legal `lm` succeeds and a one-frame
//! encode→decode round trip holds the exact-size / clean-decode
//! contract.

use libfuzzer_sys::fuzz_target;
use oxideav_celt::custom_mode::CeltCustomMode;
use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let fs = u32::from_le_bytes([data[0], data[1], data[2], 0]);
    let frame_size = usize::from(u16::from_le_bytes([data[3], data[4]]));
    let cfg = data[5];
    let Ok(mode) = CeltCustomMode::new(fs, frame_size) else {
        return;
    };
    // The documented legal envelope: 8–96 kHz, 40–1024 samples.
    assert!(
        (8_000..=96_000).contains(&fs) && (40..=1024).contains(&frame_size),
        "accepted geometry outside the documented envelope: {fs} Hz / {frame_size}"
    );
    let lm = u32::from(cfg & 3).min(mode.max_lm);
    let channels = 1 + usize::from((cfg >> 2) & 1);
    let mut enc = CeltRefEncoder::new_custom(&mode, lm, channels).expect("mode-legal encoder");
    let mut dec = CeltRefDecoder::new_custom(&mode, lm, channels).expect("mode-legal decoder");
    let frame_bytes = 2 + usize::from(cfg >> 3) % 200;
    let spf = enc.input_frame_size() * channels;
    let rest = &data[6..];
    let mut pcm = vec![0f32; spf];
    for (i, s) in pcm.iter_mut().enumerate() {
        let lo = rest.get(2 * i).copied().unwrap_or(0);
        let hi = rest.get(2 * i + 1).copied().unwrap_or(0);
        *s = f32::from(i16::from_le_bytes([lo, hi])) / 32768.0;
    }
    let bytes = enc
        .encode_frame(&pcm, frame_bytes)
        .expect("custom-mode encode");
    assert_eq!(bytes.len(), frame_bytes, "CBR frames are exactly-sized");
    let out = dec
        .decode_frame(&bytes)
        .expect("custom-mode own-stream decode");
    assert_eq!(out.len(), dec.output_frame_size() * channels);
    assert!(out.iter().all(|x| x.is_finite()));
});
