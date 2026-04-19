//! Public-API encode -> decode roundtrip.
//!
//! Exercises the `Encoder` and `Decoder` traits from `oxideav-codec` via
//! `make_decoder`/`make_encoder` instead of the module-level primitives in
//! `tests/roundtrip.rs`. This is the surface that external callers and the
//! aggregator registry talk to.

#![allow(clippy::while_let_loop)]

use oxideav_celt::decoder::CeltDecoder;
use oxideav_celt::encoder::{CeltEncoder, FRAME_SAMPLES, SAMPLE_RATE};
use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, SampleFormat, TimeBase};

fn build_params(channels: u16) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(oxideav_celt::CODEC_ID_STR));
    p.channels = Some(channels);
    p.sample_rate = Some(SAMPLE_RATE);
    p
}

fn pcm_frame_f32(samples: &[f32], channels: u16) -> Frame {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        format: SampleFormat::F32,
        channels,
        sample_rate: SAMPLE_RATE,
        samples: (samples.len() / channels as usize) as u32,
        pts: None,
        time_base: TimeBase::new(1, SAMPLE_RATE as i64),
        data: vec![bytes],
    })
}

fn decoded_f32(frame: &Frame) -> Vec<f32> {
    match frame {
        Frame::Audio(a) => {
            let bytes = &a.data[0];
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out
        }
        _ => panic!("expected audio frame"),
    }
}

#[test]
fn silence_roundtrip_through_public_api() {
    let params = build_params(1);
    let mut enc = CeltEncoder::new(&params).unwrap();
    let mut dec = CeltDecoder::new(&params).unwrap();

    let n_frames = 3usize;
    for _ in 0..n_frames {
        let pcm = vec![0.0f32; FRAME_SAMPLES];
        enc.send_frame(&pcm_frame_f32(&pcm, 1)).unwrap();
    }
    enc.flush().unwrap();

    let mut total = 0usize;
    loop {
        match enc.receive_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).unwrap();
                while let Ok(frame) = dec.receive_frame() {
                    let pcm = decoded_f32(&frame);
                    assert_eq!(pcm.len(), FRAME_SAMPLES);
                    assert!(
                        pcm.iter().all(|v| v.is_finite()),
                        "decoded silence must be finite"
                    );
                    let energy: f32 = pcm.iter().map(|v| v * v).sum();
                    assert!(
                        energy < 1e-6,
                        "silence should decode to near-zero (got energy {energy})"
                    );
                    total += 1;
                }
            }
            Err(_) => break,
        }
    }
    assert_eq!(
        total, n_frames,
        "expected one decoded frame per encoded frame"
    );
}

#[test]
fn sine_roundtrip_through_public_api() {
    let params = build_params(1);
    let mut enc = CeltEncoder::new(&params).unwrap();
    let mut dec = CeltDecoder::new(&params).unwrap();

    let n_frames = 4usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let freq = 1000.0f32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();

    for chunk in signal.chunks(FRAME_SAMPLES) {
        if chunk.len() == FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();

    let mut decoded: Vec<f32> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).unwrap();
                while let Ok(frame) = dec.receive_frame() {
                    decoded.extend(decoded_f32(&frame));
                }
            }
            Err(_) => break,
        }
    }
    assert_eq!(decoded.len(), n_samples);
    assert!(decoded.iter().all(|v| v.is_finite()));

    let goertzel = |samples: &[f32], f: f32| -> f32 {
        let w = 2.0 * std::f32::consts::PI * f / SAMPLE_RATE as f32;
        let cw = w.cos();
        let (mut s0, mut s1, mut s2) = (0f32, 0f32, 0f32);
        for &x in samples {
            s0 = 2.0 * cw * s1 - s2 + x;
            s2 = s1;
            s1 = s0;
        }
        let _ = s0;
        (s1 * s1 + s2 * s2 - 2.0 * cw * s1 * s2).sqrt()
    };
    let start = FRAME_SAMPLES * 2;
    let end = FRAME_SAMPLES * 3;
    let slice = &decoded[start..end];
    let mag_target = goertzel(slice, freq);
    let mag_off = goertzel(slice, 5000.0);
    let energy: f32 = slice.iter().map(|v| v * v).sum();
    assert!(energy > 1e-6, "decoded output is silent (energy {energy})");
    assert!(
        mag_target > 0.3 * mag_off,
        "target tone buried in decoder output (tgt {mag_target:.3}, off {mag_off:.3})"
    );
}

#[test]
fn stereo_roundtrip_through_public_api() {
    let params = build_params(2);
    let mut enc = CeltEncoder::new(&params).unwrap();
    let mut dec = CeltDecoder::new(&params).unwrap();

    let n_frames = 3usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let f_l = 1000.0f32;
    let f_r = 1500.0f32;
    let mut interleaved: Vec<f32> = Vec::with_capacity(n_samples * 2);
    for i in 0..n_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        interleaved.push((2.0 * std::f32::consts::PI * f_l * t).sin() * 0.3);
        interleaved.push((2.0 * std::f32::consts::PI * f_r * t).sin() * 0.3);
    }

    let frame_interleaved = FRAME_SAMPLES * 2;
    for chunk in interleaved.chunks(frame_interleaved) {
        if chunk.len() == frame_interleaved {
            enc.send_frame(&pcm_frame_f32(chunk, 2)).unwrap();
        }
    }
    enc.flush().unwrap();

    let mut decoded: Vec<f32> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).unwrap();
                while let Ok(frame) = dec.receive_frame() {
                    decoded.extend(decoded_f32(&frame));
                }
            }
            Err(_) => break,
        }
    }
    assert_eq!(decoded.len(), n_samples * 2);
    assert!(decoded.iter().all(|v| v.is_finite()));
}
