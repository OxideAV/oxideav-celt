//! Criterion benchmarks for the reference-exact CELT decoder hot
//! path (r454 depth-mode: the encoder measures at/above the listing
//! on every axis, so the round's tail is a decode throughput
//! baseline for future speed work).
//!
//! Streams are produced in-bench by the crate's own
//! reference-compatible encoder over a deterministic mixed signal
//! (tones + transient burst + noise tail) — no `docs/` fixtures or
//! external files are read — and every iteration decodes a whole
//! stream through ONE stateful decoder, mirroring real playback
//! (coarse-energy prediction, overlap, post-filter history all
//! live). Throughput is reported in decoded samples so the numbers
//! read directly as x-realtime at 48 kHz.
//!
//! Scenarios:
//!
//!   - **cbr_lm3_mono_160B / cbr_lm3_stereo_239B**: the 20 ms
//!     long/short-block mix at typical CBR rates (the general
//!     playback shape).
//!   - **cbr_lm0_mono_43B**: 2.5 ms frames — per-frame overhead
//!     dominated (400 frames/s of walk + synthesis).
//!   - **plc_lm3_mono**: one lost frame concealed per decoded frame
//!     (alternating decode/conceal), the §4.3 concealment walk's
//!     cost next to a clean decode.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;

/// Deterministic mixed test signal (tones, a hard transient burst, a
/// noise tail) at 48 kHz, interleaved.
fn test_signal(channels: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * channels);
    let mut lcg = 0x2545_F491u32;
    for t in 0..n {
        let tf = t as f32 / 48_000.0;
        for c in 0..channels {
            let f0 = if c == 0 { 440.0 } else { 523.0 };
            let mut v = 0.28 * (2.0 * std::f32::consts::PI * f0 * tf).sin()
                + 0.18 * (2.0 * std::f32::consts::PI * 3.1 * f0 * tf).sin();
            if (4_800..5_040).contains(&t) {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                v += 0.5 * ((lcg >> 16) as i16 as f32 / 32768.0);
            }
            if t >= 12_000 {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                v = 0.25 * ((lcg >> 16) as i16 as f32 / 32768.0);
            }
            out.push(v.clamp(-0.999, 0.999));
        }
    }
    out
}

/// Encode a whole stream at `frame_bytes`/frame with the crate's own
/// encoder (the streams the decoder spends its life on).
fn encode_stream(lm: u32, channels: usize, frame_bytes: usize, seconds_x10: usize) -> Vec<Vec<u8>> {
    let mut enc = CeltRefEncoder::new(lm, channels).expect("config");
    let frame = enc.frame_size();
    let n = 4_800 * seconds_x10;
    let pcm = test_signal(channels, n);
    let mut frames = Vec::new();
    for chunk in pcm.chunks_exact(frame * channels) {
        frames.push(enc.encode_frame(chunk, frame_bytes).expect("encode"));
    }
    frames
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_decode");
    for (name, lm, channels, frame_bytes) in [
        ("cbr_lm3_mono_160B", 3u32, 1usize, 160usize),
        ("cbr_lm3_stereo_239B", 3, 2, 239),
        ("cbr_lm0_mono_43B", 0, 1, 43),
    ] {
        let frames = encode_stream(lm, channels, frame_bytes, 4);
        let samples: u64 = frames.len() as u64 * (120u64 << lm);
        group.throughput(Throughput::Elements(samples));
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut dec = CeltRefDecoder::new(lm, channels).expect("config");
                let mut acc = 0f32;
                for f in &frames {
                    let pcm = dec.decode_frame(f).expect("decode");
                    // Fold the output so the decode cannot be
                    // optimised away, and catch a poisoned stream.
                    acc += pcm[0];
                }
                assert!(acc.is_finite());
                acc
            })
        });
    }
    group.finish();
}

fn bench_plc(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_plc");
    let (lm, channels) = (3u32, 1usize);
    let frames = encode_stream(lm, channels, 160, 4);
    let samples: u64 = frames.len() as u64 * 2 * (120u64 << lm);
    group.throughput(Throughput::Elements(samples));
    group.bench_function("plc_lm3_mono_alternating", |b| {
        b.iter(|| {
            let mut dec = CeltRefDecoder::new(lm, channels).expect("config");
            let mut acc = 0f32;
            for f in &frames {
                acc += dec.decode_frame(f).expect("decode")[0];
                acc += dec.decode_lost().expect("conceal")[0];
            }
            assert!(acc.is_finite());
            acc
        })
    });
    group.finish();
}

criterion_group!(benches, bench_decode, bench_plc);
criterion_main!(benches);
