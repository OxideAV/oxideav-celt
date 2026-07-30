//! Hybrid-mode CELT layer (`start = 17`): the encoder/decoder pair
//! codes bands 17..21 only (8-20 kHz; the SILK layer owns the
//! spectrum below 8 kHz in a Hybrid Opus stream), with the reference
//! `start != 0` behaviours — no post-filter fields on the wire, no
//! pitch prefilter, out-of-range energy state pinned to its reset
//! values — over the same Table-56 walk as pure CELT.

use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;

const HYBRID_START: usize = 17;

fn tone(frame: usize, channels: usize, frames: usize, freq_hz: f64, amp: f32) -> Vec<f32> {
    let n = frame * frames;
    let mut out = Vec::with_capacity(n * channels);
    for t in 0..n {
        let tf = t as f64 / 48_000.0;
        for c in 0..channels {
            let f = freq_hz * (1.0 + 0.05 * c as f64);
            out.push(amp * (2.0 * std::f64::consts::PI * f * tf).sin() as f32);
        }
    }
    out
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
}

/// Steady-state SNR of `ours` against `input` delayed by the
/// 120-sample lookahead, skipping the first `skip` frames.
fn snr_delayed(ours: &[f32], input: &[f32], channels: usize, skip: usize) -> f64 {
    let delay = 120 * channels;
    let n = (ours.len() - delay).min(input.len() - delay);
    let (mut ss, mut ee) = (0f64, 0f64);
    for i in skip..n {
        let s = input[i] as f64;
        let d = s - ours[i + delay] as f64;
        ss += s * s;
        ee += d * d;
    }
    10.0 * (ss / ee.max(1e-30)).log10()
}

/// A 10 kHz tone (inside coded band 18) round-trips through the
/// hybrid-layer codec at fidelity, at every LM and channel count.
#[test]
fn hybrid_high_tone_roundtrip() {
    for &(lm, channels) in &[
        (0u32, 1usize),
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 2),
        (2, 2),
        (3, 2),
    ] {
        let frame = 120usize << lm;
        let frames = 96usize >> lm;
        let bytes = (30 * channels) << lm;
        let input = tone(frame, channels, frames, 10_000.0, 0.3);
        let mut enc = CeltRefEncoder::new_with_start(lm, channels, HYBRID_START).expect("enc");
        let mut dec = CeltRefDecoder::new_with_start(lm, channels, HYBRID_START).expect("dec");
        let mut pcm = Vec::new();
        for chunk in input.chunks_exact(frame * channels) {
            let data = enc.encode_frame(chunk, bytes).expect("encode");
            assert_eq!(data.len(), bytes, "exact-size CBR frame");
            pcm.extend(dec.decode_frame(&data).expect("decode"));
        }
        let skip = 4 * frame * channels;
        let snr = snr_delayed(&pcm, &input, channels, skip);
        eprintln!("hybrid lm={lm} ch={channels} {bytes}B: 10 kHz tone SNR {snr:.1} dB");
        assert!(
            snr >= 10.0,
            "lm={lm} ch={channels}: hybrid tone SNR {snr:.1} dB"
        );
        // The prediction state below the start band stays at its
        // reference reset value.
        for row in dec.coarse.energy.iter() {
            assert!(row[..HYBRID_START].iter().all(|&e| e == 0.0));
        }
    }
}

/// Content wholly below the start band (a 400 Hz tone) decodes to
/// near-silence: the hybrid layer does not code the SILK territory.
#[test]
fn hybrid_low_content_stays_silent() {
    for &(lm, channels) in &[(2u32, 1usize), (3, 2)] {
        let frame = 120usize << lm;
        let frames = 24usize >> (lm - 2);
        let input = tone(frame, channels, frames, 400.0, 0.4);
        let mut enc = CeltRefEncoder::new_with_start(lm, channels, HYBRID_START).expect("enc");
        let mut dec = CeltRefDecoder::new_with_start(lm, channels, HYBRID_START).expect("dec");
        let mut pcm = Vec::new();
        for chunk in input.chunks_exact(frame * channels) {
            let data = enc.encode_frame(chunk, 60 * channels).expect("encode");
            pcm.extend(dec.decode_frame(&data).expect("decode"));
        }
        let skip = 2 * frame * channels;
        let out_rms = rms(&pcm[skip..]);
        let in_rms = rms(&input[skip..]);
        eprintln!(
            "hybrid lm={lm} ch={channels}: 400 Hz leak {:.4} (out {out_rms:.5} / in {in_rms:.5})",
            out_rms / in_rms
        );
        assert!(
            out_rms < 0.05 * in_rms,
            "lm={lm} ch={channels}: low-band content leaked ({out_rms:.5} vs {in_rms:.5})"
        );
    }
}

/// Every byte budget from the 2-byte floor to the 1275-byte wire cap
/// encodes to exactly the requested size and decodes finite.
#[test]
fn hybrid_budget_sweep_finite() {
    for &(lm, channels) in &[(0u32, 1usize), (3, 2)] {
        let frame = 120usize << lm;
        let input = tone(frame, channels, 3, 11_000.0, 0.5);
        for &bytes in &[2usize, 3, 5, 8, 13, 21, 35, 60, 100, 200, 500, 1275] {
            let mut enc = CeltRefEncoder::new_with_start(lm, channels, HYBRID_START).expect("enc");
            let mut dec = CeltRefDecoder::new_with_start(lm, channels, HYBRID_START).expect("dec");
            for chunk in input.chunks_exact(frame * channels) {
                let data = enc.encode_frame(chunk, bytes).expect("encode");
                assert_eq!(data.len(), bytes);
                let pcm = dec.decode_frame(&data).expect("decode");
                assert_eq!(pcm.len(), frame * channels);
                assert!(pcm.iter().all(|v| v.is_finite()));
            }
        }
    }
}

/// The VBR controller runs on the hybrid layer: digital silence
/// collapses to 2-byte frames, tonal content sizes itself, every
/// frame decodes finite.
#[test]
fn hybrid_vbr_sizes_and_silence() {
    for &(lm, channels) in &[(2u32, 1usize), (3, 2)] {
        let frame = 120usize << lm;
        let frames = 24usize >> (lm - 2);
        let mut input = tone(frame, channels, frames, 10_000.0, 0.35);
        // Silence in the middle third.
        let third = input.len() / 3;
        for v in &mut input[third..2 * third] {
            *v = 0.0;
        }
        let mut enc = CeltRefEncoder::new_with_start(lm, channels, HYBRID_START).expect("enc");
        let mut dec = CeltRefDecoder::new_with_start(lm, channels, HYBRID_START).expect("dec");
        let mut sizes = Vec::new();
        for chunk in input.chunks_exact(frame * channels) {
            let data = enc
                .encode_frame_vbr(chunk, 1275, 48_000, false)
                .expect("vbr");
            let pcm = dec.decode_frame(&data).expect("decode");
            assert!(pcm.iter().all(|v| v.is_finite()));
            sizes.push(data.len());
        }
        let silent = sizes.iter().filter(|&&s| s == 2).count();
        eprintln!("hybrid vbr lm={lm} ch={channels}: sizes {sizes:?}");
        assert!(silent >= 2, "lm={lm}: silence frames collapse to 2 bytes");
        assert!(
            sizes.iter().any(|&s| s > 8),
            "lm={lm}: tonal frames carry real budget"
        );
    }
}

/// A transient (noise burst) frame mid-stream keeps the hybrid walk
/// in encode/decode lockstep (short blocks + anti-collapse position).
#[test]
fn hybrid_transient_lockstep() {
    for &(lm, channels) in &[(2u32, 1usize), (3, 2)] {
        let frame = 120usize << lm;
        let frames = 8usize;
        let mut input = tone(frame, channels, frames, 9_000.0, 0.3);
        // A hard burst in frame 3 (LCG noise, full scale).
        let mut lcg: u32 = 0x1234_5678;
        let burst = 3 * frame * channels..(3 * frame * channels + frame * channels / 2);
        for v in &mut input[burst] {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((lcg >> 16) as i32 - 32_768) as f32 / 65_536.0;
        }
        let mut enc = CeltRefEncoder::new_with_start(lm, channels, HYBRID_START).expect("enc");
        let mut dec = CeltRefDecoder::new_with_start(lm, channels, HYBRID_START).expect("dec");
        for chunk in input.chunks_exact(frame * channels) {
            let data = enc.encode_frame(chunk, 80 * channels).expect("encode");
            let pcm = dec.decode_frame(&data).expect("decode");
            assert!(pcm.iter().all(|v| v.is_finite()));
        }
    }
}
