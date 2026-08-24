//! Packet-loss concealment (`decode_lost`): the pitch-locked LPC
//! extrapolation for short loss runs, the comfort-noise fallback for
//! long runs and Hybrid-layer streams, and recovery back into live
//! decode.

use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;

fn tone_stream(lm: u32, channels: usize, frames: usize, bytes: usize, f0: f64) -> Vec<Vec<u8>> {
    let mut enc = CeltRefEncoder::new(lm, channels).expect("encoder");
    let frame = enc.frame_size();
    let pcm: Vec<f32> = (0..frames * frame)
        .flat_map(|t| {
            let tf = t as f64 / 48_000.0;
            (0..channels).map(move |c| {
                (0.4 * (2.0 * std::f64::consts::PI * f0 * (1.0 + 0.02 * c as f64) * tf).sin())
                    as f32
            })
        })
        .collect();
    (0..frames)
        .map(|f| {
            enc.encode_frame(
                &pcm[f * frame * channels..(f + 1) * frame * channels],
                bytes,
            )
            .expect("encode")
        })
        .collect()
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
}

/// Goertzel power of `freq` Hz at 48 kHz.
fn goertzel(x: &[f32], freq: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / 48_000.0;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0f64, 0f64);
    for &v in x {
        let s0 = v as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

/// A single lost frame on a steady tone conceals with a
/// tone-continuing extrapolation: the concealed frame keeps most of
/// the signal level and stays tone-dominated, and live decode
/// resumes cleanly.
#[test]
fn single_loss_extrapolates_the_tone() {
    for &(lm, channels) in &[(2u32, 1usize), (3, 1), (3, 2)] {
        let frames = tone_stream(lm, channels, 16, 120 * channels, 440.0);
        let mut dec = CeltRefDecoder::new(lm, channels).expect("decoder");
        let frame = dec.frame_size();
        let mut steady = Vec::new();
        for f in &frames[..10] {
            steady = dec.decode_frame(f).expect("decode");
        }
        let concealed = dec.decode_lost().expect("conceal");
        assert_eq!(concealed.len(), channels * frame);
        assert!(concealed.iter().all(|v| v.is_finite()));
        let level = rms(&concealed) / rms(&steady).max(1e-12);
        assert!(
            level > 0.4 && level < 2.0,
            "concealed level ratio {level:.3} at lm={lm} ch={channels}"
        );
        // Mono channel 0: the tone must still dominate.
        let ch0: Vec<f32> = concealed.iter().step_by(channels).copied().collect();
        let p_tone = goertzel(&ch0, 440.0);
        let p_off = goertzel(&ch0, 1_237.0);
        assert!(
            p_tone > 20.0 * p_off,
            "conealed frame lost the tone at lm={lm} ch={channels}"
        );
        // Live decode resumes and stays finite/leveled.
        for f in &frames[10..] {
            let out = dec.decode_frame(f).expect("resume");
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}

/// A long loss run fades: the pitch arm decays across frames 2-5,
/// then the comfort-noise arm takes over and keeps decaying toward
/// the background floor.
#[test]
fn long_loss_run_decays_to_noise() {
    let frames = tone_stream(3, 1, 12, 120, 440.0);
    let mut dec = CeltRefDecoder::new(3, 1).expect("decoder");
    let mut last = Vec::new();
    for f in &frames {
        last = dec.decode_frame(f).expect("decode");
    }
    let steady_rms = rms(&last);
    let mut levels = Vec::new();
    for _ in 0..12 {
        let concealed = dec.decode_lost().expect("conceal");
        assert!(concealed.iter().all(|v| v.is_finite()));
        levels.push(rms(&concealed) / steady_rms.max(1e-12));
    }
    // The tail of the run must sit well below the live level, and
    // far below the first concealed frame.
    assert!(
        levels[11] < 0.25 && levels[11] < 0.5 * levels[0].max(1e-12),
        "loss run does not decay: {levels:?}"
    );
}

/// Hybrid-layer streams (start = 17) conceal through the
/// comfort-noise arm from the first loss (no pitch extrapolation
/// below the coded band), and stay finite.
#[test]
fn hybrid_loss_uses_noise_arm() {
    let lm = 2u32;
    let mut enc = CeltRefEncoder::new_with_start(lm, 1, 17).expect("encoder");
    let frame = enc.frame_size();
    let pcm: Vec<f32> = (0..10 * frame)
        .map(|t| (0.3 * (2.0 * std::f64::consts::PI * 10_000.0 * t as f64 / 48_000.0).sin()) as f32)
        .collect();
    let mut dec = CeltRefDecoder::new_with_start(lm, 1, 17).expect("decoder");
    for f in 0..10 {
        let bytes = enc
            .encode_frame(&pcm[f * frame..(f + 1) * frame], 80)
            .expect("encode");
        dec.decode_frame(&bytes).expect("decode");
    }
    for _ in 0..4 {
        let concealed = dec.decode_lost().expect("conceal");
        assert_eq!(concealed.len(), frame);
        assert!(concealed.iter().all(|v| v.is_finite()));
    }
}

/// Downsampled output rates conceal too: the concealed frame spans
/// `frame / d` samples and live decode resumes.
#[test]
fn loss_at_downsampled_rates() {
    let frames = tone_stream(3, 1, 10, 120, 440.0);
    for &rate in &[24_000u32, 16_000, 12_000, 8_000] {
        let mut dec = CeltRefDecoder::new_downsampled(3, 1, rate).expect("decoder");
        for f in &frames[..6] {
            dec.decode_frame(f).expect("decode");
        }
        let concealed = dec.decode_lost().expect("conceal");
        assert_eq!(concealed.len(), dec.output_frame_size());
        assert!(concealed.iter().all(|v| v.is_finite()));
        for f in &frames[6..] {
            let out = dec.decode_frame(f).expect("resume");
            assert_eq!(out.len(), dec.output_frame_size());
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}

/// Concealment is deterministic: two decoders fed the same stream
/// and loss pattern emit identical PCM.
#[test]
fn concealment_is_deterministic() {
    let frames = tone_stream(2, 2, 12, 180, 330.0);
    let run = || -> Vec<f32> {
        let mut dec = CeltRefDecoder::new(2, 2).expect("decoder");
        let mut out = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            if i % 5 == 3 {
                out.extend(dec.decode_lost().expect("conceal"));
            }
            out.extend(dec.decode_frame(f).expect("decode"));
        }
        out
    };
    let a = run();
    let b = run();
    assert_eq!(a, b);
}
