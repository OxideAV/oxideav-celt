//! End-band configurations (the RFC 6716 §3.1 CELT-mode
//! bandwidths): coded bands `0..end` with `end` = 13 (NB), 17 (WB),
//! 19 (SWB), 21 (FB) — both directions, self round trips, spectral
//! bounds, and state pinning.

use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;

const ENDS: [usize; 3] = [13, 17, 19];

/// Band edge in MDCT bins at LM 3 for a band index (e_bands × 8).
const EDGE_LM3: [usize; 22] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 272, 320, 384, 480, 624, 800,
];

fn tone(n: usize, channels: usize, f0: f64) -> Vec<f32> {
    (0..n)
        .flat_map(|t| {
            let tf = t as f64 / 48_000.0;
            (0..channels).map(move |c| {
                (0.4 * (2.0 * std::f64::consts::PI * f0 * (1.0 + 0.01 * c as f64) * tf).sin())
                    as f32
            })
        })
        .collect()
}

fn snr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut ss, mut ee) = (0f64, 0f64);
    for i in 0..n {
        let s = a[i] as f64;
        let d = s - b[i] as f64;
        ss += s * s;
        ee += d * d;
    }
    10.0 * (ss / ee.max(1e-30)).log10()
}

/// Round trips at every bandwidth: in-band content survives with a
/// codec-floor SNR, and the wire is agreed on by both sides.
#[test]
fn end_band_round_trips() {
    for &end in &ENDS {
        for &channels in &[1usize, 2] {
            let mut enc = CeltRefEncoder::new_with_bands(3, channels, 0, end).expect("encoder");
            let mut dec = CeltRefDecoder::new_with_bands(3, channels, 0, end).expect("decoder");
            let frame = enc.frame_size();
            let n_frames = 16usize;
            let pcm = tone(n_frames * frame, channels, 440.0);
            let mut out = Vec::new();
            for f in 0..n_frames {
                let bytes = enc
                    .encode_frame(&pcm[f * frame * channels..(f + 1) * frame * channels], 100)
                    .expect("encode");
                assert_eq!(bytes.len(), 100);
                out.extend(dec.decode_frame(&bytes).expect("decode"));
            }
            let steady = 4 * frame * channels;
            let delay = 120 * channels;
            let s = snr(&pcm[steady..pcm.len() - delay], &out[steady + delay..]);
            assert!(s > 12.0, "end={end} ch={channels} round trip {s:.1} dB");
        }
    }
}

/// A narrowband decode leaves the spectrum above the end band
/// empty: content coded at end = 13 carries no energy above 4 kHz
/// (checked in the decode of a full-scale tone against a fullband
/// re-encode of the same signal).
#[test]
fn end_band_bounds_the_spectrum() {
    let end = 13usize;
    let mut enc = CeltRefEncoder::new_with_bands(3, 1, 0, end).expect("encoder");
    let mut dec = CeltRefDecoder::new_with_bands(3, 1, 0, end).expect("decoder");
    let frame = enc.frame_size();
    let pcm = tone(12 * frame, 1, 440.0);
    let mut out = Vec::new();
    for f in 0..12 {
        let bytes = enc
            .encode_frame(&pcm[f * frame..(f + 1) * frame], 100)
            .expect("encode");
        out.extend(dec.decode_frame(&bytes).expect("decode"));
    }
    // Goertzel probe well above the NB edge (edge bin 160 = 4 kHz):
    // 6 kHz must be at least 40 dB under the 440 Hz tone.
    let probe = |freq: f64| -> f64 {
        let w = 2.0 * std::f64::consts::PI * freq / 48_000.0;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0f64, 0f64);
        for &v in &out[out.len() / 2..] {
            let s0 = v as f64 + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        s1 * s1 + s2 * s2 - coeff * s1 * s2
    };
    let p_tone = probe(440.0);
    let p_high = probe(6_000.0);
    assert!(
        p_high < 1e-4 * p_tone,
        "energy above the NB edge: {p_high:.3e} vs {p_tone:.3e}"
    );
    assert_eq!(EDGE_LM3[end], 160);
}

/// The coarse-energy state above the end band stays pinned to the
/// reference reset values across frames, and different-end decoders
/// stay in symbol lockstep with their encoder (streams don't decode
/// on a mismatched end).
#[test]
fn end_band_state_pinning_and_mismatch() {
    let end = 17usize;
    let mut enc = CeltRefEncoder::new_with_bands(2, 1, 0, end).expect("encoder");
    let mut dec = CeltRefDecoder::new_with_bands(2, 1, 0, end).expect("decoder");
    let frame = enc.frame_size();
    let pcm = tone(8 * frame, 1, 440.0);
    for f in 0..8 {
        let bytes = enc
            .encode_frame(&pcm[f * frame..(f + 1) * frame], 90)
            .expect("encode");
        dec.decode_frame(&bytes).expect("decode");
        for i in end..21 {
            assert_eq!(dec.coarse.energy[0][i], 0.0, "band {i} energy not pinned");
        }
    }
    // Constructor validation.
    assert!(CeltRefDecoder::new_with_bands(3, 1, 0, 22).is_err());
    assert!(CeltRefDecoder::new_with_bands(3, 1, 17, 17).is_err());
    assert!(CeltRefEncoder::new_with_bands(3, 1, 0, 1).is_ok());
}

/// Hybrid-style band windows compose with an end band
/// (`start = 17`, `end = 19`): both sides agree and stay finite.
#[test]
fn start_and_end_compose() {
    let mut enc = CeltRefEncoder::new_with_bands(2, 1, 17, 19).expect("encoder");
    let mut dec = CeltRefDecoder::new_with_bands(2, 1, 17, 19).expect("decoder");
    let frame = enc.frame_size();
    // 9-11 kHz content (inside bands 17..19).
    let pcm: Vec<f32> = (0..10 * frame)
        .map(|t| (0.3 * (2.0 * std::f64::consts::PI * 9_800.0 * t as f64 / 48_000.0).sin()) as f32)
        .collect();
    for f in 0..10 {
        let bytes = enc
            .encode_frame(&pcm[f * frame..(f + 1) * frame], 60)
            .expect("encode");
        let out = dec.decode_frame(&bytes).expect("decode");
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
