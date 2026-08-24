//! RFC 6716 downsampled output rates: the standard 48 kHz mode
//! decoded to 24/16/12/8 kHz PCM (`celt_decoder_init` at a reduced
//! rate — spectrum bounded to the output Nyquist, de-emphasis
//! decimation), and the encoder-side upsampled-input counterpart.

use oxideav_celt::ref_decode::{resampling_factor, CeltRefDecoder};
use oxideav_celt::ref_encode::CeltRefEncoder;

const RATES: [u32; 4] = [24_000, 16_000, 12_000, 8_000];

/// A tonal + noise-burst test signal at 48 kHz, band-limited to
/// `f_max` Hz so every output rate keeps the content in band.
fn test_signal(frames: usize, frame: usize, channels: usize, f_max: f64) -> Vec<f32> {
    let n = frames * frame;
    let mut out = Vec::with_capacity(n * channels);
    for t in 0..n {
        let tf = t as f64 / 48_000.0;
        for c in 0..channels {
            let f0 = if c == 0 { 440.0 } else { 554.37 };
            let v = 0.4 * (2.0 * std::f64::consts::PI * f0 * tf).sin()
                + 0.2 * (2.0 * std::f64::consts::PI * (2.7 * f0).min(f_max) * tf).sin()
                + 0.1 * (2.0 * std::f64::consts::PI * (5.3 * f0).min(f_max) * tf).sin();
            out.push(v as f32);
        }
    }
    out
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

/// Goertzel power of `freq` Hz in a signal at `rate` Hz.
fn goertzel(x: &[f32], rate: f64, freq: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / rate;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0f64, 0f64);
    for &v in x {
        let s0 = v as f64 + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - coeff * s1 * s2
}

/// Encode a 48 kHz stream once, at each LM and channel count.
fn encode_stream(lm: u32, channels: usize, frames: usize, bytes: usize) -> Vec<Vec<u8>> {
    let mut enc = CeltRefEncoder::new(lm, channels).expect("encoder");
    let frame = enc.frame_size();
    let pcm = test_signal(frames, frame, channels, 3_600.0);
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

/// Output frame sizes and PCM lengths follow `48000 / rate` at every
/// LM, and the factor table matches the reference switch.
#[test]
fn output_sizing_follows_rate() {
    assert_eq!(resampling_factor(48_000), Some(1));
    assert_eq!(resampling_factor(24_000), Some(2));
    assert_eq!(resampling_factor(16_000), Some(3));
    assert_eq!(resampling_factor(12_000), Some(4));
    assert_eq!(resampling_factor(8_000), Some(6));
    assert_eq!(resampling_factor(44_100), None);
    for lm in 0..=3u32 {
        for &rate in &RATES {
            let d = resampling_factor(rate).unwrap();
            let mut dec = CeltRefDecoder::new_downsampled(lm, 1, rate).expect("decoder");
            assert_eq!(dec.frame_size(), 120usize << lm);
            assert_eq!(dec.output_frame_size(), (120usize << lm) / d);
            let pcm = dec.decode_frame(&[0xFF, 0xFF]).expect("silence decodes");
            assert_eq!(pcm.len(), (120usize << lm) / d);
        }
    }
}

/// The decode walk is rate-independent: a decoder at every output
/// rate reconstructs bit-identical energy state from the same frames
/// (only the synthesis tail differs).
#[test]
fn energy_state_is_rate_independent() {
    for &(lm, channels) in &[(2u32, 1usize), (3, 2)] {
        let frames = encode_stream(lm, channels, 12, 96);
        let mut d48 = CeltRefDecoder::new(lm, channels).expect("decoder");
        let mut down: Vec<CeltRefDecoder> = RATES
            .iter()
            .map(|&r| CeltRefDecoder::new_downsampled(lm, channels, r).expect("decoder"))
            .collect();
        for f in &frames {
            d48.decode_frame(f).expect("decode 48k");
            for d in down.iter_mut() {
                d.decode_frame(f).expect("decode downsampled");
                assert_eq!(
                    d.coarse.energy, d48.coarse.energy,
                    "energy state diverged at lm={lm} ch={channels}"
                );
            }
        }
    }
}

/// In-band equivalence: on band-limited content the downsampled
/// output matches the decimated 48 kHz decode (the only difference
/// is the spectrum above the output Nyquist, which the encoded tone
/// barely excites).
#[test]
fn downsampled_output_matches_decimated_decode() {
    for &(lm, channels) in &[(0u32, 1usize), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2)] {
        let frames = encode_stream(lm, channels, 10, 120 * channels);
        for &rate in &RATES {
            let d = resampling_factor(rate).unwrap();
            let mut ref48 = CeltRefDecoder::new(lm, channels).expect("decoder");
            let mut dsd = CeltRefDecoder::new_downsampled(lm, channels, rate).expect("decoder");
            let mut full = Vec::new();
            let mut ds = Vec::new();
            for f in &frames {
                full.extend(ref48.decode_frame(f).expect("decode 48k"));
                ds.extend(dsd.decode_frame(f).expect("decode downsampled"));
            }
            // Interleaved decimation: keep every d-th sample frame.
            let decimated: Vec<f32> = full
                .chunks_exact(channels)
                .step_by(d)
                .flatten()
                .copied()
                .collect();
            assert_eq!(decimated.len(), ds.len());
            let s = snr(&decimated, &ds);
            // The residual is the decoded stream's genuine energy
            // above the output Nyquist (PVQ noise floor included) —
            // largest at LM 0 / 8 kHz where 5/6 of the spectrum is
            // cut.
            assert!(
                s > 30.0,
                "decimated-vs-downsampled SNR {s:.1} dB at lm={lm} ch={channels} rate={rate}"
            );
        }
    }
}

/// The tone survives at every output rate (frequency content lands
/// where it should on the decimated grid).
#[test]
fn tone_survives_at_every_rate() {
    let lm = 3u32;
    let frames = encode_stream(lm, 1, 10, 120);
    for &rate in &RATES {
        let mut dec = CeltRefDecoder::new_downsampled(lm, 1, rate).expect("decoder");
        let mut pcm = Vec::new();
        for f in &frames {
            pcm.extend(dec.decode_frame(f).expect("decode"));
        }
        // Skip the codec delay; measure over the steady tail.
        let tail = &pcm[pcm.len() / 2..];
        let p_tone = goertzel(tail, rate as f64, 440.0);
        let p_off = goertzel(tail, rate as f64, 1_777.0);
        assert!(
            p_tone > 100.0 * p_off,
            "tone not dominant at {rate} Hz: {p_tone:.3e} vs {p_off:.3e}"
        );
    }
}

/// Robustness: random payloads decode finite and full-length at
/// every output rate.
#[test]
fn random_frames_decode_finite_downsampled() {
    for &rate in &RATES {
        for &(lm, ch, len) in &[(1u32, 1usize, 47usize), (3, 2, 160)] {
            let mut dec = CeltRefDecoder::new_downsampled(lm, ch, rate).expect("decoder");
            let mut seed = 0x1234_5678u32 ^ rate ^ (lm << 20);
            for _ in 0..4 {
                let bytes: Vec<u8> = (0..len)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed >> 24) as u8
                    })
                    .collect();
                if let Ok(pcm) = dec.decode_frame(&bytes) {
                    assert_eq!(pcm.len(), ch * dec.output_frame_size());
                    assert!(pcm.iter().all(|v| v.is_finite()));
                }
            }
        }
    }
}

/// Hybrid-layer streams (start = 17) decode at reduced output rates
/// too (the Opus decoder's SWB/FB-hybrid at 8-24 kHz output): state
/// stays lockstep with the 48 kHz hybrid decode.
#[test]
fn hybrid_layer_decodes_downsampled() {
    let lm = 2u32;
    let mut enc = CeltRefEncoder::new_with_start(lm, 1, 17).expect("encoder");
    let frame = enc.frame_size();
    // High-band content (the hybrid CELT layer's 8-20 kHz territory).
    let n = 10 * frame;
    let pcm: Vec<f32> = (0..n)
        .map(|t| {
            let tf = t as f64 / 48_000.0;
            (0.3 * (2.0 * std::f64::consts::PI * 10_000.0 * tf).sin()) as f32
        })
        .collect();
    let frames: Vec<Vec<u8>> = (0..10)
        .map(|f| {
            enc.encode_frame(&pcm[f * frame..(f + 1) * frame], 80)
                .expect("encode")
        })
        .collect();
    let mut d48 = CeltRefDecoder::new_with_start(lm, 1, 17).expect("decoder");
    for &rate in &RATES {
        let d = resampling_factor(rate).unwrap();
        let mut dec = CeltRefDecoder::new_with_start_downsampled(lm, 1, 17, rate).expect("decoder");
        for f in &frames {
            let out = dec.decode_frame(f).expect("decode");
            assert_eq!(out.len(), frame / d);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
    // The 16 kHz-output decode of a 10 kHz tone stream is near
    // silence (all coded content sits above the output Nyquist).
    let mut dec = CeltRefDecoder::new_with_start_downsampled(lm, 1, 17, 16_000).expect("decoder");
    let mut energy = 0f64;
    let mut full_energy = 0f64;
    for f in &frames {
        for v in dec.decode_frame(f).expect("decode") {
            energy += (v as f64) * (v as f64);
        }
        for v in d48.decode_frame(f).expect("decode 48k") {
            full_energy += (v as f64) * (v as f64);
        }
    }
    assert!(
        energy < 1e-3 * full_energy,
        "out-of-band content leaked: {energy:.3e} vs {full_energy:.3e}"
    );
}
