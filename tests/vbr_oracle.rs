//! VBR rate-control A/B against the staged reference oracle
//! (`docs/audio/celt/fixtures/{vbr,cvbr}-lm*`): the §A.1
//! target/drift/reservoir controller in `encode_frame_vbr` is driven
//! with the fixtures' own input signal (regenerated per the staging
//! README's generator, byte-identical semantics) at the fixtures'
//! 64 kb/s target, and its per-frame byte trajectory is compared
//! frame-by-frame with the reference listing encoder's.
//!
//! The listing encoder is not bit-reproducible even across
//! toolchains (float rounding inside the rate decisions changes
//! symbol choices), so the gates are on the controller's *behaviour*:
//! digital-silence frames collapse to exactly 2 bytes at exactly the
//! oracle's frame positions, the achieved mean rate brackets the
//! oracle's, the per-frame trajectory tracks (correlation), and the
//! decoded quality at the spent rate is not below the oracle's.
//!
//! Runtime-gated: passes with a note when the staging area is absent.

use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;
use std::path::PathBuf;

fn fixture_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("../../docs/audio/celt/fixtures"),
        PathBuf::from("docs/audio/celt/fixtures"),
    ];
    candidates.into_iter().find(|p| p.is_dir())
}

/// The staging README's deterministic test input: a two-tone mix, a
/// hard LCG noise-burst transient at samples 9600..9840, digital
/// silence at 14400..19200, and an LCG noise tail from 24000 — the
/// same generator (and therefore the same bytes) as the staged
/// `input.f32` the oracle encoded.
fn test_signal(channels: usize) -> Vec<f32> {
    let n = 28_800usize;
    let mut out = Vec::with_capacity(n * channels);
    let mut lcg: u32 = 0x2545_F491;
    let noise = |lcg: &mut u32| -> f32 {
        *lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let hi = (*lcg >> 16) as i32;
        let s16 = if hi >= 32_768 { hi - 65_536 } else { hi };
        s16 as f32 / 32_768.0
    };
    for t in 0..n {
        let tf = t as f64 / 48_000.0;
        for c in 0..channels {
            let f0: f64 = if c == 0 { 440.0 } else { 523.0 };
            let mut v = (0.28 * (2.0 * std::f64::consts::PI * f0 * tf).sin()
                + 0.18 * (2.0 * std::f64::consts::PI * 3.1 * f0 * tf).sin())
                as f32;
            if (9_600..9_840).contains(&t) {
                v += 0.5 * noise(&mut lcg);
            }
            if (14_400..19_200).contains(&t) {
                v = 0.0;
            }
            if t >= 24_000 {
                v = 0.25 * noise(&mut lcg);
            }
            out.push(v.clamp(-0.999, 0.999));
        }
    }
    out
}

/// Parse a `[u16le length][bytes]` frame stream into per-frame sizes.
fn oracle_sizes(frames: &[u8]) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut pos = 0usize;
    while pos + 2 <= frames.len() {
        let len = u16::from_le_bytes([frames[pos], frames[pos + 1]]) as usize;
        pos += 2 + len;
        sizes.push(len);
    }
    assert_eq!(pos, frames.len(), "frame stream is self-consistent");
    sizes
}

/// Float SNR of `ours` against the encoder input delayed by the
/// 120-sample-per-channel algorithmic lookahead (the alignment the
/// staging README documents for `expected.f32`).
fn snr_vs_delayed_input(ours: &[f32], input: &[f32], channels: usize) -> f64 {
    let delay = 120 * channels;
    let n = (ours.len() - delay).min(input.len() - delay);
    let (mut ss, mut ee) = (0f64, 0f64);
    for i in 0..n {
        let s = input[i] as f64;
        let d = s - ours[i + delay] as f64;
        ss += s * s;
        ee += d * d;
    }
    10.0 * (ss / ee.max(1e-30)).log10()
}

struct SetResult {
    frames: usize,
    total: usize,
    mean_kbps: f64,
    silence_frames: Vec<usize>,
    max_bytes: usize,
    sizes: Vec<usize>,
}

fn stats(sizes: &[usize], frame_size: usize) -> SetResult {
    let total: usize = sizes.iter().sum();
    let silence_frames: Vec<usize> = sizes
        .iter()
        .enumerate()
        .filter(|(_, &s)| s <= 2)
        .map(|(i, _)| i)
        .collect();
    SetResult {
        frames: sizes.len(),
        total,
        mean_kbps: total as f64 * 8.0 * 48_000.0 / (sizes.len() * frame_size) as f64 / 1000.0,
        silence_frames,
        max_bytes: sizes.iter().copied().max().unwrap_or(0),
        sizes: sizes.to_vec(),
    }
}

fn pearson(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    let ma = a.iter().sum::<usize>() as f64 / n;
    let mb = b.iter().sum::<usize>() as f64 / n;
    let (mut num, mut da, mut db) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x as f64 - ma;
        let dy = y as f64 - mb;
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    num / (da * db).sqrt().max(1e-30)
}

#[test]
fn vbr_controller_tracks_the_reference_oracle() {
    let Some(dir) = fixture_dir() else {
        eprintln!("celt raw-frame fixture staging area not present; skipping");
        return;
    };
    // (set, lm, channels, constrained)
    let sets: [(&str, u32, usize, bool); 6] = [
        ("vbr-lm0-mono-64k", 0, 1, false),
        ("vbr-lm1-stereo-64k", 1, 2, false),
        ("vbr-lm2-mono-64k", 2, 1, false),
        ("vbr-lm3-stereo-64k", 3, 2, false),
        ("cvbr-lm2-mono-64k", 2, 1, true),
        ("cvbr-lm3-stereo-64k", 3, 2, true),
    ];
    let mut measured = 0usize;
    let mut mean_by_set = std::collections::HashMap::new();
    for (name, lm, channels, constrained) in sets {
        let d = dir.join(name);
        if !d.is_dir() {
            eprintln!("{name} not staged; skipping");
            continue;
        }
        let frame_size = 120usize << lm;
        let oracle = stats(
            &oracle_sizes(&std::fs::read(d.join("frames.bin")).expect("frames.bin")),
            frame_size,
        );
        let input = test_signal(channels);

        let mut enc = CeltRefEncoder::new(lm, channels).expect("encoder");
        let mut sizes = Vec::new();
        let mut stream: Vec<Vec<u8>> = Vec::new();
        for chunk in input.chunks_exact(frame_size * channels) {
            let data = enc
                .encode_frame_vbr(chunk, 1275, 64_000, constrained)
                .expect("vbr encode");
            sizes.push(data.len());
            stream.push(data);
        }
        let ours = stats(&sizes, frame_size);

        // Decode our own stream and measure quality at the spent
        // rate, against the same delayed-input alignment the oracle's
        // expected.f32 carries.
        let mut dec = CeltRefDecoder::new(lm, channels).expect("decoder");
        let mut pcm: Vec<f32> = Vec::new();
        for f in &stream {
            pcm.extend(dec.decode_frame(f).expect("decode"));
        }
        let our_snr = snr_vs_delayed_input(&pcm, &input, channels);
        let expected: Vec<f32> = std::fs::read(d.join("expected.f32"))
            .expect("expected.f32")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let oracle_snr = snr_vs_delayed_input(&expected, &input, channels);

        let corr = pearson(&ours.sizes, &oracle.sizes);
        eprintln!(
            "{name}: frames {}/{} | total B {} vs {} | mean {:.2} vs {:.2} kb/s | \
             max B {} vs {} | 2-byte frames {} vs {} | size corr {:.3} | \
             SNR-at-rate {:.1} vs {:.1} dB",
            ours.frames,
            oracle.frames,
            ours.total,
            oracle.total,
            ours.mean_kbps,
            oracle.mean_kbps,
            ours.max_bytes,
            oracle.max_bytes,
            ours.silence_frames.len(),
            oracle.silence_frames.len(),
            corr,
            our_snr,
            oracle_snr,
        );

        assert_eq!(ours.frames, oracle.frames, "{name}: frame count");
        // Reference behaviour 1 (staging README): digital silence
        // collapses to exactly 2-byte frames — at the same positions.
        assert_eq!(
            ours.silence_frames, oracle.silence_frames,
            "{name}: 2-byte silence frame positions"
        );
        // Reference behaviour 2/3: the achieved mean tracks the
        // oracle's achieved mean (which under- and over-shoots the
        // 64k target per LM exactly as the README table pins).
        let ratio = ours.mean_kbps / oracle.mean_kbps;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "{name}: mean rate {:.2} kb/s vs oracle {:.2} (ratio {ratio:.3})",
            ours.mean_kbps,
            oracle.mean_kbps
        );
        // The per-frame trajectory tracks the oracle's (the target
        // controller reacts to the same transients and silence).
        assert!(
            corr >= 0.85,
            "{name}: per-frame size correlation {corr:.3} below 0.85"
        );
        // No frame above the wire cap, and the peak stays in the
        // oracle's neighbourhood (the transient boost, not a runaway).
        assert!(
            ours.max_bytes <= (oracle.max_bytes * 3) / 2,
            "{name}: peak frame {} B vs oracle {} B",
            ours.max_bytes,
            oracle.max_bytes
        );
        // Quality at the spent rate: not below the oracle's decode of
        // its own stream (both measured against the delayed input).
        assert!(
            our_snr >= oracle_snr - 1.0,
            "{name}: SNR at rate {our_snr:.1} dB vs oracle {oracle_snr:.1} dB"
        );
        mean_by_set.insert(name, ours.mean_kbps);
        measured += 1;
    }
    assert!(measured >= 6, "expected the six staged VBR sets");
    // Reference behaviour 3: the constrained reservoir pulls the mean
    // back below the unconstrained run at the same target.
    for (v, cv) in [
        ("vbr-lm2-mono-64k", "cvbr-lm2-mono-64k"),
        ("vbr-lm3-stereo-64k", "cvbr-lm3-stereo-64k"),
    ] {
        assert!(
            mean_by_set[cv] < mean_by_set[v],
            "constrained mean {:.2} not below unconstrained {:.2}",
            mean_by_set[cv],
            mean_by_set[v]
        );
    }
}
