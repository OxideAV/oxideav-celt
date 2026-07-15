//! Reference-exactness measurement on real CELT streams: decode the
//! workspace-staged Ogg-Opus fixtures (reference-encoder output, used
//! strictly as black-box byte streams) with the reference-exact
//! decoder and compare the PCM against the staged reference decode
//! (`expected.wav`).
//!
//! Runtime-gated: the fixtures live in the umbrella workspace's
//! `docs/` staging area, which the standalone crate CI does not carry
//! — the test passes with a note when the directory is absent.

mod common;

use common::{ogg_packets, opus_frames, wav_pcm_f32};
use oxideav_celt::ref_decode::CeltRefDecoder;
use std::path::PathBuf;

fn fixture_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("../../docs/audio/opus/fixtures"),
        PathBuf::from("docs/audio/opus/fixtures"),
    ];
    candidates.into_iter().find(|p| p.is_dir())
}

struct FixtureResult {
    snr_db: f64,
    gain: f64,
}

fn decode_fixture(dir: &std::path::Path) -> FixtureResult {
    let fixture_name = dir.file_name().unwrap().to_string_lossy().to_string();
    let opus = std::fs::read(dir.join("input.opus")).expect("input.opus");
    let expected = wav_pcm_f32(&std::fs::read(dir.join("expected.wav")).expect("expected.wav"));

    let packets = ogg_packets(&opus);
    assert!(&packets[0][..8] == b"OpusHead");
    let channels = packets[0][9] as usize;
    let pre_skip = u16::from_le_bytes([packets[0][10], packets[0][11]]) as usize;
    let gain_q8 = i16::from_le_bytes([packets[0][16], packets[0][17]]);
    assert_eq!(gain_q8, 0, "fixture has no output gain");

    let max_packets: usize = std::env::var("CELT_FIXTURE_MAX_PACKETS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let mut decoder: Option<CeltRefDecoder> = None;
    let mut pcm: Vec<f32> = Vec::new();
    for packet in packets[2..].iter().take(max_packets) {
        let Some((toc, frames)) = opus_frames(packet) else {
            continue;
        };
        let config = toc >> 3;
        assert!((16..=31).contains(&config), "CELT-only fixture");
        let lm = (config & 3) as u32;
        let stereo = toc & 4 != 0;
        let ch = if stereo { 2 } else { 1 };
        assert_eq!(ch, channels, "fixture keeps its channel count");
        let d = decoder.get_or_insert_with(|| CeltRefDecoder::new(lm, ch).expect("decoder"));
        for f in &frames {
            let out = d.decode_frame(f).expect("frame decodes");
            pcm.extend_from_slice(&out);
        }
    }

    if let Ok(path) = std::env::var("CELT_DUMP_F32") {
        let path = format!("{path}.{fixture_name}");
        let mut bytes = Vec::with_capacity(pcm.len() * 4);
        for v in &pcm {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    // Trim the encoder lookahead the reference decode trims.
    let ours = &pcm[channels * pre_skip..];
    let n = ours.len().min(expected.len());
    let (mut dot, mut ee, mut oo) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let e = expected[i] as f64;
        let o = ours[i] as f64;
        dot += e * o;
        ee += e * e;
        oo += o * o;
    }
    let gain = if oo > 0.0 { dot / oo } else { 0.0 };
    // SNR of the raw (ungained) difference.
    let mut err = 0f64;
    for i in 0..n {
        let d = expected[i] as f64 - ours[i] as f64;
        err += d * d;
    }
    let snr_db = 10.0 * (ee / err.max(1e-30)).log10();
    FixtureResult { snr_db, gain }
}

#[test]
fn celt_fixture_reference_exactness() {
    let Some(dir) = fixture_dir() else {
        eprintln!("fixture staging area not present; skipping");
        return;
    };
    for (name, floor_db) in [
        ("celt-fb-stereo-128kbps", 80.0),
        ("celt-2.5ms-low-latency", 80.0),
    ] {
        let d = dir.join(name);
        if !d.is_dir() {
            eprintln!("fixture {name} not staged; skipping");
            continue;
        }
        let r = decode_fixture(&d);
        eprintln!("{name}: SNR {:.2} dB, LS gain {:.6}", r.snr_db, r.gain);
        assert!(
            r.snr_db >= floor_db,
            "{name}: SNR {:.2} dB below the {floor_db} dB floor (gain {:.6})",
            r.snr_db,
            r.gain
        );
    }
}
