//! Reference-exactness generalization sweep: encode locally-generated
//! test material with a black-box reference encoder (`opusenc`) at
//! every CELT frame size in mono and stereo, decode the streams with
//! [`CeltRefDecoder`], and compare against the black-box reference
//! decode (`opusdec --float`) at float precision.
//!
//! Runtime-gated: passes with a note when `opusenc`/`opusdec` are not
//! installed (they are invoked strictly as opaque validator binaries).

mod common;

use common::{ogg_packets, opus_frames, wav_pcm_f32, write_wav_s16};
use oxideav_celt::ref_decode::CeltRefDecoder;
use std::path::PathBuf;
use std::process::Command;

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Deterministic 0.6 s test signal: a tone mix, a hard transient
/// burst, a near-silent stretch, and an LCG noise tail — exercising
/// long and short blocks, folding, silence flags, and anti-collapse.
fn test_signal(channels: usize) -> Vec<i16> {
    let n = 28_800usize; // 0.6 s @ 48 kHz
    let mut out = Vec::with_capacity(n * channels);
    let mut lcg = 0x2545_F491u32;
    for t in 0..n {
        let tf = t as f32 / 48_000.0;
        for c in 0..channels {
            let f0 = if c == 0 { 440.0 } else { 523.0 };
            let mut v = 0.28 * (2.0 * std::f32::consts::PI * f0 * tf).sin()
                + 0.18 * (2.0 * std::f32::consts::PI * 3.1 * f0 * tf).sin();
            // Transient burst.
            if (9_600..9_840).contains(&t) {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                v += 0.5 * ((lcg >> 16) as i16 as f32 / 32768.0);
            }
            // Near-silence.
            if (14_400..19_200).contains(&t) {
                v *= 0.001;
            }
            // Noise tail.
            if t >= 24_000 {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                v = 0.25 * ((lcg >> 16) as i16 as f32 / 32768.0);
            }
            out.push((v.clamp(-0.999, 0.999) * 32767.0) as i16);
        }
    }
    out
}

struct SweepResult {
    snr_db: f64,
    frames: usize,
}

fn run_case(
    dir: &std::path::Path,
    channels: usize,
    framesize: &str,
    bitrate: &str,
) -> Option<SweepResult> {
    let wav = dir.join(format!("in_{channels}.wav"));
    let opus = dir.join(format!("c{channels}_f{framesize}.opus"));
    let refwav = dir.join(format!("c{channels}_f{framesize}_ref.wav"));

    let enc = Command::new("opusenc")
        .args([
            "--quiet",
            "--framesize",
            framesize,
            "--bitrate",
            bitrate,
            wav.to_str().unwrap(),
            opus.to_str().unwrap(),
        ])
        .status()
        .ok()?;
    assert!(enc.success(), "opusenc failed");
    let dec = Command::new("opusdec")
        .args([
            "--quiet",
            "--float",
            opus.to_str().unwrap(),
            refwav.to_str().unwrap(),
        ])
        .status()
        .ok()?;
    assert!(dec.success(), "opusdec failed");

    let opus_bytes = std::fs::read(&opus).unwrap();
    let expected = wav_pcm_f32(&std::fs::read(&refwav).unwrap());
    let packets = ogg_packets(&opus_bytes);
    assert!(&packets[0][..8] == b"OpusHead");
    let pre_skip = u16::from_le_bytes([packets[0][10], packets[0][11]]) as usize;

    let mut decoder: Option<CeltRefDecoder> = None;
    let mut pcm: Vec<f32> = Vec::new();
    let mut frames_total = 0usize;
    for packet in &packets[2..] {
        let Some((toc, frames)) = opus_frames(packet) else {
            continue;
        };
        let config = toc >> 3;
        if !(16..=31).contains(&config) {
            // The reference encoder chose a SILK/Hybrid mode for this
            // stream; the sweep only measures CELT.
            eprintln!("  non-CELT config {config}; skipping combo");
            return None;
        }
        let lm = (config & 3) as u32;
        let ch = if toc & 4 != 0 { 2 } else { 1 };
        assert_eq!(ch, channels);
        let d = decoder.get_or_insert_with(|| CeltRefDecoder::new(lm, ch).expect("decoder"));
        for f in &frames {
            let out = d.decode_frame(f).expect("frame decodes");
            pcm.extend_from_slice(&out);
            frames_total += 1;
        }
    }

    let ours = &pcm[channels * pre_skip..];
    let n = ours.len().min(expected.len());
    let mut ee = 0f64;
    let mut err = 0f64;
    for i in 0..n {
        let e = expected[i] as f64;
        ee += e * e;
        let d = e - ours[i] as f64;
        err += d * d;
    }
    Some(SweepResult {
        snr_db: 10.0 * (ee / err.max(1e-30)).log10(),
        frames: frames_total,
    })
}

#[test]
fn celt_reference_decode_sweep() {
    if !tool_available("opusenc") || !tool_available("opusdec") {
        eprintln!("opusenc/opusdec not installed; skipping black-box sweep");
        return;
    }
    let dir: PathBuf =
        std::env::temp_dir().join(format!("oxideav-celt-bbx-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    for channels in [1usize, 2] {
        write_wav_s16(
            &dir.join(format!("in_{channels}.wav")),
            channels as u16,
            &test_signal(channels),
        );
    }

    let mut measured = 0usize;
    for &(channels, framesize, bitrate) in &[
        (1usize, "2.5", "96"),
        (1, "5", "96"),
        (1, "10", "128"),
        (1, "20", "128"),
        (2, "2.5", "128"),
        (2, "5", "128"),
        (2, "10", "160"),
        (2, "20", "160"),
    ] {
        let Some(r) = run_case(&dir, channels, framesize, bitrate) else {
            continue;
        };
        eprintln!(
            "C={channels} {framesize} ms @ {bitrate} kb/s: {} frames, SNR {:.2} dB",
            r.frames, r.snr_db
        );
        assert!(
            r.snr_db >= 90.0,
            "C={channels} {framesize} ms: SNR {:.2} dB below the 90 dB float floor",
            r.snr_db
        );
        measured += 1;
    }
    assert!(measured >= 4, "too few CELT combos measured ({measured})");
    let _ = std::fs::remove_dir_all(&dir);
}
