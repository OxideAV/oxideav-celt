//! High-rate coverage (the r417 "high-rate mono" followup): the
//! encoder/decoder pair at 192-384 kb/s CBR operating points, both
//! self-consistently (always-on) and A/B against reference oracle
//! harness binaries built from the staged listing (runtime-gated on
//! `CELT_HYB_ENC` / `CELT_HYB_DEC`, invoked as opaque processes; the
//! harness takes the start band as an argument — 0 here, pure CELT).

use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The staged fixtures' deterministic mixed test input (tones, hard
/// burst, digital silence, noise tail) — same generator as the
/// staging README's, byte-identical to the staged `input.f32`.
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

fn snr_delayed(ours: &[f32], input: &[f32], channels: usize) -> f64 {
    let delay = 120 * channels;
    let n = (ours.len() - delay).min(input.len() - delay);
    snr_delayed_window(ours, input, channels, 0, n)
}

/// SNR over `input[from..to]` (interleaved-sample indices) against
/// `ours` delayed by the 120-sample lookahead.
fn snr_delayed_window(ours: &[f32], input: &[f32], channels: usize, from: usize, to: usize) -> f64 {
    let delay = 120 * channels;
    let to = to.min(ours.len() - delay).min(input.len());
    let (mut ss, mut ee) = (0f64, 0f64);
    for i in from..to {
        let s = input[i] as f64;
        let d = s - ours[i + delay] as f64;
        ss += s * s;
        ee += d * d;
    }
    10.0 * (ss / ee.max(1e-30)).log10()
}

/// Self-consistent high-rate sweep (always on): mono 192 and
/// 384 kb/s at LM 2/3 encode exact-size, decode through the crate's
/// own decoder, and clear a quality floor well above the 64 kb/s
/// regime's on the steady tonal stretch (the mixed signal's full-file
/// SNR is capped ~17.5 dB by the burst region at ANY rate — the
/// oracle saturates identically there).
#[test]
fn highrate_mono_self_roundtrip() {
    for &(lm, bytes, floor_db) in &[
        (2u32, 240usize, 30.0f64),
        (2, 480, 35.0),
        (3, 480, 30.0),
        (3, 960, 35.0),
    ] {
        let frame = 120usize << lm;
        let input = test_signal(1);
        let mut enc = CeltRefEncoder::new(lm, 1).expect("enc");
        let mut dec = CeltRefDecoder::new(lm, 1).expect("dec");
        let mut pcm = Vec::new();
        for chunk in input.chunks_exact(frame) {
            let data = enc.encode_frame(chunk, bytes).expect("encode");
            assert_eq!(data.len(), bytes);
            pcm.extend(dec.decode_frame(&data).expect("decode"));
        }
        // Steady tonal window: past the 4-frame warmup, before the
        // 9600-sample burst.
        let snr = snr_delayed_window(&pcm, &input, 1, 4 * frame, 9_000);
        let kbps = bytes * 8 * 48_000 / frame / 1000;
        eprintln!("high-rate mono lm={lm} {bytes}B ({kbps} kb/s): tonal SNR {snr:.1} dB");
        assert!(
            snr >= floor_db,
            "lm={lm} {bytes}B: tonal SNR {snr:.1} dB below the {floor_db} dB floor"
        );
    }
}

fn oracle_bins() -> Option<(PathBuf, PathBuf)> {
    let enc = std::env::var_os("CELT_HYB_ENC")?;
    let dec = std::env::var_os("CELT_HYB_DEC")?;
    Some((PathBuf::from(enc), PathBuf::from(dec)))
}

fn write_f32(path: &Path, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write f32");
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .expect("read f32")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run(cmd: &Path, args: &[&str]) {
    let out = Command::new(cmd).args(args).output().expect("oracle runs");
    assert!(
        out.status.success(),
        "{cmd:?} {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// High-rate A/B against the listing oracle (runtime-gated): the
/// oracle's high-rate streams decode reference-exactly through our
/// decoder, and our encoder's quality at the same rate is not below
/// the oracle's.
#[test]
fn highrate_oracle_ab() {
    let Some((enc_bin, dec_bin)) = oracle_bins() else {
        eprintln!("CELT_HYB_ENC/CELT_HYB_DEC not set; skipping high-rate oracle A/B");
        return;
    };
    let dir = std::env::temp_dir().join("celt-highrate-oracle-ab");
    std::fs::create_dir_all(&dir).expect("tmp dir");

    for &(lm, channels, bytes) in &[
        (1u32, 1usize, 120usize),
        (2, 1, 240),
        (2, 1, 480),
        (3, 1, 480),
        (3, 1, 960),
        (3, 2, 960),
    ] {
        let frame = 120usize << lm;
        let fs = frame.to_string();
        let ch = channels.to_string();
        let input = test_signal(channels);
        let in_f32 = dir.join(format!("in-{channels}.f32"));
        write_f32(&in_f32, &input);

        let oracle_frames = dir.join(format!("o-{lm}-{channels}-{bytes}.frames"));
        let oracle_dec = dir.join(format!("o-{lm}-{channels}-{bytes}.f32"));
        run(
            &enc_bin,
            &[
                &ch,
                &fs,
                "0",
                "cbr",
                &bytes.to_string(),
                in_f32.to_str().unwrap(),
                oracle_frames.to_str().unwrap(),
            ],
        );
        run(
            &dec_bin,
            &[
                &ch,
                &fs,
                "0",
                oracle_frames.to_str().unwrap(),
                oracle_dec.to_str().unwrap(),
            ],
        );

        // Our decoder on the oracle's high-rate stream.
        let stream = std::fs::read(&oracle_frames).expect("frames");
        let mut dec = CeltRefDecoder::new(lm, channels).expect("dec");
        let mut ours = Vec::new();
        let mut pos = 0usize;
        while pos + 2 <= stream.len() {
            let len = u16::from_le_bytes([stream[pos], stream[pos + 1]]) as usize;
            pos += 2;
            ours.extend(dec.decode_frame(&stream[pos..pos + len]).expect("decode"));
            pos += len;
        }
        let expected = read_f32(&oracle_dec);
        let n = ours.len().min(expected.len());
        let (mut ee, mut err) = (0f64, 0f64);
        for i in 0..n {
            let e = expected[i] as f64;
            let d = e - ours[i] as f64;
            ee += e * e;
            err += d * d;
        }
        let dec_snr = 10.0 * (ee / err.max(1e-30)).log10();

        // Our encoder at the same rate, oracle-decoded.
        let mut enc = CeltRefEncoder::new(lm, channels).expect("enc");
        let mut out_stream = Vec::new();
        for chunk in input.chunks_exact(frame * channels) {
            let data = enc.encode_frame(chunk, bytes).expect("encode");
            out_stream.extend_from_slice(&(data.len() as u16).to_le_bytes());
            out_stream.extend_from_slice(&data);
        }
        let ours_frames = dir.join(format!("u-{lm}-{channels}-{bytes}.frames"));
        std::fs::write(&ours_frames, &out_stream).expect("write");
        let ours_dec = dir.join(format!("u-{lm}-{channels}-{bytes}.f32"));
        run(
            &dec_bin,
            &[
                &ch,
                &fs,
                "0",
                ours_frames.to_str().unwrap(),
                ours_dec.to_str().unwrap(),
            ],
        );
        let theirs = read_f32(&ours_dec);
        let q_ours = snr_delayed(&theirs, &input, channels);
        let q_oracle = snr_delayed(&expected, &input, channels);
        // The steady tonal stretch separates the rate points the
        // full-file number hides behind the burst region.
        let (wf, wt) = (4 * frame * channels, 9_000 * channels);
        let t_ours = snr_delayed_window(&theirs, &input, channels, wf, wt);
        let t_oracle = snr_delayed_window(&expected, &input, channels, wf, wt);
        let kbps = bytes * 8 * 48_000 / frame / 1000;
        eprintln!(
            "high-rate lm={lm} ch={channels} {bytes}B ({kbps} kb/s): decode {dec_snr:.1} dB, \
             quality ours {q_ours:.1} vs oracle {q_oracle:.1} dB \
             (tonal {t_ours:.1} vs {t_oracle:.1} dB)"
        );
        assert!(
            dec_snr >= 95.0,
            "lm={lm} ch={channels} {bytes}B: decode SNR {dec_snr:.1} dB"
        );
        assert!(
            q_ours >= q_oracle - 1.0,
            "lm={lm} ch={channels} {bytes}B: quality {q_ours:.1} vs oracle {q_oracle:.1} dB"
        );
        assert!(
            t_ours >= t_oracle - 1.0,
            "lm={lm} ch={channels} {bytes}B: tonal quality {t_ours:.1} vs oracle {t_oracle:.1} dB"
        );
    }
}
