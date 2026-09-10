//! Equal-rate quality matrix against the RFC 6716 §A.1 reference
//! listing encoder: mono/stereo x 2.5/5/10/20 ms x 6-128 kb/s CBR,
//! both encoders' streams decoded through the same listing decoder
//! and scored against the (120-sample-delayed) input, on three kinds
//! of material — the steady tonal segment, a harmonic "music-like"
//! signal with vibrato, a noise floor and hits, and a partially
//! correlated stereo pair (intensity / dual / mid-side territory).
//!
//! Runtime-gated on `OXIDEAV_CELT_LISTING_ORACLE` (a directory with
//! the `celt_ref_enc` / `celt_ref_dec` harness binaries built from
//! the hash-verified §A.1 extraction); passes with a note otherwise.
//! `CELT_MATRIX_FULL=1` widens the sweep (every LM x every rate) and
//! `CELT_MATRIX_LM=n` restricts it to one frame size; the default
//! keeps a representative subset so the gate stays fast.

use oxideav_celt::ref_encode::CeltRefEncoder;
use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("OXIDEAV_CELT_LISTING_ORACLE")?);
    (dir.join("celt_ref_dec").is_file() && dir.join("celt_ref_enc").is_file()).then_some(dir)
}

const N: usize = 48_000;

/// Steady two-tone material (the regime where SNR is most telling).
fn tonal(channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(N * channels);
    for t in 0..N {
        let tf = t as f32 / 48_000.0;
        for c in 0..channels {
            let f0 = if c == 0 { 440.0 } else { 523.0 };
            let v = 0.25 * (2.0 * std::f32::consts::PI * f0 * tf).sin()
                + 0.16 * (2.0 * std::f32::consts::PI * 3.1 * f0 * tf).sin();
            out.push(v);
        }
    }
    out
}

/// Harmonic "music-like" material: a 12-partial tone with vibrato and
/// a decaying spectral tilt, a -30 dB noise floor, and percussive
/// hits every 100 ms — long and short blocks, folding, boosts.
fn music(channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(N * channels);
    let mut lcg = 0x2545_F491u32;
    let mut noise = || {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((lcg >> 16) as i16) as f32 / 32768.0
    };
    for t in 0..N {
        let tf = t as f32 / 48_000.0;
        for c in 0..channels {
            let f0 = (if c == 0 { 220.0 } else { 261.6 })
                * (1.0 + 0.006 * (2.0 * std::f32::consts::PI * 5.5 * tf).sin());
            let mut v = 0.0f32;
            for h in 1..=12u32 {
                let a = 0.22 / (h as f32).powf(1.3);
                v += a * (2.0 * std::f32::consts::PI * f0 * h as f32 * tf).sin();
            }
            v += 0.03 * noise();
            let hit = t % 4_800;
            if hit < 480 {
                let env = (-(hit as f32) / 80.0).exp();
                v += 0.45 * env * noise();
            }
            out.push(v.clamp(-0.999, 0.999));
        }
    }
    out
}

/// Partially correlated stereo: a shared mid source plus per-channel
/// side content and a slow pan — the intensity / dual / theta regime.
fn stereo_pair() -> Vec<f32> {
    let mut out = Vec::with_capacity(N * 2);
    let mut lcg = 0x9E37_79B9u32;
    let mut noise = || {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((lcg >> 16) as i16) as f32 / 32768.0
    };
    for t in 0..N {
        let tf = t as f32 / 48_000.0;
        let pan = 0.5 + 0.4 * (2.0 * std::f32::consts::PI * 0.7 * tf).sin();
        let mut mid = 0.0f32;
        for h in 1..=8u32 {
            mid += 0.2 / h as f32 * (2.0 * std::f32::consts::PI * 330.0 * h as f32 * tf).sin();
        }
        let side_l = 0.08 * (2.0 * std::f32::consts::PI * 1_760.0 * tf).sin();
        let side_r = 0.08 * (2.0 * std::f32::consts::PI * 2_093.0 * tf).sin();
        let n = 0.02 * noise();
        let l = mid * (1.0 - pan) + side_l + n;
        let r = mid * pan + side_r + 0.02 * noise();
        out.push(l.clamp(-0.999, 0.999));
        out.push(r.clamp(-0.999, 0.999));
    }
    out
}

fn write_frames(path: &Path, frames: &[Vec<u8>]) {
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(&(f.len() as u16).to_le_bytes());
        out.extend_from_slice(f);
    }
    std::fs::write(path, out).unwrap();
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let n = reference.len().min(test.len());
    let (mut ss, mut ee) = (0f64, 0f64);
    for i in 0..n {
        let s = reference[i] as f64;
        let d = s - test[i] as f64;
        ss += s * s;
        ee += d * d;
    }
    10.0 * (ss / ee.max(1e-30)).log10()
}

fn run(cmd: &Path, args: &[&str]) {
    let out = Command::new(cmd).args(args).output().expect("oracle runs");
    assert!(
        out.status.success(),
        "{cmd:?} {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

struct Point {
    name: &'static str,
    channels: usize,
    lm: u32,
    kbps: u32,
    ours: f64,
    listing: f64,
}

fn measure(
    dir: &Path,
    tmp: &Path,
    name: &'static str,
    pcm: &[f32],
    channels: usize,
    lm: u32,
    kbps: u32,
) -> Option<Point> {
    let frame = 120usize << lm;
    // bytes/frame = kb/s * 1000 * frame / 48000 / 8 = kbps * frame / 384
    let bytes = (kbps as usize * frame) / 384;
    if !(2..=1275).contains(&bytes) {
        return None;
    }
    let frames_n = N / frame;
    let mut enc = CeltRefEncoder::new(lm, channels).expect("encoder");
    let mut frames = Vec::with_capacity(frames_n);
    for f in 0..frames_n {
        let chunk = &pcm[f * frame * channels..(f + 1) * frame * channels];
        frames.push(enc.encode_frame(chunk, bytes).expect("encode"));
    }
    let tag = format!("{name}_{channels}_{lm}_{kbps}");
    let ours_frames = tmp.join(format!("o_{tag}.frames"));
    let ours_out = tmp.join(format!("o_{tag}.f32"));
    write_frames(&ours_frames, &frames);
    let ch = channels.to_string();
    let fs = frame.to_string();
    run(
        &dir.join("celt_ref_dec"),
        &[
            &ch,
            &fs,
            ours_frames.to_str().unwrap(),
            ours_out.to_str().unwrap(),
        ],
    );
    let in_f32 = tmp.join(format!("in_{name}_{channels}.f32"));
    if !in_f32.is_file() {
        let mut raw = Vec::with_capacity(pcm.len() * 4);
        for v in pcm {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&in_f32, raw).unwrap();
    }
    let ref_frames = tmp.join(format!("r_{tag}.frames"));
    let ref_out = tmp.join(format!("r_{tag}.f32"));
    run(
        &dir.join("celt_ref_enc"),
        &[
            &ch,
            &fs,
            &bytes.to_string(),
            in_f32.to_str().unwrap(),
            ref_frames.to_str().unwrap(),
        ],
    );
    run(
        &dir.join("celt_ref_dec"),
        &[
            &ch,
            &fs,
            ref_frames.to_str().unwrap(),
            ref_out.to_str().unwrap(),
        ],
    );
    let ours = read_f32(&ours_out);
    let theirs = read_f32(&ref_out);
    let delay = 120 * channels;
    let skip = 4 * frame * channels;
    let seg = &pcm[skip..frames_n * frame * channels - delay];
    Some(Point {
        name,
        channels,
        lm,
        kbps,
        ours: snr_db(seg, &ours[skip + delay..]),
        listing: snr_db(seg, &theirs[skip + delay..]),
    })
}

#[test]
fn equal_rate_matrix_vs_listing_encoder() {
    let Some(dir) = oracle_dir() else {
        eprintln!("OXIDEAV_CELT_LISTING_ORACLE not set; skipping equal-rate matrix");
        return;
    };
    let full = std::env::var("CELT_MATRIX_FULL").as_deref() == Ok("1");
    let tmp = std::env::temp_dir().join(format!("oxideav-celt-matrix-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let rates: &[u32] = if full {
        &[6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    } else {
        &[8, 16, 32, 64, 128]
    };
    // `CELT_MATRIX_LM=n` restricts the sweep to one frame size.
    let only_lm: Option<u32> = std::env::var("CELT_MATRIX_LM")
        .ok()
        .and_then(|v| v.parse().ok());
    let lms: Vec<u32> = match only_lm {
        Some(l) => vec![l],
        None if full => vec![0, 1, 2, 3],
        None => vec![1, 3],
    };
    let mut points = Vec::new();
    for &lm in &lms {
        for &kbps in rates {
            let only_mat = std::env::var("CELT_MATRIX_MATERIAL").ok();
            for (name, channels, pcm) in [
                ("tonal", 1usize, tonal(1)),
                ("music", 1, music(1)),
                ("tonal", 2, tonal(2)),
                ("music", 2, music(2)),
                ("pair", 2, stereo_pair()),
            ] {
                if only_mat.as_deref().is_some_and(|m| m != name) {
                    continue;
                }
                if let Some(p) = measure(&dir, &tmp, name, &pcm, channels, lm, kbps) {
                    points.push(p);
                }
            }
        }
    }
    eprintln!("material  C  LM  kb/s   ours  listing  delta");
    let mut behind = Vec::new();
    for p in &points {
        let delta = p.ours - p.listing;
        eprintln!(
            "{:<8} {:>2} {:>3} {:>5}  {:>6.2} {:>7.2}  {:>+6.2}",
            p.name, p.channels, p.lm, p.kbps, p.ours, p.listing, delta
        );
        if delta < -0.5 {
            behind.push(format!(
                "{} C={} LM={} {} kb/s: ours {:.2} vs listing {:.2}",
                p.name, p.channels, p.lm, p.kbps, p.ours, p.listing
            ));
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
        behind.is_empty(),
        "encoder behind the listing by more than 0.5 dB at:\n{}",
        behind.join("\n")
    );
}
