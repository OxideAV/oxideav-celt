//! Encoder interop against the RFC 6716 §A reference listing, run as
//! a black-box oracle: every stream [`CeltRefEncoder`] emits must
//! decode **identically** (to the float-rounding floor) through the
//! crate's own reference-exact decoder and through a decoder built
//! from the §A.1-extracted listing, and the decoded audio must
//! approximate the input.
//!
//! Runtime-gated: set `OXIDEAV_CELT_LISTING_ORACLE` to a directory
//! containing the two harness binaries (`celt_ref_dec`,
//! `celt_ref_enc`) built from the hash-verified §A.1 extraction
//! (SHA-1 `86a927223e73d2476646a1b933fcd3fffb6ecc8c`); the test
//! passes with a note when the variable is unset (the oracle cannot
//! be redistributed inside this crate).
//!
//! Frame-stream format shared with the harnesses:
//! `[u16le length][length bytes]` per frame.

use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;
use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("OXIDEAV_CELT_LISTING_ORACLE")?);
    let dec = dir.join("celt_ref_dec");
    dec.is_file().then_some(dir)
}

/// Deterministic mixed test signal: tones, a hard transient burst, a
/// digital-silence stretch, and a noise tail.
fn test_signal(channels: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * channels);
    let mut lcg = 0x2545_F491u32;
    for t in 0..n {
        let tf = t as f32 / 48_000.0;
        for c in 0..channels {
            let f0 = if c == 0 { 440.0 } else { 523.0 };
            let mut v = 0.28 * (2.0 * std::f32::consts::PI * f0 * tf).sin()
                + 0.18 * (2.0 * std::f32::consts::PI * 3.1 * f0 * tf).sin();
            if (9_600..9_840).contains(&t) {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                v += 0.5 * ((lcg >> 16) as i16 as f32 / 32768.0);
            }
            if (14_400..19_200).contains(&t) {
                v = 0.0;
            }
            if t >= 24_000 {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                v = 0.25 * ((lcg >> 16) as i16 as f32 / 32768.0);
            }
            out.push(v.clamp(-0.999, 0.999));
        }
    }
    out
}

fn write_frames(path: &Path, frames: &[Vec<u8>]) {
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(&(f.len() as u16).to_le_bytes());
        out.extend_from_slice(f);
    }
    std::fs::write(path, out).expect("write frames");
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .expect("read f32")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let n = reference.len().min(test.len());
    let mut ee = 0f64;
    let mut err = 0f64;
    for i in 0..n {
        let e = reference[i] as f64;
        ee += e * e;
        let d = e - test[i] as f64;
        err += d * d;
    }
    10.0 * (ee / err.max(1e-300)).log10()
}

/// Encode with this crate, decode with BOTH decoders, compare.
#[test]
fn encoder_streams_decode_identically_on_the_listing_decoder() {
    let Some(dir) = oracle_dir() else {
        eprintln!("OXIDEAV_CELT_LISTING_ORACLE not set; skipping oracle interop");
        return;
    };
    let tmp = std::env::temp_dir().join(format!("oxideav-celt-r417-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let mut measured = 0usize;
    for &(lm, channels, bytes) in &[
        (0u32, 1usize, 43usize),
        (1, 1, 60),
        (2, 1, 100),
        (3, 1, 160),
        (0, 2, 64),
        (1, 2, 90),
        (2, 2, 141),
        (3, 2, 239),
    ] {
        let frame = 120usize << lm;
        let n = 28_800usize;
        let frames_n = n / frame;
        let pcm = test_signal(channels, n);

        let mut enc = CeltRefEncoder::new(lm, channels).expect("encoder");
        let mut dec = CeltRefDecoder::new(lm, channels).expect("decoder");
        let mut frames: Vec<Vec<u8>> = Vec::with_capacity(frames_n);
        let mut ours: Vec<f32> = Vec::new();
        for f in 0..frames_n {
            let chunk = &pcm[f * frame * channels..(f + 1) * frame * channels];
            let bytes_out = enc.encode_frame(chunk, bytes).expect("encode");
            ours.extend(dec.decode_frame(&bytes_out).expect("decode"));
            frames.push(bytes_out);
        }

        let fpath = tmp.join(format!("s_{lm}_{channels}.frames"));
        let opath = tmp.join(format!("s_{lm}_{channels}.f32"));
        write_frames(&fpath, &frames);
        let status = Command::new(dir.join("celt_ref_dec"))
            .args([
                &channels.to_string(),
                &frame.to_string(),
                fpath.to_str().unwrap(),
                opath.to_str().unwrap(),
            ])
            .status()
            .expect("oracle decoder runs");
        assert!(status.success(), "oracle decoder rejected the stream");
        let theirs = read_f32(&opath);
        assert_eq!(theirs.len(), ours.len(), "decode length mismatch");

        // The two decodes must agree at the float-rounding floor.
        // The pair's numerical floor (measured on reference-encoded
        // streams through the same two decoders) is ~99 dB at LM 0
        // and ~117 dB at LM 2/3 — the listing's f32 FFT accumulates
        // more rounding than this crate's f64-accumulated IMDCT. A
        // symbol-level parse divergence would show as a localized
        // multi-orders-of-magnitude error, caught by the max-diff
        // bound below.
        let cross = snr_db(&theirs, &ours);
        let maxdiff = theirs
            .iter()
            .zip(ours.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // And both must resemble the (120-sample-delayed) input.
        let delay = 120 * channels;
        let quality = snr_db(&pcm[..ours.len() - delay], &ours[delay..]);
        eprintln!(
            "lm={lm} C={channels} {bytes}B: cross-decoder {cross:.1} dB \
             (maxdiff {maxdiff:.2e}), quality {quality:.2} dB"
        );
        assert!(
            cross >= 95.0,
            "lm={lm} C={channels}: decoders disagree ({cross:.1} dB)"
        );
        assert!(
            maxdiff <= 1e-4,
            "lm={lm} C={channels}: localized decoder divergence ({maxdiff:.2e})"
        );
        assert!(
            quality >= 5.0,
            "lm={lm} C={channels}: quality {quality:.2} dB too low"
        );
        measured += 1;
    }
    assert!(measured == 8);
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Rate/quality sweep vs the reference listing encoder at matched
/// frame sizes and byte budgets: report both SNRs (decoded through
/// the same listing decoder) and require this crate's encoder to stay
/// within a fixed window of the reference encoder's quality.
#[test]
fn rate_quality_sweep_vs_listing_encoder() {
    let Some(dir) = oracle_dir() else {
        eprintln!("OXIDEAV_CELT_LISTING_ORACLE not set; skipping oracle sweep");
        return;
    };
    if !dir.join("celt_ref_enc").is_file() {
        eprintln!("celt_ref_enc not built; skipping oracle sweep");
        return;
    }
    let tmp = std::env::temp_dir().join(format!("oxideav-celt-r417s-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Steady tonal material (the regime where SNR is comparable).
    let n = 48_000usize;
    for &(lm, channels) in &[(1u32, 1usize), (2, 1), (3, 1), (2, 2)] {
        let frame = 120usize << lm;
        let pcm: Vec<f32> = test_signal(channels, n)[..n * channels]
            .iter()
            .map(|&v| v * 0.9)
            .collect();
        let frames_n = 9_600 / frame; // first 0.2 s: pure tone segment
        for &bytes in &[40usize, 80, 160] {
            let budget = bytes.min(1275);
            // Ours.
            let mut enc = CeltRefEncoder::new(lm, channels).expect("encoder");
            let mut frames: Vec<Vec<u8>> = Vec::new();
            for f in 0..frames_n {
                let chunk = &pcm[f * frame * channels..(f + 1) * frame * channels];
                frames.push(enc.encode_frame(chunk, budget).expect("encode"));
            }
            let ours_frames = tmp.join(format!("o_{lm}_{channels}_{bytes}.frames"));
            let ours_out = tmp.join(format!("o_{lm}_{channels}_{bytes}.f32"));
            write_frames(&ours_frames, &frames);
            assert!(Command::new(dir.join("celt_ref_dec"))
                .args([
                    &channels.to_string(),
                    &frame.to_string(),
                    ours_frames.to_str().unwrap(),
                    ours_out.to_str().unwrap(),
                ])
                .status()
                .expect("oracle decoder runs")
                .success());

            // Reference encoder on the same material.
            let in_f32 = tmp.join(format!("in_{lm}_{channels}.f32"));
            let mut raw = Vec::new();
            for v in &pcm[..frames_n * frame * channels] {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(&in_f32, raw).unwrap();
            let ref_frames = tmp.join(format!("r_{lm}_{channels}_{bytes}.frames"));
            let ref_out = tmp.join(format!("r_{lm}_{channels}_{bytes}.f32"));
            assert!(Command::new(dir.join("celt_ref_enc"))
                .args([
                    &channels.to_string(),
                    &frame.to_string(),
                    &budget.to_string(),
                    in_f32.to_str().unwrap(),
                    ref_frames.to_str().unwrap(),
                ])
                .status()
                .expect("oracle encoder runs")
                .success());
            assert!(Command::new(dir.join("celt_ref_dec"))
                .args([
                    &channels.to_string(),
                    &frame.to_string(),
                    ref_frames.to_str().unwrap(),
                    ref_out.to_str().unwrap(),
                ])
                .status()
                .expect("oracle decoder runs")
                .success());

            let ours = read_f32(&ours_out);
            let theirs = read_f32(&ref_out);
            let delay = 120 * channels;
            let skip = 2 * frame * channels;
            let seg = &pcm[skip..frames_n * frame * channels - delay];
            let snr_ours = snr_db(seg, &ours[skip + delay..]);
            let snr_ref = snr_db(seg, &theirs[skip + delay..]);
            eprintln!(
                "lm={lm} C={channels} {budget}B/frame: ours {snr_ours:.2} dB, listing {snr_ref:.2} dB"
            );
            assert!(
                snr_ours >= snr_ref - 12.0,
                "lm={lm} C={channels} {budget}B: ours {snr_ours:.2} dB vs listing {snr_ref:.2} dB"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
