//! Custom-mode (non-48 kHz) black-box A/B against reference oracle
//! binaries: harnesses over the reference listing's custom-mode API
//! at arbitrary `(rate, frame_size)` operating points, invoked as
//! opaque validator processes.
//!
//! Runtime-gated on `CELT_CM_ENC` / `CELT_CM_DEC` pointing at the
//! oracle encoder/decoder harness binaries
//! (`celt_cm_enc <fs> <ch> <frame_size> <cbr|vbr|cvbr> <arg>
//! <in.f32> <out.frames>`, `celt_cm_dec <fs> <ch> <frame_size>
//! <in.frames> <out.f32>`); passes with a note when unset.
//!
//! Configurations are restricted to short-MDCT sizes whose transform
//! plan the oracle's runtime supports (half-short-size 5-smooth);
//! this crate's direct-form transform also covers the remaining
//! legal geometries (e.g. `(44100, 880)`, short 110), which the
//! oracle binary rejects at mode creation and which are therefore
//! carried by the self-consistency arm in `tests/custom_modes.rs`.

use oxideav_celt::custom_mode::CeltCustomMode;
use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;
use std::path::{Path, PathBuf};
use std::process::Command;

fn oracle_bins() -> Option<(PathBuf, PathBuf)> {
    let enc = std::env::var_os("CELT_CM_ENC")?;
    let dec = std::env::var_os("CELT_CM_DEC")?;
    Some((PathBuf::from(enc), PathBuf::from(dec)))
}

/// `(fs, frame_size, cbr bytes/frame)` A/B points: 8–96 kHz, every
/// max_lm (32 kHz appears at both its 20 ms `max_lm = 3` and 5 ms
/// `max_lm = 1` operating points), the `eff_ebands < nb_ebands`
/// family (16 kHz / 40), and a non-400×-rate 48 kHz layout.
const CONFIGS: &[(u32, usize, usize)] = &[
    (8_000, 160, 80),
    (16_000, 320, 120),
    (16_000, 40, 40),
    (24_000, 480, 160),
    (32_000, 640, 200),
    (32_000, 160, 90),
    (44_100, 720, 160),
    (48_000, 1024, 240),
    (96_000, 960, 160),
];

fn test_signal(fs: u32, channels: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * channels);
    let mut lcg = 0x2545_F491u32;
    let mut noise = || -> f32 {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((lcg >> 16) as i32 - 32_768) as f32 / 65_536.0
    };
    let f0 = fs as f64 / 109.0;
    let burst = n / 3;
    let silence = (n / 2, n / 2 + n / 6);
    for t in 0..n {
        let tf = t as f64 / fs as f64;
        for c in 0..channels {
            let f = f0 * if c == 0 { 1.0 } else { 1.19 };
            let mut v = (0.28 * (2.0 * std::f64::consts::PI * f * tf).sin()
                + 0.18 * (2.0 * std::f64::consts::PI * 3.1 * f * tf).sin())
                as f32;
            if (burst..burst + fs as usize / 200).contains(&t) {
                v += 0.5 * noise();
            }
            if (silence.0..silence.1).contains(&t) {
                v = 0.0;
            }
            out.push(v.clamp(-0.999, 0.999));
        }
    }
    out
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

fn frames_of(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    let mut pos = 0usize;
    while pos + 2 <= stream.len() {
        let len = u16::from_le_bytes([stream[pos], stream[pos + 1]]) as usize;
        pos += 2;
        frames.push(stream[pos..pos + len].to_vec());
        pos += len;
    }
    frames
}

fn to_stream(frames: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(&(f.len() as u16).to_le_bytes());
        out.extend_from_slice(f);
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

fn run(cmd: &Path, args: &[&str]) {
    let out = Command::new(cmd).args(args).output().expect("oracle runs");
    assert!(
        out.status.success(),
        "{cmd:?} {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// CBR A/B in both directions at every configuration:
///
/// * oracle streams decode identically on this crate's decoder
///   (cross-decoder float SNR against the oracle's own decode);
/// * this crate's streams decode identically on the oracle decoder
///   (cross-decoder float SNR against this crate's own decode);
/// * quality parity: the two encoders' decoded outputs measure
///   within a few dB of each other against the input.
#[test]
fn custom_mode_oracle_cbr_ab() {
    let Some((enc_bin, dec_bin)) = oracle_bins() else {
        eprintln!("CELT_CM_ENC/CELT_CM_DEC not set; skipping custom-mode oracle A/B");
        return;
    };
    let dir = std::env::temp_dir().join("celt-custom-oracle-ab");
    std::fs::create_dir_all(&dir).expect("tmpdir");

    for &(fs, frame_size, bytes) in CONFIGS {
        for channels in 1..=2usize {
            let mode = CeltCustomMode::new(fs, frame_size).expect("mode");
            let frames = 30usize;
            let pcm = test_signal(fs, channels, frame_size * frames);
            let tag = format!("{fs}-{frame_size}-{channels}");
            let in_f32 = dir.join(format!("in-{tag}.f32"));
            write_f32(&in_f32, &pcm);

            // ── Direction 1: oracle encode → both decoders ──
            let ofr = dir.join(format!("o-{tag}.frames"));
            let odec = dir.join(format!("o-{tag}.f32"));
            run(
                &enc_bin,
                &[
                    &fs.to_string(),
                    &channels.to_string(),
                    &frame_size.to_string(),
                    "cbr",
                    &bytes.to_string(),
                    in_f32.to_str().unwrap(),
                    ofr.to_str().unwrap(),
                ],
            );
            run(
                &dec_bin,
                &[
                    &fs.to_string(),
                    &channels.to_string(),
                    &frame_size.to_string(),
                    ofr.to_str().unwrap(),
                    odec.to_str().unwrap(),
                ],
            );
            let oracle_dec = read_f32(&odec);
            let mut ours = Vec::new();
            let mut dec = CeltRefDecoder::new_custom(&mode, mode.max_lm, channels).expect("dec");
            for f in frames_of(&std::fs::read(&ofr).expect("frames")) {
                ours.extend(dec.decode_frame(&f).expect("decode oracle frame"));
            }
            let d1 = snr(&oracle_dec, &ours);
            assert!(
                d1 > 60.0,
                "({fs}, {frame_size}) x{channels}: decode of oracle stream diverges ({d1:.1} dB)"
            );

            // ── Direction 2: our encode → both decoders ──
            let mut enc = CeltRefEncoder::new_custom(&mode, mode.max_lm, channels).expect("enc");
            let mut our_frames = Vec::new();
            let mut our_dec = Vec::new();
            let mut dec2 = CeltRefDecoder::new_custom(&mode, mode.max_lm, channels).expect("dec");
            for f in 0..frames {
                let chunk = &pcm[f * frame_size * channels..(f + 1) * frame_size * channels];
                let coded = enc.encode_frame(chunk, bytes).expect("encode");
                our_dec.extend(dec2.decode_frame(&coded).expect("self decode"));
                our_frames.push(coded);
            }
            let sfr = dir.join(format!("s-{tag}.frames"));
            let sdec = dir.join(format!("s-{tag}.f32"));
            std::fs::write(&sfr, to_stream(&our_frames)).expect("write frames");
            run(
                &dec_bin,
                &[
                    &fs.to_string(),
                    &channels.to_string(),
                    &frame_size.to_string(),
                    sfr.to_str().unwrap(),
                    sdec.to_str().unwrap(),
                ],
            );
            let oracle_dec_ours = read_f32(&sdec);
            let d2 = snr(&our_dec, &oracle_dec_ours);
            assert!(
                d2 > 60.0,
                "({fs}, {frame_size}) x{channels}: oracle decode of our stream diverges ({d2:.1} dB)"
            );

            // ── Quality parity on the coded (non-silent) span ──
            let delay = mode.overlap * channels;
            let skip = 2 * frame_size * channels;
            let q_oracle = snr(
                &pcm[skip..oracle_dec.len() - delay],
                &oracle_dec[skip + delay..],
            );
            let q_ours = snr(&pcm[skip..our_dec.len() - delay], &our_dec[skip + delay..]);
            eprintln!(
                "({fs}, {frame_size}) x{channels} {bytes} B: cross-dec {d1:.1}/{d2:.1} dB, \
                 quality oracle {q_oracle:.1} dB vs ours {q_ours:.1} dB"
            );
            assert!(
                q_ours > q_oracle - 6.0,
                "({fs}, {frame_size}) x{channels}: quality gap ({q_ours:.1} vs {q_oracle:.1} dB)"
            );
        }
    }
}

/// VBR A/B at a custom rate: both encoders at the same bit/s target
/// on the same 44.1 kHz signal, unconstrained **and** constrained —
/// mean rates within 20% and identical 2-byte digital-silence frame
/// positions in both modes.
#[test]
fn custom_mode_oracle_vbr_ab() {
    let Some((enc_bin, dec_bin)) = oracle_bins() else {
        eprintln!("CELT_CM_ENC/CELT_CM_DEC not set; skipping custom-mode VBR A/B");
        return;
    };
    let dir = std::env::temp_dir().join("celt-custom-oracle-ab");
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let (fs, frame_size, target) = (44_100u32, 720usize, 64_000u32);
    let mode = CeltCustomMode::new(fs, frame_size).expect("mode");
    let frames = 60usize;
    let pcm = test_signal(fs, 1, frame_size * frames);
    let in_f32 = dir.join("vbr-in.f32");
    write_f32(&in_f32, &pcm);

    for (vbr_mode, constrained) in [("vbr", false), ("cvbr", true)] {
        let ofr = dir.join(format!("{vbr_mode}-o.frames"));
        let odec = dir.join(format!("{vbr_mode}-o.f32"));
        run(
            &enc_bin,
            &[
                &fs.to_string(),
                "1",
                &frame_size.to_string(),
                vbr_mode,
                &target.to_string(),
                in_f32.to_str().unwrap(),
                ofr.to_str().unwrap(),
            ],
        );
        run(
            &dec_bin,
            &[
                &fs.to_string(),
                "1",
                &frame_size.to_string(),
                ofr.to_str().unwrap(),
                odec.to_str().unwrap(),
            ],
        );
        let oracle_frames = frames_of(&std::fs::read(&ofr).expect("frames"));

        let mut enc = CeltRefEncoder::new_custom(&mode, mode.max_lm, 1).expect("enc");
        let mut our_sizes = Vec::new();
        for f in 0..frames {
            let coded = enc
                .encode_frame_vbr(
                    &pcm[f * frame_size..(f + 1) * frame_size],
                    1275,
                    target,
                    constrained,
                )
                .expect("encode");
            our_sizes.push(coded.len());
        }
        let o_total: usize = oracle_frames.iter().map(Vec::len).sum();
        let s_total: usize = our_sizes.iter().sum();
        let o_sil: Vec<usize> = oracle_frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.len() <= 2)
            .map(|(i, _)| i)
            .collect();
        let s_sil: Vec<usize> = our_sizes
            .iter()
            .enumerate()
            .filter(|(_, &l)| l <= 2)
            .map(|(i, _)| i)
            .collect();
        eprintln!(
            "44.1 kHz {vbr_mode} at {target} b/s: oracle {o_total} B / ours {s_total} B, \
             silence {o_sil:?} vs {s_sil:?}"
        );
        assert_eq!(o_sil, s_sil, "{vbr_mode}: 2-byte silence positions differ");
        let ratio = s_total as f64 / o_total as f64;
        assert!(
            (0.8..1.25).contains(&ratio),
            "{vbr_mode}: mean-rate ratio {ratio:.3} out of range"
        );
    }
}
