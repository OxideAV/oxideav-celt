//! Hybrid-layer (start = 17) black-box A/B against reference oracle
//! binaries: harnesses over the reference listing's custom-mode API
//! with the start band set to 17 (the Hybrid CELT layer), invoked as
//! opaque validator processes.
//!
//! Runtime-gated on `CELT_HYB_ENC` / `CELT_HYB_DEC` pointing at the
//! oracle encoder/decoder harness binaries
//! (`celt_hyb_enc <ch> <frame_size> <start> <cbr|vbr|cvbr> <arg>
//! <in.f32> <out.frames>`, `celt_hyb_dec <ch> <frame_size> <start>
//! <in.frames> <out.f32>`); passes with a note when unset.

use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;
use std::path::{Path, PathBuf};
use std::process::Command;

const START: usize = 17;

fn oracle_bins() -> Option<(PathBuf, PathBuf)> {
    let enc = std::env::var_os("CELT_HYB_ENC")?;
    let dec = std::env::var_os("CELT_HYB_DEC")?;
    Some((PathBuf::from(enc), PathBuf::from(dec)))
}

/// High-band test material for the hybrid layer: a 9.5/12.5 kHz tone
/// mix, a hard noise burst, a digital-silence stretch, and a noise
/// tail (the coded 8-20 kHz territory carries all of it).
fn test_signal(channels: usize) -> Vec<f32> {
    let n = 28_800usize;
    let mut out = Vec::with_capacity(n * channels);
    let mut lcg = 0x2545_F491u32;
    let noise = |lcg: &mut u32| -> f32 {
        *lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((*lcg >> 16) as i32 - 32_768) as f32 / 65_536.0
    };
    for t in 0..n {
        let tf = t as f64 / 48_000.0;
        for c in 0..channels {
            let f0: f64 = if c == 0 { 9_500.0 } else { 12_500.0 };
            let mut v = (0.3 * (2.0 * std::f64::consts::PI * f0 * tf).sin()
                + 0.15 * (2.0 * std::f64::consts::PI * 1.31 * f0 * tf).sin())
                as f32;
            if (9_600..9_840).contains(&t) {
                v += noise(&mut lcg);
            }
            if (14_400..19_200).contains(&t) {
                v = 0.0;
            }
            if t >= 24_000 {
                v = 0.5 * noise(&mut lcg);
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

#[test]
fn hybrid_layer_oracle_ab() {
    let Some((enc_bin, dec_bin)) = oracle_bins() else {
        eprintln!("CELT_HYB_ENC/CELT_HYB_DEC not set; skipping hybrid oracle A/B");
        return;
    };
    let dir = std::env::temp_dir().join("celt-hybrid-oracle-ab");
    std::fs::create_dir_all(&dir).expect("tmp dir");

    for &(lm, channels, cbr_bytes) in &[(1u32, 1usize, 45usize), (2, 1, 90), (3, 2, 220)] {
        let frame = 120usize << lm;
        let fs = frame.to_string();
        let ch = channels.to_string();
        let input = test_signal(channels);
        let in_f32 = dir.join(format!("in-{lm}-{channels}.f32"));
        write_f32(&in_f32, &input);

        // ── Oracle encodes (CBR), both decoders decode ──
        let oracle_frames = dir.join(format!("oracle-{lm}-{channels}.frames"));
        let oracle_dec = dir.join(format!("oracle-{lm}-{channels}.f32"));
        run(
            &enc_bin,
            &[
                &ch,
                &fs,
                "17",
                "cbr",
                &cbr_bytes.to_string(),
                in_f32.to_str().unwrap(),
                oracle_frames.to_str().unwrap(),
            ],
        );
        run(
            &dec_bin,
            &[
                &ch,
                &fs,
                "17",
                oracle_frames.to_str().unwrap(),
                oracle_dec.to_str().unwrap(),
            ],
        );
        let mut dec = CeltRefDecoder::new_with_start(lm, channels, START).expect("dec");
        let mut ours = Vec::new();
        for f in frames_of(&std::fs::read(&oracle_frames).expect("frames")) {
            ours.extend(dec.decode_frame(&f).expect("decode oracle hybrid frame"));
        }
        let expected = read_f32(&oracle_dec);
        let dec_snr = snr(&expected, &ours);
        eprintln!(
            "hybrid lm={lm} ch={channels} {cbr_bytes}B: our decode of oracle stream {dec_snr:.1} dB"
        );
        assert!(
            dec_snr >= 95.0,
            "lm={lm} ch={channels}: hybrid decode SNR {dec_snr:.1} dB below the 95 dB floor"
        );

        // ── We encode (CBR), both decoders decode ──
        let mut enc = CeltRefEncoder::new_with_start(lm, channels, START).expect("enc");
        let mut our_frames: Vec<Vec<u8>> = Vec::new();
        for chunk in input.chunks_exact(frame * channels) {
            our_frames.push(enc.encode_frame(chunk, cbr_bytes).expect("encode"));
        }
        let ours_stream = dir.join(format!("ours-{lm}-{channels}.frames"));
        std::fs::write(&ours_stream, to_stream(&our_frames)).expect("write stream");
        let ours_oracle_dec = dir.join(format!("ours-oracle-{lm}-{channels}.f32"));
        run(
            &dec_bin,
            &[
                &ch,
                &fs,
                "17",
                ours_stream.to_str().unwrap(),
                ours_oracle_dec.to_str().unwrap(),
            ],
        );
        let mut dec2 = CeltRefDecoder::new_with_start(lm, channels, START).expect("dec");
        let mut ours_own = Vec::new();
        for f in &our_frames {
            ours_own.extend(dec2.decode_frame(f).expect("decode own"));
        }
        let theirs = read_f32(&ours_oracle_dec);
        let cross = snr(&theirs, &ours_own);
        // Quality of both encoders' streams (oracle-decoded, vs the
        // 120-sample-delayed input).
        let delay = 120 * channels;
        let q_ours = snr(&input[..input.len() - delay], &theirs[delay..]);
        let q_oracle = snr(&input[..input.len() - delay], &expected[delay..]);
        eprintln!(
            "hybrid lm={lm} ch={channels} {cbr_bytes}B: cross-decoder {cross:.1} dB, \
             quality ours {q_ours:.1} vs oracle {q_oracle:.1} dB"
        );
        assert!(
            cross >= 95.0,
            "lm={lm} ch={channels}: cross-decoder SNR {cross:.1} dB — wire divergence"
        );
        assert!(
            q_ours >= q_oracle - 3.0,
            "lm={lm} ch={channels}: our hybrid encode {q_ours:.1} dB vs oracle {q_oracle:.1} dB"
        );
    }
}

#[test]
fn hybrid_layer_oracle_vbr_ab() {
    let Some((enc_bin, dec_bin)) = oracle_bins() else {
        eprintln!("CELT_HYB_ENC/CELT_HYB_DEC not set; skipping hybrid VBR oracle A/B");
        return;
    };
    let dir = std::env::temp_dir().join("celt-hybrid-oracle-ab");
    std::fs::create_dir_all(&dir).expect("tmp dir");

    for &(lm, channels) in &[(2u32, 1usize), (3, 2)] {
        let frame = 120usize << lm;
        let fs = frame.to_string();
        let ch = channels.to_string();
        let input = test_signal(channels);
        let in_f32 = dir.join(format!("vin-{lm}-{channels}.f32"));
        write_f32(&in_f32, &input);

        let oracle_frames = dir.join(format!("voracle-{lm}-{channels}.frames"));
        run(
            &enc_bin,
            &[
                &ch,
                &fs,
                "17",
                "vbr",
                "48000",
                in_f32.to_str().unwrap(),
                oracle_frames.to_str().unwrap(),
            ],
        );
        let oracle_sizes: Vec<usize> = frames_of(&std::fs::read(&oracle_frames).expect("frames"))
            .iter()
            .map(|f| f.len())
            .collect();

        let mut enc = CeltRefEncoder::new_with_start(lm, channels, START).expect("enc");
        let mut sizes = Vec::new();
        let mut our_frames = Vec::new();
        for chunk in input.chunks_exact(frame * channels) {
            let data = enc
                .encode_frame_vbr(chunk, 1275, 48_000, false)
                .expect("vbr encode");
            sizes.push(data.len());
            our_frames.push(data);
        }
        // The oracle-decoded quality of our hybrid VBR stream stays
        // finite and the decode succeeds end to end.
        let ours_stream = dir.join(format!("vours-{lm}-{channels}.frames"));
        std::fs::write(&ours_stream, to_stream(&our_frames)).expect("write stream");
        let ours_dec = dir.join(format!("vours-{lm}-{channels}.f32"));
        run(
            &dec_bin,
            &[
                &ch,
                &fs,
                "17",
                ours_stream.to_str().unwrap(),
                ours_dec.to_str().unwrap(),
            ],
        );
        let our_silence: Vec<usize> = sizes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s <= 2)
            .map(|(i, _)| i)
            .collect();
        let oracle_silence: Vec<usize> = oracle_sizes
            .iter()
            .enumerate()
            .filter(|(_, &s)| s <= 2)
            .map(|(i, _)| i)
            .collect();
        let mean = |s: &[usize]| {
            s.iter().sum::<usize>() as f64 * 8.0 * 48_000.0 / (s.len() * frame) as f64
        };
        eprintln!(
            "hybrid vbr lm={lm} ch={channels}: mean {:.0} vs oracle {:.0} b/s, \
             silence {:?} vs {:?}",
            mean(&sizes),
            mean(&oracle_sizes),
            our_silence,
            oracle_silence
        );
        assert_eq!(
            our_silence, oracle_silence,
            "lm={lm}: hybrid VBR silence positions"
        );
        let ratio = mean(&sizes) / mean(&oracle_sizes);
        assert!(
            (0.7..=1.3).contains(&ratio),
            "lm={lm}: hybrid VBR mean rate ratio {ratio:.3}"
        );
    }
}
