//! Reduced-rate I/O black-box A/B against reference oracle binaries:
//! harnesses over the reference listing's standard-mode API
//! initialized at 8/12/16/24 kHz PCM rates, invoked as opaque
//! validator processes.
//!
//! Runtime-gated on `CELT_DS_ENC` / `CELT_DS_DEC` pointing at the
//! oracle harness binaries
//! (`celt_ds_enc <ch> <frame48> <rate> <start> <cbr|vbr|cvbr> <arg>
//! <in.f32> <out.frames>`, `celt_ds_dec <ch> <frame48> <rate>
//! <start> <in.frames> <out.f32>`); passes with a note when unset.

use oxideav_celt::ref_decode::{resampling_factor, CeltRefDecoder};
use oxideav_celt::ref_encode::CeltRefEncoder;
use std::path::{Path, PathBuf};
use std::process::Command;

const RATES: [u32; 4] = [24_000, 16_000, 12_000, 8_000];

fn oracle_bins() -> Option<(PathBuf, PathBuf)> {
    let enc = std::env::var_os("CELT_DS_ENC")?;
    let dec = std::env::var_os("CELT_DS_DEC")?;
    Some((PathBuf::from(enc), PathBuf::from(dec)))
}

/// Band-limited (≤ 3.4 kHz) tonal + noise material at 48 kHz so
/// every output rate keeps the content in band.
fn test_signal_48k(n: usize, channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * channels);
    let mut lcg = 0x2545_F491u32;
    let mut noise = || -> f32 {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((lcg >> 16) as i32 - 32_768) as f32 / 65_536.0
    };
    for t in 0..n {
        let tf = t as f64 / 48_000.0;
        for c in 0..channels {
            let f0: f64 = if c == 0 { 440.0 } else { 554.37 };
            let mut v = (0.35 * (2.0 * std::f64::consts::PI * f0 * tf).sin()
                + 0.15 * (2.0 * std::f64::consts::PI * 3.1 * f0 * tf).sin())
                as f32;
            if (9_600..9_840).contains(&t) {
                v += 0.5 * noise();
            }
            if t >= 14_400 {
                v += 0.1 * noise();
            }
            out.push(v.clamp(-0.999, 0.999));
        }
    }
    out
}

/// Decimate an interleaved 48 kHz signal by `d` (keep every d-th
/// sample frame — the encoder-side input grid takes the LAST slot of
/// each group, so the input fed at a reduced rate is built with
/// `skip = d - 1`).
fn decimate(x: &[f32], channels: usize, d: usize, skip: usize) -> Vec<f32> {
    x.chunks_exact(channels)
        .skip(skip)
        .step_by(d)
        .flatten()
        .copied()
        .collect()
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

/// Decode lockstep at every reduced output rate: the oracle encodes
/// at 48 kHz, both decoders decode the same frames at each PCM rate,
/// and the outputs must sit at the decoder pair's float-noise floor.
#[test]
fn downsampled_decode_lockstep() {
    let Some((enc_bin, dec_bin)) = oracle_bins() else {
        eprintln!("CELT_DS_ENC/CELT_DS_DEC not set; skipping downsample oracle A/B");
        return;
    };
    let dir = std::env::temp_dir().join("celt-ds-oracle-ab");
    std::fs::create_dir_all(&dir).expect("mkdir");

    for &(lm, channels, bytes) in &[(3u32, 1usize, 160usize), (2, 2, 180)] {
        let frame = 120usize << lm;
        let n = 24 * frame;
        let pcm48 = test_signal_48k(n, channels);
        let in_f = dir.join(format!("in48-{lm}-{channels}.f32"));
        write_f32(&in_f, &pcm48);
        let frames_f = dir.join(format!("st-{lm}-{channels}.frames"));
        run(
            &enc_bin,
            &[
                &channels.to_string(),
                &frame.to_string(),
                "48000",
                "0",
                "cbr",
                &bytes.to_string(),
                in_f.to_str().unwrap(),
                frames_f.to_str().unwrap(),
            ],
        );
        let stream = std::fs::read(&frames_f).expect("frames");
        let frames = frames_of(&stream);

        for &rate in &RATES {
            // Oracle decode at the reduced rate.
            let ora_f = dir.join(format!("ora-{lm}-{channels}-{rate}.f32"));
            run(
                &dec_bin,
                &[
                    &channels.to_string(),
                    &frame.to_string(),
                    &rate.to_string(),
                    "0",
                    frames_f.to_str().unwrap(),
                    ora_f.to_str().unwrap(),
                ],
            );
            let oracle = read_f32(&ora_f);
            // Our decode at the same rate.
            let mut dec = CeltRefDecoder::new_downsampled(lm, channels, rate).expect("decoder");
            let mut ours = Vec::new();
            for f in &frames {
                ours.extend(dec.decode_frame(f).expect("decode"));
            }
            assert_eq!(ours.len(), oracle.len());
            let s = snr(&oracle, &ours);
            eprintln!("decode lockstep lm={lm} ch={channels} rate={rate}: {s:.1} dB");
            assert!(
                s > 90.0,
                "downsampled decode diverged at lm={lm} ch={channels} rate={rate}: {s:.1} dB"
            );
        }
    }
}

/// Encode-side A/B at every reduced input rate: both encoders code
/// the same reduced-rate input; each stream is decoded by the oracle
/// decoder at the same rate. Our stream must be symbol-clean through
/// the oracle decoder (cross-decode at the decoder pair's noise
/// floor vs our own decode) and its decoded quality must be within
/// the parity band of the oracle encoder's.
#[test]
fn upsampled_encode_parity() {
    let Some((enc_bin, dec_bin)) = oracle_bins() else {
        eprintln!("CELT_DS_ENC/CELT_DS_DEC not set; skipping downsample oracle A/B");
        return;
    };
    let dir = std::env::temp_dir().join("celt-ds-oracle-ab");
    std::fs::create_dir_all(&dir).expect("mkdir");

    let lm = 3u32;
    let frame = 120usize << lm;
    let channels = 1usize;
    let n = 24 * frame;
    let pcm48 = test_signal_48k(n, channels);

    for &rate in &RATES {
        let d = resampling_factor(rate).unwrap();
        // The reduced-rate input: the grid slot the zero-stuffing
        // fills is the last of each group.
        let pcm_in = decimate(&pcm48, channels, d, d - 1);
        let in_f = dir.join(format!("in-{rate}.f32"));
        write_f32(&in_f, &pcm_in);

        // Oracle encode at the reduced rate.
        let ora_st = dir.join(format!("ora-enc-{rate}.frames"));
        run(
            &enc_bin,
            &[
                "1",
                &frame.to_string(),
                &rate.to_string(),
                "0",
                "cbr",
                "160",
                in_f.to_str().unwrap(),
                ora_st.to_str().unwrap(),
            ],
        );
        // Our encode at the reduced rate.
        let mut enc = CeltRefEncoder::new_upsampled(lm, 1, rate).expect("encoder");
        let in_frame = enc.input_frame_size();
        let our_frames: Vec<Vec<u8>> = pcm_in
            .chunks_exact(in_frame)
            .map(|c| enc.encode_frame(c, 160).expect("encode"))
            .collect();
        let our_st = dir.join(format!("our-enc-{rate}.frames"));
        std::fs::write(&our_st, to_stream(&our_frames)).expect("write stream");

        // Oracle decodes both streams at the same rate.
        let dec_of = |st: &Path, tag: &str| -> Vec<f32> {
            let out = dir.join(format!("dec-{tag}-{rate}.f32"));
            run(
                &dec_bin,
                &[
                    "1",
                    &frame.to_string(),
                    &rate.to_string(),
                    "0",
                    st.to_str().unwrap(),
                    out.to_str().unwrap(),
                ],
            );
            read_f32(&out)
        };
        let ora_dec = dec_of(&ora_st, "ora");
        let our_cross = dec_of(&our_st, "our");

        // Our own decode of our stream: the cross-decoder must agree
        // at the float-noise floor (symbol-exact interop).
        let mut dec = CeltRefDecoder::new_downsampled(lm, 1, rate).expect("decoder");
        let mut our_dec = Vec::new();
        for f in frames_of(&std::fs::read(&our_st).unwrap()) {
            our_dec.extend(dec.decode_frame(&f).expect("decode"));
        }
        let lockstep = snr(&our_dec, &our_cross);

        // Quality parity vs the input (delay-compensated).
        let delay = 120 / d;
        let steady = 4 * in_frame;
        let q = |decoded: &[f32]| -> f64 {
            snr(
                &pcm_in[steady..pcm_in.len() - delay],
                &decoded[steady + delay..],
            )
        };
        let q_ora = q(&ora_dec);
        let q_our = q(&our_cross);
        eprintln!(
            "encode parity rate={rate}: ours {q_our:.1} dB vs oracle {q_ora:.1} dB, \
             cross-decode lockstep {lockstep:.1} dB"
        );
        assert!(
            lockstep > 90.0,
            "our reduced-rate stream is not symbol-clean at {rate}: {lockstep:.1} dB"
        );
        assert!(
            q_our > q_ora - 1.5,
            "encode quality fell behind the oracle at {rate}: {q_our:.1} vs {q_ora:.1} dB"
        );
    }
}

/// Hybrid-layer (start = 17) streams decode at reduced output rates
/// in lockstep with the oracle (the Opus decoder's hybrid at
/// 8-24 kHz output).
#[test]
fn hybrid_downsampled_decode_lockstep() {
    let Some((enc_bin, dec_bin)) = oracle_bins() else {
        eprintln!("CELT_DS_ENC/CELT_DS_DEC not set; skipping downsample oracle A/B");
        return;
    };
    let dir = std::env::temp_dir().join("celt-ds-oracle-ab");
    std::fs::create_dir_all(&dir).expect("mkdir");

    let lm = 2u32;
    let frame = 120usize << lm;
    let n = 30 * frame;
    // High-band + low-band mix; the hybrid layer codes ≥ 8 kHz only.
    let pcm48: Vec<f32> = (0..n)
        .map(|t| {
            let tf = t as f64 / 48_000.0;
            (0.3 * (2.0 * std::f64::consts::PI * 9_500.0 * tf).sin()
                + 0.2 * (2.0 * std::f64::consts::PI * 400.0 * tf).sin()) as f32
        })
        .collect();
    let in_f = dir.join("hyb-in48.f32");
    write_f32(&in_f, &pcm48);
    let frames_f = dir.join("hyb-st.frames");
    run(
        &enc_bin,
        &[
            "1",
            &frame.to_string(),
            "48000",
            "17",
            "cbr",
            "80",
            in_f.to_str().unwrap(),
            frames_f.to_str().unwrap(),
        ],
    );
    let frames = frames_of(&std::fs::read(&frames_f).unwrap());

    for &rate in &RATES {
        let ora_f = dir.join(format!("hyb-ora-{rate}.f32"));
        run(
            &dec_bin,
            &[
                "1",
                &frame.to_string(),
                &rate.to_string(),
                "17",
                frames_f.to_str().unwrap(),
                ora_f.to_str().unwrap(),
            ],
        );
        let oracle = read_f32(&ora_f);
        let mut dec = CeltRefDecoder::new_with_start_downsampled(lm, 1, 17, rate).expect("decoder");
        let mut ours = Vec::new();
        for f in &frames {
            ours.extend(dec.decode_frame(f).expect("decode"));
        }
        assert_eq!(ours.len(), oracle.len());
        // At 16 kHz and below the hybrid layer's coded band sits
        // entirely above the output Nyquist: both outputs are the
        // post-filter/de-emphasis tail of (near-)zero spectra, so
        // gate on absolute agreement rather than SNR of a silent
        // signal.
        let max_diff = ours
            .iter()
            .zip(oracle.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let s = snr(&oracle, &ours);
        eprintln!("hybrid decode lockstep rate={rate}: {s:.1} dB, max diff {max_diff:.3e}");
        assert!(
            s > 90.0 || max_diff < 1e-6,
            "hybrid downsampled decode diverged at {rate}: {s:.1} dB, max diff {max_diff:.3e}"
        );
    }
}
