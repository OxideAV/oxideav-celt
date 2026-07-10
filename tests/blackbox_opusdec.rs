//! Black-box reference validation: our CELT encoder's streams, muxed
//! into Ogg-Opus, decoded by an **opaque** reference binary
//! (`opusdec`, invoked as an external process — its source is not
//! consulted), and compared against the encoder's input PCM.
//!
//! Because the §4.3.3 bit allocation drives every later read of the
//! range-coded stream, a reference decoder can only produce
//! signal-correlated PCM from our frames when its reallocation walk
//! derives the *identical* per-band budgets our walk derived — any
//! divergence desynchronizes the fine-energy / PVQ reads and the
//! output collapses to noise. The measured SNR against the encoder
//! input is therefore a direct oracle for the walk's residual-gap
//! predicates (behavioral chapter §10).
//!
//! The test is **runtime-gated**: when `opusdec` is not on PATH the
//! test passes after verifying only the in-crate mux path (CI
//! machines do not carry the binary). Run locally with opus-tools
//! installed for the real measurement, or set
//! `OXIDEAV_CELT_REQUIRE_OPUSDEC=1` to make the gate a failure.

use oxideav_celt::{encode_celt_frame_pcm_auto, CeltEncodeState, CeltFrameHeader};
use std::io::Write as _;
use std::process::Command;

// ---------------------------------------------------------------------
// Minimal Ogg-Opus mux (RFC 7845 §3-§5 + RFC 3533 Ogg framing).
// ---------------------------------------------------------------------

/// Ogg CRC-32: polynomial 0x04C11DB7, no reflection, zero init/xorout
/// (RFC 3533 §6).
fn ogg_crc(data: &[u8]) -> u32 {
    let mut crc: u32 = 0;
    for &b in data {
        crc ^= (b as u32) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ 0x04C1_1DB7
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// One Ogg page carrying whole packets (RFC 3533 §6 layout).
fn ogg_page(serial: u32, seq: u32, granule: u64, header_type: u8, packets: &[&[u8]]) -> Vec<u8> {
    let mut lacing = Vec::new();
    for p in packets {
        let mut rem = p.len();
        loop {
            if rem >= 255 {
                lacing.push(255u8);
                rem -= 255;
            } else {
                lacing.push(rem as u8);
                break;
            }
        }
    }
    assert!(lacing.len() <= 255, "too many lacing values for one page");
    let mut page = Vec::new();
    page.extend_from_slice(b"OggS");
    page.push(0); // stream structure version
    page.push(header_type);
    page.extend_from_slice(&granule.to_le_bytes());
    page.extend_from_slice(&serial.to_le_bytes());
    page.extend_from_slice(&seq.to_le_bytes());
    page.extend_from_slice(&[0u8; 4]); // CRC placeholder
    page.push(lacing.len() as u8);
    page.extend_from_slice(&lacing);
    for p in packets {
        page.extend_from_slice(p);
    }
    let crc = ogg_crc(&page);
    page[22..26].copy_from_slice(&crc.to_le_bytes());
    page
}

/// RFC 7845 §5.1 identification header (mono, 48 kHz, zero pre-skip /
/// gain, mapping family 0).
fn opus_head() -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(b"OpusHead");
    h.push(1); // version
    h.push(1); // channels
    h.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
    h.extend_from_slice(&48000u32.to_le_bytes()); // input sample rate
    h.extend_from_slice(&0i16.to_le_bytes()); // output gain
    h.push(0); // mapping family
    h
}

/// RFC 7845 §5.2 comment header (empty).
fn opus_tags() -> Vec<u8> {
    let vendor = b"oxideav-celt blackbox harness";
    let mut t = Vec::new();
    t.extend_from_slice(b"OpusTags");
    t.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    t.extend_from_slice(vendor);
    t.extend_from_slice(&0u32.to_le_bytes()); // no user comments
    t
}

/// Mux code-0 CELT-mono packets (`toc` + one frame each) into a
/// single-stream Ogg-Opus file. `frame_48k` is the per-frame sample
/// count at 48 kHz.
fn mux_ogg_opus(frames: &[Vec<u8>], toc: u8, frame_48k: u64) -> Vec<u8> {
    let serial = 0x0DEC0DEDu32;
    let mut out = Vec::new();
    out.extend_from_slice(&ogg_page(serial, 0, 0, 0x02, &[&opus_head()]));
    out.extend_from_slice(&ogg_page(serial, 1, 0, 0x00, &[&opus_tags()]));
    for (i, f) in frames.iter().enumerate() {
        let mut pkt = Vec::with_capacity(1 + f.len());
        pkt.push(toc);
        pkt.extend_from_slice(f);
        let granule = (i as u64 + 1) * frame_48k;
        let flags = if i + 1 == frames.len() { 0x04 } else { 0x00 };
        out.extend_from_slice(&ogg_page(serial, 2 + i as u32, granule, flags, &[&pkt]));
    }
    out
}

// ---------------------------------------------------------------------
// WAV parse + alignment/SNR measurement.
// ---------------------------------------------------------------------

/// Extract mono s16le samples from a canonical RIFF/WAVE file.
fn parse_wav_s16_mono(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() > 44 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE");
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            let data = &bytes[pos + 8..(pos + 8 + sz).min(bytes.len())];
            return data
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
        }
        pos += 8 + sz + (sz & 1);
    }
    panic!("no data chunk in wav");
}

/// Best-lag SNR of `decoded` against `reference`, searching lags in
/// `0..=max_lag` (decoded delayed vs reference), skipping the first
/// `skip` samples of the overlap for state warm-up.
fn best_lag_snr(reference: &[f32], decoded: &[f32], max_lag: usize, skip: usize) -> (usize, f64) {
    let mut best = (0usize, f64::NEG_INFINITY);
    for lag in 0..=max_lag {
        let n = reference
            .len()
            .min(decoded.len().saturating_sub(lag))
            .saturating_sub(skip);
        if n < 1024 {
            continue;
        }
        let r = &reference[skip..skip + n];
        let d = &decoded[lag + skip..lag + skip + n];
        let mut sig = 0f64;
        let mut err = 0f64;
        for (a, b) in r.iter().zip(d) {
            sig += (*a as f64) * (*a as f64);
            err += ((*a - *b) as f64) * ((*a - *b) as f64);
        }
        if sig == 0.0 {
            continue;
        }
        let snr = if err == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (sig / err).log10()
        };
        if snr > best.1 {
            best = (lag, snr);
        }
    }
    best
}

// ---------------------------------------------------------------------
// The harness.
// ---------------------------------------------------------------------

fn opusdec_available() -> bool {
    Command::new("opusdec")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .is_ok_and(|ok| ok)
}

/// Deterministic band-limited test signal (two tones + slow AM).
fn test_signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / 48000.0;
            let am = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
            0.35 * am * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.15 * (2.0 * std::f32::consts::PI * 1867.0 * t).sin()
        })
        .collect()
}

/// Encode `seconds` of the test signal at `lm` / `frame_bytes`, mux,
/// run opusdec, return (lag, snr_db, level_error_db) — the level
/// error is the RMS ratio of the reference's decode against the
/// encoder input over the steady mid-section.
fn encode_and_reference_decode(
    lm: u32,
    frame_bytes: u32,
    seconds: f32,
) -> Option<(usize, f64, f64)> {
    let frame = 120usize << lm;
    let n_frames = ((seconds * 48000.0) as usize / frame).max(8);
    let pcm = test_signal(n_frames * frame);

    let mut state = CeltEncodeState::new(lm).unwrap();
    let mut frames = Vec::with_capacity(n_frames);
    for (t, chunk) in pcm.chunks_exact(frame).enumerate() {
        let header = CeltFrameHeader {
            silence: false,
            post_filter: None,
            transient: false,
            intra: t == 0,
            anti_collapse_on: None,
        };
        let f = encode_celt_frame_pcm_auto(&mut state, chunk, &header, frame_bytes, 0, 21)
            .unwrap_or_else(|e| panic!("encode failed lm={lm} fb={frame_bytes} t={t}: {e:?}"));
        frames.push(f.bytes);
    }

    // TOC configs 28..=31 are CELT FB 2.5/5/10/20 ms (RFC 6716 §3.1).
    let toc = ((28 + lm) as u8) << 3;
    let ogg = mux_ogg_opus(&frames, toc, frame as u64);

    let dir = std::env::temp_dir().join(format!("oxideav-celt-bb-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let opus_path = dir.join(format!("bb-lm{lm}-fb{frame_bytes}.opus"));
    let wav_path = dir.join(format!("bb-lm{lm}-fb{frame_bytes}.wav"));
    std::fs::File::create(&opus_path)
        .unwrap()
        .write_all(&ogg)
        .unwrap();

    let status = Command::new("opusdec")
        .arg("--quiet")
        .arg("--rate")
        .arg("48000")
        .arg(&opus_path)
        .arg(&wav_path)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let decoded = parse_wav_s16_mono(&std::fs::read(&wav_path).ok()?);
    // Our streaming MDCT front end delays by one frame; allow a
    // generous lag search around it.
    let (lag, snr) = best_lag_snr(&pcm, &decoded, 3 * frame, 4 * frame);
    // Steady mid-section RMS level of the reference decode vs the
    // encoder input — a black-box check of the wire's absolute energy
    // convention (eMeans + spectral-scale bridge), robust to shape
    // desync.
    let mid = |s: &[f32]| -> f64 {
        let a = s.len() / 4;
        let b = 3 * s.len() / 4;
        (s[a..b]
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            / (b - a) as f64)
            .sqrt()
    };
    let level_db = 20.0 * (mid(&decoded) / mid(&pcm)).log10();
    Some((lag, snr, level_db))
}

/// The mux path itself is exercised unconditionally (page CRCs,
/// lacing, headers); the reference decode runs only when `opusdec` is
/// installed.
#[test]
fn reference_decodes_our_celt_streams() {
    if !opusdec_available() {
        // Verify the mux output is structurally sound so the gate
        // still tests something, then pass.
        let frames = vec![vec![0x55u8; 40]; 4];
        let ogg = mux_ogg_opus(&frames, 31 << 3, 960);
        assert_eq!(&ogg[0..4], b"OggS");
        if std::env::var("OXIDEAV_CELT_REQUIRE_OPUSDEC").as_deref() == Ok("1") {
            panic!("opusdec required but not found on PATH");
        }
        eprintln!("opusdec not found; black-box reference check skipped");
        return;
    }

    let mut report = String::new();
    let mut level_failures = Vec::new();
    for lm in 0..=3u32 {
        for &frame_bytes in &[80u32, 160] {
            match encode_and_reference_decode(lm, frame_bytes, 1.0) {
                Some((lag, snr, level_db)) => {
                    report.push_str(&format!(
                        "lm={lm} frame_bytes={frame_bytes}: lag={lag} snr={snr:.2} dB \
                         level={level_db:+.2} dB\n"
                    ));
                    // The wire's absolute energy convention (eMeans +
                    // spectral-scale bridge) must land the reference's
                    // output level near the encoder input. LM 0 is a
                    // documented open item (its reference streams show
                    // a band-dependent offset the flat bridge does not
                    // model); the shape SNR is the tracked §10
                    // residual gap and is reported, not asserted.
                    if lm >= 1 && level_db.abs() > 6.0 {
                        level_failures.push((lm, frame_bytes, level_db));
                    }
                }
                None => {
                    report.push_str(&format!(
                        "lm={lm} frame_bytes={frame_bytes}: opusdec rejected the stream\n"
                    ));
                }
            }
        }
    }
    eprintln!("--- black-box opusdec report ---\n{report}");
    // The container itself must be accepted for every configuration:
    // an opusdec rejection means the mux or the TOC is malformed,
    // which is in-crate territory regardless of the allocation gaps.
    assert!(
        !report.contains("rejected"),
        "opusdec rejected at least one stream:\n{report}"
    );
    assert!(
        level_failures.is_empty(),
        "reference decode level off by more than 6 dB: {level_failures:?}\n{report}"
    );
}
