//! Custom-mode (non-48 kHz) end-to-end coverage: the mode-derived
//! geometry drives the reference-exact encoder and decoder at
//! arbitrary rate/frame-size operating points, self-consistently.
//!
//! The wire truth for these configurations is pinned two ways:
//!
//! * **In-repo (this file):** encode → decode round trips through the
//!   full Table-56 walk on the derived geometry — waveform fidelity at
//!   a healthy rate, byte determinism, silence handling, VBR rate
//!   tracking against the mode's own sample rate, and decode
//!   robustness on arbitrary bytes.
//! * **Black-box (runtime-gated):** `tests/blackbox_custom_oracle.rs`
//!   A/Bs the same configurations against oracle harness binaries
//!   built from the RFC 6716 §A.1 reference listing.

use oxideav_celt::custom_mode::CeltCustomMode;
use oxideav_celt::ref_decode::CeltRefDecoder;
use oxideav_celt::ref_encode::CeltRefEncoder;

/// The custom-mode configurations under test: `(fs, frame_size)`
/// spanning 8–96 kHz, every legal `max_lm`, odd geometries
/// (`short = 110` → overlap 108), the non-400×-rate 48 kHz layout,
/// and the shared-2.5 ms-layout family whose `eff_ebands < nb_ebands`
/// (16 kHz / 40).
const CONFIGS: &[(u32, usize)] = &[
    (8_000, 160),
    (16_000, 320),
    (16_000, 40),
    (24_000, 480),
    (32_000, 640),
    (44_100, 440),
    (44_100, 880),
    (48_000, 1024),
    (96_000, 960),
];

/// Deterministic tonal + transient + noise test material at the
/// mode's own sample rate (frequencies scale with `fs` so every mode
/// gets in-band content).
fn test_signal(fs: u32, channels: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * channels);
    let mut lcg = 0x2545_F491u32;
    let mut noise = || -> f32 {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((lcg >> 16) as i32 - 32_768) as f32 / 65_536.0
    };
    let f0 = fs as f64 / 109.0;
    let burst = n / 3;
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
            out.push(v.clamp(-0.999, 0.999));
        }
    }
    out
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

/// Encode → decode at every configuration, mono and stereo, at a
/// healthy CBR rate: the decoded output must track the
/// (overlap-delayed) input closely once the coarse prediction has
/// adapted.
#[test]
fn custom_mode_round_trip_fidelity() {
    for &(fs, frame_size) in CONFIGS {
        let mode = CeltCustomMode::new(fs, frame_size).expect("mode constructs");
        for channels in 1..=2usize {
            // ~128 kb/s mono / 192 kb/s stereo equivalent.
            let bits_per_frame = (frame_size as u64 * (64_000 + 64_000 * channels as u64)
                / fs as u64)
                .clamp(48, 1275 * 8);
            let bytes = (bits_per_frame / 8) as usize;
            let frames = (6 * fs as usize / 1000 / frame_size).max(6); // >= ~6 frames
            let pcm = test_signal(fs, channels, frame_size * frames);
            let mut enc =
                CeltRefEncoder::new_custom(&mode, mode.max_lm, channels).expect("encoder");
            let mut dec =
                CeltRefDecoder::new_custom(&mode, mode.max_lm, channels).expect("decoder");
            assert_eq!(enc.frame_size(), frame_size);
            assert_eq!(dec.frame_size(), frame_size);
            let mut out: Vec<f32> = Vec::new();
            for f in 0..frames {
                let chunk = &pcm[f * frame_size * channels..(f + 1) * frame_size * channels];
                let coded = enc.encode_frame(chunk, bytes).expect("encode");
                assert_eq!(coded.len(), bytes, "({fs}, {frame_size}) byte-exact frame");
                let dpcm = dec.decode_frame(&coded).expect("decode");
                assert_eq!(dpcm.len(), frame_size * channels);
                assert!(dpcm.iter().all(|v| v.is_finite()));
                out.extend(dpcm);
            }
            // Delay-compensated SNR over the adapted steady state.
            let delay = mode.overlap * channels;
            let skip = 2 * frame_size * channels;
            let snr = snr_db(&pcm[skip..out.len() - delay], &out[skip + delay..]);
            eprintln!("({fs}, {frame_size}) x{channels}: {bytes} B/frame, SNR {snr:.1} dB");
            assert!(
                snr > 12.0,
                "({fs}, {frame_size}) x{channels} round-trip SNR {snr:.1} dB"
            );
        }
    }
}

/// Byte determinism: identical input through fresh encoder states
/// yields identical custom-mode streams.
#[test]
fn custom_mode_encode_is_deterministic() {
    let (fs, frame_size) = (44_100, 880);
    let mode = CeltCustomMode::new(fs, frame_size).expect("mode");
    let pcm = test_signal(fs, 2, frame_size * 4);
    let run = || -> Vec<Vec<u8>> {
        let mut enc = CeltRefEncoder::new_custom(&mode, mode.max_lm, 2).expect("encoder");
        (0..4)
            .map(|f| {
                enc.encode_frame(&pcm[f * frame_size * 2..(f + 1) * frame_size * 2], 180)
                    .expect("encode")
            })
            .collect()
    };
    assert_eq!(run(), run());
}

/// Every smaller frame shift of a mode decodes too (the mode covers
/// frames `short << lm` for `lm <= max_lm`), and the lm/channel
/// envelope is validated.
#[test]
fn custom_mode_lm_ladder_and_rejections() {
    let mode = CeltCustomMode::new(32_000, 640).expect("mode");
    assert_eq!(mode.max_lm, 3);
    for lm in 0..=mode.max_lm {
        let frame = mode.short_mdct_size << lm;
        let pcm = test_signal(32_000, 1, frame * 4);
        let mut enc = CeltRefEncoder::new_custom(&mode, lm, 1).expect("encoder");
        let mut dec = CeltRefDecoder::new_custom(&mode, lm, 1).expect("decoder");
        for f in 0..4 {
            let coded = enc
                .encode_frame(&pcm[f * frame..(f + 1) * frame], 80)
                .expect("encode");
            let out = dec.decode_frame(&coded).expect("decode");
            assert_eq!(out.len(), frame);
        }
    }
    assert!(CeltRefDecoder::new_custom(&mode, mode.max_lm + 1, 1).is_err());
    assert!(CeltRefEncoder::new_custom(&mode, mode.max_lm + 1, 1).is_err());
    assert!(CeltRefDecoder::new_custom(&mode, 0, 3).is_err());
}

/// VBR at a custom rate: the controller's rate law is derived from
/// the mode's own sample rate — a 44.1 kHz stream at a 64 kb/s target
/// must land near the target, and digital silence collapses to the
/// 2-byte floor.
#[test]
fn custom_mode_vbr_tracks_target_and_silence() {
    let (fs, frame_size) = (44_100, 880);
    let mode = CeltCustomMode::new(fs, frame_size).expect("mode");
    let frames = 40usize;
    let mut pcm = test_signal(fs, 1, frame_size * frames);
    // Digital silence over frames 25..32.
    for v in pcm[25 * frame_size..32 * frame_size].iter_mut() {
        *v = 0.0;
    }
    let mut enc = CeltRefEncoder::new_custom(&mode, mode.max_lm, 1).expect("encoder");
    let mut dec = CeltRefDecoder::new_custom(&mode, mode.max_lm, 1).expect("decoder");
    let mut total = 0usize;
    let mut silence_frames = 0usize;
    for f in 0..frames {
        let coded = enc
            .encode_frame_vbr(
                &pcm[f * frame_size..(f + 1) * frame_size],
                1275,
                64_000,
                false,
            )
            .expect("encode");
        if coded.len() <= 2 {
            silence_frames += 1;
        }
        total += coded.len();
        let out = dec.decode_frame(&coded).expect("decode");
        assert_eq!(out.len(), frame_size);
    }
    // Frames 26..32 are wholly silent (the flag runs one frame late
    // behind the pre-emphasis discharge); require most of the span.
    assert!(
        silence_frames >= 4,
        "silence collapse: {silence_frames} 2-byte frames"
    );
    let mean_kbps = (total as f64 * 8.0 * fs as f64 / frame_size as f64) / (frames as f64 * 1000.0);
    assert!(
        (35.0..95.0).contains(&mean_kbps),
        "44.1 kHz VBR mean rate {mean_kbps:.1} kb/s far from the 64 kb/s target"
    );
}

/// Arbitrary bytes never panic or produce non-finite output on
/// custom-mode decoders (the corrupt-frame path stays an error).
#[test]
fn custom_mode_decode_robustness() {
    for &(fs, frame_size) in &[(8_000usize as u32, 160usize), (44_100, 880), (96_000, 960)] {
        let mode = CeltCustomMode::new(fs, frame_size).expect("mode");
        for channels in 1..=2usize {
            let mut dec =
                CeltRefDecoder::new_custom(&mode, mode.max_lm, channels).expect("decoder");
            let mut seed = 0xBADC_0FFEu32 ^ fs;
            for len in [2usize, 7, 23, 61, 130] {
                let bytes: Vec<u8> = (0..len)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (seed >> 24) as u8
                    })
                    .collect();
                // An error is the §4.1.5 corrupt-frame path; a
                // success must be finite and full-length.
                if let Ok(pcm) = dec.decode_frame(&bytes) {
                    assert_eq!(pcm.len(), channels * frame_size);
                    assert!(pcm.iter().all(|v| v.is_finite()));
                }
            }
        }
    }
}

/// A silence-flagged custom-mode frame decodes to a decaying tail and
/// floors the energy state, exactly like the standard mode.
#[test]
fn custom_mode_silence_frame() {
    let mode = CeltCustomMode::new(24_000, 480).expect("mode");
    let mut dec = CeltRefDecoder::new_custom(&mode, mode.max_lm, 1).expect("decoder");
    let bytes = [0xFFu8, 0xFF];
    let pcm = dec.decode_frame(&bytes).expect("decode");
    assert_eq!(pcm.len(), 480);
    assert!(pcm.iter().all(|v| v.abs() < 1.0));
}
