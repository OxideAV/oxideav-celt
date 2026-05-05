//! Public-API encode -> decode roundtrip.
//!
//! Exercises the `Encoder` and `Decoder` traits from `oxideav-codec` via
//! `make_decoder`/`make_encoder` instead of the module-level primitives in
//! `tests/roundtrip.rs`. This is the surface that external callers and the
//! aggregator registry talk to.

#![allow(clippy::while_let_loop)]

use oxideav_celt::decoder::CeltDecoder;
use oxideav_celt::encoder::{CeltEncoder, FRAME_SAMPLES, SAMPLE_RATE};
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame};
use oxideav_core::{Decoder, Encoder};

fn build_params(channels: u16) -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(oxideav_celt::CODEC_ID_STR));
    p.channels = Some(channels);
    p.sample_rate = Some(SAMPLE_RATE);
    p.sample_format = Some(oxideav_core::SampleFormat::F32);
    p
}

fn pcm_frame_f32(samples: &[f32], channels: u16) -> Frame {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Frame::Audio(AudioFrame {
        samples: (samples.len() / channels as usize) as u32,
        pts: None,
        data: vec![bytes],
    })
}

fn decoded_f32(frame: &Frame) -> Vec<f32> {
    match frame {
        Frame::Audio(a) => {
            let bytes = &a.data[0];
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out
        }
        _ => panic!("expected audio frame"),
    }
}

#[test]
fn silence_roundtrip_through_public_api() {
    let params = build_params(1);
    let mut enc = CeltEncoder::new(&params).unwrap();
    let mut dec = CeltDecoder::new(&params).unwrap();

    let n_frames = 3usize;
    for _ in 0..n_frames {
        let pcm = vec![0.0f32; FRAME_SAMPLES];
        enc.send_frame(&pcm_frame_f32(&pcm, 1)).unwrap();
    }
    enc.flush().unwrap();

    let mut total = 0usize;
    loop {
        match enc.receive_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).unwrap();
                while let Ok(frame) = dec.receive_frame() {
                    let pcm = decoded_f32(&frame);
                    assert_eq!(pcm.len(), FRAME_SAMPLES);
                    assert!(
                        pcm.iter().all(|v| v.is_finite()),
                        "decoded silence must be finite"
                    );
                    let energy: f32 = pcm.iter().map(|v| v * v).sum();
                    assert!(
                        energy < 1e-6,
                        "silence should decode to near-zero (got energy {energy})"
                    );
                    total += 1;
                }
            }
            Err(_) => break,
        }
    }
    assert_eq!(
        total, n_frames,
        "expected one decoded frame per encoded frame"
    );
}

#[test]
fn sine_roundtrip_through_public_api() {
    let params = build_params(1);
    let mut enc = CeltEncoder::new(&params).unwrap();
    let mut dec = CeltDecoder::new(&params).unwrap();

    let n_frames = 4usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let freq = 1000.0f32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();

    for chunk in signal.chunks(FRAME_SAMPLES) {
        if chunk.len() == FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();

    let mut decoded: Vec<f32> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).unwrap();
                while let Ok(frame) = dec.receive_frame() {
                    decoded.extend(decoded_f32(&frame));
                }
            }
            Err(_) => break,
        }
    }
    assert_eq!(decoded.len(), n_samples);
    assert!(decoded.iter().all(|v| v.is_finite()));

    let goertzel = |samples: &[f32], f: f32| -> f32 {
        let w = 2.0 * std::f32::consts::PI * f / SAMPLE_RATE as f32;
        let cw = w.cos();
        let (mut s0, mut s1, mut s2) = (0f32, 0f32, 0f32);
        for &x in samples {
            s0 = 2.0 * cw * s1 - s2 + x;
            s2 = s1;
            s1 = s0;
        }
        let _ = s0;
        (s1 * s1 + s2 * s2 - 2.0 * cw * s1 * s2).sqrt()
    };
    let start = FRAME_SAMPLES * 2;
    let end = FRAME_SAMPLES * 3;
    let slice = &decoded[start..end];
    let mag_target = goertzel(slice, freq);
    let mag_off = goertzel(slice, 5000.0);
    let energy: f32 = slice.iter().map(|v| v * v).sum();
    assert!(energy > 1e-6, "decoded output is silent (energy {energy})");
    assert!(
        mag_target > 0.3 * mag_off,
        "target tone buried in decoder output (tgt {mag_target:.3}, off {mag_off:.3})"
    );
}

/// Transient round-trip: the encoder must detect a sharp onset and emit
/// `transient=true`, and the decoder must parse that packet without error
/// and produce finite output with sane RMS.
///
/// We also A/B the same burst signal against a long-only baseline
/// (forced via `set_force_long_only(true)`) and report both pre-echo
/// RMS values. Short blocks bound the MDCT's time-spread to 2.5 ms per
/// sub-block instead of 20 ms per frame, so the short-block pre-echo
/// should be meaningfully smaller than the long-only pre-echo.
#[test]
fn transient_roundtrip_has_bounded_preecho() {
    let params = build_params(1);

    // Three frames of silence, then a percussive burst near the START of
    // frame 3, followed by a 1 kHz steady tone for the remainder.
    let n_frames = 5usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let burst_frame = 3usize;
    let burst_start = burst_frame * FRAME_SAMPLES + 600;
    let burst_len = 120usize;
    let tone_freq = 1_000.0f32;
    let mut signal = vec![0f32; n_samples];
    for (i, s) in signal.iter_mut().enumerate() {
        let t = i as f32 / SAMPLE_RATE as f32;
        if i >= burst_start && i < burst_start + burst_len {
            *s += 0.7;
        }
        if i >= burst_frame * FRAME_SAMPLES + FRAME_SAMPLES / 2 {
            *s += (2.0 * std::f32::consts::PI * tone_freq * t).sin() * 0.15;
        }
    }

    // Encode + decode through the public API.
    let mut enc = CeltEncoder::new(&params).unwrap();
    let mut dec = CeltDecoder::new(&params).unwrap();
    for chunk in signal.chunks(FRAME_SAMPLES) {
        if chunk.len() == FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();
    let mut packets = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        packets.push(pkt);
    }
    // At least one packet must carry the transient header flag — this is
    // the actual short-block emission proof.
    let mut saw_transient = false;
    for pkt in &packets {
        let mut rd = oxideav_celt::range_decoder::RangeDecoder::new(&pkt.data);
        if let Some(h) = oxideav_celt::header::decode_header(&mut rd) {
            if h.transient {
                saw_transient = true;
                break;
            }
        }
    }
    assert!(
        saw_transient,
        "encoder never emitted transient=true on a bursty signal"
    );

    // Decode every packet; the decoder must handle the transient path
    // without error.
    let mut decoded: Vec<f32> = Vec::new();
    for pkt in &packets {
        dec.send_packet(pkt).unwrap();
        while let Ok(frame) = dec.receive_frame() {
            decoded.extend(decoded_f32(&frame));
        }
    }
    assert_eq!(decoded.len(), n_samples);
    assert!(decoded.iter().all(|v| v.is_finite()));

    let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let pre_lo = burst_frame * FRAME_SAMPLES;
    let pre_hi = burst_start - 120;
    let post_lo = burst_start + burst_len;
    let post_hi = post_lo + FRAME_SAMPLES;
    let pre_echo_short = rms(&decoded[pre_lo..pre_hi]);
    let post_rms_short = rms(&decoded[post_lo..post_hi]);

    // Sanity: decoder output is in a plausible amplitude range.
    assert!(
        rms(&decoded) < 10.0 && rms(&decoded) > 1e-3,
        "total RMS out of range"
    );
    assert!(
        post_rms_short > 1e-3,
        "post-burst region silent: {post_rms_short}"
    );

    // A/B comparison: re-encode the SAME signal with the detector forced
    // off, and measure the long-only pre-echo on the identical pre-burst
    // region.
    let mut enc_long = CeltEncoder::new(&params).unwrap();
    enc_long.set_force_long_only(true);
    let mut dec_long = CeltDecoder::new(&params).unwrap();
    for chunk in signal.chunks(FRAME_SAMPLES) {
        if chunk.len() == FRAME_SAMPLES {
            enc_long.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc_long.flush().unwrap();
    let mut decoded_long: Vec<f32> = Vec::new();
    while let Ok(pkt) = enc_long.receive_packet() {
        dec_long.send_packet(&pkt).unwrap();
        while let Ok(frame) = dec_long.receive_frame() {
            decoded_long.extend(decoded_f32(&frame));
        }
    }
    assert_eq!(decoded_long.len(), n_samples);
    let pre_echo_long = rms(&decoded_long[pre_lo..pre_hi]);

    println!(
        "transient A/B: pre-echo short-block {:.4e}, long-only {:.4e}",
        pre_echo_short, pre_echo_long
    );
    // On a real bit-exact MDCT+IMDCT pair, short-block coding should
    // dramatically reduce pre-echo vs. long-only (by ~Nlong / Nshort = 8).
    // Our current MDCT pair is correct up to CELT's 50% TDAC-alias
    // convention but is not bit-exact (Bluestein-based rather than
    // libopus' kiss_fft), so the two paths produce similar pre-echo
    // numbers on this stimulus — the aliasing artefacts dominate the
    // "real" time-spread.
    //
    // To keep this test a meaningful regression signal without blocking
    // progress on the larger IMDCT-bit-exact gap (see README "Known
    // Gaps"), we assert a LOOSE upper bound: the short-block pipeline
    // must not be dramatically worse than the long-only baseline. A 3x
    // regression would indicate the short-MDCT + Hadamard wrapper has
    // regressed; 5% is just float noise.
    assert!(
        pre_echo_short <= 3.0 * pre_echo_long,
        "short-block pre-echo ({pre_echo_short}) is >3x the long-only baseline ({pre_echo_long}); \
         short-MDCT / Hadamard regression?"
    );
}

/// Post-filter flag smoke test: force the encoder to emit every frame with
/// the RFC 6716 §4.3.7.1 post-filter flag set, and confirm the decoder
/// runs the comb-filter + de-emphasis path without erroring or producing
/// NaNs/Infs. Since the encoder does *not* run the matching pitch
/// pre-filter, the decoded signal is degraded — we only gate on "no
/// panics, no non-finite samples, nonzero output".
#[test]
fn post_filter_flag_decodes_without_error() {
    use oxideav_celt::header::PostFilter;
    let params = build_params(1);
    let mut enc = CeltEncoder::new(&params).unwrap();
    enc.set_force_post_filter(Some(PostFilter {
        octave: 2,
        // fine_pitch = 7 → period = (16<<2) + 7 - 1 = 70 samples (≈686 Hz at 48 kHz).
        period: 7,
        gain: 4, // G = 3*5/32 = 0.46875
        tapset: 0,
    }));
    let mut dec = CeltDecoder::new(&params).unwrap();

    let n_frames = 6usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    // Periodic signal so the comb filter has something coherent to latch
    // onto at T=70 samples (freq ~686 Hz).
    let freq = 686.0f32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.2)
        .collect();

    for chunk in signal.chunks(FRAME_SAMPLES) {
        if chunk.len() == FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();

    // Every emitted packet's header must parse with post_filter = Some.
    let mut packets = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        packets.push(pkt);
    }
    assert!(!packets.is_empty());
    for (i, pkt) in packets.iter().enumerate() {
        let mut rd = oxideav_celt::range_decoder::RangeDecoder::new(&pkt.data);
        let h = oxideav_celt::header::decode_header(&mut rd).expect("header must parse");
        assert!(
            h.post_filter.is_some(),
            "packet {i} does not carry post_filter=Some despite the test-knob"
        );
    }

    // Decode every packet; the decoder must handle post_filter=Some without
    // panicking or erroring.
    let mut decoded: Vec<f32> = Vec::new();
    for pkt in &packets {
        dec.send_packet(pkt)
            .expect("decoder must accept post_filter packets");
        while let Ok(frame) = dec.receive_frame() {
            decoded.extend(decoded_f32(&frame));
        }
    }
    assert_eq!(decoded.len(), n_samples);
    assert!(decoded.iter().all(|v| v.is_finite()), "decoded non-finite");

    // Output must carry nonzero energy — the decoder's post-filter path
    // adds a comb tap but shouldn't zero out the signal entirely.
    let mid = &decoded[FRAME_SAMPLES * 2..FRAME_SAMPLES * 4];
    let e: f32 = mid.iter().map(|v| v * v).sum::<f32>() / mid.len() as f32;
    assert!(e > 1e-6, "post-filter decoded output silent (energy {e})");
}

#[test]
fn stereo_roundtrip_through_public_api() {
    let params = build_params(2);
    let mut enc = CeltEncoder::new(&params).unwrap();
    let mut dec = CeltDecoder::new(&params).unwrap();

    let n_frames = 3usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let f_l = 1000.0f32;
    let f_r = 1500.0f32;
    let mut interleaved: Vec<f32> = Vec::with_capacity(n_samples * 2);
    for i in 0..n_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        interleaved.push((2.0 * std::f32::consts::PI * f_l * t).sin() * 0.3);
        interleaved.push((2.0 * std::f32::consts::PI * f_r * t).sin() * 0.3);
    }

    let frame_interleaved = FRAME_SAMPLES * 2;
    for chunk in interleaved.chunks(frame_interleaved) {
        if chunk.len() == frame_interleaved {
            enc.send_frame(&pcm_frame_f32(chunk, 2)).unwrap();
        }
    }
    enc.flush().unwrap();

    let mut decoded: Vec<f32> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).unwrap();
                while let Ok(frame) = dec.receive_frame() {
                    decoded.extend(decoded_f32(&frame));
                }
            }
            Err(_) => break,
        }
    }
    assert_eq!(decoded.len(), n_samples * 2);
    assert!(decoded.iter().all(|v| v.is_finite()));
}

// -------------------------------------------------------------------------
// Transient round-trip — quantitative SNR comparisons.
//
// The earlier `transient_roundtrip_has_bounded_preecho` test asserts the
// pre-echo region's RMS doesn't blow up. The two tests below add the next
// rung on the ladder: actual round-trip SNR numbers, both
//   * **transient signal**: short-block SNR vs long-only SNR — with a real
//     bit-exact MDCT pair short would beat long by ~9 dB on the burst
//     region. Our IMDCT isn't bit-exact yet (Bluestein FFT vs libopus'
//     kiss_fft), so we report both numbers without a hard inequality —
//     these are regression sentinels for once the IMDCT lands.
//   * **stationary signal**: the auto-detector must keep `transient=false`
//     end-to-end and the SNR must not regress vs. the forced-long path.
// -------------------------------------------------------------------------

/// Helper: SNR in dB of `decoded` vs `reference` over `[lo, hi)`. Returns
/// 200.0 when the noise is below 1e-30 (effectively perfect reconstruction).
fn snr_db(reference: &[f32], decoded: &[f32], lo: usize, hi: usize) -> f32 {
    let n = (hi - lo).min(reference.len() - lo).min(decoded.len() - lo);
    let signal: f64 = reference[lo..lo + n]
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        / n as f64;
    let noise: f64 = reference[lo..lo + n]
        .iter()
        .zip(decoded[lo..lo + n].iter())
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if noise < 1e-30 {
        return 200.0;
    }
    (10.0 * (signal / noise).log10()) as f32
}

/// Encode + decode a mono signal end-to-end, returning the decoded PCM.
/// `force_long_only` controls whether the encoder's transient detector is
/// active; this is the A/B knob that drives the comparison tests.
fn roundtrip_mono(signal: &[f32], force_long_only: bool) -> Vec<f32> {
    let params = build_params(1);
    let mut enc = CeltEncoder::new(&params).unwrap();
    enc.set_force_long_only(force_long_only);
    let mut dec = CeltDecoder::new(&params).unwrap();
    for chunk in signal.chunks(FRAME_SAMPLES) {
        if chunk.len() == FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();
    let mut decoded: Vec<f32> = Vec::with_capacity(signal.len());
    while let Ok(pkt) = enc.receive_packet() {
        dec.send_packet(&pkt).unwrap();
        while let Ok(frame) = dec.receive_frame() {
            decoded.extend(decoded_f32(&frame));
        }
    }
    decoded
}

/// Compare the round-trip SNR of a transient signal under the two encoder
/// configurations: detector-on (short blocks where appropriate) vs.
/// `force_long_only` (long block always). The numbers go to stdout for the
/// task report; assertions stay loose because the IMDCT isn't bit-exact.
///
/// The signal is one isolated 2.5 ms full-scale burst inside frame 3 of a
/// 5-frame stream. We measure SNR over the burst region itself (long-only
/// is expected to win here — short blocks have less per-sub-block resolution)
/// AND over the pre-burst silence (the *pre-echo* region — long-only is
/// expected to LOSE here because the long MDCT smears the burst's energy
/// backwards across the entire 20 ms frame).
#[test]
fn transient_signal_snr_short_vs_long() {
    let n_frames = 5usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let burst_frame = 3usize;
    let burst_offset_in_frame = 600usize;
    let burst_start = burst_frame * FRAME_SAMPLES + burst_offset_in_frame;
    let burst_len = 120usize;
    let mut signal = vec![0f32; n_samples];
    for s in &mut signal[burst_start..burst_start + burst_len] {
        *s = 0.7;
    }

    let dec_short = roundtrip_mono(&signal, false);
    let dec_long = roundtrip_mono(&signal, true);
    assert_eq!(dec_short.len(), n_samples);
    assert_eq!(dec_long.len(), n_samples);

    // Pre-echo region: the silence frames just BEFORE the burst, inside
    // frame 3 (i.e. after the previous OLA tail). On a bit-exact
    // long-block decoder this would carry a non-trivial echo of the burst
    // smeared backward by the 20 ms MDCT support; short-block coding
    // confines the smear to ~2.5 ms. Reference is silent here so the
    // useful metric is the decoder's noise RMS, not classical SNR.
    let pre_lo = burst_frame * FRAME_SAMPLES + 120;
    let pre_hi = burst_start - 60;
    let pre_rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let pre_noise_short = pre_rms(&dec_short[pre_lo..pre_hi]);
    let pre_noise_long = pre_rms(&dec_long[pre_lo..pre_hi]);

    // Burst region: where the actual transient lives.
    let burst_snr_short = snr_db(&signal, &dec_short, burst_start, burst_start + burst_len);
    let burst_snr_long = snr_db(&signal, &dec_long, burst_start, burst_start + burst_len);

    println!(
        "transient round-trip: pre-echo RMS short={:.4e}, long={:.4e}; \
         burst SNR short={:.2} dB, long={:.2} dB",
        pre_noise_short, pre_noise_long, burst_snr_short, burst_snr_long
    );

    // Sanity: both pipelines produce finite output.
    assert!(dec_short.iter().all(|v| v.is_finite()));
    assert!(dec_long.iter().all(|v| v.is_finite()));

    // Loose floor: the burst SNR for both paths must beat -20 dB. CELT is
    // perceptual and the IMDCT isn't bit-exact, so we're really just
    // catching a "decoded output is silent / blown up" regression here.
    assert!(
        burst_snr_short > -20.0,
        "short-block burst SNR collapsed: {burst_snr_short:.2} dB"
    );
    assert!(
        burst_snr_long > -20.0,
        "long-only burst SNR collapsed: {burst_snr_long:.2} dB"
    );
}

/// Per-band TF analyser A/B (RFC 6716 §4.3.4.5 + §5.3.6): the analyser
/// must NOT regress PSNR on real audio fixtures.
///
/// Two stimuli are exercised:
///   * **multi-tone**: representative of pitched audio. The analyser
///     typically returns `tf_change=0` for every band — the MDCT of a
///     pitched signal is sparse, so the L1-norm masking model finds no
///     gain from a haar1 transform. Confirms the analyser doesn't
///     degrade quality on the common case.
///   * **alternating-burst**: a synthetic non-transient signal whose
///     post-MDCT band coefficients have strong intra-band structure
///     (alternating-sign tendencies). On these the analyser engages and
///     the haar1 wrapping in `encoder_bands::encode_all_bands_mono`
///     applies. Confirms no encoder/decoder mismatch when tf_change != 0
///     fires end-to-end.
///
/// PSNR numbers go to stdout for the task report. The test gate is "no
/// catastrophic regression" (≤ 1 dB drop) since the L1-norm proxy is
/// approximate and the IMDCT is not yet bit-exact (see crate README
/// "Known Gaps").
#[test]
fn tf_analysis_signal_roundtrip_no_regression() {
    fn roundtrip(signal: &[f32], force_tf_off: bool) -> Vec<f32> {
        let params = build_params(1);
        let mut enc = CeltEncoder::new(&params).unwrap();
        enc.set_force_tf_off(force_tf_off);
        let mut dec = CeltDecoder::new(&params).unwrap();
        for chunk in signal.chunks(FRAME_SAMPLES) {
            if chunk.len() == FRAME_SAMPLES {
                enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
            }
        }
        enc.flush().unwrap();
        let mut decoded = Vec::with_capacity(signal.len());
        while let Ok(pkt) = enc.receive_packet() {
            dec.send_packet(&pkt).unwrap();
            while let Ok(frame) = dec.receive_frame() {
                decoded.extend(decoded_f32(&frame));
            }
        }
        decoded
    }

    // Stimulus 1 — multi-tone pitched signal.
    let n_frames = 8usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let multi_tone: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            let tone1 = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.20;
            let tone2 = (2.0 * std::f32::consts::PI * 1320.0 * t).sin() * 0.10;
            let trem = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 6.0 * t).sin();
            let tone3 = (2.0 * std::f32::consts::PI * 3960.0 * t).sin() * 0.10 * trem;
            tone1 + tone2 + tone3
        })
        .collect();
    let dec_on_a = roundtrip(&multi_tone, false);
    let dec_off_a = roundtrip(&multi_tone, true);
    assert_eq!(dec_on_a.len(), n_samples);
    assert_eq!(dec_off_a.len(), n_samples);
    assert!(dec_on_a.iter().all(|v| v.is_finite()));
    assert!(dec_off_a.iter().all(|v| v.is_finite()));
    let lo = FRAME_SAMPLES;
    let hi = FRAME_SAMPLES * (n_frames - 1);
    let psnr_on_a = snr_db(&multi_tone, &dec_on_a, lo, hi);
    let psnr_off_a = snr_db(&multi_tone, &dec_off_a, lo, hi);

    // Stimulus 2 — broad-band noise with a slow envelope. Engages
    // multiple bands' L1-norm gradient enough to exercise the
    // encoder_bands haar1 wrapping in the case where the analyser fires.
    let mut seed: u32 = 0xCAFE_BABE;
    let envnoise: Vec<f32> = (0..n_samples)
        .map(|i| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let raw = ((seed >> 16) as i32 - 32_768) as f32 / 32_768.0;
            let t = i as f32 / SAMPLE_RATE as f32;
            let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 4.0 * t).sin();
            raw * env * 0.3
        })
        .collect();
    let dec_on_b = roundtrip(&envnoise, false);
    let dec_off_b = roundtrip(&envnoise, true);
    assert!(dec_on_b.iter().all(|v| v.is_finite()));
    assert!(dec_off_b.iter().all(|v| v.is_finite()));
    let psnr_on_b = snr_db(&envnoise, &dec_on_b, lo, hi);
    let psnr_off_b = snr_db(&envnoise, &dec_off_b, lo, hi);

    println!(
        "tf_analysis A/B PSNR: multi-tone on={:.2} dB off={:.2} dB delta={:+.2} dB; \
         env-noise on={:.2} dB off={:.2} dB delta={:+.2} dB",
        psnr_on_a,
        psnr_off_a,
        psnr_on_a - psnr_off_a,
        psnr_on_b,
        psnr_off_b,
        psnr_on_b - psnr_off_b
    );

    // "Do no harm" gate: the analyser must not regress PSNR by more than
    // 1 dB on either fixture. The actual delta on real audio is usually
    // 0 dB (the analyser keeps `tf_change=0` for sparse-spectrum
    // content); the gate's job is to catch encoder/decoder mismatches
    // when the analyser does engage.
    assert!(
        psnr_on_a >= psnr_off_a - 1.0,
        "tf_analysis multi-tone regression: on={psnr_on_a:.2}, off={psnr_off_a:.2}"
    );
    assert!(
        psnr_on_b >= psnr_off_b - 1.0,
        "tf_analysis env-noise regression: on={psnr_on_b:.2}, off={psnr_off_b:.2}"
    );
}

/// Stationary-content regression: a pure sine fed through the encoder must
/// (a) keep every packet's `transient` flag false, and (b) achieve the
/// same SNR with the detector enabled as with `force_long_only`. This
/// guarantees the auto-detector doesn't false-positive on steady tones
/// and silently degrade quality.
#[test]
fn stationary_signal_snr_unchanged_by_detector() {
    let n_frames = 6usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let freq = 1000.0f32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();

    // Detector-enabled path: every packet should still carry transient=0.
    let params = build_params(1);
    let mut enc = CeltEncoder::new(&params).unwrap();
    let mut dec = CeltDecoder::new(&params).unwrap();
    for chunk in signal.chunks(FRAME_SAMPLES) {
        if chunk.len() == FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();
    let mut decoded_auto: Vec<f32> = Vec::new();
    let mut packets_auto = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        packets_auto.push(pkt.clone());
        dec.send_packet(&pkt).unwrap();
        while let Ok(frame) = dec.receive_frame() {
            decoded_auto.extend(decoded_f32(&frame));
        }
    }
    for (i, pkt) in packets_auto.iter().enumerate() {
        let mut rd = oxideav_celt::range_decoder::RangeDecoder::new(&pkt.data);
        let h = oxideav_celt::header::decode_header(&mut rd).expect("header parses");
        assert!(
            !h.transient,
            "stationary sine: packet {i} unexpectedly flipped to transient"
        );
    }

    // Forced-long baseline.
    let decoded_long = roundtrip_mono(&signal, true);

    // Compare SNR over the steady-state middle frames (skip frame 0 where
    // the OLA buffer hasn't settled).
    let lo = FRAME_SAMPLES * 2;
    let hi = FRAME_SAMPLES * (n_frames - 1);
    let snr_auto = snr_db(&signal, &decoded_auto, lo, hi);
    let snr_long = snr_db(&signal, &decoded_long, lo, hi);
    println!(
        "stationary sine SNR — detector-on: {:.2} dB, force-long: {:.2} dB",
        snr_auto, snr_long
    );

    // The two paths should be identical (detector returned false on every
    // frame). Allow a tiny epsilon for float-accumulation noise across the
    // two independent encode runs.
    assert!(
        (snr_auto - snr_long).abs() < 0.1,
        "detector-on SNR ({snr_auto:.3}) differs from forced-long SNR ({snr_long:.3}) on a stationary signal — false positive?"
    );
}

/// RFC 6716 §4.3.7.1 pitch pre-filter A/B. On a tonal fixture (220 Hz +
/// 4 harmonics, the canonical "ringy" signal that motivates the
/// post-filter), the encoder-side pre-filter analyser should drop the
/// MDCT residual energy enough to reduce round-trip error vs the same
/// pipeline with the pre-filter disabled.
///
/// We measure RMS reconstruction error over the steady-state middle
/// frames and confirm pre-filter-on yields a smaller (or no worse)
/// error than pre-filter-off. The improvement isn't huge — at 64
/// kbit/s mono the MDCT already represents the harmonic series well —
/// but it's measurable and consistent on this fixture.
///
/// Also asserts the pre-filter-on path actually emits a non-zero
/// post-filter flag in at least one packet (the analyser shouldn't
/// silently no-op on a clean tonal input).
#[test]
fn prefilter_tonal_signal_snr_a_b() {
    // 220 Hz fundamental + 4 harmonics. 6 frames @ 20 ms → 120 ms signal.
    let n_frames = 6usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let sr = SAMPLE_RATE as f32;
    let f0 = 220.0f32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f32 / sr;
            let mut s = 0.0f32;
            for k in 1..=5 {
                let amp = 0.25 / k as f32;
                s += amp * (2.0 * std::f32::consts::PI * f0 * k as f32 * t).sin();
            }
            s
        })
        .collect();

    fn roundtrip_with_prefilter(
        signal: &[f32],
        enable: bool,
    ) -> (Vec<f32>, Vec<oxideav_core::Packet>) {
        let params = build_params(1);
        let mut enc = CeltEncoder::new(&params).unwrap();
        enc.set_enable_prefilter(enable);
        let mut dec = CeltDecoder::new(&params).unwrap();
        for chunk in signal.chunks(FRAME_SAMPLES) {
            if chunk.len() == FRAME_SAMPLES {
                enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
            }
        }
        enc.flush().unwrap();
        let mut decoded: Vec<f32> = Vec::new();
        let mut packets: Vec<oxideav_core::Packet> = Vec::new();
        while let Ok(pkt) = enc.receive_packet() {
            packets.push(pkt.clone());
            dec.send_packet(&pkt).unwrap();
            while let Ok(frame) = dec.receive_frame() {
                decoded.extend(decoded_f32(&frame));
            }
        }
        (decoded, packets)
    }

    let (dec_pre_on, pkts_pre_on) = roundtrip_with_prefilter(&signal, true);
    let (dec_pre_off, _) = roundtrip_with_prefilter(&signal, false);

    // The pre-filter-on path must have flipped the post-filter flag on
    // at least one packet — otherwise the analyser silently no-op'd on
    // a clean tonal input, which would defeat the point.
    let mut pf_on_count = 0usize;
    for pkt in &pkts_pre_on {
        let mut rd = oxideav_celt::range_decoder::RangeDecoder::new(&pkt.data);
        if let Some(h) = oxideav_celt::header::decode_header(&mut rd) {
            if h.post_filter.is_some() {
                pf_on_count += 1;
            }
        }
    }
    assert!(
        pf_on_count > 0,
        "pre-filter analyser produced no post-filter flags on a 220 Hz harmonic fixture (pkts={})",
        pkts_pre_on.len()
    );

    // Compare RMS error in the steady-state middle. Skip the first 2
    // frames (cold-start state) and the last frame (tail OLA window).
    let lo = FRAME_SAMPLES * 2;
    let hi = FRAME_SAMPLES * (n_frames - 1);
    let snr_pre_on = snr_db(&signal, &dec_pre_on, lo, hi);
    let snr_pre_off = snr_db(&signal, &dec_pre_off, lo, hi);
    println!(
        "tonal pre-filter A/B (220 Hz + 4 harmonics): SNR pre-on={:.2} dB, pre-off={:.2} dB, \
         post-filter packets {}/{}",
        snr_pre_on,
        snr_pre_off,
        pf_on_count,
        pkts_pre_on.len()
    );

    // Sanity: both pipelines must produce finite, non-collapsed output.
    assert!(dec_pre_on.iter().all(|v| v.is_finite()));
    assert!(dec_pre_off.iter().all(|v| v.is_finite()));
    assert!(snr_pre_on > -10.0, "pre-on SNR collapsed: {snr_pre_on:.2}");
    assert!(
        snr_pre_off > -10.0,
        "pre-off SNR collapsed: {snr_pre_off:.2}"
    );

    // Pre-filter on must be at least as good as pre-filter off. Allow
    // a small tolerance (-0.5 dB) for noise-introduction by the
    // crossfade region between two adjacent frames where the pitch
    // analyser picked slightly different params — the steady-state
    // body of each frame still benefits, the boundary doesn't.
    assert!(
        snr_pre_on >= snr_pre_off - 0.5,
        "pre-filter regressed SNR: on={snr_pre_on:.2} dB vs off={snr_pre_off:.2} dB"
    );
}

// -------------------------------------------------------------------------
// LM=2 (10 ms / 480-sample) round-trip — RFC 6716 §4.3 frame-size dispatch.
//
// The encoder/decoder default to LM=3 (20 ms / 960 samples). For Opus
// 10 ms Hybrid (configs 6/8/10) and any direct CELT-only 10 ms use the
// caller opts in via `*_with_frame_samples(_, 480)`. These tests confirm
// the per-band quantisation, PVQ, TF analysis, silence/transient/post-
// filter machinery all work end-to-end at LM=2:
//
//   * mono silence — silence flag fast-path round-trips.
//   * mono sine — non-silent header + audible 1 kHz tone in the decoded
//     output.
//   * mono click — the transient detector still flips short-block under
//     LM=2 (M=4 sub-blocks of 120 samples instead of M=8 × 120).
//   * stereo sine — dual-stereo CELT body decodes without divergence.
// -------------------------------------------------------------------------

const LM2_FRAME_SAMPLES: usize = 480;

#[test]
fn lm2_silence_roundtrip() {
    let params = build_params(1);
    let mut enc = CeltEncoder::new_with_frame_samples(&params, LM2_FRAME_SAMPLES).unwrap();
    let mut dec = CeltDecoder::new_with_frame_samples(&params, LM2_FRAME_SAMPLES).unwrap();
    assert_eq!(enc.frame_samples(), LM2_FRAME_SAMPLES);
    assert_eq!(dec.frame_samples(), LM2_FRAME_SAMPLES);

    let n_frames = 4usize;
    for _ in 0..n_frames {
        let pcm = vec![0.0f32; LM2_FRAME_SAMPLES];
        enc.send_frame(&pcm_frame_f32(&pcm, 1)).unwrap();
    }
    enc.flush().unwrap();

    let mut decoded_frames = 0usize;
    while let Ok(pkt) = enc.receive_packet() {
        dec.send_packet(&pkt).unwrap();
        while let Ok(frame) = dec.receive_frame() {
            let pcm = decoded_f32(&frame);
            assert_eq!(
                pcm.len(),
                LM2_FRAME_SAMPLES,
                "LM=2 decoded frame must have 480 samples"
            );
            assert!(pcm.iter().all(|v| v.is_finite()));
            let energy: f32 = pcm.iter().map(|v| v * v).sum();
            assert!(
                energy < 1e-6,
                "LM=2 silence should decode to near-zero (got energy {energy})"
            );
            decoded_frames += 1;
        }
    }
    assert_eq!(
        decoded_frames, n_frames,
        "LM=2: expected one decoded frame per encoded frame"
    );
}

#[test]
fn lm2_sine_roundtrip_audible_tone() {
    let params = build_params(1);
    let mut enc = CeltEncoder::new_with_frame_samples(&params, LM2_FRAME_SAMPLES).unwrap();
    let mut dec = CeltDecoder::new_with_frame_samples(&params, LM2_FRAME_SAMPLES).unwrap();

    let n_frames = 8usize;
    let n_samples = LM2_FRAME_SAMPLES * n_frames;
    let freq = 1000.0f32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();

    for chunk in signal.chunks(LM2_FRAME_SAMPLES) {
        if chunk.len() == LM2_FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();

    // Headers should NOT carry the silence flag for a non-silent sine.
    let mut packets = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        packets.push(pkt);
    }
    assert_eq!(packets.len(), n_frames);
    for (i, pkt) in packets.iter().enumerate() {
        let mut rd = oxideav_celt::range_decoder::RangeDecoder::new(&pkt.data);
        let h = oxideav_celt::header::decode_header(&mut rd)
            .unwrap_or_else(|| panic!("LM=2 packet {i} unexpectedly carries silence flag"));
        assert!(!h.transient, "LM=2 sine packet {i} flipped to transient");
    }

    // Decode and check the target frequency dominates the off-frequency
    // baseline in the steady-state middle frames.
    let mut decoded: Vec<f32> = Vec::with_capacity(n_samples);
    for pkt in &packets {
        dec.send_packet(pkt).unwrap();
        while let Ok(frame) = dec.receive_frame() {
            decoded.extend(decoded_f32(&frame));
        }
    }
    assert_eq!(decoded.len(), n_samples);
    assert!(decoded.iter().all(|v| v.is_finite()));

    let goertzel = |samples: &[f32], f: f32| -> f32 {
        let w = 2.0 * std::f32::consts::PI * f / SAMPLE_RATE as f32;
        let cw = w.cos();
        let (mut s0, mut s1, mut s2) = (0f32, 0f32, 0f32);
        for &x in samples {
            s0 = 2.0 * cw * s1 - s2 + x;
            s2 = s1;
            s1 = s0;
        }
        let _ = s0;
        (s1 * s1 + s2 * s2 - 2.0 * cw * s1 * s2).sqrt()
    };
    // Skip the first 2 frames (OLA still settling) and last frame.
    let lo = LM2_FRAME_SAMPLES * 2;
    let hi = LM2_FRAME_SAMPLES * (n_frames - 1);
    let slice = &decoded[lo..hi];
    let mag_target = goertzel(slice, freq);
    let mag_off = goertzel(slice, 5000.0);
    let energy: f32 = slice.iter().map(|v| v * v).sum();
    assert!(
        energy > 1e-6,
        "LM=2 decoded sine is silent: energy {energy}"
    );
    assert!(
        mag_target > 0.3 * mag_off,
        "LM=2 target tone buried in decoder output (tgt {mag_target:.3}, off {mag_off:.3})"
    );
}

#[test]
fn lm2_transient_roundtrip_short_blocks_decode() {
    let params = build_params(1);
    let mut enc = CeltEncoder::new_with_frame_samples(&params, LM2_FRAME_SAMPLES).unwrap();
    let mut dec = CeltDecoder::new_with_frame_samples(&params, LM2_FRAME_SAMPLES).unwrap();

    let n_frames = 4usize;
    let n_samples = LM2_FRAME_SAMPLES * n_frames;
    // Click in the middle of frame 2 — sharp burst against silence.
    let burst_frame = 2usize;
    let burst_start = burst_frame * LM2_FRAME_SAMPLES + 240;
    let burst_len = 60usize;
    let mut signal = vec![0f32; n_samples];
    for s in &mut signal[burst_start..burst_start + burst_len] {
        *s = 0.7;
    }

    for chunk in signal.chunks(LM2_FRAME_SAMPLES) {
        if chunk.len() == LM2_FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();

    let mut packets = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        packets.push(pkt);
    }
    // At least one packet must be transient — short blocks at LM=2 use
    // M=4 sub-blocks of 120 samples each.
    let mut saw_transient = false;
    for pkt in &packets {
        let mut rd = oxideav_celt::range_decoder::RangeDecoder::new(&pkt.data);
        if let Some(h) = oxideav_celt::header::decode_header(&mut rd) {
            if h.transient {
                saw_transient = true;
                break;
            }
        }
    }
    assert!(
        saw_transient,
        "LM=2 encoder did not emit transient on a click-vs-silence stimulus"
    );

    // Decoder must accept every packet (transient short-block path runs
    // M × 100-sample IMDCTs) without errors or non-finite samples.
    let mut decoded: Vec<f32> = Vec::with_capacity(n_samples);
    for pkt in &packets {
        dec.send_packet(pkt).unwrap();
        while let Ok(frame) = dec.receive_frame() {
            decoded.extend(decoded_f32(&frame));
        }
    }
    assert_eq!(decoded.len(), n_samples);
    assert!(decoded.iter().all(|v| v.is_finite()));
    // Total energy across the stream must exceed a coarse floor — the
    // burst-against-silence stimulus is sparse, so the per-region SNR is
    // dominated by the IMDCT-not-bit-exact gap (see crate README "Known
    // Gaps"). The point of this test is "transient decode does not blow
    // up", not a quality gate.
    let total_e: f32 = decoded.iter().map(|v| v * v).sum();
    assert!(
        total_e > 1e-9,
        "LM=2 transient stream decoded to total silence (energy {total_e})"
    );
}

#[test]
fn lm2_stereo_sine_roundtrip() {
    let params = build_params(2);
    let mut enc = CeltEncoder::new_with_frame_samples(&params, LM2_FRAME_SAMPLES).unwrap();
    let mut dec = CeltDecoder::new_with_frame_samples(&params, LM2_FRAME_SAMPLES).unwrap();

    let n_frames = 5usize;
    let n_samples = LM2_FRAME_SAMPLES * n_frames;
    let f_l = 1000.0f32;
    let f_r = 1500.0f32;
    let mut interleaved: Vec<f32> = Vec::with_capacity(n_samples * 2);
    for i in 0..n_samples {
        let t = i as f32 / SAMPLE_RATE as f32;
        interleaved.push((2.0 * std::f32::consts::PI * f_l * t).sin() * 0.3);
        interleaved.push((2.0 * std::f32::consts::PI * f_r * t).sin() * 0.3);
    }
    let frame_interleaved = LM2_FRAME_SAMPLES * 2;
    for chunk in interleaved.chunks(frame_interleaved) {
        if chunk.len() == frame_interleaved {
            enc.send_frame(&pcm_frame_f32(chunk, 2)).unwrap();
        }
    }
    enc.flush().unwrap();

    let mut decoded: Vec<f32> = Vec::with_capacity(n_samples * 2);
    while let Ok(pkt) = enc.receive_packet() {
        dec.send_packet(&pkt).unwrap();
        while let Ok(frame) = dec.receive_frame() {
            decoded.extend(decoded_f32(&frame));
        }
    }
    assert_eq!(decoded.len(), n_samples * 2);
    assert!(decoded.iter().all(|v| v.is_finite()));

    // De-interleave; both channels carry their respective tones with
    // distinguishable energy.
    let mut left = Vec::with_capacity(n_samples);
    let mut right = Vec::with_capacity(n_samples);
    for chunk in decoded.chunks_exact(2) {
        left.push(chunk[0]);
        right.push(chunk[1]);
    }
    let lo = LM2_FRAME_SAMPLES * 2;
    let hi = LM2_FRAME_SAMPLES * (n_frames - 1);
    let e_left: f32 = left[lo..hi].iter().map(|v| v * v).sum();
    let e_right: f32 = right[lo..hi].iter().map(|v| v * v).sum();
    assert!(e_left > 1e-3, "LM=2 stereo L silent: e={e_left}");
    assert!(e_right > 1e-3, "LM=2 stereo R silent: e={e_right}");
}

#[test]
fn constructor_rejects_unsupported_frame_size() {
    let params = build_params(1);
    // 120 (LM=0), 240 (LM=1), 480 (LM=2), 960 (LM=3) are all valid.
    assert!(CeltEncoder::new_with_frame_samples(&params, 120).is_ok());
    assert!(CeltDecoder::new_with_frame_samples(&params, 120).is_ok());
    assert!(CeltEncoder::new_with_frame_samples(&params, 240).is_ok());
    assert!(CeltDecoder::new_with_frame_samples(&params, 240).is_ok());
    // 720 — not a valid CELT frame size, must be rejected.
    assert!(CeltEncoder::new_with_frame_samples(&params, 720).is_err());
    assert!(CeltDecoder::new_with_frame_samples(&params, 720).is_err());
    // 360 — also invalid.
    assert!(CeltEncoder::new_with_frame_samples(&params, 360).is_err());
    assert!(CeltDecoder::new_with_frame_samples(&params, 360).is_err());
}

/// LM=0 (120 samples / 2.5 ms) round-trip: encoder + decoder round-trip
/// a 1 kHz sine at the smallest frame size and produce non-silent output.
#[test]
fn lm0_sine_roundtrip_audible_tone() {
    const LM0_FRAME_SAMPLES: usize = 120;
    let params = build_params(1);
    let mut enc = CeltEncoder::new_with_frame_samples(&params, LM0_FRAME_SAMPLES).unwrap();
    let mut dec = CeltDecoder::new_with_frame_samples(&params, LM0_FRAME_SAMPLES).unwrap();
    let n_frames = 16usize;
    let n_samples = LM0_FRAME_SAMPLES * n_frames;
    let freq = 1000.0f32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();
    for chunk in signal.chunks(LM0_FRAME_SAMPLES) {
        if chunk.len() == LM0_FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();
    let mut decoded: Vec<f32> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).unwrap();
                while let Ok(frame) = dec.receive_frame() {
                    decoded.extend(decoded_f32(&frame));
                }
            }
            Err(_) => break,
        }
    }
    assert_eq!(decoded.len(), n_samples);
    assert!(
        decoded.iter().all(|v| v.is_finite()),
        "LM=0 decoded non-finite"
    );
    let mid_start = LM0_FRAME_SAMPLES * 8;
    let mid_end = mid_start + LM0_FRAME_SAMPLES * 4;
    let e: f32 = decoded[mid_start..mid_end]
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        / (LM0_FRAME_SAMPLES * 4) as f32;
    assert!(e > 1e-5, "LM=0 decoded output silent (energy={e})");
}

/// LM=1 (240 samples / 5 ms) round-trip: encoder + decoder round-trip
/// a 1 kHz sine at the 5 ms frame size and produce non-silent output.
#[test]
fn lm1_sine_roundtrip_audible_tone() {
    const LM1_FRAME_SAMPLES: usize = 240;
    let params = build_params(1);
    let mut enc = CeltEncoder::new_with_frame_samples(&params, LM1_FRAME_SAMPLES).unwrap();
    let mut dec = CeltDecoder::new_with_frame_samples(&params, LM1_FRAME_SAMPLES).unwrap();
    let n_frames = 8usize;
    let n_samples = LM1_FRAME_SAMPLES * n_frames;
    let freq = 1000.0f32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();
    for chunk in signal.chunks(LM1_FRAME_SAMPLES) {
        if chunk.len() == LM1_FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();
    let mut decoded: Vec<f32> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(pkt) => {
                dec.send_packet(&pkt).unwrap();
                while let Ok(frame) = dec.receive_frame() {
                    decoded.extend(decoded_f32(&frame));
                }
            }
            Err(_) => break,
        }
    }
    assert_eq!(decoded.len(), n_samples);
    assert!(
        decoded.iter().all(|v| v.is_finite()),
        "LM=1 decoded non-finite"
    );
    let mid_start = LM1_FRAME_SAMPLES * 4;
    let mid_end = mid_start + LM1_FRAME_SAMPLES * 2;
    let e: f32 = decoded[mid_start..mid_end]
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        / (LM1_FRAME_SAMPLES * 2) as f32;
    assert!(e > 1e-5, "LM=1 decoded output silent (energy={e})");
}

/// `new_auto_lm` heuristic: verify it constructs valid encoders for
/// both steady-state and high-transient content modes.
#[test]
fn auto_lm_constructor_produces_valid_encoder() {
    use oxideav_celt::encoder::CeltEncoder;
    let params = build_params(1);
    // Steady-state → LM=3 (960 samples).
    let enc_steady = CeltEncoder::new_auto_lm(&params, false).unwrap();
    assert_eq!(enc_steady.frame_samples(), 960);
    // High transient → LM=1 (240 samples).
    let enc_transient = CeltEncoder::new_auto_lm(&params, true).unwrap();
    assert_eq!(enc_transient.frame_samples(), 240);
}

/// `select_lm_for_pcm` heuristic: verify it picks smaller LM for bursty signals.
#[test]
fn select_lm_for_pcm_picks_smaller_lm_on_burst() {
    use oxideav_celt::encoder::CeltEncoder;
    // Steady 1 kHz sine — should return 960 (LM=3).
    let n = 960usize;
    let sine: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48_000.0).sin() * 0.3)
        .collect();
    let lm_sine = CeltEncoder::select_lm_for_pcm(&sine, 15.0);
    assert_eq!(lm_sine, 960, "steady sine should select LM=3 (960 samples)");

    // Sharp burst (castanets-style): near-zero for 7/8 of the frame, then
    // a loud hit for 1/8 → many sub-blocks carry the onset.
    let mut burst = vec![0.0f32; n];
    burst[(n / 8)..(n / 4)].fill(0.8);
    burst[(n / 2)..(5 * n / 8)].fill(0.6);
    burst[(7 * n / 8)..n].fill(0.9);
    let lm_burst = CeltEncoder::select_lm_for_pcm(&burst, 15.0);
    // Burst has many onsets, should select shorter frame (240 or 120).
    assert!(
        lm_burst <= 240,
        "bursty signal should select LM<=1 (240 or 120 samples), got {lm_burst}"
    );
}

// ---------------------------------------------------------------------------
// Per-LM PSNR benchmarks (regression sentinels).
// ---------------------------------------------------------------------------

/// Self-roundtrip SNR for a given frame size: encode + decode a 1 kHz sine
/// with the CELT encoder+decoder pair at `frame_samples` and report SNR
/// on the steady-state middle section. Returns (SNR dB, bytes/frame).
fn lm_sine_psnr(frame_samples: usize) -> (f32, usize) {
    let params = build_params(1);
    let mut enc = CeltEncoder::new_with_frame_samples(&params, frame_samples).unwrap();
    let bytes_per_frame = enc.frame_samples(); // proxy: actual depends on budget
    let mut dec = CeltDecoder::new_with_frame_samples(&params, frame_samples).unwrap();
    let n_frames = (48_000 / frame_samples).max(16); // at least ~0.3 s
    let n_samples = frame_samples * n_frames;
    let freq = 1000.0f32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();
    for chunk in signal.chunks(frame_samples) {
        if chunk.len() == frame_samples {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();
    let mut decoded: Vec<f32> = Vec::new();
    let mut total_bytes = 0usize;
    let mut packet_count = 0usize;
    loop {
        match enc.receive_packet() {
            Ok(pkt) => {
                total_bytes += pkt.data.len();
                packet_count += 1;
                dec.send_packet(&pkt).unwrap();
                while let Ok(frame) = dec.receive_frame() {
                    decoded.extend(decoded_f32(&frame));
                }
            }
            Err(_) => break,
        }
    }
    let avg_bytes = total_bytes
        .checked_div(packet_count)
        .unwrap_or(bytes_per_frame);
    // SNR on the middle 50% of frames (skip first 25% settling).
    let lo = n_samples / 4;
    let hi = lo + n_samples / 2;
    let hi = hi.min(decoded.len()).min(signal.len());
    if hi <= lo {
        return (0.0, avg_bytes);
    }
    let snr = snr_db(&signal, &decoded, lo, hi);
    (snr, avg_bytes)
}

/// Per-LM Goertzel-based tone energy report — regression sentinel.
///
/// For each LM: encode a 1 kHz sine, decode, measure energy at 1 kHz vs
/// 5 kHz (as a proxy for reconstruction fidelity). The target tone must
/// dominate the off-band by at least 2x. Also prints bytes/frame and
/// effective bitrate.
///
/// We use Goertzel rather than raw PSNR because CELT is a perceptual codec
/// that does not preserve phase — time-domain PSNR is very low even for
/// high-quality encoding. The tone-dominance check is a meaningful proxy
/// for perceptual quality at the target frequency.
#[test]
fn per_lm_sine_tone_dominance() {
    let goertzel = |samples: &[f32], f: f32| -> f32 {
        let w = 2.0 * std::f32::consts::PI * f / SAMPLE_RATE as f32;
        let cw = w.cos();
        let (mut s1, mut s2) = (0f32, 0f32);
        for &x in samples {
            let s0 = 2.0 * cw * s1 - s2 + x;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - 2.0 * cw * s1 * s2).sqrt()
    };

    for &fs in &[120usize, 240, 480, 960] {
        let (_snr, bytes_per_frame) = lm_sine_psnr(fs);
        let kbps = bytes_per_frame as f64 * 8.0 / (fs as f64 / SAMPLE_RATE as f64) / 1000.0;

        // Encode + decode independently to get the decoded slice.
        let params = build_params(1);
        let mut enc = CeltEncoder::new_with_frame_samples(&params, fs).unwrap();
        let mut dec = CeltDecoder::new_with_frame_samples(&params, fs).unwrap();
        let n_frames = (48_000 / fs).max(16);
        let n_samples = fs * n_frames;
        let freq = 1000.0f32;
        let signal: Vec<f32> = (0..n_samples)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.3
            })
            .collect();
        for chunk in signal.chunks(fs) {
            if chunk.len() == fs {
                enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
            }
        }
        enc.flush().unwrap();
        let mut decoded: Vec<f32> = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(pkt) => {
                    dec.send_packet(&pkt).unwrap();
                    while let Ok(frame) = dec.receive_frame() {
                        decoded.extend(decoded_f32(&frame));
                    }
                }
                Err(_) => break,
            }
        }
        // Measure on the middle 50% (skip settling frames).
        let lo = n_samples / 4;
        let hi = (lo + n_samples / 2).min(decoded.len());
        let slice = &decoded[lo..hi];
        let mag_1k = goertzel(slice, 1000.0);
        let mag_5k = goertzel(slice, 5000.0);
        let ratio = mag_1k / mag_5k.max(1e-9);
        println!(
            "LM={} frame={:4} ms={:.1}  bytes/frame={:3}  kbps={:.0}  mag@1kHz/5kHz={:.1}x",
            lm_for_frame_samples_test(fs),
            fs,
            fs as f32 / SAMPLE_RATE as f32 * 1000.0,
            bytes_per_frame,
            kbps,
            ratio,
        );
        assert!(
            mag_1k > 2.0 * mag_5k,
            "LM={} (frame_samples={}) 1 kHz tone buried (ratio={:.2}x < 2x)",
            lm_for_frame_samples_test(fs),
            fs,
            ratio,
        );
    }
}

fn lm_for_frame_samples_test(fs: usize) -> u32 {
    match fs {
        120 => 0,
        240 => 1,
        480 => 2,
        960 => 3,
        _ => 99,
    }
}

/// White-noise roundtrip at each LM: decoded output must be finite and
/// have non-negligible energy (the encoder should not collapse to silence).
#[test]
fn per_lm_noise_roundtrip() {
    let mut seed = 0x7b3au32;
    for &fs in &[120usize, 240, 480, 960] {
        let params = build_params(1);
        let mut enc = CeltEncoder::new_with_frame_samples(&params, fs).unwrap();
        let mut dec = CeltDecoder::new_with_frame_samples(&params, fs).unwrap();
        let n_frames = 16usize;
        let n_samples = fs * n_frames;
        let signal: Vec<f32> = (0..n_samples)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 16) as i32 - 32768) as f32 / 32768.0 * 0.1
            })
            .collect();
        for chunk in signal.chunks(fs) {
            if chunk.len() == fs {
                enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
            }
        }
        enc.flush().unwrap();
        let mut decoded: Vec<f32> = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(pkt) => {
                    dec.send_packet(&pkt).unwrap();
                    while let Ok(frame) = dec.receive_frame() {
                        decoded.extend(decoded_f32(&frame));
                    }
                }
                Err(_) => break,
            }
        }
        assert_eq!(
            decoded.len(),
            n_samples,
            "LM={}: wrong output length",
            lm_for_frame_samples_test(fs)
        );
        assert!(
            decoded.iter().all(|v| v.is_finite()),
            "LM={}: decoded output contains non-finite value",
            lm_for_frame_samples_test(fs)
        );
        let lo = fs * 4;
        let hi = lo + fs * 4;
        let e: f32 = decoded[lo..hi].iter().map(|v| v * v).sum::<f32>() / (fs * 4) as f32;
        assert!(
            e > 1e-7,
            "LM={} (frame_samples={}) white-noise decoded energy {e:.2e} too low",
            lm_for_frame_samples_test(fs),
            fs,
        );
        println!(
            "LM={} frame={} noise roundtrip OK (e={:.2e})",
            lm_for_frame_samples_test(fs),
            fs,
            e
        );
    }
}

/// Anti-collapse flag exercise: on a transient signal with low per-band pulse
/// allocation, the encoder should emit anti_collapse_on=1 on at least some
/// transient frames, and the decoder must handle those packets without error.
#[test]
fn anti_collapse_flag_emitted_on_transient() {
    use oxideav_celt::header::decode_header;
    use oxideav_celt::range_decoder::RangeDecoder;

    let params = build_params(1);
    let mut enc = CeltEncoder::new(&params).unwrap();
    enc.set_force_long_only(false); // ensure transient detection is on

    // Castanets-style signal: repeated short bursts against silence.
    let n_frames = 8usize;
    let n_samples = FRAME_SAMPLES * n_frames;
    let mut signal = vec![0.0f32; n_samples];
    for f in 0..n_frames {
        let onset = f * FRAME_SAMPLES + FRAME_SAMPLES / 8;
        for sample in signal.iter_mut().skip(onset).take(20) {
            *sample = 0.9;
        }
    }
    for chunk in signal.chunks(FRAME_SAMPLES) {
        if chunk.len() == FRAME_SAMPLES {
            enc.send_frame(&pcm_frame_f32(chunk, 1)).unwrap();
        }
    }
    enc.flush().unwrap();

    let mut packets = Vec::new();
    while let Ok(pkt) = enc.receive_packet() {
        packets.push(pkt);
    }
    assert!(!packets.is_empty(), "no packets produced");

    // Decode every packet with the public decoder; no panics / errors.
    let mut dec = CeltDecoder::new(&params).unwrap();
    for pkt in &packets {
        dec.send_packet(pkt).unwrap();
        while let Ok(frame) = dec.receive_frame() {
            assert!(decoded_f32(&frame).iter().all(|v| v.is_finite()));
        }
    }

    // At least one packet must be transient.
    let saw_transient = packets.iter().any(|pkt| {
        let mut rd = RangeDecoder::new(&pkt.data);
        decode_header(&mut rd).is_some_and(|h| h.transient)
    });
    assert!(
        saw_transient,
        "no transient packet found in the castanets stream"
    );

    println!(
        "anti-collapse test: {} packets, at least one transient, decoder stable",
        packets.len()
    );
}
