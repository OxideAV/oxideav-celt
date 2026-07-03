//! End-to-end **stereo** PCM codec-loop integration tests (RFC 6716
//! §4.3.7.2 / §4.3.7 / §5.3 per channel, Table 56 dual-stereo wire):
//! the self-contained `encode_stereo_celt_frame_pcm_auto` →
//! `decode_stereo_frame_auto` loop over interleaved L/R/L/R PCM, with
//! no out-of-band data — the stereo mirror of `pcm_codec_loop.rs`.

use oxideav_celt::coarse_energy::NUM_BANDS;
use oxideav_celt::frame_header::CeltFrameHeader;
use oxideav_celt::frame_synthesis::StereoCeltDecodeState;
use oxideav_celt::pcm_encode::{
    encode_stereo_celt_frame_pcm, encode_stereo_celt_frame_pcm_auto, StereoCeltEncodeState,
};

fn plain_header(intra: bool) -> CeltFrameHeader {
    CeltFrameHeader {
        silence: false,
        post_filter: None,
        transient: false,
        intra,
        anti_collapse_on: None,
    }
}

/// Deterministic pseudo-random values in [-0.5, 0.5).
fn test_signal(len: usize, mut seed: u32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// **Full quantized stereo PCM → bytes → PCM loop at every LM.** A
/// multi-frame interleaved stream encodes to fixed-size dual-stereo
/// frames and decodes caller-input-free: PCM is finite, the frame
/// selects dual stereo on the wire, both channels' coarse prediction
/// stays in exact encoder/decoder lockstep, and the decoded PCM
/// equals the decode-side synthesis of the encoder's bit-exact
/// per-channel reconstructions.
#[test]
fn stereo_quantized_codec_loop_all_lm() {
    for lm in 0..=3u32 {
        let frame_bytes = 120u32;
        let frames = 3usize;
        let mut enc = StereoCeltEncodeState::new(lm).unwrap();
        let mut dec = StereoCeltDecodeState::new(lm).unwrap();
        // Parallel reference back end fed the encoder's own
        // reconstructions (post-filter off, so the chains coincide).
        let mut reference = StereoCeltDecodeState::new(lm).unwrap();
        let n = enc.frame_size();
        assert_eq!(n, 120usize << lm);

        let input = test_signal(frames * 2 * n, 0x57E0 + lm);
        for t in 0..frames {
            let header = plain_header(t == 0);
            let frame = encode_stereo_celt_frame_pcm_auto(
                &mut enc,
                &input[t * 2 * n..(t + 1) * 2 * n],
                &header,
                frame_bytes,
                0,
                NUM_BANDS,
            )
            .unwrap();
            assert_eq!(frame.bytes.len(), frame_bytes as usize);
            assert!(frame.prefix.allocation.dual_stereo, "lm={lm} frame {t}");

            let decoded = dec
                .decode_stereo_frame_auto(&frame.bytes, 0, NUM_BANDS)
                .unwrap();
            assert_eq!(decoded.pcm.len(), 2 * n);
            assert!(decoded.pcm.iter().all(|s| s.is_finite()));
            assert_eq!(
                enc.coarse_energy().energy,
                dec.coarse_energy().energy,
                "lm={lm} frame {t}: stereo coarse lockstep broken"
            );
            assert_eq!(frame.envelope_q8, decoded.envelope_q8);

            let expected = reference
                .synthesize_stereo_frame(
                    &frame.reconstructed_left,
                    &frame.reconstructed_right,
                    0,
                    NUM_BANDS,
                    None,
                )
                .unwrap();
            for (i, (e, d)) in expected.iter().zip(&decoded.pcm).enumerate() {
                assert!(
                    (e - d).abs() <= 1e-5,
                    "lm={lm} frame {t} sample {i}: decoder {d} vs reconstruction chain {e}"
                );
            }
        }
    }
}

/// **Stereo loop fidelity.** For a low-band tonal stereo signal with
/// distinct L/R content, each decoded channel tracks its own delayed
/// input: steady-state relative L2 error well below one and strongly
/// positive per-channel correlation — the loop transports both
/// waveforms, and the channels stay unswapped.
#[test]
fn stereo_quantized_loop_tracks_tonal_input_per_channel() {
    let lm = 3u32;
    let n = 120usize << lm;
    let frame_bytes = 220u32;
    let frames = 6usize;

    // Distinct low-frequency tones per channel.
    let mut input = Vec::with_capacity(frames * 2 * n);
    for i in 0..frames * n {
        let t = i as f32;
        input.push(0.6 * (0.02 * t).sin() + 0.2 * (0.05 * t + 0.3).sin()); // L
        input.push(0.5 * (0.031 * t + 1.1).sin() + 0.25 * (0.011 * t).sin()); // R
    }

    let mut enc = StereoCeltEncodeState::new(lm).unwrap();
    let mut dec = StereoCeltDecodeState::new(lm).unwrap();
    let mut output = Vec::with_capacity(frames * 2 * n);
    for t in 0..frames {
        let header = plain_header(t == 0);
        let frame = encode_stereo_celt_frame_pcm_auto(
            &mut enc,
            &input[t * 2 * n..(t + 1) * 2 * n],
            &header,
            frame_bytes,
            0,
            NUM_BANDS,
        )
        .unwrap();
        let decoded = dec
            .decode_stereo_frame_auto(&frame.bytes, 0, NUM_BANDS)
            .unwrap();
        output.extend_from_slice(&decoded.pcm);
    }

    // One frame (2n interleaved samples) of delay; steady state skips
    // the first two recovered frames and the last in-flight frame.
    for ch in 0..2usize {
        let mut err2 = 0.0f64;
        let mut sig2 = 0.0f64;
        let mut dot = 0.0f64;
        let mut out2 = 0.0f64;
        for i in (2 * n)..((frames - 1) * n) {
            let want = input[2 * i + ch] as f64;
            let got = output[2 * (n + i) + ch] as f64;
            err2 += (got - want) * (got - want);
            sig2 += want * want;
            dot += got * want;
            out2 += got * got;
        }
        let rel = (err2 / sig2).sqrt();
        let corr = dot / (sig2.sqrt() * out2.sqrt());
        assert!(
            rel < 0.8,
            "channel {ch}: steady-state relative L2 error too high: {rel}"
        );
        assert!(corr > 0.6, "channel {ch}: correlation too low: {corr}");
    }
}

/// **Stereo silence run.** After a loud frame, silence-flagged frames
/// over silent input carry only the prefix; the decoder plays out
/// both channels' overlap tails toward numerical silence while the
/// stereo coarse prediction stays in lockstep throughout.
#[test]
fn stereo_silence_run_decays_and_stays_in_lockstep() {
    let lm = 1u32;
    let mut enc = StereoCeltEncodeState::new(lm).unwrap();
    let mut dec = StereoCeltDecodeState::new(lm).unwrap();
    let n = enc.frame_size();
    let frame_bytes = 96u32;

    let loud = test_signal(2 * n, 0x10AD);
    let frame =
        encode_stereo_celt_frame_pcm_auto(&mut enc, &loud, &plain_header(true), frame_bytes, 0, 21)
            .unwrap();
    let decoded = dec.decode_stereo_frame_auto(&frame.bytes, 0, 21).unwrap();
    assert!(decoded.pcm.iter().any(|&s| s != 0.0));

    let silent_header = CeltFrameHeader {
        silence: true,
        ..plain_header(false)
    };
    let zeros = vec![0.0f32; 2 * n];
    let mut peak = f32::MAX;
    for t in 0..4 {
        let frame =
            encode_stereo_celt_frame_pcm_auto(&mut enc, &zeros, &silent_header, frame_bytes, 0, 21)
                .unwrap();
        assert!(frame.prefix.header.silence);
        assert!(frame.reconstructed_left.iter().all(|&s| s == 0.0));
        assert!(frame.reconstructed_right.iter().all(|&s| s == 0.0));
        let decoded = dec.decode_stereo_frame_auto(&frame.bytes, 0, 21).unwrap();
        assert_eq!(
            enc.coarse_energy().energy,
            dec.coarse_energy().energy,
            "silence frame {t}: stereo coarse lockstep broken"
        );
        let frame_peak = decoded
            .pcm
            .iter()
            .fold(0.0f32, |m, &v| if v.abs() > m { v.abs() } else { m });
        assert!(
            frame_peak < peak,
            "silence frame {t}: output must keep decaying ({frame_peak} vs {peak})"
        );
        peak = frame_peak;
    }
    assert!(peak < 1e-2, "stereo silence run did not decay: peak {peak}");
}

/// Byte-level determinism + `reset()`: the same interleaved stream
/// through two fresh states produces identical frames, stream memory
/// matters, and a reset restores fresh-stream bytes.
#[test]
fn stereo_pcm_encode_deterministic_and_resettable() {
    let lm = 2u32;
    let frame_bytes = 110u32;
    let mut a = StereoCeltEncodeState::new(lm).unwrap();
    let mut b = StereoCeltEncodeState::new(lm).unwrap();
    assert_eq!(a.lm(), lm);
    let n = a.frame_size();
    let header = plain_header(true);

    let pcm = test_signal(2 * n, 0xD00D);
    let fa = encode_stereo_celt_frame_pcm_auto(&mut a, &pcm, &header, frame_bytes, 0, 21).unwrap();
    let fb = encode_stereo_celt_frame_pcm_auto(&mut b, &pcm, &header, frame_bytes, 0, 21).unwrap();
    assert_eq!(fa.bytes, fb.bytes);
    assert_eq!(fa.envelope_q8, fb.envelope_q8);

    // Stream memory influences the second frame; reset restores the
    // fresh first-frame bytes.
    let second =
        encode_stereo_celt_frame_pcm_auto(&mut a, &pcm, &header, frame_bytes, 0, 21).unwrap();
    assert_ne!(fa.bytes, second.bytes, "stream memory must matter");
    a.reset();
    let after_reset =
        encode_stereo_celt_frame_pcm_auto(&mut a, &pcm, &header, frame_bytes, 0, 21).unwrap();
    assert_eq!(fa.bytes, after_reset.bytes);
}

/// A narrowed coded-band window (`end < 21`, the reduced-bandwidth
/// configuration) closes the same self-contained loop: dual selector
/// on the wire, coarse lockstep, finite PCM, and determinism.
#[test]
fn stereo_loop_narrow_band_window() {
    let lm = 2u32;
    let frame_bytes = 100u32;
    let end = 17usize; // narrower audio bandwidth: bands 0..17
    let mut enc = StereoCeltEncodeState::new(lm).unwrap();
    let mut dec = StereoCeltDecodeState::new(lm).unwrap();
    let n = enc.frame_size();

    for t in 0..3u32 {
        let pcm = test_signal(2 * n, 0x0B0E + t);
        let header = plain_header(t == 0);
        let frame = encode_stereo_celt_frame_pcm_auto(&mut enc, &pcm, &header, frame_bytes, 0, end)
            .unwrap();
        assert!(frame.prefix.allocation.dual_stereo, "frame {t}");
        assert_eq!(frame.prefix.allocation.intensity_band_offset, end as u32);
        assert_eq!(frame.prefix.end, end);
        let decoded = dec.decode_stereo_frame_auto(&frame.bytes, 0, end).unwrap();
        assert_eq!(decoded.pcm.len(), 2 * n);
        assert!(decoded.pcm.iter().all(|s| s.is_finite()));
        assert_eq!(
            enc.coarse_energy().energy,
            dec.coarse_energy().energy,
            "frame {t}: narrow-window coarse lockstep broken"
        );
    }
}

/// A longer mixed stream — sound, a silence run, sound again, with
/// the intra flag only on the opener — stays in lockstep end to end
/// and re-acquires non-silent output after the silence run (the
/// §4.3.2.1 prediction survives the silence frames on both sides).
#[test]
fn stereo_loop_mixed_sound_silence_stream() {
    let lm = 1u32;
    let frame_bytes = 110u32;
    let mut enc = StereoCeltEncodeState::new(lm).unwrap();
    let mut dec = StereoCeltDecodeState::new(lm).unwrap();
    let n = enc.frame_size();

    // Schedule: 3 sound, 3 silence, 3 sound.
    let mut last_peak = 0.0f32;
    for t in 0..9usize {
        let silent = (3..6).contains(&t);
        let header = CeltFrameHeader {
            silence: silent,
            ..plain_header(t == 0)
        };
        let pcm = if silent {
            vec![0.0f32; 2 * n]
        } else {
            test_signal(2 * n, 0x9000 + t as u32)
        };
        let frame =
            encode_stereo_celt_frame_pcm_auto(&mut enc, &pcm, &header, frame_bytes, 0, 21).unwrap();
        let decoded = dec.decode_stereo_frame_auto(&frame.bytes, 0, 21).unwrap();
        assert_eq!(
            enc.coarse_energy().energy,
            dec.coarse_energy().energy,
            "frame {t}: lockstep broken across the sound/silence schedule"
        );
        last_peak = decoded
            .pcm
            .iter()
            .fold(0.0f32, |m, &v| if v.abs() > m { v.abs() } else { m });
    }
    // The final sound frames re-acquired non-silent output.
    assert!(
        last_peak > 1e-4,
        "stream did not re-acquire sound after the silence run ({last_peak})"
    );
}

/// Rejections leave the streaming front end untouched, and the
/// explicit-allocation variant round-trips through
/// `decode_stereo_frame_coded`.
#[test]
fn stereo_pcm_rejections_and_explicit_variant() {
    let lm = 1u32;
    let frame_bytes = 100u32;
    let mut state = StereoCeltEncodeState::new(lm).unwrap();
    let n = state.frame_size();
    let pcm = test_signal(2 * n, 0x777);
    let good = plain_header(true);

    // Wrong PCM length.
    assert!(encode_stereo_celt_frame_pcm_auto(
        &mut state,
        &pcm[..2 * n - 1],
        &good,
        frame_bytes,
        0,
        21
    )
    .is_err());
    // Transient header.
    let transient = CeltFrameHeader {
        transient: true,
        ..good
    };
    assert!(
        encode_stereo_celt_frame_pcm_auto(&mut state, &pcm, &transient, frame_bytes, 0, 21)
            .is_err()
    );
    // Post-filter signalled.
    let post_filter = CeltFrameHeader {
        post_filter: Some(oxideav_celt::frame_header::PostFilter {
            octave: 2,
            period: 100,
            gain: 3,
            tapset: 1,
        }),
        ..good
    };
    assert!(
        encode_stereo_celt_frame_pcm_auto(&mut state, &pcm, &post_filter, frame_bytes, 0, 21)
            .is_err()
    );
    // A tiny budget cannot carry the dual selector.
    assert!(encode_stereo_celt_frame_pcm_auto(&mut state, &pcm, &good, 3, 0, 21).is_err());

    // After the rejections the state still behaves fresh.
    let after =
        encode_stereo_celt_frame_pcm_auto(&mut state, &pcm, &good, frame_bytes, 0, 21).unwrap();
    let mut fresh = StereoCeltEncodeState::new(lm).unwrap();
    let expected =
        encode_stereo_celt_frame_pcm_auto(&mut fresh, &pcm, &good, frame_bytes, 0, 21).unwrap();
    assert_eq!(after.bytes, expected.bytes);

    // Explicit allocations round-trip through the coded decoder.
    let mut fine_bits = [0u32; NUM_BANDS];
    fine_bits[..4].fill(1);
    let mut band_k = vec![0u32; 21];
    band_k[..8].fill(2);
    let mut enc2 = StereoCeltEncodeState::new(lm).unwrap();
    let frame = encode_stereo_celt_frame_pcm(
        &mut enc2,
        &pcm,
        &good,
        frame_bytes,
        0,
        21,
        &fine_bits,
        &band_k,
    )
    .unwrap();
    let mut dec = StereoCeltDecodeState::new(lm).unwrap();
    let decoded = dec
        .decode_stereo_frame_coded(&frame.bytes, 0, 21, &fine_bits, &band_k)
        .unwrap();
    assert!(decoded.pcm.iter().all(|s| s.is_finite()));
    assert_eq!(enc2.coarse_energy().energy, dec.coarse_energy().energy);
    assert_eq!(frame.envelope_q8, decoded.envelope_q8);
}
