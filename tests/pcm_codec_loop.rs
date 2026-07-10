//! End-to-end PCM codec-loop integration tests (RFC 6716 §4.3.7.2 /
//! §4.3.7 / §5.3): the encode front end (pre-emphasis → windowed
//! forward MDCT → coded-window extraction) against the decode back end
//! (spectrum placement → IMDCT + WOLA → de-emphasis), first without
//! quantization (the exact inverse-pair identity), then through the
//! full quantized `encode_celt_frame_pcm_auto` →
//! `decode_celt_frame_auto` loop.

use oxideav_celt::analysis::LongMdctAnalysis;
use oxideav_celt::coarse_energy::NUM_BANDS;
use oxideav_celt::deemphasis::{Deemphasis, Preemphasis};
use oxideav_celt::derive_pulses::decode_celt_frame_auto;
use oxideav_celt::frame_header::CeltFrameHeader;
use oxideav_celt::frame_synthesis::CeltDecodeState;
use oxideav_celt::pcm_encode::{encode_celt_frame_pcm_auto, CeltEncodeState};
use oxideav_celt::synthesis::LongMdctSynthesis;

fn plain_header(intra: bool) -> CeltFrameHeader {
    CeltFrameHeader {
        silence: false,
        post_filter: None,
        transient: false,
        intra,
        anti_collapse_on: None,
    }
}

/// Deterministic pseudo-random values in [-1, 1).
fn test_signal(len: usize, mut seed: u32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

/// **Unquantized inverse-pair identity.** A band-limited PCM stream
/// (synthesized from known coded-window spectra, then de-emphasized so
/// the §4.3.7.2 pair is part of the loop) runs through the encode
/// front end and the decode back end with **no quantization** in
/// between: pre-emphasis → analysis → [identity] → synthesis →
/// de-emphasis. In steady state the output reproduces the input
/// exactly, delayed by one frame — every front-end stage is the exact
/// inverse of its back-end mirror.
#[test]
fn unquantized_front_end_back_end_identity() {
    for lm in 0..=3u32 {
        let n = 120usize << lm;
        let frames = 5usize;

        // Build a band-limited input: synthesize PCM from random
        // coded-window spectra, then de-emphasize (so pre-emphasis
        // recovers the synthesized stream bit-for-bit up to f32
        // rounding).
        let mut gen_synth = LongMdctSynthesis::new(lm).unwrap();
        let mut gen_deemph = Deemphasis::new();
        let mut input = Vec::with_capacity(frames * n);
        for t in 0..frames {
            let spec_len = 100usize << lm;
            let spec = test_signal(spec_len, 0x600D + lm * 131 + t as u32);
            let mut pcm = gen_synth.synthesize(&spec, 0, NUM_BANDS).unwrap();
            gen_deemph.apply_in_place(&mut pcm);
            input.extend_from_slice(&pcm);
        }

        // Encode front end → (no quantization) → decode back end.
        let mut preemph = Preemphasis::new();
        let mut analysis = LongMdctAnalysis::new(lm).unwrap();
        let mut synth = LongMdctSynthesis::new(lm).unwrap();
        let mut deemph = Deemphasis::new();
        let mut output = Vec::with_capacity(frames * n);
        for t in 0..frames {
            let mut frame = input[t * n..(t + 1) * n].to_vec();
            preemph.apply_in_place(&mut frame);
            let spectrum = analysis.analyze(&frame, 0, NUM_BANDS).unwrap();
            let mut out = synth.synthesize(&spectrum, 0, NUM_BANDS).unwrap();
            deemph.apply_in_place(&mut out);
            output.extend_from_slice(&out);
        }

        // One frame of delay; steady state from the second recovered
        // frame on (the first recovered frame folded a zero history
        // against a zero tail).
        let scale = input
            .iter()
            .fold(0.0f32, |m, &v| if v.abs() > m { v.abs() } else { m })
            .max(1.0);
        for i in n..(frames - 1) * n {
            let got = output[n + i];
            let want = input[i];
            assert!(
                (got - want).abs() <= 3e-4 * scale,
                "lm={lm}: unquantized identity broken at sample {i}: {got} vs {want}"
            );
        }
    }
}

/// **Full quantized PCM → bytes → PCM loop.** The self-contained
/// encoder/decoder pair over a multi-frame stream: the decoded PCM is
/// finite and non-silent, the decoder reproduces the encoder's
/// bit-exact residual reconstruction (pinned indirectly: the decoded
/// PCM equals the decode-side synthesis chain run on the encoder's
/// `reconstructed_spectrum`), and both coarse states stay in lockstep.
#[test]
fn quantized_codec_loop_decoder_matches_encoder_reconstruction() {
    let lm = 2u32;
    let n = 120usize << lm;
    let frame_bytes = 120u32;
    let frames = 4usize;

    let mut enc = CeltEncodeState::new(lm).unwrap();
    let mut dec = CeltDecodeState::new(lm).unwrap();
    // Reference decode chain fed with the encoder's own bit-exact
    // reconstruction of the residual spectrum.
    let mut ref_synth = LongMdctSynthesis::new(lm).unwrap();
    let mut ref_deemph = Deemphasis::new();

    let input = test_signal(frames * n, 0xFA57);
    for t in 0..frames {
        let header = plain_header(t == 0);
        let frame = encode_celt_frame_pcm_auto(
            &mut enc,
            &input[t * n..(t + 1) * n],
            &header,
            frame_bytes,
            0,
            NUM_BANDS,
        )
        .unwrap();
        let decoded = decode_celt_frame_auto(&mut dec, &frame.bytes, 0, NUM_BANDS).unwrap();
        assert!(decoded.pcm.iter().all(|s| s.is_finite()));

        // The decoder's PCM must equal the §4.3.6→§4.3.7→§4.3.7.2
        // chain run on the encoder's own reconstruction (the residual
        // decode is bit-exact, the post-filter is off, and both
        // de-emphasis memories started at zero).
        let mut expected = ref_synth
            .synthesize(&frame.reconstructed_spectrum, 0, NUM_BANDS)
            .unwrap();
        ref_deemph.apply_in_place(&mut expected);
        assert_eq!(expected.len(), decoded.pcm.len());
        for (i, (e, d)) in expected.iter().zip(&decoded.pcm).enumerate() {
            assert!(
                (e - d).abs() <= 1e-5,
                "frame {t} sample {i}: decoder {d} vs encoder-reconstruction chain {e}"
            );
        }

        assert_eq!(
            enc.coarse_energy().energy,
            dec.coarse_energy().energy,
            "frame {t}: coarse lockstep broken"
        );
    }
}

/// **Quantized loop fidelity.** For a low-band-concentrated tonal
/// signal (where the allocation buys plenty of pulses) the decoded
/// output tracks the delayed input: the steady-state relative L2
/// error stays far below one (an all-zero output would score exactly
/// one) and the normalized correlation is near-unity. The r393
/// derived fine/shape split funds per-band fine-energy refinement, so
/// the envelope error no longer dominates the loop.
#[test]
fn quantized_codec_loop_tracks_tonal_input() {
    let lm = 3u32;
    let n = 120usize << lm;
    let frame_bytes = 160u32;
    let frames = 6usize;

    // A two-tone low-frequency signal (bins land in the well-funded
    // low bands) with a touch of amplitude variation.
    let input: Vec<f32> = (0..frames * n)
        .map(|i| {
            let t = i as f32;
            0.6 * (0.02 * t).sin() + 0.25 * (0.055 * t + 0.7).sin()
        })
        .collect();

    let mut enc = CeltEncodeState::new(lm).unwrap();
    let mut dec = CeltDecodeState::new(lm).unwrap();
    let mut output = Vec::with_capacity(frames * n);
    for t in 0..frames {
        let header = plain_header(t == 0);
        let frame = encode_celt_frame_pcm_auto(
            &mut enc,
            &input[t * n..(t + 1) * n],
            &header,
            frame_bytes,
            0,
            NUM_BANDS,
        )
        .unwrap();
        let decoded = decode_celt_frame_auto(&mut dec, &frame.bytes, 0, NUM_BANDS).unwrap();
        output.extend_from_slice(&decoded.pcm);
    }

    // Compare output (delayed one frame) to input over the steady
    // state, skipping the first two recovered frames (stream-open
    // transients) and the last input frame (still in the pipeline).
    let lo = 2 * n;
    let hi = (frames - 1) * n;
    let mut err2 = 0.0f64;
    let mut sig2 = 0.0f64;
    let mut dot = 0.0f64;
    let mut out2 = 0.0f64;
    for i in lo..hi {
        let want = input[i] as f64;
        let got = output[n + i] as f64;
        err2 += (got - want) * (got - want);
        sig2 += want * want;
        dot += got * want;
        out2 += got * got;
    }
    let rel = (err2 / sig2).sqrt();
    let corr = dot / (sig2.sqrt() * out2.sqrt());
    // Thresholds tightened after the r393 derived fine/shape split
    // landed (measured: rel ~0.039, corr ~0.9992 at lm=3/160B; ~2.5x
    // headroom retained).
    assert!(
        rel < 0.1,
        "steady-state relative L2 error too high: {rel} (all-zero output scores 1.0)"
    );
    assert!(corr > 0.99, "normalized correlation too low: {corr}");
}

/// **Re-analysis closes the loop on the spectrum axis.** The decoded
/// PCM is (by the second test) the synthesis of the encoder's
/// bit-exact reconstructed residual spectra — so running the encode
/// front end (pre-emphasis + windowed forward MDCT) over the decoded
/// PCM must recover those reconstructed spectra, one frame delayed, in
/// steady state. This pins the whole PCM loop to the quantized
/// envelope + shape for every band the allocation funded, with no
/// hand-tuned per-band tolerance.
#[test]
fn quantized_loop_reanalysis_recovers_reconstruction() {
    let lm = 2u32;
    let n = 120usize << lm;
    let frame_bytes = 120u32;
    let frames = 5usize;

    let input = test_signal(frames * n, 0xE55E);
    let mut enc = CeltEncodeState::new(lm).unwrap();
    let mut dec = CeltDecodeState::new(lm).unwrap();

    // Re-analysis chain over the decoded output (the §4.3.7.2 +
    // §4.3.7 front end again).
    let mut re_preemph = Preemphasis::new();
    let mut re_analysis = LongMdctAnalysis::new(lm).unwrap();

    let mut reconstructed: Vec<Vec<f32>> = Vec::new();
    let mut re_spectra: Vec<Vec<f32>> = Vec::new();
    for t in 0..frames {
        let header = plain_header(t == 0);
        let frame = encode_celt_frame_pcm_auto(
            &mut enc,
            &input[t * n..(t + 1) * n],
            &header,
            frame_bytes,
            0,
            NUM_BANDS,
        )
        .unwrap();
        let decoded = decode_celt_frame_auto(&mut dec, &frame.bytes, 0, NUM_BANDS).unwrap();
        reconstructed.push(frame.reconstructed_spectrum.clone());
        let mut re_in = decoded.pcm.clone();
        re_preemph.apply_in_place(&mut re_in);
        re_spectra.push(re_analysis.analyze(&re_in, 0, NUM_BANDS).unwrap());
    }

    // Tolerance scales with the spectrum magnitude (the analysis
    // round-trip is exact to ~2e-4 relative; see the analysis module
    // tests).
    let max_abs =
        reconstructed
            .iter()
            .flatten()
            .fold(0.0f32, |m, &v| if v.abs() > m { v.abs() } else { m });
    let tol = 5e-4 * max_abs.max(1e-3);

    // Re-analysis frame t sees decoded block t-1 (one frame of
    // delay); steady state from t = 2.
    for t in 2..frames {
        let want = &reconstructed[t - 1];
        let got = &re_spectra[t];
        assert_eq!(want.len(), got.len());
        for k in 0..want.len() {
            assert!(
                (want[k] - got[k]).abs() <= tol,
                "frame {} bin {k}: re-analyzed {} vs encoder reconstruction {} (tol {tol})",
                t - 1,
                got[k],
                want[k]
            );
        }
    }
}

/// **Budget fit is rigorous across byte budgets and frame sizes.**
/// The §4.3.4.1 derivation prices shape symbols from the bit-exact
/// pulse cache (`ceil(8*log2 V(N, K))` in its exact-identity region)
/// while the measured §5.1.4 `enc_uint` wire cost can exceed that
/// price by one eighth-bit per symbol; the derivation provisions one
/// eighth-bit per coded PVQ symbol so every frame provably fits its
/// byte budget. This sweep drives the full PCM loop across every LM
/// and a ladder of byte budgets: the encoder must always produce a
/// frame of exactly the requested size and the decoder must accept
/// it.
#[test]
fn quantized_loop_fits_across_byte_budgets() {
    for lm in 0..=3u32 {
        let n = 120usize << lm;
        for &frame_bytes in &[20u32, 32, 48, 64, 96, 128, 160, 200] {
            let input = test_signal(3 * n, 0xB1D5 + lm * 977 + frame_bytes);
            let mut enc = CeltEncodeState::new(lm).unwrap();
            let mut dec = CeltDecodeState::new(lm).unwrap();
            for t in 0..3 {
                let header = plain_header(t == 0);
                let frame = encode_celt_frame_pcm_auto(
                    &mut enc,
                    &input[t * n..(t + 1) * n],
                    &header,
                    frame_bytes,
                    0,
                    NUM_BANDS,
                )
                .unwrap_or_else(|e| {
                    panic!("lm={lm} bytes={frame_bytes} frame {t}: encode failed: {e:?}")
                });
                assert_eq!(
                    frame.bytes.len(),
                    frame_bytes as usize,
                    "lm={lm} bytes={frame_bytes}: frame size mismatch"
                );
                let decoded = decode_celt_frame_auto(&mut dec, &frame.bytes, 0, NUM_BANDS)
                    .unwrap_or_else(|e| {
                        panic!("lm={lm} bytes={frame_bytes} frame {t}: decode failed: {e:?}")
                    });
                assert_eq!(decoded.pcm.len(), n);
                for &s in &decoded.pcm {
                    assert!(s.is_finite());
                }
            }
        }
    }
}

/// **§5.3.1 pitch pre-filter through the full PCM loop.** The encoder
/// runs the §5.3.1 pitch search over its input, signals the chosen
/// §4.3.7.1 parameters in the Table-56 prefix, and applies the pitch
/// pre-filter (the exact inverse of the decoder's post-filter) after
/// pre-emphasis; the decoder undoes it with the same signalled
/// parameters. The loop must stay self-contained (no out-of-band
/// data), track the waveform, and actually engage the filter.
#[test]
fn quantized_loop_with_pitch_postfilter() {
    use oxideav_celt::frame_header::PostFilter;
    use oxideav_celt::{choose_post_filter_params, pitch_search};

    let lm = 2u32;
    let n = 120usize << lm;
    let frame_bytes = 96u32;
    let frames = 6usize;

    // A strongly periodic tone (period ~97 samples) with a soft
    // second harmonic.
    let period_true = 97.0f32;
    let input: Vec<f32> = (0..frames * n)
        .map(|i| {
            let ph = 2.0 * std::f32::consts::PI * i as f32 / period_true;
            0.6 * ph.sin() + 0.15 * (2.0 * ph).sin()
        })
        .collect();

    let mut enc = CeltEncodeState::new(lm).unwrap();
    let mut dec = CeltDecodeState::new(lm).unwrap();
    let mut output = Vec::with_capacity(frames * n);
    let mut prev_period: Option<u16> = None;
    let mut engaged = 0usize;
    for t in 0..frames {
        // §5.3.1 pitch search over the input seen so far (the current
        // frame is the analysis window; earlier frames are the lag
        // history).
        let hist_end = (t + 1) * n;
        let est = pitch_search(&input[..hist_end], n, prev_period);
        let pf = est
            .and_then(choose_post_filter_params)
            .and_then(|p| PostFilter::from_period(p.period, p.gain_index, p.tapset));
        if pf.is_some() {
            engaged += 1;
            prev_period = pf.as_ref().map(|p| p.period);
        }
        let header = CeltFrameHeader {
            silence: false,
            post_filter: pf,
            transient: false,
            intra: t == 0,
            anti_collapse_on: None,
        };
        let frame = encode_celt_frame_pcm_auto(
            &mut enc,
            &input[t * n..(t + 1) * n],
            &header,
            frame_bytes,
            0,
            NUM_BANDS,
        )
        .unwrap();
        let decoded = decode_celt_frame_auto(&mut dec, &frame.bytes, 0, NUM_BANDS).unwrap();
        output.extend_from_slice(&decoded.pcm);
    }
    assert!(
        engaged >= frames - 1,
        "pitch search engaged the post-filter on only {engaged} frames"
    );

    // Steady-state fidelity, one frame of latency.
    let lo = 2 * n;
    let hi = (frames - 1) * n;
    let (mut err2, mut sig2, mut dot, mut out2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in lo..hi {
        let want = input[i] as f64;
        let got = output[n + i] as f64;
        err2 += (got - want) * (got - want);
        sig2 += want * want;
        dot += got * want;
        out2 += got * got;
    }
    let rel = (err2 / sig2).sqrt();
    let corr = dot / (sig2.sqrt() * out2.sqrt());
    assert!(
        rel < 0.15,
        "post-filtered loop error too high: {rel} (all-zero output scores 1.0)"
    );
    assert!(
        corr > 0.98,
        "post-filtered loop correlation too low: {corr}"
    );
}

/// **Transient (short-block) codec loop, bit-exact.** Mixed
/// long/transient multi-frame streams at every `LM`: the decoder's PCM
/// equals the decode-side chain run on the encoder's own bit-exact
/// residual reconstruction (the §4.3.1 short-block synthesis, §4.3.4.5
/// TF transform, and §4.3.5 anti-collapse bit all in lockstep), and
/// the coarse states match after every frame.
#[test]
fn transient_codec_loop_decoder_matches_encoder_reconstruction() {
    for lm in 0..=3u32 {
        let n = 120usize << lm;
        let frame_bytes = 60 + 30 * lm;
        let kinds = [false, true, true, false, true];
        let frames = kinds.len();

        let mut enc = CeltEncodeState::new(lm).unwrap();
        let mut dec = CeltDecodeState::new(lm).unwrap();
        let mut ref_synth = LongMdctSynthesis::new(lm).unwrap();
        let mut ref_deemph = Deemphasis::new();

        let input = test_signal(frames * n, 0x7A9 + lm);
        for (t, &transient) in kinds.iter().enumerate() {
            let header = CeltFrameHeader {
                transient,
                ..plain_header(t == 0)
            };
            let frame = encode_celt_frame_pcm_auto(
                &mut enc,
                &input[t * n..(t + 1) * n],
                &header,
                frame_bytes,
                0,
                NUM_BANDS,
            )
            .unwrap();
            let decoded = decode_celt_frame_auto(&mut dec, &frame.bytes, 0, NUM_BANDS).unwrap();
            assert_eq!(decoded.prefix.header.transient, transient, "lm={lm} t={t}");
            assert!(decoded.pcm.iter().all(|s| s.is_finite()));

            // Decoder PCM ≡ decode chain over the encoder's bit-exact
            // reconstruction (anti-collapse defaults off, so no
            // injection perturbs the identity).
            let mut expected = ref_synth
                .synthesize_frame(&frame.reconstructed_spectrum, 0, NUM_BANDS, transient)
                .unwrap();
            ref_deemph.apply_in_place(&mut expected);
            for (i, (e, d)) in expected.iter().zip(&decoded.pcm).enumerate() {
                assert!(
                    (e - d).abs() <= 1e-5,
                    "lm={lm} frame {t} (transient={transient}) sample {i}: {d} vs {e}"
                );
            }
            assert_eq!(
                enc.coarse_energy().energy,
                dec.coarse_energy().energy,
                "lm={lm} frame {t}: coarse lockstep broken"
            );
        }
    }
}

/// **Transient loop fidelity.** An all-transient tonal stream still
/// tracks the delayed input through the short-block analysis/synthesis
/// pair and the §4.3.4.5 TF transform round trip.
#[test]
fn transient_codec_loop_tracks_tonal_input() {
    let lm = 3u32;
    let n = 120usize << lm;
    let frame_bytes = 160u32;
    let frames = 6usize;

    let input: Vec<f32> = (0..frames * n)
        .map(|i| {
            let t = i as f32;
            0.6 * (0.02 * t).sin() + 0.25 * (0.055 * t + 0.7).sin()
        })
        .collect();

    let mut enc = CeltEncodeState::new(lm).unwrap();
    let mut dec = CeltDecodeState::new(lm).unwrap();
    let mut output = Vec::with_capacity(frames * n);
    for t in 0..frames {
        let header = CeltFrameHeader {
            transient: true,
            ..plain_header(t == 0)
        };
        let frame = encode_celt_frame_pcm_auto(
            &mut enc,
            &input[t * n..(t + 1) * n],
            &header,
            frame_bytes,
            0,
            NUM_BANDS,
        )
        .unwrap();
        let decoded = decode_celt_frame_auto(&mut dec, &frame.bytes, 0, NUM_BANDS).unwrap();
        output.extend_from_slice(&decoded.pcm);
    }

    let lo = 2 * n;
    let hi = (frames - 1) * n;
    let mut err2 = 0.0f64;
    let mut sig2 = 0.0f64;
    let mut dot = 0.0f64;
    let mut out2 = 0.0f64;
    for i in lo..hi {
        let want = input[i] as f64;
        let got = output[n + i] as f64;
        err2 += (got - want) * (got - want);
        sig2 += want * want;
        dot += got * want;
        out2 += got * got;
    }
    let rel = (err2 / sig2).sqrt();
    let corr = dot / (sig2.sqrt() * out2.sqrt());
    // Measured at lm=3/160B: rel ~0.044, corr ~0.9990 — on par with
    // the long-MDCT loop (~0.039); ~2x headroom retained.
    assert!(
        rel < 0.1,
        "transient steady-state relative L2 error too high: {rel}"
    );
    assert!(
        corr > 0.99,
        "transient normalized correlation too low: {corr}"
    );
}

/// **Anti-collapse wire round trip.** With `anti_collapse_on =
/// Some(true)` on transient frames the bit crosses the wire (the
/// decoded prefix reports it), the loop stays in coarse lockstep (the
/// bit is read exactly once at its Table-56 position), and the decoded
/// PCM stays finite whether or not any band actually collapsed.
#[test]
fn transient_codec_loop_carries_anti_collapse_bit() {
    let lm = 2u32; // LM > 1 so the §4.3.3 reservation is made
    let n = 120usize << lm;
    let frame_bytes = 100u32;
    let frames = 4usize;

    let input = test_signal(frames * n, 0xAC0);
    for on in [false, true] {
        let mut enc = CeltEncodeState::new(lm).unwrap();
        let mut dec = CeltDecodeState::new(lm).unwrap();
        for t in 0..frames {
            let header = CeltFrameHeader {
                transient: true,
                anti_collapse_on: Some(on),
                ..plain_header(t == 0)
            };
            let frame = encode_celt_frame_pcm_auto(
                &mut enc,
                &input[t * n..(t + 1) * n],
                &header,
                frame_bytes,
                0,
                NUM_BANDS,
            )
            .unwrap();
            assert_eq!(frame.prefix.header.anti_collapse_on, Some(on));
            let decoded = decode_celt_frame_auto(&mut dec, &frame.bytes, 0, NUM_BANDS).unwrap();
            assert_eq!(
                decoded.prefix.header.anti_collapse_on,
                Some(on),
                "on={on} frame {t}: anti-collapse bit did not round-trip"
            );
            assert!(decoded.pcm.iter().all(|s| s.is_finite()));
            assert_eq!(
                enc.coarse_energy().energy,
                dec.coarse_energy().energy,
                "on={on} frame {t}: coarse lockstep broken"
            );
        }
    }
}
