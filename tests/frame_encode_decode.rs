//! Integration test: the full mono CELT frame **encoder**
//! (`encode_celt_frame`) produces a fixed-size frame the full frame
//! **decoder** (`decode_celt_frame`) turns back into PCM, with the
//! decoder's residual spectrum matching the encoder's own
//! reconstruction **bit-exactly**.
//!
//! Pipeline (RFC 6716 §4.3.6 analysis → Table 56 → §4.3.7 synthesis):
//!
//! encoder: analyze bands → prefix (header/coarse/TF/spread/boosts/
//! allocation) → fine energy → PVQ shapes → §5.1.5 fixed-size frame.
//!
//! decoder: `decode_celt_frame` (prefix → fine → residual → IMDCT/WOLA
//! → post-filter → de-emphasis → PCM), plus a manual
//! `decode_frame_prefix` → `decode_fine_energy` →
//! `decode_residual_bands` walk to compare the raw residual spectrum
//! against `EncodedFrame::reconstructed_spectrum`.

use oxideav_celt::{
    band_bins, coded_total_bins, decode_celt_frame, decode_celt_frame_auto, decode_fine_energy,
    decode_frame_prefix, decode_residual_bands, encode_celt_frame, encode_celt_frame_auto, v_count,
    CeltDecodeState, CeltFrameHeader, CoarseEnergyState, PostFilter, RangeDecoder, NUM_BANDS,
};

const FRAME_BYTES: u32 = 160;

/// Deterministic non-silent coded-window spectrum for `lm`.
fn build_spectrum(lm: u32, start: usize, end: usize) -> Vec<f32> {
    let total = coded_total_bins(start, end, lm).unwrap() as usize;
    (0..total)
        .map(|i| {
            let t = i as f32;
            0.7 * (t * 0.13).sin() + 0.4 * (t * 0.031).cos() + 0.1
        })
        .collect()
}

/// A modest per-band pulse allocation that never trips the §4.3.4.4
/// split gate at the given `lm`.
fn small_band_k(lm: u32, start: usize, end: usize) -> Vec<u32> {
    (start..end)
        .map(|band| {
            let n = band_bins(band, lm).unwrap();
            // Aim for a few pulses; back off until V(N, K) fits 32 bits.
            let mut k = 4u32.min(n.max(1));
            while k > 0 && v_count(n, k) == u32::MAX {
                k -= 1;
            }
            k
        })
        .collect()
}

fn default_header() -> CeltFrameHeader {
    CeltFrameHeader {
        silence: false,
        post_filter: None,
        transient: false,
        intra: true,
        anti_collapse_on: None,
    }
}

/// Full encode → decode: the decoder's residual spectrum equals the
/// encoder's reconstruction bit-exactly, and `decode_celt_frame`
/// produces finite, non-trivial PCM from the same bytes.
#[test]
fn encode_then_decode_matches_encoder_reconstruction() {
    let lm = 2u32;
    let (start, end) = (0usize, NUM_BANDS);
    let spectrum = build_spectrum(lm, start, end);
    let band_k = small_band_k(lm, start, end);
    let fine_bits = [2u32; NUM_BANDS];

    // --- encode ---
    let mut enc_state = CoarseEnergyState::new();
    let encoded = encode_celt_frame(
        &mut enc_state,
        &spectrum,
        &default_header(),
        lm,
        FRAME_BYTES,
        start,
        end,
        &fine_bits,
        &band_k,
    )
    .unwrap();
    assert_eq!(encoded.bytes.len(), FRAME_BYTES as usize);

    // --- manual decode: prefix → fine → residual ---
    let mut dec = RangeDecoder::new(&encoded.bytes);
    let mut dec_state = CoarseEnergyState::new();
    let prefix =
        decode_frame_prefix(&mut dec, &mut dec_state, lm, FRAME_BYTES, false, start, end).unwrap();
    assert!(prefix.header.intra);
    assert!(!prefix.header.transient);
    assert_eq!(prefix.spread, encoded.prefix.spread);
    assert_eq!(prefix.allocation, encoded.prefix.allocation);
    assert_eq!(prefix.boosts, encoded.prefix.boosts);

    let fine_q14 = decode_fine_energy(&mut dec, &fine_bits);
    let env_q8 =
        oxideav_celt::assemble_band_log_energy_q8(&dec_state, 0, Some(&fine_q14), None).unwrap();

    let window_energy: Vec<i32> = env_q8[start..end].to_vec();
    let mut residual = decode_residual_bands(
        &mut dec,
        lm,
        start,
        end,
        false,
        prefix.tf.tf_select,
        &prefix.tf.tf_changes,
        &band_k,
        prefix.spread,
        &window_energy,
    )
    .unwrap();
    assert!(!dec.has_error());

    // §4.3.2.2 finalize (r406): the leftover raw bits refine the
    // envelope; the manual walk mirrors the driver.
    let priorities = oxideav_celt::finalize_priorities_from_k(&band_k);
    let leftover = dec.storage_bits().saturating_sub(dec.tell());
    let fin = oxideav_celt::finalize_extra_bits_depth(
        &mut dec,
        &priorities,
        &fine_bits,
        start,
        end,
        1,
        leftover,
    );
    let env_final = oxideav_celt::assemble_band_log_energy_q8(
        &dec_state,
        0,
        Some(&fine_q14),
        Some(&fin.corrections_q14[0]),
    )
    .unwrap();
    assert_eq!(env_final, encoded.envelope_q8, "envelope diverged");
    assert!(oxideav_celt::apply_finalize_scale_f32(
        &mut residual.samples,
        lm,
        start,
        end,
        &fin.corrections_q14[0],
    ));
    assert_eq!(
        residual.samples, encoded.reconstructed_spectrum,
        "decoded residual spectrum != encoder reconstruction"
    );

    // §4.3.2.1 fine feedback (r406): the encoder's prediction state
    // now carries the final (coarse + fine + finalize) envelope; the
    // manual decode side performs the identical fold and lands on the
    // same state.
    let env_f32 = oxideav_celt::assemble_band_log_energy_f32(
        &dec_state,
        0,
        Some(&fine_q14),
        Some(&fin.corrections_q14[0]),
    )
    .unwrap();
    dec_state.energy[0][start..end].copy_from_slice(&env_f32[start..end]);
    assert_eq!(
        enc_state.energy[0], dec_state.energy[0],
        "coarse state diverged after the fine feedback fold"
    );

    // --- full decode_celt_frame to PCM over the same bytes ---
    let mut state = CeltDecodeState::new(lm).unwrap();
    let frame =
        decode_celt_frame(&mut state, &encoded.bytes, start, end, &fine_bits, &band_k).unwrap();
    assert_eq!(frame.pcm.len(), 120 << lm);
    assert!(frame.pcm.iter().all(|s| s.is_finite()));
    assert!(
        frame.pcm.iter().any(|&s| s != 0.0),
        "PCM is silent for a non-silent frame"
    );
}

/// The energy of each decoded band tracks the analyzed input energy to
/// the coarse+fine quantization tolerance.
#[test]
fn encoded_frame_preserves_band_energy() {
    let lm = 1u32;
    let (start, end) = (0usize, NUM_BANDS);
    let spectrum = build_spectrum(lm, start, end);
    let band_k = small_band_k(lm, start, end);
    let fine_bits = [3u32; NUM_BANDS];

    let mut enc_state = CoarseEnergyState::new();
    let encoded = encode_celt_frame(
        &mut enc_state,
        &spectrum,
        &default_header(),
        lm,
        FRAME_BYTES,
        start,
        end,
        &fine_bits,
        &band_k,
    )
    .unwrap();

    // Per band: reconstructed L2 norm vs original L2 norm. The envelope
    // is quantized to coarse (±1/2 step) + fine (3 bits → ±1/16 step),
    // so the amplitude ratio is within 2^(±(1/16)/2) of the coarse
    // rounding-free ideal; allow the fine-step bound with slack.
    let mut offset = 0usize;
    for band in start..end {
        let n = band_bins(band, lm).unwrap() as usize;
        let orig: f32 = spectrum[offset..offset + n]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        let recon: f32 = encoded.reconstructed_spectrum[offset..offset + n]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        offset += n;
        if orig < 1e-6 {
            continue;
        }
        let ratio = recon / orig;
        assert!(
            (0.95..=1.06).contains(&ratio),
            "band {band}: amplitude ratio {ratio} outside the fine tolerance"
        );
    }
}

/// Two consecutive frames through one encoder state and one decoder
/// state: the §4.3.2.1 inter-frame prediction stays in lockstep and the
/// second (inter) frame still decodes to the encoder's reconstruction.
#[test]
fn consecutive_frames_stay_in_lockstep() {
    let lm = 2u32;
    let (start, end) = (0usize, NUM_BANDS);
    let band_k = small_band_k(lm, start, end);
    let fine_bits = [2u32; NUM_BANDS];

    let mut enc_state = CoarseEnergyState::new();
    let mut dec_pcm_state = CeltDecodeState::new(lm).unwrap();

    for (i, gain) in [1.0f32, 0.6].iter().enumerate() {
        let spectrum: Vec<f32> = build_spectrum(lm, start, end)
            .iter()
            .map(|&s| s * gain)
            .collect();
        let header = CeltFrameHeader {
            intra: i == 0,
            ..default_header()
        };
        let encoded = encode_celt_frame(
            &mut enc_state,
            &spectrum,
            &header,
            lm,
            FRAME_BYTES,
            start,
            end,
            &fine_bits,
            &band_k,
        )
        .unwrap();
        let frame = decode_celt_frame(
            &mut dec_pcm_state,
            &encoded.bytes,
            start,
            end,
            &fine_bits,
            &band_k,
        )
        .unwrap();
        // The decoder's coarse state (inside dec_pcm_state) must track
        // the encoder's.
        assert_eq!(
            enc_state.energy[0],
            dec_pcm_state.coarse_energy().energy[0],
            "frame {i}: coarse prediction diverged"
        );
        assert_eq!(frame.prefix.header.intra, i == 0);
        assert!(frame.pcm.iter().all(|s| s.is_finite()));
    }
}

/// A frame with the post-filter on carries its raw-bit parameters
/// through the fixed-size assembly intact.
#[test]
fn post_filter_parameters_survive_roundtrip() {
    let lm = 2u32;
    let (start, end) = (0usize, NUM_BANDS);
    let spectrum = build_spectrum(lm, start, end);
    let band_k = small_band_k(lm, start, end);
    let fine_bits = [1u32; NUM_BANDS];
    let header = CeltFrameHeader {
        post_filter: Some(PostFilter {
            octave: 3,
            period: 150,
            gain: 5,
            tapset: 2,
        }),
        ..default_header()
    };

    let mut enc_state = CoarseEnergyState::new();
    let encoded = encode_celt_frame(
        &mut enc_state,
        &spectrum,
        &header,
        lm,
        FRAME_BYTES,
        start,
        end,
        &fine_bits,
        &band_k,
    )
    .unwrap();

    let mut state = CeltDecodeState::new(lm).unwrap();
    let frame =
        decode_celt_frame(&mut state, &encoded.bytes, start, end, &fine_bits, &band_k).unwrap();
    assert_eq!(frame.prefix.header.post_filter, header.post_filter);
}

/// Transient / silent headers and mismatched inputs are rejected.
#[test]
fn encoder_rejects_out_of_scope_frames() {
    let lm = 2u32;
    let (start, end) = (0usize, NUM_BANDS);
    let spectrum = build_spectrum(lm, start, end);
    let band_k = small_band_k(lm, start, end);
    let fine_bits = [0u32; NUM_BANDS];
    let mut state = CoarseEnergyState::new();

    // A transient header encodes (r406 short-block support) — on a
    // scratch state so the rejection checks below start clean.
    let transient = CeltFrameHeader {
        transient: true,
        ..default_header()
    };
    let mut scratch = CoarseEnergyState::new();
    encode_celt_frame(
        &mut scratch,
        &spectrum,
        &transient,
        lm,
        FRAME_BYTES,
        start,
        end,
        &fine_bits,
        &band_k,
    )
    .expect("transient mono encode");

    // Spectrum length mismatch.
    assert!(encode_celt_frame(
        &mut state,
        &spectrum[1..],
        &default_header(),
        lm,
        FRAME_BYTES,
        start,
        end,
        &fine_bits,
        &band_k
    )
    .is_err());

    // band_k length mismatch.
    assert!(encode_celt_frame(
        &mut state,
        &spectrum,
        &default_header(),
        lm,
        FRAME_BYTES,
        start,
        end,
        &fine_bits,
        &band_k[1..]
    )
    .is_err());
}

/// The fully self-contained codec loop: `encode_celt_frame_auto` →
/// `decode_celt_frame_auto` exchanges NO allocation out of band — both
/// sides derive `band_k` from the (bit-identical) prefix via the same
/// documented §4.3.3 → §4.3.4.1 seam.
#[test]
fn auto_encode_auto_decode_self_contained_loop() {
    let lm = 2u32;
    let (start, end) = (0usize, NUM_BANDS);
    let spectrum = build_spectrum(lm, start, end);

    let mut enc_state = CoarseEnergyState::new();
    let encoded = encode_celt_frame_auto(
        &mut enc_state,
        &spectrum,
        &default_header(),
        lm,
        FRAME_BYTES,
        start,
        end,
    )
    .unwrap();
    assert_eq!(encoded.bytes.len(), FRAME_BYTES as usize);

    // Auto-decode: derives band_k internally from the decoded prefix.
    let mut state = CeltDecodeState::new(lm).unwrap();
    let frame = decode_celt_frame_auto(&mut state, &encoded.bytes, start, end).unwrap();
    assert_eq!(frame.pcm.len(), 120 << lm);
    assert!(frame.pcm.iter().all(|s| s.is_finite()));
    assert!(
        frame.pcm.iter().any(|&s| s != 0.0),
        "auto loop produced silent PCM"
    );
    // Encoder and decoder coarse prediction stay in lockstep.
    assert_eq!(enc_state.energy[0], state.coarse_energy().energy[0]);

    // Bit-exact spectrum check: derive band_k AND fine_bits from the
    // encoder's returned prefix (the same values the auto-decoder
    // derives from its decoded copy) and walk the residual manually.
    let alloc = oxideav_celt::derive_band_allocation(&encoded.prefix, lm, 1, false).unwrap();
    let band_k = alloc.band_k;
    let mut dec = RangeDecoder::new(&encoded.bytes);
    let mut dec_state = CoarseEnergyState::new();
    let prefix =
        decode_frame_prefix(&mut dec, &mut dec_state, lm, FRAME_BYTES, false, start, end).unwrap();
    let fine_q14 = decode_fine_energy(&mut dec, &alloc.fine_bits);
    let env_q8 =
        oxideav_celt::assemble_band_log_energy_q8(&dec_state, 0, Some(&fine_q14), None).unwrap();
    let window_energy: Vec<i32> = env_q8[start..end].to_vec();
    let mut residual = decode_residual_bands(
        &mut dec,
        lm,
        start,
        end,
        false,
        prefix.tf.tf_select,
        &prefix.tf.tf_changes,
        &band_k,
        prefix.spread,
        &window_energy,
    )
    .unwrap();
    // The finalize step (r406): mirror the driver's leftover-bit walk.
    let priorities = oxideav_celt::finalize_priorities_from_k(&band_k);
    let leftover = dec.storage_bits().saturating_sub(dec.tell());
    let fin = oxideav_celt::finalize_extra_bits_depth(
        &mut dec,
        &priorities,
        &alloc.fine_bits,
        start,
        end,
        1,
        leftover,
    );
    let env_final = oxideav_celt::assemble_band_log_energy_q8(
        &dec_state,
        0,
        Some(&fine_q14),
        Some(&fin.corrections_q14[0]),
    )
    .unwrap();
    assert_eq!(env_final, encoded.envelope_q8);
    assert!(oxideav_celt::apply_finalize_scale_f32(
        &mut residual.samples,
        lm,
        start,
        end,
        &fin.corrections_q14[0],
    ));
    assert_eq!(
        residual.samples, encoded.reconstructed_spectrum,
        "auto loop residual != encoder reconstruction"
    );

    // Determinism: encoding the same spectrum from the same state
    // yields identical bytes.
    let mut enc_state2 = CoarseEnergyState::new();
    let encoded2 = encode_celt_frame_auto(
        &mut enc_state2,
        &spectrum,
        &default_header(),
        lm,
        FRAME_BYTES,
        start,
        end,
    )
    .unwrap();
    assert_eq!(encoded2.bytes, encoded.bytes);
}

/// The auto loop stays in lockstep across a multi-frame stream at
/// every frame size.
#[test]
fn auto_loop_multi_frame_all_lm() {
    for lm in 0..=3u32 {
        let (start, end) = (0usize, NUM_BANDS);
        let mut enc_state = CoarseEnergyState::new();
        let mut dec_state = CeltDecodeState::new(lm).unwrap();
        for (i, gain) in [1.0f32, 0.5, 1.4].iter().enumerate() {
            let spectrum: Vec<f32> = build_spectrum(lm, start, end)
                .iter()
                .map(|&s| s * gain)
                .collect();
            let header = CeltFrameHeader {
                intra: i == 0,
                ..default_header()
            };
            let encoded = encode_celt_frame_auto(
                &mut enc_state,
                &spectrum,
                &header,
                lm,
                FRAME_BYTES,
                start,
                end,
            )
            .unwrap();
            let frame = decode_celt_frame_auto(&mut dec_state, &encoded.bytes, start, end).unwrap();
            assert!(frame.pcm.iter().all(|s| s.is_finite()), "lm={lm} frame {i}");
            assert_eq!(
                enc_state.energy[0],
                dec_state.coarse_energy().energy[0],
                "lm={lm} frame {i}: prediction diverged"
            );
        }
    }
}
