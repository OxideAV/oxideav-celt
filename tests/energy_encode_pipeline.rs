//! Integration test: the encoder energy front-end composes with the
//! decoder energy back-end end-to-end.
//!
//! Pipeline exercised (RFC 6716 §4.3.6 → §4.3.2.1 → §4.3.2 → §4.3.6):
//!
//! 1. **Analyze** an MDCT-domain spectrum into per-band coarse-energy
//!    targets and unit-norm shapes ([`analyze_bands_f32`], §4.3.6
//!    inverse).
//! 2. **Encode** the coarse energy from those targets
//!    ([`encode_coarse_energy`], §4.3.2.1) into a frame.
//! 3. **Decode** the coarse energy back ([`decode_coarse_energy`]) and
//!    assemble the Q8 envelope ([`assemble_band_log_energy_q8`],
//!    §4.3.2).
//! 4. **Denormalize** the analyzed shapes at the reconstructed energies
//!    ([`denormalize_bands_f32`], §4.3.6) and confirm the recovered
//!    spectrum matches the original to the coarse (6 dB) resolution.
//!
//! The frame carries only range-coded coarse-energy symbols (no raw
//! bits), so the finished bytes are padded to a fixed length to give
//! the decoder the same `storage_bits()` the encoder budgeted, keeping
//! the budget dispatch in lockstep. The §4.3.2.2 fine-energy tightening
//! is demonstrated arithmetically (the fine quantizer is a pure
//! function) on top of the reconstructed coarse envelope.

use oxideav_celt::{
    analyze_bands_f32, assemble_band_log_energy_q8, decode_coarse_energy, denormalize_bands_f32,
    encode_coarse_energy, fine_correction_q14, log_energy_f32_to_q8, quantize_fine_energy_band,
    CoarseEnergyState, RangeDecoder, RangeEncoder, BAND_BINS_LM, MAX_CHANNELS, NUM_BANDS,
};

const FRAME_BYTES: usize = 192;
const BUDGET_BITS: u32 = (FRAME_BYTES as u32) * 8;

/// Build a deterministic non-silent MDCT spectrum for LM = `lm`,
/// laid out band-contiguously per `BAND_BINS_LM[lm]`.
fn build_spectrum(lm: usize) -> (Vec<f32>, Vec<u32>) {
    let bins: Vec<u32> = BAND_BINS_LM[lm].to_vec();
    let total: u32 = bins.iter().sum();
    let mut spectrum = Vec::with_capacity(total as usize);
    let mut idx = 0u32;
    for (b, &n) in bins.iter().enumerate() {
        // A per-band amplitude that varies across the frame, with a
        // shape that is clearly non-constant so the analysis exercises
        // a real direction (never all-zero).
        let amp = 0.2 + 0.9 * (b as f32 * 0.31).sin().abs();
        for _ in 0..n {
            let phase = (idx as f32 * 0.17 + b as f32 * 0.9).sin();
            spectrum.push(amp * (0.5 + phase));
            idx += 1;
        }
    }
    (spectrum, bins)
}

#[test]
fn analyze_encode_decode_denormalize_pipeline() {
    let lm = 2usize; // 10 ms
    let (original, bins) = build_spectrum(lm);

    // 1. Analyze into coarse-energy targets + unit shapes.
    let analyzed = analyze_bands_f32(&original, &bins).unwrap();
    let mut target: [[f32; NUM_BANDS]; MAX_CHANNELS] = [[0.0; NUM_BANDS]; MAX_CHANNELS];
    let mut target_q8 = [0i32; NUM_BANDS];
    for (b, a) in analyzed.iter().enumerate() {
        target[0][b] = a.log_energy;
        target_q8[b] = log_energy_f32_to_q8(a.log_energy);
    }

    // 2. Encode the coarse energy from the analyzed targets.
    let mut enc = RangeEncoder::new();
    let mut enc_state = CoarseEnergyState::new();
    encode_coarse_energy(
        &mut enc,
        &mut enc_state,
        &target,
        true, // intra
        lm as u32,
        0,
        NUM_BANDS,
        1,
        BUDGET_BITS,
    )
    .unwrap();
    let mut frame = enc.finish();
    assert!(frame.len() <= FRAME_BYTES, "frame overflowed the budget");
    frame.resize(FRAME_BYTES, 0);

    // 3. Decode the coarse energy back and assemble the Q8 envelope.
    let mut dec = RangeDecoder::new(&frame);
    let mut dec_state = CoarseEnergyState::new();
    decode_coarse_energy(&mut dec, &mut dec_state, true, lm as u32, 0, NUM_BANDS, 1).unwrap();
    assert!(!dec.has_error());
    assert_eq!(
        enc_state.energy[0], dec_state.energy[0],
        "coarse state diverged"
    );
    let reconstructed_q8 = assemble_band_log_energy_q8(&dec_state, 0, None, None).unwrap();

    // The coarse reconstruction is within half a 6 dB step (128 Q8) of
    // the target for every band.
    for b in 0..NUM_BANDS {
        let err = (reconstructed_q8[b] - target_q8[b]).abs();
        assert!(
            err <= 129,
            "band {b} coarse error {err} Q8 exceeds a half step"
        );
    }

    // 3b. The §4.3.2.2 fine step tightens the reconstruction. Using
    // b_bits fine bits, quantize the residual and confirm the combined
    // coarse+fine error never grows and is bounded by the fine step.
    let b_bits = 4u32;
    for b in 0..NUM_BANDS {
        let residual_f32 = target[0][b] - dec_state.energy[0][b];
        let residual_q14 = (residual_f32 * 16384.0).round() as i32;
        let f = quantize_fine_energy_band(residual_q14, b_bits);
        let fine_q8 = (fine_correction_q14(f, b_bits) + 32) >> 6;
        let recon_fine_q8 = reconstructed_q8[b] + fine_q8;
        let coarse_err = (reconstructed_q8[b] - target_q8[b]).abs();
        let fine_err = (recon_fine_q8 - target_q8[b]).abs();
        assert!(
            fine_err <= coarse_err + 1,
            "band {b}: fine step worsened the error ({coarse_err} -> {fine_err})"
        );
        // Fine step in Q8 is 256 >> b_bits; error is bounded by ~half of
        // it plus coarse+fine rounding slack.
        let fine_step_q8 = 256i32 >> b_bits;
        assert!(
            fine_err <= fine_step_q8 + 2,
            "band {b}: coarse+fine error {fine_err} exceeds fine step {fine_step_q8}"
        );
    }

    // 4. Denormalize the analyzed shapes at the reconstructed energies
    // and confirm the recovered spectrum tracks the original to the
    // coarse resolution (per-band amplitude ratio within 2^±0.25).
    let shapes: Vec<Vec<f32>> = analyzed.iter().map(|a| a.shape.clone()).collect();
    let shape_refs: Vec<&[f32]> = shapes.iter().map(|s| s.as_slice()).collect();
    let mut reconstructed = vec![0.0f32; original.len()];
    assert!(denormalize_bands_f32(
        &shape_refs,
        &reconstructed_q8,
        &mut reconstructed
    ));

    // Per band, compare the L2 norm of the reconstructed vs original
    // samples: the ratio is the amplitude ratio 2^((E_recon-E_target)/2),
    // bounded by 2^±0.25 because |E_recon - E_target| <= 0.5.
    let mut offset = 0usize;
    for (b, &n) in bins.iter().enumerate() {
        let n = n as usize;
        let orig_norm: f32 = original[offset..offset + n]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        let recon_norm: f32 = reconstructed[offset..offset + n]
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        offset += n;
        if orig_norm < 1e-6 {
            continue;
        }
        let ratio = recon_norm / orig_norm;
        assert!(
            (0.83..=1.20).contains(&ratio),
            "band {b}: amplitude ratio {ratio} outside the coarse tolerance"
        );
    }
}
