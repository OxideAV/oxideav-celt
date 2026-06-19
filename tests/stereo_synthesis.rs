//! Integration test for the documented CELT **stereo** synthesis chain
//! (RFC 6716 §4.3.2 energy → §4.3.6 denormalization → §4.3.7 inverse
//! MDCT + weighted overlap-add → interleaved PCM).
//!
//! The CELT decode splits cleanly at the band-shape boundary: everything
//! from the two channels' decoded **unit-norm band shapes** + per-channel
//! log-2 energy envelope onward is fully specified per channel by the
//! RFC (§4.3.6 "each decoded normalized band is multiplied by the square
//! root of the decoded energy"; §4.3.7 inverse MDCT "has no special
//! characteristics"). This file drives that documented part end-to-end
//! for a two-channel frame and confirms:
//!
//! 1. The per-channel energy assembly ([`assemble_band_log_energy_q8`])
//!    produces independent per-channel envelopes (channel 0 from
//!    `energy[0]`, channel 1 from `energy[1]`).
//! 2. Per-channel §4.3.6 denormalization
//!    ([`denormalize_bands_in_place_f32`]) scales each channel's unit
//!    shapes by `sqrt(2^(E_q8/256))` independently.
//! 3. The stereo synthesis state
//!    ([`StereoCeltDecodeState::synthesize_stereo_frame`]) runs the
//!    per-channel §4.3.7 inverse MDCT + WOLA and interleaves the two
//!    channels' PCM into one L/R/L/R buffer, with each channel's overlap
//!    tail / de-emphasis memory carried independently.
//!
//! The §4.3.4.4 `itheta` mid/side band coupling that produces the two
//! channel shapes from the bitstream is the documented docs gap and is
//! NOT exercised here; this test starts from the decoded per-channel
//! shapes, the same input boundary the synthesis spine draws.
//!
//! Wall: this test reads only the crate's own public API and RFC 6716
//! §4.3 (cited in the module docs). No external library source.

use oxideav_celt::{
    assemble_band_log_energy_q8, denormalize_bands_in_place_f32, log_energy_q8_to_amplitude_f32,
    mdct_size, CoarseEnergyState, Deemphasis, LongMdctSynthesis, StereoCeltDecodeState,
    BAND_BINS_LM, NUM_BANDS,
};

/// Build a deterministic unit-norm-ish shape for one channel's full
/// coded spectrum at frame-size shift `lm`. (The synthesis chain does
/// not require exact unit norm; it just multiplies by the band
/// amplitude. The values here are arbitrary but per-band-contiguous so
/// the denormalization seam is exercised band by band.)
fn channel_shapes(lm: usize, seed: u32) -> Vec<f32> {
    let coded: usize = BAND_BINS_LM[lm].iter().map(|&n| n as usize).sum();
    (0..coded)
        .map(|i| (((i as u32).wrapping_mul(seed).wrapping_add(1)) % 7) as f32 - 3.0)
        .collect()
}

/// The full documented stereo chain matches two independent mono chains:
/// per-channel envelope assembly → per-channel denormalization →
/// per-channel synthesis + de-emphasis, interleaved.
#[test]
fn stereo_chain_matches_two_mono_chains() {
    let lm = 1usize;
    let lm_u = lm as u32;

    // Two distinct per-channel coarse-energy envelopes (channel 0 vs 1).
    let mut coarse = CoarseEnergyState::new();
    for b in 0..NUM_BANDS {
        coarse.energy[0][b] = ((b as f32) * 0.1) - 1.0;
        coarse.energy[1][b] = 1.0 - ((b as f32) * 0.07);
    }

    // Per-channel Q8 envelopes (no fine/finalize corrections here — the
    // energy-assembly module has its own dedicated tests for those).
    let env0 = assemble_band_log_energy_q8(&coarse, 0, None, None).unwrap();
    let env1 = assemble_band_log_energy_q8(&coarse, 1, None, None).unwrap();
    // The two channels assemble independently.
    assert_ne!(env0, env1);

    // Per-channel decoded shapes.
    let mut shape0 = channel_shapes(lm, 37);
    let mut shape1 = channel_shapes(lm, 53);

    // §4.3.6 per-channel denormalization (full-band window).
    let bins = BAND_BINS_LM[lm];
    assert!(denormalize_bands_in_place_f32(&mut shape0, &bins, &env0));
    assert!(denormalize_bands_in_place_f32(&mut shape1, &bins, &env1));

    // Stereo synthesis → interleaved L/R/L/R PCM.
    let mut stereo = StereoCeltDecodeState::new(lm_u).unwrap();
    let out = stereo
        .synthesize_stereo_frame(&shape0, &shape1, 0, 21, None)
        .unwrap();
    let n = mdct_size(lm_u).unwrap();
    assert_eq!(out.len(), 2 * n);

    // Reference: two independent mono synthesis + de-emphasis chains over
    // the same per-channel denormalized spectra.
    let mut sl = LongMdctSynthesis::new(lm_u).unwrap();
    let mut sr = LongMdctSynthesis::new(lm_u).unwrap();
    let mut dl = Deemphasis::new();
    let mut dr = Deemphasis::new();
    let mut l = sl.synthesize(&shape0, 0, 21).unwrap();
    let mut r = sr.synthesize(&shape1, 0, 21).unwrap();
    dl.apply_in_place(&mut l);
    dr.apply_in_place(&mut r);

    for i in 0..n {
        assert!((out[2 * i] - l[i]).abs() <= 1e-5, "left[{i}]");
        assert!((out[2 * i + 1] - r[i]).abs() <= 1e-5, "right[{i}]");
    }
    assert!(out.iter().all(|x| x.is_finite()));
}

/// A louder channel (higher per-band energy) yields a larger-amplitude
/// denormalized spectrum, and that ordering survives the linear synthesis
/// transform: the louder channel's PCM has the larger RMS.
#[test]
fn louder_channel_has_larger_rms() {
    let lm = 0usize;
    let lm_u = lm as u32;

    let mut coarse = CoarseEnergyState::new();
    for b in 0..NUM_BANDS {
        coarse.energy[0][b] = 2.0; // +2 log-2 steps everywhere (loud)
        coarse.energy[1][b] = -2.0; // -2 log-2 steps everywhere (quiet)
    }
    let env_loud = assemble_band_log_energy_q8(&coarse, 0, None, None).unwrap();
    let env_quiet = assemble_band_log_energy_q8(&coarse, 1, None, None).unwrap();

    // The loud channel's per-band amplitude is strictly larger.
    for b in 0..NUM_BANDS {
        let a_loud = log_energy_q8_to_amplitude_f32(env_loud[b]);
        let a_quiet = log_energy_q8_to_amplitude_f32(env_quiet[b]);
        assert!(a_loud > a_quiet);
    }

    // Same base shape on both channels; only the energy differs.
    let base = channel_shapes(lm, 41);
    let mut loud = base.clone();
    let mut quiet = base.clone();
    let bins = BAND_BINS_LM[lm];
    assert!(denormalize_bands_in_place_f32(&mut loud, &bins, &env_loud));
    assert!(denormalize_bands_in_place_f32(
        &mut quiet, &bins, &env_quiet
    ));

    let mut stereo = StereoCeltDecodeState::new(lm_u).unwrap();
    let out = stereo
        .synthesize_stereo_frame(&loud, &quiet, 0, 21, None)
        .unwrap();
    let n = mdct_size(lm_u).unwrap();

    let rms = |v: &[f32]| -> f64 {
        let s: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
        (s / v.len() as f64).sqrt()
    };
    let left: Vec<f32> = (0..n).map(|i| out[2 * i]).collect();
    let right: Vec<f32> = (0..n).map(|i| out[2 * i + 1]).collect();
    assert!(
        rms(&left) > rms(&right),
        "loud channel RMS {} not greater than quiet {}",
        rms(&left),
        rms(&right)
    );
}

/// Across two frames, each channel's overlap tail and de-emphasis memory
/// carry independently: de-interleaving the second stereo frame matches
/// two independent two-frame mono chains.
#[test]
fn per_channel_state_carries_across_two_frames() {
    let lm = 1usize;
    let lm_u = lm as u32;
    let n = mdct_size(lm_u).unwrap();
    let bins = BAND_BINS_LM[lm];

    let mut coarse = CoarseEnergyState::new();
    for b in 0..NUM_BANDS {
        coarse.energy[0][b] = 0.5;
        coarse.energy[1][b] = -0.5;
    }
    let env0 = assemble_band_log_energy_q8(&coarse, 0, None, None).unwrap();
    let env1 = assemble_band_log_energy_q8(&coarse, 1, None, None).unwrap();

    let make = |seed: u32, env: &[i32; NUM_BANDS]| -> Vec<f32> {
        let mut s = channel_shapes(lm, seed);
        denormalize_bands_in_place_f32(&mut s, &bins, env);
        s
    };
    let f1l = make(11, &env0);
    let f1r = make(13, &env1);
    let f2l = make(17, &env0);
    let f2r = make(19, &env1);

    let mut stereo = StereoCeltDecodeState::new(lm_u).unwrap();
    let _ = stereo
        .synthesize_stereo_frame(&f1l, &f1r, 0, 21, None)
        .unwrap();
    let frame2 = stereo
        .synthesize_stereo_frame(&f2l, &f2r, 0, 21, None)
        .unwrap();

    // Reference: two independent mono chains run for two frames.
    let mut sl = LongMdctSynthesis::new(lm_u).unwrap();
    let mut sr = LongMdctSynthesis::new(lm_u).unwrap();
    let mut dl = Deemphasis::new();
    let mut dr = Deemphasis::new();
    let mut l1 = sl.synthesize(&f1l, 0, 21).unwrap();
    let mut r1 = sr.synthesize(&f1r, 0, 21).unwrap();
    dl.apply_in_place(&mut l1);
    dr.apply_in_place(&mut r1);
    let mut l2 = sl.synthesize(&f2l, 0, 21).unwrap();
    let mut r2 = sr.synthesize(&f2r, 0, 21).unwrap();
    dl.apply_in_place(&mut l2);
    dr.apply_in_place(&mut r2);

    for i in 0..n {
        assert!((frame2[2 * i] - l2[i]).abs() <= 1e-5, "left frame2[{i}]");
        assert!(
            (frame2[2 * i + 1] - r2[i]).abs() <= 1e-5,
            "right frame2[{i}]"
        );
    }
}
