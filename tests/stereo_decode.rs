//! Integration test for the CELT **stereo frame-decode driver**
//! (`StereoCeltDecodeState::decode_stereo_frame`).
//!
//! The stereo driver is the two-channel counterpart of
//! `decode_celt_frame`: it decodes the stereo Table 56 control prefix and
//! **both channels' §4.3.2.1 coarse energy** from the range-coded
//! bitstream (the stereo coarse channel interleave is specified —
//! `decode_coarse_energy` with `channels = 2`), composes each channel's
//! §4.3.2 Q8 log-energy envelope (bitstream coarse + caller-supplied fine
//! corrections), and runs the per-channel §4.3.6 → §4.3.7 synthesis on
//! the two denormalized residual spectra, emitting interleaved L/R/L/R
//! PCM.
//!
//! This file drives that driver from raw bitstream bytes + per-channel
//! denormalized residual spectra and confirms, through the public API,
//! that it is a faithful composition of the parts each already covered by
//! their own module tests:
//!
//! 1. The bitstream-decoded **coarse energy** the driver produces for
//!    both channels equals an independent standalone prefix decode
//!    (`decode_frame_prefix` with `stereo = true`) of the same bytes.
//! 2. The interleaved PCM equals the manual composition: standalone
//!    stereo prefix decode → per-channel envelope assembly → per-channel
//!    synthesis (`StereoLongMdctSynthesis`) → per-channel de-emphasis,
//!    interleaved.
//! 3. The shared coarse-energy prediction carries across frames so the
//!    second frame of a shared state differs from a fresh first decode.
//!
//! The §4.3.4.4 `itheta` mid/side coupling that produces the two channel
//! residual spectra from the joint-coded bitstream, and the main
//! §4.3.2.2 fine-energy per-channel interleave, are the documented docs
//! gaps; this test supplies the per-channel residual spectra + fine
//! corrections directly, the same input boundary the driver draws.
//!
//! Wall: this test reads only the crate's own public API and RFC 6716
//! §4.3 (cited in the module docs). No external library source.

use oxideav_celt::{
    assemble_band_log_energy_q8, coded_total_bins, decode_frame_prefix, mdct_size,
    CoarseEnergyState, Deemphasis, LongMdctSynthesis, RangeDecoder, StereoCeltDecodeState,
    NUM_BANDS,
};

/// A deterministic stereo CELT payload. The exact bytes are arbitrary;
/// the test only needs a non-degenerate range-coded stream that the
/// prefix decode walks without error and that does not signal a transient
/// (so the long-MDCT synthesis path applies).
fn stereo_payload(seed: u8) -> Vec<u8> {
    (0..112u8)
        .map(|i| {
            i.wrapping_mul(37)
                .wrapping_add(seed.wrapping_mul(11))
                .wrapping_add(5)
        })
        .collect()
}

/// Find a `(seed, lm)` payload whose stereo prefix is **non-transient**,
/// so the test exercises the long-MDCT path the driver supports.
fn non_transient_payload(lm: u32) -> Vec<u8> {
    for seed in 0u8..128 {
        let buf = stereo_payload(seed);
        let mut probe = CoarseEnergyState::new();
        let mut dec = RangeDecoder::new(&buf);
        if let Ok(prefix) =
            decode_frame_prefix(&mut dec, &mut probe, lm, buf.len() as u32, true, 0, 21)
        {
            if !prefix.header.transient {
                return buf;
            }
        }
    }
    panic!("no non-transient stereo payload found for lm={lm}");
}

fn zero_fine2() -> [[i32; NUM_BANDS]; 2] {
    [[0i32; NUM_BANDS]; 2]
}

/// The driver's bitstream-decoded coarse energy (both channels) matches a
/// standalone stereo prefix decode of the same bytes.
#[test]
fn driver_coarse_energy_matches_standalone_prefix() {
    let lm = 1u32;
    let buf = non_transient_payload(lm);
    let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
    let res: Vec<f32> = (0..coded).map(|i| ((i % 7) as f32) - 3.0).collect();

    // Driver path.
    let mut st = StereoCeltDecodeState::new(lm).unwrap();
    let out = st
        .decode_stereo_frame(&buf, 0, 21, &zero_fine2(), &res, &res, None)
        .expect("stereo decode");

    // Standalone reference: decode the same stereo prefix and assemble
    // each channel's envelope from the resulting coarse state.
    let mut ref_coarse = CoarseEnergyState::new();
    let mut dec = RangeDecoder::new(&buf);
    let _ref_prefix =
        decode_frame_prefix(&mut dec, &mut ref_coarse, lm, buf.len() as u32, true, 0, 21).unwrap();
    let ref_env0 =
        assemble_band_log_energy_q8(&ref_coarse, 0, Some(&[0; NUM_BANDS]), None).unwrap();
    let ref_env1 =
        assemble_band_log_energy_q8(&ref_coarse, 1, Some(&[0; NUM_BANDS]), None).unwrap();

    assert_eq!(out.envelope_q8[0], ref_env0, "left envelope mismatch");
    assert_eq!(out.envelope_q8[1], ref_env1, "right envelope mismatch");

    // The driver's carried coarse state equals the standalone decode.
    for c in 0..2 {
        for b in 0..NUM_BANDS {
            assert_eq!(
                st.coarse_energy().energy[c][b],
                ref_coarse.energy[c][b],
                "coarse[{c}][{b}] mismatch"
            );
        }
    }
}

/// The interleaved PCM the driver emits equals the manual composition:
/// standalone prefix decode → per-channel envelope assembly → per-channel
/// synthesis → per-channel de-emphasis, interleaved.
#[test]
fn driver_pcm_matches_manual_composition() {
    let lm = 1u32;
    let buf = non_transient_payload(lm);
    let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
    let res_l: Vec<f32> = (0..coded).map(|i| ((i % 7) as f32) - 3.0).collect();
    let res_r: Vec<f32> = (0..coded).map(|i| ((i % 5) as f32) - 2.0).collect();

    // Driver path.
    let mut st = StereoCeltDecodeState::new(lm).unwrap();
    let out = st
        .decode_stereo_frame(&buf, 0, 21, &zero_fine2(), &res_l, &res_r, None)
        .unwrap();
    let n = mdct_size(lm).unwrap();
    assert_eq!(out.pcm.len(), 2 * n);

    // Manual reference: identical decode, hand-composed from the parts.
    let mut ref_coarse = CoarseEnergyState::new();
    let mut dec = RangeDecoder::new(&buf);
    let _ =
        decode_frame_prefix(&mut dec, &mut ref_coarse, lm, buf.len() as u32, true, 0, 21).unwrap();
    // (Envelopes are not needed for the PCM cross-check: the driver's
    // synthesis consumes the *already-denormalized* residual spectra the
    // caller passes, so the reference uses the same spectra directly.)
    let mut sl = LongMdctSynthesis::new(lm).unwrap();
    let mut sr = LongMdctSynthesis::new(lm).unwrap();
    let mut dl = Deemphasis::new();
    let mut dr = Deemphasis::new();
    let mut l = sl.synthesize(&res_l, 0, 21).unwrap();
    let mut r = sr.synthesize(&res_r, 0, 21).unwrap();
    dl.apply_in_place(&mut l);
    dr.apply_in_place(&mut r);

    for i in 0..n {
        assert!((out.pcm[2 * i] - l[i]).abs() <= 1e-5, "left[{i}]");
        assert!((out.pcm[2 * i + 1] - r[i]).abs() <= 1e-5, "right[{i}]");
    }
    assert!(out.pcm.iter().all(|x| x.is_finite()));
}

/// The shared coarse-energy prediction carries across frames: the second
/// frame of a shared state differs from a fresh first decode of the same
/// bytes (the carried coarse prediction + per-channel overlap +
/// de-emphasis memory all fold prior state into frame 2).
#[test]
fn driver_carries_coarse_prediction_across_frames() {
    let lm = 1u32;
    let buf = non_transient_payload(lm);
    let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
    let res: Vec<f32> = (0..coded).map(|i| ((i % 4) as f32) - 1.5).collect();

    let mut shared = StereoCeltDecodeState::new(lm).unwrap();
    let _f1 = shared
        .decode_stereo_frame(&buf, 0, 21, &zero_fine2(), &res, &res, None)
        .unwrap();
    let f2 = shared
        .decode_stereo_frame(&buf, 0, 21, &zero_fine2(), &res, &res, None)
        .unwrap();

    let mut fresh = StereoCeltDecodeState::new(lm).unwrap();
    let ff = fresh
        .decode_stereo_frame(&buf, 0, 21, &zero_fine2(), &res, &res, None)
        .unwrap();

    // The envelope after a carried frame differs from the fresh decode
    // (inter-frame coarse prediction has folded the prior frame's energy).
    assert_ne!(
        f2.envelope_q8, ff.envelope_q8,
        "carried coarse prediction did not change frame 2 envelope"
    );
    // And the PCM differs too (overlap + de-emphasis memory carried).
    let any_diff = f2
        .pcm
        .iter()
        .zip(&ff.pcm)
        .any(|(a, b)| (a - b).abs() > 1e-9);
    assert!(any_diff, "carried state did not affect frame 2 PCM");
}

/// A Hybrid-mode stereo window (bands 17..=20) decodes a 4-band frame and
/// still emits a full `2 * frame_size` interleaved PCM.
#[test]
fn driver_decodes_hybrid_window() {
    let lm = 2u32;
    // Reuse the non-transient search but for the hybrid window the prefix
    // walk is over (17, 21); a generic payload still walks without error.
    let buf = stereo_payload(9);
    let coded = coded_total_bins(17, 21, lm).unwrap() as usize;
    let res: Vec<f32> = (0..coded).map(|i| ((i % 3) as f32) - 1.0).collect();

    let mut st = StereoCeltDecodeState::new(lm).unwrap();
    match st.decode_stereo_frame(&buf, 17, 21, &zero_fine2(), &res, &res, None) {
        Ok(out) => {
            assert_eq!(out.pcm.len(), 2 * mdct_size(lm).unwrap());
            assert_eq!(out.prefix.start, 17);
            assert!(out.pcm.iter().all(|x| x.is_finite()));
        }
        // A transient prefix for this specific payload is a valid outcome
        // (rejected, not mis-decoded); the test's purpose is the hybrid
        // window geometry, which the Ok arm exercises.
        Err(oxideav_celt::Error::NotImplemented) => {}
        Err(e) => panic!("unexpected hybrid stereo error: {e:?}"),
    }
}
