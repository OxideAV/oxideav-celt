//! Integration test: the §4.3.2.1 coarse-energy *encode* composes into
//! a Table-56-ordered frame that the matching decoders read back, with
//! the reconstructed per-band energy envelope recovered bit-exactly.
//!
//! This closes the gap `control_encode_roundtrip.rs` documents: that
//! test omits the coarse-energy Laplace block because the encode side
//! did not exist. With `encode_coarse_energy` landed, the block now
//! sits in its real frame position — between the frame prefix and the
//! time-frequency parameters (RFC 6716 §4.3, Table 56) — and this test
//! chains prefix → coarse energy → TF parameters → spread into one
//! `RangeEncoder` frame, then walks the decoders through the same
//! sequence.
//!
//! To keep the coarse-energy budget dispatch (`budget - tell()`) in
//! lockstep between encoder and decoder, the frame carries no raw-bit
//! symbols (the post-filter is off, so the prefix is fully range-coded,
//! and TF/spread are range-coded): the finished range bytes are padded
//! with trailing zeros to a fixed length so the decoder's
//! `storage_bits()` equals the encoder's `budget`. The range decoder
//! zero-extends past the real bytes, so the padding is transparent.

use oxideav_celt::{
    decode_coarse_energy, decode_spread, decode_tf_parameters, encode_coarse_energy, encode_spread,
    encode_tf_parameters, CeltFrameHeader, CoarseEnergyState, RangeDecoder, RangeEncoder, Spread,
    TfParameters, MAX_CHANNELS, NUM_BANDS,
};

/// Frame budget in bytes; large enough that every coarse-energy band
/// stays on the full Laplace path for both encoder and decoder.
const FRAME_BYTES: usize = 128;
const BUDGET_BITS: u32 = (FRAME_BYTES as u32) * 8;

fn mono_target() -> [[f32; NUM_BANDS]; MAX_CHANNELS] {
    [
        std::array::from_fn(|b| 5.0 - 0.15 * b as f32 + 0.4 * (b as f32 * 0.7).sin()),
        [0.0; NUM_BANDS],
    ]
}

fn stereo_target() -> [[f32; NUM_BANDS]; MAX_CHANNELS] {
    [
        std::array::from_fn(|b| 4.5 - 0.13 * b as f32),
        std::array::from_fn(|b| 3.9 + 0.11 * b as f32 - 0.3 * (b as f32 * 0.5).cos()),
    ]
}

/// Mono frame: prefix (post-filter off) → coarse energy → TF → spread,
/// coarse energies recovered bit-exactly, intra mode.
#[test]
fn mono_frame_prefix_coarse_tf_spread_roundtrip() {
    let header = CeltFrameHeader {
        silence: false,
        post_filter: None,
        transient: false,
        intra: true,
        anti_collapse_on: None,
    };
    let lm = 2u8; // 10 ms
    let coded_bands = NUM_BANDS;
    let target = mono_target();
    let tf = TfParameters {
        tf_changes: vec![false; coded_bands],
        tf_select: 0,
        tf_select_decoded: false,
    };
    let spread = Spread::Normal;

    // --- encode ---
    let mut enc = RangeEncoder::new();
    let mut enc_state = CoarseEnergyState::new();
    header.encode_prefix(&mut enc).unwrap();
    encode_coarse_energy(
        &mut enc,
        &mut enc_state,
        &target,
        header.intra,
        lm as u32,
        0,
        NUM_BANDS,
        1,
        BUDGET_BITS,
    )
    .unwrap();
    encode_tf_parameters(&mut enc, &tf, header.transient, lm).unwrap();
    encode_spread(&mut enc, spread).unwrap();
    let mut frame = enc.finish();
    assert!(frame.len() <= FRAME_BYTES, "frame overflowed the budget");
    frame.resize(FRAME_BYTES, 0);

    // --- decode ---
    let mut dec = RangeDecoder::new(&frame);
    let decoded_header = CeltFrameHeader::decode_prefix(&mut dec);
    assert_eq!(decoded_header.intra, header.intra);
    assert_eq!(decoded_header.transient, header.transient);
    assert_eq!(decoded_header.post_filter, header.post_filter);

    let mut dec_state = CoarseEnergyState::new();
    decode_coarse_energy(
        &mut dec,
        &mut dec_state,
        decoded_header.intra,
        lm as u32,
        0,
        NUM_BANDS,
        1,
    )
    .unwrap();
    assert_eq!(
        enc_state.energy[0], dec_state.energy[0],
        "mono coarse-energy envelope mismatch"
    );

    let decoded_tf = decode_tf_parameters(&mut dec, 0, coded_bands, header.transient, lm);
    assert_eq!(decoded_tf.tf_changes, tf.tf_changes);
    assert_eq!(decoded_tf.tf_select, tf.tf_select);

    assert_eq!(decode_spread(&mut dec), spread);
    assert!(!dec.has_error(), "decoder latched an error");
}

/// Stereo frame: both channels' coarse energy interleave through the
/// same chain, inter mode, and recover bit-exactly.
#[test]
fn stereo_frame_coarse_energy_roundtrip() {
    let header = CeltFrameHeader {
        silence: false,
        post_filter: None,
        transient: false,
        intra: false,
        anti_collapse_on: None,
    };
    let lm = 3u8; // 20 ms
    let coded_bands = NUM_BANDS;
    let target = stereo_target();
    let tf = TfParameters {
        tf_changes: vec![false; coded_bands],
        tf_select: 0,
        tf_select_decoded: false,
    };
    let spread = Spread::Light;

    let mut enc = RangeEncoder::new();
    let mut enc_state = CoarseEnergyState::new();
    header.encode_prefix(&mut enc).unwrap();
    encode_coarse_energy(
        &mut enc,
        &mut enc_state,
        &target,
        header.intra,
        lm as u32,
        0,
        NUM_BANDS,
        2,
        BUDGET_BITS,
    )
    .unwrap();
    encode_tf_parameters(&mut enc, &tf, header.transient, lm).unwrap();
    encode_spread(&mut enc, spread).unwrap();
    let mut frame = enc.finish();
    assert!(frame.len() <= FRAME_BYTES);
    frame.resize(FRAME_BYTES, 0);

    let mut dec = RangeDecoder::new(&frame);
    let decoded_header = CeltFrameHeader::decode_prefix(&mut dec);
    let mut dec_state = CoarseEnergyState::new();
    decode_coarse_energy(
        &mut dec,
        &mut dec_state,
        decoded_header.intra,
        lm as u32,
        0,
        NUM_BANDS,
        2,
    )
    .unwrap();
    for c in 0..2 {
        assert_eq!(
            enc_state.energy[c], dec_state.energy[c],
            "stereo coarse-energy channel {c} mismatch"
        );
    }
    let decoded_tf = decode_tf_parameters(&mut dec, 0, coded_bands, header.transient, lm);
    assert_eq!(decoded_tf.tf_changes, tf.tf_changes);
    assert_eq!(decode_spread(&mut dec), spread);
    assert!(!dec.has_error());
}

/// Two consecutive frames share a `CoarseEnergyState`: the second
/// frame's inter-frame time-arm prediction runs against the first
/// frame's reconstructed energies. Encoder and decoder must carry the
/// identical state across the boundary.
#[test]
fn consecutive_frames_share_prediction_state() {
    let lm = 2u8;
    let targets = [mono_target(), {
        let mut t = mono_target();
        for (b, e) in t[0].iter_mut().enumerate() {
            *e += 0.5 - 0.05 * b as f32;
        }
        t
    }];

    let mut enc_state = CoarseEnergyState::new();
    let mut dec_state = CoarseEnergyState::new();

    for (i, target) in targets.iter().enumerate() {
        // First frame intra, second frame inter (uses the carried state).
        let intra = i == 0;
        let header = CeltFrameHeader {
            silence: false,
            post_filter: None,
            transient: false,
            intra,
            anti_collapse_on: None,
        };
        let mut enc = RangeEncoder::new();
        header.encode_prefix(&mut enc).unwrap();
        encode_coarse_energy(
            &mut enc,
            &mut enc_state,
            target,
            intra,
            lm as u32,
            0,
            NUM_BANDS,
            1,
            BUDGET_BITS,
        )
        .unwrap();
        let mut frame = enc.finish();
        frame.resize(FRAME_BYTES, 0);

        let mut dec = RangeDecoder::new(&frame);
        let decoded_header = CeltFrameHeader::decode_prefix(&mut dec);
        assert_eq!(decoded_header.intra, intra);
        decode_coarse_energy(&mut dec, &mut dec_state, intra, lm as u32, 0, NUM_BANDS, 1).unwrap();
        assert_eq!(
            enc_state.energy[0], dec_state.energy[0],
            "frame {i} coarse-energy state diverged"
        );
        assert!(!dec.has_error());
    }
}
