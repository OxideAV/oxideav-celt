//! Integration test: the §4.3 CELT control-symbol *encoders* compose
//! into one frame that the matching *decoders* read back bit-exactly.
//!
//! Each encode primitive landed this round is the inverse of an
//! existing decode primitive. This test chains a representative subset
//! of them in their RFC 6716 Table-56 relative order — frame prefix
//! (§4.3 / §4.3.7.1), time-frequency parameters (§4.3.4.5), spread
//! (§4.3.4.3), band-allocation fields (§4.3.3), fine energy (§4.3.2.2),
//! and a PVQ band shape (§4.3.4.2) — into a single `RangeEncoder`
//! frame, then walks the decoders through the same sequence and asserts
//! every value is recovered. The coarse-energy §4.3.2.1 Laplace block
//! that sits between the prefix and the TF parameters in a real frame is
//! DOCS-GAP blocked on the encode side, so it is omitted; this test
//! exercises the contiguous chain of fully-specified encoders.

use oxideav_celt::{
    decode_band_allocation, decode_fine_energy_band, decode_pulses, decode_spread,
    decode_tf_parameters, encode_band_allocation, encode_fine_energy_band, encode_shape,
    encode_spread, encode_tf_parameters, fine_correction_q14, BandAllocation, BandAllocationGates,
    CeltFrameHeader, PostFilter, RangeDecoder, RangeEncoder, Spread, TfParameters,
};

#[test]
fn full_control_chain_encode_then_decode() {
    // --- build the control values an encoder would choose ---
    let header = CeltFrameHeader {
        silence: false,
        post_filter: Some(PostFilter {
            octave: 3,
            period: 200,
            gain: 4,
            tapset: 1,
        }),
        transient: false,
        intra: true,
        anti_collapse_on: None,
    };
    let lm = 2u8; // 10 ms
    let coded_bands = 5usize;
    let tf = TfParameters {
        tf_changes: vec![false, true, true, false, true],
        tf_select: 0,
        tf_select_decoded: false,
    };
    let spread = Spread::Aggressive;
    let gates = BandAllocationGates {
        trim_gated: true,
        skip_gated: true,
        intensity_gated: false,
        dual_gated: false,
        coded_bands: coded_bands as u32,
    };
    let alloc = BandAllocation {
        alloc_trim: 7,
        skip: true,
        intensity_band_offset: 0,
        dual_stereo: false,
    };
    // A fine-energy correction per coded band, with per-band bit widths.
    let fine_bits = [3u32, 2, 4, 1, 5];
    let fine_f = [5u32, 1, 9, 0, 17];
    // One PVQ band shape (input vector → quantised pulses).
    let shape_x = [0.6f32, -0.2, 0.1, -0.5, 0.3, 0.05, -0.1, 0.2];
    let shape_n = shape_x.len() as u32;
    let shape_k = 4u32;

    // --- encode the whole chain into one frame ---
    let mut enc = RangeEncoder::new();
    header.encode_prefix(&mut enc).unwrap();
    encode_tf_parameters(&mut enc, &tf, header.transient, lm).unwrap();
    encode_spread(&mut enc, spread).unwrap();
    encode_band_allocation(&mut enc, gates, &alloc).unwrap();
    for (&f, &b) in fine_f.iter().zip(fine_bits.iter()) {
        encode_fine_energy_band(&mut enc, f, b).unwrap();
    }
    let quantised = encode_shape(&mut enc, &shape_x, shape_n, shape_k).unwrap();
    let frame = enc.finish();

    // --- decode the chain back and assert every value matches ---
    let mut dec = RangeDecoder::new(&frame);

    let decoded_header = CeltFrameHeader::decode_prefix(&mut dec);
    assert_eq!(decoded_header.silence, header.silence);
    assert_eq!(decoded_header.post_filter, header.post_filter);
    assert_eq!(decoded_header.transient, header.transient);
    assert_eq!(decoded_header.intra, header.intra);

    let decoded_tf = decode_tf_parameters(&mut dec, 0, coded_bands, header.transient, lm);
    assert_eq!(decoded_tf.tf_changes, tf.tf_changes);
    assert_eq!(decoded_tf.tf_select, tf.tf_select);

    let decoded_spread = decode_spread(&mut dec);
    assert_eq!(decoded_spread, spread);

    let decoded_alloc = decode_band_allocation(&mut dec, gates);
    assert_eq!(decoded_alloc.alloc_trim, alloc.alloc_trim);
    assert_eq!(decoded_alloc.skip, alloc.skip);

    for (&f, &b) in fine_f.iter().zip(fine_bits.iter()) {
        let correction = decode_fine_energy_band(&mut dec, b);
        assert_eq!(
            correction,
            fine_correction_q14(f, b),
            "fine band b={b} f={f}"
        );
    }

    let decoded_pulses = decode_pulses(&mut dec, shape_n, shape_k).unwrap();
    assert_eq!(decoded_pulses, quantised, "PVQ shape mismatch");

    assert!(
        !dec.has_error(),
        "decoder latched an error walking the chain"
    );
}

/// The same chain with a transient header (so the TF PDFs and the
/// anti-collapse bit path differ), and a post-filter-off header.
#[test]
fn transient_no_postfilter_control_chain() {
    let header = CeltFrameHeader {
        silence: false,
        post_filter: None,
        transient: true,
        intra: false,
        anti_collapse_on: None,
    };
    let lm = 3u8;
    let tf = TfParameters {
        tf_changes: vec![true, false, true],
        tf_select: 1,
        tf_select_decoded: true,
    };

    let mut enc = RangeEncoder::new();
    header.encode_prefix(&mut enc).unwrap();
    encode_tf_parameters(&mut enc, &tf, header.transient, lm).unwrap();
    encode_spread(&mut enc, Spread::Normal).unwrap();
    let frame = enc.finish();

    let mut dec = RangeDecoder::new(&frame);
    let decoded_header = CeltFrameHeader::decode_prefix(&mut dec);
    assert_eq!(decoded_header.post_filter, None);
    assert!(decoded_header.transient);
    assert!(!decoded_header.intra);

    let decoded_tf = decode_tf_parameters(&mut dec, 0, tf.tf_changes.len(), header.transient, lm);
    assert_eq!(decoded_tf.tf_changes, tf.tf_changes);
    // tf_select is only recovered if the gate is open for this pattern.
    if decoded_tf.tf_select_decoded {
        assert_eq!(decoded_tf.tf_select, tf.tf_select);
    }

    assert_eq!(decode_spread(&mut dec), Spread::Normal);
    assert!(!dec.has_error());
}
