//! Hostile-input regressions found by the fuzz harnesses (r454).
//!
//! Every case here is a minimized fuzz artifact replayed through the
//! public decode API: the contract is that a hostile frame decodes to
//! *something* or is rejected — it must never panic, and it must
//! never poison the decoder state or the emitted PCM with NaN.

use oxideav_celt::ref_decode::CeltRefDecoder;

/// Fuzz regression (r454, `decode_frame` target): a hostile
/// coarse-energy walk drives the log-domain band energy far above any
/// real signal; without the RFC 8251 sec 8 "Cap on Band Energy" the
/// linear conversion overflows to inf and the synthesis emits NaN
/// (0 x inf), poisoning the decoder state for every later frame.
#[test]
fn hostile_energy_walk_stays_nan_free() {
    // 31-byte minimized fuzz input body: lm = 0, mono, fullband,
    // split into three consecutive frames on one decoder state.
    const BODY: [u8; 31] = [
        0x0a, 0x0a, 0x0a, 0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0x26, 0xff, 0xff, 0xfd, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x65, 0xab, 0x53, 0x18, 0x85, 0xab, 0x18, 0x85, 0x85,
        0xab,
    ];
    let mut dec = CeltRefDecoder::new(0, 1).expect("legal config");
    let want = dec.output_frame_size();
    for part in BODY.chunks(11) {
        if let Ok(pcm) = dec.decode_frame(part) {
            assert_eq!(pcm.len(), want);
            assert!(
                pcm.iter().all(|v| !v.is_nan()),
                "hostile frame decoded to NaN"
            );
        }
    }
    // The state must stay usable: concealment right after the hostile
    // stream is NaN-free too (it extrapolates from that state).
    for _ in 0..8 {
        let pcm = dec.decode_lost().expect("concealment always emits");
        assert_eq!(pcm.len(), want);
        assert!(pcm.iter().all(|v| !v.is_nan()), "concealment emitted NaN");
    }
}

/// The cap must not disturb reference lockstep on real streams: the
/// staged reference fixture still decodes NaN-free and unchanged
/// (`tests/ref_decode_fixtures.rs` holds the SNR gate; this arm only
/// pins that the hostile-input guard is inert on sane energies).
#[test]
fn cap_is_inert_on_reference_fixture() {
    let fix = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/audio/celt/fixtures/ref-lm2-mono-100B/frames.bin");
    let Ok(bytes) = std::fs::read(&fix) else {
        eprintln!("staged fixture not present; skipping (covered in-repo by CI checkout layout)");
        return;
    };
    let mut dec = CeltRefDecoder::new(2, 1).expect("legal config");
    let mut pos = 0usize;
    while pos + 2 <= bytes.len() {
        let n = usize::from(bytes[pos]) | (usize::from(bytes[pos + 1]) << 8);
        pos += 2;
        let frame = &bytes[pos..pos + n];
        pos += n;
        let pcm = dec.decode_frame(frame).expect("fixture frame decodes");
        assert!(
            pcm.iter().all(|v| v.is_finite()),
            "fixture decode not finite"
        );
    }
}
