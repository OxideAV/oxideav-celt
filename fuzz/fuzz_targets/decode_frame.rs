#![no_main]

//! Coverage-guided harness for the raw-CELT-frame decode path:
//! `CeltRefDecoder::decode_frame` across the standard-mode band
//! windows (fullband, the RFC 6716 §3.1 NB/WB/SWB bandwidths, and
//! the Hybrid `start = 17` layer).
//!
//! The input is carved into up to 8 consecutive frames fed to ONE
//! stateful decoder, so cross-frame state (§4.3.2.1 coarse-energy
//! prediction, MDCT overlap, §4.3.7.1 post-filter history, the
//! anti-collapse energy history) is exercised, not just single-shot
//! decode.
//!
//! Contract under test: every byte sequence produces `Ok(pcm)` (one
//! frame of the exact configured size, NaN-free) or `Err(..)`.
//! Panics, debug-mode integer overflows, and index-out-of-bounds are
//! all bugs — a hostile frame must be decoded to *something* or
//! rejected, never crash the decoder or poison its state with NaN.

use libfuzzer_sys::fuzz_target;
use oxideav_celt::ref_decode::CeltRefDecoder;

/// `(start, end)` coded-band windows: fullband, the §3.1 CELT-mode
/// bandwidths (NB/WB/SWB), and the Hybrid-mode CELT layer.
const BANDS: [(usize, usize); 6] = [(0, 21), (0, 13), (0, 17), (0, 19), (17, 21), (17, 19)];

fuzz_target!(|data: &[u8]| {
    let (Some(&b0), Some(&b1)) = (data.first(), data.get(1)) else {
        return;
    };
    let lm = u32::from(b0 & 3);
    let channels = 1 + usize::from((b0 >> 2) & 1);
    let (start, end) = BANDS[usize::from(b0 >> 3) % BANDS.len()];
    let n = 1 + usize::from(b1 & 7);
    let mut dec = CeltRefDecoder::new_with_bands(lm, channels, start, end).expect("legal config");
    let want = dec.output_frame_size() * channels;
    let body = &data[2..];
    if body.is_empty() {
        // The wire limits are 1..=1275 bytes: empty must be rejected.
        assert!(dec.decode_frame(body).is_err(), "empty frame rejected");
        return;
    }
    let chunk = body.len().div_ceil(n).max(1);
    for part in body.chunks(chunk) {
        if let Ok(pcm) = dec.decode_frame(part) {
            assert_eq!(pcm.len(), want, "decode emits exactly one frame");
            assert!(
                pcm.iter().all(|v| !v.is_nan()),
                "hostile input must not poison the PCM with NaN"
            );
        }
    }
});
