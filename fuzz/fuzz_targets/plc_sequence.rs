#![no_main]

//! Coverage-guided harness for packet-loss concealment sequences:
//! arbitrary interleavings of `decode_frame` (possibly-hostile
//! frames) and `decode_lost` on one stateful `CeltRefDecoder`.
//!
//! The 16-step loss mask reaches every concealment regime — the
//! pitch-locked LPC extrapolation of the first five losses, the
//! comfort-noise fallback beyond (and on Hybrid `start = 17`
//! streams, which take it immediately), and recovery frames decoded
//! after a concealed stretch — including on the reduced-rate
//! (downsampled-output) decoder timeline.
//!
//! Contract under test: `decode_lost` ALWAYS emits exactly one frame
//! of NaN-free PCM (there is no invalid input to concealment), and
//! no decode/loss interleaving panics, overflows, or leaves state
//! that crashes a later call.

use libfuzzer_sys::fuzz_target;
use oxideav_celt::ref_decode::CeltRefDecoder;

/// §4.2.9-style output rates: the concealment walk also runs on the
/// decimated de-emphasis timeline.
const RATES: [u32; 5] = [48_000, 24_000, 16_000, 12_000, 8_000];

fuzz_target!(|data: &[u8]| {
    let (Some(&b0), Some(&b1), Some(&b2)) = (data.first(), data.get(1), data.get(2)) else {
        return;
    };
    let lm = u32::from(b0 & 3);
    let channels = 1 + usize::from((b0 >> 2) & 1);
    // Hybrid streams conceal via comfort noise from the first loss.
    let start = if (b0 >> 3) & 1 == 1 { 17 } else { 0 };
    let rate = RATES[usize::from(b0 >> 4) % RATES.len()];
    let mut dec =
        CeltRefDecoder::new_with_config(lm, channels, start, 0, rate).expect("legal config");
    let want = dec.output_frame_size() * channels;
    let mask = u16::from_le_bytes([b1, b2]);
    let body = &data[3..];
    let chunk = (body.len() / 8).max(1);
    let mut frames = body.chunks(chunk);
    for step in 0..16 {
        let pcm = if (mask >> step) & 1 == 1 {
            dec.decode_lost().expect("concealment always emits a frame")
        } else {
            match frames.next() {
                Some(part) => match dec.decode_frame(part) {
                    Ok(pcm) => pcm,
                    // A rejected frame is a legal outcome; the state
                    // must still carry the rest of the sequence.
                    Err(_) => continue,
                },
                // Out of input: treat the tail of the mask as losses.
                None => dec.decode_lost().expect("concealment always emits a frame"),
            }
        };
        assert_eq!(pcm.len(), want, "one frame per call, at the output rate");
        assert!(
            pcm.iter().all(|v| !v.is_nan()),
            "concealment must not emit NaN"
        );
    }
});
