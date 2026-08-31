#![no_main]

//! Differential harness for the two decode output paths: the same
//! frame bytes through the full-rate (48 kHz) decoder and through
//! the reduced-rate decoder (`new_downsampled` — spectrum bounded to
//! the output Nyquist, decimated de-emphasis).
//!
//! The two paths parse IDENTICAL symbols — downsampling only changes
//! the synthesis tail — so their Ok/Err verdicts must stay in
//! lockstep on every input, hostile or not, and their output lengths
//! must relate by exactly the resampling factor (concealment
//! included). Any divergence is a real decoder bug.

use libfuzzer_sys::fuzz_target;
use oxideav_celt::ref_decode::{resampling_factor, CeltRefDecoder};

/// Reduced output rates under test (the §4.2.9 ladder below 48 kHz).
const RATES: [u32; 4] = [24_000, 16_000, 12_000, 8_000];

fuzz_target!(|data: &[u8]| {
    let Some(&b0) = data.first() else {
        return;
    };
    let lm = u32::from(b0 & 3);
    let channels = 1 + usize::from((b0 >> 2) & 1);
    let rate = RATES[usize::from(b0 >> 3) % RATES.len()];
    let factor = resampling_factor(rate).expect("ladder rate");
    let mut full = CeltRefDecoder::new(lm, channels).expect("legal config");
    let mut down = CeltRefDecoder::new_downsampled(lm, channels, rate).expect("legal config");
    let body = &data[1..];
    let chunk = body.len().div_ceil(4).max(1);
    for part in body.chunks(chunk) {
        let a = full.decode_frame(part);
        let b = down.decode_frame(part);
        assert_eq!(
            a.is_ok(),
            b.is_ok(),
            "full-rate and reduced-rate verdicts must stay in lockstep"
        );
        if let (Ok(a), Ok(b)) = (a, b) {
            assert_eq!(a.len(), b.len() * factor, "outputs relate by the factor");
        }
    }
    // Concealment runs on both timelines too.
    let a = full.decode_lost().expect("concealment always emits");
    let b = down.decode_lost().expect("concealment always emits");
    assert_eq!(
        a.len(),
        b.len() * factor,
        "PLC outputs relate by the factor"
    );
});
