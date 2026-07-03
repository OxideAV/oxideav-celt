//! CELT long-MDCT analysis spine — time-domain PCM → coded-window
//! MDCT spectrum (RFC 6716 §4.3.7 forward direction, per §5.3).
//!
//! ## What this module covers
//!
//! [`crate::synthesis::LongMdctSynthesis`] is the decode-side spine:
//! coded-window residual spectrum → full `120 << lm`-bin MDCT spectrum
//! → §4.3.7 inverse MDCT + weighted overlap-add → PCM. This module is
//! its exact mirror for the encode direction (§5.3: "the filters and
//! rotations in the encoder are simply the inverse of the operation
//! performed by the decoder"):
//!
//! 1. [`LongMdctAnalysis`] windows the sliding `[history | input]`
//!    block with the same fixed-overlap §4.3.7 window
//!    ([`CELT_OVERLAP`](crate::synthesis::CELT_OVERLAP) = 120) and runs
//!    the forward MDCT ([`crate::mdct::MdctAnalysis`]), producing the
//!    full `120 << lm`-bin spectrum per frame.
//! 2. [`extract_coded_spectrum`] slices the coded-band window
//!    `[band_edge(start), band_edge(end))` out of that full spectrum —
//!    the exact inverse of
//!    [`place_residual_spectrum`](crate::synthesis::place_residual_spectrum).
//!    The bins above the `100 << lm` coding top (and below a Hybrid
//!    `start`) are **dropped**: §4.3 Table 55 codes only the lower
//!    `100 << lm` bins, so the encoder discards the remainder — the
//!    same bins the decode side reconstructs as zero.
//!
//! The output of [`LongMdctAnalysis::analyze`] is band-contiguous in
//! the [`crate::residual::ResidualSpectrum`] layout — exactly what
//! [`crate::frame_encode::encode_celt_frame`] consumes.
//!
//! ## Why long-MDCT only
//!
//! The transient (short-block) analysis needs the §4.3.1 per-frame
//! short-MDCT geometry and the within-frame frequency-vector
//! interleave, which RFC 6716 defers to the reference implementation —
//! the same documented docs gap the synthesis spine keeps. This spine
//! therefore handles only the single-long-MDCT case.
//!
//! ## Clean-room provenance
//!
//! The `120 << lm` MDCT size, the `100 << lm` coded top, the fixed
//! 120-sample overlap, and the window construction are RFC 6716 §4.3
//! Table 55 + §4.3.7 (`docs/audio/opus/rfc6716-opus.txt`), the same
//! sections the synthesis spine transcribes; the forward direction is
//! the §5.3 encoder-is-inverse prose. No external library source was
//! consulted.

use crate::band_layout::{band_edge, coded_total_bins};
use crate::coarse_energy::NUM_BANDS;
use crate::mdct::{build_low_overlap_window_f32, MdctAnalysis};
use crate::synthesis::{mdct_size, CELT_OVERLAP};
use crate::Error;

/// Slice the coded-band window out of a full per-channel MDCT
/// spectrum — the exact inverse of
/// [`place_residual_spectrum`](crate::synthesis::place_residual_spectrum).
///
/// `spectrum` is the full `120 << lm`-bin forward-MDCT output; the
/// returned vector holds the `coded_total_bins(start, end, lm)` bins
/// of the absolute range `[band_edge(start), band_edge(end))`. The
/// remaining bins (below a Hybrid `start` and the `20 << lm`-bin gap
/// above the `100 << lm` coding top) are dropped — they are not
/// band-coded (RFC 6716 §4.3 Table 55) and the decode side
/// reconstructs them as zero.
///
/// Returns [`Error::InvalidParameter`] when `lm > 3`, the window is
/// out of range, or `spectrum.len() != 120 << lm`.
pub fn extract_coded_spectrum(
    spectrum: &[f32],
    lm: u32,
    start: usize,
    end: usize,
) -> Result<Vec<f32>, Error> {
    if lm > 3 || start > end || end > NUM_BANDS {
        return Err(Error::InvalidParameter);
    }
    let full = mdct_size(lm).ok_or(Error::InvalidParameter)?;
    if spectrum.len() != full {
        return Err(Error::InvalidParameter);
    }
    let coded = coded_total_bins(start, end, lm).ok_or(Error::InvalidParameter)? as usize;
    let lo = band_edge(start, lm).ok_or(Error::InvalidParameter)? as usize;
    Ok(spectrum[lo..lo + coded].to_vec())
}

/// Streaming long-MDCT **analysis** state — the encode-direction
/// mirror of [`LongMdctSynthesis`](crate::synthesis::LongMdctSynthesis).
///
/// Wraps a [`MdctAnalysis`] and the fixed-overlap §4.3.7 window for a
/// fixed frame-size shift `lm`. Each [`analyze`](Self::analyze) call
/// consumes `mdct_size(lm)` new PCM samples (pre-emphasized per
/// §4.3.7.2 — the caller runs [`crate::deemphasis::Preemphasis`]
/// first) and emits the coded-window MDCT spectrum, the
/// band-contiguous input [`crate::frame_encode::encode_celt_frame`]
/// takes.
///
/// The analysis inherits [`MdctAnalysis`]'s one-frame-delay latency
/// contract: the spectrum emitted at call `t` represents the block
/// covering input frames `t-1` and `t`, and the matching decode-side
/// synthesis finishes that block's first half at its call `t` — so a
/// full PCM → encode → decode → PCM chain reproduces the input with
/// exactly one frame of delay (plus quantization).
#[derive(Debug, Clone)]
pub struct LongMdctAnalysis {
    lm: u32,
    /// The `2 * mdct_size(lm)`-sample §4.3.7 window (fixed-overlap
    /// 120) — identical to the synthesis window.
    window: Vec<f32>,
    ana: MdctAnalysis,
}

impl LongMdctAnalysis {
    /// Create a long-MDCT analysis state for frame-size shift `lm`
    /// (`0..=3`), with a zero input history.
    ///
    /// Returns `None` if `lm > 3`.
    pub fn new(lm: u32) -> Option<Self> {
        let n = mdct_size(lm)?;
        let window = build_low_overlap_window_f32(n, CELT_OVERLAP)?;
        Some(Self {
            lm,
            window,
            ana: MdctAnalysis::new(n),
        })
    }

    /// The frame-size shift this state was created for.
    #[inline]
    pub fn lm(&self) -> u32 {
        self.lm
    }

    /// The per-channel MDCT size (`120 << lm`) — the number of PCM
    /// samples [`analyze`](Self::analyze) consumes per frame.
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.ana.frame_size()
    }

    /// Zero the carried input history (encoder reset, the mirror of
    /// the §4.5.2 decoder reset).
    pub fn reset(&mut self) {
        self.ana.reset();
    }

    /// Analyze one frame: window the sliding `[history | pcm]` block,
    /// forward-MDCT it, and return the **full** `120 << lm`-bin
    /// spectrum.
    ///
    /// Returns [`Error::InvalidParameter`] when
    /// `pcm.len() != mdct_size(lm)`. The history advances only on
    /// success.
    pub fn analyze_full(&mut self, pcm: &[f32]) -> Result<Vec<f32>, Error> {
        let n = self.ana.frame_size();
        if pcm.len() != n {
            return Err(Error::InvalidParameter);
        }
        let mut spectrum = vec![0.0f32; n];
        if !self.ana.frame(pcm, &self.window, &mut spectrum) {
            return Err(Error::InvalidParameter);
        }
        Ok(spectrum)
    }

    /// Analyze one frame and return the **coded-window** spectrum for
    /// the band window `[start, end)` — [`analyze_full`](Self::analyze_full)
    /// followed by [`extract_coded_spectrum`]. This is the
    /// band-contiguous residual-layout input
    /// [`encode_celt_frame`](crate::frame_encode::encode_celt_frame)
    /// consumes.
    ///
    /// Returns [`Error::InvalidParameter`] on a PCM length mismatch or
    /// an out-of-range band window; the history advances only on
    /// success.
    pub fn analyze(&mut self, pcm: &[f32], start: usize, end: usize) -> Result<Vec<f32>, Error> {
        // Validate the window before advancing the streaming history.
        if start > end || end > NUM_BANDS {
            return Err(Error::InvalidParameter);
        }
        let full = self.analyze_full(pcm)?;
        extract_coded_spectrum(&full, self.lm, start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::{place_residual_spectrum, LongMdctSynthesis};

    /// Deterministic pseudo-random signal in [-1, 1).
    fn test_signal(len: usize, mut seed: u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// `extract_coded_spectrum` is the exact inverse of
    /// `place_residual_spectrum` over every `lm` and both the pure-CELT
    /// and Hybrid windows.
    #[test]
    fn extract_is_inverse_of_place() {
        for lm in 0..=3u32 {
            for &(start, end) in &[(0usize, NUM_BANDS), (17, NUM_BANDS), (5, 12)] {
                let coded = coded_total_bins(start, end, lm).unwrap() as usize;
                let residual = test_signal(coded, 0x1234 + lm * 7 + start as u32);
                let full = place_residual_spectrum(&residual, lm, start, end).unwrap();
                let back = extract_coded_spectrum(&full, lm, start, end).unwrap();
                assert_eq!(back.len(), coded);
                for (i, (a, b)) in residual.iter().zip(&back).enumerate() {
                    assert_eq!(a, b, "lm={lm} window=({start},{end}) bin {i}");
                }
            }
        }
    }

    /// Out-of-range parameters and length mismatches are rejected.
    #[test]
    fn extract_rejects_bad_inputs() {
        let full = vec![0.0f32; 120];
        assert!(extract_coded_spectrum(&full, 4, 0, NUM_BANDS).is_err());
        assert!(extract_coded_spectrum(&full, 0, 5, 3).is_err());
        assert!(extract_coded_spectrum(&full, 0, 0, NUM_BANDS + 1).is_err());
        // Wrong spectrum length for lm=1 (needs 240).
        assert!(extract_coded_spectrum(&full, 1, 0, NUM_BANDS).is_err());
    }

    /// The full analysis → synthesis loop recovers the coded-window
    /// spectra with one frame of delay, at every `lm`: synthesize PCM
    /// from known band-limited spectra, analyze it back, compare.
    #[test]
    fn analysis_recovers_synthesized_coded_spectra_all_lm() {
        for lm in 0..=3u32 {
            let coded = coded_total_bins(0, NUM_BANDS, lm).unwrap() as usize;
            let frames = 4usize;
            let spectra: Vec<Vec<f32>> = (0..frames)
                .map(|t| test_signal(coded, 0xC0DE + lm * 31 + t as u32))
                .collect();
            let mut synth = LongMdctSynthesis::new(lm).unwrap();
            let mut ana = LongMdctAnalysis::new(lm).unwrap();
            assert_eq!(ana.lm(), lm);
            assert_eq!(ana.frame_size(), synth.frame_size());
            let mut recovered: Vec<Vec<f32>> = Vec::new();
            for spec in &spectra {
                let pcm = synth.synthesize(spec, 0, NUM_BANDS).unwrap();
                recovered.push(ana.analyze(&pcm, 0, NUM_BANDS).unwrap());
            }
            // Analysis frame t sees synthesis block t-1; block t-1 is
            // fully reconstructed once its first half overlapped block
            // t-2's tail, so from t = 2 the recovery is exact.
            for t in 2..frames {
                for k in 0..coded {
                    assert!(
                        (recovered[t][k] - spectra[t - 1][k]).abs() <= 2e-4,
                        "lm={lm} frame {} bin {k}: {} vs {}",
                        t - 1,
                        recovered[t][k],
                        spectra[t - 1][k]
                    );
                }
            }
        }
    }

    /// A Hybrid-window analysis extracts only the bands-17..21 bins:
    /// synthesizing from a Hybrid residual and analyzing with the same
    /// window recovers the residual (steady state).
    #[test]
    fn hybrid_window_analysis_round_trip() {
        let lm = 2u32;
        let (start, end) = (17usize, NUM_BANDS);
        let coded = coded_total_bins(start, end, lm).unwrap() as usize;
        let frames = 4usize;
        let spectra: Vec<Vec<f32>> = (0..frames)
            .map(|t| test_signal(coded, 0xAB0 + t as u32))
            .collect();
        let mut synth = LongMdctSynthesis::new(lm).unwrap();
        let mut ana = LongMdctAnalysis::new(lm).unwrap();
        let mut recovered: Vec<Vec<f32>> = Vec::new();
        for spec in &spectra {
            let pcm = synth.synthesize(spec, start, end).unwrap();
            recovered.push(ana.analyze(&pcm, start, end).unwrap());
        }
        for t in 2..frames {
            for k in 0..coded {
                assert!(
                    (recovered[t][k] - spectra[t - 1][k]).abs() <= 2e-4,
                    "hybrid frame {} bin {k}",
                    t - 1
                );
            }
        }
    }

    /// The uncoded top bins (`100 << lm ..= 120 << lm`) of a signal
    /// synthesized from band-coded spectra analyze back to (near)
    /// zero — the energy the encoder drops is exactly the energy the
    /// synthesis never produced.
    #[test]
    fn uncoded_top_bins_are_zero_for_band_limited_input() {
        let lm = 1u32;
        let coded = coded_total_bins(0, NUM_BANDS, lm).unwrap() as usize;
        let mut synth = LongMdctSynthesis::new(lm).unwrap();
        let mut ana = LongMdctAnalysis::new(lm).unwrap();
        for t in 0..4 {
            let spec = test_signal(coded, 0xFACE + t);
            let pcm = synth.synthesize(&spec, 0, NUM_BANDS).unwrap();
            let full = ana.analyze_full(&pcm).unwrap();
            if t >= 2 {
                for (k, &v) in full.iter().enumerate().skip(coded) {
                    assert!(
                        v.abs() <= 2e-4,
                        "top bin {k} should be ~0 for band-limited input, got {v}"
                    );
                }
            }
        }
    }

    /// PCM length mismatches and bad windows are rejected without
    /// advancing the streaming history.
    #[test]
    fn analyze_rejects_bad_inputs_without_state_advance() {
        let mut ana = LongMdctAnalysis::new(0).unwrap();
        let n = ana.frame_size();
        assert_eq!(n, 120);
        let pcm = test_signal(n, 0x77);
        // Wrong PCM length.
        assert!(ana.analyze(&pcm[..n - 1], 0, NUM_BANDS).is_err());
        // Bad band window (validated before the history advances).
        assert!(ana.analyze(&pcm, 5, 3).is_err());
        assert!(ana.analyze(&pcm, 0, NUM_BANDS + 1).is_err());
        // The failed calls must not have advanced the history: a fresh
        // state produces the identical first-frame spectrum.
        let got = ana.analyze(&pcm, 0, NUM_BANDS).unwrap();
        let mut fresh = LongMdctAnalysis::new(0).unwrap();
        let expected = fresh.analyze(&pcm, 0, NUM_BANDS).unwrap();
        assert_eq!(got, expected);
        // lm out of range at construction.
        assert!(LongMdctAnalysis::new(4).is_none());
    }

    /// `reset()` restores the fresh-state output.
    #[test]
    fn reset_restores_fresh_state() {
        let mut ana = LongMdctAnalysis::new(1).unwrap();
        let pcm = test_signal(ana.frame_size(), 0x99);
        let first = ana.analyze(&pcm, 0, NUM_BANDS).unwrap();
        let second = ana.analyze(&pcm, 0, NUM_BANDS).unwrap();
        assert_ne!(first, second, "history must influence the analysis");
        ana.reset();
        let after_reset = ana.analyze(&pcm, 0, NUM_BANDS).unwrap();
        assert_eq!(first, after_reset);
    }
}
