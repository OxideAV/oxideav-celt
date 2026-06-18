//! CELT long-MDCT synthesis spine — residual spectrum → time-domain
//! PCM (RFC 6716 §4.3.6 → §4.3.7).
//!
//! ## What this module covers
//!
//! [`crate::residual::decode_residual_bands`] produces the denormalized
//! MDCT-domain spectrum for the coded-band window `[start, end)`.
//! [`crate::mdct::MdctSynthesis`] runs the §4.3.7 inverse MDCT and the
//! weighted overlap-add (WOLA) for one MDCT block. What was missing
//! between them was the **spectrum-placement spine**: a driver that maps
//! the coded-window residual spectrum into the full per-channel MDCT
//! spectrum the §4.3.7 inverse MDCT actually transforms, then runs the
//! WOLA to emit the frame's time-domain samples.
//!
//! [`LongMdctSynthesis`] is that driver for the **long-MDCT** (i.e.
//! non-transient, single-MDCT) case. For each frame it:
//!
//! 1. Allocates the full per-channel MDCT spectrum of [`mdct_size`] bins
//!    (`120 << lm`) — the size the §4.3.7 inverse MDCT operates on.
//! 2. Copies the coded residual spectrum (length
//!    [`coded_total_bins(start, end, lm)`](crate::band_layout::coded_total_bins))
//!    into its absolute bin range `[band_edge(start), band_edge(end))`,
//!    leaving the uncoded low bins (below the Hybrid `start`) and the
//!    `20 << lm`-bin gap above the 20 kHz coding limit at zero (§4.3 Table
//!    55: the coded range tops out at `100 << lm`, while the MDCT spans
//!    `120 << lm`).
//! 3. Runs [`MdctSynthesis::frame`](crate::mdct::MdctSynthesis::frame)
//!    with the fixed-overlap §4.3.7 window ([`CELT_OVERLAP`] = 120),
//!    emitting `mdct_size(lm)` finished output samples.
//!
//! The emitted samples are exactly "the output of the inverse MDCT
//! (after weighted overlap-add)" that §4.3.7.1 names as the post-filter's
//! input, so a caller chains [`crate::post_filter`] then
//! [`crate::deemphasis`] onto this output to finish the §4.3.7 chain.
//!
//! ## Why long-MDCT only
//!
//! When the transient flag is set, the frame's spectrum represents
//! several short MDCTs (§4.3.1), each transformed and overlap-added
//! separately. The precise within-frame layout that maps the band-loop
//! output to the per-short-block frequency vectors — and the
//! inter-short-block overlap-add scheme — is delegated to the reference
//! implementation by §4.3.1 ("defined in ... celt.c") and §4.3.7
//! ("performed by mdct_backward (mdct.c)"); the RFC narrative does not
//! pin it. This spine therefore handles only the unambiguous
//! single-long-MDCT case (`nb_blocks == 1`), where the residual spectrum
//! maps one-to-one onto the §4.3.7 inverse-MDCT input with no
//! interleaving. The transient short-block reassembly remains a
//! documented docs gap, the same boundary
//! [`crate::residual::decode_residual_bands`] keeps for the
//! per-short-block geometry.
//!
//! ## Clean-room provenance
//!
//! The `120 << lm` MDCT size and the `100 << lm` coded-range top are RFC
//! 6716 §4.3 Table 55 / its surrounding narrative
//! (`docs/audio/opus/rfc6716-opus.txt` lines 5813–5870; the band-layout
//! module records the edge transcription). The fixed 240-sample basic
//! window / 120-sample overlap and the `1/2`-scaled inverse-MDCT WOLA are
//! §4.3.7 (lines 6738–6754). The placement of the coded window into the
//! full spectrum with the uncoded high-frequency gap zeroed is the
//! direct reading of the Table 55 coded-range top vs. the MDCT span. No
//! external library source was consulted.

use crate::band_layout::{band_edge, coded_total_bins};
use crate::coarse_energy::NUM_BANDS;
use crate::mdct::{build_low_overlap_window_f32, MdctSynthesis, BASIC_WINDOW_HALF};
use crate::Error;

/// The fixed CELT MDCT overlap, in per-channel samples (RFC 6716
/// §4.3.7: the window is derived from the basic 240-sample version, so
/// the rising/falling halves — the overlap — are 120 samples each).
pub const CELT_OVERLAP: usize = BASIC_WINDOW_HALF;

/// The full per-channel MDCT size at frame-size shift `lm`, in bins:
/// `120 << lm` (RFC 6716 §4.3 Table 55 narrative — the MDCT spans
/// `120 << lm` bins per channel while only the lower `100 << lm` are
/// band-coded).
///
/// Returns `None` if `lm > 3`.
#[inline]
pub fn mdct_size(lm: u32) -> Option<usize> {
    if lm > 3 {
        return None;
    }
    Some(120usize << lm)
}

/// Place a coded-window residual spectrum into the full per-channel
/// MDCT spectrum (RFC 6716 §4.3.7 inverse-MDCT input).
///
/// `residual` is the denormalized spectrum for the coded-band window
/// `[start, end)` — length [`coded_total_bins(start, end, lm)`]. It is
/// copied into the absolute bin range `[band_edge(start), band_edge(end))`
/// of a freshly-zeroed `120 << lm`-bin buffer; every other bin (the
/// uncoded low bins below a Hybrid `start`, and the `20 << lm`-bin gap
/// above the `100 << lm` coding top) stays zero.
///
/// Returns the full spectrum on success, or [`Error::InvalidParameter`]
/// when `lm > 3`, the window is out of range (`start > end` or
/// `end > 21`), or `residual.len()` disagrees with the window's coded
/// bin count.
pub fn place_residual_spectrum(
    residual: &[f32],
    lm: u32,
    start: usize,
    end: usize,
) -> Result<Vec<f32>, Error> {
    if lm > 3 || start > end || end > NUM_BANDS {
        return Err(Error::InvalidParameter);
    }
    let coded = coded_total_bins(start, end, lm).ok_or(Error::InvalidParameter)? as usize;
    if residual.len() != coded {
        return Err(Error::InvalidParameter);
    }
    let full = mdct_size(lm).ok_or(Error::InvalidParameter)?;
    let lo = band_edge(start, lm).ok_or(Error::InvalidParameter)? as usize;

    let mut spectrum = vec![0.0f32; full];
    spectrum[lo..lo + coded].copy_from_slice(residual);
    Ok(spectrum)
}

/// Streaming long-MDCT synthesis state (RFC 6716 §4.3.6 → §4.3.7).
///
/// Wraps a [`MdctSynthesis`] and the fixed-overlap §4.3.7 window for a
/// fixed frame-size shift `lm`. Each [`synthesize`](Self::synthesize)
/// call places a coded-window residual spectrum into the full MDCT
/// spectrum and runs the inverse MDCT + weighted overlap-add, emitting
/// `mdct_size(lm)` time-domain samples — the input the §4.3.7.1
/// post-filter consumes.
///
/// This handles the non-transient (single long MDCT) case only; the
/// transient short-block reassembly is a documented docs gap (see the
/// module docs).
#[derive(Debug, Clone)]
pub struct LongMdctSynthesis {
    lm: u32,
    /// The `2 * mdct_size(lm)`-sample §4.3.7 synthesis window
    /// (fixed-overlap 120).
    window: Vec<f32>,
    synth: MdctSynthesis,
}

impl LongMdctSynthesis {
    /// Create a long-MDCT synthesis state for frame-size shift `lm`
    /// (`0..=3`), with a zero overlap tail.
    ///
    /// Returns `None` if `lm > 3` (the §4.3.7 window construction is
    /// total for every in-range `lm`, so this is the only failure mode).
    pub fn new(lm: u32) -> Option<Self> {
        let n = mdct_size(lm)?;
        let window = build_low_overlap_window_f32(n, CELT_OVERLAP)?;
        Some(Self {
            lm,
            window,
            synth: MdctSynthesis::new(n),
        })
    }

    /// The frame-size shift this state was created for.
    #[inline]
    pub fn lm(&self) -> u32 {
        self.lm
    }

    /// The per-channel MDCT size (`120 << lm`) — the number of output
    /// samples [`synthesize`](Self::synthesize) emits per frame.
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.synth.frame_size()
    }

    /// Zero the carried overlap tail (decoder reset per §4.5.2).
    pub fn reset(&mut self) {
        self.synth.reset();
    }

    /// Synthesize one long-MDCT frame: place the coded-window residual
    /// spectrum into the full MDCT spectrum, run the §4.3.7 inverse MDCT
    /// with weighted overlap-add, and return the `mdct_size(lm)`
    /// finished time-domain output samples.
    ///
    /// * `residual` — the denormalized spectrum for the coded-band
    ///   window `[start, end)` (the
    ///   [`ResidualSpectrum::samples`](crate::residual::ResidualSpectrum::samples)
    ///   output), length
    ///   [`coded_total_bins(start, end, lm)`](crate::band_layout::coded_total_bins).
    /// * `start` / `end` — the coded-band window, `start <= end <= 21`.
    ///
    /// Returns the WOLA output on success, or [`Error::InvalidParameter`]
    /// when the window / `residual` length is inconsistent (via
    /// [`place_residual_spectrum`]). The overlap tail is advanced only on
    /// success.
    pub fn synthesize(
        &mut self,
        residual: &[f32],
        start: usize,
        end: usize,
    ) -> Result<Vec<f32>, Error> {
        let spectrum = place_residual_spectrum(residual, self.lm, start, end)?;
        let n = self.synth.frame_size();
        let mut out = vec![0.0f32; n];
        // The window/spectrum lengths are guaranteed consistent (both
        // derive from `self.lm`), so `frame` cannot reject them.
        if !self.synth.frame(&spectrum, &self.window, &mut out) {
            return Err(Error::InvalidParameter);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band_layout::coded_total_bins;
    use crate::mdct::imdct_naive_f32;

    #[test]
    fn mdct_size_is_120_shifted() {
        assert_eq!(mdct_size(0), Some(120));
        assert_eq!(mdct_size(1), Some(240));
        assert_eq!(mdct_size(2), Some(480));
        assert_eq!(mdct_size(3), Some(960));
        assert_eq!(mdct_size(4), None);
    }

    #[test]
    fn overlap_is_fixed_120() {
        assert_eq!(CELT_OVERLAP, 120);
    }

    /// Placing a full-band window puts the coded spectrum in the low
    /// `100 << lm` bins and leaves the `20 << lm` high-frequency gap at
    /// zero.
    #[test]
    fn place_full_band_zeros_high_gap() {
        let lm = 1;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        assert_eq!(coded, 100 << lm);
        let residual: Vec<f32> = (0..coded).map(|i| (i as f32) + 1.0).collect();

        let full = place_residual_spectrum(&residual, lm, 0, 21).unwrap();
        assert_eq!(full.len(), mdct_size(lm).unwrap());
        assert_eq!(full.len(), 240);
        // The coded range is copied verbatim.
        assert_eq!(&full[..coded], &residual[..]);
        // The 20 << lm = 40-bin gap above 20 kHz is zero.
        assert!(full[coded..].iter().all(|&x| x == 0.0));
        assert_eq!(full.len() - coded, 20 << lm);
    }

    /// A Hybrid window (start = 17) is placed at its absolute band edge,
    /// with the bins below `start` and the high gap both zeroed.
    #[test]
    fn place_hybrid_window_offsets_to_band_edge() {
        let lm = 2;
        let (start, end) = (17usize, 21usize);
        let coded = coded_total_bins(start, end, lm).unwrap() as usize;
        let residual: Vec<f32> = (0..coded).map(|i| (i as f32) + 0.5).collect();

        let full = place_residual_spectrum(&residual, lm, start, end).unwrap();
        let lo = band_edge(start, lm).unwrap() as usize;
        let hi = band_edge(end, lm).unwrap() as usize;
        // Low bins (below the Hybrid coding start) are silent.
        assert!(full[..lo].iter().all(|&x| x == 0.0));
        // The coded window lands at its absolute edge.
        assert_eq!(&full[lo..hi], &residual[..]);
        // The high gap above the coded top is silent.
        assert!(full[hi..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn place_rejects_inconsistent_length() {
        // Right window, wrong residual length.
        let lm = 0;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let residual = vec![1.0f32; coded - 1];
        assert!(matches!(
            place_residual_spectrum(&residual, lm, 0, 21),
            Err(Error::InvalidParameter)
        ));
    }

    #[test]
    fn place_rejects_invalid_window() {
        assert!(matches!(
            place_residual_spectrum(&[], 4, 0, 0),
            Err(Error::InvalidParameter)
        ));
        assert!(matches!(
            place_residual_spectrum(&[], 0, 0, 22),
            Err(Error::InvalidParameter)
        ));
        assert!(matches!(
            place_residual_spectrum(&[], 0, 10, 5),
            Err(Error::InvalidParameter)
        ));
    }

    #[test]
    fn synthesis_new_rejects_out_of_range_lm() {
        assert!(LongMdctSynthesis::new(4).is_none());
        for lm in 0..=3u32 {
            let s = LongMdctSynthesis::new(lm).unwrap();
            assert_eq!(s.lm(), lm);
            assert_eq!(s.frame_size(), 120 << lm);
        }
    }

    /// One frame of synthesis emits `mdct_size(lm)` samples and, with a
    /// zero overlap tail (stream open), equals the windowed IMDCT first
    /// half of the placed spectrum.
    #[test]
    fn first_frame_is_windowed_imdct_first_half() {
        let lm = 0; // N = 120
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        // A simple deterministic spectrum.
        let residual: Vec<f32> = (0..coded).map(|i| ((i % 7) as f32) - 3.0).collect();

        let mut synth = LongMdctSynthesis::new(lm).unwrap();
        let out = synth.synthesize(&residual, 0, 21).unwrap();
        let n = mdct_size(lm).unwrap();
        assert_eq!(out.len(), n);

        // Reference: place, IMDCT, window, take the first half (the tail
        // is zero on the first frame).
        let spectrum = place_residual_spectrum(&residual, lm, 0, 21).unwrap();
        let window = build_low_overlap_window_f32(n, CELT_OVERLAP).unwrap();
        let mut block = vec![0.0f32; 2 * n];
        assert!(imdct_naive_f32(&spectrum, &mut block));
        for i in 0..n {
            let expected = block[i] * window[i];
            assert!(
                (out[i] - expected).abs() <= 1e-5,
                "frame[{i}] {} != windowed-imdct {}",
                out[i],
                expected
            );
        }
    }

    /// Two consecutive frames overlap-add: the second frame's first half
    /// equals its own windowed IMDCT first half plus the first frame's
    /// saved (windowed IMDCT second half) tail.
    #[test]
    fn second_frame_overlap_adds_previous_tail() {
        let lm = 0;
        let n = mdct_size(lm).unwrap();
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let r1: Vec<f32> = (0..coded).map(|i| ((i % 5) as f32) - 2.0).collect();
        let r2: Vec<f32> = (0..coded).map(|i| ((i % 3) as f32) - 1.0).collect();

        let mut synth = LongMdctSynthesis::new(lm).unwrap();
        let _f1 = synth.synthesize(&r1, 0, 21).unwrap();
        let f2 = synth.synthesize(&r2, 0, 21).unwrap();

        // Reference WOLA for the same two spectra.
        let window = build_low_overlap_window_f32(n, CELT_OVERLAP).unwrap();
        let s1 = place_residual_spectrum(&r1, lm, 0, 21).unwrap();
        let s2 = place_residual_spectrum(&r2, lm, 0, 21).unwrap();
        let mut b1 = vec![0.0f32; 2 * n];
        let mut b2 = vec![0.0f32; 2 * n];
        assert!(imdct_naive_f32(&s1, &mut b1));
        assert!(imdct_naive_f32(&s2, &mut b2));
        for i in 0..n {
            // f1 saved tail = b1 second half windowed; f2 = b2 first half
            // windowed + that tail.
            let tail = b1[n + i] * window[n + i];
            let expected = b2[i] * window[i] + tail;
            assert!(
                (f2[i] - expected).abs() <= 1e-5,
                "frame2[{i}] {} != overlap-add {}",
                f2[i],
                expected
            );
        }
    }

    /// `reset()` zeroes the overlap tail so the next frame folds against
    /// silence again (§4.5.2 decoder reset): a reset frame equals a
    /// fresh-state first frame of the same spectrum.
    #[test]
    fn reset_clears_overlap_tail() {
        let lm = 1;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let residual: Vec<f32> = (0..coded).map(|i| ((i % 11) as f32) - 5.0).collect();

        let mut synth = LongMdctSynthesis::new(lm).unwrap();
        let fresh = synth.synthesize(&residual, 0, 21).unwrap();
        // Run another frame to dirty the tail, then reset.
        let _ = synth.synthesize(&residual, 0, 21).unwrap();
        synth.reset();
        let after_reset = synth.synthesize(&residual, 0, 21).unwrap();
        for (a, b) in fresh.iter().zip(&after_reset) {
            assert!((a - b).abs() <= 1e-7);
        }
    }

    /// A zero residual spectrum synthesizes to all-zero output (the WOLA
    /// of silence is silence).
    #[test]
    fn zero_residual_is_silence() {
        let lm = 2;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let residual = vec![0.0f32; coded];
        let mut synth = LongMdctSynthesis::new(lm).unwrap();
        let out = synth.synthesize(&residual, 0, 21).unwrap();
        assert!(out.iter().all(|&x| x == 0.0));
    }

    /// The synthesizer rejects a residual whose length disagrees with the
    /// coded window (delegated to `place_residual_spectrum`).
    #[test]
    fn synthesize_rejects_bad_residual_length() {
        let mut synth = LongMdctSynthesis::new(0).unwrap();
        let r = synth.synthesize(&[1.0, 2.0, 3.0], 0, 21);
        assert!(matches!(r, Err(Error::InvalidParameter)));
    }
}
