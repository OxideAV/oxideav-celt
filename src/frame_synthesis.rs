//! End-to-end CELT frame decode → PCM orchestrator (RFC 6716 §4.3).
//!
//! ## What this module covers
//!
//! Every CELT decode stage is implemented and unit-tested as a
//! standalone module: the range decoder ([`crate::range_decoder`]), the
//! Table 56 control-symbol prefix ([`crate::frame_decode`]), the
//! §4.3.2.1 coarse / §4.3.2.2 fine energy decode
//! ([`crate::coarse_energy`], [`crate::fine_energy`]), the §4.3.2 final
//! per-band log-energy assembly ([`crate::band_energy`]), the §4.3.4
//! multi-band residual (shape) loop ([`crate::residual`]), the §4.3.6 →
//! §4.3.7 long-MDCT synthesis spine ([`crate::synthesis`]), the
//! §4.3.7.1 post-filter ([`crate::post_filter`]), and the §4.3.7.2
//! de-emphasis ([`crate::deemphasis`]). What was missing was the
//! **top-level driver** that walks them in Table 56 order, threading the
//! decoded envelope from the energy stages into the residual loop and
//! the residual spectrum into the synthesis chain, to turn a CELT
//! range-coded frame into time-domain PCM.
//!
//! [`decode_celt_frame`] is that driver. For one **mono, non-transient
//! (single long MDCT)** CELT frame it:
//!
//! 1. Decodes the Table 56 prefix ([`crate::frame_decode::decode_frame_prefix`]),
//!    yielding the header scalars (silence / post-filter / transient /
//!    intra), the §4.3.4.5 TF parameters, the §4.3.4.3 spread selector,
//!    and the §4.3.3 band-allocation fields. The prefix mutates the
//!    caller's [`CoarseEnergyState`] in place (so §4.3.2.1 inter-frame
//!    prediction carries across frames).
//! 2. Decodes the §4.3.2.2 fine-energy refinement
//!    ([`crate::fine_energy::decode_fine_energy`]) for the per-band
//!    bit counts the caller supplies (`fine_bits`).
//! 3. Assembles the §4.3.2 final per-band Q8 log-energy envelope
//!    ([`crate::band_energy::assemble_band_log_energy_q8`]) from the
//!    coarse + fine corrections, and slices it to the coded-band window.
//! 4. Decodes the §4.3.4 residual (shape) bands
//!    ([`crate::residual::decode_residual_bands`]) for the per-band
//!    pulse counts the caller supplies (`band_k`), producing the
//!    denormalized MDCT-domain spectrum.
//! 5. Runs the §4.3.6 → §4.3.7 long-MDCT synthesis
//!    ([`crate::synthesis::LongMdctSynthesis`]) to obtain the
//!    `120 << lm` time-domain samples (the §4.3.7.1 post-filter input).
//! 6. Applies the §4.3.7.1 post-filter
//!    ([`crate::post_filter::apply_post_filter_f32`]) when the prefix
//!    signalled one, then the §4.3.7.2 de-emphasis
//!    ([`crate::deemphasis::Deemphasis`]), yielding the final PCM.
//!
//! The streaming state ([`CeltDecodeState`]) carries the cross-frame
//! synthesis overlap tail, the de-emphasis memory, and the post-filter
//! history exactly as §4.5.2 requires for gapless playback.
//!
//! ## Why the allocation is an input
//!
//! The per-band pulse counts (`band_k`, the §4.3.4.1 bits-to-pulses
//! output) and the per-band fine-bit counts (`fine_bits`, the §4.3.2.2
//! allocation) are the output of the §4.3.3 reallocation pass: the
//! per-band bisection subject to caps/minimums with concurrent skip
//! decoding and the fine-energy vs. shape split. RFC 6716 §4.3.3
//! delegates the precise procedure to the reference implementation
//! ("implementers are free to implement the procedure in any way that
//! produces identical results"), so it remains a documented docs gap.
//! By taking `band_k` and `fine_bits` as parameters, this driver stays
//! entirely within fully-specified §4.3 territory — the same boundary
//! [`crate::residual::decode_residual_bands`] and
//! [`crate::bits_to_pulses::bits_to_pulses_band_loop`] keep. When the
//! reallocation pass lands, the only change here is to compute these two
//! vectors from the [`FramePrefix`] rather than accept them.
//!
//! ## Scope: mono, non-transient
//!
//! This driver handles the unambiguous **mono single-long-MDCT** case.
//! Three CELT features are deferred to the reference by the RFC and
//! therefore out of scope:
//!
//! * **Transient short blocks** (§4.3.1 / §4.3.7): the per-short-block
//!   frequency layout and inter-block overlap-add are delegated to
//!   `celt.c` / `mdct.c`; [`crate::synthesis::LongMdctSynthesis`] keeps
//!   that boundary.
//! * **Stereo joint coding** (§4.3.4.4 `itheta` mid/side): the angle
//!   quantisation PDF is deferred to the reference.
//! * **Anti-collapse injection** (§4.3.5): the §4.3.5 narrative
//!   describes the *intent* — "a pseudo-random signal is inserted with
//!   an energy corresponding to the minimum energy over the two
//!   previous frames" — but gives **no** collapse-detection threshold,
//!   pseudo-random generator, or injection magnitude formula; those live
//!   in `bands.c::anti_collapse()`, outside the staged docs. This is a
//!   documented docs gap; since this driver only handles non-transient
//!   frames (where §4.3.5 says the anti-collapse bit is not even
//!   decoded), the gap does not block the mono long-MDCT path.
//!
//! A `transient` or `stereo` request is rejected with
//! [`Error::NotImplemented`] rather than silently mis-decoded.
//!
//! ## Clean-room provenance
//!
//! The stage order is RFC 6716 Table 56
//! (`docs/audio/opus/rfc6716-opus.txt` lines 5943–5985) and §4.3.7's
//! "output of the inverse MDCT (after weighted overlap-add) is sent to
//! the post-filter" → de-emphasis chain (lines 6738–6760). Every stage
//! delegates to an existing RFC-grounded module whose own provenance is
//! recorded in that module. No external library source was consulted.

use crate::band_energy::assemble_band_log_energy_q8;
use crate::coarse_energy::{CoarseEnergyState, NUM_BANDS};
use crate::deemphasis::Deemphasis;
use crate::fine_energy::decode_fine_energy;
use crate::frame_decode::{decode_frame_prefix, FramePrefix};
use crate::post_filter::apply_post_filter_f32;
use crate::range_decoder::RangeDecoder;
use crate::residual::decode_residual_bands;
use crate::synthesis::LongMdctSynthesis;
use crate::Error;

/// Streaming CELT decoder state carried across frames (RFC 6716 §4.5.2).
///
/// Holds the three pieces of inter-frame memory the §4.3 decode chain
/// needs for gapless playback:
///
/// * the §4.3.2.1 coarse-energy inter-frame prediction
///   ([`CoarseEnergyState`]),
/// * the §4.3.6 → §4.3.7 long-MDCT synthesis overlap tail
///   ([`LongMdctSynthesis`]),
/// * the §4.3.7.1 post-filter history (the previous frame's
///   pre-de-emphasis output) and the §4.3.7.2 de-emphasis filter memory
///   ([`Deemphasis`]).
///
/// One state instance decodes a single logical CELT stream of a fixed
/// frame-size shift `lm`. Call [`reset`](Self::reset) on a §4.5.2
/// decoder reset (packet loss boundary, mode switch) to zero every
/// carried memory.
#[derive(Debug, Clone)]
pub struct CeltDecodeState {
    lm: u32,
    coarse: CoarseEnergyState,
    synth: LongMdctSynthesis,
    deemph: Deemphasis,
    /// The previous frame's post-filtered (pre-de-emphasis) output, the
    /// §4.3.7.1 post-filter's cross-frame history. Length equals the
    /// frame size; zero on a fresh / reset state.
    post_filter_history: Vec<f32>,
}

impl CeltDecodeState {
    /// Create a fresh decoder state for frame-size shift `lm` (`0..=3`,
    /// i.e. 2.5 / 5 / 10 / 20 ms frames), with all inter-frame memory
    /// zeroed.
    ///
    /// Returns `None` if `lm > 3` (the only failure mode — the synthesis
    /// window construction is total for every in-range `lm`).
    pub fn new(lm: u32) -> Option<Self> {
        let synth = LongMdctSynthesis::new(lm)?;
        let frame_size = synth.frame_size();
        Some(Self {
            lm,
            coarse: CoarseEnergyState::new(),
            synth,
            deemph: Deemphasis::new(),
            post_filter_history: vec![0.0; frame_size],
        })
    }

    /// The frame-size shift this state decodes (`log2(frame_size / 120)`).
    #[inline]
    pub fn lm(&self) -> u32 {
        self.lm
    }

    /// The per-channel output sample count per frame (`120 << lm`).
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.synth.frame_size()
    }

    /// Read-only view of the §4.3.2.1 coarse-energy prediction state.
    #[inline]
    pub fn coarse_energy(&self) -> &CoarseEnergyState {
        &self.coarse
    }

    /// Zero every carried inter-frame memory (RFC 6716 §4.5.2 decoder
    /// reset): coarse-energy prediction, synthesis overlap tail,
    /// post-filter history, and de-emphasis memory.
    pub fn reset(&mut self) {
        self.coarse.reset();
        self.synth.reset();
        self.deemph.reset();
        self.post_filter_history.iter_mut().for_each(|s| *s = 0.0);
    }
}

/// The result of decoding one CELT frame to PCM.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// The final time-domain PCM samples, `frame_size()` (= `120 << lm`)
    /// of them, after synthesis → post-filter → de-emphasis.
    pub pcm: Vec<f32>,
    /// The fully-decoded Table 56 control prefix (header scalars, TF
    /// parameters, spread, band allocation), for callers that want to
    /// inspect the decoded control state.
    pub prefix: FramePrefix,
}

/// Decode one mono, non-transient CELT frame to time-domain PCM
/// (RFC 6716 §4.3, Table 56 → §4.3.7 chain).
///
/// Walks the documented decode chain end-to-end:
/// prefix → fine energy → band-energy assembly → residual shape decode →
/// long-MDCT synthesis → post-filter → de-emphasis. The streaming
/// [`CeltDecodeState`] carries the cross-frame memory.
///
/// ## Parameters
///
/// * `state` — the streaming decoder state (mutated in place: coarse
///   energy, overlap tail, post-filter history, de-emphasis memory).
/// * `frame_bytes` — the CELT range-coded payload for this frame. The
///   range decoder is initialised over it; its byte length drives the
///   §4.3.3 frame budget.
/// * `start` / `end` — the coded-band window, `start <= end <= 21`.
///   Pure CELT is `(0, 21)`; Hybrid mode is `(17, 21)`; narrower
///   bandwidths reduce `end`.
/// * `fine_bits` — the §4.3.2.2 per-band fine-bit counts `B_i` the
///   §4.3.3 allocator assigned, indexed by **absolute** band
///   (`0..NUM_BANDS`). The coded window `[start, end)` slice is the part
///   that matters; out-of-window entries are ignored. (The §4.3.3
///   reallocation pass that produces this is a documented docs gap — see
///   the module docs.)
/// * `band_k` — the §4.3.4.1 per-band pulse counts `K`, one per coded
///   band in `start..end` order. Length MUST equal `end - start`. (Same
///   docs-gap boundary.)
///
/// ## Returns
///
/// The [`DecodedFrame`] (PCM + decoded prefix) on success, or:
///
/// * [`Error::InvalidParameter`] when the band window or `lm` is out of
///   range, or `band_k.len() != end - start`.
/// * [`Error::NotImplemented`] when the decoded prefix signals a
///   transient frame (the §4.3.1 / §4.3.7 short-block reassembly gap),
///   or when a band's codebook saturates (the §4.3.4.4 split gap
///   surfaced by [`decode_residual_bands`]).
///
/// A `silence`-flagged frame (§4.3 silence bit) decodes to all-zero
/// PCM: the residual is the zero spectrum, so synthesis emits the WOLA
/// of silence and de-emphasis carries the previous memory through.
pub fn decode_celt_frame(
    state: &mut CeltDecodeState,
    frame_bytes: &[u8],
    start: usize,
    end: usize,
    fine_bits: &[u32; NUM_BANDS],
    band_k: &[u32],
) -> Result<DecodedFrame, Error> {
    if start > end || end > NUM_BANDS {
        return Err(Error::InvalidParameter);
    }
    let coded_bands = end - start;
    if band_k.len() != coded_bands {
        return Err(Error::InvalidParameter);
    }
    let lm = state.lm;

    let mut dec = RangeDecoder::new(frame_bytes);

    // Table 56 prefix: silence / post-filter / transient / intra →
    // coarse energy → TF / spread → caps → boosts → reservations →
    // band allocation. Mutates the coarse-energy prediction state.
    let prefix = decode_frame_prefix(
        &mut dec,
        &mut state.coarse,
        lm,
        frame_bytes.len() as u32,
        false,
        start,
        end,
    )?;

    // This driver handles the non-transient (single long MDCT) case
    // only; the transient short-block reassembly is a documented gap.
    if prefix.header.transient {
        return Err(Error::NotImplemented);
    }

    // §4.3.2.2 fine-energy refinement for the allocator's per-band bit
    // counts. Reads raw bits from the back of the frame.
    let fine_q14 = decode_fine_energy(&mut dec, fine_bits);

    // §4.3.2 final per-band Q8 log-energy envelope (coarse + fine).
    // The finalize step is part of the gap'd allocation tail, so it is
    // omitted here (it contributes at most ±1 fine bit per band).
    let env_q8 = assemble_band_log_energy_q8(&state.coarse, 0, Some(&fine_q14), None)
        .ok_or(Error::InvalidParameter)?;

    // Slice the absolute-band envelope to the coded window for the
    // residual loop, which indexes per-coded-band.
    let window_energy: Vec<i32> = env_q8[start..end].to_vec();

    // §4.3.4 residual (shape) decode → denormalized MDCT-domain spectrum.
    let residual = decode_residual_bands(
        &mut dec,
        lm,
        start,
        end,
        false,
        prefix.tf.tf_select,
        &prefix.tf.tf_changes,
        band_k,
        prefix.spread,
        &window_energy,
    )?;

    // §4.3.6 → §4.3.7 long-MDCT synthesis: place the residual spectrum
    // and run the inverse MDCT + weighted overlap-add.
    let mut pcm = state.synth.synthesize(&residual.samples, start, end)?;

    // §4.3.7.1 post-filter (when the prefix signalled one), against the
    // previous frame's pre-de-emphasis output as cross-frame history.
    if let Some(pf) = &prefix.header.post_filter {
        apply_post_filter_f32(
            &mut pcm,
            &state.post_filter_history,
            pf.period,
            pf.gain,
            pf.tapset,
        );
    }
    // Save this frame's post-filtered (pre-de-emphasis) output as the
    // next frame's post-filter history.
    state.post_filter_history.copy_from_slice(&pcm);

    // §4.3.7.2 single-pole de-emphasis, carrying the filter memory.
    state.deemph.apply_in_place(&mut pcm);

    Ok(DecodedFrame { pcm, prefix })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::mdct_size;

    fn zero_fine() -> [u32; NUM_BANDS] {
        [0u32; NUM_BANDS]
    }

    /// A full pure-CELT mono frame decodes to `frame_size` PCM samples
    /// without panicking, threading every stage.
    #[test]
    fn decodes_full_mono_frame() {
        let buf: Vec<u8> = (0..96u8)
            .map(|i| i.wrapping_mul(37).wrapping_add(11))
            .collect();
        let mut state = CeltDecodeState::new(3).unwrap();
        let band_k = vec![2u32; 21];
        let out =
            decode_celt_frame(&mut state, &buf, 0, 21, &zero_fine(), &band_k).expect("decode");
        assert_eq!(out.pcm.len(), mdct_size(3).unwrap());
        assert_eq!(out.prefix.start, 0);
        assert_eq!(out.prefix.end, 21);
    }

    /// Out-of-range band window / mismatched `band_k` is rejected.
    #[test]
    fn rejects_invalid_parameters() {
        let buf = [0x5au8; 64];
        let mut state = CeltDecodeState::new(2).unwrap();

        // end > NUM_BANDS.
        assert!(matches!(
            decode_celt_frame(&mut state, &buf, 0, 22, &zero_fine(), &[1u32; 22]),
            Err(Error::InvalidParameter)
        ));
        // start > end.
        assert!(matches!(
            decode_celt_frame(&mut state, &buf, 10, 5, &zero_fine(), &[]),
            Err(Error::InvalidParameter)
        ));
        // band_k length disagrees with the window.
        assert!(matches!(
            decode_celt_frame(&mut state, &buf, 0, 21, &zero_fine(), &[1u32; 20]),
            Err(Error::InvalidParameter)
        ));
    }

    /// State constructor rejects out-of-range `lm`.
    #[test]
    fn state_rejects_bad_lm() {
        assert!(CeltDecodeState::new(4).is_none());
        for lm in 0..=3u32 {
            let s = CeltDecodeState::new(lm).unwrap();
            assert_eq!(s.lm(), lm);
            assert_eq!(s.frame_size(), 120 << lm);
        }
    }

    /// A Hybrid-mode window (bands 17..=20) decodes a 4-band frame and
    /// still emits a full `frame_size` of PCM (the uncoded low bins are
    /// zero in the spectrum).
    #[test]
    fn decodes_hybrid_window() {
        let buf: Vec<u8> = (0..64u8)
            .map(|i| i.wrapping_mul(29).wrapping_add(3))
            .collect();
        let mut state = CeltDecodeState::new(2).unwrap();
        let band_k = vec![3u32; 4];
        let out = decode_celt_frame(&mut state, &buf, 17, 21, &zero_fine(), &band_k)
            .expect("hybrid decode");
        assert_eq!(out.pcm.len(), mdct_size(2).unwrap());
        assert_eq!(out.prefix.start, 17);
    }

    /// Two consecutive frames carry the synthesis overlap tail: the
    /// second frame's PCM is not simply the first-frame-from-fresh-state
    /// output, because the overlap-add folds in the prior tail.
    #[test]
    fn overlap_tail_carries_across_frames() {
        let buf: Vec<u8> = (0..96u8)
            .map(|i| i.wrapping_mul(53).wrapping_add(7))
            .collect();
        let band_k = vec![2u32; 21];

        // Two frames sharing one state.
        let mut shared = CeltDecodeState::new(1).unwrap();
        let _f1 = decode_celt_frame(&mut shared, &buf, 0, 21, &zero_fine(), &band_k).unwrap();
        let f2 = decode_celt_frame(&mut shared, &buf, 0, 21, &zero_fine(), &band_k).unwrap();

        // The same second-position frame decoded from a fresh state
        // (no carried overlap / de-emphasis memory).
        let mut fresh = CeltDecodeState::new(1).unwrap();
        let f_fresh = decode_celt_frame(&mut fresh, &buf, 0, 21, &zero_fine(), &band_k).unwrap();

        // The carried-tail frame differs from the fresh decode (the
        // overlap-add and de-emphasis memory both fold prior state in).
        let any_diff = f2
            .pcm
            .iter()
            .zip(&f_fresh.pcm)
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(any_diff, "carried state did not affect the second frame");
    }

    /// `reset()` zeroes the carried memory so a post-reset frame equals a
    /// fresh-state first frame of the same bytes.
    #[test]
    fn reset_restores_fresh_decode() {
        let buf: Vec<u8> = (0..96u8)
            .map(|i| i.wrapping_mul(37).wrapping_add(11))
            .collect();
        let band_k = vec![1u32; 21];

        let mut state = CeltDecodeState::new(3).unwrap();
        let fresh = decode_celt_frame(&mut state, &buf, 0, 21, &zero_fine(), &band_k).unwrap();
        // Dirty the carried memory, then reset.
        let _ = decode_celt_frame(&mut state, &buf, 0, 21, &zero_fine(), &band_k).unwrap();
        state.reset();
        let after = decode_celt_frame(&mut state, &buf, 0, 21, &zero_fine(), &band_k).unwrap();
        for (a, b) in fresh.pcm.iter().zip(&after.pcm) {
            assert!((a - b).abs() <= 1e-6, "post-reset PCM differs: {a} vs {b}");
        }
    }

    /// A frame whose decoded prefix signals a transient is rejected with
    /// `NotImplemented` (the §4.3.1 / §4.3.7 short-block reassembly gap),
    /// never silently mis-decoded as a long MDCT.
    #[test]
    fn transient_frame_is_not_implemented() {
        // This byte pattern decodes a transient prefix at lm=1.
        let buf: Vec<u8> = (0..96u8)
            .map(|i| i.wrapping_mul(41).wrapping_add(13))
            .collect();
        let mut state = CeltDecodeState::new(1).unwrap();
        let band_k = vec![1u32; 21];
        let r = decode_celt_frame(&mut state, &buf, 0, 21, &zero_fine(), &band_k);
        assert!(
            matches!(r, Err(Error::NotImplemented)),
            "expected NotImplemented for a transient frame, got {r:?}"
        );
    }

    /// A coded window of zero pulses everywhere with a flat zero envelope
    /// produces finite (non-NaN) PCM — the all-zero-shape edge.
    #[test]
    fn zero_pulses_is_finite_pcm() {
        let buf = [0x80u8; 64];
        let mut state = CeltDecodeState::new(2).unwrap();
        let band_k = vec![0u32; 21];
        let out =
            decode_celt_frame(&mut state, &buf, 0, 21, &zero_fine(), &band_k).expect("decode");
        assert!(out.pcm.iter().all(|x| x.is_finite()));
    }
}
