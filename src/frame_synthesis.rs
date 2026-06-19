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
use crate::synthesis::{LongMdctSynthesis, StereoLongMdctSynthesis};
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

/// Streaming **stereo** CELT synthesis state carried across frames
/// (RFC 6716 §4.5.2), two independent per-channel synthesis chains.
///
/// The §4.3.6 denormalization, §4.3.7 inverse MDCT + weighted
/// overlap-add, §4.3.7.1 post-filter, and §4.3.7.2 de-emphasis all run
/// **per channel** — there is no cross-channel state in the synthesis
/// stage (the stereo joint coding is entirely upstream, in the §4.3.4
/// band decode). This state therefore holds two independent copies of
/// the cross-frame memory the mono [`CeltDecodeState`] keeps:
///
/// * the §4.3.7 long-MDCT overlap tail for each channel
///   (one [`StereoLongMdctSynthesis`] wrapping two spines),
/// * the §4.3.7.1 post-filter history for each channel,
/// * the §4.3.7.2 de-emphasis memory for each channel.
///
/// [`Self::synthesize_stereo_frame`] takes the two channels' **already
/// denormalized** residual spectra and runs that per-channel chain,
/// producing interleaved L/R/L/R PCM. The §4.3.4.4 `itheta` mid/side
/// band coupling (and the §2.7 reallocation) that produces those two
/// spectra from the bitstream is the documented docs gap; this spine
/// covers everything from the denormalized spectra onward, which RFC 6716
/// §4.3.6–§4.3.7 fully specify per channel.
#[derive(Debug, Clone)]
pub struct StereoCeltDecodeState {
    lm: u32,
    synth: StereoLongMdctSynthesis,
    deemph: [Deemphasis; 2],
    /// Per-channel previous-frame post-filtered (pre-de-emphasis) output.
    post_filter_history: [Vec<f32>; 2],
}

/// Per-channel §4.3.7.1 post-filter parameters for a stereo synthesis
/// frame.
///
/// CELT codes a single post-filter in the frame prefix (Table 56), so in
/// practice both channels share the same parameters; this struct lets a
/// caller pass them explicitly per channel for generality (and for the
/// `None` = post-filter-off case).
#[derive(Debug, Clone, Copy)]
pub struct PostFilterParams {
    /// The §4.3.7.1 pitch period (bounded 15..=1022).
    pub period: u16,
    /// The 3-bit raw gain index (`G = 3*(gain+1)/32`).
    pub gain: u8,
    /// The tapset selector (0..=2).
    pub tapset: u8,
}

impl StereoCeltDecodeState {
    /// Create a fresh stereo synthesis state for frame-size shift `lm`
    /// (`0..=3`), with all per-channel inter-frame memory zeroed.
    ///
    /// Returns `None` if `lm > 3`.
    pub fn new(lm: u32) -> Option<Self> {
        let synth = StereoLongMdctSynthesis::new(lm)?;
        let frame_size = synth.frame_size();
        Some(Self {
            lm,
            synth,
            deemph: [Deemphasis::new(), Deemphasis::new()],
            post_filter_history: [vec![0.0; frame_size], vec![0.0; frame_size]],
        })
    }

    /// The frame-size shift this state decodes.
    #[inline]
    pub fn lm(&self) -> u32 {
        self.lm
    }

    /// The per-channel output sample count per frame (`120 << lm`); the
    /// interleaved stereo output is twice this.
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.synth.frame_size()
    }

    /// Zero every carried per-channel inter-frame memory (§4.5.2 reset).
    pub fn reset(&mut self) {
        self.synth.reset();
        self.deemph[0].reset();
        self.deemph[1].reset();
        for h in &mut self.post_filter_history {
            h.iter_mut().for_each(|s| *s = 0.0);
        }
    }

    /// Synthesize one stereo frame from the two channels' **denormalized**
    /// residual spectra, running the full §4.3.6 → §4.3.7 chain per
    /// channel and returning interleaved L/R/L/R PCM.
    ///
    /// * `left_residual` / `right_residual` — each channel's denormalized
    ///   spectrum for the coded-band window `[start, end)`, length
    ///   [`coded_total_bins(start, end, lm)`](crate::band_layout::coded_total_bins).
    /// * `start` / `end` — the coded-band window, `start <= end <= 21`.
    /// * `post_filter` — the §4.3.7.1 parameters to apply per channel
    ///   (`None` leaves the channel un-post-filtered). Both channels share
    ///   the same `Option` because CELT codes a single post-filter.
    ///
    /// Each channel is placed, transformed (§4.3.7 IMDCT + WOLA against
    /// its own overlap tail), post-filtered against its own cross-frame
    /// history, and de-emphasized with its own filter memory, then the two
    /// channels are interleaved.
    ///
    /// Returns [`Error::InvalidParameter`] when either residual length or
    /// the window is inconsistent. The carried state is advanced only on
    /// success (the synthesis spine validates both channels before
    /// mutating either overlap tail).
    pub fn synthesize_stereo_frame(
        &mut self,
        left_residual: &[f32],
        right_residual: &[f32],
        start: usize,
        end: usize,
        post_filter: Option<PostFilterParams>,
    ) -> Result<Vec<f32>, Error> {
        // §4.3.7 per-channel IMDCT + WOLA → interleaved L/R/L/R. The
        // synthesis spine advances both overlap tails atomically (it
        // validates both channels' placement before transforming either).
        let interleaved = self
            .synth
            .synthesize(left_residual, right_residual, start, end)?;

        let n = self.synth.frame_size();
        // De-interleave to per-channel buffers for the per-channel
        // post-filter + de-emphasis (each filter is a running IIR that
        // needs contiguous channel samples).
        let mut chans: [Vec<f32>; 2] = [vec![0.0f32; n], vec![0.0f32; n]];
        for i in 0..n {
            chans[0][i] = interleaved[2 * i];
            chans[1][i] = interleaved[2 * i + 1];
        }

        for ((chan, history), deemph) in chans
            .iter_mut()
            .zip(self.post_filter_history.iter_mut())
            .zip(self.deemph.iter_mut())
        {
            // §4.3.7.1 post-filter against this channel's history.
            if let Some(pf) = post_filter {
                apply_post_filter_f32(chan, history, pf.period, pf.gain, pf.tapset);
            }
            // Save this frame's post-filtered (pre-de-emphasis) output as
            // the next frame's post-filter history.
            history.copy_from_slice(chan);
            // §4.3.7.2 de-emphasis, carrying this channel's filter memory.
            deemph.apply_in_place(chan);
        }

        // Re-interleave the finished per-channel PCM.
        let mut out = vec![0.0f32; 2 * n];
        for i in 0..n {
            out[2 * i] = chans[0][i];
            out[2 * i + 1] = chans[1][i];
        }
        Ok(out)
    }
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

    // --- Stereo synthesis (per-channel-spectrum → interleaved PCM) ---

    use crate::band_layout::coded_total_bins;

    #[test]
    fn stereo_state_rejects_bad_lm() {
        assert!(StereoCeltDecodeState::new(4).is_none());
        for lm in 0..=3u32 {
            let s = StereoCeltDecodeState::new(lm).unwrap();
            assert_eq!(s.lm(), lm);
            assert_eq!(s.frame_size(), 120 << lm);
        }
    }

    /// A stereo synthesis frame produces `2 * frame_size` interleaved
    /// samples and the de-interleaved channels match a per-channel mono
    /// synthesis + de-emphasis (no post-filter).
    #[test]
    fn stereo_frame_matches_per_channel_chain() {
        let lm = 1;
        let n = 120usize << lm;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let rl: Vec<f32> = (0..coded).map(|i| ((i % 7) as f32) - 3.0).collect();
        let rr: Vec<f32> = (0..coded).map(|i| ((i % 5) as f32) - 2.0).collect();

        let mut st = StereoCeltDecodeState::new(lm).unwrap();
        let out = st.synthesize_stereo_frame(&rl, &rr, 0, 21, None).unwrap();
        assert_eq!(out.len(), 2 * n);

        // Reference: two independent mono synthesis + de-emphasis chains.
        let mut sl = LongMdctSynthesis::new(lm).unwrap();
        let mut sr = LongMdctSynthesis::new(lm).unwrap();
        let mut dl = Deemphasis::new();
        let mut dr = Deemphasis::new();
        let mut l = sl.synthesize(&rl, 0, 21).unwrap();
        let mut r = sr.synthesize(&rr, 0, 21).unwrap();
        dl.apply_in_place(&mut l);
        dr.apply_in_place(&mut r);
        for i in 0..n {
            assert!((out[2 * i] - l[i]).abs() <= 1e-6, "left[{i}]");
            assert!((out[2 * i + 1] - r[i]).abs() <= 1e-6, "right[{i}]");
        }
    }

    /// Each channel's de-emphasis + overlap memory carries across frames
    /// (the second frame differs from a fresh first frame of the same
    /// spectra).
    #[test]
    fn stereo_frame_carries_per_channel_memory() {
        let lm = 0;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let rl: Vec<f32> = (0..coded).map(|i| ((i % 3) as f32) - 1.0).collect();
        let rr: Vec<f32> = (0..coded).map(|i| ((i % 4) as f32) - 1.5).collect();

        let mut shared = StereoCeltDecodeState::new(lm).unwrap();
        let _f1 = shared
            .synthesize_stereo_frame(&rl, &rr, 0, 21, None)
            .unwrap();
        let f2 = shared
            .synthesize_stereo_frame(&rl, &rr, 0, 21, None)
            .unwrap();

        let mut fresh = StereoCeltDecodeState::new(lm).unwrap();
        let ff = fresh
            .synthesize_stereo_frame(&rl, &rr, 0, 21, None)
            .unwrap();

        let any_diff = f2.iter().zip(&ff).any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(any_diff, "carried per-channel state did not affect frame 2");
    }

    /// `reset()` restores a fresh-state decode.
    #[test]
    fn stereo_reset_restores_fresh() {
        let lm = 1;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let rl: Vec<f32> = (0..coded).map(|i| ((i % 9) as f32) - 4.0).collect();
        let rr: Vec<f32> = (0..coded).map(|i| ((i % 11) as f32) - 5.0).collect();

        let mut st = StereoCeltDecodeState::new(lm).unwrap();
        let fresh = st.synthesize_stereo_frame(&rl, &rr, 0, 21, None).unwrap();
        let _ = st.synthesize_stereo_frame(&rl, &rr, 0, 21, None).unwrap();
        st.reset();
        let after = st.synthesize_stereo_frame(&rl, &rr, 0, 21, None).unwrap();
        for (a, b) in fresh.iter().zip(&after) {
            assert!((a - b).abs() <= 1e-6);
        }
    }

    /// The post-filter is applied per channel against per-channel history;
    /// with a non-trivial gain the post-filtered output differs from the
    /// no-post-filter output.
    #[test]
    fn stereo_post_filter_applies_per_channel() {
        let lm = 1;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let rl: Vec<f32> = (0..coded).map(|i| ((i % 7) as f32) - 3.0).collect();
        let rr: Vec<f32> = (0..coded).map(|i| ((i % 5) as f32) - 2.0).collect();
        let pf = PostFilterParams {
            period: 64,
            gain: 7,
            tapset: 0,
        };

        let mut with = StereoCeltDecodeState::new(lm).unwrap();
        // Prime a non-zero post-filter history with one frame, then a
        // second frame where the post-filter sees that history.
        let _ = with
            .synthesize_stereo_frame(&rl, &rr, 0, 21, Some(pf))
            .unwrap();
        let a = with
            .synthesize_stereo_frame(&rl, &rr, 0, 21, Some(pf))
            .unwrap();

        let mut without = StereoCeltDecodeState::new(lm).unwrap();
        let _ = without
            .synthesize_stereo_frame(&rl, &rr, 0, 21, None)
            .unwrap();
        let b = without
            .synthesize_stereo_frame(&rl, &rr, 0, 21, None)
            .unwrap();

        let any_diff = a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6);
        assert!(any_diff, "post-filter had no effect");
        assert!(a.iter().all(|x| x.is_finite()));
    }

    /// Inconsistent residual length is rejected.
    #[test]
    fn stereo_frame_rejects_bad_length() {
        let lm = 0;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let rl = vec![0.5f32; coded];
        let rr = vec![-0.5f32; coded - 1];
        let mut st = StereoCeltDecodeState::new(lm).unwrap();
        assert!(matches!(
            st.synthesize_stereo_frame(&rl, &rr, 0, 21, None),
            Err(Error::InvalidParameter)
        ));
    }

    /// Zero spectra on both channels synthesize to interleaved silence
    /// (with no post-filter / fresh de-emphasis the WOLA of silence is
    /// silence).
    #[test]
    fn stereo_zero_spectra_is_silence() {
        let lm = 2;
        let coded = coded_total_bins(0, 21, lm).unwrap() as usize;
        let z = vec![0.0f32; coded];
        let mut st = StereoCeltDecodeState::new(lm).unwrap();
        let out = st.synthesize_stereo_frame(&z, &z, 0, 21, None).unwrap();
        assert!(out.iter().all(|&x| x == 0.0));
    }
}
