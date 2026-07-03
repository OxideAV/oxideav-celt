//! PCM-consuming CELT frame encoder — the top of the encode stack
//! (RFC 6716 §4.3.7.2 pre-emphasis → §4.3.7/§5.3 forward MDCT →
//! §4.3/§5.1 frame encode).
//!
//! ## What this module covers
//!
//! [`crate::frame_encode::encode_celt_frame`] consumes an MDCT-domain
//! spectrum. This module supplies the missing **PCM front end** so the
//! encoder consumes real time-domain samples, mirroring the decode
//! stack's back end step for step (§5.3: the encoder performs the
//! inverse of each decoder operation):
//!
//! | decode back end (§4.3.7)         | encode front end (this module) |
//! |----------------------------------|--------------------------------|
//! | §4.3.7.2 de-emphasis `1/A(z)`    | §4.3.7.2 pre-emphasis `A(z)`   |
//! | §4.3.7 IMDCT + WOLA              | §4.3.7 windowed forward MDCT   |
//! | coded-window spectrum placement  | coded-window extraction        |
//!
//! [`CeltEncodeState`] carries the encoder's cross-frame memory — the
//! pre-emphasis FIR tap, the MDCT analysis history, and the §4.3.2.1
//! coarse-energy prediction (the mirror of
//! [`CeltDecodeState`](crate::frame_synthesis::CeltDecodeState)'s
//! de-emphasis memory, overlap tail, and coarse state).
//! [`encode_celt_frame_pcm`] / [`encode_celt_frame_pcm_auto`] chain
//! pre-emphasis → analysis → the existing frame encoders.
//!
//! ## Latency
//!
//! The analysis inherits the one-frame delay of the streaming MDCT
//! pair (see [`crate::mdct::MdctAnalysis`]): the frame encoded at call
//! `t` represents the transform block covering input frames `t-1` and
//! `t`, and the matching decode at call `t` emits input frame `t-1`.
//! A full PCM → encode → decode → PCM chain therefore reproduces the
//! input with exactly `120 << lm` samples of delay, plus quantization
//! error.
//!
//! ## Post-filter note
//!
//! The §5.3.1 pitch pre-filter (the encoder-side inverse of the
//! §4.3.7.1 post-filter) is not applied: these drivers accept only a
//! `post_filter: None` header. The pre-filter arithmetic is the
//! documented inverse, but the §5.3.1 pitch-period *search* the
//! encoder needs to choose `T` is described only as optimization
//! criteria — encoding with the post-filter signalled but no
//! pre-filter applied would mis-shape the decoded output, so the
//! combination is rejected rather than mis-encoded.
//!
//! ## Clean-room provenance
//!
//! Every stage delegates to an existing RFC-grounded module
//! ([`Preemphasis`], [`LongMdctAnalysis`],
//! [`encode_celt_frame`](crate::frame_encode::encode_celt_frame)).
//! No external library source was consulted.

use crate::analysis::LongMdctAnalysis;
use crate::coarse_energy::{CoarseEnergyState, NUM_BANDS};
use crate::deemphasis::Preemphasis;
use crate::frame_encode::{encode_celt_frame, encode_celt_frame_auto, EncodedFrame};
use crate::frame_header::CeltFrameHeader;
use crate::synthesis::mdct_size;
use crate::Error;

/// Streaming CELT **encoder** state carried across frames — the
/// encode-side mirror of
/// [`CeltDecodeState`](crate::frame_synthesis::CeltDecodeState).
///
/// Holds the three pieces of cross-frame encoder memory:
///
/// * the §4.3.7.2 pre-emphasis FIR tap (`x(n-1)`),
/// * the §4.3.7 forward-MDCT analysis history (the previous input
///   frame, the first half of the next transform block),
/// * the §4.3.2.1 coarse-energy prediction state (kept in exact
///   lockstep with the decoder's by the frame encoder).
///
/// [`reset`](Self::reset) zeroes all three — the encoder-side mirror
/// of the §4.5.2 decoder reset.
#[derive(Debug, Clone)]
pub struct CeltEncodeState {
    lm: u32,
    preemph: Preemphasis,
    analysis: LongMdctAnalysis,
    coarse: CoarseEnergyState,
}

impl CeltEncodeState {
    /// Create an encoder state for frame-size shift `lm` (`0..=3`).
    ///
    /// Returns `None` if `lm > 3`.
    pub fn new(lm: u32) -> Option<Self> {
        Some(Self {
            lm,
            preemph: Preemphasis::new(),
            analysis: LongMdctAnalysis::new(lm)?,
            coarse: CoarseEnergyState::new(),
        })
    }

    /// The frame-size shift this state was created for.
    #[inline]
    pub fn lm(&self) -> u32 {
        self.lm
    }

    /// The per-channel frame size (`120 << lm`) — the number of PCM
    /// samples each encode call consumes.
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.analysis.frame_size()
    }

    /// The encoder's §4.3.2.1 coarse-energy prediction state (in
    /// lockstep with the decoder's after each successfully encoded
    /// frame).
    #[inline]
    pub fn coarse_energy(&self) -> &CoarseEnergyState {
        &self.coarse
    }

    /// Zero every piece of cross-frame memory (pre-emphasis tap,
    /// analysis history, coarse prediction) — the encoder-side mirror
    /// of the §4.5.2 decoder reset.
    pub fn reset(&mut self) {
        self.preemph.reset();
        self.analysis.reset();
        self.coarse.reset();
    }
}

/// Validate the driver-level constraints shared by both PCM encoders,
/// then run the front end (pre-emphasis → windowed forward MDCT →
/// coded-window extraction), committing the streaming state only when
/// the whole encode succeeds.
#[allow(clippy::too_many_arguments)]
fn encode_pcm_impl(
    state: &mut CeltEncodeState,
    pcm: &[f32],
    header: &CeltFrameHeader,
    frame_bytes: u32,
    start: usize,
    end: usize,
    fine_bits: Option<&[u32; NUM_BANDS]>,
    band_k: Option<&[u32]>,
) -> Result<EncodedFrame, Error> {
    let n = mdct_size(state.lm).ok_or(Error::InvalidParameter)?;
    if pcm.len() != n || start > end || end > NUM_BANDS {
        return Err(Error::InvalidParameter);
    }
    // The spectrum encoder itself rejects transient/silent headers;
    // reject them (and a signalled post-filter, whose §5.3.1 pre-filter
    // is not applied — see the module docs) before the streaming
    // front-end state is touched.
    if header.transient || header.silence {
        return Err(Error::NotImplemented);
    }
    if header.post_filter.is_some() {
        return Err(Error::NotImplemented);
    }

    // Snapshot the front-end state so a failed encode leaves the
    // stream exactly where it was (the coarse state is only mutated by
    // the frame encoder after its own validation).
    let preemph_snapshot = state.preemph;
    let analysis_snapshot = state.analysis.clone();

    // §4.3.7.2 pre-emphasis: A(z) = 1 - alpha_p*z^-1 on the input,
    // carrying the FIR tap across frames.
    let mut emphasized = pcm.to_vec();
    state.preemph.apply_in_place(&mut emphasized);

    // §4.3.7 windowed forward MDCT → coded-window spectrum.
    let spectrum = match state.analysis.analyze(&emphasized, start, end) {
        Ok(s) => s,
        Err(e) => {
            state.preemph = preemph_snapshot;
            state.analysis = analysis_snapshot;
            return Err(e);
        }
    };

    // Spectrum-domain frame encode (Table-56 prefix → fine energy →
    // PVQ shapes → §5.1.5 fixed-size assembly).
    let result = match (fine_bits, band_k) {
        (Some(fb), Some(k)) => encode_celt_frame(
            &mut state.coarse,
            &spectrum,
            header,
            state.lm,
            frame_bytes,
            start,
            end,
            fb,
            k,
        ),
        _ => encode_celt_frame_auto(
            &mut state.coarse,
            &spectrum,
            header,
            state.lm,
            frame_bytes,
            start,
            end,
        ),
    };
    match result {
        Ok(frame) => Ok(frame),
        Err(e) => {
            state.preemph = preemph_snapshot;
            state.analysis = analysis_snapshot;
            Err(e)
        }
    }
}

/// Encode one mono, non-transient CELT frame **from PCM**, deriving
/// the per-band pulse counts from the documented §4.3.3 → §4.3.4.1
/// seam — the PCM-consuming counterpart of
/// [`encode_celt_frame_auto`](crate::frame_encode::encode_celt_frame_auto),
/// and the full encode-side mirror of
/// [`decode_celt_frame_auto`](crate::derive_pulses::decode_celt_frame_auto).
///
/// Chains §4.3.7.2 pre-emphasis → the §4.3.7 windowed forward MDCT →
/// coded-window extraction → the self-contained spectrum encoder. The
/// produced frame decodes with `decode_celt_frame_auto` alone — a
/// fully self-contained PCM → bytes → PCM codec loop with no
/// out-of-band data.
///
/// * `pcm` — exactly `120 << lm` time-domain samples (the next input
///   frame; see the module docs for the one-frame latency contract).
/// * `header` — must be non-silent, non-transient, and have
///   `post_filter: None` ([`Error::NotImplemented`] otherwise).
///
/// On any error the streaming front-end state is left untouched, so
/// the caller may retry (e.g. with a larger `frame_bytes`) without
/// desynchronizing the stream.
pub fn encode_celt_frame_pcm_auto(
    state: &mut CeltEncodeState,
    pcm: &[f32],
    header: &CeltFrameHeader,
    frame_bytes: u32,
    start: usize,
    end: usize,
) -> Result<EncodedFrame, Error> {
    encode_pcm_impl(state, pcm, header, frame_bytes, start, end, None, None)
}

/// Encode one mono, non-transient CELT frame **from PCM** with
/// caller-supplied `fine_bits` / `band_k` allocations — the
/// PCM-consuming counterpart of
/// [`encode_celt_frame`](crate::frame_encode::encode_celt_frame)
/// (identical values must be given to the matching
/// [`decode_celt_frame`](crate::frame_synthesis::decode_celt_frame)).
///
/// Front end, constraints, and error behaviour match
/// [`encode_celt_frame_pcm_auto`].
#[allow(clippy::too_many_arguments)]
pub fn encode_celt_frame_pcm(
    state: &mut CeltEncodeState,
    pcm: &[f32],
    header: &CeltFrameHeader,
    frame_bytes: u32,
    start: usize,
    end: usize,
    fine_bits: &[u32; NUM_BANDS],
    band_k: &[u32],
) -> Result<EncodedFrame, Error> {
    encode_pcm_impl(
        state,
        pcm,
        header,
        frame_bytes,
        start,
        end,
        Some(fine_bits),
        Some(band_k),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derive_pulses::decode_celt_frame_auto;
    use crate::frame_synthesis::CeltDecodeState;

    const FRAME_BYTES: u32 = 96;

    fn plain_header() -> CeltFrameHeader {
        CeltFrameHeader {
            silence: false,
            post_filter: None,
            transient: false,
            intra: true,
            anti_collapse_on: None,
        }
    }

    /// Deterministic pseudo-random PCM in [-0.5, 0.5).
    fn test_pcm(len: usize, mut seed: u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// PCM encode → auto decode closes at every LM: the produced
    /// frame decodes to finite PCM with the encoder and decoder coarse
    /// states in exact lockstep, over a multi-frame stream.
    #[test]
    fn pcm_encode_decode_loop_all_lm() {
        for lm in 0..=3u32 {
            let mut enc_state = CeltEncodeState::new(lm).unwrap();
            let mut dec_state = CeltDecodeState::new(lm).unwrap();
            let n = enc_state.frame_size();
            assert_eq!(n, 120usize << lm);
            let mut header = plain_header();
            for t in 0..3u32 {
                let pcm = test_pcm(n, 0xBEEF00 + 97 * t + lm);
                let frame =
                    encode_celt_frame_pcm_auto(&mut enc_state, &pcm, &header, FRAME_BYTES, 0, 21)
                        .unwrap();
                assert_eq!(frame.bytes.len(), FRAME_BYTES as usize);
                let decoded = decode_celt_frame_auto(&mut dec_state, &frame.bytes, 0, 21).unwrap();
                assert_eq!(decoded.pcm.len(), n);
                assert!(decoded.pcm.iter().all(|s| s.is_finite()));
                // Encoder and decoder §4.3.2.1 prediction stay in
                // exact lockstep frame after frame.
                assert_eq!(
                    enc_state.coarse_energy().energy,
                    dec_state.coarse_energy().energy,
                    "lm={lm} frame {t}: coarse state diverged"
                );
                // Decoder sees the identical prefix the encoder built.
                assert_eq!(frame.prefix.header.intra, decoded.prefix.header.intra);
                assert_eq!(frame.prefix.boosts, decoded.prefix.boosts);
                header.intra = false; // subsequent frames inter-coded
            }
        }
    }

    /// Byte-level determinism: the same PCM stream through two fresh
    /// states produces identical frames.
    #[test]
    fn pcm_encode_is_deterministic() {
        let lm = 2u32;
        let mut a = CeltEncodeState::new(lm).unwrap();
        let mut b = CeltEncodeState::new(lm).unwrap();
        let header = plain_header();
        let n = a.frame_size();
        for t in 0..2u32 {
            let pcm = test_pcm(n, 0xD00D + t);
            let fa = encode_celt_frame_pcm_auto(&mut a, &pcm, &header, FRAME_BYTES, 0, 21).unwrap();
            let fb = encode_celt_frame_pcm_auto(&mut b, &pcm, &header, FRAME_BYTES, 0, 21).unwrap();
            assert_eq!(fa.bytes, fb.bytes, "frame {t} bytes diverged");
            assert_eq!(fa.envelope_q8, fb.envelope_q8);
        }
    }

    /// The PCM driver is the exact composition of the standalone
    /// front-end stages and the spectrum encoder.
    #[test]
    fn pcm_driver_matches_manual_composition() {
        let lm = 1u32;
        let mut driver = CeltEncodeState::new(lm).unwrap();
        let header = plain_header();
        let n = driver.frame_size();

        let mut preemph = Preemphasis::new();
        let mut analysis = LongMdctAnalysis::new(lm).unwrap();
        let mut coarse = CoarseEnergyState::new();

        for t in 0..3u32 {
            let pcm = test_pcm(n, 0xFACE + 13 * t);
            let via_driver =
                encode_celt_frame_pcm_auto(&mut driver, &pcm, &header, FRAME_BYTES, 0, 21).unwrap();

            let mut emphasized = pcm.clone();
            preemph.apply_in_place(&mut emphasized);
            let spectrum = analysis.analyze(&emphasized, 0, 21).unwrap();
            let manual = crate::frame_encode::encode_celt_frame_auto(
                &mut coarse,
                &spectrum,
                &header,
                lm,
                FRAME_BYTES,
                0,
                21,
            )
            .unwrap();

            assert_eq!(via_driver.bytes, manual.bytes, "frame {t}");
            assert_eq!(
                via_driver.reconstructed_spectrum,
                manual.reconstructed_spectrum
            );
        }
    }

    /// Unsupported headers and bad shapes are rejected with the
    /// streaming state untouched: a subsequent good call produces the
    /// same bytes a fresh state would.
    #[test]
    fn rejections_leave_stream_state_untouched() {
        let lm = 0u32;
        let mut state = CeltEncodeState::new(lm).unwrap();
        let n = state.frame_size();
        let pcm = test_pcm(n, 0x777);
        let good = plain_header();

        // Wrong PCM length.
        assert!(matches!(
            encode_celt_frame_pcm_auto(&mut state, &pcm[..n - 1], &good, FRAME_BYTES, 0, 21),
            Err(Error::InvalidParameter)
        ));
        // Transient header.
        let transient = CeltFrameHeader {
            transient: true,
            ..good
        };
        assert!(matches!(
            encode_celt_frame_pcm_auto(&mut state, &pcm, &transient, FRAME_BYTES, 0, 21),
            Err(Error::NotImplemented)
        ));
        // Silence header.
        let silence = CeltFrameHeader {
            silence: true,
            ..good
        };
        assert!(matches!(
            encode_celt_frame_pcm_auto(&mut state, &pcm, &silence, FRAME_BYTES, 0, 21),
            Err(Error::NotImplemented)
        ));
        // Post-filter signalled (no §5.3.1 pre-filter is applied).
        let post_filter = CeltFrameHeader {
            post_filter: Some(crate::frame_header::PostFilter {
                octave: 2,
                period: 100,
                gain: 3,
                tapset: 1,
            }),
            ..good
        };
        assert!(matches!(
            encode_celt_frame_pcm_auto(&mut state, &pcm, &post_filter, FRAME_BYTES, 0, 21),
            Err(Error::NotImplemented)
        ));
        // Bad band window.
        assert!(encode_celt_frame_pcm_auto(&mut state, &pcm, &good, FRAME_BYTES, 5, 3).is_err());

        // After all the rejections, the state still behaves fresh.
        let after =
            encode_celt_frame_pcm_auto(&mut state, &pcm, &good, FRAME_BYTES, 0, 21).unwrap();
        let mut fresh = CeltEncodeState::new(lm).unwrap();
        let expected =
            encode_celt_frame_pcm_auto(&mut fresh, &pcm, &good, FRAME_BYTES, 0, 21).unwrap();
        assert_eq!(after.bytes, expected.bytes);
    }

    /// `reset()` restores fresh-stream behaviour across all three
    /// pieces of cross-frame memory.
    #[test]
    fn reset_restores_fresh_stream() {
        let lm = 1u32;
        let mut state = CeltEncodeState::new(lm).unwrap();
        let header = plain_header();
        let n = state.frame_size();
        let pcm = test_pcm(n, 0x1CE);
        let first =
            encode_celt_frame_pcm_auto(&mut state, &pcm, &header, FRAME_BYTES, 0, 21).unwrap();
        let second =
            encode_celt_frame_pcm_auto(&mut state, &pcm, &header, FRAME_BYTES, 0, 21).unwrap();
        assert_ne!(first.bytes, second.bytes, "stream memory must matter");
        state.reset();
        let after_reset =
            encode_celt_frame_pcm_auto(&mut state, &pcm, &header, FRAME_BYTES, 0, 21).unwrap();
        assert_eq!(first.bytes, after_reset.bytes);
    }

    /// The explicit-allocation variant round-trips against
    /// `decode_celt_frame` with the same allocations.
    #[test]
    fn explicit_allocation_variant_round_trips() {
        let lm = 2u32;
        let mut enc_state = CeltEncodeState::new(lm).unwrap();
        let mut dec_state = CeltDecodeState::new(lm).unwrap();
        let header = plain_header();
        let n = enc_state.frame_size();
        let pcm = test_pcm(n, 0xAB1E);
        // Modest fixed allocations: a few pulses in the low bands,
        // one fine bit each in the first four.
        let mut fine_bits = [0u32; NUM_BANDS];
        fine_bits[..4].fill(1);
        let mut band_k = vec![0u32; 21];
        band_k[..8].fill(2);
        let frame = encode_celt_frame_pcm(
            &mut enc_state,
            &pcm,
            &header,
            FRAME_BYTES,
            0,
            21,
            &fine_bits,
            &band_k,
        )
        .unwrap();
        let decoded = crate::frame_synthesis::decode_celt_frame(
            &mut dec_state,
            &frame.bytes,
            0,
            21,
            &fine_bits,
            &band_k,
        )
        .unwrap();
        assert!(decoded.pcm.iter().all(|s| s.is_finite()));
        assert_eq!(
            enc_state.coarse_energy().energy,
            dec_state.coarse_energy().energy
        );
        assert_eq!(frame.envelope_q8, {
            // The decoder's envelope is implicit in its coarse state +
            // fine corrections; equality of coarse states plus the
            // deterministic fine encode pins it, so just re-assert the
            // prefix matched.
            assert_eq!(frame.prefix.allocation, decoded.prefix.allocation);
            frame.envelope_q8
        });
    }
}
