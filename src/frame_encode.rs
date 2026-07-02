//! CELT frame-prefix encode driver (RFC 6716 §4.3, Table 56) — the
//! exact inverse of [`crate::frame_decode::decode_frame_prefix`].
//!
//! ## What this module covers
//!
//! Every control symbol in the Table-56 prefix now has an encode
//! primitive: the header scalars
//! ([`CeltFrameHeader::encode_prefix`]), the §4.3.2.1 coarse energy
//! ([`encode_coarse_energy`]), the §4.3.4.5 TF parameters
//! ([`encode_tf_parameters`]), the §4.3.4.3 spread
//! ([`encode_spread`]), the §4.3.3 dynalloc band boosts
//! ([`encode_band_boosts`]), and the §4.3.3 band-allocation scalar
//! fields ([`encode_band_allocation`]). What was missing was the
//! **integration spine**: the single driver that writes them in the
//! exact bitstream order Table 56 prescribes, threading the running
//! reservation/boost budget between the steps so every gate
//! (`trim_gated`, the stereo reservations, the band-boost
//! `total_bits`) is evaluated against the correct intermediate budget
//! — the same threading [`decode_frame_prefix`] performs on the read
//! side.
//!
//! [`encode_frame_prefix`] is that driver. Because the encoder's
//! [`RangeEncoder::tell_frac`] reports the same value the decoder's
//! `tell_frac` reports after the same symbols (RFC 6716 §5.1.6 /
//! §4.1.6), each budget read lands on the identical number on both
//! sides, and a [`decode_frame_prefix`] pass over the finished frame
//! reconstructs the identical [`FramePrefix`] — the §4.3.3
//! encoder/decoder lockstep requirement ("identical coding decisions
//! must be made in the encoder and decoder").
//!
//! ## Normalization
//!
//! The returned [`FramePrefix`] is the *decoder's view* of what was
//! written: a gated-off allocation field lands on its §4.3.3 default
//! (not the caller's requested value), a gated-off `tf_select` lands
//! on 0, and the boosts are the gate-truncated
//! [`BoostResult`](crate::band_cap::BoostResult) actually encoded.
//! Callers that need a request honoured exactly compare the returned
//! prefix against their spec.
//!
//! ## Clean-room provenance
//!
//! The symbol order is RFC 6716 Table 56
//! (`docs/audio/opus/rfc6716-opus.txt` lines 5943–5985); the budget
//! threading is RFC 6716 §4.3.3 and the clean-room narrative
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md`
//! §§2.1–2.7 — the same sources the decode driver transcribes. Every
//! written field is produced by an existing encode module whose own
//! provenance is recorded in that module. No external library source
//! was consulted.

use crate::allocation_budget::compute_initial_reservations;
use crate::band_cap::{compute_band_caps, encode_band_boosts};
use crate::bit_allocation::{encode_band_allocation, BandAllocation};
use crate::coarse_energy::{encode_coarse_energy, CoarseEnergyState, MAX_CHANNELS, NUM_BANDS};
use crate::frame_decode::FramePrefix;
use crate::frame_header::CeltFrameHeader;
use crate::range_encoder::RangeEncoder;
use crate::spread::{encode_spread, Spread};
use crate::tf_change::{encode_tf_parameters, tf_select_matters, TfParameters};
use crate::Error;

/// Per-band MDCT-bin layout for the coded band window (same helper the
/// decode driver uses; RFC 6716 §4.3 Table 55).
fn bins_for_window(lm: usize, start: usize, end: usize) -> Vec<u32> {
    crate::band_minimums::BAND_BINS_LM[lm][start..end].to_vec()
}

/// The encoder's chosen per-frame control parameters — everything
/// [`encode_frame_prefix`] serialises ahead of the `fine energy`
/// Table-56 entry.
#[derive(Debug, Clone)]
pub struct FramePrefixSpec<'a> {
    /// Header scalars: silence / post-filter / transient / intra.
    /// `anti_collapse_on` is ignored here (the §4.3.5 bit is written
    /// *after* the band shapes, not in the prefix).
    pub header: CeltFrameHeader,
    /// Per-channel per-band target log-2 energies (`1.0` = 6 dB) for
    /// the §4.3.2.1 coarse quantization. Channel 1 is ignored for a
    /// mono frame.
    pub energy_target: &'a [[f32; NUM_BANDS]; MAX_CHANNELS],
    /// The §4.3.4.5 TF parameters. `tf_select` must be 0 when the
    /// "can it have an impact" gate is closed (`tf_select_decoded` is
    /// recomputed, not trusted).
    pub tf: TfParameters,
    /// The §4.3.4.3 spreading selector.
    pub spread: Spread,
    /// Desired §4.3.3 per-band boosts in 1/8 bits, one per coded band
    /// (`end - start` entries). Values floor to the band's quanta grid
    /// and truncate at the budget/cap gates.
    pub target_boost: &'a [i32],
    /// The §4.3.3 band-allocation scalar fields. Gated-off fields are
    /// not written (the decoder lands on the §4.3.3 defaults).
    pub allocation: BandAllocation,
}

/// Encode the CELT frame prefix in RFC 6716 Table 56 order — the exact
/// inverse of [`decode_frame_prefix`](crate::frame_decode::decode_frame_prefix).
///
/// Writes, in order: the header scalars (§4.3), the coarse energy
/// (§4.3.2.1, mutating `coarse_state` with the actually-coded
/// reconstruction so inter-frame prediction carries across frames),
/// the TF parameters (§4.3.4.5), the spread (§4.3.4.3), the dynalloc
/// band boosts (§4.3.3, gate-truncated), and the band-allocation
/// scalar fields (§4.3.3, each only when its gate is open). The
/// encoder is left positioned exactly at the Table-56 `fine energy`
/// entry.
///
/// * `frame_bytes` — the **target** coded frame size in bytes. It
///   drives the §4.3.2.1 budget dispatch and the §4.3.3 reservation
///   walk; the caller must pad the finished frame to exactly this many
///   bytes so the decoder's `storage_bits()` sees the same budget.
/// * `stereo` / `start` / `end` — as on the decode side.
///
/// Returns the [`FramePrefix`] a matching `decode_frame_prefix` over
/// the finished (padded) frame reconstructs — the decoder's view, with
/// gated-off fields normalized to their §4.3.3 defaults.
///
/// Returns [`Error::InvalidParameter`] for an out-of-range `lm` / band
/// window, a `tf_changes` length mismatch, a `target_boost` length
/// mismatch, or an out-of-range allocation field; a closed-gate
/// non-zero `tf_select` is rejected by the underlying TF encoder.
#[allow(clippy::too_many_arguments)]
pub fn encode_frame_prefix(
    enc: &mut RangeEncoder,
    coarse_state: &mut CoarseEnergyState,
    spec: &FramePrefixSpec<'_>,
    lm: u32,
    frame_bytes: u32,
    stereo: bool,
    start: usize,
    end: usize,
) -> Result<FramePrefix, Error> {
    if lm > 3 || start > end || end > NUM_BANDS {
        return Err(Error::InvalidParameter);
    }
    let lm_usize = lm as usize;
    let channels = if stereo { 2usize } else { 1usize };
    let coded_bands = (end - start) as u32;
    if spec.tf.tf_changes.len() != end - start {
        return Err(Error::InvalidParameter);
    }

    // Table 56 step 1: silence / post-filter / transient / intra.
    spec.header.encode_prefix(enc)?;

    // Table 56 step 2: coarse energy (§4.3.2.1). Budget-keyed dispatch
    // against the target frame size, mirroring the decoder's
    // storage_bits(); mutates the prediction state with the coded
    // reconstruction.
    encode_coarse_energy(
        enc,
        coarse_state,
        spec.energy_target,
        spec.header.intra,
        lm,
        start,
        end,
        channels,
        frame_bytes * 8,
    )?;

    // Table 56 step 3: tf_change + tf_select (§4.3.4.5).
    encode_tf_parameters(enc, &spec.tf, spec.header.transient, lm as u8)?;

    // Table 56 step 4: spread (§4.3.4.3).
    encode_spread(enc, spec.spread)?;

    // §4.3.3 setup: per-band caps for the coded window.
    let bins = bins_for_window(lm_usize, start, end);
    let mut caps = vec![0i16; bins.len()];
    if !compute_band_caps(lm, stereo, channels as u32, &bins, &mut caps) {
        return Err(Error::InvalidParameter);
    }

    // §4.3.3 setup: the initial reservation walk, read at the
    // post-spread tell_frac — the identical position the decoder reads.
    let tell_after_spread = enc.tell_frac();
    let reservations = compute_initial_reservations(
        frame_bytes,
        tell_after_spread,
        spec.header.transient,
        lm,
        stereo,
        coded_bands,
    );

    // Table 56 step 5: dynalloc band boosts (§4.3.3), gate-truncated.
    let boosts = encode_band_boosts(
        enc,
        start as u32,
        end as u32,
        &bins,
        &caps,
        reservations.total,
        spec.target_boost,
    )?;

    // Table 56 steps 6-9: trim / skip / intensity / dual (§4.3.3),
    // gated against the post-boost tell_frac and total_boost.
    let tell_after_boosts = enc.tell_frac();
    let gates = reservations.gates_for_band_allocation(tell_after_boosts, boosts.total_boost);
    encode_band_allocation(enc, gates, &spec.allocation)?;

    // Normalize to the decoder's view: gated-off fields land on their
    // §4.3.3 defaults.
    let defaults = BandAllocation::defaults();
    let allocation = BandAllocation {
        alloc_trim: if gates.trim_gated {
            spec.allocation.alloc_trim
        } else {
            defaults.alloc_trim
        },
        skip: if gates.skip_gated {
            spec.allocation.skip
        } else {
            defaults.skip
        },
        intensity_band_offset: if gates.intensity_gated {
            spec.allocation.intensity_band_offset
        } else {
            defaults.intensity_band_offset
        },
        dual_stereo: if gates.dual_gated {
            spec.allocation.dual_stereo
        } else {
            defaults.dual_stereo
        },
    };
    let tf_gate = tf_select_matters(spec.header.transient, lm as u8, &spec.tf.tf_changes);
    let tf = TfParameters {
        tf_changes: spec.tf.tf_changes.clone(),
        tf_select: if tf_gate { spec.tf.tf_select } else { 0 },
        tf_select_decoded: tf_gate,
    };

    Ok(FramePrefix {
        header: spec.header,
        tf,
        spread: spec.spread,
        caps,
        reservations,
        boosts,
        allocation,
        start,
        end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_decode::decode_frame_prefix;
    use crate::frame_header::PostFilter;
    use crate::range_decoder::RangeDecoder;

    const FRAME_BYTES: u32 = 96;

    fn mono_target() -> [[f32; NUM_BANDS]; MAX_CHANNELS] {
        [
            std::array::from_fn(|b| 4.0 - 0.11 * b as f32),
            [0.0; NUM_BANDS],
        ]
    }

    fn encode_and_decode(
        spec: &FramePrefixSpec<'_>,
        lm: u32,
        stereo: bool,
        start: usize,
        end: usize,
    ) -> (FramePrefix, FramePrefix) {
        let mut enc = RangeEncoder::new();
        let mut enc_state = CoarseEnergyState::new();
        let enc_prefix = encode_frame_prefix(
            &mut enc,
            &mut enc_state,
            spec,
            lm,
            FRAME_BYTES,
            stereo,
            start,
            end,
        )
        .unwrap();
        // §5.1 fixed-size frame assembly: range bytes at the front, raw
        // bits (the post-filter fine_pitch/gain) at the back.
        let frame = enc.finish_to_size(FRAME_BYTES as usize).unwrap();
        let mut dec = RangeDecoder::new(&frame);
        let mut dec_state = CoarseEnergyState::new();
        let dec_prefix = decode_frame_prefix(
            &mut dec,
            &mut dec_state,
            lm,
            FRAME_BYTES,
            stereo,
            start,
            end,
        )
        .unwrap();
        assert!(!dec.has_error());
        assert_eq!(enc_state.energy, dec_state.energy, "coarse state diverged");
        (enc_prefix, dec_prefix)
    }

    fn assert_prefix_eq(a: &FramePrefix, b: &FramePrefix) {
        assert_eq!(a.header.silence, b.header.silence);
        assert_eq!(a.header.post_filter, b.header.post_filter);
        assert_eq!(a.header.transient, b.header.transient);
        assert_eq!(a.header.intra, b.header.intra);
        assert_eq!(a.tf.tf_changes, b.tf.tf_changes);
        assert_eq!(a.tf.tf_select, b.tf.tf_select);
        assert_eq!(a.tf.tf_select_decoded, b.tf.tf_select_decoded);
        assert_eq!(a.spread, b.spread);
        assert_eq!(a.caps, b.caps);
        assert_eq!(a.reservations, b.reservations);
        assert_eq!(a.boosts, b.boosts);
        assert_eq!(a.allocation, b.allocation);
        assert_eq!(a.start, b.start);
        assert_eq!(a.end, b.end);
    }

    /// A full pure-CELT mono prefix round-trips: the decoder
    /// reconstructs the identical FramePrefix, field for field.
    #[test]
    fn mono_prefix_roundtrip_full_window() {
        let target = mono_target();
        let mut tf_changes = vec![false; NUM_BANDS];
        tf_changes[3] = true;
        tf_changes[10] = true;
        let mut target_boost = vec![0i32; NUM_BANDS];
        // Band 2 at LM=2 spans N = 4 bins, so quanta = min(8*4,
        // max(48, 4)) = 32; two quanta = 64.
        target_boost[2] = 64;
        let spec = FramePrefixSpec {
            header: CeltFrameHeader {
                silence: false,
                post_filter: Some(PostFilter {
                    octave: 2,
                    period: 100,
                    gain: 3,
                    tapset: 1,
                }),
                transient: false,
                intra: true,
                anti_collapse_on: None,
            },
            energy_target: &target,
            tf: TfParameters {
                tf_changes,
                tf_select: 0,
                tf_select_decoded: false,
            },
            spread: Spread::Normal,
            target_boost: &target_boost,
            allocation: BandAllocation {
                alloc_trim: 7,
                skip: true,
                intensity_band_offset: 0,
                dual_stereo: false,
            },
        };
        let (enc_prefix, dec_prefix) = encode_and_decode(&spec, 2, false, 0, NUM_BANDS);
        assert_prefix_eq(&enc_prefix, &dec_prefix);
        // The requested boost landed (ample budget).
        assert_eq!(dec_prefix.boosts.boost[2], 64);
        // The requested trim landed (gate open at this budget).
        assert_eq!(dec_prefix.allocation.alloc_trim, 7);
        assert!(dec_prefix.allocation.skip);
    }

    /// A stereo inter prefix round-trips, including the stereo
    /// reservations and the intensity/dual fields.
    #[test]
    fn stereo_prefix_roundtrip() {
        let target: [[f32; NUM_BANDS]; MAX_CHANNELS] = [
            std::array::from_fn(|b| 3.0 - 0.1 * b as f32),
            std::array::from_fn(|b| 2.5 - 0.08 * b as f32),
        ];
        let spec = FramePrefixSpec {
            header: CeltFrameHeader {
                silence: false,
                post_filter: None,
                transient: false,
                intra: false,
                anti_collapse_on: None,
            },
            energy_target: &target,
            tf: TfParameters {
                tf_changes: vec![false; NUM_BANDS],
                tf_select: 0,
                tf_select_decoded: false,
            },
            spread: Spread::Aggressive,
            target_boost: &[0i32; NUM_BANDS],
            allocation: BandAllocation {
                alloc_trim: 4,
                skip: false,
                intensity_band_offset: 5,
                dual_stereo: true,
            },
        };
        let (enc_prefix, dec_prefix) = encode_and_decode(&spec, 1, true, 0, NUM_BANDS);
        assert_prefix_eq(&enc_prefix, &dec_prefix);
        assert_eq!(dec_prefix.allocation.intensity_band_offset, 5);
        assert!(dec_prefix.allocation.dual_stereo);
    }

    /// A Hybrid-window (bands 17..21) prefix round-trips.
    #[test]
    fn hybrid_window_prefix_roundtrip() {
        let target = mono_target();
        let spec = FramePrefixSpec {
            header: CeltFrameHeader {
                silence: false,
                post_filter: None,
                transient: false,
                intra: true,
                anti_collapse_on: None,
            },
            energy_target: &target,
            tf: TfParameters {
                tf_changes: vec![false; 4],
                tf_select: 0,
                tf_select_decoded: false,
            },
            spread: Spread::None,
            target_boost: &[0, 0, 0, 0],
            allocation: BandAllocation::defaults(),
        };
        let (enc_prefix, dec_prefix) = encode_and_decode(&spec, 3, false, 17, NUM_BANDS);
        assert_prefix_eq(&enc_prefix, &dec_prefix);
        assert_eq!(dec_prefix.start, 17);
        assert_eq!(dec_prefix.end, NUM_BANDS);
    }

    /// Bad inputs are rejected before anything is written.
    #[test]
    fn rejects_bad_inputs() {
        let target = mono_target();
        let base = FramePrefixSpec {
            header: CeltFrameHeader {
                silence: false,
                post_filter: None,
                transient: false,
                intra: true,
                anti_collapse_on: None,
            },
            energy_target: &target,
            tf: TfParameters {
                tf_changes: vec![false; NUM_BANDS],
                tf_select: 0,
                tf_select_decoded: false,
            },
            spread: Spread::Normal,
            target_boost: &[0i32; NUM_BANDS],
            allocation: BandAllocation::defaults(),
        };
        let mut enc = RangeEncoder::new();
        let mut state = CoarseEnergyState::new();
        // lm out of range.
        assert!(encode_frame_prefix(
            &mut enc,
            &mut state,
            &base,
            4,
            FRAME_BYTES,
            false,
            0,
            NUM_BANDS
        )
        .is_err());
        // band window out of range.
        assert!(encode_frame_prefix(
            &mut enc,
            &mut state,
            &base,
            2,
            FRAME_BYTES,
            false,
            0,
            NUM_BANDS + 1
        )
        .is_err());
        // tf length mismatch (window is 4 bands, tf has 21).
        assert!(
            encode_frame_prefix(&mut enc, &mut state, &base, 2, FRAME_BYTES, false, 17, 21)
                .is_err()
        );
    }
}
