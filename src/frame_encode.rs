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
use crate::band_analysis::analyze_band_f32;
use crate::band_cap::{compute_band_caps, encode_band_boosts};
use crate::band_energy::assemble_band_log_energy_q8;
use crate::band_layout::{band_bins, coded_total_bins};
use crate::band_split::band_needs_split;
use crate::bit_allocation::{encode_band_allocation, BandAllocation};
use crate::coarse_energy::{encode_coarse_energy, CoarseEnergyState, MAX_CHANNELS, NUM_BANDS};
use crate::denormalization::denormalize_band_in_place_f32;
use crate::fine_energy::{encode_fine_energy_band, fine_correction_q14, quantize_fine_energy_f32};
use crate::frame_decode::FramePrefix;
use crate::frame_header::CeltFrameHeader;
use crate::pvq::{encode_pulses, normalize_to_unit_l2, pvq_search};
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

/// The result of encoding one mono CELT frame from an MDCT-domain
/// spectrum.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// The complete range-coded frame, exactly `frame_bytes` long
    /// (§5.1 layout: range symbols from the front, §5.1.3 raw bits
    /// from the back).
    pub bytes: Vec<u8>,
    /// The control prefix as the decoder will reconstruct it.
    pub prefix: FramePrefix,
    /// The final per-band Q8 log-energy envelope (coarse + fine), on
    /// the absolute band axis — the envelope the §4.3.6
    /// denormalization on the decode side reproduces.
    pub envelope_q8: [i32; NUM_BANDS],
    /// The encoder's own reconstruction of the coded-window
    /// denormalized spectrum (`coded_total_bins(start, end, lm)`
    /// samples): each band is its quantized PVQ pulse vector,
    /// unit-normalized and scaled by the quantized envelope. A
    /// matching `decode_residual_bands` pass over `bytes` reproduces
    /// these samples **bit-exactly** (both sides run the identical
    /// §4.3.4.2 normalization and §4.3.6 denormalization arithmetic).
    pub reconstructed_spectrum: Vec<f32>,
}

/// Encode one **mono, non-transient (single long MDCT)** CELT frame
/// from an MDCT-domain spectrum — the inverse of
/// [`crate::frame_synthesis::decode_celt_frame`]'s bitstream walk.
///
/// Chains the whole documented encode pipeline: §4.3.6 band analysis
/// (energy + unit shape, via [`analyze_band_f32`]) → Table-56 prefix
/// ([`encode_frame_prefix`], including the §4.3.2.1 coarse-energy
/// quantization against `coarse_state`) → §4.3.2.2 fine-energy
/// quantization + encode → §4.3.4.2 PVQ shape search + encode per band
/// → §5.1.5 fixed-size frame assembly
/// ([`RangeEncoder::finish_to_size`]).
///
/// ## Parameters
///
/// * `coarse_state` — the encoder's §4.3.2.1 inter-frame prediction
///   state, mutated with the coded reconstruction (it tracks the
///   decoder's state exactly).
/// * `spectrum` — the coded-window MDCT-domain samples, band-contiguous
///   (`coded_total_bins(start, end, lm)` samples; the
///   [`crate::synthesis::place_residual_spectrum`] layout).
/// * `header` — the frame's control scalars. Must be non-silent and
///   non-transient (`Error::NotImplemented` otherwise — the §4.3.1
///   short-block geometry is the same docs-gap boundary the decoder
///   keeps).
/// * `fine_bits` / `band_k` — the §4.3.2.2 fine-bit and §4.3.4.1 pulse
///   allocations, the same RFC-deferred §4.3.3 inputs
///   `decode_celt_frame` takes (identical values must be given to the
///   decoder).
///
/// ## Encoder decisions pinned by this driver
///
/// The spread is forced to [`Spread::None`] and every `tf_change` to
/// zero (both legal encoder choices): the decoder applies the §4.3.4.3
/// rotation and the §4.3.4.5 Hadamard *after* PVQ decode, so encoding
/// with a non-identity spread/TF would require searching the PVQ
/// codebook against the inverse-rotated target. Those inverses are
/// fully specified (orthonormal transforms) and can land later without
/// reshaping this driver; forcing the identity keeps the quantized
/// shape exactly the vector the decoder reconstructs. No band boosts
/// are requested and the allocation fields stay at their §4.3.3
/// defaults.
///
/// Returns the [`EncodedFrame`] — including the encoder's own
/// bit-exact reconstruction of what the decoder will produce — or
/// [`Error::InvalidParameter`] / [`Error::NotImplemented`] on an
/// out-of-range window, a spectrum length mismatch, a `band_k` length
/// mismatch, a band whose `V(N, K)` saturates (the §4.3.4.4 split
/// gap), or a frame the target byte budget cannot hold.
#[allow(clippy::too_many_arguments)]
pub fn encode_celt_frame(
    coarse_state: &mut CoarseEnergyState,
    spectrum: &[f32],
    header: &CeltFrameHeader,
    lm: u32,
    frame_bytes: u32,
    start: usize,
    end: usize,
    fine_bits: &[u32; NUM_BANDS],
    band_k: &[u32],
) -> Result<EncodedFrame, Error> {
    encode_celt_frame_impl(
        coarse_state,
        spectrum,
        header,
        lm,
        frame_bytes,
        start,
        end,
        fine_bits,
        Some(band_k),
        None,
    )
}

/// Encode one mono, non-transient CELT frame **deriving the per-band
/// pulse counts** from the documented §4.3.3 / §4.3.4.1 allocation
/// arithmetic rather than taking them as a caller input — the encode
/// counterpart of [`crate::derive_pulses::decode_celt_frame_auto`].
///
/// After the Table-56 prefix is encoded, the pulse counts are derived
/// from the *returned* [`FramePrefix`] via
/// [`crate::derive_pulses::derive_band_pulses`] — the identical
/// function the auto-decoder runs over the *decoded* prefix. The two
/// prefixes are bit-identical (proven by the `encode_frame_prefix`
/// round-trips), so both sides land on the same `band_k` with **no
/// allocation exchanged out of band**: `encode_celt_frame_auto` →
/// [`decode_celt_frame_auto`](crate::derive_pulses::decode_celt_frame_auto)
/// is a fully self-contained codec loop. Like the auto-decoder, no
/// fine-energy refinement is spent (`fine_bits = 0`, the RFC-deferred
/// fine/shape split approximated by treating the whole combined
/// allocation as shape).
///
/// The §4.3.3 dynalloc band boosts are chosen by the §5.3.4.1
/// contrast rule
/// ([`choose_band_boosts`](crate::encoder_decisions::choose_band_boosts))
/// over the analyzed band energies — a band poking far enough above
/// its neighbors requests one or two boost quanta, gate-truncated
/// against the live budget by the boost encoder. The decoder reads
/// the boosts back from the prefix, so the derived `band_k` stays in
/// lockstep automatically. Use [`encode_celt_frame_auto_boosted`] to
/// override the decision.
///
/// Parameters and error behaviour match [`encode_celt_frame`], minus
/// the two allocation inputs.
pub fn encode_celt_frame_auto(
    coarse_state: &mut CoarseEnergyState,
    spectrum: &[f32],
    header: &CeltFrameHeader,
    lm: u32,
    frame_bytes: u32,
    start: usize,
    end: usize,
) -> Result<EncodedFrame, Error> {
    let fine_bits = [0u32; NUM_BANDS];
    encode_celt_frame_impl(
        coarse_state,
        spectrum,
        header,
        lm,
        frame_bytes,
        start,
        end,
        &fine_bits,
        None,
        None,
    )
}

/// [`encode_celt_frame_auto`] with **caller-chosen §4.3.3 band-boost
/// targets** instead of the automatic §5.3.4.1 decision.
///
/// `target_boost` holds one 1/8-bit boost target per coded band
/// (`end - start` entries — the
/// [`choose_band_boosts`](crate::encoder_decisions::choose_band_boosts)
/// output shape). Targets floor to the band's dynalloc quanta grid
/// and truncate at the budget/cap gates, exactly like the underlying
/// [`encode_band_boosts`] — boosts are encoder freedom (§5.3.4), so
/// any choice keeps the decoder in lockstep. Everything else matches
/// [`encode_celt_frame_auto`].
#[allow(clippy::too_many_arguments)]
pub fn encode_celt_frame_auto_boosted(
    coarse_state: &mut CoarseEnergyState,
    spectrum: &[f32],
    header: &CeltFrameHeader,
    lm: u32,
    frame_bytes: u32,
    start: usize,
    end: usize,
    target_boost: &[i32],
) -> Result<EncodedFrame, Error> {
    let fine_bits = [0u32; NUM_BANDS];
    encode_celt_frame_impl(
        coarse_state,
        spectrum,
        header,
        lm,
        frame_bytes,
        start,
        end,
        &fine_bits,
        None,
        Some(target_boost),
    )
}

/// Shared engine for [`encode_celt_frame`] (caller-supplied `band_k`,
/// zero boosts), [`encode_celt_frame_auto`] (`band_k` derived from
/// the encoded prefix via the documented §4.3.3 → §4.3.4.1 seam,
/// §5.3.4.1 automatic boosts), and
/// [`encode_celt_frame_auto_boosted`] (derived `band_k`,
/// caller-chosen boosts).
#[allow(clippy::too_many_arguments)]
fn encode_celt_frame_impl(
    coarse_state: &mut CoarseEnergyState,
    spectrum: &[f32],
    header: &CeltFrameHeader,
    lm: u32,
    frame_bytes: u32,
    start: usize,
    end: usize,
    fine_bits: &[u32; NUM_BANDS],
    band_k: Option<&[u32]>,
    target_boost: Option<&[i32]>,
) -> Result<EncodedFrame, Error> {
    if lm > 3 || start > end || end > NUM_BANDS {
        return Err(Error::InvalidParameter);
    }
    if header.transient || header.silence {
        return Err(Error::NotImplemented);
    }
    let coded_bands = end - start;
    if let Some(k) = band_k {
        if k.len() != coded_bands {
            return Err(Error::InvalidParameter);
        }
    }
    let total_bins = coded_total_bins(start, end, lm).ok_or(Error::InvalidParameter)? as usize;
    if spectrum.len() != total_bins {
        return Err(Error::InvalidParameter);
    }

    // §4.3.6 analysis: split each band into its log-2 energy target and
    // its unit-norm shape.
    let mut energy_target: [[f32; NUM_BANDS]; MAX_CHANNELS] = coarse_state.energy;
    let mut shapes: Vec<Vec<f32>> = Vec::with_capacity(coded_bands);
    {
        let mut offset = 0usize;
        for (band, slot) in energy_target[0]
            .iter_mut()
            .enumerate()
            .take(end)
            .skip(start)
        {
            let n = band_bins(band, lm).ok_or(Error::InvalidParameter)? as usize;
            let analysis = analyze_band_f32(&spectrum[offset..offset + n]);
            *slot = analysis.log_energy;
            shapes.push(analysis.shape);
            offset += n;
        }
    }

    // Resolve the §4.3.3 boost targets: caller-chosen, the §5.3.4.1
    // contrast rule over the analyzed energies (auto mode), or zero
    // (explicit-allocation mode, where the caller's fixed `band_k`
    // ignores the prefix-derived allocation anyway).
    let derived_boost: Vec<i32>;
    let target_boost: &[i32] = match target_boost {
        Some(tb) => {
            if tb.len() != coded_bands {
                return Err(Error::InvalidParameter);
            }
            tb
        }
        None if band_k.is_none() => {
            derived_boost = crate::encoder_decisions::choose_band_boosts(
                &energy_target[0][start..end],
                lm,
                start,
                end,
            )
            .ok_or(Error::InvalidParameter)?;
            &derived_boost
        }
        None => {
            derived_boost = vec![0i32; coded_bands];
            &derived_boost
        }
    };

    // Table-56 prefix: header → coarse energy → TF (identity) → spread
    // (None) → boosts (§5.3.4.1 / caller-chosen, gate-truncated) →
    // allocation (defaults).
    let mut enc = RangeEncoder::new();
    let spec = FramePrefixSpec {
        header: CeltFrameHeader {
            anti_collapse_on: None,
            ..*header
        },
        energy_target: &energy_target,
        tf: TfParameters {
            tf_changes: vec![false; coded_bands],
            tf_select: 0,
            tf_select_decoded: false,
        },
        spread: Spread::None,
        target_boost,
        allocation: BandAllocation::defaults(),
    };
    let prefix = encode_frame_prefix(
        &mut enc,
        coarse_state,
        &spec,
        lm,
        frame_bytes,
        false,
        start,
        end,
    )?;

    // Resolve the per-band pulse counts: caller-supplied, or derived
    // from the just-encoded prefix via the documented §4.3.3 →
    // §4.3.4.1 seam (the identical derivation the auto-decoder runs
    // over the decoded prefix).
    let derived_k;
    let band_k: &[u32] = match band_k {
        Some(k) => k,
        None => {
            derived_k = crate::derive_pulses::derive_band_pulses(&prefix, lm, 1, false)
                .ok_or(Error::InvalidParameter)?;
            &derived_k
        }
    };

    // §4.3.2.2 fine energy: quantize the residual left by the coarse
    // step and write each band's B_i raw bits, walking all 21 bands in
    // the same order `decode_fine_energy` reads them.
    let mut fine_q14 = [0i32; NUM_BANDS];
    for band in 0..NUM_BANDS {
        let b_bits = fine_bits[band];
        let residual = if (start..end).contains(&band) {
            energy_target[0][band] - coarse_state.energy[0][band]
        } else {
            0.0
        };
        let f = quantize_fine_energy_f32(residual, b_bits);
        encode_fine_energy_band(&mut enc, f, b_bits)?;
        fine_q14[band] = fine_correction_q14(f, b_bits);
    }

    // §4.3.2 final envelope (coarse + fine), the amplitude the §4.3.6
    // denormalization applies on both sides.
    let envelope_q8 = assemble_band_log_energy_q8(coarse_state, 0, Some(&fine_q14), None)
        .ok_or(Error::InvalidParameter)?;

    // §4.3.4.2 PVQ shape search + encode per band, reconstructing the
    // decoder's view as we go.
    let mut reconstructed = Vec::with_capacity(total_bins);
    for (i, band) in (start..end).enumerate() {
        let n = band_bins(band, lm).ok_or(Error::InvalidParameter)?;
        let k = band_k[i];
        if band_needs_split(n, k) {
            // The §4.3.4.4 split-gain path is the documented docs gap.
            return Err(Error::NotImplemented);
        }
        if k == 0 || n == 0 {
            // No pulses: the decoder reconstructs the zero shape (its
            // §4.3.4.2 normalization degrades all-zero to all-zero).
            reconstructed.resize(reconstructed.len() + n as usize, 0.0f32);
            continue;
        }
        let pulses = pvq_search(&shapes[i], n, k).ok_or(Error::InvalidParameter)?;
        encode_pulses(&mut enc, &pulses, n, k)?;
        // The decoder's exact reconstruction arithmetic: unit-normalize
        // the integer pulses, then scale by the quantized envelope.
        let mut band_samples = normalize_to_unit_l2(&pulses);
        denormalize_band_in_place_f32(&mut band_samples, envelope_q8[band]);
        reconstructed.extend_from_slice(&band_samples);
    }

    // §5.1.5 fixed-size assembly: range symbols from the front, raw
    // bits at the frame end, matching the decoder's storage_bits().
    let bytes = enc.finish_to_size(frame_bytes as usize)?;

    Ok(EncodedFrame {
        bytes,
        prefix,
        envelope_q8,
        reconstructed_spectrum: reconstructed,
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

    /// A spectrum with one band poking far above its neighbors earns a
    /// §5.3.4.1 boost in the auto encoder, and the boosted frame still
    /// closes the self-contained auto codec loop (bit-identical
    /// prefix, coarse lockstep).
    #[test]
    fn auto_boost_lands_and_round_trips() {
        use crate::band_layout::band_bin_range;
        use crate::derive_pulses::decode_celt_frame_auto;
        use crate::frame_synthesis::CeltDecodeState;

        let lm = 2u32;
        let total = coded_total_bins(0, NUM_BANDS, lm).unwrap() as usize;
        // Baseline amplitude everywhere, one band 8x louder:
        // E_peak - E_neighbor = log2(64) = 6, so D = 12 > t2 = 4 —
        // boosted twice.
        let peak_band = 10usize;
        let (lo, hi) = band_bin_range(peak_band, lm).unwrap();
        let mut spectrum = vec![0.05f32; total];
        for s in &mut spectrum[lo as usize..hi as usize] {
            *s = 0.4;
        }

        let header = CeltFrameHeader {
            silence: false,
            post_filter: None,
            transient: false,
            intra: true,
            anti_collapse_on: None,
        };
        let mut enc_state = CoarseEnergyState::new();
        let frame = encode_celt_frame_auto(
            &mut enc_state,
            &spectrum,
            &header,
            lm,
            FRAME_BYTES,
            0,
            NUM_BANDS,
        )
        .unwrap();
        // Two quanta for band 10 (N = 8 at LM=2 → quanta = 48).
        let n = crate::band_minimums::BAND_BINS_LM[lm as usize][peak_band];
        assert_eq!(
            frame.prefix.boosts.boost[peak_band],
            2 * crate::encoder_decisions::boost_quanta_8th(n),
            "the §5.3.4.1 double boost must land on the wire"
        );
        assert!(frame
            .prefix
            .boosts
            .boost
            .iter()
            .enumerate()
            .all(|(b, &v)| b == peak_band || v == 0));

        // The boosted frame still decodes caller-input-free with the
        // identical prefix and coarse state.
        let mut dec_state = CeltDecodeState::new(lm).unwrap();
        let decoded = decode_celt_frame_auto(&mut dec_state, &frame.bytes, 0, NUM_BANDS).unwrap();
        assert_eq!(frame.prefix.boosts, decoded.prefix.boosts);
        assert_eq!(enc_state.energy, dec_state.coarse_energy().energy);
        assert!(decoded.pcm.iter().all(|s| s.is_finite()));
    }

    /// The explicit-boost variant honours a caller-chosen target (on
    /// the quanta grid, within budget) and rejects a length mismatch.
    #[test]
    fn explicit_boost_variant() {
        use crate::derive_pulses::decode_celt_frame_auto;
        use crate::frame_synthesis::CeltDecodeState;

        let lm = 2u32;
        let total = coded_total_bins(0, NUM_BANDS, lm).unwrap() as usize;
        let spectrum = vec![0.1f32; total];
        let header = CeltFrameHeader {
            silence: false,
            post_filter: None,
            transient: false,
            intra: true,
            anti_collapse_on: None,
        };
        // One quantum on band 3 (N = 4 → quanta = 32).
        let mut target = vec![0i32; NUM_BANDS];
        target[3] = 32;
        let mut enc_state = CoarseEnergyState::new();
        let frame = encode_celt_frame_auto_boosted(
            &mut enc_state,
            &spectrum,
            &header,
            lm,
            FRAME_BYTES,
            0,
            NUM_BANDS,
            &target,
        )
        .unwrap();
        assert_eq!(frame.prefix.boosts.boost[3], 32);
        let mut dec_state = CeltDecodeState::new(lm).unwrap();
        let decoded = decode_celt_frame_auto(&mut dec_state, &frame.bytes, 0, NUM_BANDS).unwrap();
        assert_eq!(frame.prefix.boosts, decoded.prefix.boosts);

        // Length mismatch is rejected.
        let mut fresh = CoarseEnergyState::new();
        assert!(encode_celt_frame_auto_boosted(
            &mut fresh,
            &spectrum,
            &header,
            lm,
            FRAME_BYTES,
            0,
            NUM_BANDS,
            &target[..5],
        )
        .is_err());
    }
}
