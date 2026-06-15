//! CELT multi-band residual (shape) decode loop — the §4.3.4
//! integration spine (RFC 6716 §4.3.4 → §4.3.6).
//!
//! ## What this module covers
//!
//! The single-band shape chain ([`crate::band_decode::decode_band_shape`])
//! decodes one band: PVQ index → unit shape → §4.3.4.3 spreading →
//! §4.3.4.5 TF resolution change → §4.3.6 denormalization. What was
//! missing was the **band loop**: a driver that walks the coded-band
//! window `[start, end)` in bitstream order, invoking the single-band
//! chain for each band with that band's own dimension `N`, short-block
//! count, TF adjustment, and decoded energy, and assembling the per-band
//! outputs into the contiguous MDCT-domain spectrum the §4.3.7 inverse
//! MDCT consumes.
//!
//! [`decode_residual_bands`] is that driver. It is the residual-section
//! counterpart of [`crate::frame_decode::decode_frame_prefix`] (which is
//! the *prefix*-section spine): given the per-band pulse counts the
//! §4.3.3 allocation produced, it decodes every band's shape and lays
//! the samples out in spectral order.
//!
//! ## What it consumes, and why that boundary
//!
//! The per-band pulse counts `K[band]` are an **input** here, not a
//! computed quantity. They are the output of the §4.3.4.1 bits-to-pulses
//! search ([`crate::bits_to_pulses`]) driven by the §4.3.3 per-band
//! shape allocation. The precise §4.3.3 reallocation pass that produces
//! that allocation — the per-band bisection subject to caps/minimums,
//! the concurrent skip decoding, and the fine-energy vs. shape split —
//! is delegated to the reference implementation by RFC 6716 §4.3.3
//! ("the methods used by the reference implementation should be used")
//! and remains a documented docs gap. By taking `K[]` as a parameter,
//! this loop stays entirely within the fully-specified §4.3.4 territory:
//! the per-band decode chain and the Table 55 band layout. The same
//! boundary is why [`crate::bits_to_pulses::bits_to_pulses_band_loop`]
//! takes per-band *targets* rather than computing them.
//!
//! ## Band geometry
//!
//! For each coded band `b`:
//!
//! * The per-channel sample count is `N = band_bins(b, lm)`
//!   ([`crate::band_layout::band_bins`], RFC 6716 §4.3 Table 55) — the
//!   2.5 ms "Bins" column scaled by `1 << lm`.
//! * The short-block count is `B = 2^lm` when the frame is transient
//!   (the "several short MDCTs" of §4.3.1) and `1` otherwise. The band's
//!   `N` samples are interleaved across the `B` blocks; the single-band
//!   chain ([`decode_band_shape`](crate::band_decode::decode_band_shape))
//!   takes `nb_blocks = B` and applies the §4.3.4.3 rotation and the
//!   §4.3.4.5 TF Hadamard independently per block. `N` is always
//!   divisible by `B` because every Table 55 entry is a multiple of
//!   `1 << lm`.
//! * The TF adjustment is
//!   [`tf_adjustment`](crate::tf_change::tf_adjustment) evaluated from
//!   the frame transient flag, the global `tf_select`, `lm`, and the
//!   band's own `tf_change` bit.
//!
//! The decoded samples for band `b` occupy the half-open MDCT-bin range
//! `band_bin_range(b, lm)` ([`crate::band_layout::band_bin_range`]),
//! offset to the window start so a Hybrid-mode window (`start = 17`)
//! produces a spectrum indexed from its first coded band.
//!
//! ## Scope
//!
//! This is the **mono / per-channel, non-split** band loop. A band whose
//! codebook `V(N, K)` overflows 32 bits needs the §4.3.4.4 recursive
//! split ([`crate::band_split`]), whose quantized split-gain PDF the RFC
//! defers to the reference (a documented gap); such a band makes the
//! single-band chain return `None`, which this loop surfaces as an
//! error rather than silently mis-decoding. Stereo joint coding (the
//! §4.3.4.4 `itheta` mid/side angle) and the §4.3.5 anti-collapse pass
//! are likewise out of scope for the same docs-gap reason; this loop is
//! the per-channel shape assembly that precedes anti-collapse.
//!
//! ## Clean-room provenance
//!
//! The decode chain order is RFC 6716 §4.3.4 (lines 6462–6474); the
//! short-block count semantics are §4.3.1 (lines 6009–6022); the band
//! layout is §4.3 Table 55 (lines 5813–5870). Every step delegates to an
//! existing RFC-grounded primitive whose own provenance is recorded in
//! its module. No external library source was consulted.

use crate::band_decode::decode_band_shape;
use crate::band_layout::{band_bin_range, band_bins, coded_total_bins};
use crate::coarse_energy::NUM_BANDS;
use crate::range_decoder::RangeDecoder;
use crate::spread::Spread;
use crate::tf_change::tf_adjustment;
use crate::Error;

/// The decoded residual spectrum for the coded-band window, plus the
/// per-band pulse counts the loop consumed.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualSpectrum {
    /// The denormalized MDCT-domain samples for the whole coded-band
    /// window `[start, end)`, laid out contiguously in spectral order
    /// (band `start` first). Length is
    /// [`coded_total_bins(start, end, lm)`](crate::band_layout::coded_total_bins),
    /// i.e. the sum of every coded band's `N`. For a transient frame the
    /// samples within a band are interleaved across the `2^lm` short
    /// blocks (the layout the §4.3.7 inverse MDCT de-interleaves).
    pub samples: Vec<f32>,
    /// The pulse count `K` consumed for each coded band, in band order
    /// from `start` to `end - 1`. Length is `end - start`. Echoes the
    /// `k` input so callers driving the §4.3.4.1 balance accumulator can
    /// confirm the per-band consumption.
    pub band_k: Vec<u32>,
}

/// Decode the §4.3.4 residual (shape) section across the coded-band
/// window and assemble the MDCT-domain spectrum.
///
/// Walks `start..end` in bitstream order. For each band it computes the
/// band dimension `N` and short-block count `B` from the Table 55 layout
/// and the transient flag, evaluates the §4.3.4.5 TF adjustment, and
/// invokes the single-band chain
/// ([`decode_band_shape`](crate::band_decode::decode_band_shape)),
/// placing the result in the band's spectral slot.
///
/// Parameters:
///
/// * `dec` — the range decoder positioned at the first band's PVQ index
///   (i.e. just past the §4.3.3 fine-energy / allocation section). It is
///   advanced through every band's index in order.
/// * `lm` — frame-size shift `log2(frame_size / 120)`, in `0..=3`.
/// * `start` / `end` — coded-band window, `start <= end <= 21`. Pure
///   CELT is `(0, 21)`; Hybrid mode is `(17, 21)`; narrower bandwidths
///   reduce `end`.
/// * `is_transient` — the frame transient flag (§4.3.1). Selects the
///   short-block count `B = 2^lm` (vs. `B = 1` for a long MDCT) and the
///   TF-adjustment table family.
/// * `tf_select` — the global `tf_select` value (`0` if the bit was
///   gated off), from
///   [`TfParameters`](crate::tf_change::TfParameters).
/// * `tf_changes` — the per-band `tf_change` bits, one per coded band in
///   `start..end` order. Length MUST equal `end - start`.
/// * `band_k` — the per-band pulse count `K`, one per coded band in
///   `start..end` order (the §4.3.4.1 bits-to-pulses output). Length
///   MUST equal `end - start`.
/// * `spread` — the §4.3.4.3 spreading selector
///   ([`crate::spread::decode_spread`]).
/// * `log_energy_q8` — the per-band reconstructed band energy in the Q8
///   base-2 log domain, one per coded band in `start..end` order (the
///   §4.3.2 coarse+fine envelope, ×256 and rounded). Length MUST equal
///   `end - start`.
///
/// Returns the assembled [`ResidualSpectrum`] on success, or:
///
/// * [`Error::InvalidParameter`] when `lm > 3`, the band window is
///   out of range (`start > end` or `end > 21`), or any of the
///   per-band slices' lengths disagree with `end - start`.
/// * [`Error::NotImplemented`] when a band's codebook saturates (the
///   §4.3.4.4 split path, a documented gap), `N` is not divisible by
///   the short-block count, or the single-band chain otherwise rejects
///   the band parameterisation — surfaced rather than silently emitting
///   a wrong-energy band.
///
/// A sticky range-decoder error mid-walk propagates through the
/// single-band chain (it falls back to its defined low-budget
/// behaviour); callers should check [`RangeDecoder::has_error`] after
/// the call for the §4.1.5 corrupt-frame path.
#[allow(clippy::too_many_arguments)]
pub fn decode_residual_bands(
    dec: &mut RangeDecoder<'_>,
    lm: u32,
    start: usize,
    end: usize,
    is_transient: bool,
    tf_select: u8,
    tf_changes: &[bool],
    band_k: &[u32],
    spread: Spread,
    log_energy_q8: &[i32],
) -> Result<ResidualSpectrum, Error> {
    if lm > 3 || start > end || end > NUM_BANDS {
        return Err(Error::InvalidParameter);
    }
    let coded_bands = end - start;
    if tf_changes.len() != coded_bands
        || band_k.len() != coded_bands
        || log_energy_q8.len() != coded_bands
    {
        return Err(Error::InvalidParameter);
    }

    // The §4.3.1 short-block count: 2^lm short MDCTs on a transient
    // frame, a single long MDCT otherwise.
    let nb_blocks: u32 = if is_transient { 1u32 << lm } else { 1 };

    // The contiguous output spectrum spans every coded band's N. The
    // window-overflow / lm guards are already pinned above, so this
    // cannot be None.
    let total_bins = coded_total_bins(start, end, lm).ok_or(Error::InvalidParameter)? as usize;
    let mut samples = vec![0f32; total_bins];

    // The window origin in absolute MDCT-bin coordinates, so a
    // Hybrid-mode window (start = 17) maps its first coded band to
    // output offset 0.
    let window_origin = band_bin_range(start.min(NUM_BANDS - 1), lm)
        .map(|(s, _)| s)
        .unwrap_or(0);

    let mut out_k = Vec::with_capacity(coded_bands);

    for (i, band) in (start..end).enumerate() {
        // §4.3 Table 55: this band's per-channel sample count at lm.
        let n = band_bins(band, lm).ok_or(Error::InvalidParameter)?;
        let (abs_start, abs_end) = band_bin_range(band, lm).ok_or(Error::InvalidParameter)?;

        let k = band_k[i];
        let energy = log_energy_q8[i];

        // §4.3.4.5 per-band TF adjustment.
        let tf_adj = tf_adjustment(is_transient, tf_select, lm as u8, tf_changes[i]);

        // §4.3.4 → §4.3.6 single-band chain. A `None` here is a
        // saturated codebook (the §4.3.4.4 split gap), an indivisible
        // block count, or an over-large TF request — none of which this
        // mono non-split loop can resolve, so surface it as
        // NotImplemented rather than emit a wrong band.
        let shape = decode_band_shape(dec, n, k, spread, nb_blocks, tf_adj, energy)
            .ok_or(Error::NotImplemented)?;

        // Lay the band's samples into its spectral slot.
        let lo = (abs_start - window_origin) as usize;
        let hi = (abs_end - window_origin) as usize;
        debug_assert_eq!(hi - lo, shape.samples.len());
        samples[lo..hi].copy_from_slice(&shape.samples);

        out_k.push(shape.k);
    }

    Ok(ResidualSpectrum {
        samples,
        band_k: out_k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band_layout::coded_total_bins;

    fn l2(samples: &[f32]) -> f64 {
        samples
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt()
    }

    /// A pure-CELT mono long-MDCT frame decodes every band into the
    /// contiguous spectrum; the output length is the window's total bin
    /// count and the per-band slots tile it without gaps.
    #[test]
    fn decodes_full_window_mono_long() {
        let buf = [0xA5u8; 256];
        let mut dec = RangeDecoder::new(&buf);
        let lm = 3;
        let (start, end) = (0usize, 21usize);
        let n_bands = end - start;
        // Two pulses per band keeps every codebook well inside 32 bits;
        // the 256-byte buffer is ample for 21 PVQ indices.
        let band_k = vec![2u32; n_bands];
        let tf_changes = vec![false; n_bands];
        let log_e = vec![0i32; n_bands];

        let res = decode_residual_bands(
            &mut dec,
            lm,
            start,
            end,
            false,
            0,
            &tf_changes,
            &band_k,
            Spread::Normal,
            &log_e,
        )
        .expect("residual decode");

        let total = coded_total_bins(start, end, lm).unwrap() as usize;
        assert_eq!(res.samples.len(), total);
        assert_eq!(res.band_k, band_k);

        // Each band's slot has the unit-norm energy (E_q8 = 0 ⇒
        // amplitude 1) so the whole window's energy is sqrt(n_bands)
        // (orthogonal per-band unit vectors), up to PVQ rounding.
        let total_energy = l2(&res.samples);
        assert!(
            (total_energy - (n_bands as f64).sqrt()).abs() < 1e-2,
            "window energy {total_energy} != sqrt({n_bands})"
        );
    }

    /// A transient frame uses `2^lm` short blocks; the loop still
    /// produces the same total bin count and unit-norm per band. The
    /// short-block geometry (`N` divisible by `2^lm`, samples
    /// interleaved per block) is exercised with `Spread::None`, which
    /// keeps the per-block sub-vector decode a pure norm-preserving
    /// pass-through even for the narrow low bands (where `N / 2^lm`
    /// drops to a single sample).
    #[test]
    fn decodes_transient_short_blocks() {
        let buf = [0x3Cu8; 256];
        let mut dec = RangeDecoder::new(&buf);
        let lm = 2; // 4 short blocks
        let (start, end) = (0usize, 21usize);
        let n_bands = end - start;
        // One pulse per band keeps every per-band PVQ index cheap so the
        // 256-byte buffer never exhausts across all 21 bands.
        let band_k = vec![1u32; n_bands];
        // tf_change = true at LM=2, tf_select=0 ⇒ TABLE_62[2][1] = 0, a
        // no-op TF adjustment. This isolates the transient short-block
        // geometry from the §4.3.4.5 Hadamard pass, which a +N
        // adjustment would request more levels of than a 1-sample
        // per-block sub-vector can provide (a constraint the single-band
        // chain correctly rejects).
        let tf_changes = vec![true; n_bands];
        let log_e = vec![0i32; n_bands];

        let res = decode_residual_bands(
            &mut dec,
            lm,
            start,
            end,
            true,
            0,
            &tf_changes,
            &band_k,
            Spread::None,
            &log_e,
        )
        .expect("transient residual decode");

        assert_eq!(
            res.samples.len(),
            coded_total_bins(start, end, lm).unwrap() as usize
        );
        // Unit per-band shapes ⇒ window energy ≈ sqrt(n_bands).
        assert!((l2(&res.samples) - (n_bands as f64).sqrt()).abs() < 1e-2);
    }

    /// A Hybrid-mode window (bands 17..=20) produces a 4-band spectrum
    /// indexed from offset 0, with length = the window's total bins.
    #[test]
    fn decodes_hybrid_window() {
        let buf = [0xD7u8; 64];
        let mut dec = RangeDecoder::new(&buf);
        let lm = 2;
        let (start, end) = (17usize, 21usize);
        let n_bands = end - start;
        let band_k = vec![3u32; n_bands];
        let tf_changes = vec![false; n_bands];
        let log_e = vec![0i32; n_bands];

        let res = decode_residual_bands(
            &mut dec,
            lm,
            start,
            end,
            false,
            0,
            &tf_changes,
            &band_k,
            Spread::Normal,
            &log_e,
        )
        .expect("hybrid residual decode");

        let total = coded_total_bins(start, end, lm).unwrap() as usize;
        assert_eq!(res.samples.len(), total);
        assert_eq!(res.band_k.len(), n_bands);
    }

    /// Per-band energy scales the band's slot: a band with a higher
    /// `E_q8` carries more energy than one at zero.
    #[test]
    fn per_band_energy_scales_slot() {
        let buf = [0x6Bu8; 64];
        let mut dec = RangeDecoder::new(&buf);
        let lm = 3;
        let (start, end) = (0usize, 21usize);
        let n_bands = end - start;
        let band_k = vec![2u32; n_bands];
        let tf_changes = vec![false; n_bands];
        // Band 0 at zero energy (amplitude 1); band 20 at E_q8 = 512.
        // The §4.3.6 amplitude is sqrt(2^(E_q8/256)) = 2^(E_q8/512), so
        // E_q8 = 512 ⇒ amplitude 2.0.
        let mut log_e = vec![0i32; n_bands];
        log_e[n_bands - 1] = 512;

        let res = decode_residual_bands(
            &mut dec,
            lm,
            start,
            end,
            false,
            0,
            &tf_changes,
            &band_k,
            Spread::None,
            &log_e,
        )
        .unwrap();

        // The last band's slot is the last band_bins(20, lm) samples.
        let (s, e) = band_bin_range(end - 1, lm).unwrap();
        let origin = band_bin_range(start, lm).unwrap().0;
        let slot = &res.samples[(s - origin) as usize..(e - origin) as usize];
        // Amplitude 2 ⇒ slot energy ≈ 2.
        assert!((l2(slot) - 2.0).abs() < 1e-2, "band-20 energy {}", l2(slot));
    }

    /// A `K = 0` band decodes to an all-zero slot (no PVQ index
    /// consumed) without disturbing the rest of the spectrum.
    #[test]
    fn zero_k_band_is_zero_slot() {
        let buf = [0xFFu8; 64];
        let mut dec = RangeDecoder::new(&buf);
        let lm = 3;
        let (start, end) = (0usize, 21usize);
        let n_bands = end - start;
        let mut band_k = vec![2u32; n_bands];
        band_k[0] = 0; // first band carries no pulses
        let tf_changes = vec![false; n_bands];
        let log_e = vec![100i32; n_bands];

        let res = decode_residual_bands(
            &mut dec,
            lm,
            start,
            end,
            false,
            0,
            &tf_changes,
            &band_k,
            Spread::None,
            &log_e,
        )
        .unwrap();

        // Band 0 occupies bin 0 only (N=1 at LM=3 ⇒ 8 bins). Its slot is
        // all zero because K=0 ⇒ zero pulse vector ⇒ zero shape.
        let (s, e) = band_bin_range(0, lm).unwrap();
        assert!(res.samples[s as usize..e as usize]
            .iter()
            .all(|&x| x == 0.0));
    }

    /// Length-mismatched per-band slices are rejected.
    #[test]
    fn rejects_length_mismatch() {
        let buf = [0x11u8; 32];
        let mut dec = RangeDecoder::new(&buf);
        // 21 bands but only 20 tf_changes.
        let r = decode_residual_bands(
            &mut dec,
            3,
            0,
            21,
            false,
            0,
            &[false; 20],
            &[1u32; 21],
            Spread::Normal,
            &[0i32; 21],
        );
        assert!(matches!(r, Err(Error::InvalidParameter)));
    }

    /// Out-of-range `lm` / window is rejected before touching the
    /// decoder.
    #[test]
    fn rejects_invalid_window() {
        let buf = [0x22u8; 32];
        let mut dec = RangeDecoder::new(&buf);
        let r = decode_residual_bands(
            &mut dec,
            4, // lm > 3
            0,
            21,
            false,
            0,
            &[false; 21],
            &[1u32; 21],
            Spread::Normal,
            &[0i32; 21],
        );
        assert!(matches!(r, Err(Error::InvalidParameter)));
    }

    /// A band whose codebook saturates surfaces as NotImplemented (the
    /// §4.3.4.4 split gap) rather than a silent mis-decode.
    #[test]
    fn saturated_band_is_not_implemented() {
        let buf = [0xFFu8; 64];
        let mut dec = RangeDecoder::new(&buf);
        let lm = 3;
        let (start, end) = (0usize, 21usize);
        let n_bands = end - start;
        // Band 20 (N = 22<<3 = 176) with a huge K saturates V(N, K)
        // beyond 32 bits, forcing the split path the loop can't take.
        let mut band_k = vec![2u32; n_bands];
        band_k[n_bands - 1] = 176;
        let tf_changes = vec![false; n_bands];
        let log_e = vec![0i32; n_bands];

        let r = decode_residual_bands(
            &mut dec,
            lm,
            start,
            end,
            false,
            0,
            &tf_changes,
            &band_k,
            Spread::None,
            &log_e,
        );
        assert!(matches!(r, Err(Error::NotImplemented)));
    }

    /// The empty window (`start == end`) decodes to an empty spectrum
    /// without consuming any bits.
    #[test]
    fn empty_window_is_empty() {
        let buf = [0x44u8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let tell = dec.tell();
        let res =
            decode_residual_bands(&mut dec, 3, 10, 10, false, 0, &[], &[], Spread::Normal, &[])
                .unwrap();
        assert!(res.samples.is_empty());
        assert!(res.band_k.is_empty());
        assert_eq!(dec.tell(), tell);
    }
}
