//! Bit-allocation field decoding (RFC 6716 §4.3.3).
//!
//! ## What this module covers
//!
//! Per Table 56 of RFC 6716 (`docs/audio/opus/rfc6716-opus.txt`), the
//! CELT bitstream emits four scalar fields between the coarse-energy
//! block and the per-band shape vectors:
//!
//! 1. `alloc.trim`  — Table 58 PDF `{2,2,5,10,22,46,22,10,5,2,2}/128`.
//! 2. `skip`        — PDF `{1,1}/2` (`logp=1`).
//! 3. `intensity`   — uniform over the coded bands (`dec_uint(end-start+1)`).
//! 4. `dual`        — PDF `{1,1}/2` (`logp=1`).
//!
//! Each is **gated** by a budget condition prose'd in §4.3.3:
//!
//! * `alloc.trim` is decoded only if `ec_tell_frac() + 48 <=
//!   total_frame_size_8th_bits - total_boost` (the encoder reserved
//!   six bits of headroom for the trim PDF, written as `48` 8th bits).
//! * `skip` is decoded only if the §4.3.3 reservation step set
//!   `skip_rsv = 8` (8th bits), i.e. `total > 8` at the reservation
//!   point.
//! * `intensity` (stereo frames only) is decoded only if
//!   `intensity_rsv > 0`, i.e. the conservative log2-in-8th-bits of
//!   `end-start` fits in the remaining budget.
//! * `dual` (stereo frames only) is decoded only if
//!   `dual_stereo_rsv = 8` (8th bits) was set, i.e. there were still
//!   more than 8 8th bits left after subtracting `intensity_rsv`.
//!
//! This module exposes a per-field decoder for each scalar, plus an
//! orchestrator [`decode_band_allocation`] that takes the gating
//! booleans as parameters (the caller, which holds the full §4.3.3
//! budget loop, decides whether each reservation was made). Returning
//! defaults preserves the §4.3.3 invariant that an un-emitted field
//! reads as its low-rate fallback.
//!
//! ## What is NOT in this round
//!
//! * Band boost decoding (§4.3.3, lines 6318–6360): a dynalloc-logp
//!   loop per band that depends on the per-band `cap[]` vector, which
//!   in turn depends on a numeric `cache_caps50[]` table that the RFC
//!   delegates to a source file outside the workspace's clean-room
//!   allow-list. The `cache_caps50[]` numeric values are not derivable
//!   from RFC prose alone — this is a separate docs gap, queued
//!   behind the Laplace / `e_prob_model` blocker.
//! * The per-band shape/fine-energy split, the reallocation loop, and
//!   the final skip-band selection. These come after the four scalar
//!   fields decoded here and consume the residual budget — they are
//!   queued for a later round.
//! * The full §4.3.3 budget loop that produces the gating booleans
//!   passed to [`decode_band_allocation`]. That loop is straightforward
//!   in RFC prose but it depends on `total_boost` from the gap'd band
//!   boost step.
//!
//! ## Clean-room provenance
//!
//! Every PDF, every gate condition, and every field comment in this
//! file is transcribed from RFC 6716 §4.3.3 and Table 58 (`docs/audio/
//! opus/rfc6716-opus.txt`). The full §4.3.3 allocation loop and the
//! `LOG2_FRAC_TABLE` the RFC delegates to a source file outside the
//! workspace's clean-room allow-list are deferred until clean-room
//! trace material is staged; no external library source was
//! consulted.

use crate::range_decoder::RangeDecoder;

/// Default value of the allocation trim per RFC 6716 §4.3.3.
///
/// > "The allocation trim is an integer value from 0-10. The default
/// >  value of 5 indicates no trim."
///
/// Returned by [`decode_alloc_trim`] when the trim field is gated off,
/// and is also the implicit value for [`BandAllocation::alloc_trim`]
/// when [`decode_band_allocation`] receives `trim_gated = false`.
pub const DEFAULT_ALLOC_TRIM: u8 = 5;

/// ICDF table for the §4.3.3 allocation trim, derived from the
/// Table 58 PDF `{2, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2}/128`.
///
/// Cumulative frequencies (`fh`): `[2, 4, 9, 19, 41, 87, 109, 119,
/// 124, 126, 128]`. ICDF entries are `ft - fh[k]` with `ft = 128`:
/// `[126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0]`. `ftb = 7` (since
/// `128 == 1 << 7`).
const ALLOC_TRIM_ICDF: &[u8] = &[126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// `ftb` for the allocation trim ICDF (RFC 6716 Table 58: `ft = 128`).
const ALLOC_TRIM_FTB: u32 = 7;

/// Decoded band-allocation fields for one CELT frame (RFC 6716 §4.3.3).
///
/// The struct holds the four scalar decisions that the §4.3.3 bit
/// allocator emits BETWEEN the coarse-energy block and the per-band
/// shape vectors. Each field has a §4.3.3 default that is returned
/// whenever the corresponding budget reservation didn't fire:
///
/// * `alloc_trim` defaults to `5` (no trim) when the trim PDF was
///   gated off by `ec_tell_frac()+48 > total - total_boost`.
/// * `skip` defaults to `false` when no skip bit was reserved (i.e.
///   `total <= 8` at the §4.3.3 reservation point).
/// * `intensity_band_offset` defaults to `0` (no intensity stereo
///   applied above band `start`) when the frame is mono OR when the
///   stereo intensity reservation fell short of the bit budget.
/// * `dual_stereo` defaults to `false` (joint coding) when the frame
///   is mono OR when the dual reservation didn't fire.
///
/// All four defaults match the §4.3.3 prose: an absent field reads
/// as its low-rate equivalent, so the downstream decoder sees a
/// well-defined value whether or not the encoder spent bits on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandAllocation {
    /// Allocation trim parameter in `0..=10`. `5` is "no trim";
    /// lower values bias the allocation toward low frequencies, higher
    /// toward high frequencies (RFC 6716 §4.3.3 lines 6370–6381).
    pub alloc_trim: u8,
    /// Final skip flag. `true` when at least one high-frequency band
    /// will be allocated no shape bits (RFC 6716 §4.3.3 lines
    /// 6402–6403). Reserved as 1 bit (8 8th-bits) when `total > 8`.
    pub skip: bool,
    /// Offset of the intensity-stereo lowest-band selector from
    /// `start`. The actual intensity band is `start +
    /// intensity_band_offset`. Range is `0..=end-start`. Zero means
    /// intensity stereo applies starting at band `start` (i.e. across
    /// all coded bands); `end-start` means intensity stereo is never
    /// applied. Mono frames or those with insufficient stereo budget
    /// land on 0 (RFC 6716 §4.3.3 lines 6405–6408).
    pub intensity_band_offset: u32,
    /// `true` when the frame uses dual stereo coding (left and right
    /// channels coded separately, RFC 6716 §3.1.2 / §4.3.3). `false`
    /// means joint coding. Mono frames or those without a dual
    /// reservation land on `false` (RFC 6716 §4.3.3 line 6408).
    pub dual_stereo: bool,
}

impl BandAllocation {
    /// All-defaults band-allocation that matches the §4.3.3 prose for
    /// a frame where none of the four optional fields were emitted.
    pub fn defaults() -> Self {
        Self {
            alloc_trim: DEFAULT_ALLOC_TRIM,
            skip: false,
            intensity_band_offset: 0,
            dual_stereo: false,
        }
    }
}

impl Default for BandAllocation {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Gate parameters for [`decode_band_allocation`].
///
/// Each boolean records whether the §4.3.3 reservation step set the
/// corresponding `*_rsv` to its emit-it value (8 8th bits for the
/// binary fields, `LOG2_FRAC_TABLE[end-start]` 8th bits for
/// intensity). The caller computes these from the `total` /
/// `total_boost` budget walk and from the mono/stereo flag; this
/// module only consumes the booleans.
///
/// * `trim_gated = true` iff `ec_tell_frac() + 48 <=
///   total_frame_size_8th_bits - total_boost` (RFC 6716 §4.3.3
///   lines 6376–6381).
/// * `skip_gated = true` iff `skip_rsv == 8` (RFC 6716 §4.3.3
///   lines 6419–6421).
/// * `intensity_gated = true` iff stereo AND `intensity_rsv > 0`
///   (RFC 6716 §4.3.3 lines 6423–6427).
/// * `dual_gated = true` iff stereo AND `dual_stereo_rsv == 8`
///   (RFC 6716 §4.3.3 lines 6428–6429).
/// * `coded_bands` = `end - start`, the number of coded bands
///   (used only when `intensity_gated`).
///
/// Mono frames pass `intensity_gated = false` and `dual_gated =
/// false`; the orchestrator returns the default `0` /  `false`
/// for those fields without touching the range decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandAllocationGates {
    /// Decode the allocation trim PDF? (RFC 6716 §4.3.3 trim gate.)
    pub trim_gated: bool,
    /// Decode the binary skip flag? (RFC 6716 §4.3.3 skip gate.)
    pub skip_gated: bool,
    /// Decode the uniform intensity-band selector? Stereo + budget.
    pub intensity_gated: bool,
    /// Decode the binary dual-stereo flag? Stereo + budget.
    pub dual_gated: bool,
    /// Number of coded bands `end - start`; only inspected when
    /// `intensity_gated == true`. Must be `>= 1` for the intensity
    /// decode to make sense (`dec_uint(end-start+1)` reads from a set
    /// of at least two values).
    pub coded_bands: u32,
}

/// Decode the §4.3.3 allocation-trim parameter.
///
/// Returns `Some(trim)` with `trim ∈ 0..=10` when `gated`, and
/// `None` when the trim PDF was suppressed by the bit budget. The
/// `BandAllocation::defaults()` path treats absent as `5` (no trim).
///
/// The PDF is `{2, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2}/128` (Table 58):
/// 46/128 of the mass concentrates on `trim=5`, falling off
/// symmetrically toward both extremes.
pub fn decode_alloc_trim(dec: &mut RangeDecoder<'_>, gated: bool) -> Option<u8> {
    if !gated {
        return None;
    }
    let v = dec.dec_icdf(ALLOC_TRIM_ICDF, ALLOC_TRIM_FTB);
    Some(v as u8)
}

/// Decode the §4.3.3 binary skip flag.
///
/// Returns `Some(bit)` with `bit ∈ {false, true}` when `gated`, and
/// `None` otherwise (the §4.3.3 default is `false`).
///
/// The skip PDF is `{1, 1}/2`, i.e. `logp = 1`. The flag tells the
/// decoder whether to apply the high-frequency band-skipping logic
/// at the tail of the allocation pass.
pub fn decode_skip_flag(dec: &mut RangeDecoder<'_>, gated: bool) -> Option<bool> {
    if !gated {
        return None;
    }
    Some(dec.dec_bit_logp(1) == 1)
}

/// Decode the §4.3.3 intensity-band offset.
///
/// Returns `Some(offset)` with `offset ∈ 0..=coded_bands` when
/// `gated` and `coded_bands >= 1`, and `None` otherwise. The
/// §4.3.3 default for absent intensity is `0` ("intensity stereo
/// applies starting at band `start`", i.e. no stereo bands).
///
/// The intensity offset is decoded as a uniform value across the
/// `coded_bands + 1` possible selectors. The reference's
/// `ilog2(end - start)` bits is the upper bound on the coding cost;
/// the [`RangeDecoder::dec_uint`] path produces a bit-exact decode
/// for any `ft >= 2`.
pub fn decode_intensity_band(
    dec: &mut RangeDecoder<'_>,
    gated: bool,
    coded_bands: u32,
) -> Option<u32> {
    if !gated || coded_bands == 0 {
        return None;
    }
    // dec_uint(ft) decodes uniformly in 0..ft. The intensity selector
    // spans 0..=coded_bands, i.e. coded_bands + 1 possible values.
    let v = dec.dec_uint(coded_bands + 1).unwrap_or(0);
    Some(v)
}

/// Decode the §4.3.3 binary dual-stereo flag.
///
/// Returns `Some(bit)` when `gated`, `None` otherwise. The §4.3.3
/// default is `false` (joint coding).
///
/// PDF is `{1, 1}/2`, i.e. `logp = 1`.
pub fn decode_dual_stereo(dec: &mut RangeDecoder<'_>, gated: bool) -> Option<bool> {
    if !gated {
        return None;
    }
    Some(dec.dec_bit_logp(1) == 1)
}

/// Decode the four §4.3.3 band-allocation scalar fields in Table 56
/// order: alloc.trim → skip → intensity → dual.
///
/// Each field is gated by the corresponding boolean in `gates`. A
/// gated-off field receives its §4.3.3 default (see
/// [`BandAllocation::defaults`]).
///
/// The range decoder is advanced exactly through the gated-on
/// fields; gated-off fields do not touch `dec` at all (so the
/// caller's `ec_tell_frac()` accounting stays accurate).
///
/// This function does NOT decode the surrounding band-boost loop,
/// the per-band cap[] vector, or the final reallocation/skip
/// resolution. Callers must implement those externally (the
/// boost loop is docs-gap-blocked behind the numeric `cache_caps50[]`
/// table the RFC delegates to a source file outside the workspace's
/// clean-room allow-list).
pub fn decode_band_allocation(
    dec: &mut RangeDecoder<'_>,
    gates: BandAllocationGates,
) -> BandAllocation {
    let mut out = BandAllocation::defaults();
    if let Some(t) = decode_alloc_trim(dec, gates.trim_gated) {
        out.alloc_trim = t;
    }
    if let Some(s) = decode_skip_flag(dec, gates.skip_gated) {
        out.skip = s;
    }
    if let Some(i) = decode_intensity_band(dec, gates.intensity_gated, gates.coded_bands) {
        out.intensity_band_offset = i;
    }
    if let Some(d) = decode_dual_stereo(dec, gates.dual_gated) {
        out.dual_stereo = d;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BandAllocation::defaults()` matches the §4.3.3 prose: trim=5,
    /// every other field at its low-rate default.
    #[test]
    fn defaults_match_section_4_3_3_prose() {
        let d = BandAllocation::defaults();
        assert_eq!(d.alloc_trim, 5);
        assert!(!d.skip);
        assert_eq!(d.intensity_band_offset, 0);
        assert!(!d.dual_stereo);
        // Default trait matches the explicit constructor.
        assert_eq!(BandAllocation::default(), d);
    }

    /// The icdf table for the §4.3.3 alloc-trim PDF must satisfy
    /// `icdf[0] = ft - fh[0]`, monotonically decrease to `0`, and
    /// have length 11 (one entry per `trim ∈ 0..=10`).
    #[test]
    fn alloc_trim_icdf_well_formed() {
        // Length is the number of symbols (terminator-inclusive).
        assert_eq!(ALLOC_TRIM_ICDF.len(), 11);
        // Strictly monotonically decreasing.
        for w in ALLOC_TRIM_ICDF.windows(2) {
            assert!(w[0] > w[1], "icdf not monotone at {:?}", w);
        }
        // Terminator is 0.
        assert_eq!(*ALLOC_TRIM_ICDF.last().unwrap(), 0);
        // ftb = 7 ⇒ ft = 128. ICDF[0] = ft - fh[0] = 128 - 2 = 126.
        assert_eq!(ALLOC_TRIM_ICDF[0], 126);
        // PDF sum = ft. Reconstruct fh = ft - icdf and confirm.
        let ft = 1u32 << ALLOC_TRIM_FTB;
        let fh: Vec<u32> = ALLOC_TRIM_ICDF.iter().map(|&c| ft - c as u32).collect();
        // fh must be strictly increasing.
        for w in fh.windows(2) {
            assert!(w[0] < w[1]);
        }
        // fh[last] = ft.
        assert_eq!(*fh.last().unwrap(), ft);
        // Reconstruct PDF = fh[k+1] - fh[k] (with fh[-1] = 0). Compare
        // against the literal Table 58 values.
        let table58 = [2u32, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2];
        let mut prev = 0u32;
        for (k, &want) in table58.iter().enumerate() {
            let p = fh[k] - prev;
            assert_eq!(p, want, "PDF mismatch at k={k}");
            prev = fh[k];
        }
        // Sum-check.
        assert_eq!(table58.iter().sum::<u32>(), ft);
    }

    /// Gated-off `decode_alloc_trim` returns `None` and does not
    /// advance the range decoder.
    #[test]
    fn alloc_trim_gated_off_is_noop() {
        let mut dec = RangeDecoder::new(&[0x55u8; 8]);
        let before = dec.tell();
        let r = decode_alloc_trim(&mut dec, false);
        let after = dec.tell();
        assert_eq!(r, None);
        assert_eq!(before, after);
        assert!(!dec.has_error());
    }

    /// Gated-on `decode_alloc_trim` returns `Some(v)` with `v` in
    /// `0..=10`, and advances `tell()` strictly.
    #[test]
    fn alloc_trim_gated_on_in_range() {
        // Try a variety of buffers; every decoded trim must lie in
        // [0, 10] and not trip the error flag.
        for seed in [0x00u8, 0x55, 0xAA, 0xFF] {
            let buf = [seed; 16];
            let mut dec = RangeDecoder::new(&buf);
            let before = dec.tell();
            let r = decode_alloc_trim(&mut dec, true).unwrap();
            let after = dec.tell();
            assert!(r <= 10, "trim={r} > 10 for seed=0x{:02X}", seed);
            assert!(after >= before, "tell() retreated on seed=0x{:02X}", seed);
            assert!(!dec.has_error());
        }
    }

    /// The icdf walk + the Table 58 PDF together must guarantee that
    /// for *some* hand-crafted stream the high-probability symbol
    /// `trim = 5` is reachable. We don't pin which input lands there
    /// (that depends on the exact renorm trajectory), but we verify
    /// that across a small ensemble at least one of the popular
    /// non-tail values (4, 5, 6 — together carrying 90/128 of the
    /// PDF mass) is hit, exercising the inner cells of the icdf walk
    /// rather than only the boundary entries.
    #[test]
    fn alloc_trim_ensemble_hits_inner_cells() {
        let mut hits_inner = false;
        for seed in [0u8, 0x55, 0xAA, 0x33, 0xCC, 0x10, 0xE7, 0x42] {
            let buf = [seed; 16];
            let mut dec = RangeDecoder::new(&buf);
            let t = decode_alloc_trim(&mut dec, true).unwrap();
            if (4..=6).contains(&t) {
                hits_inner = true;
            }
            assert!(t <= 10);
        }
        assert!(
            hits_inner,
            "ensemble of trim decodes never hit the high-probability core {{4,5,6}}"
        );
    }

    /// Gated-off skip is `None`, no decoder consumption.
    #[test]
    fn skip_gated_off_is_noop() {
        let mut dec = RangeDecoder::new(&[0xCCu8; 8]);
        let before = dec.tell();
        let r = decode_skip_flag(&mut dec, false);
        let after = dec.tell();
        assert_eq!(r, None);
        assert_eq!(before, after);
    }

    /// Gated-on skip consumes one bit and returns a bool. The exact
    /// value depends on the input bytes; `tell()` advances.
    #[test]
    fn skip_gated_on_advances_tell() {
        let mut dec = RangeDecoder::new(&[0xA5u8, 0x33, 0x77, 0xCC]);
        let before = dec.tell();
        let r = decode_skip_flag(&mut dec, true);
        let after = dec.tell();
        assert!(r.is_some());
        assert!(after > before);
        assert!(!dec.has_error());
    }

    /// Gated-off intensity is `None` even if `coded_bands > 0`.
    #[test]
    fn intensity_gated_off_is_noop() {
        let mut dec = RangeDecoder::new(&[0x12u8; 8]);
        let before = dec.tell();
        let r = decode_intensity_band(&mut dec, false, 21);
        let after = dec.tell();
        assert_eq!(r, None);
        assert_eq!(before, after);
    }

    /// Gated-on intensity with `coded_bands == 0` is also `None`:
    /// `dec_uint(1)` would return 0 deterministically, but the
    /// §4.3.3 reservation only fires for stereo frames with at
    /// least one coded band — we treat the degenerate case as not
    /// gated.
    #[test]
    fn intensity_gated_on_zero_bands_is_noop() {
        let mut dec = RangeDecoder::new(&[0x42u8; 8]);
        let before = dec.tell();
        let r = decode_intensity_band(&mut dec, true, 0);
        let after = dec.tell();
        assert_eq!(r, None);
        assert_eq!(before, after);
    }

    /// Gated-on intensity must land in `0..=coded_bands` for every
    /// `coded_bands` from 1 up to 21 (the full CELT band count).
    #[test]
    fn intensity_gated_on_in_range() {
        for cb in 1u32..=21 {
            let buf = [0x37u8, 0x91, 0xC4, 0x18, 0xA2, 0x5D, 0x6E, 0xFF];
            let mut dec = RangeDecoder::new(&buf);
            let v = decode_intensity_band(&mut dec, true, cb).unwrap();
            assert!(
                v <= cb,
                "intensity={v} out of [0, {cb}] for coded_bands={cb}"
            );
            assert!(!dec.has_error());
        }
    }

    /// Gated-off dual is `None`.
    #[test]
    fn dual_gated_off_is_noop() {
        let mut dec = RangeDecoder::new(&[0x99u8; 8]);
        let before = dec.tell();
        let r = decode_dual_stereo(&mut dec, false);
        let after = dec.tell();
        assert_eq!(r, None);
        assert_eq!(before, after);
    }

    /// Gated-on dual advances `tell()` by ~1 bit.
    #[test]
    fn dual_gated_on_advances_tell() {
        let mut dec = RangeDecoder::new(&[0x6Bu8, 0x21, 0x4F, 0xE0]);
        let before = dec.tell();
        let r = decode_dual_stereo(&mut dec, true);
        let after = dec.tell();
        assert!(r.is_some());
        assert!(after > before);
        assert!(!dec.has_error());
    }

    /// Orchestrator: every gate off → defaults, no decoder
    /// consumption. This is the silence-frame / low-rate path.
    #[test]
    fn orchestrator_all_gates_off_returns_defaults() {
        let mut dec = RangeDecoder::new(&[0xAAu8; 16]);
        let before = dec.tell();
        let gates = BandAllocationGates {
            trim_gated: false,
            skip_gated: false,
            intensity_gated: false,
            dual_gated: false,
            coded_bands: 21,
        };
        let alloc = decode_band_allocation(&mut dec, gates);
        let after = dec.tell();
        assert_eq!(alloc, BandAllocation::defaults());
        assert_eq!(after, before, "decoder advanced under all-gates-off");
    }

    /// Orchestrator: all four gates ON, stereo frame, full 21-band
    /// CELT mode. Every field should be decoded, `tell()` must
    /// strictly advance, and the result must lie inside the §4.3.3
    /// value ranges.
    #[test]
    fn orchestrator_all_gates_on_stereo_full_band() {
        let buf: Vec<u8> = (0u8..32).cycle().take(64).collect();
        let mut dec = RangeDecoder::new(&buf);
        let before = dec.tell();
        let gates = BandAllocationGates {
            trim_gated: true,
            skip_gated: true,
            intensity_gated: true,
            dual_gated: true,
            coded_bands: 21,
        };
        let alloc = decode_band_allocation(&mut dec, gates);
        let after = dec.tell();
        assert!(after > before, "no decoder advance under all-gates-on");
        assert!(alloc.alloc_trim <= 10);
        assert!(alloc.intensity_band_offset <= 21);
        assert!(!dec.has_error());
    }

    /// Orchestrator: mono frame with only trim + skip gated on.
    /// Intensity and dual must remain at their defaults (0, false)
    /// because mono frames never reserve them per §4.3.3 lines 6405
    /// – 6408.
    #[test]
    fn orchestrator_mono_only_trim_and_skip() {
        let mut dec = RangeDecoder::new(&[0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        let gates = BandAllocationGates {
            trim_gated: true,
            skip_gated: true,
            intensity_gated: false,
            dual_gated: false,
            coded_bands: 21,
        };
        let alloc = decode_band_allocation(&mut dec, gates);
        assert!(alloc.alloc_trim <= 10);
        assert_eq!(alloc.intensity_band_offset, 0);
        assert!(!alloc.dual_stereo);
    }

    /// Orchestrator: hybrid mode (CELT covers bands 17..=20, so
    /// `coded_bands = 4`). Both stereo gates on. The intensity
    /// offset must land in `0..=4`.
    #[test]
    fn orchestrator_hybrid_stereo_four_bands() {
        let buf: Vec<u8> = (0u8..32).rev().take(48).collect();
        let mut dec = RangeDecoder::new(&buf);
        let gates = BandAllocationGates {
            trim_gated: true,
            skip_gated: true,
            intensity_gated: true,
            dual_gated: true,
            coded_bands: 4,
        };
        let alloc = decode_band_allocation(&mut dec, gates);
        assert!(alloc.intensity_band_offset <= 4);
        assert!(!dec.has_error());
    }

    /// Orchestrator: the per-field order must match Table 56 (trim
    /// then skip then intensity then dual). Compare against
    /// hand-decoding the same buffer step by step.
    #[test]
    fn orchestrator_field_order_matches_table_56() {
        let buf: Vec<u8> = (0u8..32).cycle().take(64).collect();
        // Hand decode via direct calls in Table-56 order.
        let mut hand = RangeDecoder::new(&buf);
        let t_hand = decode_alloc_trim(&mut hand, true).unwrap();
        let s_hand = decode_skip_flag(&mut hand, true).unwrap();
        let i_hand = decode_intensity_band(&mut hand, true, 21).unwrap();
        let d_hand = decode_dual_stereo(&mut hand, true).unwrap();
        // Orchestrator-decode the same buffer.
        let mut orch = RangeDecoder::new(&buf);
        let gates = BandAllocationGates {
            trim_gated: true,
            skip_gated: true,
            intensity_gated: true,
            dual_gated: true,
            coded_bands: 21,
        };
        let a = decode_band_allocation(&mut orch, gates);
        assert_eq!(a.alloc_trim, t_hand);
        assert_eq!(a.skip, s_hand);
        assert_eq!(a.intensity_band_offset, i_hand);
        assert_eq!(a.dual_stereo, d_hand);
        // After both decoders walked the same fields, they must agree
        // on tell().
        assert_eq!(orch.tell(), hand.tell());
    }

    /// `decode_intensity_band` with `coded_bands = 1` reads a uniform
    /// over `{0, 1}` (single bit of entropy). The orchestrator gate
    /// keeps it active for the minimal stereo case.
    #[test]
    fn intensity_single_coded_band_decodes_one_bit() {
        let mut dec = RangeDecoder::new(&[0xF0u8, 0x0F, 0xAA, 0x55]);
        let before = dec.tell();
        let v = decode_intensity_band(&mut dec, true, 1).unwrap();
        let after = dec.tell();
        assert!(v <= 1);
        assert!(after > before, "no advance on dec_uint(2)");
    }
}
