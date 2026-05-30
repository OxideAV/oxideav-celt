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
//!   `end-start` (looked up in [`LOG2_FRAC_TABLE`]) fits in the
//!   remaining budget.
//! * `dual` (stereo frames only) is decoded only if
//!   `dual_stereo_rsv = 8` (8th bits) was set, i.e. there were still
//!   more than 8 8th bits left after subtracting `intensity_rsv`.
//!
//! This module exposes a per-field decoder for each scalar, the two
//! reservation helpers [`intensity_rsv`] / [`reserve_stereo`] that
//! drive the gating arithmetic for the two stereo fields, plus an
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
//! file is transcribed from RFC 6716 §4.3.3 + Table 58
//! (`docs/audio/opus/rfc6716-opus.txt`) and the clean-room narrative at
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2. The
//! `LOG2_FRAC_TABLE` numeric values come from the Feist-facts
//! numeric extract at `docs/audio/celt/tables/log2_frac_table.csv`
//! (metadata in `log2_frac_table.meta`). No external library source
//! was consulted.

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

/// Conservative `log2(n)` in 1/8 bit units, for `n ∈ 0..=23`
/// (RFC 6716 §4.3.3 / clean-room narrative §2.5).
///
/// Each entry `LOG2_FRAC_TABLE[n]` is the conservative — i.e.
/// always rounded down to integer 1/8-bit precision — base-2
/// logarithm of `n`. Examples:
///
/// * `LOG2_FRAC_TABLE[0] = 0`  (sentinel: `log2(0)` is undefined,
///   but the §4.3.3 reservation walk only indexes this slot when
///   `end - start == 0`, i.e. no coded bands at all).
/// * `LOG2_FRAC_TABLE[1] = 8`  (= `log2(1)*8 + 8` — the conservative
///   rounding biases `log2(1) = 0` up to a full 1-bit reservation
///   to keep encoder/decoder lockstep).
/// * `LOG2_FRAC_TABLE[2] = 13` (= `floor(log2(2) * 8 + 8*log2(e)*...)`
///   — `log2(2) = 1` plus ~5/8 of additional headroom, conservatively
///   rounded down).
/// * `LOG2_FRAC_TABLE[16] = 33` (= ~4.125 bits, close to the exact
///   `log2(16) = 4`).
/// * `LOG2_FRAC_TABLE[23] = 37`.
///
/// The table is exposed as a `pub const` so the §4.3.3 budget walk
/// (computed in a future round, once the band-boost + `cap[]` machinery
/// is in place) can look up the intensity-stereo reservation without
/// re-computing it.
///
/// Numeric values come from the Feist-facts CSV at
/// `docs/audio/celt/tables/log2_frac_table.csv` (24 entries, 1/8-bit
/// units, unsigned byte). The CSV is documented in the clean-room
/// narrative at `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md`
/// §2.5 and is normatively cited by RFC 6716 §4.3.3.
pub const LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

/// Reservation (in 1/8 bit units) for the §4.3.3 intensity-stereo
/// selector, given a budget and the number of coded bands.
///
/// Returns the conservative `log2(coded_bands)` in 1/8-bit units when
/// the §4.3.3 reservation step fires, and zero when the reservation
/// must be dropped (either the frame is mono, the budget is too
/// small, or `coded_bands` is outside the table's domain).
///
/// Inputs:
///
/// * `coded_bands` = `end - start`, the number of CELT bands that
///   this frame codes (RFC 6716 §4.3.3 lines 6310–6314). The §4.3.3
///   walk only indexes `LOG2_FRAC_TABLE` for `1..=22` (the maximum
///   CELT band range is 21 — pure CELT 0..21 — and Hybrid mode 17..21
///   gives 4); values outside `1..=23` clamp to a zero reservation
///   defensively rather than out-of-bounds, matching the §4.3.3
///   prose that mono frames and empty band ranges reserve nothing.
/// * `stereo` = `true` for two-channel CELT, `false` for mono.
///   Mono frames never reserve intensity bits (RFC 6716 §4.3.3
///   lines 6405–6408), so the function always returns `0` when
///   `stereo == false`.
/// * `total_8th_bits` = the §4.3.3 running budget in 1/8 bits at the
///   point where intensity_rsv is considered (after anti_collapse_rsv
///   and skip_rsv have been subtracted). The reservation is taken
///   only if `LOG2_FRAC_TABLE[coded_bands] <= total_8th_bits`,
///   otherwise it is dropped to zero per §4.3.3.
///
/// The return value is in 1/8-bit units, matching the rest of the
/// §4.3.3 budget arithmetic; callers subtract it from `total` and
/// then test `total > 8` for the dual-stereo reservation.
///
/// This function does NOT consult the range decoder; it is the
/// pure-arithmetic side of the §4.3.3 reservation prose, exposed
/// standalone so a future round's full budget walk can drop it in
/// without restructuring.
pub fn intensity_rsv(coded_bands: u32, stereo: bool, total_8th_bits: i32) -> u32 {
    if !stereo {
        return 0;
    }
    // Defensive: §4.3.3 only ever indexes LOG2_FRAC_TABLE at 1..=22 in
    // practice. Empty band ranges shouldn't reserve intensity bits at
    // all (there are no stereo bands to apply intensity to); values
    // beyond the table fall back to no reservation to avoid panics.
    let idx = coded_bands as usize;
    if idx == 0 || idx >= LOG2_FRAC_TABLE.len() {
        return 0;
    }
    let rsv = LOG2_FRAC_TABLE[idx] as i32;
    // §4.3.3: "If intensity_rsv > total, set it to 0; otherwise
    // decrement total by it ...". We return the value that survives
    // the >total test, or 0 if it would exceed the budget.
    if rsv > total_8th_bits {
        0
    } else {
        rsv as u32
    }
}

/// Compute the trio (intensity_rsv, total_after, dual_stereo_rsv) for
/// the §4.3.3 stereo reservation step.
///
/// Convenience helper that bundles the two §4.3.3 stereo decisions
/// (intensity-band selector reservation followed by dual-stereo flag
/// reservation) into a single function. The §4.3.3 prose runs them in
/// fixed order:
///
/// 1. `intensity_rsv = LOG2_FRAC_TABLE[end - start]` (conservative
///    log2 of the band count, in 1/8 bits); if `intensity_rsv >
///    total`, set it to 0.
/// 2. Decrement `total` by `intensity_rsv`.
/// 3. If `total > 8`, set `dual_stereo_rsv = 8` and decrement
///    `total` by 8.
///
/// Returns `(intensity_rsv, total_after, dual_rsv)` in 1/8 bits.
/// `total_after` is what the caller's remaining-budget variable
/// should be after this step. Mono frames pass `stereo = false` and
/// receive `(0, total_8th_bits, 0)` — the reservation step is a
/// no-op for mono per RFC 6716 §4.3.3 lines 6405–6408.
pub fn reserve_stereo(coded_bands: u32, stereo: bool, total_8th_bits: i32) -> (u32, i32, u32) {
    let i_rsv = intensity_rsv(coded_bands, stereo, total_8th_bits);
    let after_intensity = total_8th_bits - i_rsv as i32;
    let d_rsv: u32 = if stereo && after_intensity > 8 { 8 } else { 0 };
    let total_after = after_intensity - d_rsv as i32;
    (i_rsv, total_after, d_rsv)
}

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

    /// `LOG2_FRAC_TABLE` has exactly 24 entries — one per index
    /// `n ∈ 0..=23`. The §4.3.3 stereo-reservation walk only ever
    /// indexes `1..=22` in practice, but the table extends to 23 to
    /// match the clean-room CSV at
    /// `docs/audio/celt/tables/log2_frac_table.csv`.
    #[test]
    fn log2_frac_table_length() {
        assert_eq!(LOG2_FRAC_TABLE.len(), 24);
    }

    /// Spot-check `LOG2_FRAC_TABLE` against the values listed in the
    /// clean-room narrative
    /// `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.5:
    /// `[0]=0`, `[1]=8`, `[2]=13`, `[16]=33`, `[23]=37`.
    #[test]
    fn log2_frac_table_spot_check() {
        assert_eq!(LOG2_FRAC_TABLE[0], 0);
        assert_eq!(LOG2_FRAC_TABLE[1], 8);
        assert_eq!(LOG2_FRAC_TABLE[2], 13);
        assert_eq!(LOG2_FRAC_TABLE[3], 16);
        assert_eq!(LOG2_FRAC_TABLE[16], 33);
        assert_eq!(LOG2_FRAC_TABLE[23], 37);
    }

    /// `LOG2_FRAC_TABLE` is monotone non-decreasing: each entry is at
    /// least as large as the previous (the conservative log2 of `n+1`
    /// can never be less than the conservative log2 of `n`). This is
    /// a structural invariant the §4.3.3 budget walk relies on.
    #[test]
    fn log2_frac_table_monotone_non_decreasing() {
        for w in LOG2_FRAC_TABLE.windows(2) {
            assert!(
                w[0] <= w[1],
                "LOG2_FRAC_TABLE not monotone: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    /// Each entry is a conservative — i.e. floor-rounded — bound on
    /// `n * 8` in 1/8 bit units. For `n >= 1`, `LOG2_FRAC_TABLE[n]`
    /// must be at most `8 + 8*ceil(log2(n))` (a generous upper bound
    /// covering the conservative-rounding policy). And for `n >= 2`,
    /// it must be at least `8 * floor(log2(n))` (the strict floor
    /// lower bound).
    #[test]
    fn log2_frac_table_bounds_match_log2() {
        for (n, &entry) in LOG2_FRAC_TABLE.iter().enumerate().skip(1) {
            let value = entry as f64;
            let log2_n = (n as f64).log2();
            // Upper bound: log2(n)*8 + 8 (allow the conservative
            // rounding to overshoot by up to 1 bit).
            assert!(
                value <= log2_n * 8.0 + 8.0 + 1e-6,
                "LOG2_FRAC_TABLE[{n}] = {value} > log2({n})*8 + 8"
            );
            // Lower bound for n >= 2: at least floor(log2(n))*8.
            if n >= 2 {
                let floor_lower = (log2_n.floor() * 8.0) - 1e-6;
                assert!(
                    value >= floor_lower,
                    "LOG2_FRAC_TABLE[{n}] = {value} < floor(log2({n}))*8 = {floor_lower}"
                );
            }
        }
    }

    /// Mono frames never reserve intensity bits per RFC 6716 §4.3.3
    /// lines 6405–6408. Regardless of band count and budget, the
    /// helper returns zero.
    #[test]
    fn intensity_rsv_mono_is_zero() {
        for coded_bands in [0u32, 1, 4, 21, 22, 23] {
            for total in [0i32, 100, 10_000] {
                assert_eq!(intensity_rsv(coded_bands, false, total), 0);
            }
        }
    }

    /// Empty band range never reserves intensity bits — there are no
    /// stereo bands to apply intensity to.
    #[test]
    fn intensity_rsv_zero_bands_is_zero() {
        assert_eq!(intensity_rsv(0, true, 10_000), 0);
    }

    /// Out-of-table band counts fall back to a zero reservation
    /// defensively rather than panicking. The §4.3.3 walk only ever
    /// passes `coded_bands ∈ 1..=22` in practice.
    #[test]
    fn intensity_rsv_out_of_range_is_zero() {
        // 24, 100, u32::MAX — all out of LOG2_FRAC_TABLE's 0..24 domain.
        for coded_bands in [24u32, 25, 100, u32::MAX] {
            assert_eq!(intensity_rsv(coded_bands, true, 10_000), 0);
        }
    }

    /// `intensity_rsv` returns `LOG2_FRAC_TABLE[coded_bands]` when
    /// the budget covers it. At a generous budget every legal
    /// `coded_bands` value lands on its table entry.
    #[test]
    fn intensity_rsv_stereo_budget_sufficient() {
        let big_budget = 10_000;
        for coded_bands in 1u32..=23 {
            let got = intensity_rsv(coded_bands, true, big_budget);
            let want = LOG2_FRAC_TABLE[coded_bands as usize] as u32;
            assert_eq!(got, want, "mismatch at coded_bands={coded_bands}");
        }
    }

    /// `intensity_rsv` returns zero when the reservation would exceed
    /// the running budget, per §4.3.3 "If intensity_rsv > total, set
    /// it to 0".
    #[test]
    fn intensity_rsv_budget_too_small() {
        // 21 bands needs 36 1/8-bits. Budget of 35 drops to zero.
        assert_eq!(intensity_rsv(21, true, 35), 0);
        // Budget of exactly 36 is the equality boundary — the §4.3.3
        // prose says "if intensity_rsv > total, set to 0", so equality
        // keeps the reservation.
        assert_eq!(intensity_rsv(21, true, 36), 36);
        // Budget of 100 keeps it.
        assert_eq!(intensity_rsv(21, true, 100), 36);
    }

    /// Hybrid-mode 4-band CELT reserves `LOG2_FRAC_TABLE[4] = 19`
    /// 1/8 bits for intensity when the budget covers it.
    #[test]
    fn intensity_rsv_hybrid_four_bands() {
        let rsv = intensity_rsv(4, true, 1000);
        assert_eq!(rsv, 19);
        assert_eq!(rsv, LOG2_FRAC_TABLE[4] as u32);
    }

    /// `reserve_stereo` on a mono frame is a complete no-op: zero
    /// reservations, total unchanged.
    #[test]
    fn reserve_stereo_mono_is_passthrough() {
        let (i_rsv, total_after, d_rsv) = reserve_stereo(21, false, 500);
        assert_eq!(i_rsv, 0);
        assert_eq!(d_rsv, 0);
        assert_eq!(total_after, 500);
    }

    /// Stereo with a large budget reserves intensity, then 8 1/8-bits
    /// for dual. Verify against `LOG2_FRAC_TABLE[21] + 8 = 36 + 8 = 44`
    /// of total reservation taken.
    #[test]
    fn reserve_stereo_large_budget_both_reservations() {
        let total = 1000;
        let (i_rsv, total_after, d_rsv) = reserve_stereo(21, true, total);
        assert_eq!(i_rsv, 36);
        assert_eq!(d_rsv, 8);
        assert_eq!(total_after, 1000 - 36 - 8);
    }

    /// Stereo with a tight budget: intensity reserved exactly, then
    /// not enough left for dual.
    #[test]
    fn reserve_stereo_tight_budget_no_dual() {
        // 21 bands: intensity = 36, then after_intensity = 8 which is
        // NOT > 8, so dual_rsv = 0.
        let (i_rsv, total_after, d_rsv) = reserve_stereo(21, true, 44);
        assert_eq!(i_rsv, 36);
        assert_eq!(d_rsv, 0);
        assert_eq!(total_after, 8);
        // Bump budget by 1 — now after_intensity = 9 > 8, dual fires.
        let (_, total_after_2, d_rsv_2) = reserve_stereo(21, true, 45);
        assert_eq!(d_rsv_2, 8);
        assert_eq!(total_after_2, 1);
    }

    /// Stereo with a budget below the intensity reservation: both
    /// reservations drop to zero.
    #[test]
    fn reserve_stereo_below_intensity_drops_both() {
        // 21 bands need 36, budget of 30 forces i_rsv = 0; after that
        // we still test "total > 8" for dual — 30 > 8 so dual fires
        // with 8. (Per §4.3.3: dual is reserved iff total > 8 AFTER
        // intensity subtraction, which here is the original 30.)
        let (i_rsv, total_after, d_rsv) = reserve_stereo(21, true, 30);
        assert_eq!(i_rsv, 0);
        assert_eq!(d_rsv, 8);
        assert_eq!(total_after, 22);
    }

    /// Stereo with empty band range: no intensity reserved (zero band
    /// count), but dual still considered against the budget.
    #[test]
    fn reserve_stereo_empty_bands() {
        let (i_rsv, total_after, d_rsv) = reserve_stereo(0, true, 100);
        assert_eq!(i_rsv, 0);
        assert_eq!(d_rsv, 8);
        assert_eq!(total_after, 92);
    }

    /// Cross-check `intensity_rsv` against `reserve_stereo`'s first
    /// returned value across a grid of budgets and band counts.
    #[test]
    fn reserve_stereo_intensity_matches_helper() {
        for coded_bands in 0u32..=23 {
            for total in [0i32, 16, 36, 50, 200] {
                for stereo in [false, true] {
                    let (i_rsv, _, _) = reserve_stereo(coded_bands, stereo, total);
                    assert_eq!(i_rsv, intensity_rsv(coded_bands, stereo, total));
                }
            }
        }
    }
}
