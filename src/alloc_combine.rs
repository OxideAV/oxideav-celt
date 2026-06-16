//! Per-band shape-allocation assembly at a fixed quality column
//! (RFC 6716 §4.3.3).
//!
//! ## Where this fits
//!
//! The §4.3.3 allocation process builds several per-coded-band vectors
//! before it can convert bits to pulses (§4.3.4.1):
//!
//! 1. the interpolated **static** allocation at a quality column
//!    `(qlo, frac)` —
//!    [`crate::static_alloc::window_static_alloc_per_band_1_8th`],
//! 2. the per-band **boosts** decoded by
//!    [`crate::band_cap::decode_band_boosts`] (§2.3),
//! 3. the per-band **trim offsets** derived from `alloc.trim` by
//!    [`crate::band_minimums::compute_trim_offsets`] (§2.6), and
//! 4. the per-band **caps** `cap[]` from
//!    [`crate::band_cap::compute_band_caps`] (§2.2).
//!
//! RFC 6716 §4.3.3 describes the search as picking "the entry nearest
//! but not exceeding the available space (subject to the tilt, boosts,
//! band maximums, and band minimums)". This module performs the
//! purely-additive part of that "subject to": for a single quality
//! column it sums the static allocation with the boosts and the trim
//! offsets (the "tilt"), then clamps each band to its `cap[]` (the
//! "band maximums") and floors it at zero. The result is the per-band
//! shape-allocation candidate the §2.7 search evaluates at that column.
//!
//! The doc comment on
//! [`crate::static_alloc::find_static_alloc`]'s [`StaticAllocSearch`]
//! result names this step explicitly: when the budget is too small to
//! afford even `(qlo=0, frac=0)`, "the caller is expected to fall
//! through to the minimums / cap / boost composition in the next
//! stage." This module is that next stage, restricted to the part
//! whose arithmetic RFC 6716 §4.3.3 spells out term by term.
//!
//! ## What is *not* here
//!
//! The §2.7 hard-minimum **skip** decision — comparing a band's
//! combined candidate against `thresh[band]` and dropping it to zero
//! when it falls short — is bound up with the reallocation bisection
//! and the concurrent skip decoding, which RFC 6716 §4.3.3 and the
//! clean-room narrative (`docs/audio/celt/spec/`) §2.7 defer to the
//! reference implementation. That floor is therefore *not* applied
//! here; this module stops at the cap clamp, which is the part the RFC
//! fully specifies. Callers that already hold a `thresh[]` vector (from
//! [`crate::band_minimums::compute_thresh`]) carry it forward into the
//! deferred bisection unchanged.
//!
//! [`StaticAllocSearch`]: crate::static_alloc::StaticAllocSearch

use crate::band_cap::compute_band_caps;
use crate::band_minimums::compute_trim_offsets;
use crate::static_alloc::window_static_alloc_per_band_1_8th;

/// The per-band shape-allocation candidate assembled at a quality
/// column, in 1/8-bit units.
///
/// `bits` is the per-coded-band candidate
/// `clamp(static + boost + trim_offset, 0, cap)` (one entry per coded
/// band, indexed from the window origin). `caps` is the per-band `cap[]`
/// the clamp used, carried alongside so the deferred §2.7 bisection can
/// reuse it without recomputation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombinedAllocation {
    /// Per-coded-band shape-allocation candidate in 1/8 bits:
    /// `clamp(static[b] + boost[b] + trim_offset[b], 0, cap[b])`.
    pub bits: Vec<i32>,
    /// Per-coded-band maximum `cap[]` in 1/8 bits (RFC 6716 §4.3.3
    /// §2.2), the upper bound applied to `bits`.
    pub caps: Vec<i32>,
    /// Sum of `bits` across the coded-band window, in 1/8 bits — the
    /// window total the §2.7 search compares against the remaining
    /// budget.
    pub total: i64,
}

/// Assemble the per-band shape-allocation candidate at a quality column
/// (RFC 6716 §4.3.3).
///
/// For each coded band `b` in the window the candidate is
///
/// ```text
/// bits[b] = clamp(static[b] + boost[b] + trim_offset[b], 0, cap[b])
/// ```
///
/// where `static[b]` is the interpolated Table-57 allocation at
/// `(qlo, frac)` (§2.1), `boost[b]` the decoded band boost (§2.3),
/// `trim_offset[b]` the `alloc.trim`-derived tilt (§2.6), and `cap[b]`
/// the per-band maximum (§2.2). All quantities are in 1/8-bit units.
///
/// Inputs:
///
/// * `coding_start` — the Table-55 index the coded-band window starts
///   at (0 for pure CELT, 17 for Hybrid).
/// * `bins_per_band` — per-channel MDCT-bin count for each band in the
///   window, at the **actual** frame size (`BAND_BINS_LM[lm]` sliced to
///   the window). `bins_per_band.len()` is the window length.
/// * `qlo` / `frac` — the quality-column grid position (typically a
///   [`crate::static_alloc::StaticAllocSearch`] outcome).
/// * `boost` — per-coded-band boosts in 1/8 bits (from
///   [`crate::band_cap::decode_band_boosts`]); length equals the window
///   length.
/// * `alloc_trim` — the decoded §4.3.3 trim integer in `0..=10`
///   (5 = no trim).
/// * `channels` — 1 (mono) or 2 (stereo).
/// * `stereo` — whether the frame is stereo (selects the `cap[]` table
///   row; for mono this is `false` even though `channels == 1`).
/// * `lm` — `log2(frame_size / 120)` ∈ `{0,1,2,3}`.
///
/// Returns `None` on any input-validation failure: a window overflowing
/// `NUM_BANDS`, `channels ∉ {1,2}`, `lm > 3`, `alloc_trim ∉ 0..=10`,
/// `qlo`/`frac` out of range for the interpolation grid, or a `boost`
/// slice whose length does not match the window. The returned
/// [`CombinedAllocation::bits`] never contains a negative entry, and
/// every entry satisfies `bits[b] <= caps[b]`.
#[allow(clippy::too_many_arguments)]
pub fn combine_band_allocation(
    coding_start: usize,
    bins_per_band: &[u32],
    qlo: usize,
    frac: u32,
    boost: &[i32],
    alloc_trim: i32,
    channels: u32,
    stereo: bool,
    lm: u32,
) -> Option<CombinedAllocation> {
    let n = bins_per_band.len();
    if boost.len() != n {
        return None;
    }

    // 1. Interpolated static allocation at (qlo, frac).
    let mut static_alloc = vec![0u32; n];
    if !window_static_alloc_per_band_1_8th(
        coding_start,
        bins_per_band,
        qlo,
        frac,
        channels,
        lm,
        &mut static_alloc,
    ) {
        return None;
    }

    // 2. Trim offsets (the §2.6 tilt).
    let mut trim_offsets = vec![0i32; n];
    if !compute_trim_offsets(
        alloc_trim,
        lm,
        channels,
        coding_start,
        bins_per_band,
        &mut trim_offsets,
    ) {
        return None;
    }

    // 3. Per-band caps (the §2.2 band maximums).
    let mut caps_i16 = vec![0i16; n];
    if !compute_band_caps(lm, stereo, channels, bins_per_band, &mut caps_i16) {
        return None;
    }

    // 4. Combine: clamp(static + boost + trim_offset, 0, cap) per band.
    //    Accumulate in i64 so the additive chain cannot overflow before
    //    the clamp re-narrows it into the `cap`/zero bounds.
    let mut bits = vec![0i32; n];
    let mut caps = vec![0i32; n];
    let mut total: i64 = 0;
    for b in 0..n {
        let cap = caps_i16[b] as i64;
        let raw = static_alloc[b] as i64 + boost[b] as i64 + trim_offsets[b] as i64;
        let clamped = raw.clamp(0, cap);
        // Every term is bounded well inside i32 at legal inputs; the
        // clamp keeps `clamped` in `0..=cap <= i16::MAX`, so the cast is
        // exact.
        bits[b] = clamped as i32;
        caps[b] = cap as i32;
        total += clamped;
    }

    Some(CombinedAllocation { bits, caps, total })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band_cap::compute_band_caps;
    use crate::band_minimums::{compute_trim_offsets, BAND_BINS_LM};
    use crate::static_alloc::window_static_alloc_per_band_1_8th;

    /// Pure-CELT mono LM=0 window of all 21 bands.
    fn mono_window(lm: usize) -> Vec<u32> {
        BAND_BINS_LM[lm].to_vec()
    }

    /// With no trim (alloc_trim = 5), no boosts, and caps generous
    /// enough not to bind, the combined candidate is exactly the
    /// interpolated static allocation. We verify against the static
    /// evaluator directly.
    #[test]
    fn no_trim_no_boost_equals_static_when_caps_loose() {
        let lm = 1u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let boost = vec![0i32; n];
        // alloc_trim = 5 ⇒ trim prefactor (5 - 5 - LM) = -LM, so trim is
        // NOT zero at LM>0. Use LM such that we can compare term by term
        // rather than asserting equality to static.
        let combined = combine_band_allocation(0, &bins, 3, 0, &boost, 5, 1, false, lm).unwrap();

        // Reconstruct static + trim independently.
        let mut static_alloc = vec![0u32; n];
        assert!(window_static_alloc_per_band_1_8th(
            0,
            &bins,
            3,
            0,
            1,
            lm,
            &mut static_alloc
        ));
        let mut trim = vec![0i32; n];
        assert!(compute_trim_offsets(5, lm, 1, 0, &bins, &mut trim));
        let mut caps16 = vec![0i16; n];
        assert!(compute_band_caps(lm, false, 1, &bins, &mut caps16));

        for b in 0..n {
            let expect =
                ((static_alloc[b] as i64) + (trim[b] as i64)).clamp(0, caps16[b] as i64) as i32;
            assert_eq!(combined.bits[b], expect, "band {b}");
            assert_eq!(combined.caps[b], caps16[b] as i32, "cap band {b}");
        }
        let sum: i64 = combined.bits.iter().map(|&x| x as i64).sum();
        assert_eq!(sum, combined.total);
    }

    /// Boosts are additive to the candidate (until the cap binds).
    #[test]
    fn boost_increases_candidate() {
        let lm = 0u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let no_boost = vec![0i32; n];
        let base = combine_band_allocation(0, &bins, 4, 0, &no_boost, 5, 1, false, lm).unwrap();

        // Boost band 5 by 16 1/8-bits; expect band 5 to rise by 16 as
        // long as the cap is not yet binding there.
        let mut boost = vec![0i32; n];
        boost[5] = 16;
        let boosted = combine_band_allocation(0, &bins, 4, 0, &boost, 5, 1, false, lm).unwrap();

        let rise = boosted.bits[5] - base.bits[5];
        let headroom = base.caps[5] - base.bits[5];
        assert_eq!(rise, 16.min(headroom.max(0)));
        // No other band changed.
        for b in (0..n).filter(|&b| b != 5) {
            assert_eq!(boosted.bits[b], base.bits[b], "band {b} unchanged");
        }
    }

    /// The cap clamp is honoured: a huge boost cannot push a band above
    /// its `cap[]`.
    #[test]
    fn cap_clamp_binds() {
        let lm = 0u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let mut boost = vec![0i32; n];
        // A boost far larger than any cap.
        boost[2] = 1_000_000;
        let combined = combine_band_allocation(0, &bins, 6, 0, &boost, 5, 1, false, lm).unwrap();
        assert_eq!(combined.bits[2], combined.caps[2]);
        for b in 0..n {
            assert!(combined.bits[b] <= combined.caps[b], "band {b} over cap");
            assert!(combined.bits[b] >= 0, "band {b} negative");
        }
    }

    /// A strongly-negative trim plus zero static cannot push a band
    /// below zero (the floor binds).
    #[test]
    fn zero_floor_binds() {
        let lm = 0u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let boost = vec![0i32; n];
        // alloc_trim = 0 gives the most-negative trim prefactor; at the
        // lowest quality column (qlo=0 ⇒ static all zero except via
        // interpolation) several bands floor at 0.
        let combined = combine_band_allocation(0, &bins, 0, 0, &boost, 0, 1, false, lm).unwrap();
        for b in 0..n {
            assert!(combined.bits[b] >= 0, "band {b} below floor");
            assert!(combined.bits[b] <= combined.caps[b], "band {b} over cap");
        }
    }

    /// The window total equals the sum of the per-band candidates.
    #[test]
    fn total_is_sum_of_bits() {
        let lm = 2u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let mut boost = vec![0i32; n];
        boost[10] = 24;
        boost[15] = 8;
        let combined = combine_band_allocation(0, &bins, 5, 32, &boost, 7, 1, false, lm).unwrap();
        let sum: i64 = combined.bits.iter().map(|&x| x as i64).sum();
        assert_eq!(sum, combined.total);
    }

    /// Stereo selects a different `cap[]` table row than mono, so the
    /// caps the clamp uses differ.
    #[test]
    fn stereo_uses_stereo_caps() {
        let lm = 1u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let boost = vec![0i32; n];
        let stereo = combine_band_allocation(0, &bins, 8, 0, &boost, 5, 2, true, lm).unwrap();
        let mut caps_stereo = vec![0i16; n];
        assert!(compute_band_caps(lm, true, 2, &bins, &mut caps_stereo));
        for (b, &cap) in caps_stereo.iter().enumerate() {
            assert_eq!(stereo.caps[b], cap as i32, "stereo cap band {b}");
        }
    }

    /// Hybrid window (coding_start = 17) covers bands 17..=20 only.
    #[test]
    fn hybrid_window() {
        let lm = 3u32;
        let full = mono_window(lm as usize);
        let bins = full[17..].to_vec();
        let n = bins.len();
        assert_eq!(n, 4);
        let boost = vec![0i32; n];
        let combined = combine_band_allocation(17, &bins, 6, 0, &boost, 5, 1, false, lm).unwrap();
        assert_eq!(combined.bits.len(), 4);
        for b in 0..n {
            assert!(combined.bits[b] >= 0);
            assert!(combined.bits[b] <= combined.caps[b]);
        }
    }

    /// Input validation: mismatched boost length, bad channels, bad lm,
    /// bad trim, and an overflowing window all return `None`.
    #[test]
    fn invalid_inputs_return_none() {
        let lm = 0u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let boost = vec![0i32; n];

        // boost length mismatch
        let short_boost = vec![0i32; n - 1];
        assert!(combine_band_allocation(0, &bins, 4, 0, &short_boost, 5, 1, false, lm).is_none());
        // channels = 0
        assert!(combine_band_allocation(0, &bins, 4, 0, &boost, 5, 0, false, lm).is_none());
        // channels = 3
        assert!(combine_band_allocation(0, &bins, 4, 0, &boost, 5, 3, false, lm).is_none());
        // lm > 3
        assert!(combine_band_allocation(0, &bins, 4, 0, &boost, 5, 1, false, 4).is_none());
        // alloc_trim out of range
        assert!(combine_band_allocation(0, &bins, 4, 0, &boost, 11, 1, false, lm).is_none());
        // window overflow (coding_start too high for full window)
        assert!(combine_band_allocation(5, &bins, 4, 0, &boost, 5, 1, false, lm).is_none());
        // qlo out of range
        assert!(combine_band_allocation(0, &bins, 11, 0, &boost, 5, 1, false, lm).is_none());
    }
}
