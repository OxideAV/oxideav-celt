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
use crate::band_minimums::{compute_trim_offsets, SHORT_FRAME_BAND_BINS};
use crate::static_alloc::{window_static_alloc_per_band_1_8th, INTERP_STEPS, NUM_Q};

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
///   the window). `bins_per_band.len()` is the window length. These
///   widths drive the trim offsets and the caps; the static-allocation
///   term internally uses the 2.5 ms **base** widths
///   ([`SHORT_FRAME_BAND_BINS`]) because the RFC's `<< LM` factor
///   already restores the actual bin count (clean-room narrative,
///   reallocation-walk chapter §2).
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
    //
    // Dimensional note (clean-room narrative, reallocation-walk chapter
    // §2): the `N` in the RFC's `channels * N * alloc[band][q] << LM
    // >> 2` is the band's MDCT-bin width at the 2.5 ms BASE frame size
    // (LM = 0); the `<< LM` inside the evaluator restores the actual
    // bin count. Feeding the evaluator the LM-scaled widths would
    // double-apply the `2^LM` factor, inflating the static allocation
    // on every frame size above 2.5 ms. The caller's `bins_per_band`
    // (actual widths) still drives the trim offsets and caps below,
    // which the RFC defines on the actual frame-size widths.
    if coding_start + n > SHORT_FRAME_BAND_BINS.len() {
        return None;
    }
    let base_bins = &SHORT_FRAME_BAND_BINS[coding_start..coding_start + n];
    let mut static_alloc = vec![0u32; n];
    if !window_static_alloc_per_band_1_8th(
        coding_start,
        base_bins,
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

/// The result of the §4.3.3 quality-column search over the *combined*
/// (cap-clamped) candidate: the chosen grid position plus the
/// [`CombinedAllocation`] assembled there.
///
/// `qlo` / `frac` is the selected grid position on the same 1/64-step
/// interpolation grid [`crate::static_alloc::find_static_alloc`] uses;
/// `alloc` is the per-band candidate `combine_band_allocation` produced
/// at that position. `alloc.total <= budget` is the search's exit
/// invariant whenever the budget can afford even the `(0, 0)` cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombinedAllocSearch {
    /// Lower quality column at the selected grid position
    /// (`0..=NUM_Q-1`).
    pub qlo: usize,
    /// 1/64-step sub-column position between `qlo` and `qlo+1`
    /// (`0..=INTERP_STEPS`). `INTERP_STEPS == 64` is the canonical
    /// "exactly on the upper edge" pin.
    pub frac: u32,
    /// The combined per-band candidate assembled at `(qlo, frac)`.
    pub alloc: CombinedAllocation,
}

/// Search the §4.3.3 1/64-step interpolation grid for the highest
/// quality column whose **combined** (cap-clamped, boost- and
/// trim-inclusive) window total does not exceed `budget_1_8th`.
///
/// This is [`crate::static_alloc::find_static_alloc`] taken one step
/// further into the §4.3.3 prose: that search finds the highest column
/// whose *static-only* window total fits; this one searches the column
/// grid against the total of the full per-band candidate
///
/// ```text
/// bits[b] = clamp(static[b] + boost[b] + trim_offset[b], 0, cap[b])
/// ```
///
/// i.e. the entry "nearest but not exceeding the available space,
/// subject to the tilt, boosts, [and] band maximums" before the linear
/// interpolation — exactly the words RFC 6716 §4.3.3 uses for the
/// allocation search. The boosts and trim offsets do not depend on the
/// column, and `clamp(static + const, 0, cap)` is non-decreasing in
/// `static`, which is itself non-decreasing in `(qlo, frac)`; the
/// combined window total is therefore monotonically non-decreasing
/// along the grid, so the same two-phase bisection
/// [`find_static_alloc`](crate::static_alloc::find_static_alloc) uses is
/// valid here.
///
/// What this does **not** do: the §2.7 hard-minimum **skip** decision
/// (comparing each band against `thresh[band]`, the concurrent skip
/// decoding, and the fine-energy/shape split) stays deferred to the
/// reference per RFC 6716 §4.3.3 / the clean-room narrative §2.7. The
/// `thresh[]` floor is *not* folded into the search budget here; a
/// caller that holds it carries it into that deferred bisection.
///
/// Inputs mirror [`combine_band_allocation`] (the column `(qlo, frac)`
/// is replaced by the `budget_1_8th` the search drives toward):
///
/// * `coding_start`, `bins_per_band`, `boost`, `alloc_trim`,
///   `channels`, `stereo`, `lm` — passed unchanged to
///   `combine_band_allocation` at each probed column.
/// * `budget_1_8th` — the §4.3.3 remaining budget in 1/8 bits the
///   combined window total must not exceed.
///
/// Returns `None` on the same input-validation paths as
/// [`combine_band_allocation`] (mismatched `boost` length, bad
/// `channels` / `lm` / `alloc_trim`, or an overflowing window).
///
/// When even the `(qlo = 0, frac = 0)` cell exceeds the budget the
/// search returns that cell anyway — its combined total is the minimum
/// achievable, and the §4.3.3 prose notes minimums and boosts dominate
/// at "very low rates"; the caller falls through to the deferred
/// minimums/skip stage.
#[allow(clippy::too_many_arguments)]
pub fn find_combined_alloc(
    coding_start: usize,
    bins_per_band: &[u32],
    boost: &[i32],
    alloc_trim: i32,
    channels: u32,
    stereo: bool,
    lm: u32,
    budget_1_8th: i64,
) -> Option<CombinedAllocSearch> {
    // Evaluate the combined candidate at one grid position. `frac ==
    // INTERP_STEPS` is the canonical upper-column-edge pin: it is the
    // `(qlo + 1, 0)` cell, evaluated through `combine_band_allocation`
    // so the cap clamp and boost/trim addition are identical to every
    // other probe.
    let probe = |qlo: usize, frac: u32| -> Option<CombinedAllocation> {
        if frac == INTERP_STEPS {
            combine_band_allocation(
                coding_start,
                bins_per_band,
                qlo + 1,
                0,
                boost,
                alloc_trim,
                channels,
                stereo,
                lm,
            )
        } else {
            combine_band_allocation(
                coding_start,
                bins_per_band,
                qlo,
                frac,
                boost,
                alloc_trim,
                channels,
                stereo,
                lm,
            )
        }
    };

    // The `(0, 0)` probe pins every input-validation guard and is the
    // search's lower bound. If it already exceeds the budget we return
    // it (its total is the floor; see the "very low rates" note above).
    let zero = probe(0, 0)?;
    if zero.total > budget_1_8th {
        return Some(CombinedAllocSearch {
            qlo: 0,
            frac: 0,
            alloc: zero,
        });
    }

    // Phase 1: coarse column scan. Largest `qlo` whose integer-column
    // combined total fits. Round-up midpoint so `hi == lo + 1` makes
    // progress.
    let mut lo: usize = 0;
    let mut hi: usize = NUM_Q - 1;
    let mut lo_alloc = zero;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let cand = probe(mid, 0)?;
        if cand.total <= budget_1_8th {
            lo = mid;
            lo_alloc = cand;
        } else {
            hi = mid - 1;
        }
    }
    let qlo = lo;

    // At the top column there is no upper neighbour to interpolate
    // toward; the grid degenerates to the integer column.
    if qlo == NUM_Q - 1 {
        return Some(CombinedAllocSearch {
            qlo,
            frac: 0,
            alloc: lo_alloc,
        });
    }

    // Phase 2: fine fractional scan. Largest `frac ∈ 0..=INTERP_STEPS`
    // whose interpolated combined total fits.
    let mut flo: u32 = 0;
    let mut fhi: u32 = INTERP_STEPS;
    let mut flo_alloc = lo_alloc;
    while flo < fhi {
        let mid = flo + (fhi - flo).div_ceil(2);
        let cand = probe(qlo, mid)?;
        if cand.total <= budget_1_8th {
            flo = mid;
            flo_alloc = cand;
        } else {
            fhi = mid - 1;
        }
    }

    Some(CombinedAllocSearch {
        qlo,
        frac: flo,
        alloc: flo_alloc,
    })
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

        // Reconstruct static + trim independently. The static term is
        // evaluated on the 2.5 ms base widths (the evaluator's `<< LM`
        // restores the actual bin count — reallocation-walk chapter §2).
        let mut static_alloc = vec![0u32; n];
        assert!(window_static_alloc_per_band_1_8th(
            0,
            &SHORT_FRAME_BAND_BINS[..n],
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

    // ----- find_combined_alloc -----

    /// The brute-force reference: the highest grid position whose
    /// combined total fits, scanning every `(qlo, frac)` cell in
    /// monotone order. Returns the same `(qlo, frac)` the bisection
    /// must land on.
    #[allow(clippy::too_many_arguments)]
    fn brute_combined(
        coding_start: usize,
        bins: &[u32],
        boost: &[i32],
        trim: i32,
        channels: u32,
        stereo: bool,
        lm: u32,
        budget: i64,
    ) -> (usize, u32, i64) {
        let mut best = (0usize, 0u32, i64::MIN);
        // Enumerate columns in ascending grid order: for each qlo,
        // frac 0..INTERP_STEPS, then the qlo==NUM_Q-1 integer column.
        for qlo in 0..NUM_Q {
            let frac_hi = if qlo == NUM_Q - 1 {
                0
            } else {
                INTERP_STEPS - 1
            };
            for frac in 0..=frac_hi {
                let c = combine_band_allocation(
                    coding_start,
                    bins,
                    qlo,
                    frac,
                    boost,
                    trim,
                    channels,
                    stereo,
                    lm,
                )
                .unwrap();
                if c.total <= budget {
                    best = (qlo, frac, c.total);
                }
            }
            // The fractional upper-edge pin (qlo+1, 0) is the same cell
            // as the next column's frac==0; it is reached by stepping
            // qlo, so the per-qlo loop above already covers it.
        }
        best
    }

    /// The bisection agrees with the brute-force scan across a sweep of
    /// budgets, for mono and stereo, across all four frame sizes.
    #[test]
    fn search_matches_brute_force() {
        for lm in 0u32..=3 {
            let bins = mono_window(lm as usize);
            let n = bins.len();
            for &(channels, stereo) in &[(1u32, false), (2u32, true)] {
                let mut boost = vec![0i32; n];
                boost[3] = 24;
                boost[12] = 8;
                for trim in [0, 5, 7, 10] {
                    // The full-grid total bounds the budget sweep.
                    let top = combine_band_allocation(
                        0,
                        &bins,
                        NUM_Q - 1,
                        0,
                        &boost,
                        trim,
                        channels,
                        stereo,
                        lm,
                    )
                    .unwrap()
                    .total;
                    for &budget in &[0i64, 1, top / 4, top / 2, top - 1, top, top + 100] {
                        let got = find_combined_alloc(
                            0, &bins, &boost, trim, channels, stereo, lm, budget,
                        )
                        .unwrap();
                        let (eqlo, efrac, etotal) =
                            brute_combined(0, &bins, &boost, trim, channels, stereo, lm, budget);
                        // When even (0,0) overflows, both the search and
                        // the brute force "best" sit at (0,0); compare
                        // position and the assembled total.
                        assert_eq!(
                            got.qlo, eqlo,
                            "lm={lm} ch={channels} trim={trim} b={budget} qlo"
                        );
                        assert_eq!(
                            got.frac, efrac,
                            "lm={lm} ch={channels} trim={trim} b={budget} frac"
                        );
                        // etotal is i64::MIN only if no cell fit (budget
                        // below (0,0)); then the search returns (0,0).
                        if etotal != i64::MIN {
                            assert_eq!(got.alloc.total, etotal, "lm={lm} b={budget} total");
                        } else {
                            assert_eq!((got.qlo, got.frac), (0, 0));
                        }
                    }
                }
            }
        }
    }

    /// The chosen combined total never exceeds the budget when the
    /// budget can afford the (0,0) cell, and the next grid position up
    /// would exceed it (the "nearest but not exceeding" exit).
    #[test]
    fn search_total_within_budget_and_tight() {
        let lm = 2u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let boost = vec![0i32; n];
        let top = combine_band_allocation(0, &bins, NUM_Q - 1, 0, &boost, 5, 1, false, lm)
            .unwrap()
            .total;
        // A mid budget that admits a non-trivial column.
        let budget = top / 2;
        let got = find_combined_alloc(0, &bins, &boost, 5, 1, false, lm, budget).unwrap();
        assert!(got.alloc.total <= budget, "total over budget");

        // Stepping one frac up (or one column if at the top of the
        // fraction range) must exceed the budget — otherwise the search
        // stopped early.
        let next = if got.frac < INTERP_STEPS {
            combine_band_allocation(0, &bins, got.qlo, got.frac + 1, &boost, 5, 1, false, lm)
        } else {
            combine_band_allocation(0, &bins, got.qlo + 1, 0, &boost, 5, 1, false, lm)
        };
        if let Some(next) = next {
            assert!(next.total > budget, "search stopped before the budget edge");
        }
    }

    /// A generous budget lands on the top column; the assembled
    /// candidate equals `combine_band_allocation` at `(NUM_Q-1, 0)`.
    #[test]
    fn search_saturates_top_column() {
        let lm = 1u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let boost = vec![0i32; n];
        let got = find_combined_alloc(0, &bins, &boost, 5, 1, false, lm, i64::MAX / 2).unwrap();
        assert_eq!(got.qlo, NUM_Q - 1);
        assert_eq!(got.frac, 0);
        let top = combine_band_allocation(0, &bins, NUM_Q - 1, 0, &boost, 5, 1, false, lm).unwrap();
        assert_eq!(got.alloc, top);
    }

    /// A zero budget returns the `(0, 0)` cell (the floor); its total is
    /// the minimum achievable and may exceed the budget.
    #[test]
    fn search_zero_budget_returns_floor() {
        let lm = 0u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let mut boost = vec![0i32; n];
        // A boost forces a non-zero (0,0) total, so the floor exceeds a
        // zero budget.
        boost[4] = 200;
        let got = find_combined_alloc(0, &bins, &boost, 5, 1, false, lm, 0).unwrap();
        assert_eq!((got.qlo, got.frac), (0, 0));
        let floor = combine_band_allocation(0, &bins, 0, 0, &boost, 5, 1, false, lm).unwrap();
        assert_eq!(got.alloc, floor);
        assert!(floor.total > 0);
    }

    /// Hybrid window (coding_start = 17) searches bands 17..=20 only.
    #[test]
    fn search_hybrid_window() {
        let lm = 3u32;
        let full = mono_window(lm as usize);
        let bins = full[17..].to_vec();
        let n = bins.len();
        assert_eq!(n, 4);
        let boost = vec![0i32; n];
        let top = combine_band_allocation(17, &bins, NUM_Q - 1, 0, &boost, 5, 1, false, lm)
            .unwrap()
            .total;
        let got = find_combined_alloc(17, &bins, &boost, 5, 1, false, lm, top / 2).unwrap();
        assert_eq!(got.alloc.bits.len(), 4);
        assert!(got.alloc.total <= top / 2);
        let (eqlo, efrac, _) = brute_combined(17, &bins, &boost, 5, 1, false, lm, top / 2);
        assert_eq!((got.qlo, got.frac), (eqlo, efrac));
    }

    /// Input validation propagates from `combine_band_allocation`.
    #[test]
    fn search_invalid_inputs_return_none() {
        let lm = 0u32;
        let bins = mono_window(lm as usize);
        let n = bins.len();
        let boost = vec![0i32; n];
        // boost length mismatch
        let short = vec![0i32; n - 1];
        assert!(find_combined_alloc(0, &bins, &short, 5, 1, false, lm, 1000).is_none());
        // bad channels
        assert!(find_combined_alloc(0, &bins, &boost, 5, 0, false, lm, 1000).is_none());
        // bad lm
        assert!(find_combined_alloc(0, &bins, &boost, 5, 1, false, 4, 1000).is_none());
        // bad trim
        assert!(find_combined_alloc(0, &bins, &boost, 11, 1, false, lm, 1000).is_none());
        // overflowing window
        assert!(find_combined_alloc(5, &bins, &boost, 5, 1, false, lm, 1000).is_none());
    }
}
