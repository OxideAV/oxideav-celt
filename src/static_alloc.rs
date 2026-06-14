//! CELT static allocation table (RFC 6716 §4.3.3, Table 57).
//!
//! ## What this module covers
//!
//! After the §4.3.3 reservation walk
//! ([`crate::allocation_budget::compute_initial_reservations`]), the
//! band-boost loop ([`crate::band_cap::decode_band_boosts`]), the trim
//! decode ([`crate::bit_allocation::decode_alloc_trim`]), and the
//! §2.6 per-band minimums + trim offsets
//! ([`crate::band_minimums`]), the §4.3.3 allocator selects a
//! starting quality `q` from the **static allocation table** (RFC 6716
//! Table 57). The static allocation in 1/8 bits for a given band/q is:
//!
//! ```text
//!     channels * N * alloc[band][q] << LM >> 2
//! ```
//!
//! where `alloc[band][q]` is in units of 1/32 bit per MDCT bin
//! (RFC 6716 §4.3.3 lines 6223–6229), `N` is the per-channel MDCT-bin
//! count for the band, and `LM` is `log2(frame_size/120)` ∈ `{0,1,2,3}`.
//!
//! ## Table 57 transcription
//!
//! Table 57 is rendered as ASCII art directly inside RFC 6716 §4.3.3
//! (lines 6234–6286 of `docs/audio/opus/rfc6716-opus.txt`). Unlike
//! `e_prob_model` / `cache_caps50` / `LOG2_FRAC_TABLE`, which the RFC
//! delegates to external source files, Table 57 lives entirely inside
//! the RFC body — so we can transcribe its 11 × 21 = 231 cell values
//! directly with the docs/ allow-list satisfied. The CSV staging step
//! is therefore unnecessary; the table is reproduced here verbatim.
//!
//! ## Interpolation
//!
//! The actual per-band static allocation is obtained by linear
//! interpolation between two adjacent `q` columns in steps of 1/64
//! (RFC 6716 §4.3.3 lines 6227–6229). With the lower column `qlo` and
//! the upper column `qlo+1`, and a sub-column position `frac ∈ 0..=63`
//! representing the 1/64 step, the interpolated 1/32-bit-per-bin
//! coefficient is:
//!
//! ```text
//!   interp[band] = ((64 - frac) * alloc[band][qlo]
//!                 +       frac  * alloc[band][qlo+1] + 32) / 64
//! ```
//!
//! (rounding to nearest with a 32-half-bit step to mirror the RFC's
//! "highest allocation that does not exceed the bits remaining"
//! contract; the additive constant +32 is the half-LSB round-to-nearest
//! tweak that keeps the inner search deterministic across forward and
//! reverse evaluation orders).
//!
//! Together with [`band_static_alloc_1_8th`] (which folds in the
//! `channels * N << LM >> 2` scaling), the caller can search the
//! interpolation grid for the highest sub-column position whose total
//! over the coded-band window does not exceed the §4.3.3 remaining
//! budget. The search itself is provided by [`find_static_alloc`],
//! which bisects the `(qlo, frac)` grid in two phases (coarse column
//! scan then fine fractional scan) and returns a
//! [`StaticAllocSearch`] carrier with the chosen position and the
//! committed 1/8-bit window total.

use crate::coarse_energy::NUM_BANDS;

/// CELT static allocation table (RFC 6716 §4.3.3, Table 57).
///
/// Indexed as `STATIC_ALLOC[band][q]`. Units are 1/32 bit per MDCT
/// bin. `band ∈ 0..NUM_BANDS` (21 bands per RFC Table 55), `q ∈ 0..11`
/// (the 11 quality columns).
///
/// Values transcribed verbatim from RFC 6716 §4.3.3 Table 57
/// (`docs/audio/opus/rfc6716-opus.txt` lines 6234–6286).
pub const STATIC_ALLOC: [[u8; NUM_Q]; NUM_BANDS] = [
    // band 0
    [0, 90, 110, 118, 126, 134, 144, 152, 162, 172, 200],
    // band 1
    [0, 80, 100, 110, 119, 127, 137, 145, 155, 165, 200],
    // band 2
    [0, 75, 90, 103, 112, 120, 130, 138, 148, 158, 200],
    // band 3
    [0, 69, 84, 93, 104, 114, 124, 132, 142, 152, 200],
    // band 4
    [0, 63, 78, 86, 95, 103, 113, 123, 133, 143, 200],
    // band 5
    [0, 56, 71, 80, 89, 97, 107, 117, 127, 137, 200],
    // band 6
    [0, 49, 65, 75, 83, 91, 101, 111, 121, 131, 200],
    // band 7
    [0, 40, 58, 70, 78, 85, 95, 105, 115, 125, 200],
    // band 8
    [0, 34, 51, 65, 72, 78, 88, 98, 108, 118, 198],
    // band 9
    [0, 29, 45, 59, 66, 72, 82, 92, 102, 112, 193],
    // band 10
    [0, 20, 39, 53, 60, 66, 76, 86, 96, 106, 188],
    // band 11
    [0, 18, 32, 47, 54, 60, 70, 80, 90, 100, 183],
    // band 12
    [0, 10, 26, 40, 47, 54, 64, 74, 84, 94, 178],
    // band 13
    [0, 0, 20, 31, 39, 47, 57, 67, 77, 87, 173],
    // band 14
    [0, 0, 12, 23, 32, 41, 51, 61, 71, 81, 168],
    // band 15
    [0, 0, 0, 15, 25, 35, 45, 55, 65, 75, 163],
    // band 16
    [0, 0, 0, 4, 17, 29, 39, 49, 59, 69, 158],
    // band 17
    [0, 0, 0, 0, 12, 23, 33, 43, 53, 63, 153],
    // band 18
    [0, 0, 0, 0, 1, 16, 26, 36, 46, 56, 148],
    // band 19
    [0, 0, 0, 0, 0, 10, 15, 20, 30, 45, 129],
    // band 20
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 20, 104],
];

/// Number of quality columns (`q` axis) in [`STATIC_ALLOC`].
///
/// RFC 6716 Table 57 enumerates eleven columns, `q ∈ 0..=10`.
pub const NUM_Q: usize = 11;

/// 1/64-step sub-column count used by the §4.3.3 linear interpolation
/// between adjacent quality columns (RFC 6716 lines 6227–6229).
pub const INTERP_STEPS: u32 = 64;

/// Half-step rounding offset for the interpolation (round-to-nearest
/// with a 1/2-unit additive constant, matching the §4.3.3 "highest
/// allocation that does not exceed the bits remaining" inner-search
/// contract).
const INTERP_ROUND: u32 = INTERP_STEPS / 2;

/// Return the per-band interpolated `alloc` coefficient in units of
/// 1/32 bit per MDCT bin, for a sub-column position `frac ∈ 0..=63`
/// between adjacent quality columns `qlo` and `qlo+1`.
///
/// `qlo + 1` MUST be `<= NUM_Q - 1` (i.e. `qlo ∈ 0..=NUM_Q-2`). The
/// boundary case `frac == 0` yields the exact `STATIC_ALLOC[band][qlo]`
/// entry; `frac == 63` yields a value `(63 * upper + 1 * lower + 32) / 64`
/// just below the upper-column entry; the integer endpoint is reached
/// by stepping `qlo` itself, not by extrapolating `frac == 64`.
///
/// Returns `None` if `band >= NUM_BANDS` or `qlo + 1 >= NUM_Q` or
/// `frac >= INTERP_STEPS`. All three error paths leave the caller in a
/// clean state (no clamping).
pub fn interp_alloc_1_32nd(band: usize, qlo: usize, frac: u32) -> Option<u32> {
    if band >= NUM_BANDS || qlo + 1 >= NUM_Q || frac >= INTERP_STEPS {
        return None;
    }
    let lower = STATIC_ALLOC[band][qlo] as u32;
    let upper = STATIC_ALLOC[band][qlo + 1] as u32;
    let weighted = (INTERP_STEPS - frac) * lower + frac * upper + INTERP_ROUND;
    Some(weighted / INTERP_STEPS)
}

/// Return the §4.3.3 static allocation for one band in **1/8 bits**.
///
/// Computes `channels * N * alloc << LM >> 2` per RFC 6716 §4.3.3
/// (line 6225), where `alloc` is the interpolated 1/32-bit-per-bin
/// coefficient produced by [`interp_alloc_1_32nd`] for the requested
/// `(band, qlo, frac)` position, `N = bins_per_channel`, and `LM` is
/// `log2(frame_size/120)`.
///
/// Returns `None` when [`interp_alloc_1_32nd`] would, or when
/// `channels ∉ {1, 2}`, or when `lm > 3`, or when the multiply would
/// exceed the `u32` range (defensively — the legitimate range is well
/// inside `u32`, but the saturation guard means a caller passing a
/// pathological `bins_per_channel` gets `None` instead of a wrap).
///
/// The shape of the multiply matches the RFC literally: the `<< LM`
/// scales the result by the per-frame-size factor, and the `>> 2`
/// reduces the 1/32-bit units to 1/8-bit units (factor of 4).
pub fn band_static_alloc_1_8th(
    band: usize,
    qlo: usize,
    frac: u32,
    channels: u32,
    bins_per_channel: u32,
    lm: u32,
) -> Option<u32> {
    if !(1..=2).contains(&channels) || lm > 3 {
        return None;
    }
    let alloc = interp_alloc_1_32nd(band, qlo, frac)?;
    // channels (∈ 1..=2) * N (≤ 176 per Table 55 at LM=3) * alloc
    // (≤ 200) ≤ 2 * 176 * 200 = 70_400 < 2^17 before the LM shift.
    // After `<< LM` (LM ≤ 3) the product is ≤ 70_400 * 8 = 563_200 <
    // 2^20, well inside `u32`.
    let raw = channels
        .checked_mul(bins_per_channel)?
        .checked_mul(alloc)?
        .checked_shl(lm)?;
    Some(raw >> 2)
}

/// Window static-allocation evaluator at an integer column `q` (no
/// interpolation), in 1/8 bits. This is the helper the search's
/// coarse phase calls — distinct from [`window_static_alloc_1_8th`]
/// because the interpolated path requires `qlo + 1 < NUM_Q`, so the
/// top column `q == NUM_Q - 1` is not reachable through it.
///
/// Returns `None` on the same input-validation paths as
/// [`band_static_alloc_1_8th`], plus `q >= NUM_Q` and a window that
/// overflows `NUM_BANDS`.
fn window_static_alloc_at_column_1_8th(
    coding_start: usize,
    bins_per_band: &[u32],
    q: usize,
    channels: u32,
    lm: u32,
) -> Option<u32> {
    if !(1..=2).contains(&channels) || lm > 3 || q >= NUM_Q {
        return None;
    }
    if coding_start + bins_per_band.len() > NUM_BANDS {
        return None;
    }
    let mut total: u32 = 0;
    for (i, &bins) in bins_per_band.iter().enumerate() {
        let band = coding_start + i;
        // RFC 6716 §4.3.3 line 6225: `channels * N * alloc << LM >> 2`.
        let alloc = STATIC_ALLOC[band][q] as u32;
        let raw = channels
            .checked_mul(bins)?
            .checked_mul(alloc)?
            .checked_shl(lm)?;
        total = total.checked_add(raw >> 2)?;
    }
    Some(total)
}

/// Convenience: evaluate [`band_static_alloc_1_8th`] across a window of
/// coded bands and accumulate the total.
///
/// `coding_start` is the first band of the window (0 for pure CELT, 17
/// for Hybrid). `bins_per_band` is the per-channel MDCT-bin count for
/// each band in the window (`bins_per_band.len()` is the window
/// length). The total in 1/8 bits is summed across the window, with
/// each band evaluated at the same `(qlo, frac)` interpolation point.
///
/// Returns `None` on the same error paths as
/// [`band_static_alloc_1_8th`], including a window that overflows
/// `NUM_BANDS`.
pub fn window_static_alloc_1_8th(
    coding_start: usize,
    bins_per_band: &[u32],
    qlo: usize,
    frac: u32,
    channels: u32,
    lm: u32,
) -> Option<u32> {
    if coding_start + bins_per_band.len() > NUM_BANDS {
        return None;
    }
    let mut total: u32 = 0;
    for (i, &bins) in bins_per_band.iter().enumerate() {
        let band = coding_start + i;
        let cell = band_static_alloc_1_8th(band, qlo, frac, channels, bins, lm)?;
        total = total.checked_add(cell)?;
    }
    Some(total)
}

/// Fill `out[i]` with the §4.3.3 per-band static allocation, in 1/8
/// bits, at the interpolation grid position `(qlo, frac)`.
///
/// This is the per-band breakdown of the same window the search
/// ([`find_static_alloc`]) sums: where [`window_static_alloc_1_8th`]
/// returns only the scalar window total at `(qlo, frac)`, this routine
/// emits the individual per-band 1/8-bit allocations that total to it.
/// The §4.3.3 reallocation pass (RFC 6716 §4.3.3 lines 6431–6460 and
/// the §2.7 outcome prose) consumes this per-band vector together with
/// the per-band minimums (`thresh[]`,
/// [`crate::band_minimums::compute_thresh`]), the trim offsets
/// (`trim_offsets[]`, [`crate::band_minimums::compute_trim_offsets`]),
/// and the per-band caps (`cap[]`, [`crate::band_cap::compute_band_caps`])
/// to derive the final shape / fine-energy split. Exposing the
/// per-band vector here keeps that arithmetic decoupled from the
/// interpolation it depends on.
///
/// The per-band value is
///
/// ```text
///   channels * N[band] * interp_alloc(band, qlo, frac) << LM >> 2
/// ```
///
/// exactly as [`band_static_alloc_1_8th`] computes it for one band,
/// where `interp_alloc` is the §2.1 1/64-step linear interpolation
/// between columns `qlo` and `qlo+1` ([`interp_alloc_1_32nd`]).
///
/// The top-column position is reachable here: when `qlo == NUM_Q - 1`
/// (the search's saturated exit) there is no upper column to
/// interpolate toward, so `frac` MUST be 0 and the integer column
/// `NUM_Q - 1` is evaluated directly (mirroring the column path
/// [`find_static_alloc`] uses at saturation). Any non-zero `frac` at
/// the top column is rejected, since the grid has no sub-column there.
///
/// Inputs match [`window_static_alloc_1_8th`]:
///
/// * `coding_start` — first band of the coded-band window (0 for pure
///   CELT, 17 for Hybrid).
/// * `bins_per_band` — per-channel MDCT-bin count for each band in the
///   window; `bins_per_band.len()` is the window length.
/// * `qlo` / `frac` — the grid position (typically a
///   [`StaticAllocSearch`] outcome).
/// * `channels` — 1 (mono) or 2 (stereo).
/// * `lm` — `log2(frame_size / 120)` ∈ `{0,1,2,3}`.
/// * `out` — receives the per-band 1/8-bit allocations; its length MUST
///   equal `bins_per_band.len()`.
///
/// Returns `false` (and leaves `out` unchanged) on any input-validation
/// failure: `out.len() != bins_per_band.len()`, a window overflowing
/// `NUM_BANDS`, `channels ∉ {1,2}`, `lm > 3`, `qlo >= NUM_Q`, a
/// non-zero `frac` at the top column, or `frac >= INTERP_STEPS` below
/// the top column. Returns `true` on success.
///
/// On success the emitted vector satisfies
/// `out.iter().sum() == window_static_alloc_1_8th(...).unwrap()` (or the
/// column-evaluator total at the top column) — the per-band split is a
/// faithful decomposition of the window total.
pub fn window_static_alloc_per_band_1_8th(
    coding_start: usize,
    bins_per_band: &[u32],
    qlo: usize,
    frac: u32,
    channels: u32,
    lm: u32,
    out: &mut [u32],
) -> bool {
    if out.len() != bins_per_band.len() {
        return false;
    }
    if !(1..=2).contains(&channels) || lm > 3 || qlo >= NUM_Q {
        return false;
    }
    if coding_start.saturating_add(bins_per_band.len()) > NUM_BANDS {
        return false;
    }
    // The top column has no upper neighbour to interpolate toward; the
    // grid degenerates to the integer column there, so `frac` must be 0.
    let top_column = qlo == NUM_Q - 1;
    if top_column {
        if frac != 0 {
            return false;
        }
    } else if frac >= INTERP_STEPS {
        return false;
    }

    // First compute every cell into a scratch so a late overflow leaves
    // `out` untouched (the documented "leaves out unchanged" contract).
    let mut scratch = [0u32; NUM_BANDS];
    for (i, &bins) in bins_per_band.iter().enumerate() {
        let band = coding_start + i;
        let cell = if top_column {
            // Direct integer-column evaluation: `channels * N * alloc
            // << LM >> 2` with `alloc = STATIC_ALLOC[band][NUM_Q-1]`.
            let alloc = STATIC_ALLOC[band][qlo] as u32;
            let raw = match channels
                .checked_mul(bins)
                .and_then(|v| v.checked_mul(alloc))
                .and_then(|v| v.checked_shl(lm))
            {
                Some(v) => v,
                None => return false,
            };
            raw >> 2
        } else {
            match band_static_alloc_1_8th(band, qlo, frac, channels, bins, lm) {
                Some(v) => v,
                None => return false,
            }
        };
        scratch[i] = cell;
    }
    out.copy_from_slice(&scratch[..bins_per_band.len()]);
    true
}

/// Outcome of the §4.3.3 static-allocation search.
///
/// The search picks the highest grid position `(qlo, frac)` whose
/// per-band-summed static allocation, in 1/8 bits, does not exceed the
/// supplied budget. `qlo ∈ 0..=NUM_Q-1` is the lower quality column,
/// `frac ∈ 0..=INTERP_STEPS` is the 1/64-step sub-column position
/// between `qlo` and `qlo+1`. `frac == INTERP_STEPS` is the canonical
/// representation of "exactly on the upper-column edge" — i.e. the
/// caller is sitting on `qlo+1` with zero residual fraction — and only
/// arises when the budget happens to land precisely on a column.
///
/// `total_1_8th` is the window sum the search committed to, in 1/8
/// bits; `total_1_8th <= budget` is the search's exit invariant.
///
/// When the budget is too small to afford even `(qlo=0, frac=0)` the
/// returned [`StaticAllocSearch::total_1_8th`] is 0 and both
/// `qlo == 0` and `frac == 0`. The RFC notes that minimums and boosts
/// dominate at "very low rates" — that is the search's "zero
/// allocation" exit; the caller is expected to fall through to the
/// minimums / cap / boost composition in the next stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticAllocSearch {
    /// Lower quality column at the selected grid position
    /// (`0..=NUM_Q-1`).
    pub qlo: usize,
    /// 1/64-step sub-column position between `qlo` and `qlo+1`
    /// (`0..=INTERP_STEPS`). `INTERP_STEPS == 64` is the canonical
    /// "exactly on the upper edge" pin.
    pub frac: u32,
    /// Window sum in 1/8 bits at the selected `(qlo, frac)`; satisfies
    /// `total_1_8th <= budget`.
    pub total_1_8th: u32,
}

/// Search the §4.3.3 1/64-step interpolation grid for the highest
/// `(qlo, frac)` position whose window static allocation does not
/// exceed `budget_1_8th`.
///
/// This is the inner search the §4.3.3 prose describes as "linearly
/// interpolating between two values of q (in steps of 1/64) to find
/// the highest allocation that does not exceed the number of bits
/// remaining" (RFC 6716 lines 6227–6229). The budget supplied here is
/// the §4.3.3 "remaining" budget — i.e. the §4.3.3 §2.5 `total` minus
/// the band-boost loop's `total_boost`, with the per-band minimums
/// (`thresh[]`) deducted by the caller. The minimums and the per-band
/// `cap[]` are not folded into the search itself; the §4.3.3
/// reallocation pass redistributes the residual budget against them.
///
/// The search runs in two phases:
///
/// 1. **Coarse column scan.** Bisect `q ∈ 0..=NUM_Q-1` for the largest
///    `qlo` such that the window sum at `frac == 0` (i.e. the integer
///    `qlo` column) is `<= budget_1_8th`. The window sum is
///    monotonically non-decreasing in `qlo` because every row of
///    Table 57 is non-decreasing in q.
/// 2. **Fine fractional scan.** When `qlo < NUM_Q - 1`, bisect
///    `frac ∈ 0..=INTERP_STEPS` for the largest `frac` such that the
///    window sum at `(qlo, frac)` is `<= budget_1_8th`. The window
///    sum is monotonically non-decreasing in `frac` for the same row
///    reason. `frac == INTERP_STEPS` is the canonical hit on the
///    upper column edge.
///
/// Returns `None` on the same input-validation paths as
/// [`window_static_alloc_1_8th`] (window overflows `NUM_BANDS`,
/// `channels ∉ {1, 2}`, or `lm > 3`).
///
/// The zero-allocation exit (budget below the `(0, 0)` cell, which
/// the §4.3.3 prose calls out as the "very low rates" case) returns
/// `Some(StaticAllocSearch { qlo: 0, frac: 0, total_1_8th: 0 })`.
pub fn find_static_alloc(
    coding_start: usize,
    bins_per_band: &[u32],
    channels: u32,
    lm: u32,
    budget_1_8th: u32,
) -> Option<StaticAllocSearch> {
    // Input validation: a probe evaluation pins the channels / lm /
    // window-overflow guards. We use the column evaluator so the
    // top-column probe is reachable (the interpolated evaluator
    // rejects `qlo == NUM_Q - 1` since there is no upper column).
    let probe_zero =
        window_static_alloc_at_column_1_8th(coding_start, bins_per_band, 0, channels, lm)?;

    // Zero-allocation exit: even q=0 exceeds the budget. (q=0 column
    // is all zero per Table 57, so `probe_zero` is necessarily zero
    // for legitimate inputs; the explicit check covers a defensive
    // future-proofing path.)
    if probe_zero > budget_1_8th {
        return Some(StaticAllocSearch {
            qlo: 0,
            frac: 0,
            total_1_8th: 0,
        });
    }

    // Phase 1: coarse column scan. Find the largest qlo ∈ 0..=NUM_Q-1
    // such that the integer-column window total is <= budget. Window
    // total at integer q is monotonically non-decreasing in q because
    // every row of Table 57 is non-decreasing in q (proven by the
    // `rows_monotonic_in_q` test).
    //
    // Invariant: `lo` always fits, `hi` is the open upper bound of the
    // search range. We bisect with the round-up midpoint rule so we
    // make progress when `hi == lo + 1`.
    let mut lo: usize = 0;
    let mut hi: usize = NUM_Q - 1;
    let mut lo_total: u32 = probe_zero;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let total =
            window_static_alloc_at_column_1_8th(coding_start, bins_per_band, mid, channels, lm)?;
        if total <= budget_1_8th {
            lo = mid;
            lo_total = total;
        } else {
            hi = mid - 1;
        }
    }
    let qlo = lo;

    // At qlo == NUM_Q - 1 there is no upper column to interpolate
    // toward; the search exits at frac == 0 with the lo_total we
    // already have.
    if qlo == NUM_Q - 1 {
        return Some(StaticAllocSearch {
            qlo,
            frac: 0,
            total_1_8th: lo_total,
        });
    }

    // Phase 2: fine fractional scan. Find the largest frac ∈
    // 0..=INTERP_STEPS such that the interpolated window total fits.
    // Lower bound: frac == 0 fits (lo_total). Upper bound:
    // INTERP_STEPS (the canonical upper-column-edge pin; we evaluate
    // it as the qlo+1 column at frac=0 to keep the interpolated
    // evaluator's frac ∈ 0..INTERP_STEPS contract intact).
    let mut flo: u32 = 0;
    let mut fhi: u32 = INTERP_STEPS;
    let mut flo_total: u32 = lo_total;
    while flo < fhi {
        let mid = flo + (fhi - flo).div_ceil(2);
        let total = if mid == INTERP_STEPS {
            window_static_alloc_at_column_1_8th(coding_start, bins_per_band, qlo + 1, channels, lm)?
        } else {
            window_static_alloc_1_8th(coding_start, bins_per_band, qlo, mid, channels, lm)?
        };
        if total <= budget_1_8th {
            flo = mid;
            flo_total = total;
        } else {
            fhi = mid - 1;
        }
    }

    Some(StaticAllocSearch {
        qlo,
        frac: flo,
        total_1_8th: flo_total,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spot-check shape: 21 bands × 11 quality columns per RFC Table 57.
    #[test]
    fn table_shape_matches_rfc() {
        assert_eq!(STATIC_ALLOC.len(), NUM_BANDS);
        for row in STATIC_ALLOC.iter() {
            assert_eq!(row.len(), NUM_Q);
        }
        assert_eq!(NUM_Q, 11);
    }

    /// `q = 0` column is identically zero per the RFC: no static
    /// allocation at the lowest quality level (all bits must come from
    /// boosts / reservations / minimums).
    #[test]
    fn q0_column_is_all_zero() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            assert_eq!(row[0], 0, "band {band} q=0 should be 0");
        }
    }

    /// `q = 10` column saturates at 200 for the first 8 bands (low
    /// frequencies) per RFC Table 57, and decreases for higher bands.
    /// Pin a few representative values to catch transcription errors.
    #[test]
    fn q10_endpoints_match_rfc() {
        // Bands 0..=7 saturate at 200.
        for (band, row) in STATIC_ALLOC.iter().enumerate().take(8) {
            assert_eq!(row[10], 200, "band {band} q=10");
        }
        // Bands 8..=20 fall off according to the RFC.
        assert_eq!(STATIC_ALLOC[8][10], 198);
        assert_eq!(STATIC_ALLOC[9][10], 193);
        assert_eq!(STATIC_ALLOC[10][10], 188);
        assert_eq!(STATIC_ALLOC[11][10], 183);
        assert_eq!(STATIC_ALLOC[12][10], 178);
        assert_eq!(STATIC_ALLOC[13][10], 173);
        assert_eq!(STATIC_ALLOC[14][10], 168);
        assert_eq!(STATIC_ALLOC[15][10], 163);
        assert_eq!(STATIC_ALLOC[16][10], 158);
        assert_eq!(STATIC_ALLOC[17][10], 153);
        assert_eq!(STATIC_ALLOC[18][10], 148);
        assert_eq!(STATIC_ALLOC[19][10], 129);
        assert_eq!(STATIC_ALLOC[20][10], 104);
    }

    /// Each row of Table 57 is monotonically non-decreasing in `q`:
    /// higher quality allocates more bits per bin. This is the spec's
    /// implicit invariant that drives the linear-interpolation search.
    #[test]
    fn rows_monotonic_in_q() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            for q in 1..NUM_Q {
                assert!(
                    row[q] >= row[q - 1],
                    "band {band} q={q} ({}) < q={} ({})",
                    row[q],
                    q - 1,
                    row[q - 1]
                );
            }
        }
    }

    /// `frac = 0` recovers the exact lower-column value.
    #[test]
    fn interp_frac_zero_is_lower_column() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            for (qlo, &lo) in row.iter().enumerate().take(NUM_Q - 1) {
                assert_eq!(
                    interp_alloc_1_32nd(band, qlo, 0),
                    Some(lo as u32),
                    "band {band} qlo {qlo}"
                );
            }
        }
    }

    /// Interpolation lands strictly between (or equal to) the two
    /// flanking columns for any sub-column position.
    #[test]
    fn interp_within_bracket() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            for (qlo, pair) in row.windows(2).enumerate() {
                let lo = pair[0] as u32;
                let hi = pair[1] as u32;
                let (min, max) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                for frac in 0..INTERP_STEPS {
                    let v = interp_alloc_1_32nd(band, qlo, frac).expect("valid (band, qlo, frac)");
                    assert!(
                        v >= min && v <= max + 1,
                        "band {band} qlo {qlo} frac {frac}: {v} not in [{min},{max}+1]"
                    );
                }
            }
        }
    }

    /// Interpolation is monotonically non-decreasing in `frac` whenever
    /// the upper column dominates the lower (i.e. the row is rising).
    #[test]
    fn interp_monotonic_when_row_rising() {
        for (band, row) in STATIC_ALLOC.iter().enumerate() {
            for (qlo, pair) in row.windows(2).enumerate() {
                let lo = pair[0] as u32;
                let hi = pair[1] as u32;
                if hi < lo {
                    continue;
                }
                let mut prev = interp_alloc_1_32nd(band, qlo, 0).unwrap();
                for frac in 1..INTERP_STEPS {
                    let v = interp_alloc_1_32nd(band, qlo, frac).unwrap();
                    assert!(v >= prev, "band {band} qlo {qlo} frac {frac}");
                    prev = v;
                }
            }
        }
    }

    /// Out-of-range inputs return `None` rather than panicking.
    #[test]
    fn interp_rejects_out_of_range() {
        assert_eq!(interp_alloc_1_32nd(NUM_BANDS, 0, 0), None);
        assert_eq!(interp_alloc_1_32nd(0, NUM_Q - 1, 0), None);
        assert_eq!(interp_alloc_1_32nd(0, 0, INTERP_STEPS), None);
    }

    /// `band_static_alloc_1_8th` evaluates the RFC formula directly:
    /// `channels * N * alloc << LM >> 2`. Pick a representative
    /// (band 0, q = 5, channels = 1, N = 4, LM = 0): alloc = 134;
    /// `1 * 4 * 134 << 0 >> 2 = 134`.
    #[test]
    fn band_static_alloc_hand_check() {
        // (band 0, qlo = 5, frac = 0) -> alloc = 134
        let v = band_static_alloc_1_8th(0, 5, 0, 1, 4, 0).unwrap();
        assert_eq!(v, 134);
        // Stereo doubles the channels factor.
        let v_stereo = band_static_alloc_1_8th(0, 5, 0, 2, 4, 0).unwrap();
        assert_eq!(v_stereo, 268);
        // LM = 1 doubles via the `<< LM` term.
        let v_lm1 = band_static_alloc_1_8th(0, 5, 0, 1, 4, 1).unwrap();
        assert_eq!(v_lm1, 268);
    }

    /// `band_static_alloc_1_8th` rejects out-of-range channels / LM.
    #[test]
    fn band_static_alloc_rejects_invalid() {
        assert_eq!(band_static_alloc_1_8th(0, 5, 0, 0, 4, 0), None);
        assert_eq!(band_static_alloc_1_8th(0, 5, 0, 3, 4, 0), None);
        assert_eq!(band_static_alloc_1_8th(0, 5, 0, 1, 4, 4), None);
    }

    /// Window sum matches the manual sum of two adjacent bands for a
    /// representative configuration.
    #[test]
    fn window_static_alloc_sums_window() {
        // Bands 0..2 (window of length 2), channels = 1, LM = 0.
        // bins_per_band = [4, 4]; qlo = 5, frac = 0.
        // alloc[0][5] = 134, alloc[1][5] = 127.
        // band 0: 1 * 4 * 134 << 0 >> 2 = 134
        // band 1: 1 * 4 * 127 << 0 >> 2 = 127
        // total: 261
        let bins = [4u32, 4];
        let total = window_static_alloc_1_8th(0, &bins, 5, 0, 1, 0).unwrap();
        assert_eq!(total, 134 + 127);
    }

    /// Hybrid-mode window starting at band 17 walks the right rows of
    /// Table 57. At q = 9 the cells are 63, 56, 45, 20.
    #[test]
    fn window_static_alloc_hybrid_q9() {
        // bins = 4 per band, channels = 1, LM = 0:
        //   band: channels * N * alloc << LM >> 2
        //       = 1 * 4 * alloc << 0 >> 2
        //       = alloc
        // (the `<< LM` cancels with the `>> 2` for this configuration)
        let bins = [4u32, 4, 4, 4];
        let total = window_static_alloc_1_8th(17, &bins, 9, 0, 1, 0).unwrap();
        // alloc cells (qlo = 9 ⇒ STATIC_ALLOC[band][9]):
        //   band 17 -> 63, band 18 -> 56, band 19 -> 45, band 20 -> 20
        // sum = 184
        assert_eq!(total, 63 + 56 + 45 + 20);
    }

    /// A window that overflows `NUM_BANDS` returns `None`.
    #[test]
    fn window_static_alloc_rejects_overflow() {
        let bins = [4u32; 5];
        // coding_start = 17 + 5 = 22 > 21.
        assert_eq!(window_static_alloc_1_8th(17, &bins, 5, 0, 1, 0), None);
    }

    /// The interpolation rounding constant matches the documented
    /// half-step (so the unit tests above can rely on it).
    #[test]
    fn interp_round_is_half_step() {
        assert_eq!(INTERP_ROUND, 32);
        assert_eq!(INTERP_STEPS, 64);
    }

    /// Cross-check the §4.3.3 prose against a representative 5 ms
    /// frame (LM = 1) stereo configuration. Use band 0 (where the
    /// table caps at 200 at q = 10) and bins_per_channel = 8.
    ///
    /// channels * N * alloc << LM >> 2 = 2 * 8 * 200 << 1 >> 2 = 1600.
    #[test]
    fn band_static_alloc_5ms_stereo() {
        let v = band_static_alloc_1_8th(0, 9, 63, 2, 8, 1).unwrap();
        // frac=63 ≈ upper column; with rounding +32 this rounds toward
        // 200 (the upper column for band 0):
        //   weighted = 1*172 + 63*200 + 32 = 172 + 12600 + 32 = 12804
        //   alloc = 12804 / 64 = 200
        // band_static = 2 * 8 * 200 << 1 >> 2 = 6400 >> 2 + shift…
        // 2 * 8 = 16; 16 * 200 = 3200; << 1 = 6400; >> 2 = 1600.
        assert_eq!(v, 1600);
    }

    // -----------------------------------------------------------------
    // §4.3.3 static-allocation search (find_static_alloc) tests.
    // -----------------------------------------------------------------

    /// Search with a generous budget pins to (qlo = NUM_Q-1, frac = 0):
    /// at any q above the highest column there is no further
    /// allocation to interpolate toward.
    #[test]
    fn search_saturates_at_top_column() {
        // Single-band window, band 0 (peaks at q=10 alloc=200), LM=0,
        // channels=1, bins=4 ⇒ q=10 cell is 200. Give the search a
        // huge budget; it pins at qlo=10, frac=0, total=200.
        let bins = [4u32];
        let r = find_static_alloc(0, &bins, 1, 0, u32::MAX / 2).unwrap();
        assert_eq!(r.qlo, NUM_Q - 1);
        assert_eq!(r.frac, 0);
        assert_eq!(r.total_1_8th, 200);
    }

    /// Search at exactly the qlo = 0 column total picks qlo = 0,
    /// frac = 0 (no fractional advance possible because the next
    /// step would exceed).
    #[test]
    fn search_lands_on_q0_when_budget_just_fits() {
        // Band 0, channels=1, bins=4, LM=0. q=0 cell is 0. Set budget
        // to 0: search picks qlo=0, frac=0, total=0 — and that is
        // the highest position that fits.
        let bins = [4u32];
        let r = find_static_alloc(0, &bins, 1, 0, 0).unwrap();
        assert_eq!(r.qlo, 0);
        // frac may step into the rising side of the (0,1) interp; check
        // total instead — it must fit exactly.
        assert_eq!(r.total_1_8th, 0);
    }

    /// Search with a budget that exactly matches a column value
    /// commits to that column and reports the matching total.
    ///
    /// The exact `frac` value is not specified by the RFC: when the
    /// interpolation's integer division produces the same window
    /// total for multiple fractional positions, the §4.3.3 "highest
    /// allocation that does not exceed the bits remaining" contract
    /// allows the search to land on any of them. The invariant the
    /// search promises is `total_1_8th == 134` and `total_1_8th <=
    /// budget`.
    #[test]
    fn search_exact_column_match() {
        // Band 0 row: q=5 cell is 134, q=6 cell is 144. channels=1,
        // bins=4, LM=0 ⇒ band_static = alloc.
        let bins = [4u32];
        let r = find_static_alloc(0, &bins, 1, 0, 134).unwrap();
        assert_eq!(r.qlo, 5);
        assert_eq!(r.total_1_8th, 134);
        // Stepping by one position must strictly exceed budget — the
        // "highest allocation that does not exceed" contract.
        let next_total = if r.frac + 1 == INTERP_STEPS {
            window_static_alloc_at_column_1_8th(0, &bins, r.qlo + 1, 1, 0).unwrap()
        } else {
            window_static_alloc_1_8th(0, &bins, r.qlo, r.frac + 1, 1, 0).unwrap()
        };
        assert!(next_total > 134, "next-step {next_total} should exceed 134");
    }

    /// Search budget strictly between two adjacent columns lands at a
    /// fractional position whose window sum is the highest under
    /// budget.
    #[test]
    fn search_lands_in_fractional_bracket() {
        // Band 0 row, q=5 → 134, q=6 → 144 (delta 10). Budget = 140
        // sits between the two columns. The search should pick qlo=5
        // and the largest frac such that interp <= 140.
        let bins = [4u32];
        let r = find_static_alloc(0, &bins, 1, 0, 140).unwrap();
        assert_eq!(r.qlo, 5);
        // The total must fit (≤ 140) and stepping frac+1 must overrun.
        assert!(r.total_1_8th <= 140);
        // Stepping by one position must strictly exceed the budget.
        let next_total = if r.frac + 1 == INTERP_STEPS {
            window_static_alloc_at_column_1_8th(0, &bins, r.qlo + 1, 1, 0).unwrap()
        } else {
            window_static_alloc_1_8th(0, &bins, r.qlo, r.frac + 1, 1, 0).unwrap()
        };
        assert!(
            next_total > 140,
            "next-step total {next_total} should exceed 140 at frac={}",
            r.frac
        );
    }

    /// Search invariants hold for every budget value: the chosen
    /// total ≤ budget, and stepping by one grid position (one frac,
    /// or wrap to next qlo at frac=INTERP_STEPS) overruns the budget
    /// (unless we are saturated at the top column).
    #[test]
    fn search_invariants_hold_across_budgets() {
        let bins = [4u32];
        // Band 0 row, channels=1, LM=0 ⇒ search-space cells are the
        // raw alloc values. Walk budgets 0..=210 and check.
        for budget in 0..=210u32 {
            let r = find_static_alloc(0, &bins, 1, 0, budget).unwrap();
            assert!(
                r.total_1_8th <= budget,
                "budget {budget}: total {} > budget",
                r.total_1_8th
            );

            // If we are not at the top column, the next-step total
            // (frac+1, or column step) must strictly exceed budget.
            let saturated = r.qlo == NUM_Q - 1;
            if !saturated {
                let next_total = if r.frac + 1 == INTERP_STEPS {
                    // Wrap to next integer column. Use the column
                    // evaluator so the top column is reachable.
                    window_static_alloc_at_column_1_8th(0, &bins, r.qlo + 1, 1, 0).unwrap()
                } else {
                    window_static_alloc_1_8th(0, &bins, r.qlo, r.frac + 1, 1, 0).unwrap()
                };
                assert!(
                    next_total > budget,
                    "budget {budget}: next-step total {next_total} <= budget at (qlo={}, frac={})",
                    r.qlo,
                    r.frac
                );
            }
        }
    }

    /// Multi-band window: budget that is the exact sum at q = 5 lands
    /// at qlo = 5 with the matching total. The exact `frac` value is
    /// unspecified (integer-division collapse may let `frac > 0`
    /// reproduce the same window sum).
    #[test]
    fn search_multi_band_window_exact() {
        // Bands 0..=2, channels=1, bins=4 each, LM=0:
        //   band 0 q=5 -> 134, band 1 q=5 -> 127, band 2 q=5 -> 120
        //   sum = 381.
        let bins = [4u32, 4, 4];
        let r = find_static_alloc(0, &bins, 1, 0, 381).unwrap();
        assert_eq!(r.qlo, 5);
        assert_eq!(r.total_1_8th, 381);
        // Stepping by one position strictly exceeds the budget.
        let next_total = if r.frac + 1 == INTERP_STEPS {
            window_static_alloc_at_column_1_8th(0, &bins, r.qlo + 1, 1, 0).unwrap()
        } else {
            window_static_alloc_1_8th(0, &bins, r.qlo, r.frac + 1, 1, 0).unwrap()
        };
        assert!(next_total > 381);
    }

    /// Hybrid-mode window starting at band 17: a representative budget
    /// search lands inside the expected column.
    #[test]
    fn search_hybrid_window() {
        // bands 17..=20, channels=1, bins=4 each, LM=0:
        //   q=9: 63+56+45+20 = 184
        //   q=10: 153+148+129+104 = 534
        // budget = 184 ⇒ qlo=9, frac=0, total=184.
        let bins = [4u32, 4, 4, 4];
        let r = find_static_alloc(17, &bins, 1, 0, 184).unwrap();
        assert_eq!(r.qlo, 9);
        assert_eq!(r.frac, 0);
        assert_eq!(r.total_1_8th, 184);
    }

    /// Search rejects invalid input the same way the underlying
    /// evaluator does.
    #[test]
    fn search_rejects_invalid_input() {
        let bins = [4u32];
        // channels = 0
        assert_eq!(find_static_alloc(0, &bins, 0, 0, 100), None);
        // lm = 4
        assert_eq!(find_static_alloc(0, &bins, 1, 4, 100), None);
        // window overflow
        let big = [4u32; 5];
        assert_eq!(find_static_alloc(17, &big, 1, 0, 100), None);
    }

    /// Search is monotonic in budget: a larger budget never produces a
    /// smaller `total_1_8th`.
    #[test]
    fn search_monotonic_in_budget() {
        let bins = [4u32, 4, 4];
        let mut prev_total = 0u32;
        for budget in (0..=500u32).step_by(7) {
            let r = find_static_alloc(0, &bins, 1, 0, budget).unwrap();
            assert!(
                r.total_1_8th >= prev_total,
                "budget {budget}: total {} < prev_total {}",
                r.total_1_8th,
                prev_total
            );
            prev_total = r.total_1_8th;
        }
    }

    /// Search reaches frac = INTERP_STEPS as the canonical upper-
    /// column pin when the budget exactly matches the upper column
    /// AND the coarse phase committed to qlo (rather than qlo+1).
    ///
    /// The §4.3.3 prose lets the search land on either side of an
    /// upper-edge boundary; this test pins the behaviour for a budget
    /// that the coarse phase reaches qlo+1 directly.
    #[test]
    fn search_upper_edge_pin_at_exact_column() {
        // Band 0, channels=1, bins=4, LM=0. q=5 cell = 134, q=6 = 144.
        // Budget = 144 ⇒ the coarse phase reaches qlo=6 (since 144 ≤
        // 144). The fine phase commits to the matching total; the
        // total must be 144.
        let bins = [4u32];
        let r = find_static_alloc(0, &bins, 1, 0, 144).unwrap();
        assert_eq!(r.qlo, 6);
        assert_eq!(r.total_1_8th, 144);
        // Stepping by one position strictly exceeds.
        let next_total = if r.frac + 1 == INTERP_STEPS {
            window_static_alloc_at_column_1_8th(0, &bins, r.qlo + 1, 1, 0).unwrap()
        } else {
            window_static_alloc_1_8th(0, &bins, r.qlo, r.frac + 1, 1, 0).unwrap()
        };
        assert!(next_total > 144);
    }

    /// `window_static_alloc_at_column_1_8th` reaches the top column,
    /// unlike the interpolated path which rejects `qlo == NUM_Q - 1`.
    #[test]
    fn column_evaluator_reaches_top_column() {
        let bins = [4u32];
        // Band 0, q=10, channels=1, LM=0:
        //   alloc = 200; total = 1 * 4 * 200 << 0 >> 2 = 200.
        let v = window_static_alloc_at_column_1_8th(0, &bins, NUM_Q - 1, 1, 0).unwrap();
        assert_eq!(v, 200);
    }

    /// `window_static_alloc_at_column_1_8th` rejects q >= NUM_Q.
    #[test]
    fn column_evaluator_rejects_out_of_range_q() {
        let bins = [4u32];
        assert_eq!(
            window_static_alloc_at_column_1_8th(0, &bins, NUM_Q, 1, 0),
            None
        );
    }

    // -----------------------------------------------------------------
    // §4.3.3 per-band interpolated allocation
    // (window_static_alloc_per_band_1_8th) tests.
    // -----------------------------------------------------------------

    /// The per-band breakdown matches `band_static_alloc_1_8th` cell by
    /// cell for a non-top-column position.
    #[test]
    fn per_band_matches_single_band_evaluator() {
        let bins = [4u32, 4, 4];
        let mut out = [0u32; 3];
        assert!(window_static_alloc_per_band_1_8th(
            0, &bins, 5, 0, 1, 0, &mut out
        ));
        for (i, &n) in bins.iter().enumerate() {
            let expect = band_static_alloc_1_8th(i, 5, 0, 1, n, 0).unwrap();
            assert_eq!(out[i], expect, "band {i}");
        }
    }

    /// The per-band vector sums to the scalar window total at the same
    /// grid position (the decomposition invariant).
    #[test]
    fn per_band_sums_to_window_total() {
        let bins = [4u32, 4, 4, 4];
        // Walk a representative grid of (qlo, frac) positions below the
        // top column and confirm the per-band split totals the window
        // sum every time.
        for qlo in 0..NUM_Q - 1 {
            for frac in [0u32, 17, 31, 63] {
                let mut out = [0u32; 4];
                assert!(
                    window_static_alloc_per_band_1_8th(0, &bins, qlo, frac, 1, 0, &mut out),
                    "qlo {qlo} frac {frac}"
                );
                let window = window_static_alloc_1_8th(0, &bins, qlo, frac, 1, 0).unwrap();
                let sum: u32 = out.iter().sum();
                assert_eq!(sum, window, "qlo {qlo} frac {frac}");
            }
        }
    }

    /// The top column (qlo == NUM_Q-1, frac == 0) is reachable and
    /// totals the column evaluator's window sum — the search's saturated
    /// exit position must produce a usable per-band vector.
    #[test]
    fn per_band_top_column_reachable() {
        let bins = [4u32, 4, 4];
        let mut out = [0u32; 3];
        assert!(window_static_alloc_per_band_1_8th(
            0,
            &bins,
            NUM_Q - 1,
            0,
            1,
            0,
            &mut out
        ));
        let window = window_static_alloc_at_column_1_8th(0, &bins, NUM_Q - 1, 1, 0).unwrap();
        let sum: u32 = out.iter().sum();
        assert_eq!(sum, window);
        // Band 0 at q=10 (channels=1, bins=4, LM=0) is alloc 200.
        assert_eq!(out[0], 200);
    }

    /// A non-zero `frac` at the top column is rejected (no sub-column
    /// exists there) and leaves `out` unchanged.
    #[test]
    fn per_band_top_column_rejects_nonzero_frac() {
        let bins = [4u32];
        let mut out = [9999u32];
        assert!(!window_static_alloc_per_band_1_8th(
            0,
            &bins,
            NUM_Q - 1,
            1,
            1,
            0,
            &mut out
        ));
        assert_eq!(out[0], 9999, "out must be untouched on rejection");
    }

    /// Feeding a `StaticAllocSearch` outcome straight back into the
    /// per-band split reproduces the search's committed window total.
    #[test]
    fn per_band_consumes_search_outcome() {
        let bins = [4u32, 4, 4, 4];
        for budget in (0..=600u32).step_by(13) {
            let r = find_static_alloc(0, &bins, 1, 0, budget).unwrap();
            let mut out = [0u32; 4];
            assert!(
                window_static_alloc_per_band_1_8th(0, &bins, r.qlo, r.frac, 1, 0, &mut out),
                "budget {budget}: (qlo={}, frac={})",
                r.qlo,
                r.frac
            );
            let sum: u32 = out.iter().sum();
            assert_eq!(sum, r.total_1_8th, "budget {budget}");
        }
    }

    /// Input-validation rejections all leave `out` unchanged and return
    /// false.
    #[test]
    fn per_band_rejects_invalid_input() {
        let bins = [4u32, 4];
        // Length mismatch.
        let mut short = [0u32; 1];
        assert!(!window_static_alloc_per_band_1_8th(
            0, &bins, 5, 0, 1, 0, &mut short
        ));
        // channels = 0.
        let mut out = [0u32; 2];
        assert!(!window_static_alloc_per_band_1_8th(
            0, &bins, 5, 0, 0, 0, &mut out
        ));
        // lm = 4.
        assert!(!window_static_alloc_per_band_1_8th(
            0, &bins, 5, 0, 1, 4, &mut out
        ));
        // qlo = NUM_Q (out of range).
        assert!(!window_static_alloc_per_band_1_8th(
            0, &bins, NUM_Q, 0, 1, 0, &mut out
        ));
        // frac == INTERP_STEPS below the top column.
        assert!(!window_static_alloc_per_band_1_8th(
            0,
            &bins,
            5,
            INTERP_STEPS,
            1,
            0,
            &mut out
        ));
        // Window overflow (coding_start 17 + 5 bands > 21).
        let big = [4u32; 5];
        let mut wide = [0u32; 5];
        assert!(!window_static_alloc_per_band_1_8th(
            17, &big, 5, 0, 1, 0, &mut wide
        ));
    }

    /// Hybrid-mode window (coding_start = 17) per-band split totals the
    /// window evaluator and walks the correct Table 57 rows.
    #[test]
    fn per_band_hybrid_window() {
        let bins = [4u32, 4, 4, 4];
        let mut out = [0u32; 4];
        assert!(window_static_alloc_per_band_1_8th(
            17, &bins, 9, 0, 1, 0, &mut out
        ));
        // q=9 cells: band 17 -> 63, 18 -> 56, 19 -> 45, 20 -> 20.
        assert_eq!(out, [63, 56, 45, 20]);
        assert_eq!(out.iter().sum::<u32>(), 184);
    }
}
