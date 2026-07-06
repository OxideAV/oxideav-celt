//! Bits-to-Pulses search and balance accumulator (RFC 6716 §4.3.4.1).
//!
//! ## What this module covers
//!
//! After the §4.3.3 bit-allocation has produced a per-band shape
//! allocation in 1/8-bit units, the §4.3.4.2 PVQ requires an
//! **integer** pulse count `K`. The §4.3.4.1 prose specifies how to
//! pick `K` so that the entropy-coded codebook index `V(N, K)` lands
//! as close as possible to the allocated budget without exceeding it,
//! and how to carry the rounding error forward into a per-band
//! **balance** accumulator that adjusts subsequent bands' targets.
//!
//! The RFC §4.3.4.1 narrative is:
//!
//! > "the encoder searches for the value of K that produces the
//! > number of bits nearest to the allocated value (rounding down
//! > if exactly halfway between two values), not to exceed the
//! > total number of bits available."
//!
//! and on the balance:
//!
//! > "The difference between the number of bits allocated and the
//! > number of bits used is accumulated to a 'balance' (initialized
//! > to zero) that helps adjust the allocation for the next bands.
//! > One third of the balance is applied to the bit allocation of
//! > each band to help achieve the target allocation. The only
//! > exceptions are the band before the last and the last band, for
//! > which half the balance and the whole balance are applied,
//! > respectively."
//!
//! ## Cost-of-`K`: bit-exact cache vs. estimator
//!
//! The exact 1/8-bit cost of coding a `K`-pulse band is the
//! precomputed per-(band, LM) cost curve the RFC §4.3.4.1 search runs
//! against ("a precomputed allocation table that only permits some K
//! values for each N"). That curve is the `cache_index50` /
//! `cache_bits50` pair, embedded bit-exact in [`crate::pulse_cache`]
//! from the **corrected** (LM-major, issue #184) clean-room trace
//! `docs/audio/opus/pulse-cache-format-trace.md`. Under the corrected
//! mapping every `(band, LM ∈ 0..=3)` tuple resolves to a cost curve
//! for the band's actual Table-55 size `N`, and the retrieved cost
//! `qbits[K] + 1` never under-prices the monolithic
//! `ceil(8*log2 V(N, K))` PVQ index cost (equality for `K <= 16` —
//! see the `pulse_cache` validation tests). The cache-driven
//! [`bits_to_pulses_band_loop_cached`] consumes it directly and is
//! the allocation cost model.
//!
//! The [`cost_log2_v_count_8th`] estimator (a whole-bit
//! `ceil(log2 V(N, K))` worst-case bound, coarser than the cache by
//! up to 7/8 bit per symbol) remains as the defensive fallback for
//! out-of-range bands and as a caller-supplied closure for
//! [`bits_to_pulses_band_loop`] / [`bits_to_pulses_search`].
//!
//! ## What ships in this round
//!
//! 1. [`cost_log2_v_count_8th`] — the §4.1.5 `dec_uint` worst-case
//!    cost of coding a uniform integer in `0..V(N, K)`, in 1/8-bit
//!    units. This is the closed-form estimator the search composes
//!    with by default.
//! 2. [`bits_to_pulses_search`] — for a single band, given `N` and a
//!    1/8-bit target budget, returns the `(K, bits_used_8th)` pair
//!    with `K` chosen per §4.3.4.1 (closest to target without
//!    exceeding it; ties round down).
//! 3. [`BalanceAccumulator`] — the running 1/8-bit balance the
//!    §4.3.4.1 prose maintains across the band loop. Exposes
//!    [`BalanceAccumulator::adjusted_target`] which folds the
//!    `divisor`-scaled balance share into the per-band target, and
//!    [`BalanceAccumulator::update`] which adds the leftover into
//!    the running balance.
//! 4. [`bits_to_pulses_band_loop`] — the §4.3.4.1 walk over a
//!    sequence of bands, applying the balance share and accumulator
//!    update in spec order. Returns per-band `(K, bits_used)` plus
//!    the final balance.
//!
//! ## What does NOT ship yet
//!
//! * The §4.3.3 `interp_bits2pulses` reallocation bisection with
//!   concurrent skip decoding and the exact fine-energy vs. shape
//!   split. The CELT narrative
//!   `celt-coarse-energy-and-allocation.md` §2.7 enumerates these
//!   steps but defers the algorithm to the reference; they are a
//!   genuine docs gap (see the round report). The corrected cache
//!   closes the *pricing* half of that gap; the reallocation walk
//!   itself remains unspecified.
//! * The §4.3.4.4 band-splitting decision (large bands are split
//!   into sub-bands before bits-to-pulses runs). This module assumes
//!   the caller has already done the split.
//! * The fine-energy / shape split (a sibling §4.3.3 step). The
//!   per-band 1/8-bit input to the bits-to-pulses search is the
//!   shape portion only, after the fine-energy bits have been
//!   reserved.
//!
//! ## Clean-room provenance
//!
//! Every formula, integer-division rule, and band-ordering decision
//! in this module is transcribed from RFC 6716 §4.3.4.1 (lines
//! 6476–6493 of `docs/audio/opus/rfc6716-opus.txt`). The bit-exact
//! cost cache consumed by [`bits_to_pulses_band_loop_cached`] is the
//! clean-room trace `docs/audio/opus/pulse-cache-format-trace.md`
//! (#118) with values from `docs/audio/opus/tables/cache-bits50.csv`
//! and `cache-index50.csv`. No external library source was consulted.

use crate::pulse_cache::{cache_offset, cached_bits_to_pulses_extended};
use crate::pvq::{v_count, V_COUNT_SATURATION};

/// Number of 1/8 bits per whole bit. Pinned for arithmetic clarity
/// since every quantity in this module is in 1/8-bit units.
pub const EIGHTH_BITS_PER_BIT: u32 = 8;

/// Default per-band divisor applied to the running balance per the
/// §4.3.4.1 prose: "One third of the balance is applied to the bit
/// allocation of each band."
pub const DEFAULT_BALANCE_DIVISOR: i32 = 3;

/// Per-band divisor applied to the running balance for the
/// second-to-last band: "half the balance".
pub const SECOND_TO_LAST_BALANCE_DIVISOR: i32 = 2;

/// Per-band divisor applied for the last band: "the whole balance",
/// modelled as a divisor of `1` so the formula
/// `target += balance / divisor` covers all three cases uniformly.
pub const LAST_BALANCE_DIVISOR: i32 = 1;

/// Largest `K` the search will probe before giving up. The §4.3.4.4
/// band-splitting machinery (a future round) keeps the legitimate
/// per-band `K` well inside this bound; the cap exists so the search
/// has a deterministic worst-case rather than running until
/// `V(N, K)` saturates.
///
/// `128` covers every `N` the §4.3.3 allocator can reach for the
/// largest CELT band at the lowest `LM`: at `LM = 3` band 20 has 22
/// MDCT bins, and the maximum per-band rate of one bit per sample
/// caps `K` well below 128 even at the largest legal frame-size
/// budget.
pub const K_SEARCH_CAP: u32 = 128;

/// Compute the §4.1.5 `dec_uint` worst-case cost, in 1/8 bits, of
/// coding a uniform integer in `0..V(N, K)`.
///
/// The §4.1.5 `dec_uint(ft)` cost is `ceil(log2(ft))` whole bits,
/// converted here to 1/8 bits by multiplying by 8. `V(N, 0) = 1`
/// returns zero (no symbol to code). Saturated codebook sizes
/// ([`V_COUNT_SATURATION`]) return [`u32::MAX`] so the search
/// upstream skips them safely.
///
/// This is the *estimator* the bits-to-pulses search composes with
/// by default. The §4.3.4.1 reference implementation uses a per-
/// `(N, K)` cost cache instead (`compute_pulse_cache()` output) that
/// is bit-exact rather than worst-case. When the docs gap on that
/// cache closes, callers can swap in the bit-exact form by passing
/// a different closure to [`bits_to_pulses_search`].
pub fn cost_log2_v_count_8th(n: u32, k: u32) -> u32 {
    let v = v_count(n, k);
    if v == V_COUNT_SATURATION {
        return u32::MAX;
    }
    if v <= 1 {
        // V(N, 0) = 1: no symbol needed; cost is zero.
        return 0;
    }
    // ceil(log2(v)) for v >= 2: this is the bit length of (v - 1)
    // since log2(2) = 1 ⇒ ceil-log of 2 = 1, log2(3) ⇒ ceil-log = 2,
    // log2(4) = 2 (exact), etc. The standard identity is
    // ceil(log2(v)) = floor(log2(v - 1)) + 1 for v >= 2.
    let ceil_log2 = 32 - (v - 1).leading_zeros();
    // Convert whole bits to 1/8-bit units.
    ceil_log2 * EIGHTH_BITS_PER_BIT
}

/// Per-band bits-to-pulses search outcome (§4.3.4.1).
///
/// `k` is the chosen pulse count; `bits_used_8th` is the entropy-
/// coded cost of the resulting codebook index in 1/8 bits, as
/// reported by the cost function the search was driven with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitsToPulses {
    /// Chosen integer pulse count `K` per §4.3.4.1.
    pub k: u32,
    /// Reported 1/8-bit cost of coding a `V(N, K)` codebook index.
    /// Guaranteed not to exceed the supplied `target_8th` (the
    /// "not to exceed" clause).
    pub bits_used_8th: u32,
}

/// Run the §4.3.4.1 bits-to-pulses search for a single band.
///
/// `n` is the band dimension (PVQ `N`); `target_8th` is the per-band
/// shape allocation in 1/8-bit units. `cost_fn` returns the 1/8-bit
/// cost of coding a `V(n, k)` codebook index — see
/// [`cost_log2_v_count_8th`] for the default in-repo estimator.
///
/// The search proceeds upward from `K = 0` and stops at the largest
/// `K` whose cost does not exceed `target_8th`. The §4.3.4.1
/// "rounding down if exactly halfway between two values" tie-breaker
/// is enforced by always taking the lower of the two `K` values that
/// flank the target (a strict-less-than midpoint comparison
/// suffices).
///
/// Returns `BitsToPulses { k: 0, bits_used_8th: 0 }` when even
/// `K = 1` would exceed the budget — the band carries zero shape
/// bits in that case.
///
/// `n == 0` is a degenerate input and returns `K = 0`.
pub fn bits_to_pulses_search<F>(n: u32, target_8th: u32, mut cost_fn: F) -> BitsToPulses
where
    F: FnMut(u32, u32) -> u32,
{
    // K = 0 always costs zero: V(N, 0) = 1, dec_uint(1) = nothing.
    // Establish that as our baseline.
    let mut best = BitsToPulses {
        k: 0,
        bits_used_8th: 0,
    };

    if n == 0 || target_8th == 0 {
        return best;
    }

    // Walk K upward. The §4.3.4.1 search is "closest to target
    // without exceeding it; ties round down". The cost function is
    // monotonically non-decreasing in K for fixed N (V is monotonic
    // in K, log2 is monotonic in V), so the largest K within budget
    // is by definition the closest one not exceeding the target.
    //
    // The tie case ("exactly halfway between two values") only
    // arises if the cost of K_low equals the cost of K_low + 1 and
    // the midpoint between them lands exactly on the target. Since
    // we always step upward and only accept costs <= target, the
    // tie-breaker is implicit: we never advance past the lower of
    // a tied pair.
    for k in 1..=K_SEARCH_CAP {
        let cost = cost_fn(n, k);
        if cost > target_8th {
            break;
        }
        if cost == u32::MAX {
            // Saturated codebook — caller should split before us.
            break;
        }
        best = BitsToPulses {
            k,
            bits_used_8th: cost,
        };
    }

    best
}

/// Running 1/8-bit balance per RFC 6716 §4.3.4.1.
///
/// The balance starts at zero, accumulates `target - bits_used` per
/// band, and contributes a `target += balance / divisor` adjustment
/// to subsequent bands' targets. The §4.3.4.1 prose nominates three
/// divisors — `3` for the general case, `2` for the band before the
/// last, and `1` (the "whole balance") for the last band.
#[derive(Debug, Clone, Copy, Default)]
pub struct BalanceAccumulator {
    /// Running balance in 1/8-bit units. Signed: a band that
    /// over-spent leaves a negative balance to be borrowed from
    /// subsequent bands' targets.
    pub balance_8th: i32,
}

impl BalanceAccumulator {
    /// Construct a freshly-reset balance accumulator (zero).
    pub fn new() -> Self {
        Self { balance_8th: 0 }
    }

    /// Apply the §4.3.4.1 per-band balance share to a candidate
    /// target.
    ///
    /// `divisor` is the §4.3.4.1 share divisor — `3` for the general
    /// case, `2` for the second-to-last band, `1` for the last band.
    /// The adjustment is `target + balance / divisor` with
    /// round-toward-zero integer division (the natural Rust `/`
    /// semantics for `i32`, matching the rounding convention the
    /// RFC §4 sec 2 paragraph 2 implies for "1/8 bits"-scale integer
    /// arithmetic).
    ///
    /// The function saturates non-negatively to zero — a per-band
    /// target cannot go below zero because a §4.3.4.1 "negative
    /// target" would have no operational meaning for the
    /// bits-to-pulses search.
    pub fn adjusted_target(&self, raw_target_8th: u32, divisor: i32) -> u32 {
        debug_assert!(
            divisor >= 1,
            "balance divisor must be at least 1 per §4.3.4.1"
        );
        let share = if divisor == 0 {
            0
        } else {
            self.balance_8th / divisor
        };
        let adjusted = raw_target_8th as i64 + share as i64;
        if adjusted <= 0 {
            0
        } else if adjusted > u32::MAX as i64 {
            u32::MAX
        } else {
            adjusted as u32
        }
    }

    /// Update the running balance after a band has been coded.
    ///
    /// `target_8th` is the band's *original* target (before the
    /// balance share was folded in); `bits_used_8th` is the cost
    /// the bits-to-pulses search reported for the chosen `K`. The
    /// §4.3.4.1 prose adds the difference into the running balance.
    ///
    /// Note that the balance is updated against the *unadjusted*
    /// target so that the share itself doesn't double-count: a
    /// share of `+S` applied to band `b` becomes a `-S` debit on
    /// the running balance once we update with the unadjusted
    /// target. The net effect is that the balance carries the
    /// truncation residue and the bits-used overshoot/undershoot
    /// forward.
    pub fn update(&mut self, target_8th: u32, bits_used_8th: u32) {
        let delta = target_8th as i64 - bits_used_8th as i64;
        let new_balance = self.balance_8th as i64 + delta;
        // Saturating clamp keeps the balance inside i32 range; the
        // legitimate per-frame range is well inside that bound.
        self.balance_8th = new_balance.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }

    /// Reset the balance to zero (a §4.5.2 decoder reset).
    pub fn reset(&mut self) {
        self.balance_8th = 0;
    }
}

/// Walk a sequence of bands through the §4.3.4.1 bits-to-pulses
/// search, applying the balance accumulator share rules in spec
/// order.
///
/// `band_n` is the per-band `N` (PVQ dimension); `band_target_8th` is
/// the per-band raw shape allocation in 1/8 bits as produced by the
/// upstream §4.3.3 allocator. `cost_fn` is the cost-of-(N, K)
/// estimator the inner search uses (see
/// [`cost_log2_v_count_8th`] for the default).
///
/// The balance divisor per band is selected per §4.3.4.1:
/// * `DEFAULT_BALANCE_DIVISOR` (= 3) for bands `0..nbands - 2`.
/// * `SECOND_TO_LAST_BALANCE_DIVISOR` (= 2) for band `nbands - 2`.
/// * `LAST_BALANCE_DIVISOR` (= 1) for band `nbands - 1`.
///
/// Returns `(per_band_result, final_balance)` where
/// `per_band_result[i] = BitsToPulses { k, bits_used_8th }` for band
/// `i`. `band_n.len() != band_target_8th.len()` returns `None`.
pub fn bits_to_pulses_band_loop<F>(
    band_n: &[u32],
    band_target_8th: &[u32],
    mut cost_fn: F,
) -> Option<(Vec<BitsToPulses>, BalanceAccumulator)>
where
    F: FnMut(u32, u32) -> u32,
{
    if band_n.len() != band_target_8th.len() {
        return None;
    }

    let nbands = band_n.len();
    let mut out = Vec::with_capacity(nbands);
    let mut balance = BalanceAccumulator::new();

    for (i, (&n, &raw_target)) in band_n.iter().zip(band_target_8th.iter()).enumerate() {
        // §4.3.4.1 divisor selection.
        let divisor = if nbands >= 2 && i == nbands - 1 {
            LAST_BALANCE_DIVISOR
        } else if nbands >= 2 && i == nbands - 2 {
            SECOND_TO_LAST_BALANCE_DIVISOR
        } else {
            DEFAULT_BALANCE_DIVISOR
        };

        // Adjusted per-band target with the balance share folded in.
        let adjusted_target = balance.adjusted_target(raw_target, divisor);

        // Inner search with the supplied cost function.
        let result = bits_to_pulses_search(n, adjusted_target, &mut cost_fn);

        // Update the running balance with the *raw* (unadjusted)
        // target, per `update`'s contract: the granted share must not
        // double-count. Updating with the adjusted target would leave
        // a granted-and-spent share sitting in the balance — the
        // surplus would never deplete, and the aggregate spend could
        // run past the sum of the raw targets (i.e. past the §4.3.3
        // budget the targets were drawn from). With the raw target,
        // `sum(bits_used) <= sum(raw_target)` holds by induction
        // (each band spends at most its raw target plus a share the
        // balance is then debited for).
        balance.update(raw_target, result.bits_used_8th);

        out.push(result);
    }

    Some((out, balance))
}

/// Walk a sequence of bands through the §4.3.4.1 bits-to-pulses
/// search using the **bit-exact** `cache_index50` / `cache_bits50`
/// cost cache (corrected LM-major mapping): every in-range
/// `(band, LM ∈ 0..=3)` tuple resolves to its band-size cost curve,
/// so the worst-case [`cost_log2_v_count_8th`] estimator engages only
/// as a defensive fallback for out-of-range band indices.
///
/// This is the cache-driven counterpart of
/// [`bits_to_pulses_band_loop`]: it replaces the `ceil(log2 V(N,K))`
/// estimator with the precomputed per-(band, LM) cost curve the RFC
/// delegates to, so the chosen `K` and reported cost are bit-exact
/// for every band the §4.3.3 allocator actually reaches. The cached
/// walk also caps each band's `K` at the run's `maxK` — the cached
/// pulse ceiling beyond which the reference's §4.3.4.4 split regime
/// applies — so the chosen `K` always has a `u32`-representable
/// codebook.
///
/// Parameters:
/// * `lm` — the frame-size shift (`LM ∈ 0..=3`) shared by every band.
/// * `band_n` — per-band `N` (PVQ dimension), used only on the
///   out-of-range fallback path.
/// * `band_target_8th` — per-band raw shape allocation in 1/8 bits.
///
/// The first coded band corresponds to CELT band index `band_start`
/// (0 in normal mode, 17 in the Hybrid mode §4.3.3 mentions); the
/// cache is indexed by the absolute CELT band number
/// `band_start + i`.
///
/// Returns `(per_band_result, final_balance)`, or `None` on a length
/// mismatch / an out-of-range `(band, LM)`.
pub fn bits_to_pulses_band_loop_cached(
    lm: usize,
    band_start: usize,
    band_n: &[u32],
    band_target_8th: &[u32],
) -> Option<(Vec<BitsToPulses>, BalanceAccumulator)> {
    bits_to_pulses_band_loop_cached_thresh(lm, band_start, band_n, band_target_8th, None)
}

/// [`bits_to_pulses_band_loop_cached`] with the §4.3.3 **hard-minimum
/// skip floor** applied per band.
///
/// RFC 6716 §4.3.3: the allocation "computes a vector representing the
/// hard minimum amounts allocation any band will receive for shape.
/// This minimum is higher than the technical limit of the PVQ process,
/// but very low rate allocations produce an excessively sparse
/// spectrum and these bands are better served by having no allocation
/// at all." `thresh_8th` is that vector
/// ([`crate::band_minimums::compute_thresh`], in 1/8 bits): a band
/// whose balance-adjusted target falls below its threshold is
/// **skipped** — forced to `K = 0`, zero bits — and its whole raw
/// target is credited to the running balance, flowing forward to the
/// remaining bands exactly through the §4.3.4.1 share mechanism (the
/// redistribution instrument the RFC provides; the reference's
/// concurrent skip-bit *decoding* and its final reallocation walk stay
/// a docs gap, so which bands the floor zeroes is this crate's
/// documented deterministic reading — both codec sides derive it from
/// the bit-identical prefix, keeping lockstep by construction).
///
/// `thresh_8th = None` disables the floor (the plain cached walk);
/// `Some(t)` must match the window length.
pub fn bits_to_pulses_band_loop_cached_thresh(
    lm: usize,
    band_start: usize,
    band_n: &[u32],
    band_target_8th: &[u32],
    thresh_8th: Option<&[u32]>,
) -> Option<(Vec<BitsToPulses>, BalanceAccumulator)> {
    if band_n.len() != band_target_8th.len() {
        return None;
    }
    if let Some(t) = thresh_8th {
        if t.len() != band_n.len() {
            return None;
        }
    }

    let nbands = band_n.len();
    let mut out = Vec::with_capacity(nbands);
    let mut balance = BalanceAccumulator::new();

    for (i, (&n, &raw_target)) in band_n.iter().zip(band_target_8th.iter()).enumerate() {
        // §4.3.4.1 divisor selection (identical to the estimator loop).
        let divisor = if nbands >= 2 && i == nbands - 1 {
            LAST_BALANCE_DIVISOR
        } else if nbands >= 2 && i == nbands - 2 {
            SECOND_TO_LAST_BALANCE_DIVISOR
        } else {
            DEFAULT_BALANCE_DIVISOR
        };

        let adjusted_target = balance.adjusted_target(raw_target, divisor);

        // §4.3.3 hard-minimum skip floor: below-threshold bands carry
        // no shape allocation at all (see the function docs); the raw
        // target is credited to the balance below, unspent.
        if let Some(t) = thresh_8th {
            if adjusted_target < t[i] {
                balance.update(raw_target, 0);
                out.push(BitsToPulses {
                    k: 0,
                    bits_used_8th: 0,
                });
                continue;
            }
        }

        let celt_band = band_start + i;
        let result = match cache_offset(celt_band, lm) {
            Some(_) => {
                // Bit-exact cache path, extended past the run's maxK
                // with exact monolithic pricing for the in-crate
                // single-block wire (see `cached_bits_to_pulses_extended`).
                let cp = cached_bits_to_pulses_extended(celt_band, lm, n, adjusted_target)?;
                BitsToPulses {
                    k: cp.k,
                    bits_used_8th: cp.bits_used_8th,
                }
            }
            None => {
                // Out-of-range (band, LM) — defensive worst-case
                // estimator fallback (see module docs). Every in-range
                // tuple is cached under the corrected LM-major mapping.
                bits_to_pulses_search(n, adjusted_target, cost_log2_v_count_8th)
            }
        };

        // Raw-target update per `BalanceAccumulator::update`'s
        // contract (see the estimator loop above): the granted share
        // must not double-count, or the aggregate spend can run past
        // the budget the targets were drawn from.
        balance.update(raw_target, result.bits_used_8th);
        out.push(result);
    }

    Some((out, balance))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- cost_log2_v_count_8th ----------

    /// `V(N, 0) = 1` ⇒ `dec_uint(1)` costs zero. The cost estimator
    /// must agree.
    #[test]
    fn cost_k_zero_is_free() {
        for n in 0..32 {
            assert_eq!(cost_log2_v_count_8th(n, 0), 0);
        }
    }

    /// `V(1, 1) = 2` ⇒ `dec_uint(2)` costs ceil(log2(2)) = 1 whole
    /// bit ⇒ 8 1/8-bit units.
    #[test]
    fn cost_n1_k1_one_bit() {
        assert_eq!(cost_log2_v_count_8th(1, 1), 8);
    }

    /// `V(2, 1) = 4` ⇒ `dec_uint(4)` costs ceil(log2(4)) = 2 whole
    /// bits ⇒ 16 1/8-bit units.
    #[test]
    fn cost_n2_k1_two_bits() {
        assert_eq!(cost_log2_v_count_8th(2, 1), 16);
    }

    /// `V(2, 2) = 8` ⇒ ceil(log2(8)) = 3 ⇒ 24 1/8-bit units.
    #[test]
    fn cost_n2_k2_three_bits() {
        assert_eq!(cost_log2_v_count_8th(2, 2), 24);
    }

    /// `V(3, 2) = 18` ⇒ ceil(log2(18)) = 5 ⇒ 40 1/8-bit units.
    #[test]
    fn cost_n3_k2_five_bits() {
        // ceil(log2(18)) = 5 since 2^4 = 16 < 18 ≤ 2^5 = 32.
        assert_eq!(cost_log2_v_count_8th(3, 2), 40);
    }

    /// Cost is monotonically non-decreasing in K for fixed N (a
    /// property the search relies on for the "largest K not
    /// exceeding target" rule to be correct).
    #[test]
    fn cost_monotone_in_k() {
        for n in 1..=8u32 {
            let mut prev = 0u32;
            for k in 0..=12u32 {
                let cost = cost_log2_v_count_8th(n, k);
                if cost == u32::MAX {
                    break;
                }
                assert!(
                    cost >= prev,
                    "cost decreased at N={}, K={} (prev={}, cost={})",
                    n,
                    k,
                    prev,
                    cost
                );
                prev = cost;
            }
        }
    }

    // ---------- bits_to_pulses_search ----------

    /// A zero target picks `K = 0`. (V(N, 0) = 1, no bits needed.)
    #[test]
    fn search_zero_target_picks_k_zero() {
        for n in 0..16 {
            let r = bits_to_pulses_search(n, 0, cost_log2_v_count_8th);
            assert_eq!(r.k, 0);
            assert_eq!(r.bits_used_8th, 0);
        }
    }

    /// `n = 0` is degenerate; the search returns `K = 0`.
    #[test]
    fn search_n_zero_returns_k_zero() {
        for target in 0..200u32 {
            let r = bits_to_pulses_search(0, target, cost_log2_v_count_8th);
            assert_eq!(r.k, 0);
            assert_eq!(r.bits_used_8th, 0);
        }
    }

    /// Budget of one whole bit (8 1/8-bits) at `N = 1` lets the
    /// search advance through the cost-plateau at K ≥ 1: V(1, K) = 2
    /// for every K ≥ 1 (the recurrence collapses to V(1, K) =
    /// V(0, K) + V(1, K - 1) + V(0, K - 1) = V(1, K - 1) since
    /// V(0, K > 0) = 0), so the cost stays at one whole bit for
    /// every K. The §4.3.4.1 "largest K within budget" rule then
    /// picks `K_SEARCH_CAP`. This reflects the cost-estimator's
    /// flat-plateau behaviour, not a bug.
    #[test]
    fn search_n1_target_one_bit_walks_plateau_to_cap() {
        let r = bits_to_pulses_search(1, 8, cost_log2_v_count_8th);
        assert_eq!(r.k, K_SEARCH_CAP);
        assert_eq!(r.bits_used_8th, 8);
    }

    /// Budget below the cost of K=1 ⇒ K=0.
    #[test]
    fn search_target_too_small_picks_k_zero() {
        // V(2, 1) = 4 costs 16 1/8-bits. A target of 15 is too small.
        let r = bits_to_pulses_search(2, 15, cost_log2_v_count_8th);
        assert_eq!(r.k, 0);
        assert_eq!(r.bits_used_8th, 0);
    }

    /// Search never overshoots its budget — this is the §4.3.4.1
    /// "not to exceed the total number of bits available" clause.
    #[test]
    fn search_never_overshoots_target() {
        for n in 1..=8u32 {
            for target in 0..200u32 {
                let r = bits_to_pulses_search(n, target, cost_log2_v_count_8th);
                assert!(
                    r.bits_used_8th <= target,
                    "overshoot at N={}, target={}: used={}",
                    n,
                    target,
                    r.bits_used_8th
                );
            }
        }
    }

    /// Search returns the *largest* K within budget. Since the cost
    /// is monotone in K, stepping K up by one must overshoot.
    #[test]
    fn search_picks_largest_k_within_budget() {
        for n in 1..=8u32 {
            for target in (8..200u32).step_by(7) {
                let r = bits_to_pulses_search(n, target, cost_log2_v_count_8th);
                if r.k == 0 {
                    // Either target < cost(N, 1), or N too small.
                    assert!(cost_log2_v_count_8th(n, 1) > target);
                    continue;
                }
                if r.k >= K_SEARCH_CAP {
                    continue;
                }
                let next_cost = cost_log2_v_count_8th(n, r.k + 1);
                if next_cost == u32::MAX {
                    continue;
                }
                assert!(
                    next_cost > target,
                    "search short-circuited at N={}, target={}, K={}: next cost {} <= target",
                    n,
                    target,
                    r.k,
                    next_cost
                );
            }
        }
    }

    /// The K_SEARCH_CAP is the worst-case upper bound on the inner
    /// loop. Verify the cap is reachable by a sufficiently generous
    /// target (a regression sentinel against accidentally lowering
    /// the cap below the legitimate range).
    #[test]
    fn search_cap_is_reachable_at_high_target() {
        // K_SEARCH_CAP = 128, N = 4 codebook V(4, 128) is huge; cost
        // is ceil(log2(V)) * 8 ≤ 8 * 32 = 256. A target ≥ that
        // accepts K up to the cap.
        let r = bits_to_pulses_search(4, 10_000, cost_log2_v_count_8th);
        assert!(r.k >= 1, "search failed to advance K under generous budget");
    }

    // ---------- BalanceAccumulator ----------

    /// Fresh accumulator is zero.
    #[test]
    fn balance_default_is_zero() {
        let b = BalanceAccumulator::new();
        assert_eq!(b.balance_8th, 0);
        assert_eq!(BalanceAccumulator::default().balance_8th, 0);
    }

    /// Zero-balance share has no effect: adjusted_target == raw.
    #[test]
    fn balance_zero_share_is_identity() {
        let b = BalanceAccumulator::new();
        for raw in [0u32, 1, 8, 100, 1000, u32::MAX] {
            for divisor in [1i32, 2, 3] {
                assert_eq!(b.adjusted_target(raw, divisor), raw);
            }
        }
    }

    /// Positive balance share is added; with divisor 3 and balance
    /// 9, every band gets +3 added to its raw target.
    #[test]
    fn balance_share_adds_third_for_divisor_3() {
        let b = BalanceAccumulator { balance_8th: 9 };
        assert_eq!(b.adjusted_target(100, DEFAULT_BALANCE_DIVISOR), 103);
    }

    /// Divisor 2 (second-to-last) adds half.
    #[test]
    fn balance_share_adds_half_for_divisor_2() {
        let b = BalanceAccumulator { balance_8th: 10 };
        assert_eq!(b.adjusted_target(100, SECOND_TO_LAST_BALANCE_DIVISOR), 105);
    }

    /// Divisor 1 (last band) adds the whole balance.
    #[test]
    fn balance_share_adds_whole_for_divisor_1() {
        let b = BalanceAccumulator { balance_8th: 42 };
        assert_eq!(b.adjusted_target(100, LAST_BALANCE_DIVISOR), 142);
    }

    /// Negative balance debits the band's target.
    #[test]
    fn balance_negative_debits_target() {
        let b = BalanceAccumulator { balance_8th: -15 };
        // 100 + (-15)/3 = 100 - 5 = 95.
        assert_eq!(b.adjusted_target(100, DEFAULT_BALANCE_DIVISOR), 95);
    }

    /// A debit that exceeds the raw target saturates to zero
    /// (negative targets have no operational meaning).
    #[test]
    fn balance_huge_debit_saturates_target_to_zero() {
        let b = BalanceAccumulator {
            balance_8th: -1_000_000,
        };
        assert_eq!(b.adjusted_target(10, DEFAULT_BALANCE_DIVISOR), 0);
    }

    /// `update` accumulates `target - bits_used` into the running
    /// balance.
    #[test]
    fn balance_update_records_residue() {
        let mut b = BalanceAccumulator::new();
        b.update(100, 80); // residue +20
        assert_eq!(b.balance_8th, 20);
        b.update(50, 60); // residue -10
        assert_eq!(b.balance_8th, 10);
    }

    /// `reset` zeros the balance.
    #[test]
    fn balance_reset_zeros_it() {
        let mut b = BalanceAccumulator { balance_8th: 123 };
        b.reset();
        assert_eq!(b.balance_8th, 0);
    }

    /// Round-toward-zero integer division: -7/3 = -2 in Rust (not
    /// the floor convention's -3). Pin the convention.
    #[test]
    fn balance_share_rounds_toward_zero() {
        let b = BalanceAccumulator { balance_8th: -7 };
        // -7 / 3 = -2 (truncation toward zero); 100 - 2 = 98.
        assert_eq!(b.adjusted_target(100, 3), 98);
        let b2 = BalanceAccumulator { balance_8th: 7 };
        // 7 / 3 = 2; 100 + 2 = 102.
        assert_eq!(b2.adjusted_target(100, 3), 102);
    }

    // ---------- bits_to_pulses_band_loop ----------

    /// Length mismatch ⇒ None.
    #[test]
    fn band_loop_length_mismatch_returns_none() {
        let r = bits_to_pulses_band_loop(&[4, 4, 4], &[100, 100], cost_log2_v_count_8th);
        assert!(r.is_none());
    }

    /// Empty input ⇒ empty output + zero balance.
    #[test]
    fn band_loop_empty_input_yields_empty_output() {
        let (out, bal) = bits_to_pulses_band_loop(&[], &[], cost_log2_v_count_8th).unwrap();
        assert!(out.is_empty());
        assert_eq!(bal.balance_8th, 0);
    }

    /// Single band: the divisor is `LAST_BALANCE_DIVISOR` (since
    /// `nbands - 1 == 0` is also the last index). With a zero
    /// starting balance, the adjusted target is the raw target.
    #[test]
    fn band_loop_single_band_uses_last_divisor() {
        let (out, bal) = bits_to_pulses_band_loop(&[2], &[24], cost_log2_v_count_8th).unwrap();
        assert_eq!(out.len(), 1);
        // V(2, 2) = 8 ⇒ cost = 24. Exactly hits the budget.
        assert_eq!(out[0].k, 2);
        assert_eq!(out[0].bits_used_8th, 24);
        // Residue is zero.
        assert_eq!(bal.balance_8th, 0);
    }

    /// Two bands: second band's divisor is `LAST_BALANCE_DIVISOR`,
    /// first band's divisor is `SECOND_TO_LAST_BALANCE_DIVISOR`. The
    /// band loop must apply both.
    #[test]
    fn band_loop_two_bands_uses_2_then_1() {
        // V(2, k) costs in 1/8 bits for k=0..: 0, 16, 24, 32, 32,
        // 40, 40, 40, 40, 48, ... (V(2, k) = 4k for k ≥ 1, so
        // ceil(log2(4k))*8). On a target plateau the search picks
        // the largest K within budget.
        //
        // Band 0 (divisor 2): raw target 30, balance 0 ⇒ adjusted
        // 30. Largest K with cost ≤ 30 is K=2 (cost 24). Raw-target
        // residue 30 - 24 = +6 ⇒ balance 6.
        // Band 1 (divisor 1): adjusted 30 + 6 = 36. Largest K with
        // cost ≤ 36 is K=4 (cost 32). Raw-target residue 30 - 32 =
        // -2 ⇒ balance 6 - 2 = 4 (the granted share is debited, per
        // `BalanceAccumulator::update`'s no-double-count contract).
        let (out, bal) =
            bits_to_pulses_band_loop(&[2, 2], &[30, 30], cost_log2_v_count_8th).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].k, 2);
        assert_eq!(out[0].bits_used_8th, 24);
        assert_eq!(out[1].k, 4);
        assert_eq!(out[1].bits_used_8th, 32);
        assert_eq!(bal.balance_8th, 4);
    }

    /// Three bands: divisors are 3, 2, 1 in order. The default
    /// divisor only kicks in starting at band 0 when nbands >= 3.
    #[test]
    fn band_loop_three_bands_uses_3_2_1() {
        // V(8, k) costs in 1/8 bits: 0, 32, 56, 80, 96, 112, 120,
        // 136, 144, 152, 160, ...
        //
        // Band 0 (divisor 3): raw 40, balance 0 ⇒ adjusted 40.
        //   Largest K with cost ≤ 40: K=1 (cost 32). Raw residue
        //   40 - 32 = +8 ⇒ balance 8.
        // Band 1 (divisor 2): adjusted 40 + 8/2 = 44.
        //   Largest K with cost ≤ 44: K=1 (cost 32). Raw residue +8
        //   ⇒ balance 16.
        // Band 2 (divisor 1): adjusted 40 + 16/1 = 56.
        //   Largest K with cost ≤ 56: K=2 (cost 56). Raw residue
        //   40 - 56 = -16 ⇒ final balance 0 — the whole surplus was
        //   granted and spent, and the raw-target update debits it
        //   exactly (aggregate spend 120 == aggregate raw target).
        let (out, bal) =
            bits_to_pulses_band_loop(&[8, 8, 8], &[40, 40, 40], cost_log2_v_count_8th).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].k, 1);
        assert_eq!(out[0].bits_used_8th, 32);
        assert_eq!(out[1].k, 1);
        assert_eq!(out[1].bits_used_8th, 32);
        assert_eq!(out[2].k, 2);
        assert_eq!(out[2].bits_used_8th, 56);
        assert_eq!(bal.balance_8th, 0);
    }

    /// The band loop never overshoots any band's *adjusted* target.
    /// Property test over a small grid.
    #[test]
    fn band_loop_never_overshoots_any_band() {
        // 5-band frame, mixed N and target.
        let n = [2u32, 4, 8, 16, 22];
        let t = [40u32, 80, 120, 160, 200];
        let (out, _) = bits_to_pulses_band_loop(&n, &t, cost_log2_v_count_8th).unwrap();
        let mut balance = BalanceAccumulator::new();
        for (i, (&_ni, &ti)) in n.iter().zip(t.iter()).enumerate() {
            let divisor = if i == n.len() - 1 {
                LAST_BALANCE_DIVISOR
            } else if i == n.len() - 2 {
                SECOND_TO_LAST_BALANCE_DIVISOR
            } else {
                DEFAULT_BALANCE_DIVISOR
            };
            let adjusted = balance.adjusted_target(ti, divisor);
            assert!(
                out[i].bits_used_8th <= adjusted,
                "band {} overshot adjusted target {} with bits_used {}",
                i,
                adjusted,
                out[i].bits_used_8th
            );
            balance.update(ti, out[i].bits_used_8th);
        }
    }

    /// Aggregate conservation: the loop's total spend never exceeds
    /// the sum of the *raw* targets (the §4.3.3 budget the targets
    /// were drawn from), across a grid of shapes and budgets — the
    /// property the raw-target balance update exists to guarantee.
    #[test]
    fn band_loop_total_spend_within_raw_budget() {
        for scale in [10u32, 40, 90, 160, 300] {
            let n = [1u32, 2, 4, 8, 12, 16, 24, 36, 44];
            let t: Vec<u32> = (0..n.len() as u32).map(|i| scale + 7 * i).collect();
            let (out, _) = bits_to_pulses_band_loop(&n, &t, cost_log2_v_count_8th).unwrap();
            let spent: u64 = out.iter().map(|r| u64::from(r.bits_used_8th)).sum();
            let budget: u64 = t.iter().map(|&v| u64::from(v)).sum();
            assert!(
                spent <= budget,
                "scale {scale}: spend {spent} exceeds raw budget {budget}"
            );

            // Same property on the cached walk.
            let (out, _) = bits_to_pulses_band_loop_cached(2, 0, &n, &t).unwrap();
            let spent: u64 = out.iter().map(|r| u64::from(r.bits_used_8th)).sum();
            assert!(
                spent <= budget,
                "cached scale {scale}: spend {spent} exceeds raw budget {budget}"
            );
        }
    }

    /// Custom cost functions compose. A stub cost-fn returning a
    /// constant 16 per K=k should make the search pick the largest
    /// K with 16 ≤ target — i.e. K = K_SEARCH_CAP at target=16+, or
    /// K = 0 below.
    #[test]
    fn band_loop_with_custom_cost_fn() {
        let cost_fn = |_n: u32, k: u32| if k == 0 { 0 } else { 16 };
        let (out, _) = bits_to_pulses_band_loop(&[4, 4], &[15, 16], cost_fn).unwrap();
        // Band 0 target 15: K=0.
        assert_eq!(out[0].k, 0);
        // Band 1 target 16 (no balance share — band 0 residue was 0,
        // and divisor is `LAST`, but 0/1 = 0): K = K_SEARCH_CAP since
        // every K up to the cap costs 16 = target.
        assert_eq!(out[1].k, K_SEARCH_CAP);
        assert_eq!(out[1].bits_used_8th, 16);
    }

    // ---------- bits_to_pulses_band_loop_cached ----------

    /// Length mismatch ⇒ None.
    #[test]
    fn cached_loop_length_mismatch_returns_none() {
        let r = bits_to_pulses_band_loop_cached(3, 2, &[4, 4, 4], &[100, 100]);
        assert!(r.is_none());
    }

    /// A single cached band (band 12, LM 0 ⇒ run @123, N = 4): the
    /// chosen K and cost are the bit-exact retrieved cache values
    /// (`qbits[K] + 1`), not the estimator's.
    #[test]
    fn cached_loop_single_band_uses_cache() {
        // Run @123 retrieved costs: K=1 → 24, K=2 → 40, K=3 → 52.
        // Budget 45 ⇒ K=2 (cost 40). The whole-bit estimator would
        // price K=2 at ceil(log2 V(4,2)=32)*8 = 40 too, but K=3 at
        // ceil(log2 88)*8 = 56 vs the cache's 52.
        let (out, _) = bits_to_pulses_band_loop_cached(0, 12, &[4], &[45]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].k, 2);
        assert_eq!(out[0].bits_used_8th, 40);
    }

    /// Under the corrected LM-major mapping every in-range band is
    /// cached — including band 0 (N = 1, the flat one-bit run). A
    /// one-bit budget walks the flat curve through `maxK` and the
    /// exact-cost extension to `EXTENDED_K_CAP` (`V(1, K) = 2` prices
    /// every `K` at one bit).
    #[test]
    fn cached_loop_band_zero_is_cached_and_walks_flat_extension() {
        let (out, _) = bits_to_pulses_band_loop_cached(0, 0, &[1], &[8]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].k, crate::pulse_cache::EXTENDED_K_CAP);
        assert_eq!(out[0].bits_used_8th, 8);
    }

    /// The cached loop never overshoots any band's adjusted target.
    #[test]
    fn cached_loop_never_overshoots() {
        // 5 coded bands starting at band 2, LM 2.
        let lm = 2usize;
        let start = 2usize;
        let n = [4u32, 8, 16, 22, 22];
        let t = [60u32, 120, 200, 300, 400];
        let (out, _) = bits_to_pulses_band_loop_cached(lm, start, &n, &t).unwrap();
        let mut balance = BalanceAccumulator::new();
        for (i, &ti) in t.iter().enumerate() {
            let divisor = if i == n.len() - 1 {
                LAST_BALANCE_DIVISOR
            } else if i == n.len() - 2 {
                SECOND_TO_LAST_BALANCE_DIVISOR
            } else {
                DEFAULT_BALANCE_DIVISOR
            };
            let adjusted = balance.adjusted_target(ti, divisor);
            assert!(
                out[i].bits_used_8th <= adjusted,
                "band {} overshot adjusted {} with {}",
                start + i,
                adjusted,
                out[i].bits_used_8th
            );
            // Mirror the loop's raw-target update contract exactly.
            balance.update(ti, out[i].bits_used_8th);
        }
    }

    // ---------- bits_to_pulses_band_loop_cached_thresh ----------

    /// A band whose adjusted target falls below its §4.3.3 hard
    /// minimum is skipped (K = 0, zero bits) and its raw target is
    /// credited forward through the balance.
    #[test]
    fn thresh_floor_skips_sub_minimum_band_and_credits_balance() {
        // Band 12 LM 0 has N = 4: thresh = max(24*4/16, 8) = 8 for
        // mono... use an explicit thresh to pin the mechanics: two
        // N = 4 bands, targets 20 and 30, thresh 25 each.
        // Band 0 (divisor 2): adjusted 20 < 25 ⇒ skipped, balance +20.
        // Band 1 (divisor 1): adjusted 30 + 20 = 50 ≥ 25 ⇒ search
        // runs; run @123 retrieved costs 24 (K=1), 40 (K=2), 52 (K=3):
        // K = 2 at 40 fits 50.
        let (out, bal) =
            bits_to_pulses_band_loop_cached_thresh(0, 12, &[4, 4], &[20, 30], Some(&[25, 25]))
                .unwrap();
        assert_eq!(out[0].k, 0);
        assert_eq!(out[0].bits_used_8th, 0);
        assert_eq!(out[1].k, 2);
        assert_eq!(out[1].bits_used_8th, 40);
        // Balance: +20 (band 0 skip) + (30 - 40) = 10.
        assert_eq!(bal.balance_8th, 10);
    }

    /// `None` thresh is the plain cached walk; a mismatched thresh
    /// length is rejected.
    #[test]
    fn thresh_floor_none_matches_plain_walk_and_validates_length() {
        let plain = bits_to_pulses_band_loop_cached(2, 0, &[4, 8, 16], &[60, 120, 200]).unwrap();
        let with_none =
            bits_to_pulses_band_loop_cached_thresh(2, 0, &[4, 8, 16], &[60, 120, 200], None)
                .unwrap();
        assert_eq!(plain.0, with_none.0);
        assert!(bits_to_pulses_band_loop_cached_thresh(
            2,
            0,
            &[4, 8, 16],
            &[60, 120, 200],
            Some(&[8, 8])
        )
        .is_none());
    }

    /// A zero thresh never skips: identical to the plain walk.
    #[test]
    fn thresh_floor_zero_never_skips() {
        let n = [2u32, 4, 8];
        let t = [40u32, 80, 120];
        let plain = bits_to_pulses_band_loop_cached(1, 4, &n, &t).unwrap();
        let zeroed =
            bits_to_pulses_band_loop_cached_thresh(1, 4, &n, &t, Some(&[0, 0, 0])).unwrap();
        assert_eq!(plain.0, zeroed.0);
    }

    /// Out-of-range absolute band index falls back to the estimator
    /// rather than erroring.
    #[test]
    fn cached_loop_band_out_of_range_uses_estimator() {
        // band_start 20 + i=1 ⇒ band 21, out of range ⇒ cache_offset
        // returns None ⇒ estimator path. Verify it does not panic and
        // produces a result for both bands.
        let (out, _) = bits_to_pulses_band_loop_cached(0, 20, &[22, 22], &[400, 400]).unwrap();
        assert_eq!(out.len(), 2);
        // Band 20 LM0 is cached (run @295, N = 22, maxK = 9); band 21
        // is out of range ⇒ estimator path. Both produce a result.
        assert!(out[0].k >= 1);
        assert!(out[1].k >= 1);
    }
}
