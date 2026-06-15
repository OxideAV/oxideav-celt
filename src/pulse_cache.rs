//! Bit-exact pulse-cost cache for §4.3.4.1 bits-to-pulses inversion.
//!
//! ## What this module covers
//!
//! RFC 6716 §4.3.4.1 converts a per-band 1/8-bit budget into an
//! integer pulse count `K` by searching "a precomputed allocation
//! table that only permits some K values for each N" for the largest
//! `K` whose entropy cost does not exceed the budget. That table is
//! the `cache_index50` / `cache_bits50` pair: a 105-entry per-(band,
//! LM) offset table indexing a 392-byte packed sequence of per-band
//! cost curves.
//!
//! This module embeds both tables as factual numeric data and exposes
//! the walk that turns a `(band, LM, budget)` triple into a `(K,
//! bits-used)` pair — bit-exact rather than the worst-case
//! `ceil(log2 V(N, K))` estimator carried in [`crate::bits_to_pulses`]
//! for the uncached path.
//!
//! ## Layout (per the clean-room trace)
//!
//! `cache_index50` has `21 bands × 5 LM = 105` entries in **band-major
//! order**: entry `band*5 + LM` holds the byte offset into
//! `cache_bits50` where that tuple's cost curve begins, or `-1` when
//! the tuple is small enough that the allocator uses a closed-form
//! path instead of the cache.
//!
//! A cost curve ("run") at offset `off` is:
//!
//! ```text
//! cache_bits50[off]         = maxK      (largest K this run supports)
//! cache_bits50[off + 1]     = qbits[1]  (cost for K = 1, 1/8-bit units)
//! ...
//! cache_bits50[off + maxK]  = qbits[maxK]
//! ```
//!
//! `K = 0` is implicit (cost 0, never stored). `qbits[K]` is the cost
//! in 1/8-bit fractional units (the `BITRES = 3` convention) and is
//! monotonically non-decreasing in `K` within a run. Multiple (band,
//! LM) tuples that share a cost profile share one run, so the 105
//! offsets resolve to only 23 distinct runs.
//!
//! ## Clean-room provenance
//!
//! Every numeric value, the band-major indexing, and the run format
//! are transcribed from the clean-room trace
//! `docs/audio/opus/pulse-cache-format-trace.md` (#118) and the
//! Feist-facts numeric extracts
//! `docs/audio/opus/tables/cache-index50.csv` (+ `.meta`) and
//! `docs/audio/opus/tables/cache-bits50.csv` (+ `.meta`), with the
//! §4.3.4.1 search prose from
//! `docs/audio/opus/rfc6716-opus.txt` lines 6476–6493. No external
//! library source was consulted.

/// Number of CELT nominal bands (RFC 6716 §4.3.2 boundary table).
pub const NUM_BANDS: usize = 21;

/// Number of distinct `LM` (frame-size shift) values the cache spans.
/// `LM ∈ {0,1,2,3,4}` selects 120/240/480/960-sample frames plus the
/// transient short-block packing variant.
pub const NUM_LM: usize = 5;

/// Per-(band, LM) offset table into [`CACHE_BITS50`], band-major:
/// `CACHE_INDEX50[band*5 + LM]`. A value of `-1` is the sentinel
/// "no cached cost — use the closed-form path".
///
/// Numeric values from `docs/audio/opus/tables/cache-index50.csv`;
/// band-major layout from the trace §2.
pub const CACHE_INDEX50: [i16; NUM_BANDS * NUM_LM] = [
    -1, -1, -1, -1, -1, // band 0
    -1, -1, -1, 0, 0, // band 1
    0, 0, 41, 41, 41, // band 2
    82, 82, 123, 164, 200, // band 3
    222, 0, 0, 0, 0, // band 4
    0, 0, 0, 0, 41, // band 5
    41, 41, 41, 123, 123, // band 6
    123, 164, 164, 240, 266, // band 7
    283, 295, 41, 41, 41, // band 8
    41, 41, 41, 41, 41, // band 9
    123, 123, 123, 123, 240, // band 10
    240, 240, 266, 266, 305, // band 11
    318, 328, 336, 123, 123, // band 12
    123, 123, 123, 123, 123, // band 13
    123, 240, 240, 240, 240, // band 14
    305, 305, 305, 318, 318, // band 15
    343, 351, 358, 364, 240, // band 16
    240, 240, 240, 240, 240, // band 17
    240, 240, 305, 305, 305, // band 18
    305, 343, 343, 343, 351, // band 19
    351, 370, 376, 382, 387, // band 20
];

/// Packed run-encoded per-band cost curves, 392 bytes.
/// Numeric values from `docs/audio/opus/tables/cache-bits50.csv`.
pub const CACHE_BITS50: [u8; 392] = [
    // run @ 0  (maxK = 40)
    40, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, //
    // run @ 41 (maxK = 40)
    40, 15, 23, 28, 31, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 47, 49, 50, 51, 52, 53, 54, 55,
    55, 57, 58, 59, 60, 61, 62, 63, 63, 65, 66, 67, 68, 69, 70, 71, 71, //
    // run @ 82 (maxK = 40)
    40, 20, 33, 41, 48, 53, 57, 61, 64, 66, 69, 71, 73, 75, 76, 78, 80, 82, 85, 87, 89, 91, 92, 94,
    96, 98, 101, 103, 105, 107, 108, 110, 112, 114, 117, 119, 121, 123, 124, 126, 128, //
    // run @ 123 (maxK = 40)
    40, 23, 39, 51, 60, 67, 73, 79, 83, 87, 91, 94, 97, 100, 102, 105, 107, 111, 115, 118, 121, 124,
    126, 129, 131, 135, 139, 142, 145, 148, 150, 153, 155, 159, 163, 166, 169, 172, 174, 177,
    179, //
    // run @ 164 (maxK = 35)
    35, 28, 49, 65, 78, 89, 99, 107, 114, 120, 126, 132, 136, 141, 145, 149, 153, 159, 165, 171,
    176, 180, 185, 189, 192, 199, 205, 211, 216, 220, 225, 229, 232, 239, 245, 251, //
    // run @ 200 (maxK = 21)
    21, 33, 58, 79, 97, 112, 125, 137, 148, 157, 166, 174, 182, 189, 195, 201, 207, 217, 227, 235,
    243, 251, //
    // run @ 222 (maxK = 17)
    17, 35, 63, 86, 106, 123, 139, 152, 165, 177, 187, 197, 206, 214, 222, 230, 237, 250, //
    // run @ 240 (maxK = 25)
    25, 31, 55, 75, 91, 105, 117, 128, 138, 146, 154, 161, 168, 174, 180, 185, 190, 200, 208, 215,
    222, 229, 235, 240, 245, 255, //
    // run @ 266 (maxK = 16)
    16, 36, 65, 89, 110, 128, 144, 159, 173, 185, 196, 207, 217, 226, 234, 242, 250, //
    // run @ 283 (maxK = 11)
    11, 41, 74, 103, 128, 151, 172, 191, 209, 225, 241, 255, //
    // run @ 295 (maxK = 9)
    9, 43, 79, 110, 138, 163, 186, 207, 227, 246, //
    // run @ 305 (maxK = 12)
    12, 39, 71, 99, 123, 144, 164, 182, 198, 214, 228, 241, 253, //
    // run @ 318 (maxK = 9)
    9, 44, 81, 113, 142, 168, 192, 214, 235, 255, //
    // run @ 328 (maxK = 7)
    7, 49, 90, 127, 160, 191, 220, 247, //
    // run @ 336 (maxK = 6)
    6, 51, 95, 134, 170, 203, 234, //
    // run @ 343 (maxK = 7)
    7, 47, 87, 123, 155, 184, 212, 237, //
    // run @ 351 (maxK = 6)
    6, 52, 97, 137, 174, 208, 240, //
    // run @ 358 (maxK = 5)
    5, 57, 106, 151, 192, 231, //
    // run @ 364 (maxK = 5)
    5, 59, 111, 158, 202, 243, //
    // run @ 370 (maxK = 5)
    5, 55, 103, 147, 187, 224, //
    // run @ 376 (maxK = 5)
    5, 60, 113, 161, 206, 248, //
    // run @ 382 (maxK = 4)
    4, 65, 122, 175, 224, //
    // run @ 387 (maxK = 4)
    4, 67, 127, 182, 234,
];

/// Look up the byte offset of the cost run for a `(band, LM)` tuple,
/// or `None` for the `-1` sentinel (closed-form path).
///
/// `band` must be `< NUM_BANDS` and `lm` must be `< NUM_LM`; out-of-
/// range inputs return `None`.
#[inline]
pub fn cache_offset(band: usize, lm: usize) -> Option<usize> {
    if band >= NUM_BANDS || lm >= NUM_LM {
        return None;
    }
    let off = CACHE_INDEX50[band * NUM_LM + lm];
    if off < 0 {
        None
    } else {
        Some(off as usize)
    }
}

/// The largest `K` the cached run for `(band, LM)` supports, or `None`
/// for the sentinel.
#[inline]
pub fn cache_max_k(band: usize, lm: usize) -> Option<u32> {
    cache_offset(band, lm).map(|off| u32::from(CACHE_BITS50[off]))
}

/// The bit-exact 1/8-bit cost of coding `K` pulses for `(band, LM)`
/// from the cache, or `None` when the tuple is a sentinel or `K`
/// exceeds the run's `maxK`.
///
/// `K = 0` is free (cost 0) for every cached tuple. For `1 ≤ K ≤
/// maxK` the cost is `cache_bits50[off + K]`.
#[inline]
pub fn cache_cost_8th(band: usize, lm: usize, k: u32) -> Option<u32> {
    let off = cache_offset(band, lm)?;
    if k == 0 {
        return Some(0);
    }
    let max_k = u32::from(CACHE_BITS50[off]);
    if k > max_k {
        return None;
    }
    Some(u32::from(CACHE_BITS50[off + k as usize]))
}

/// Result of a cached bits-to-pulses inversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedPulses {
    /// Chosen integer pulse count `K`.
    pub k: u32,
    /// Bit-exact 1/8-bit cost of coding `K` pulses (0 when `k == 0`).
    pub bits_used_8th: u32,
}

/// Invert a per-band 1/8-bit budget into the largest cached `K` whose
/// cost does not exceed it, for the `(band, LM)` tuple.
///
/// Implements the §4.3.4.1 inner loop (trace §7): walk `K = 1 ..
/// maxK`, stop at the first `K` whose cost exceeds `budget_8th`, and
/// take the last `K` that fit. `K = maxK` is permitted. Returns
/// `None` for sentinel tuples — the caller must take the closed-form
/// path for those.
pub fn cached_bits_to_pulses(band: usize, lm: usize, budget_8th: u32) -> Option<CachedPulses> {
    let off = cache_offset(band, lm)?;
    let max_k = usize::from(CACHE_BITS50[off]);
    let mut best = CachedPulses {
        k: 0,
        bits_used_8th: 0,
    };
    for k in 1..=max_k {
        let cost = u32::from(CACHE_BITS50[off + k]);
        if cost > budget_8th {
            break;
        }
        best = CachedPulses {
            k: k as u32,
            bits_used_8th: cost,
        };
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded tables must match the staged corpus sizes exactly.
    #[test]
    fn table_sizes_match_corpus() {
        assert_eq!(CACHE_INDEX50.len(), 105);
        assert_eq!(CACHE_BITS50.len(), 392);
    }

    /// Exactly 8 sentinels, per the trace §5: all of band 0 (5) plus
    /// band 1 LM0..2 (3).
    #[test]
    fn sentinel_count_and_placement() {
        let sentinels: Vec<usize> = CACHE_INDEX50
            .iter()
            .enumerate()
            .filter(|(_, &v)| v < 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(sentinels.len(), 8);
        // band 0: indices 0..5, band 1 LM0..2: indices 5,6,7.
        assert_eq!(sentinels, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        for lm in 0..NUM_LM {
            assert_eq!(cache_offset(0, lm), None, "band 0 LM {lm} not sentinel");
        }
        for lm in 0..3 {
            assert_eq!(cache_offset(1, lm), None, "band 1 LM {lm} not sentinel");
        }
        assert_eq!(cache_offset(1, 3), Some(0));
        assert_eq!(cache_offset(1, 4), Some(0));
    }

    /// The 23 distinct run offsets, and each run's `maxK` equals its
    /// length-minus-one (the runs tile `cache_bits50` exactly).
    #[test]
    fn runs_tile_exactly() {
        let mut distinct: Vec<usize> = CACHE_INDEX50
            .iter()
            .filter(|&&v| v >= 0)
            .map(|&v| v as usize)
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 23, "expected 23 distinct runs");
        // Boundaries: each run spans from its offset to the next.
        let mut bounds = distinct.clone();
        bounds.push(CACHE_BITS50.len());
        for w in bounds.windows(2) {
            let off = w[0];
            let run_len = w[1] - w[0];
            let max_k = usize::from(CACHE_BITS50[off]);
            assert_eq!(
                max_k,
                run_len - 1,
                "run @ {off}: maxK {max_k} != len-1 {}",
                run_len - 1
            );
        }
    }

    /// Within each run, `qbits[1..=maxK]` is monotonically
    /// non-decreasing (required for the search to be correct).
    #[test]
    fn runs_are_monotone() {
        let mut distinct: Vec<usize> = CACHE_INDEX50
            .iter()
            .filter(|&&v| v >= 0)
            .map(|&v| v as usize)
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        for &off in &distinct {
            let max_k = usize::from(CACHE_BITS50[off]);
            let mut prev = 0u8;
            for k in 1..=max_k {
                let cost = CACHE_BITS50[off + k];
                assert!(cost >= prev, "run @ {off} not monotone at K={k}");
                prev = cost;
            }
        }
    }

    /// K = 0 is free for every cached tuple; sentinels return None.
    #[test]
    fn cost_k_zero_is_free_for_cached() {
        for band in 0..NUM_BANDS {
            for lm in 0..NUM_LM {
                match cache_offset(band, lm) {
                    Some(_) => assert_eq!(cache_cost_8th(band, lm, 0), Some(0)),
                    None => assert_eq!(cache_cost_8th(band, lm, 0), None),
                }
            }
        }
    }

    /// Spot-check the trace §4 sharing/values: band 3 LM0 starts run
    /// @82 with qbits[1] = 20, qbits[40] = 128.
    #[test]
    fn spot_check_run_at_82() {
        assert_eq!(cache_offset(3, 0), Some(82));
        assert_eq!(cache_max_k(3, 0), Some(40));
        assert_eq!(cache_cost_8th(3, 0, 1), Some(20));
        assert_eq!(cache_cost_8th(3, 0, 40), Some(128));
        assert_eq!(cache_cost_8th(3, 0, 41), None); // exceeds maxK
    }

    /// Spot-check the smallest run (band 20 LM4 @387, maxK=4).
    #[test]
    fn spot_check_run_at_387() {
        assert_eq!(cache_offset(20, 4), Some(387));
        assert_eq!(cache_max_k(20, 4), Some(4));
        assert_eq!(cache_cost_8th(20, 4, 1), Some(67));
        assert_eq!(cache_cost_8th(20, 4, 4), Some(234));
    }

    /// The degenerate flat run @0: qbits is constant 7 for all K.
    #[test]
    fn flat_run_at_zero() {
        assert_eq!(cache_offset(1, 3), Some(0));
        for k in 1..=40 {
            assert_eq!(cache_cost_8th(1, 3, k), Some(7), "K={k}");
        }
    }

    /// Inversion picks the largest K within budget; never overshoots.
    #[test]
    fn inversion_picks_largest_within_budget() {
        // Run @82 (band 3 LM0): qbits[1]=20, qbits[2]=33, qbits[3]=41.
        // Budget 40 admits K up to 2 (cost 33), not 3 (cost 41).
        let r = cached_bits_to_pulses(3, 0, 40).unwrap();
        assert_eq!(r.k, 2);
        assert_eq!(r.bits_used_8th, 33);
        assert!(r.bits_used_8th <= 40);
        // Budget below qbits[1] picks K=0.
        let r0 = cached_bits_to_pulses(3, 0, 19).unwrap();
        assert_eq!(r0.k, 0);
        assert_eq!(r0.bits_used_8th, 0);
        // A generous budget reaches maxK.
        let rmax = cached_bits_to_pulses(3, 0, 10_000).unwrap();
        assert_eq!(rmax.k, 40);
        assert_eq!(rmax.bits_used_8th, 128);
    }

    /// Inversion of a sentinel returns None (caller uses closed form).
    #[test]
    fn inversion_sentinel_is_none() {
        assert_eq!(cached_bits_to_pulses(0, 0, 1000), None);
        assert_eq!(cached_bits_to_pulses(1, 0, 1000), None);
    }

    /// Inversion never overshoots its budget across the whole cache.
    #[test]
    fn inversion_never_overshoots() {
        for band in 0..NUM_BANDS {
            for lm in 0..NUM_LM {
                if let Some(off) = cache_offset(band, lm) {
                    let max_k = usize::from(CACHE_BITS50[off]);
                    let top = u32::from(CACHE_BITS50[off + max_k]);
                    for budget in 0..=top + 8 {
                        let r = cached_bits_to_pulses(band, lm, budget).unwrap();
                        assert!(
                            r.bits_used_8th <= budget,
                            "band {band} LM {lm} budget {budget}: used {}",
                            r.bits_used_8th
                        );
                        // Stepping K up by one must overshoot (or hit maxK).
                        if (r.k as usize) < max_k {
                            let next = u32::from(CACHE_BITS50[off + r.k as usize + 1]);
                            assert!(next > budget);
                        }
                    }
                }
            }
        }
    }
}
