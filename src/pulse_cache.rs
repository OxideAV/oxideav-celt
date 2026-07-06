//! Bit-exact pulse-cost cache for §4.3.4.1 bits-to-pulses inversion.
//!
//! ## What this module covers
//!
//! RFC 6716 §4.3.4.1 converts a per-band 1/8-bit budget into an
//! integer pulse count `K` by searching "a precomputed allocation
//! table that only permits some K values for each N" for the largest
//! `K` whose entropy cost does not exceed the budget. That table is
//! the `cache_index50` / `cache_bits50` pair: a 105-entry per-(LM,
//! band) offset table indexing a 392-byte packed sequence of per-`N`
//! cost curves.
//!
//! This module embeds both tables as factual numeric data and exposes
//! the walk that turns a `(band, LM, budget)` triple into a `(K,
//! bits-used)` pair — bit-exact rather than the worst-case
//! `ceil(log2 V(N, K))` estimator carried in [`crate::bits_to_pulses`]
//! for the uncached path.
//!
//! ## Layout (per the corrected clean-room trace)
//!
//! `cache_index50` has `5 LM rows × 21 bands = 105` entries in
//! **LM-major order**: entry `(LM + 1)*21 + band` holds the byte
//! offset into `cache_bits50` where that tuple's cost curve begins.
//! The five rows correspond to `LM ∈ {-1, 0, 1, 2, 3}`:
//!
//! * rows 1–4 are the four coded frame sizes (`LM = 0..=3`, i.e.
//!   2.5/5/10/20 ms) — see [`cache_offset`];
//! * row 0 is the `LM = -1` half-block row (`N` = half the 2.5 ms
//!   Table-55 bin count, used for transient/short-block accounting) —
//!   see [`cache_offset_half_block`]. Bands 0–7 have a 2.5 ms width
//!   of one MDCT bin, so their half-block `N` would be `1/2`; those
//!   eight entries carry the `-1` sentinel.
//!
//! Every `(band, LM)` tuple at the coded frame sizes resolves to a
//! run — there are **no sentinels** outside the `LM = -1` row.
//!
//! A cost curve ("run") at offset `off` is:
//!
//! ```text
//! cache_bits50[off]         = maxK      (largest K this run supports)
//! cache_bits50[off + 1]     = qbits[1]  (stored cost byte for K = 1)
//! ...
//! cache_bits50[off + maxK]  = qbits[maxK]
//! ```
//!
//! `K = 0` is implicit (cost 0, never stored). Each run prices the
//! PVQ codebook for a **single** band size `N` (a property the
//! corrected LM-major mapping guarantees; see the validation tests),
//! and multiple `(band, LM)` tuples sharing an `N` share one run, so
//! the 105 offsets resolve to only 23 distinct runs.
//!
//! ## Cost convention: `stored + 1`
//!
//! The stored byte `qbits[K]` is one **less** than the cost in 1/8-bit
//! fractional units (the `BITRES = 3` convention): the retrieved cost
//! of coding `K` pulses is `qbits[K] + 1` eighth-bits. Under that
//! convention the retrieved cost satisfies, for every run in the
//! table,
//!
//! ```text
//! qbits[K] + 1  >=  ceil(8 * log2(V(N, K)))
//! ```
//!
//! with **equality for every `K <= 16`** except a single entry (the
//! `N = 11` run at `K = 9`, one eighth-bit high — the safe direction),
//! where `V(N, K)` is the §4.3.4.2 PVQ codebook size. Above `K = 16`
//! the stored bytes climb further above the monolithic combinatorial
//! cost: that surplus is the splitting-aware accounting for high pulse
//! counts (the §4.3.4.4 recursive-split regime). The retrieved cost is
//! therefore a *tight, never-under-pricing* upper bound on the true
//! information cost of a PVQ index — the property that makes it safe
//! to drive encoder budget decisions from the cache. The validation
//! tests below re-derive both properties from this crate's own
//! `V(N, K)` combinatorics ([`crate::pvq::v_count`]).
//!
//! ## Clean-room provenance
//!
//! Every numeric value, the LM-major indexing, the run format, and the
//! `stored + 1` cost convention are transcribed from the corrected
//! clean-room trace `docs/audio/opus/pulse-cache-format-trace.md`
//! (#118, LM-major correction per issue #184) and the Feist-facts
//! numeric extracts `docs/audio/opus/tables/cache-index50.csv` /
//! `cache-bits50.csv` (+ `.meta`), with the §4.3.4.1 search prose from
//! `docs/audio/opus/rfc6716-opus.txt` lines 6476–6493. No external
//! library source was consulted.

/// Number of CELT nominal bands (RFC 6716 §4.3.2 boundary table).
pub const NUM_BANDS: usize = 21;

/// Number of LM rows the cache spans: `LM ∈ {-1, 0, 1, 2, 3}`. Rows
/// 1–4 are the coded frame sizes 2.5/5/10/20 ms; row 0 is the
/// half-block (`LM = -1`) accounting row.
pub const NUM_LM_ROWS: usize = 5;

/// Number of coded frame-size shifts (`LM ∈ 0..=3`).
pub const NUM_FRAME_LM: usize = 4;

/// Per-(LM, band) offset table into [`CACHE_BITS50`], LM-major:
/// `CACHE_INDEX50[(LM + 1)*21 + band]` for `LM ∈ {-1, 0, 1, 2, 3}`.
/// A value of `-1` is the sentinel "no cached cost" and occurs only
/// for bands 0–7 on the `LM = -1` row (their half-block `N` would be
/// `1/2`, not a valid band size).
///
/// Numeric values from `docs/audio/opus/tables/cache-index50.csv`;
/// LM-major layout from the corrected trace §2.1.
pub const CACHE_INDEX50: [i16; NUM_BANDS * NUM_LM_ROWS] = [
    // Row 0: LM = -1 (half-block). Bands 0-7 are sentinels.
    -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 41, 41, 41, 82, 82, 123, 164, 200, 222,
    // Row 1: LM = 0 (2.5 ms).
    0, 0, 0, 0, 0, 0, 0, 0, 41, 41, 41, 41, 123, 123, 123, 164, 164, 240, 266, 283, 295,
    // Row 2: LM = 1 (5 ms).
    41, 41, 41, 41, 41, 41, 41, 41, 123, 123, 123, 123, 240, 240, 240, 266, 266, 305, 318, 328, 336,
    // Row 3: LM = 2 (10 ms).
    123, 123, 123, 123, 123, 123, 123, 123, 240, 240, 240, 240, 305, 305, 305, 318, 318, 343, 351,
    358, 364, // Row 4: LM = 3 (20 ms).
    240, 240, 240, 240, 240, 240, 240, 240, 305, 305, 305, 305, 343, 343, 343, 351, 351, 370, 376,
    382, 387,
];

/// Packed run-encoded per-band cost curves, 392 bytes.
/// Numeric values from `docs/audio/opus/tables/cache-bits50.csv`.
pub const CACHE_BITS50: [u8; 392] = [
    // run @ 0  (maxK = 40, N = 1: V(1, K) = 2 for every K, flat cost)
    40, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, //
    // run @ 41 (maxK = 40, N = 2)
    40, 15, 23, 28, 31, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 47, 49, 50, 51, 52, 53, 54, 55,
    55, 57, 58, 59, 60, 61, 62, 63, 63, 65, 66, 67, 68, 69, 70, 71, 71, //
    // run @ 82 (maxK = 40, N = 3)
    40, 20, 33, 41, 48, 53, 57, 61, 64, 66, 69, 71, 73, 75, 76, 78, 80, 82, 85, 87, 89, 91, 92, 94,
    96, 98, 101, 103, 105, 107, 108, 110, 112, 114, 117, 119, 121, 123, 124, 126, 128, //
    // run @ 123 (maxK = 40, N = 4)
    40, 23, 39, 51, 60, 67, 73, 79, 83, 87, 91, 94, 97, 100, 102, 105, 107, 111, 115, 118, 121, 124,
    126, 129, 131, 135, 139, 142, 145, 148, 150, 153, 155, 159, 163, 166, 169, 172, 174, 177,
    179, //
    // run @ 164 (maxK = 35, N = 6)
    35, 28, 49, 65, 78, 89, 99, 107, 114, 120, 126, 132, 136, 141, 145, 149, 153, 159, 165, 171,
    176, 180, 185, 189, 192, 199, 205, 211, 216, 220, 225, 229, 232, 239, 245, 251, //
    // run @ 200 (maxK = 21, N = 9)
    21, 33, 58, 79, 97, 112, 125, 137, 148, 157, 166, 174, 182, 189, 195, 201, 207, 217, 227, 235,
    243, 251, //
    // run @ 222 (maxK = 17, N = 11)
    17, 35, 63, 86, 106, 123, 139, 152, 165, 177, 187, 197, 206, 214, 222, 230, 237, 250, //
    // run @ 240 (maxK = 25, N = 8)
    25, 31, 55, 75, 91, 105, 117, 128, 138, 146, 154, 161, 168, 174, 180, 185, 190, 200, 208, 215,
    222, 229, 235, 240, 245, 255, //
    // run @ 266 (maxK = 16, N = 12)
    16, 36, 65, 89, 110, 128, 144, 159, 173, 185, 196, 207, 217, 226, 234, 242, 250, //
    // run @ 283 (maxK = 11, N = 18)
    11, 41, 74, 103, 128, 151, 172, 191, 209, 225, 241, 255, //
    // run @ 295 (maxK = 9, N = 22)
    9, 43, 79, 110, 138, 163, 186, 207, 227, 246, //
    // run @ 305 (maxK = 12, N = 16)
    12, 39, 71, 99, 123, 144, 164, 182, 198, 214, 228, 241, 253, //
    // run @ 318 (maxK = 9, N = 24)
    9, 44, 81, 113, 142, 168, 192, 214, 235, 255, //
    // run @ 328 (maxK = 7, N = 36)
    7, 49, 90, 127, 160, 191, 220, 247, //
    // run @ 336 (maxK = 6, N = 44)
    6, 51, 95, 134, 170, 203, 234, //
    // run @ 343 (maxK = 7, N = 32)
    7, 47, 87, 123, 155, 184, 212, 237, //
    // run @ 351 (maxK = 6, N = 48)
    6, 52, 97, 137, 174, 208, 240, //
    // run @ 358 (maxK = 5, N = 72)
    5, 57, 106, 151, 192, 231, //
    // run @ 364 (maxK = 5, N = 88)
    5, 59, 111, 158, 202, 243, //
    // run @ 370 (maxK = 5, N = 64)
    5, 55, 103, 147, 187, 224, //
    // run @ 376 (maxK = 5, N = 96)
    5, 60, 113, 161, 206, 248, //
    // run @ 382 (maxK = 4, N = 144)
    4, 65, 122, 175, 224, //
    // run @ 387 (maxK = 4, N = 176)
    4, 67, 127, 182, 234,
];

/// Look up the byte offset of the cost run for a `(band, LM)` tuple at
/// a **coded frame size** (`lm ∈ 0..=3`), i.e. row `lm + 1` of the
/// LM-major index.
///
/// Every in-range tuple resolves (the coded rows carry no sentinels);
/// `band >= NUM_BANDS` or `lm > 3` returns `None`.
#[inline]
pub fn cache_offset(band: usize, lm: usize) -> Option<usize> {
    if band >= NUM_BANDS || lm >= NUM_FRAME_LM {
        return None;
    }
    let off = CACHE_INDEX50[(lm + 1) * NUM_BANDS + band];
    debug_assert!(off >= 0, "coded rows carry no sentinels");
    if off < 0 {
        None
    } else {
        Some(off as usize)
    }
}

/// Look up the byte offset of the cost run for a band on the
/// `LM = -1` half-block row (row 0 of the LM-major index), whose `N`
/// is half the band's 2.5 ms Table-55 bin count.
///
/// Returns `None` for the eight sentinel bands 0–7 (half-block `N`
/// would be `1/2`) and for `band >= NUM_BANDS`.
#[inline]
pub fn cache_offset_half_block(band: usize) -> Option<usize> {
    if band >= NUM_BANDS {
        return None;
    }
    let off = CACHE_INDEX50[band];
    if off < 0 {
        None
    } else {
        Some(off as usize)
    }
}

/// The largest `K` the cached run for `(band, LM)` supports at a coded
/// frame size, or `None` for an out-of-range tuple.
#[inline]
pub fn cache_max_k(band: usize, lm: usize) -> Option<u32> {
    cache_offset(band, lm).map(|off| u32::from(CACHE_BITS50[off]))
}

/// The bit-exact 1/8-bit cost of coding `K` pulses for `(band, LM)`
/// from the cache — the retrieved cost `qbits[K] + 1` per the trace's
/// stored-vs-actual convention (module docs) — or `None` when the
/// tuple is out of range or `K` exceeds the run's `maxK`.
///
/// `K = 0` is free (cost 0) for every cached tuple.
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
    Some(u32::from(CACHE_BITS50[off + k as usize]) + 1)
}

/// The raw stored cost byte `qbits[K]` for `(band, LM, K)` at a coded
/// frame size, without the `+1` retrieval adjustment. Exposed for
/// table inspection / cross-validation; allocation arithmetic should
/// use [`cache_cost_8th`].
#[inline]
pub fn cache_stored_qbits(band: usize, lm: usize, k: u32) -> Option<u8> {
    let off = cache_offset(band, lm)?;
    if k == 0 || k > u32::from(CACHE_BITS50[off]) {
        return None;
    }
    Some(CACHE_BITS50[off + k as usize])
}

/// Result of a cached bits-to-pulses inversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CachedPulses {
    /// Chosen integer pulse count `K`.
    pub k: u32,
    /// Bit-exact 1/8-bit retrieved cost of coding `K` pulses
    /// (`qbits[K] + 1`; 0 when `k == 0`).
    pub bits_used_8th: u32,
}

/// Invert a per-band 1/8-bit budget into the largest cached `K` whose
/// retrieved cost does not exceed it, for the `(band, LM)` tuple at a
/// coded frame size.
///
/// Implements the §4.3.4.1 inner loop (trace §7) under the trace §2.3
/// `stored + 1` retrieval convention: walk `K = 1 .. maxK`, stop at
/// the first `K` whose retrieved cost exceeds `budget_8th`, and take
/// the last `K` that fit. `K = maxK` is permitted — `maxK` is the
/// band's cached pulse ceiling (beyond it lies the §4.3.4.4 split
/// regime). Returns `None` for out-of-range `(band, LM)`.
pub fn cached_bits_to_pulses(band: usize, lm: usize, budget_8th: u32) -> Option<CachedPulses> {
    let off = cache_offset(band, lm)?;
    let max_k = usize::from(CACHE_BITS50[off]);
    let mut best = CachedPulses {
        k: 0,
        bits_used_8th: 0,
    };
    for k in 1..=max_k {
        let cost = u32::from(CACHE_BITS50[off + k]) + 1;
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
    use crate::band_minimums::BAND_BINS_LM;
    use crate::pvq::{v_count, V_COUNT_SATURATION};

    /// The band size `N` a `(band, LM)` tuple prices, per RFC 6716
    /// Table 55: the coded rows read `BAND_BINS_LM[lm][band]`; the
    /// half-block row is half the 2.5 ms count (`None` when odd).
    fn n_of(band: usize, lm: i32) -> Option<u32> {
        let base = BAND_BINS_LM[0][band];
        if lm >= 0 {
            Some(base << lm)
        } else if base % 2 == 0 {
            Some(base / 2)
        } else {
            None
        }
    }

    /// `ceil(8 * log2(v))` for a codebook size `v >= 1`. Exact for
    /// powers of two; for the table's non-power-of-two sizes the f64
    /// evaluation is safe — the nearest `8*log2(v)` comes to an
    /// integer across every `(N, K)` the cache covers is ~2.8e-5
    /// (verified offline with exact big-integer arithmetic), ten
    /// orders of magnitude above f64 log error.
    fn ceil_8_log2(v: u32) -> u32 {
        if v <= 1 {
            return 0;
        }
        if v.is_power_of_two() {
            return 8 * v.trailing_zeros();
        }
        (8.0 * f64::from(v).log2()).ceil() as u32
    }

    /// The embedded tables must match the staged corpus sizes exactly.
    #[test]
    fn table_sizes_match_corpus() {
        assert_eq!(CACHE_INDEX50.len(), 105);
        assert_eq!(CACHE_BITS50.len(), 392);
    }

    /// Exactly 8 sentinels, and they are precisely bands 0–7 of the
    /// `LM = -1` row (trace §5): the bands whose 2.5 ms width is one
    /// bin, so the half-block `N` would be `1/2`.
    #[test]
    fn sentinel_count_and_placement() {
        let sentinels: Vec<usize> = CACHE_INDEX50
            .iter()
            .enumerate()
            .filter(|(_, &v)| v < 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(sentinels, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        for band in 0..8 {
            assert_eq!(cache_offset_half_block(band), None, "band {band}");
            assert_eq!(n_of(band, -1), None, "band {band} half-block N");
        }
        for band in 8..NUM_BANDS {
            assert!(cache_offset_half_block(band).is_some(), "band {band}");
        }
        // The coded rows carry no sentinels: every (band, lm) resolves.
        for lm in 0..NUM_FRAME_LM {
            for band in 0..NUM_BANDS {
                assert!(cache_offset(band, lm).is_some(), "band {band} lm {lm}");
            }
        }
        // Out-of-range tuples are None, not panics.
        assert_eq!(cache_offset(NUM_BANDS, 0), None);
        assert_eq!(cache_offset(0, NUM_FRAME_LM), None);
        assert_eq!(cache_offset_half_block(NUM_BANDS), None);
    }

    /// The 23 distinct run offsets tile `cache_bits50` exactly, and each
    /// run's `maxK` equals its length minus one.
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

    /// **The LM-major mapping's decisive property**: every run resolves
    /// to a single band size `N` across all the `(band, LM)` tuples that
    /// share it — the consistency requirement a per-`(N, K)` cost cache
    /// must satisfy, and the property the old band-major reading
    /// violated (trace §2.2).
    #[test]
    fn every_run_prices_a_single_n() {
        use std::collections::BTreeMap;
        let mut run_n: BTreeMap<usize, u32> = BTreeMap::new();
        for lm in -1i32..=3 {
            for band in 0..NUM_BANDS {
                let off = if lm < 0 {
                    cache_offset_half_block(band)
                } else {
                    cache_offset(band, lm as usize)
                };
                let Some(off) = off else { continue };
                let n = n_of(band, lm).expect("non-sentinel tuple has integral N");
                match run_n.get(&off) {
                    Some(&prev) => assert_eq!(
                        prev, n,
                        "run @ {off} shared by different N: {prev} vs {n} (band {band}, LM {lm})"
                    ),
                    None => {
                        run_n.insert(off, n);
                    }
                }
            }
        }
        assert_eq!(run_n.len(), 23, "all 23 runs reached");
        // Spot-pin a few trace §4 rows: run 0 is N=1, run 41 is N=2,
        // run 387 is N=176.
        assert_eq!(run_n[&0], 1);
        assert_eq!(run_n[&41], 2);
        assert_eq!(run_n[&123], 4);
        assert_eq!(run_n[&387], 176);
    }

    /// **Combinatoric validation of the cost bytes** against this
    /// crate's own §4.3.4.2 `V(N, K)` recursion: for every run and
    /// every `K`, the retrieved cost `qbits[K] + 1` never under-prices
    /// the monolithic PVQ index cost `ceil(8*log2(V(N, K)))`, and for
    /// `K <= 16` it *equals* it — except the single `(N = 11, K = 9)`
    /// entry, which is one eighth-bit high (the safe direction). Above
    /// `K = 16` the surplus is the splitting-aware accounting the
    /// trace's §2.3 caveat documents.
    #[test]
    fn cost_bytes_match_own_combinatorics() {
        for lm in -1i32..=3 {
            for band in 0..NUM_BANDS {
                let off = if lm < 0 {
                    cache_offset_half_block(band)
                } else {
                    cache_offset(band, lm as usize)
                };
                let Some(off) = off else { continue };
                let n = n_of(band, lm).unwrap();
                let max_k = usize::from(CACHE_BITS50[off]);
                for k in 1..=max_k {
                    let retrieved = u32::from(CACHE_BITS50[off + k]) + 1;
                    let v = v_count(n, k as u32);
                    assert_ne!(
                        v, V_COUNT_SATURATION,
                        "cached region must stay inside the 32-bit codebook bound \
                         (N={n}, K={k})"
                    );
                    let monolithic = ceil_8_log2(v);
                    assert!(
                        retrieved >= monolithic,
                        "under-priced: band {band} LM {lm} N={n} K={k}: \
                         retrieved {retrieved} < ceil(8log2 V)={monolithic}"
                    );
                    if k <= 16 && !(n == 11 && k == 9) {
                        assert_eq!(
                            retrieved, monolithic,
                            "exact-identity region: band {band} LM {lm} N={n} K={k}"
                        );
                    }
                }
            }
        }
        // The single documented exception: N = 11 (the band-20
        // half-block run @222), K = 9 prices one eighth-bit high.
        let off = cache_offset_half_block(20).unwrap();
        assert_eq!(off, 222);
        let retrieved = u32::from(CACHE_BITS50[off + 9]) + 1;
        assert_eq!(retrieved, 178);
        assert_eq!(ceil_8_log2(v_count(11, 9)), 177);
    }

    /// The `V(2, 40)` regression the mapping correction fixed: under
    /// the corrected index an `N = 2` tuple (band 8, LM 0) prices
    /// `K = 40` at 72 eighth-bits (9 bits) — the splitting-aware upper
    /// bound on the ~59-eighth-bit monolithic cost — not the absurd 7
    /// the old band-major mapping produced by routing `N = 2` onto the
    /// flat `N = 1` run.
    #[test]
    fn v_2_40_regression() {
        // Band 8 at LM 0 has N = 2 (Table 55) and must land on run @41.
        assert_eq!(BAND_BINS_LM[0][8], 2);
        assert_eq!(cache_offset(8, 0), Some(41));
        let cost = cache_cost_8th(8, 0, 40).unwrap();
        assert_eq!(cost, 72);
        // Monolithic cost of V(2, 40) = 160: ceil(8*log2(160)) = 59.
        assert_eq!(v_count(2, 40), 160);
        assert_eq!(ceil_8_log2(160), 59);
        assert!(cost >= 59, "upper bound holds");
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

    /// K = 0 is free for every coded tuple; the raw-byte accessor and
    /// the retrieved cost differ by exactly one.
    #[test]
    fn cost_conventions() {
        for band in 0..NUM_BANDS {
            for lm in 0..NUM_FRAME_LM {
                assert_eq!(cache_cost_8th(band, lm, 0), Some(0));
                let max_k = cache_max_k(band, lm).unwrap();
                for k in 1..=max_k {
                    let stored = cache_stored_qbits(band, lm, k).unwrap();
                    assert_eq!(
                        cache_cost_8th(band, lm, k),
                        Some(u32::from(stored) + 1),
                        "band {band} lm {lm} K={k}"
                    );
                }
                assert_eq!(cache_cost_8th(band, lm, max_k + 1), None);
                assert_eq!(cache_stored_qbits(band, lm, max_k + 1), None);
            }
        }
    }

    /// Spot-check trace §4 sharing: band 3 at LM 0 has N = 1 and sits
    /// on the flat run @0; band 12 at LM 0 has N = 4 on run @123 with
    /// retrieved costs 24 (K=1) and 180 (K=40).
    #[test]
    fn spot_check_lm_major_offsets() {
        assert_eq!(cache_offset(3, 0), Some(0));
        assert_eq!(cache_max_k(3, 0), Some(40));
        assert_eq!(cache_offset(12, 0), Some(123));
        assert_eq!(cache_cost_8th(12, 0, 1), Some(24));
        assert_eq!(cache_cost_8th(12, 0, 40), Some(180));
        // Band 20 LM 3 is the largest band at the largest frame:
        // run @387 (N = 176), maxK = 4.
        assert_eq!(cache_offset(20, 3), Some(387));
        assert_eq!(cache_max_k(20, 3), Some(4));
        assert_eq!(cache_cost_8th(20, 3, 1), Some(68));
        assert_eq!(cache_cost_8th(20, 3, 4), Some(235));
        // Half-block row: band 8 halves to N = 1 (flat run @0).
        assert_eq!(cache_offset_half_block(8), Some(0));
        assert_eq!(cache_offset_half_block(20), Some(222));
    }

    /// The flat run @0 prices every K at one bit — exactly what
    /// `V(1, K) = 2` (sign only) requires under the +1 convention.
    #[test]
    fn flat_run_prices_one_bit() {
        for k in 1..=40 {
            assert_eq!(cache_cost_8th(0, 0, k), Some(8), "K={k}");
            assert_eq!(v_count(1, k), 2);
        }
    }

    /// Inversion picks the largest K within budget; never overshoots.
    #[test]
    fn inversion_picks_largest_within_budget() {
        // Band 12 LM 0 (run @123, N = 4): retrieved costs 24, 40, 52...
        let r = cached_bits_to_pulses(12, 0, 45).unwrap();
        assert_eq!(r.k, 2);
        assert_eq!(r.bits_used_8th, 40);
        // Budget below the K=1 retrieved cost picks K=0.
        let r0 = cached_bits_to_pulses(12, 0, 23).unwrap();
        assert_eq!(r0.k, 0);
        assert_eq!(r0.bits_used_8th, 0);
        // A generous budget reaches maxK.
        let rmax = cached_bits_to_pulses(12, 0, 10_000).unwrap();
        assert_eq!(rmax.k, 40);
        assert_eq!(rmax.bits_used_8th, 180);
    }

    /// Inversion of an out-of-range tuple returns None.
    #[test]
    fn inversion_out_of_range_is_none() {
        assert_eq!(cached_bits_to_pulses(NUM_BANDS, 0, 1000), None);
        assert_eq!(cached_bits_to_pulses(0, NUM_FRAME_LM, 1000), None);
    }

    /// Inversion never overshoots its budget across the whole cache,
    /// and stepping K up by one always would.
    #[test]
    fn inversion_never_overshoots() {
        for band in 0..NUM_BANDS {
            for lm in 0..NUM_FRAME_LM {
                let off = cache_offset(band, lm).unwrap();
                let max_k = usize::from(CACHE_BITS50[off]);
                let top = u32::from(CACHE_BITS50[off + max_k]) + 1;
                for budget in 0..=top + 8 {
                    let r = cached_bits_to_pulses(band, lm, budget).unwrap();
                    assert!(
                        r.bits_used_8th <= budget,
                        "band {band} LM {lm} budget {budget}: used {}",
                        r.bits_used_8th
                    );
                    if (r.k as usize) < max_k {
                        let next = u32::from(CACHE_BITS50[off + r.k as usize + 1]) + 1;
                        assert!(next > budget);
                    }
                }
            }
        }
    }
}
