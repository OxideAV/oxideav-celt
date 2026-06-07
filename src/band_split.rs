//! PVQ band-split gating and recursion geometry (RFC 6716 §4.3.4.4,
//! page 118 of `docs/audio/opus/rfc6716-opus.txt`).
//!
//! ## What this module covers
//!
//! Per RFC 6716 §4.3.4.4:
//!
//! > To avoid the need for multi-precision calculations when decoding
//! > PVQ codevectors, the maximum size allowed for codebooks is 32
//! > bits. When larger codebooks are needed, the vector is instead
//! > split in two sub-vectors of size N/2. A quantized gain parameter
//! > with precision derived from the current allocation is entropy
//! > coded to represent the relative gains of each side of the split,
//! > and the entire decoding process is recursively applied. Multiple
//! > levels of splitting may be applied up to a limit of LM+1 splits.
//! > The same recursive mechanism is applied for the joint coding of
//! > stereo audio.
//!
//! This module supplies the **gating + geometry** layer:
//!
//! 1. [`band_needs_split`] — returns `true` when the codebook size
//!    `V(N, K)` would not fit in 32 bits and a split is therefore
//!    required.
//! 2. [`split_dimensions`] — returns the `(N_lo, N_hi)` half-split
//!    sizes for the band, matching the RFC's "two sub-vectors of size
//!    N/2" with even-vs-odd `N` partitioning preserving the total.
//! 3. [`max_split_levels`] — returns the `LM+1` cap on recursion
//!    depth.
//! 4. [`BandSplitNode`] — the recursive descriptor of how a band
//!    decomposes into a tree of leaf-PVQ sub-bands the higher-level
//!    band-decode walker traverses.
//! 5. [`plan_band_split`] — descends the recursion from the given
//!    `(N, K, LM)` building the tree of N's, stopping when the leaf
//!    codebook fits in 32 bits or when the `LM+1` depth cap is
//!    reached.
//!
//! ## What is NOT in this module (deferred to a docs gap)
//!
//! The quantized gain parameter that splits the relative L2 norm
//! between the two halves and the precision (1/8-bit precision) the
//! §4.3.4.4 prose derives from "the current allocation" are not
//! defined in RFC 6716. The narrative defers to the reference
//! implementation for the exact `qb` derivation. The split-gain
//! decode is therefore queued for a future round once the
//! corresponding clean-room trace document covers it; the structural
//! backbone here can be wired up to a gain-aware leaf walker without
//! re-shaping the tree.
//!
//! ## Allocation distribution across the split
//!
//! Once a band is split, the §4.3.4.4 recursive descent re-applies
//! the §4.3.4.1 bits-to-pulses search on each half. The bit budget
//! distribution across the split (after the gain bits are paid for)
//! is governed by the same gain parameter that the docs gap above
//! covers; this module does not commit to a split rule yet. Callers
//! that need a placeholder for a leaf-level `K` while the gain
//! decode is missing can pass `K = 0` to descend the geometry
//! without invoking the PVQ leaf decode at the wrong allocation.
//!
//! ## Clean-room provenance
//!
//! The §4.3.4.4 quotation, the 32-bit codebook budget, the N/2 split,
//! and the `LM+1` recursion cap are reproduced from RFC 6716 §4.3.4.4
//! (`docs/audio/opus/rfc6716-opus.txt` lines 6601–6619). The
//! `V(N, K)` saturation flag is reused from [`crate::pvq`]. No
//! external library source was consulted.

use crate::pvq::{v_count, V_COUNT_SATURATION};

/// Maximum CELT frame-size index `LM` recognised by RFC 6716 §4.3:
/// `LM ∈ {0, 1, 2, 3}` ↔ frame durations `{2.5, 5, 10, 20}` ms.
///
/// The §4.3.4.4 recursion cap is `LM + 1`; callers passing `LM` above
/// this constant get a clamped descent rather than silent acceptance.
pub const MAX_LM: u32 = 3;

/// Returns `true` iff the codebook size `V(N, K)` would not fit in 32
/// bits — the §4.3.4.4 split trigger.
///
/// The check uses [`v_count`], which saturates to [`V_COUNT_SATURATION`]
/// (= `u32::MAX`) when the recurrence overflows; treating that
/// saturation as "needs split" keeps the test bit-exact against the
/// §4.3.4.4 32-bit codebook budget. For `K == 0` the codebook is
/// trivially the all-zero codeword (`V(N, 0) = 1`); no split is ever
/// needed.
///
/// Hot-path constraint: this function is `O(N·K)` per the [`v_count`]
/// recurrence. The §4.3.4.1 bits-to-pulses search already computes
/// `V(N, K)` for each candidate `K`, so the caller can typically short-
/// circuit the predicate by checking `v_count == V_COUNT_SATURATION`
/// directly. The predicate here is exposed for callers driving the
/// geometry without an in-loop `V(N, K)` value.
pub fn band_needs_split(n: u32, k: u32) -> bool {
    if k == 0 || n == 0 {
        return false;
    }
    v_count(n, k) == V_COUNT_SATURATION
}

/// Split a band of `N` samples into two halves per RFC 6716 §4.3.4.4
/// ("two sub-vectors of size N/2").
///
/// Returns `(N_lo, N_hi)` with `N_lo + N_hi == N`:
///
/// * Even `N` yields `(N/2, N/2)` — the spec wording exactly.
/// * Odd `N` yields `(N/2, N/2 + 1)` (floor + ceil). The §4.3.4.4
///   prose is silent on the parity; the canonical CELT band sizes in
///   `BAND_BINS_LM` are all even for `N > 1`, so odd splits only arise
///   at the deepest recursion level. The lower index gets the smaller
///   half so the leaf-PVQ walker sees a deterministic ordering.
/// * `N <= 1` cannot be split further and yields `(0, 0)`. Callers
///   should check `band_needs_split` (which is `false` for `N == 0`
///   and for `K == 0`) before invoking this helper.
pub fn split_dimensions(n: u32) -> (u32, u32) {
    if n <= 1 {
        return (0, 0);
    }
    let lo = n / 2;
    let hi = n - lo;
    (lo, hi)
}

/// Returns the §4.3.4.4 recursion cap `LM + 1`.
///
/// `LM` is clamped to [`MAX_LM`] (= 3) so callers passing
/// out-of-range LM values get the canonical 20 ms cap instead of an
/// unbounded descent.
pub fn max_split_levels(lm: u32) -> u32 {
    let clamped = if lm > MAX_LM { MAX_LM } else { lm };
    clamped + 1
}

/// Node in the recursive band-split decomposition tree.
///
/// A `Leaf { n }` describes a sub-band of dimension `n` that the
/// §4.3.4.2 PVQ decoder can handle directly — its codebook `V(n, K)`
/// fits in 32 bits for some plausible `K` (the leaf-level allocation
/// chooses `K` later via the §4.3.4.1 search).
///
/// A `Split { lo, hi }` describes a sub-band that the §4.3.4.4 rule
/// requires to be halved further; `lo` covers the lower-index half
/// and `hi` covers the upper-index half.
///
/// The tree is built by [`plan_band_split`] and is consumed by the
/// higher-level band-decode walker (a future round).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BandSplitNode {
    /// A leaf sub-band of dimension `n`; the PVQ decoder runs here.
    Leaf {
        /// Number of MDCT bins in this leaf sub-band.
        n: u32,
    },
    /// An internal split node; descend into both halves.
    Split {
        /// Lower-index half (`N / 2` MDCT bins).
        lo: Box<BandSplitNode>,
        /// Upper-index half (`N - N / 2` MDCT bins).
        hi: Box<BandSplitNode>,
    },
}

impl BandSplitNode {
    /// Total dimension covered by this node — recursively sums the
    /// leaf `n` values.
    pub fn total_n(&self) -> u32 {
        match self {
            BandSplitNode::Leaf { n } => *n,
            BandSplitNode::Split { lo, hi } => lo.total_n() + hi.total_n(),
        }
    }

    /// Number of leaf sub-bands beneath this node.
    pub fn leaf_count(&self) -> u32 {
        match self {
            BandSplitNode::Leaf { .. } => 1,
            BandSplitNode::Split { lo, hi } => lo.leaf_count() + hi.leaf_count(),
        }
    }

    /// Depth of this node's deepest split (a bare `Leaf` returns 0).
    pub fn depth(&self) -> u32 {
        match self {
            BandSplitNode::Leaf { .. } => 0,
            BandSplitNode::Split { lo, hi } => 1 + lo.depth().max(hi.depth()),
        }
    }

    /// Visit every leaf in left-to-right (low-to-high MDCT-bin index)
    /// order, calling the supplied closure on each leaf's `n`.
    ///
    /// Useful for composing the leaf-sequence layout the band-decode
    /// walker needs without allocating an intermediate `Vec`.
    pub fn for_each_leaf<F: FnMut(u32)>(&self, mut f: F) {
        self.for_each_leaf_impl(&mut f);
    }

    fn for_each_leaf_impl<F: FnMut(u32)>(&self, f: &mut F) {
        match self {
            BandSplitNode::Leaf { n } => f(*n),
            BandSplitNode::Split { lo, hi } => {
                lo.for_each_leaf_impl(f);
                hi.for_each_leaf_impl(f);
            }
        }
    }

    /// Collect every leaf's `n` into a `Vec` in left-to-right order.
    pub fn leaf_dims(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.leaf_count() as usize);
        self.for_each_leaf(|n| out.push(n));
        out
    }
}

/// Plan the §4.3.4.4 recursive split of a band of dimension `n` for a
/// hypothetical pulse count `k` at frame-size index `lm`.
///
/// The descent halves the band on each step until *either*:
///
/// * `band_needs_split` is `false` for the current `(n, k_for_node)`
///   — the leaf codebook fits in 32 bits.
/// * The depth reaches `max_split_levels(lm)` — the §4.3.4.4 `LM+1`
///   recursion cap.
/// * The dimension drops to `n <= 1` — no further split is geometric-
///   ally meaningful.
///
/// The `k_for_node` walked into each sub-band is **the same** `k` the
/// caller supplied; the §4.3.4.4 gain-parameter mechanism that would
/// redistribute the pulse budget across the split is a docs gap (see
/// module docs). Passing the gating `k` here is enough to drive the
/// **geometry** correctly: the predicate's monotonicity means that if
/// `V(n, k)` overflows then `V(n/2, k)` may still overflow (for very
/// large k) and the split continues recursively. The leaf `k` the PVQ
/// decoder ultimately consumes is chosen later by the bits-to-pulses
/// search at leaf time.
///
/// Returns `BandSplitNode::Leaf { n: 0 }` for `n == 0`.
pub fn plan_band_split(n: u32, k: u32, lm: u32) -> BandSplitNode {
    let cap = max_split_levels(lm);
    plan_band_split_rec(n, k, cap)
}

fn plan_band_split_rec(n: u32, k: u32, levels_remaining: u32) -> BandSplitNode {
    if n <= 1 || levels_remaining == 0 || !band_needs_split(n, k) {
        return BandSplitNode::Leaf { n };
    }
    let (lo_n, hi_n) = split_dimensions(n);
    let lo = plan_band_split_rec(lo_n, k, levels_remaining - 1);
    let hi = plan_band_split_rec(hi_n, k, levels_remaining - 1);
    BandSplitNode::Split {
        lo: Box::new(lo),
        hi: Box::new(hi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_zero_never_splits() {
        // V(N, 0) = 1, fits in 32 bits, no split needed.
        for n in [1u32, 2, 8, 16, 88, 176, 1024] {
            assert!(!band_needs_split(n, 0), "n={n}");
        }
    }

    #[test]
    fn n_zero_never_splits() {
        for k in [0u32, 1, 8, 16, 128] {
            assert!(!band_needs_split(0, k), "k={k}");
        }
    }

    #[test]
    fn small_k_does_not_trigger_split() {
        // Small pulse counts on realistic band sizes stay well inside
        // the 32-bit codebook budget. (V(N, K) ~ (2N)^K / K!, so
        // V(176, 4) = 639_716_352 still fits, but V(176, 5) crosses
        // the 2^32 boundary — see large_k_on_wide_band_triggers_split.)
        assert!(!band_needs_split(8, 1));
        assert!(!band_needs_split(16, 2));
        assert!(!band_needs_split(22, 4));
        assert!(!band_needs_split(88, 4));
        assert!(!band_needs_split(176, 4));
    }

    #[test]
    fn large_k_on_wide_band_triggers_split() {
        // Sufficiently aggressive K on a wide band overflows the
        // 32-bit codebook budget — the §4.3.4.4 trigger.
        //
        // V(176, 5) ≈ 4.5e10 crosses 2^32 (~4.3e9) — first split-
        // requiring point for the canonical maximum band width.
        assert!(band_needs_split(176, 5));
        // V(176, 100) is astronomically larger than 2^32; v_count
        // saturates.
        assert!(band_needs_split(176, 100));
        // V(88, 64) also saturates u32.
        assert!(band_needs_split(88, 64));
    }

    #[test]
    fn split_even_band_into_equal_halves() {
        assert_eq!(split_dimensions(2), (1, 1));
        assert_eq!(split_dimensions(8), (4, 4));
        assert_eq!(split_dimensions(16), (8, 8));
        assert_eq!(split_dimensions(176), (88, 88));
    }

    #[test]
    fn split_odd_band_into_floor_ceil() {
        // Smaller half on the low side, ceil on the high side.
        assert_eq!(split_dimensions(3), (1, 2));
        assert_eq!(split_dimensions(5), (2, 3));
        assert_eq!(split_dimensions(7), (3, 4));
        assert_eq!(split_dimensions(11), (5, 6));
    }

    #[test]
    fn split_preserves_total_dimension() {
        for n in 1u32..=200 {
            let (lo, hi) = split_dimensions(n);
            if n <= 1 {
                assert_eq!((lo, hi), (0, 0), "n={n}");
            } else {
                assert_eq!(lo + hi, n, "n={n}");
                assert!(lo <= hi, "n={n} lo={lo} hi={hi}");
            }
        }
    }

    #[test]
    fn max_split_levels_is_lm_plus_one() {
        assert_eq!(max_split_levels(0), 1);
        assert_eq!(max_split_levels(1), 2);
        assert_eq!(max_split_levels(2), 3);
        assert_eq!(max_split_levels(3), 4);
    }

    #[test]
    fn max_split_levels_clamps_out_of_range_lm() {
        // Defensive clamp on out-of-range LM matches the canonical
        // 20-ms (LM=3) cap.
        assert_eq!(max_split_levels(4), 4);
        assert_eq!(max_split_levels(255), 4);
    }

    #[test]
    fn plan_leaf_when_codebook_fits() {
        // V(8, 1) = 16 — small codebook, no split.
        let tree = plan_band_split(8, 1, 3);
        assert_eq!(tree, BandSplitNode::Leaf { n: 8 });
        assert_eq!(tree.total_n(), 8);
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn plan_splits_when_codebook_overflows() {
        // V(176, 100) overflows — must split.
        let tree = plan_band_split(176, 100, 3);
        assert!(matches!(tree, BandSplitNode::Split { .. }));
        assert_eq!(tree.total_n(), 176);
        assert!(tree.depth() >= 1);
    }

    #[test]
    fn plan_respects_lm_plus_one_cap() {
        // Force a configuration that *would* keep splitting forever
        // (very high K) and confirm the tree depth never exceeds
        // LM + 1 = 4 for LM = 3.
        for lm in 0..=3 {
            let tree = plan_band_split(176, 1000, lm);
            assert!(
                tree.depth() <= max_split_levels(lm),
                "lm={lm} depth={} cap={}",
                tree.depth(),
                max_split_levels(lm)
            );
            assert_eq!(tree.total_n(), 176, "lm={lm}");
        }
    }

    #[test]
    fn plan_leaf_dims_sum_to_total() {
        let tree = plan_band_split(176, 200, 3);
        let dims = tree.leaf_dims();
        let sum: u32 = dims.iter().sum();
        assert_eq!(sum, 176);
        assert_eq!(dims.len(), tree.leaf_count() as usize);
    }

    #[test]
    fn plan_leaf_traversal_is_left_to_right() {
        // Construct a known split and confirm the for_each_leaf
        // visitor walks the low-index half first.
        let tree = BandSplitNode::Split {
            lo: Box::new(BandSplitNode::Leaf { n: 4 }),
            hi: Box::new(BandSplitNode::Split {
                lo: Box::new(BandSplitNode::Leaf { n: 2 }),
                hi: Box::new(BandSplitNode::Leaf { n: 2 }),
            }),
        };
        let mut order = Vec::new();
        tree.for_each_leaf(|n| order.push(n));
        assert_eq!(order, vec![4, 2, 2]);
        assert_eq!(tree.total_n(), 8);
        assert_eq!(tree.leaf_count(), 3);
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    fn plan_lm_zero_caps_at_single_split() {
        // LM = 0 ⇒ at most one split level. Force a K large enough to
        // overflow even after one halving, and confirm the recursion
        // stops at depth 1.
        let tree = plan_band_split(176, 10_000, 0);
        assert!(tree.depth() <= 1);
        assert_eq!(tree.total_n(), 176);
    }

    #[test]
    fn plan_terminates_when_dimension_reaches_one() {
        // Even with a huge LM budget, a band of N=2 can split exactly
        // once into (1, 1); further descent stops.
        let tree = plan_band_split(2, 10_000, 3);
        // Either leaf-of-2 (V(2, 10_000) overflows but split goes to
        // halves of 1 each, which don't overflow) or a single-level
        // split with leaves of 1.
        assert_eq!(tree.total_n(), 2);
        // Depth bounded by the geometric cap: log2(2) = 1.
        assert!(tree.depth() <= 1);
    }

    #[test]
    fn plan_n_zero_is_zero_leaf() {
        let tree = plan_band_split(0, 0, 3);
        assert_eq!(tree, BandSplitNode::Leaf { n: 0 });
        assert_eq!(tree.total_n(), 0);
    }
}
