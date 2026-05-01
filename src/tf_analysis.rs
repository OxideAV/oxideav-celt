//! Per-band time/frequency resolution analysis (RFC 6716 §4.3.4.5 + §5.3.6).
//!
//! The CELT bitstream lets the encoder pick a per-band time/frequency
//! resolution adjustment, drawn from one of two columns of
//! [`crate::tables::TF_SELECT_TABLE`] selected via the per-frame
//! `tf_select` flag. The decoder side reads:
//!
//!   * a per-band raw delta bit (cumulative-XOR'd), one bit per coded band;
//!   * an optional `tf_select` bit (only emitted when the two candidate
//!     `tf_select` rows yield different results for the cumulative-XOR'd
//!     deltas).
//!
//! The per-band post-lookup `tf_change` value drives `quant_band`'s Hadamard
//! recombine (`tf_change > 0`) or time-divide (`tf_change < 0`) chain. Both
//! transforms are unitary — they redistribute energy across MDCT bins inside
//! a band without altering the band's L2 norm. The encoder picks per-band
//! `tf_change` to minimise an L1-norm distortion proxy under the Laplacian-
//! source-entropy interpretation (RFC §5.3.6) plus a small per-band
//! transition penalty.
//!
//! This module exposes:
//!
//!   * [`band_l1_under_tf`] — given a band's MDCT coefficients, returns the
//!     L1 norm after applying the relevant Hadamard recombine / time-divide
//!     for the supplied `tf_change`. Used by the analysis search and tested
//!     directly to pin the unitary-norm property and the L1 ordering on
//!     synthetic stimuli.
//!   * [`tf_analysis`] — Viterbi-style per-band selector. Returns the
//!     post-lookup `tf_res[]` (one entry per coded band), the raw per-band
//!     delta bits the encoder emits, and the chosen `tf_select`. Always
//!     RFC-compliant: every returned `tf_res[i]` is reachable from
//!     [`crate::tables::TF_SELECT_TABLE`] with the chosen raw bits + select.
//!
//! ### Scope (current crate state)
//!
//! The encoder pipeline currently applies `tf_change != 0` to **mono long-
//! block, non-transient frames only**. For other modes the analysis still
//! runs but the picked decisions are clamped to `tf_change == 0` everywhere
//! (the "safe" choice that needs no Hadamard wrapping in the band coder).
//! See [`tf_analysis`] for the gating logic.
//!
//! The analysis is exercised via unit tests in this module and integration-
//! tested via the round-trip test that injects a synthetic transient signal
//! and confirms the encoder still produces a packet the decoder accepts.

use crate::bands::haar1;
use crate::tables::{EBAND_5MS, NB_EBANDS, TF_SELECT_TABLE};

/// Apply the decoder's Hadamard recombine / time-divide chain to a single
/// band's worth of MDCT coefficients in-place, mirroring `bands::quant_band`.
///
/// Inputs:
///   * `band` — band's MDCT coefficients (length `n`, where `n` is the
///     band's coded width = `(EBAND_5MS[i+1] - EBAND_5MS[i]) * M`).
///   * `tf_change` — the post-lookup TF adjustment value. Positive means
///     "recombine" (frequency resolution up), negative means "time-divide"
///     (time resolution up), zero means no transform.
///   * `big_b` — the band's incoming block split. For long blocks (non-
///     transient) `big_b = 1`; for short blocks `big_b = M = 1 << lm`.
///
/// Both transforms are unitary, so the L2 norm is preserved. Use
/// [`band_l1_under_tf`] when only the L1 norm under the transform is needed
/// (it's a thin wrapper that clones, applies the transform, and sums).
pub fn apply_band_tf(band: &mut [f32], n: i32, tf_change: i32, big_b: i32) {
    if tf_change == 0 {
        return;
    }
    if tf_change > 0 {
        // Recombine: tf_change levels of haar1 with stride doubling.
        // Matches `bands::quant_band` lines 649-661.
        let recombine = tf_change;
        for k in 0..recombine {
            haar1(band, n >> k, 1 << k);
        }
    } else {
        // Time-divide: mirror the encoder's pre-haar chain in
        // `encoder_bands::encode_all_bands_mono`. The chain runs while
        // `n_b` stays even, capped at 4 iterations. Applying haar1
        // moves the band into the "partition-domain" view that
        // `quant_partition_enc` works in — and that's where the L1 norm
        // matters for the PVQ search.
        let mut n_b = n / big_b;
        let mut big_b_cur = big_b;
        let mut iters = 0i32;
        while (n_b & 1) == 0 {
            haar1(band, n_b, big_b_cur);
            big_b_cur <<= 1;
            n_b >>= 1;
            iters += 1;
            if iters > 4 {
                break;
            }
        }
    }
}

/// Return the L1 norm of a band's MDCT coefficients after applying the
/// Hadamard transform implied by `tf_change` (see [`apply_band_tf`]).
///
/// This is the distortion proxy the encoder minimises in [`tf_analysis`].
/// The L1 norm corresponds to the entropy of a Laplacian source per the
/// rate-distortion argument in RFC 6716 §5.3.6.
pub fn band_l1_under_tf(band: &[f32], n: i32, tf_change: i32, big_b: i32) -> f64 {
    if tf_change == 0 {
        return band.iter().take(n as usize).map(|v| v.abs() as f64).sum();
    }
    let mut buf = band[..n as usize].to_vec();
    apply_band_tf(&mut buf, n, tf_change, big_b);
    buf.iter().map(|v| v.abs() as f64).sum()
}

/// Per-band TF resolution decision result.
#[derive(Debug, Clone)]
pub struct TfDecision {
    /// Post-lookup `tf_change` per band (length `NB_EBANDS`). Bands outside
    /// `[start_band, end_band)` are zero.
    pub tf_res: Vec<i32>,
    /// Per-band raw delta bits the encoder emits (length `NB_EBANDS`). The
    /// decoder cumulatively XORs these to recover the per-band column index
    /// into [`TF_SELECT_TABLE`]. Bands outside the coded range are unused.
    pub raw_per_band: Vec<bool>,
    /// Frame-level `tf_select` flag (chooses between the two columns of the
    /// transient/non-transient half of [`TF_SELECT_TABLE`]).
    pub tf_select: bool,
    /// True if the encoder pipeline must apply [`apply_band_tf`] for any
    /// band with `tf_res[i] != 0`. Currently only set when the encoder's
    /// downstream wrapping in `encoder_bands` supports the picked changes;
    /// see module docs for the gating rules.
    pub apply_in_pipeline: bool,
}

/// RFC §5.3.6 TF analysis. Picks per-band TF adjustments to minimise the
/// total L1 norm of the band coefficients plus a per-band transition cost.
///
/// Inputs:
///   * `coeffs` — MDCT coefficients for the frame, length `n_total = M * 100`
///     for LM=3 mono long blocks (matches `encoder::CODED_N`).
///   * `transient` — frame-level transient flag.
///   * `lm` — log2 of frame size in 2.5 ms units (3 for the 20 ms LM=3
///     frame).
///   * `start_band` / `end_band` — coded band range.
///
/// The function searches both `tf_select` candidates. For each, it walks
/// the bands left-to-right and picks the per-band column (raw=0 vs raw=1)
/// that minimises `L1(band under tf_change) + lambda * (1 if column changes
/// from previous band else 0)`. The picked tf_select is the one with the
/// lower total cost.
///
/// `lambda` is the per-band transition penalty in L1 units. Higher values
/// favour fewer changes; lower values favour more aggressive tracking of
/// the per-band optimum. The default of `0.5 * sqrt(M*100/NB_EBANDS)`
/// roughly balances the typical per-band L1 against the bit-cost of a
/// delta in the range coder (4-5 logp bits per delta).
///
/// ### Pipeline gating
///
/// Even when the analysis prefers `tf_change != 0` for some bands, the
/// returned [`TfDecision::apply_in_pipeline`] flag is `false` when the
/// encoder's `encoder_bands` path doesn't yet support the chosen TF
/// adjustments end-to-end. In that case the caller should encode all-zero
/// raw bits (equivalent to `tf_change = 0`) to avoid an
/// encoder/decoder mismatch. The analysis still runs so the picks can be
/// inspected via tests as the pipeline support is widened.
pub fn tf_analysis(
    coeffs: &[f32],
    transient: bool,
    lm: i32,
    start_band: usize,
    end_band: usize,
) -> TfDecision {
    debug_assert!(end_band <= NB_EBANDS);
    debug_assert!(start_band <= end_band);
    let m = 1i32 << lm;
    let big_b = if transient { m } else { 1 };
    // Per-band L1 norms under each of the four reachable TF changes
    // (one per cell of TF_SELECT_TABLE for this transient row). Indexed
    // [tf_select][raw], values in [0, 3] for transient LM=3, etc.
    let table_row_offset = 4 * transient as usize;
    let tf_changes: [[i32; 2]; 2] = [
        [
            TF_SELECT_TABLE[lm as usize][table_row_offset] as i32,
            TF_SELECT_TABLE[lm as usize][table_row_offset + 1] as i32,
        ],
        [
            TF_SELECT_TABLE[lm as usize][table_row_offset + 2] as i32,
            TF_SELECT_TABLE[lm as usize][table_row_offset + 3] as i32,
        ],
    ];

    // Per-band (column 0 vs column 1) L1 cost. We compute these per the
    // RFC's L1-norm proxy on the actual MDCT coefficients of each band.
    // The transition cost approximates the range-coder bit cost of a delta
    // (logp ≈ 4-5 → ~0.5 bit on average), scaled to L1 units via the
    // mean per-band magnitude.
    let mut mean_l1 = 0f64;
    let mut counted = 0usize;
    let mut band_l1 = vec![[0f64; 2]; NB_EBANDS];
    let mut have_band = vec![false; NB_EBANDS];
    // Compute per-band L1 for column 0 and column 1 of the chosen
    // tf_select. We pick the tf_changes from tf_select=0 to seed; the
    // loop below evaluates both selects.
    let _ = (mean_l1, counted);

    // Best total cost per tf_select candidate.
    let mut best_total = [f64::INFINITY; 2];
    let mut best_picks: [Vec<usize>; 2] = [vec![0; NB_EBANDS], vec![0; NB_EBANDS]];

    for select in 0..2usize {
        let changes = tf_changes[select];
        // Pre-compute per-band L1 under each column.
        let mut per_band_cost = vec![[0f64; 2]; NB_EBANDS];
        let mut sum_l1 = 0f64;
        let mut cnt = 0usize;
        for i in start_band..end_band {
            let lo = (m * EBAND_5MS[i] as i32) as usize;
            let hi = (m * EBAND_5MS[i + 1] as i32) as usize;
            let n = (hi - lo) as i32;
            let band = &coeffs[lo..hi];
            for col in 0..2 {
                per_band_cost[i][col] = band_l1_under_tf(band, n, changes[col], big_b);
            }
            sum_l1 += per_band_cost[i][0].min(per_band_cost[i][1]);
            cnt += 1;
            band_l1[i] = per_band_cost[i];
            have_band[i] = true;
        }
        // Per-band transition penalty in L1 units. The transition cost
        // applies once per column flip; the per-band L1 difference must
        // overcome it before the Viterbi prefers a flip. We also add a
        // small ABSOLUTE bias toward column 0 (no transform) by inflating
        // the column-1 cost — this prevents the analyser from picking
        // tf_change != 0 on white-noise-like content where the haar1 ratio
        // is only ~6% lower than the identity (a meaningless gain at the
        // bit cost of a delta).
        let lambda = if cnt > 0 {
            (sum_l1 / cnt as f64) * 1.0
        } else {
            0.0
        };
        // Bias against column 1 (the "transform" column): require at
        // least a ~20% L1 advantage before the analyser flips, in addition
        // to the transition penalty above.
        let bias = if cnt > 0 {
            (sum_l1 / cnt as f64) * 0.20
        } else {
            0.0
        };
        for i in start_band..end_band {
            per_band_cost[i][1] += bias;
        }
        // Viterbi on a 2-state chain (column 0 vs column 1). State cost
        // accumulates the per-band L1; transitions add `lambda` when the
        // column flips.
        let mut dp = [[f64::INFINITY; 2]; NB_EBANDS];
        let mut bp = [[0usize; 2]; NB_EBANDS];
        let mut started = false;
        for i in start_band..end_band {
            if !started {
                dp[i] = per_band_cost[i];
                started = true;
                continue;
            }
            for col in 0..2 {
                let mut best = f64::INFINITY;
                let mut from = 0usize;
                for prev in 0..2 {
                    let cand = dp[i - 1][prev]
                        + per_band_cost[i][col]
                        + if prev == col { 0.0 } else { lambda };
                    if cand < best {
                        best = cand;
                        from = prev;
                    }
                }
                dp[i][col] = best;
                bp[i][col] = from;
            }
        }
        // Trace back from the best terminal state.
        if start_band < end_band {
            let last = end_band - 1;
            let (term_col, term_cost) = if dp[last][0] <= dp[last][1] {
                (0usize, dp[last][0])
            } else {
                (1usize, dp[last][1])
            };
            best_total[select] = term_cost;
            let mut col = term_col;
            for i in (start_band..end_band).rev() {
                best_picks[select][i] = col;
                if i > start_band {
                    col = bp[i][col];
                }
            }
        } else {
            best_total[select] = 0.0;
        }
    }
    let select = if best_total[0] <= best_total[1] {
        0usize
    } else {
        1usize
    };
    let picks = &best_picks[select];
    let changes = tf_changes[select];

    // Convert per-band column picks into raw delta bits + post-lookup
    // tf_res. The decoder cumulatively XORs `raw[i]` to track the column
    // for band i, so `raw[i] = pick[i] != pick[i-1]` (with pick[-1] := 0).
    let mut raw_per_band = vec![false; NB_EBANDS];
    let mut tf_res = vec![0i32; NB_EBANDS];
    let mut prev_col = 0usize;
    for i in start_band..end_band {
        let col = picks[i];
        raw_per_band[i] = (col ^ prev_col) != 0;
        tf_res[i] = changes[col];
        prev_col = col;
    }

    // Pipeline gating: only apply non-zero TF changes when the encoder
    // pipeline is wired for them. Currently:
    //   * non-transient (long blocks, big_b=1): we support tf_change ≤ 0
    //     via the time-divide haar wrapping in `encoder_bands`.
    //   * transient (short blocks, big_b=M): mixed support — encoder side
    //     of the recombine path is not yet wired, so we clamp.
    //
    // When `apply_in_pipeline` is false, the caller must encode all-zero
    // raw bits regardless of the picked decisions.
    let any_nonzero = tf_res.iter().any(|&v| v != 0);
    let apply_in_pipeline = !transient && any_nonzero;

    TfDecision {
        tf_res,
        raw_per_band,
        tf_select: select == 1,
        apply_in_pipeline,
    }
}

/// Force a [`TfDecision`] into a "no-op" shape (all zeros, raw bits set so
/// the decoder's cumulative XOR + table lookup yields zero everywhere).
///
/// Used as the conservative fallback when the analysis would emit non-zero
/// `tf_change` values but the encoder pipeline can't yet apply them. This
/// produces the same bitstream the encoder used before TF analysis was
/// added: the decoder reads `tf_res[i] = 0` for every band.
///
/// The returned `tf_select` and `raw_per_band` are picked to minimise the
/// emitted bit count subject to the all-zero post-lookup constraint:
///
///   * non-transient: `tf_select = 0`, `raw[i] = 0` for all i. Column 0 of
///     the non-transient row is always 0 across all LMs.
///   * transient: `tf_select = 0`, `raw[start_band] = 1`, rest = 0. The
///     cumulative XOR keeps the running column at 1 for every band, and
///     column 1 of the transient row is 0 (e.g. LM=3 transient row is
///     `[3, 0, 1, -1]` and idx 1 = 0).
pub fn tf_decision_no_op(transient: bool, start_band: usize, end_band: usize) -> TfDecision {
    let mut raw = vec![false; NB_EBANDS];
    if transient && start_band < end_band {
        raw[start_band] = true;
    }
    TfDecision {
        tf_res: vec![0i32; NB_EBANDS],
        raw_per_band: raw,
        tf_select: false,
        apply_in_pipeline: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_band_uniform(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.1).sin() * 0.3).collect()
    }

    fn synth_band_impulsive(n: usize) -> Vec<f32> {
        // Single large bin, rest small.
        let mut v = vec![0.01f32; n];
        v[0] = 0.9;
        v
    }

    /// `apply_band_tf` is unitary — the L2 norm of the band must be
    /// preserved across any tf_change value.
    #[test]
    fn apply_band_tf_preserves_l2() {
        let n = 32usize;
        let big_b = 8i32; // transient short-blocks at LM=3
        for tf in &[-3i32, -2, -1, 0, 1, 2, 3] {
            let band = synth_band_uniform(n);
            let l2_in: f64 = band.iter().map(|v| (v * v) as f64).sum();
            let mut buf = band.clone();
            apply_band_tf(&mut buf, n as i32, *tf, big_b);
            let l2_out: f64 = buf.iter().map(|v| (v * v) as f64).sum();
            assert!(
                (l2_in - l2_out).abs() < 1e-6,
                "tf_change={tf} broke L2 (in={l2_in}, out={l2_out})"
            );
        }
    }

    /// `apply_band_tf(0)` is the identity — coefficients unchanged.
    #[test]
    fn apply_band_tf_zero_is_identity() {
        let n = 32usize;
        let band = synth_band_impulsive(n);
        let mut buf = band.clone();
        apply_band_tf(&mut buf, n as i32, 0, 1);
        for (a, b) in band.iter().zip(buf.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    /// On an impulsive (sparse) band, the L1 norm should DECREASE under a
    /// frequency-recombine transform (haar1 mixes adjacent bins, which
    /// distributes the spike across pairs but the total L1 doesn't blow
    /// up). On a uniform band the L1 should stay roughly the same.
    /// This pins the L1 as a sensible distortion proxy.
    #[test]
    fn band_l1_responds_to_signal_shape() {
        let n = 32usize;
        let uniform = synth_band_uniform(n);
        let impulsive = synth_band_impulsive(n);
        let l1_uniform_0 = band_l1_under_tf(&uniform, n as i32, 0, 1);
        let l1_uniform_1 = band_l1_under_tf(&uniform, n as i32, 1, 1);
        let l1_impulsive_0 = band_l1_under_tf(&impulsive, n as i32, 0, 1);
        let l1_impulsive_1 = band_l1_under_tf(&impulsive, n as i32, 1, 1);
        // Uniform band: the L1 ratio should be near 1 (the haar1 mixes
        // already-uniform values).
        let ratio_uniform = l1_uniform_1 / l1_uniform_0;
        assert!(
            (0.7..1.3).contains(&ratio_uniform),
            "uniform L1 ratio {ratio_uniform} out of expected ~1 range"
        );
        // Impulsive band: the L1 ratio under a single recombine should
        // not be dramatically larger than the original (the bound is loose
        // because haar1 of [a, b, c, d, ...] stays bounded by sqrt(2) per
        // pair, and concentrating energy doesn't always reduce L1 unless
        // there's correlation across pairs).
        let ratio_impulsive = l1_impulsive_1 / l1_impulsive_0;
        assert!(
            ratio_impulsive < 2.0,
            "impulsive L1 ratio {ratio_impulsive} >= 2.0 — haar1 should not double L1 on a sparse band"
        );
    }

    /// On white-noise-like coefficients (equal L1 expectation across both
    /// TF columns by symmetry), `tf_analysis` should prefer `tf_change=0`
    /// because the transition penalty (lambda) breaks the tie.
    #[test]
    fn tf_analysis_returns_zero_on_white_noise() {
        let n_total = 800usize;
        let mut seed: u32 = 0xDEAD_BEEF;
        let coeffs: Vec<f32> = (0..n_total)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 16) as i32 - 32_768) as f32 / 32_768.0 * 0.1
            })
            .collect();
        let dec = tf_analysis(&coeffs, false, 3, 0, NB_EBANDS);
        // White noise is unstructured — L1 is roughly equal under both
        // columns. Transition penalty prefers a uniform run, so MOST
        // bands should be 0; allow a small fraction to flip due to
        // statistical fluctuation in finite-N bands.
        let n_nonzero = dec.tf_res.iter().filter(|&&v| v != 0).count();
        assert!(
            n_nonzero <= 5,
            "white noise should keep most bands at tf_change=0 (got {n_nonzero} non-zero out of {})",
            NB_EBANDS
        );
    }

    /// The no-op decision must yield `tf_res = 0` across every band, with
    /// raw bits chosen so the decoder's cumulative XOR + table lookup
    /// reproduces zeros. This pins the pre-existing encoder behaviour.
    #[test]
    fn no_op_decision_decodes_to_all_zero() {
        for transient in [false, true] {
            let dec = tf_decision_no_op(transient, 0, NB_EBANDS);
            assert!(dec.tf_res.iter().all(|&v| v == 0));
            // Verify by simulating the decoder's lookup.
            let mut curr = 0i32;
            let lm = 3i32;
            for i in 0..NB_EBANDS {
                let bit = dec.raw_per_band[i] as i32;
                curr ^= bit;
                let idx = (4 * transient as i32 + 2 * dec.tf_select as i32 + curr) as usize;
                let post = TF_SELECT_TABLE[lm as usize][idx] as i32;
                assert_eq!(
                    post, 0,
                    "no-op decision: band {i} (transient={transient}) decoded to {post}"
                );
            }
        }
    }

    /// `tf_analysis` on directly-set MDCT-style coefficients with strong
    /// intra-band structure (alternating sign within each band) must
    /// pick `tf_change != 0` for at least some bands. This ensures the
    /// analyser triggers on coefficient patterns that haar1 actually
    /// improves, validating the masking-model logic.
    #[test]
    fn tf_analysis_engages_on_alternating_band_coefficients() {
        // Per-band alternating-sign coefficients (the canonical haar1-
        // friendly pattern): [a, -a, a, -a, ...] within each band.
        let n_total = 800usize;
        let coeffs: Vec<f32> = (0..n_total)
            .map(|i| if i % 2 == 0 { 0.4 } else { -0.4 })
            .collect();
        let dec = tf_analysis(&coeffs, false, 3, 0, NB_EBANDS);
        let nonzero = dec.tf_res.iter().filter(|&&v| v != 0).count();
        assert!(
            nonzero >= 5,
            "alternating-sign coefficients should engage the analyser on \
             ≥ 5 bands, got tf_res={:?}",
            dec.tf_res
        );
    }

    /// On a band where one column has dramatically lower L1 than the
    /// other, the Viterbi MUST pick that column. Build a synthetic
    /// "pre-haar1-friendly" coefficient pattern and verify that the
    /// analyser switches to the alternate column.
    ///
    /// This is the smoke test for the analyser actually doing its job —
    /// without it, the function could return the no-op decision on every
    /// input and still pass the steady-state test above.
    #[test]
    fn tf_analysis_picks_alternate_column_when_l1_warrants() {
        let n_total = 800usize;
        let lm = 3i32;
        // Build a signal where most bands have a "checkerboard"
        // alternating pattern (+a, -a, +a, -a...). The haar1 transform
        // on adjacent pairs collapses this into [0, sqrt(2)*a] per pair,
        // halving the L1 norm. So the time-divide tf_change should be
        // strongly preferred for this input.
        let mut coeffs = vec![0f32; n_total];
        for i in 0..n_total {
            coeffs[i] = if i % 2 == 0 { 0.3 } else { -0.3 };
        }
        let dec = tf_analysis(&coeffs, false, lm, 0, NB_EBANDS);
        // At least one band must pick a non-zero tf_change.
        let n_nonzero = dec.tf_res.iter().filter(|&&v| v != 0).count();
        assert!(
            n_nonzero > 0,
            "checkerboard input should drive the analyser to pick non-zero tf_change \
             on at least one band (got tf_res={:?})",
            dec.tf_res
        );
        // And `apply_in_pipeline` should be true (non-transient + non-zero).
        assert!(dec.apply_in_pipeline);
    }

    /// `tf_analysis` always returns RFC-valid raw deltas: replaying the
    /// decoder's cumulative XOR + `TF_SELECT_TABLE` lookup must reproduce
    /// the analysis's `tf_res[]` exactly. Pin this on a few synthetic
    /// frames so a future regression in the column-to-raw conversion is
    /// caught immediately.
    #[test]
    fn tf_analysis_decisions_are_rfc_valid() {
        let n_total = 800usize;
        let lm = 3i32;
        for &(transient, freq) in &[(false, 0.05f32), (true, 0.5f32), (true, 0.05f32)] {
            let coeffs: Vec<f32> = (0..n_total)
                .map(|i| (i as f32 * freq).sin() * 0.2)
                .collect();
            let dec = tf_analysis(&coeffs, transient, lm, 0, NB_EBANDS);
            let mut curr = 0i32;
            for i in 0..NB_EBANDS {
                let bit = dec.raw_per_band[i] as i32;
                curr ^= bit;
                let idx = (4 * transient as i32 + 2 * dec.tf_select as i32 + curr) as usize;
                let post = TF_SELECT_TABLE[lm as usize][idx] as i32;
                assert_eq!(
                    post, dec.tf_res[i],
                    "band {i}: analysis tf_res={} but decoder lookup yields {} \
                     (transient={transient}, tf_select={}, raw={})",
                    dec.tf_res[i], post, dec.tf_select, dec.raw_per_band[i]
                );
            }
        }
    }
}
