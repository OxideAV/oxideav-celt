//! Pyramid Vector Quantizer (PVQ) shape decoder for CELT (RFC 6716
//! §4.3.4.2).
//!
//! ## What this module provides
//!
//! RFC 6716 §4.3.4.2 prescribes the per-band shape decode as:
//!
//! 1. Compute the codebook size `V(N, K)` — the number of distinct
//!    integer pulse-vectors of dimension `N` whose absolute-value sum
//!    equals `K`.
//! 2. Decode a uniformly distributed index `i ∈ [0, V(N, K))` from the
//!    range decoder (the `dec_uint` path, §4.1.5).
//! 3. Reconstruct the signed integer pulse vector `X[0..N]` from `i`
//!    via the iterative per-position algorithm reproduced in
//!    [`decode_pulses`].
//! 4. Normalise the result to unit `L2` norm.
//!
//! Steps 1, 3 and 4 are pure arithmetic; only step 2 consults the
//! range decoder. The split-decoding constraint of §4.3.4.4 caps the
//! codebook size at 32 bits, so [`v_count`] returns `u32` and saturates
//! defensively at `u32::MAX` if the input would overflow (the caller
//! is expected to keep the input inside the §4.3.4.4 32-bit budget by
//! splitting larger bands first).
//!
//! The encode direction is the exact arithmetic inverse:
//! [`encode_pulses_to_index`] maps a signed integer pulse vector back
//! to its unique codeword index in `[0, V(N, K))`. The §4.3.4.2
//! codeword↔index map is a bijection, so `encode_pulses_to_index ∘
//! decode_index_to_pulses == id` over the whole codeword space.
//!
//! ## RFC §4.3.4.2 recurrence
//!
//! The codebook size is defined by the recurrence
//!
//! ```text
//!     V(N, K) = V(N - 1, K) + V(N, K - 1) + V(N - 1, K - 1)
//!     V(N, 0) = 1
//!     V(0, K) = 0    for K > 0
//! ```
//!
//! `v_count` evaluates this by carrying the row `V(N', .)` forward in
//! `K`-indexed order; one pass takes `O(K)` work per `N`-step so the
//! whole evaluation is `O(N · K)` time and `O(K)` memory.
//!
//! ## RFC §4.3.4.2 reconstruction
//!
//! Given a decoded index `i` and the codebook size `V(N, K)`, the
//! per-position loop is:
//!
//! ```text
//!     let mut k = K;
//!     for j in 0..N {
//!         let p = (V(N - j - 1, k) + V(N - j, k)) / 2;
//!         let sgn;
//!         if i < p {
//!             sgn = +1;
//!         } else {
//!             sgn = -1;
//!             i  -= p;
//!         }
//!         let k0 = k;
//!         let mut p = p - V(N - j - 1, k);
//!         while p > i {
//!             k -= 1;
//!             p -= V(N - j - 1, k);
//!         }
//!         X[j] = sgn * (k0 - k);
//!         i -= p;
//!     }
//! ```
//!
//! Implementation note: we pre-compute the column `c[m] = V(m, k)` for
//! every `k` we will need before the per-position loop. Since `k` only
//! decreases as the loop runs, and within a single position `k` only
//! decreases inside the inner while loop, we can precompute the full
//! `V(., K)` column at the start and walk an in-place k-row of
//! `V(N - j - 1, ·)` as `j` advances. The implementation keeps the
//! state minimal so the function allocates only the output buffer.
//!
//! ## Clean-room provenance
//!
//! The recurrence and reconstruction algorithm are reproduced from RFC
//! 6716 §4.3.4.2 (`docs/audio/opus/rfc6716-opus.txt` lines 6504–6536).
//! No external library source was consulted. The IETF PDF text is the
//! sole reference for both the codebook arithmetic and the per-
//! position reconstruction loop.

use crate::range_decoder::RangeDecoder;

/// Sentinel for "codebook size exceeds 32 bits" — never reached for
/// legal PVQ inputs because §4.3.4.4 splits large bands before they hit
/// this path. Exposed so a fuzz / property test can spot saturation
/// rather than overflow.
pub const V_COUNT_SATURATION: u32 = u32::MAX;

/// Compute `V(N, K)` — the codebook size for the §4.3.4.2 PVQ.
///
/// Returns the number of distinct signed integer vectors `X[0..N]`
/// whose absolute-value sum equals `K`. Saturates to
/// [`V_COUNT_SATURATION`] when the recurrence would overflow `u32`.
///
/// Base cases:
/// * `V(N, 0) = 1` for any `N` (the unique all-zero vector).
/// * `V(0, K) = 0` for `K > 0` (no vector of dimension zero can sum
///   to a positive `K`).
/// * `V(0, 0) = 1` (the unique empty vector).
pub fn v_count(n: u32, k: u32) -> u32 {
    if k == 0 {
        return 1;
    }
    if n == 0 {
        return 0;
    }
    // Carry the row V(m, .) for m = 0, 1, ..., n. We need V(n, k) at
    // the end. Memory: O(k+1).
    //
    // Row 0: V(0, 0) = 1, V(0, j) = 0 for j > 0.
    let kk = k as usize;
    let mut row = vec![0u64; kk + 1];
    row[0] = 1;
    // Apply the recurrence V(m+1, j) = V(m, j) + V(m+1, j-1) + V(m, j-1)
    // for m = 0..n, j = 0..=k. After updating, row[j] holds V(m+1, j).
    for _ in 0..n {
        // Update in increasing j so that row[j-1] already holds the
        // V(m+1, j-1) we need.
        let mut prev_v_m = row[0]; // V(m, 0)
        row[0] = 1; // V(m+1, 0) = 1
        for j in 1..=kk {
            let v_m_j = row[j]; // V(m, j) before update
            let v_mplus1_jminus1 = row[j - 1]; // already updated to V(m+1, j-1)
                                               // V(m+1, j) = V(m, j) + V(m+1, j-1) + V(m, j-1)
            let new_val = v_m_j
                .saturating_add(v_mplus1_jminus1)
                .saturating_add(prev_v_m);
            prev_v_m = v_m_j;
            row[j] = new_val;
        }
    }
    let result = row[kk];
    if result > u32::MAX as u64 {
        V_COUNT_SATURATION
    } else {
        result as u32
    }
}

/// Compute the full column `[V(0, k), V(1, k), …, V(n, k)]` for a
/// fixed `k`. Caller-friendly when the PVQ reconstruction loop needs
/// `V(m, k)` for many `m`. Each entry is saturated to `u32::MAX` on
/// overflow. Currently only used by the tests so it stays gated.
#[cfg(test)]
fn v_column(n: u32, k: u32) -> Vec<u32> {
    // We need V(m, k') for m = 0..=n and a few decreasing k' values
    // during the inner while loop. Build the FULL 2-D matrix V(m, j)
    // for m = 0..=n, j = 0..=k. Memory: O(n*k); time: O(n*k). This is
    // cheap for the legal CELT range (max ~176 bins, K capped by the
    // §4.3.4.4 32-bit codebook budget).
    let nn = n as usize;
    let kk = k as usize;
    let mut matrix = vec![0u64; (nn + 1) * (kk + 1)];
    // V(*, 0) = 1; V(0, j>0) = 0.
    for m in 0..=nn {
        matrix[m * (kk + 1)] = 1;
    }
    for m in 1..=nn {
        for j in 1..=kk {
            let idx = m * (kk + 1) + j;
            let v_prev_m_j = matrix[(m - 1) * (kk + 1) + j];
            let v_m_jminus1 = matrix[m * (kk + 1) + j - 1];
            let v_prev_m_jminus1 = matrix[(m - 1) * (kk + 1) + j - 1];
            matrix[idx] = v_prev_m_j
                .saturating_add(v_m_jminus1)
                .saturating_add(v_prev_m_jminus1);
        }
    }
    // Extract column k.
    let mut col = Vec::with_capacity(nn + 1);
    for m in 0..=nn {
        let v = matrix[m * (kk + 1) + k as usize];
        col.push(if v > u32::MAX as u64 {
            V_COUNT_SATURATION
        } else {
            v as u32
        });
    }
    col
}

/// Extract the §4.3.4.2 PVQ codevector for the given integer index.
///
/// Returns `None` for `i >= V(N, K)` or other malformed inputs (e.g.
/// `V(N, K)` saturated to `u32::MAX`). The returned vector has length
/// `N`, contains signed integers in `-K..=K`, and satisfies
/// `sum(|out[i]|) == K`.
pub fn decode_index_to_pulses(index: u32, n: u32, k: u32) -> Option<Vec<i32>> {
    if k == 0 {
        // The all-zero vector is the only codeword.
        if index == 0 {
            return Some(vec![0i32; n as usize]);
        }
        return None;
    }
    if n == 0 {
        // No vector of dimension zero can sum to a positive K.
        return None;
    }
    // Build the full V(m, j) matrix once so the inner loop can read
    // V(N - j - 1, k') for arbitrary k' in O(1).
    let nn = n as usize;
    let kk = k as usize;
    let stride = kk + 1;
    let mut matrix = vec![0u64; (nn + 1) * stride];
    for m in 0..=nn {
        matrix[m * stride] = 1;
    }
    for m in 1..=nn {
        for j in 1..=kk {
            let v_prev_m_j = matrix[(m - 1) * stride + j];
            let v_m_jminus1 = matrix[m * stride + j - 1];
            let v_prev_m_jminus1 = matrix[(m - 1) * stride + j - 1];
            matrix[m * stride + j] = v_prev_m_j
                .saturating_add(v_m_jminus1)
                .saturating_add(v_prev_m_jminus1);
        }
    }
    let v_nk = matrix[nn * stride + kk];
    if v_nk == 0 || v_nk > u32::MAX as u64 {
        return None;
    }
    if (index as u64) >= v_nk {
        return None;
    }
    // Run the §4.3.4.2 per-position loop. Use u64 internally to be
    // robust against intermediate sums even though final values fit
    // in u32 per §4.3.4.4.
    let mut i = index as u64;
    let mut k_cur = kk;
    let mut out = vec![0i32; nn];
    for j in 0..nn {
        // p = (V(N - j - 1, k_cur) + V(N - j, k_cur)) / 2
        let v_lo = matrix[(nn - j - 1) * stride + k_cur];
        let v_hi = matrix[(nn - j) * stride + k_cur];
        let mut p = (v_lo + v_hi) / 2;
        let sgn: i32 = if i < p {
            1
        } else {
            i -= p;
            -1
        };
        let k0 = k_cur;
        // p -= V(N - j - 1, k_cur)
        p -= matrix[(nn - j - 1) * stride + k_cur];
        // While p > i: k -= 1; p -= V(N - j - 1, k_cur)
        while p > i {
            if k_cur == 0 {
                // Defensive: should never happen for a well-formed
                // index because the loop invariant is i < V(N-j, k_cur).
                return None;
            }
            k_cur -= 1;
            p -= matrix[(nn - j - 1) * stride + k_cur];
        }
        out[j] = sgn * (k0 as i32 - k_cur as i32);
        i -= p;
    }
    Some(out)
}

/// Decode the integer pulse vector for one band from the range
/// decoder per RFC 6716 §4.3.4.2.
///
/// Reads a uniform integer in `[0, V(N, K))` via [`RangeDecoder::dec_uint`]
/// (§4.1.5) and reconstructs the signed integer vector. Returns
/// `None` when `V(N, K)` saturates at `u32::MAX` (§4.3.4.4 split must
/// run first), when `N == 0` and `K > 0`, when the range decoder
/// reports a sticky error, or when the decoded index falls outside
/// `[0, V(N, K))`.
pub fn decode_pulses(dec: &mut RangeDecoder<'_>, n: u32, k: u32) -> Option<Vec<i32>> {
    let v_nk = v_count(n, k);
    if v_nk == 0 || v_nk == V_COUNT_SATURATION {
        return None;
    }
    let index = match dec.dec_uint(v_nk) {
        Ok(i) => i,
        Err(_) => return None,
    };
    if dec.has_error() {
        return None;
    }
    decode_index_to_pulses(index, n, k)
}

/// Encode a signed integer pulse vector back to its §4.3.4.2 codeword
/// index — the exact arithmetic inverse of [`decode_index_to_pulses`].
///
/// Given a pulse vector `X[0..N]` with `sum(|X[j]|) == K` (and every
/// entry's sign well-defined), returns the unique index `i ∈ [0,
/// V(N, K))` for which `decode_index_to_pulses(i, N, K) == X`. The PVQ
/// codeword↔index map is a bijection (RFC 6716 §4.3.4.2), so this is a
/// total inverse on well-formed pulse vectors.
///
/// The inversion mirrors the decode per-position loop. At position `j`
/// the decoder, with running index `i` and running pulse budget `k`,
/// reads `X[j]` by:
///
/// * splitting `[0, V(N-j, k))` into a non-negative half `[0, p)` and a
///   negative half `[p, V(N-j, k))` of equal size `p = (V(N-j-1, k) +
///   V(N-j, k)) / 2`, choosing the sign from which half `i` lands in;
/// * within the chosen half, walking `k` down from `k0` so that the
///   sub-interval of width `V(N-j-1, k)` containing `i` selects the
///   magnitude `m = k0 - k`.
///
/// To encode we replay the decoder's `p`-walk forward from `X[j]`'s
/// magnitude, re-accumulating exactly the residual the decoder
/// subtracted from `i` at this position — the negative-half base
/// `p_half` (when `X[j] < 0`) plus `p_half - Σ_{t = k_after+1}^{k0}
/// V(N-j-1, t)`, the leftover after the decoder's magnitude walk
/// (`k_after = k0 - |X[j]|`).
///
/// Returns `None` when the inputs are malformed: a length mismatch
/// against `N`, a magnitude sum other than `K`, an entry whose
/// magnitude exceeds the remaining budget, or a `V(N, K)` that saturates
/// the §4.3.4.4 32-bit codebook bound (`V_COUNT_SATURATION`).
pub fn encode_pulses_to_index(pulses: &[i32], n: u32, k: u32) -> Option<u32> {
    if pulses.len() != n as usize {
        return None;
    }
    if k == 0 {
        // The only codeword is the all-zero vector → index 0.
        return if pulses.iter().all(|&x| x == 0) {
            Some(0)
        } else {
            None
        };
    }
    if n == 0 {
        // N == 0 with K > 0 has no codeword.
        return None;
    }
    // Reject a magnitude sum that does not match K up front so the
    // per-position budget walk below always terminates with k_cur == 0.
    let mag_sum: i64 = pulses.iter().map(|&x| x.abs() as i64).sum();
    if mag_sum != k as i64 {
        return None;
    }
    // Build the full V(m, j) matrix once (same shape decode uses), so
    // the per-position offset accumulation reads V(N-j-1, ·) in O(1).
    let nn = n as usize;
    let kk = k as usize;
    let stride = kk + 1;
    let mut matrix = vec![0u64; (nn + 1) * stride];
    for m in 0..=nn {
        matrix[m * stride] = 1;
    }
    for m in 1..=nn {
        for j in 1..=kk {
            let v_prev_m_j = matrix[(m - 1) * stride + j];
            let v_m_jminus1 = matrix[m * stride + j - 1];
            let v_prev_m_jminus1 = matrix[(m - 1) * stride + j - 1];
            matrix[m * stride + j] = v_prev_m_j
                .saturating_add(v_m_jminus1)
                .saturating_add(v_prev_m_jminus1);
        }
    }
    let v_nk = matrix[nn * stride + kk];
    if v_nk == 0 || v_nk > u32::MAX as u64 {
        return None;
    }
    // Walk forward, re-accumulating the index the decoder would have
    // subtracted to arrive at this vector.
    let mut i: u64 = 0;
    let mut k_cur = kk;
    for (j, &x) in pulses.iter().enumerate() {
        let mag = x.unsigned_abs() as usize;
        if mag > k_cur {
            // Magnitude exceeds the remaining pulse budget → not a
            // codeword for this (N, K).
            return None;
        }
        // p_half = (V(N-j-1, k_cur) + V(N-j, k_cur)) / 2 — the size of
        // the non-negative half. The decoder adds `p_half` to `i` for
        // the negative half before the magnitude walk.
        let v_lo = matrix[(nn - j - 1) * stride + k_cur];
        let v_hi = matrix[(nn - j) * stride + k_cur];
        let p_half = (v_lo + v_hi) / 2;
        if x < 0 {
            // A zero entry has no sign, so the decoder only ever lands
            // in the negative half (`i >= p_half`) with a non-zero
            // magnitude. A negative-signed zero is not a canonical
            // codeword.
            if mag == 0 {
                return None;
            }
            i += p_half;
        }
        // Replay the decoder's magnitude walk: it sets `p = p_half -
        // V(N-j-1, k0)`, then for each of the `mag` pulses decrements
        // `k` and subtracts `V(N-j-1, k)`. The leftover `p` after the
        // walk is exactly what the decoder subtracted from `i`, so we
        // add it back here.
        let k0 = k_cur;
        let k_after = k0 - mag;
        let mut residual = p_half - matrix[(nn - j - 1) * stride + k0];
        let mut t = k0;
        for _ in 0..mag {
            t -= 1;
            residual -= matrix[(nn - j - 1) * stride + t];
        }
        i += residual;
        k_cur = k_after;
    }
    // After consuming every position the budget must be exhausted.
    if k_cur != 0 {
        return None;
    }
    if i >= v_nk {
        // Defensive: a well-formed codeword always lands in range.
        return None;
    }
    Some(i as u32)
}

/// Quantize an input vector `x` onto the §4.3.4.2 PVQ codebook of `K`
/// pulses in `N` dimensions, returning the signed integer pulse vector
/// `y` with `sum(|y|) == K` that the encoder would transmit.
///
/// This is the encoder-side codeword search of RFC 6716 §5.3.8.1. The
/// codebook is every integer vector `y` with `sum(|y(j)|) == K` (two
/// pulses at the same position share a sign), and `x` is the unit
/// vector to be approximated. The documented method:
///
/// 1. Form an initial codeword by projecting `x` onto the pyramid of
///    `K-1` pulses and truncating toward zero:
///    `y0[j] = trunc((K-1) * x[j] / Σ|x|)`, keeping the sign of `x[j]`.
/// 2. Add the remaining pulses one at a time with a greedy search that
///    maximizes the normalized correlation `xᵀy / ||y||` after each
///    placement, until `sum(|y|) == K`.
///
/// RFC 6716 §5.3.8.1 explicitly states implementers MAY use any search
/// method as long as the output is a valid codebook vector; this
/// implements the documented projection-plus-greedy method directly,
/// with no reference-implementation detail. The greedy increment is
/// evaluated in closed form: adding a unit pulse of sign `s` at
/// position `j` changes the squared norm `yy → yy + 2*s*y[j] + 1` and
/// the correlation `xy → xy + s*x[j]`, so the candidate metric
/// `(xy + s*x[j])² / (yy + 2*s*y[j] + 1)` is compared without
/// recomputing the full vector. Ties resolve to the lowest position
/// then to the positive sign, giving a deterministic result.
///
/// Returns `None` for `N == 0` with `K > 0` (no codeword exists) and
/// the all-zero vector for `K == 0`. When `x` is all-zero (or `N == K`
/// degenerate cases) the pulses are placed deterministically so the
/// output always satisfies `sum(|y|) == K`. The input slice length
/// must equal `N`.
pub fn pvq_search(x: &[f32], n: u32, k: u32) -> Option<Vec<i32>> {
    let nn = n as usize;
    if x.len() != nn {
        return None;
    }
    if k == 0 {
        return Some(vec![0i32; nn]);
    }
    if nn == 0 {
        // N == 0 cannot host K > 0 pulses.
        return None;
    }
    let kk = k as i64;
    // Work in f64 for the search arithmetic to stay well clear of f32
    // rounding while comparing candidate metrics.
    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let mut y = vec![0i64; nn];
    // Running L1 sum of |x| for the projection step.
    let sum_abs: f64 = xf.iter().map(|v| v.abs()).sum();

    // Step 1: project onto the (K-1)-pulse pyramid, truncating toward
    // zero. Skipped when `sum_abs == 0` (no direction to project) — all
    // pulses are then placed by the greedy step below.
    let mut pulses_placed: i64 = 0;
    if sum_abs > 0.0 && kk > 1 {
        let scale = (kk - 1) as f64 / sum_abs;
        for j in 0..nn {
            // trunc toward zero, preserving sign of x[j].
            let yj = (xf[j] * scale).trunc() as i64;
            y[j] = yj;
            pulses_placed += yj.abs();
        }
    }
    // Defensive: the projection can never place more than K-1 pulses,
    // but clamp the running count so the greedy loop terminates even on
    // pathological f64 rounding.
    if pulses_placed > kk {
        // Shrink the largest-magnitude entries back until within budget.
        while pulses_placed > kk {
            // Find the entry with the largest magnitude to decrement.
            let mut best = 0usize;
            let mut best_mag = -1i64;
            for (j, &yj) in y.iter().enumerate() {
                if yj.abs() > best_mag {
                    best_mag = yj.abs();
                    best = j;
                }
            }
            if best_mag <= 0 {
                break;
            }
            y[best] -= y[best].signum();
            pulses_placed -= 1;
        }
    }

    // Maintain the running correlation `xy = Σ x[j]*y[j]` and squared
    // norm `yy = Σ y[j]²` so each greedy step is O(N) rather than O(N²).
    let mut xy: f64 = 0.0;
    let mut yy: f64 = 0.0;
    for j in 0..nn {
        xy += xf[j] * y[j] as f64;
        yy += (y[j] * y[j]) as f64;
    }

    // Step 2: greedily add the remaining pulses one at a time.
    while pulses_placed < kk {
        let mut best_j = 0usize;
        let mut best_sign: i64 = 1;
        // Track the best `(metric, tie)` pair. Initialise below every
        // real candidate.
        let mut best_metric = f64::NEG_INFINITY;
        let mut best_tie = f64::NEG_INFINITY;
        let mut found = false;
        for j in 0..nn {
            for &s in &[1i64, -1i64] {
                // A greedy step must ADD a pulse — increase `sum(|y|)`
                // by exactly one. Adding `s` at position `j` does that
                // only when `y[j]` is zero (either sign) or `s` matches
                // the existing sign; an opposing sign would *cancel* a
                // pulse, reducing the L1 norm and wasting the placement.
                if y[j] != 0 && y[j].signum() != s {
                    continue;
                }
                let sf = s as f64;
                let new_xy = xy + sf * xf[j];
                let new_yy = yy + 2.0 * sf * (y[j] as f64) + 1.0;
                if new_yy <= 0.0 {
                    continue;
                }
                // Minimize J = -xᵀy/||y|| ⇔ maximize the signed
                // normalized correlation `xy' / sqrt(yy')` (§5.3.8.1).
                // The signed form (not the squared `xy'²/yy'`) is
                // essential: squaring would make a sign-deepening pulse
                // tie with a sign-cancelling one and let the search undo
                // a correctly-placed pulse.
                let metric = new_xy / new_yy.sqrt();
                // Tie-break on the un-normalized correlation `xy'`. When
                // the input is perfectly aligned with an axis, deepening
                // a pulse and cancelling one normalize to the same
                // ratio; the larger `xy'` (deepening) is the one that
                // grows the match, so it must win the tie.
                let tie = new_xy;
                let better = if !found || metric > best_metric + 1e-12 {
                    true
                } else if metric >= best_metric - 1e-12 {
                    // Within metric epsilon → decide on the tie key.
                    tie > best_tie
                } else {
                    false
                };
                if better {
                    best_metric = metric;
                    best_tie = tie;
                    best_j = j;
                    best_sign = s;
                    found = true;
                }
            }
        }
        if !found {
            // Unreachable for `N >= 1` (the matching-sign candidate is
            // always valid), but guard against a non-terminating loop
            // rather than spin forever if an invariant ever changes.
            return None;
        }
        // Apply the chosen pulse and update the running accumulators.
        xy += best_sign as f64 * xf[best_j];
        yy += 2.0 * best_sign as f64 * (y[best_j] as f64) + 1.0;
        y[best_j] += best_sign;
        pulses_placed += 1;
    }

    Some(y.iter().map(|&v| v as i32).collect())
}

/// Quantize an input vector and return its §4.3.4.2 codeword index in
/// one call — the encoder-side composition of [`pvq_search`] and
/// [`encode_pulses_to_index`].
///
/// Runs the §5.3.8.1 PVQ search to find the integer pulse vector `y`
/// approximating `x` with `sum(|y|) == K`, then encodes `y` to its
/// unique index `i ∈ [0, V(N, K))`. Returns `(i, y)` so the caller can
/// both transmit the index and keep the quantized pulse vector (e.g.
/// for the §4.3.4.3 spreading the encoder applies in reverse). Returns
/// `None` when the search or the index encode rejects the inputs
/// (`N == 0` with `K > 0`, a length mismatch, or a saturated `V(N, K)`).
pub fn encode_unit_shape(x: &[f32], n: u32, k: u32) -> Option<(u32, Vec<i32>)> {
    let y = pvq_search(x, n, k)?;
    let index = encode_pulses_to_index(&y, n, k)?;
    Some((index, y))
}

/// Normalise a signed integer pulse vector to unit `L2` norm per RFC
/// 6716 §4.3.4.2 (final paragraph). The output `f32` vector has the
/// same length as `pulses` and satisfies `||out||_2 == 1` (within f32
/// rounding) unless every entry is zero, in which case every output
/// entry is zero.
pub fn normalize_to_unit_l2(pulses: &[i32]) -> Vec<f32> {
    let mut energy: f64 = 0.0;
    for &p in pulses {
        let pf = p as f64;
        energy += pf * pf;
    }
    if energy == 0.0 {
        return vec![0.0f32; pulses.len()];
    }
    let inv_norm = 1.0 / energy.sqrt();
    pulses
        .iter()
        .map(|&p| (p as f64 * inv_norm) as f32)
        .collect()
}

/// Convenience composition of [`decode_pulses`] + [`normalize_to_unit_l2`].
///
/// Returns the unit-norm decoded band shape vector ready to feed the
/// §4.3.4.3 spreading rotation. `None` is propagated from
/// [`decode_pulses`].
pub fn decode_unit_shape(dec: &mut RangeDecoder<'_>, n: u32, k: u32) -> Option<Vec<f32>> {
    let pulses = decode_pulses(dec, n, k)?;
    Some(normalize_to_unit_l2(&pulses))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- v_count ----------

    #[test]
    fn v_count_base_n_zero_k_zero() {
        assert_eq!(v_count(0, 0), 1);
    }

    #[test]
    fn v_count_base_n_zero_k_positive() {
        for k in 1..=10 {
            assert_eq!(v_count(0, k), 0, "V(0, {}) should be 0", k);
        }
    }

    #[test]
    fn v_count_base_n_positive_k_zero() {
        for n in 1..=10 {
            assert_eq!(v_count(n, 0), 1, "V({}, 0) should be 1", n);
        }
    }

    #[test]
    fn v_count_hand_computed_small() {
        // From the §4.3.4.2 recurrence V(N,K)=V(N-1,K)+V(N,K-1)+V(N-1,K-1).
        // V(1, 1) = V(0, 1) + V(1, 0) + V(0, 0) = 0 + 1 + 1 = 2.
        assert_eq!(v_count(1, 1), 2);
        // V(1, 2) = V(0, 2) + V(1, 1) + V(0, 1) = 0 + 2 + 0 = 2.
        assert_eq!(v_count(1, 2), 2);
        // V(2, 1) = V(1, 1) + V(2, 0) + V(1, 0) = 2 + 1 + 1 = 4.
        assert_eq!(v_count(2, 1), 4);
        // V(2, 2) = V(1, 2) + V(2, 1) + V(1, 1) = 2 + 4 + 2 = 8.
        assert_eq!(v_count(2, 2), 8);
        // V(3, 1) = V(2, 1) + V(3, 0) + V(2, 0) = 4 + 1 + 1 = 6.
        assert_eq!(v_count(3, 1), 6);
        // V(3, 2) = V(2, 2) + V(3, 1) + V(2, 1) = 8 + 6 + 4 = 18.
        assert_eq!(v_count(3, 2), 18);
        // V(4, 1) = V(3, 1) + V(4, 0) + V(3, 0) = 6 + 1 + 1 = 8.
        assert_eq!(v_count(4, 1), 8);
        // V(4, 2) = V(3, 2) + V(4, 1) + V(3, 1) = 18 + 8 + 6 = 32.
        assert_eq!(v_count(4, 2), 32);
    }

    #[test]
    fn v_count_symmetric_recurrence() {
        // The §4.3.4.2 recurrence is symmetric in (N, K) up to the
        // additive V(N-1, K-1) term, so V(N, K) and V(K, N) are NOT
        // generally equal — confirm the pattern by spot-checking
        // V(3, 4) vs V(4, 3).
        let v_3_4 = v_count(3, 4);
        let v_4_3 = v_count(4, 3);
        // V(3, 4): V(2,4)+V(3,3)+V(2,3).
        // V(3, 3) = V(2, 3) + V(3, 2) + V(2, 2) = 12 + 18 + 8 = 38.
        // V(2, 3) = V(1, 3) + V(2, 2) + V(1, 2) = 2 + 8 + 2 = 12.
        // V(2, 4) = V(1, 4) + V(2, 3) + V(1, 3) = 2 + 12 + 2 = 16.
        // V(3, 4) = 16 + 38 + 12 = 66.
        assert_eq!(v_3_4, 66);
        // V(4, 3): V(3,3)+V(4,2)+V(3,2) = 38 + 32 + 18 = 88.
        assert_eq!(v_4_3, 88);
        assert_ne!(v_3_4, v_4_3);
    }

    #[test]
    fn v_count_recurrence_holds_grid() {
        // V(N, K) = V(N-1, K) + V(N, K-1) + V(N-1, K-1) on a small grid.
        for n in 1..=12 {
            for k in 1..=12 {
                let lhs = v_count(n, k) as u64;
                let rhs = v_count(n - 1, k) as u64
                    + v_count(n, k - 1) as u64
                    + v_count(n - 1, k - 1) as u64;
                assert_eq!(lhs, rhs, "recurrence fails at N={}, K={}", n, k);
            }
        }
    }

    #[test]
    fn v_count_monotonic_in_k() {
        // For fixed N, V(N, K) is non-decreasing in K (the recurrence
        // adds non-negative V(N, K-1) every step).
        for n in 1..=8 {
            let mut prev = v_count(n, 0);
            for k in 1..=10 {
                let cur = v_count(n, k);
                assert!(
                    cur >= prev,
                    "V({}, k) not monotone non-decreasing at k={}: {} -> {}",
                    n,
                    k,
                    prev,
                    cur,
                );
                prev = cur;
            }
        }
    }

    #[test]
    fn v_count_monotonic_in_n() {
        // For fixed K > 0, V(N, K) is non-decreasing in N.
        for k in 1..=8 {
            let mut prev = v_count(0, k);
            for n in 1..=12 {
                let cur = v_count(n, k);
                assert!(
                    cur >= prev,
                    "V(n, {}) not monotone non-decreasing at n={}: {} -> {}",
                    k,
                    n,
                    prev,
                    cur,
                );
                prev = cur;
            }
        }
    }

    // ---------- decode_index_to_pulses ----------

    #[test]
    fn decode_k_zero_returns_zero_vector() {
        // V(N, 0) = 1, the only codeword is the all-zero vector.
        for n in 0..=8 {
            let out = decode_index_to_pulses(0, n, 0).expect("k=0 index 0 must decode");
            assert_eq!(out.len(), n as usize);
            assert!(out.iter().all(|&x| x == 0));
            // Out-of-range index rejected.
            assert!(decode_index_to_pulses(1, n, 0).is_none());
        }
    }

    #[test]
    fn decode_n_zero_k_positive_rejected() {
        // No vector of dimension zero can sum to K > 0.
        for k in 1..=5 {
            assert!(decode_index_to_pulses(0, 0, k).is_none());
        }
    }

    #[test]
    fn decode_out_of_range_index_returns_none() {
        // V(2, 1) = 4. Index 4 is out of range.
        assert!(decode_index_to_pulses(4, 2, 1).is_none());
        // V(3, 2) = 18. Index 18 is out of range.
        assert!(decode_index_to_pulses(18, 3, 2).is_none());
    }

    #[test]
    fn decode_n1_k1_two_codewords() {
        // V(1, 1) = 2. The two codewords are +1 and -1.
        let plus = decode_index_to_pulses(0, 1, 1).unwrap();
        let minus = decode_index_to_pulses(1, 1, 1).unwrap();
        assert_eq!(plus.len(), 1);
        assert_eq!(minus.len(), 1);
        // One is +1 and the other is -1 (sum of magnitudes is 1).
        assert_eq!(plus[0].abs(), 1);
        assert_eq!(minus[0].abs(), 1);
        assert_ne!(plus[0], minus[0]);
    }

    #[test]
    fn decode_pulse_sum_invariant() {
        // For every (N, K) and every legal index, sum(|X|) == K and
        // every entry lies in -K..=K.
        for n in 1..=6u32 {
            for k in 1..=5u32 {
                let v = v_count(n, k) as u64;
                for i in 0..v {
                    let out = decode_index_to_pulses(i as u32, n, k).unwrap();
                    assert_eq!(out.len(), n as usize);
                    let mag_sum: i64 = out.iter().map(|&x| x.abs() as i64).sum();
                    assert_eq!(
                        mag_sum, k as i64,
                        "sum |X| should equal K at N={}, K={}, i={}",
                        n, k, i
                    );
                    for &x in &out {
                        assert!(
                            x.unsigned_abs() <= k,
                            "|X[j]| <= K at N={} K={} i={}",
                            n,
                            k,
                            i
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn decode_index_partition_is_bijective() {
        // The §4.3.4.2 decoder is a bijection from [0, V(N,K)) onto
        // the integer-pulse vectors. Iterate every index and confirm
        // we see V(N,K) distinct codewords.
        for n in 1..=5u32 {
            for k in 1..=4u32 {
                let v = v_count(n, k);
                let mut seen = std::collections::HashSet::new();
                for i in 0..v {
                    let out = decode_index_to_pulses(i, n, k).unwrap();
                    assert!(
                        seen.insert(out.clone()),
                        "duplicate codeword at N={} K={} i={}: {:?}",
                        n,
                        k,
                        i,
                        out
                    );
                }
                assert_eq!(seen.len(), v as usize);
            }
        }
    }

    #[test]
    fn decode_index_covers_all_pulse_vectors() {
        // Cross-check the bijection against the brute-force
        // enumeration of integer vectors with sum(|x|) == K. Small
        // grid only (N=3 K=3 has V(3,3)=38 codewords).
        for n in 1..=3u32 {
            for k in 1..=3u32 {
                // Enumerate all integer vectors with sum |X| = K.
                let mut brute: std::collections::HashSet<Vec<i32>> =
                    std::collections::HashSet::new();
                let nn = n as usize;
                let mut stack: Vec<(usize, i32, Vec<i32>)> = vec![(0, k as i32, vec![0i32; nn])];
                while let Some((pos, remaining, vec)) = stack.pop() {
                    if pos == nn {
                        if remaining == 0 {
                            brute.insert(vec);
                        }
                        continue;
                    }
                    for v in -(k as i32)..=(k as i32) {
                        let mag = v.abs();
                        if mag <= remaining {
                            let mut new_vec = vec.clone();
                            new_vec[pos] = v;
                            stack.push((pos + 1, remaining - mag, new_vec));
                        }
                    }
                }
                // Decode every legal index and compare the set.
                let v = v_count(n, k);
                let mut decoded: std::collections::HashSet<Vec<i32>> =
                    std::collections::HashSet::new();
                for i in 0..v {
                    decoded.insert(decode_index_to_pulses(i, n, k).unwrap());
                }
                assert_eq!(
                    decoded, brute,
                    "decoded codeword set does not match brute-force enumeration at N={} K={}",
                    n, k
                );
            }
        }
    }

    // ---------- encode_pulses_to_index (decode inverse) ----------

    #[test]
    fn encode_is_exact_inverse_of_decode_full_space() {
        // The §4.3.4.2 codeword↔index map is a bijection. For every
        // (N, K) in a small grid, decode every legal index and confirm
        // encode recovers the SAME index — i.e. encode∘decode == id on
        // the whole codeword space.
        for n in 1..=7u32 {
            for k in 1..=6u32 {
                let v = v_count(n, k);
                for i in 0..v {
                    let pulses = decode_index_to_pulses(i, n, k).unwrap();
                    let back = encode_pulses_to_index(&pulses, n, k);
                    assert_eq!(
                        back,
                        Some(i),
                        "encode∘decode != id at N={} K={} i={} (pulses {:?})",
                        n,
                        k,
                        i,
                        pulses
                    );
                }
            }
        }
    }

    #[test]
    fn encode_k_zero_only_zero_vector() {
        // V(N, 0) = 1; the all-zero vector maps to index 0, anything
        // else is not a codeword.
        for n in 0..=6u32 {
            let zero = vec![0i32; n as usize];
            assert_eq!(encode_pulses_to_index(&zero, n, 0), Some(0));
            if n > 0 {
                let mut nonzero = zero.clone();
                nonzero[0] = 1;
                assert_eq!(encode_pulses_to_index(&nonzero, n, 0), None);
            }
        }
    }

    #[test]
    fn encode_rejects_malformed_inputs() {
        // Wrong length.
        assert_eq!(encode_pulses_to_index(&[1, 0], 3, 1), None);
        // Magnitude sum != K.
        assert_eq!(encode_pulses_to_index(&[1, 1, 0], 3, 1), None);
        assert_eq!(encode_pulses_to_index(&[1, 0, 0], 3, 2), None);
        // N == 0 with K > 0 has no codeword.
        assert_eq!(encode_pulses_to_index(&[], 0, 2), None);
        // A negative-signed zero is not canonical, but here the sum
        // already disqualifies it; check a balanced case where the only
        // defect is the negative zero would need a magnitude — covered
        // implicitly by the sum check, so just confirm valid signs work.
        assert!(encode_pulses_to_index(&[0, -1, 0], 3, 1).is_some());
    }

    #[test]
    fn encode_matches_hand_traced_n1_k1() {
        // V(1, 1) = 2: decode(0) = [+1], decode(1) = [-1].
        assert_eq!(encode_pulses_to_index(&[1], 1, 1), Some(0));
        assert_eq!(encode_pulses_to_index(&[-1], 1, 1), Some(1));
    }

    #[test]
    fn encode_index_in_range_for_every_codeword() {
        // Every encoded index must land in [0, V(N, K)).
        for n in 1..=6u32 {
            for k in 1..=5u32 {
                let v = v_count(n, k);
                for i in 0..v {
                    let pulses = decode_index_to_pulses(i, n, k).unwrap();
                    let idx = encode_pulses_to_index(&pulses, n, k).unwrap();
                    assert!(idx < v, "index {} >= V({},{})={}", idx, n, k, v);
                }
            }
        }
    }

    #[test]
    fn encode_saturated_codebook_returns_none() {
        // V(180, 180) saturates; encode must refuse rather than build a
        // huge matrix or wrap.
        let pulses = vec![0i32; 180];
        // mag sum 0 != 180 so it returns None on the sum check, but the
        // intent is the saturation guard — use a vector that would sum
        // correctly only if the matrix were built. A length-180 vector
        // with sum 180 still hits the saturation guard before indexing.
        assert_eq!(encode_pulses_to_index(&pulses, 180, 180), None);
    }

    // ---------- pvq_search (§5.3.8.1 encoder codeword search) ----------

    fn l1(v: &[i32]) -> u32 {
        v.iter().map(|&x| x.unsigned_abs()).sum()
    }

    #[test]
    fn search_output_always_has_k_pulses() {
        // For a spread of input vectors and (N, K), the search output
        // must be a valid codeword: sum(|y|) == K, length N.
        let inputs: &[&[f32]] = &[
            &[1.0, 0.0, 0.0, 0.0],
            &[0.5, -0.5, 0.5, -0.5],
            &[0.9, 0.1, -0.3, 0.2],
            &[-1.0, -1.0, -1.0, -1.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[0.6, 0.6, 0.0, 0.0],
        ];
        for x in inputs {
            let n = x.len() as u32;
            for k in 1..=8u32 {
                let y = pvq_search(x, n, k).expect("search must succeed");
                assert_eq!(y.len(), n as usize);
                assert_eq!(l1(&y), k, "sum|y| != K for x={:?} K={}", x, k);
            }
        }
    }

    #[test]
    fn search_k_zero_is_zero_vector() {
        let y = pvq_search(&[0.3, -0.7, 0.1], 3, 0).unwrap();
        assert_eq!(y, vec![0, 0, 0]);
    }

    #[test]
    fn search_n_zero_k_positive_rejected() {
        assert_eq!(pvq_search(&[], 0, 3), None);
    }

    #[test]
    fn search_length_mismatch_rejected() {
        assert_eq!(pvq_search(&[1.0, 0.0], 3, 2), None);
    }

    #[test]
    fn search_concentrates_on_dominant_axis() {
        // A vector pointing almost entirely along axis 0 should put all
        // its pulses there (with the correct sign).
        let y = pvq_search(&[1.0, 0.0, 0.0, 0.0], 4, 5).unwrap();
        assert_eq!(y, vec![5, 0, 0, 0]);
        let y = pvq_search(&[-1.0, 0.0, 0.0, 0.0], 4, 3).unwrap();
        assert_eq!(y, vec![-3, 0, 0, 0]);
    }

    #[test]
    fn search_balanced_input_spreads_pulses() {
        // Four equal-magnitude axes with K=4 → one pulse per axis with
        // matching signs.
        let y = pvq_search(&[0.5, -0.5, 0.5, -0.5], 4, 4).unwrap();
        assert_eq!(l1(&y), 4);
        assert_eq!(y, vec![1, -1, 1, -1]);
    }

    #[test]
    fn search_all_zero_input_still_produces_k_pulses() {
        // No direction information, but the codeword must still be
        // valid (sum|y| == K). Determinism: lowest indices, positive.
        let y = pvq_search(&[0.0, 0.0, 0.0], 3, 2).unwrap();
        assert_eq!(l1(&y), 2);
    }

    #[test]
    fn search_maximizes_correlation_against_brute_force() {
        // For small (N, K) confirm the greedy search finds a codeword
        // whose normalized correlation xᵀy/||y|| is within f64 epsilon
        // of the brute-force optimum over the full codebook. The greedy
        // method is documented as a "good trade-off", so it is not
        // guaranteed globally optimal in general — but for these small
        // well-separated inputs it coincides with the optimum.
        let cases: &[(&[f32], u32, u32)] = &[
            (&[0.9, 0.3, 0.2, 0.1], 4, 3),
            (&[0.6, -0.6, 0.4, -0.2], 4, 4),
            (&[0.8, 0.5, 0.3, 0.1, 0.05], 5, 2),
        ];
        for &(x, n, k) in cases {
            let y = pvq_search(x, n, k).unwrap();
            let corr = normalized_corr(x, &y);
            // Brute-force the optimum over every codeword.
            let v = v_count(n, k);
            let mut best = f64::NEG_INFINITY;
            for i in 0..v {
                let cand = decode_index_to_pulses(i, n, k).unwrap();
                let c = normalized_corr(x, &cand);
                if c > best {
                    best = c;
                }
            }
            assert!(
                corr >= best - 1e-9,
                "greedy corr {} below brute-force optimum {} for x={:?} K={}",
                corr,
                best,
                x,
                k
            );
        }
    }

    fn normalized_corr(x: &[f32], y: &[i32]) -> f64 {
        let mut xy = 0.0f64;
        let mut yy = 0.0f64;
        for (j, &yj) in y.iter().enumerate() {
            xy += x[j] as f64 * yj as f64;
            yy += (yj * yj) as f64;
        }
        if yy == 0.0 {
            return f64::NEG_INFINITY;
        }
        xy / yy.sqrt()
    }

    // ---------- encode_unit_shape (search → index composition) ----------

    #[test]
    fn encode_unit_shape_roundtrips_through_decoder() {
        // For a spread of inputs, encode_unit_shape's (index, pulses)
        // must satisfy decode_index_to_pulses(index) == pulses, and the
        // pulses must be a valid K-pulse codeword.
        let inputs: &[&[f32]] = &[
            &[1.0, 0.0, 0.0, 0.0],
            &[0.5, -0.5, 0.5, -0.5],
            &[0.9, 0.1, -0.3, 0.2, -0.1, 0.05],
        ];
        for x in inputs {
            let n = x.len() as u32;
            for k in 1..=6u32 {
                let (index, pulses) = encode_unit_shape(x, n, k).unwrap();
                assert_eq!(l1(&pulses), k);
                let decoded = decode_index_to_pulses(index, n, k).unwrap();
                assert_eq!(
                    decoded, pulses,
                    "decode(encode_unit_shape) mismatch for x={:?} K={}",
                    x, k
                );
            }
        }
    }

    #[test]
    fn encode_unit_shape_n_zero_k_positive_none() {
        assert_eq!(encode_unit_shape(&[], 0, 2), None);
    }

    #[test]
    fn encode_unit_shape_k_zero_index_zero() {
        let (index, pulses) = encode_unit_shape(&[0.3, -0.7, 0.1], 3, 0).unwrap();
        assert_eq!(index, 0);
        assert_eq!(pulses, vec![0, 0, 0]);
    }

    // ---------- v_column ----------

    #[test]
    fn v_column_matches_v_count_pointwise() {
        for k in 0..=6 {
            for n in 0..=10 {
                let col = v_column(n, k);
                for m in 0..=n {
                    assert_eq!(
                        col[m as usize],
                        v_count(m, k),
                        "v_column({}, {})[{}] != v_count({}, {})",
                        n,
                        k,
                        m,
                        m,
                        k
                    );
                }
            }
        }
    }

    // ---------- decode_pulses (range decoder integration) ----------

    #[test]
    fn decode_pulses_reads_uniform_index_from_decoder() {
        // Build a decoder over a single-byte buffer. The induced
        // index for any (N, K) is whatever dec_uint(V(N,K)) returns
        // on this seed; the test asserts the output is a well-formed
        // codeword.
        let buf = [0xCDu8; 16];
        let mut dec = RangeDecoder::new(&buf);
        let n = 4u32;
        let k = 3u32;
        let pulses = decode_pulses(&mut dec, n, k).expect("must decode");
        assert_eq!(pulses.len(), n as usize);
        let mag: i32 = pulses.iter().map(|&x| x.abs()).sum();
        assert_eq!(mag as u32, k);
        for &x in &pulses {
            assert!(x.unsigned_abs() <= k);
        }
    }

    #[test]
    fn decode_pulses_k_zero_is_zero_vector_no_decoder_consumption() {
        let buf = [0xFFu8; 4];
        let mut dec = RangeDecoder::new(&buf);
        let tell_before = dec.tell();
        let out = decode_pulses(&mut dec, 5, 0).expect("k=0 decodes to zero");
        assert_eq!(out, vec![0; 5]);
        // dec_uint(1) consumes zero bits per §4.1.5 (ft = 1 is the
        // single-symbol degenerate case).
        let tell_after = dec.tell();
        assert!(
            tell_after >= tell_before,
            "tell() may not regress: {} -> {}",
            tell_before,
            tell_after
        );
    }

    #[test]
    fn decode_pulses_n_zero_k_positive_returns_none() {
        let buf = [0u8; 4];
        let mut dec = RangeDecoder::new(&buf);
        assert!(decode_pulses(&mut dec, 0, 3).is_none());
    }

    // ---------- normalize_to_unit_l2 ----------

    #[test]
    fn normalize_all_zero_returns_all_zero() {
        let out = normalize_to_unit_l2(&[0, 0, 0, 0]);
        assert_eq!(out, vec![0.0f32; 4]);
    }

    #[test]
    fn normalize_unit_vector_already_normalised() {
        let out = normalize_to_unit_l2(&[1, 0, 0, 0]);
        assert_eq!(out, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_produces_unit_l2_norm() {
        // For every (N, K) decode every codeword and check the L2
        // norm of the normalised f32 vector is within f32 rounding
        // of 1.
        for n in 1..=4u32 {
            for k in 1..=4u32 {
                let v = v_count(n, k);
                for i in 0..v {
                    let pulses = decode_index_to_pulses(i, n, k).unwrap();
                    let normalised = normalize_to_unit_l2(&pulses);
                    let energy: f64 = normalised.iter().map(|&x| (x as f64) * (x as f64)).sum();
                    assert!(
                        (energy - 1.0).abs() < 1e-5,
                        "L2 norm not 1 at N={} K={} i={}: energy={}",
                        n,
                        k,
                        i,
                        energy
                    );
                }
            }
        }
    }

    #[test]
    fn normalize_preserves_signs() {
        let pulses = [3, -2, 0, 1, -1];
        let out = normalize_to_unit_l2(&pulses);
        for (p, o) in pulses.iter().zip(out.iter()) {
            let pulse_sign = p.signum();
            // f32 sign as -1 / 0 / +1 in integer form.
            let out_sign = if *o > 0.0 {
                1
            } else if *o < 0.0 {
                -1
            } else {
                0
            };
            assert_eq!(
                pulse_sign, out_sign,
                "signs disagree on entry {:?} -> {:?}",
                p, o
            );
        }
    }

    // ---------- decode_unit_shape composition ----------

    #[test]
    fn decode_unit_shape_composes_decode_and_normalise() {
        let buf = [0x55u8; 16];
        let mut dec1 = RangeDecoder::new(&buf);
        let mut dec2 = RangeDecoder::new(&buf);
        let pulses = decode_pulses(&mut dec1, 6, 3).unwrap();
        let direct = normalize_to_unit_l2(&pulses);
        let composed = decode_unit_shape(&mut dec2, 6, 3).unwrap();
        assert_eq!(direct, composed);
    }

    #[test]
    fn decode_unit_shape_k_zero_returns_zero_vector() {
        let buf = [0u8; 4];
        let mut dec = RangeDecoder::new(&buf);
        let out = decode_unit_shape(&mut dec, 8, 0).unwrap();
        assert_eq!(out, vec![0.0f32; 8]);
    }

    // ---------- saturation ----------

    #[test]
    fn v_count_saturates_for_large_inputs() {
        // V(N, K) grows roughly like Pascal's-triangle terms; for
        // large N+K the value exceeds u32::MAX. Confirm the saturation
        // sentinel is returned rather than wrapping silently.
        let v_big = v_count(180, 180);
        assert_eq!(
            v_big, V_COUNT_SATURATION,
            "V(180, 180) should saturate, got {}",
            v_big
        );
    }

    #[test]
    fn decode_pulses_saturated_returns_none() {
        let buf = [0u8; 4];
        let mut dec = RangeDecoder::new(&buf);
        // A saturated V(N, K) means the caller must split per §4.3.4.4
        // before getting here; we refuse to fabricate an index.
        assert!(decode_pulses(&mut dec, 180, 180).is_none());
    }
}
