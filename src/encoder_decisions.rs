//! Encoder-side perceptual decisions: spread (rotation) and dynalloc band
//! boosts.
//!
//! These two decisions live outside the bit-allocator and PVQ shape coder —
//! they shape the *content* of those stages without touching their bit
//! layout. Both are RFC-valid for any encoder; they're just heuristics that
//! libopus runs and we now run too.
//!
//! ## Spread (RFC 6716 §4.3.4.4)
//!
//! The spread parameter (3-bit ICDF, Table 56) chooses one of
//! `{NONE, LIGHT, NORMAL, AGGRESSIVE}` and feeds the PVQ rotation step
//! ([`crate::bands::exp_rotation`]). The rotation pre-conditions the PVQ
//! search by mixing adjacent samples — without it, sparse PVQ vectors
//! produce audibly buzzy reconstruction on tonal content. Aggressive
//! spreading helps noise-like signals, while spread=NONE leaves clean tonal
//! peaks untouched. The libopus decision is based on a per-band
//! "tonality" measure — we use a clean-room equivalent.
//!
//! ## Dynalloc (RFC 6716 §4.3.3 dynalloc symbol)
//!
//! The bit allocator's base table ([`crate::tables::BAND_ALLOCATION`])
//! distributes pulses according to a fixed average loudness profile. When
//! a frame has a per-band energy spike that the profile doesn't anticipate
//! (e.g. an isolated high-frequency tone), the encoder can request a
//! per-band "boost" that the decoder reads back via the dynalloc loop.
//! The boost is signalled as 1..K bits per band: a single `false` means
//! "no boost" (cheap), and N `true`s followed by a `false` add `N * quanta`
//! bits to that band's pulse budget. The loop's per-bit cost falls from
//! `dynalloc_logp = 6` to `dynalloc_logp = 1` after the first `true`,
//! giving the boost a soft floor.

use crate::tables::{
    EBAND_5MS, NB_EBANDS, SPREAD_AGGRESSIVE, SPREAD_LIGHT, SPREAD_NONE, SPREAD_NORMAL,
};

/// Pick the per-frame spread mode from the post-MDCT, post-normalisation
/// shape coefficients.
///
/// Heuristic (clean-room — RFC §4.3.4.4 only specifies the bit layout, not
/// the encoder decision):
///
///   * Compute per-band peak-to-RMS ratio on the *normalised* shape (each
///     band's RMS is 1 by construction). A band where one bin carries most
///     of the energy has a high peak/RMS ratio (close to `sqrt(N)`); a
///     band with energy evenly spread across bins has a peak/RMS close to
///     `sqrt(1)`.
///   * Average the ratio across coded bands. Tonal frames score high,
///     noise-like frames score low.
///   * Map the score to one of {NONE, LIGHT, NORMAL, AGGRESSIVE} via two
///     thresholds tuned on synthetic test signals.
///
/// Caller passes the normalised shape `x` (length `n = 100*M` for full-band
/// CELT) and `lm`. Returns one of `SPREAD_NONE`, `SPREAD_LIGHT`,
/// `SPREAD_NORMAL`, or `SPREAD_AGGRESSIVE`.
pub fn spread_decision(x: &[f32], lm: i32, start_band: usize, end_band: usize) -> i32 {
    if end_band <= start_band {
        return SPREAD_NORMAL;
    }
    let m = 1i32 << lm;

    // Sum a tonality score across coded bands. Score per band:
    //   peak^2 / mean(x^2)  in [1, n]
    // We use squared values so we don't sqrt every band — the threshold is
    // tuned in the same domain.
    let mut score = 0f64;
    let mut counted = 0usize;
    for band in start_band..end_band {
        let lo = (m * EBAND_5MS[band] as i32) as usize;
        let hi = (m * EBAND_5MS[band + 1] as i32) as usize;
        if hi <= lo || hi > x.len() {
            continue;
        }
        let n = (hi - lo) as f64;
        if n < 2.0 {
            continue;
        }
        let mut peak = 0f64;
        let mut sum_sq = 0f64;
        for &v in &x[lo..hi] {
            let v = v as f64;
            let v2 = v * v;
            if v2 > peak {
                peak = v2;
            }
            sum_sq += v2;
        }
        if sum_sq <= 1e-30 {
            continue;
        }
        // peak/mean ranges from 1 (uniform) to n (Kronecker delta).
        // Normalise to [0, 1] so the threshold is independent of band width.
        let ratio = peak / (sum_sq / n);
        let normalised = ((ratio - 1.0) / (n - 1.0)).clamp(0.0, 1.0);
        score += normalised;
        counted += 1;
    }
    if counted == 0 {
        return SPREAD_NORMAL;
    }
    let mean = score / counted as f64;

    // Empirical mapping (calibrated against pure tone, white noise, music
    // mixtures). Pure tones land near 0.95-1.0 → SPREAD_NONE / LIGHT.
    // White noise lands near 0.0-0.05 → SPREAD_AGGRESSIVE.
    if mean > 0.85 {
        SPREAD_NONE
    } else if mean > 0.55 {
        SPREAD_LIGHT
    } else if mean > 0.15 {
        SPREAD_NORMAL
    } else {
        SPREAD_AGGRESSIVE
    }
}

/// Per-band dynalloc decision — pick the band that the encoder should
/// boost by one quanta. Returns the band index, or `None` to leave
/// every band at the (cheap) "no boost" emission.
///
/// Heuristic: pick the band whose normalised energy (`band_log_e[i]`,
/// already mean-removed via `E_MEANS`) most exceeds the median of the
/// coded bands. We require at least a ~1.5 dB excess (the median itself
/// captures most of the dynamic range; small deviations don't deserve a
/// boost). The caller computes the actual boost quanta from its own
/// per-channel width formula and emits the matching range-coder bits.
///
/// `band_log_e` covers the full NB_EBANDS range; only the slice
/// `[start_band, end_band)` is examined.
pub fn pick_dynalloc_boost_band(
    band_log_e: &[f32],
    start_band: usize,
    end_band: usize,
) -> Option<usize> {
    debug_assert!(end_band <= NB_EBANDS);
    if end_band <= start_band {
        return None;
    }
    let mut sorted: Vec<f32> = band_log_e[start_band..end_band].to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        0f32
    } else {
        sorted[sorted.len() / 2]
    };
    // ~6 dB excess in log2(linear) space (1.0 in log2 units). Bands sitting
    // near the median don't trigger a boost — the dynalloc emission costs
    // ~7 bits per boosted band, so the band must really stand out for the
    // pulse-budget shift to be a net win.
    const MIN_EXCESS_LOG2: f32 = 1.0;
    let mut best_band: Option<usize> = None;
    let mut best_excess: f32 = MIN_EXCESS_LOG2;
    for i in start_band..end_band {
        let excess = band_log_e[i] - median;
        if excess > best_excess {
            best_excess = excess;
            best_band = Some(i);
        }
    }
    let _ = EBAND_5MS;
    best_band
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_normalised_shape_per_band<F: Fn(usize, usize, usize) -> f32>(
        lm: i32,
        f: F,
    ) -> Vec<f32> {
        let m = 1usize << lm;
        let n = 100 * m;
        let mut x = vec![0f32; n];
        for band in 0..NB_EBANDS {
            let lo = (EBAND_5MS[band] as usize) * m;
            let hi = (EBAND_5MS[band + 1] as usize) * m;
            let bw = hi - lo;
            let mut sum_sq = 0f32;
            for i in lo..hi {
                let v = f(band, i - lo, bw);
                x[i] = v;
                sum_sq += v * v;
            }
            let g = if sum_sq > 0.0 {
                1.0 / sum_sq.sqrt()
            } else {
                0.0
            };
            for i in lo..hi {
                x[i] *= g;
            }
        }
        x
    }

    #[test]
    fn spread_picks_aggressive_on_white_noise() {
        // White noise: every bin equal magnitude → minimum tonality.
        let lm = 3;
        let x = build_normalised_shape_per_band(lm, |_band, _i, _bw| 1.0);
        let s = spread_decision(&x, lm, 0, NB_EBANDS);
        assert_eq!(s, SPREAD_AGGRESSIVE, "white noise must trip aggressive");
    }

    #[test]
    fn spread_picks_none_or_light_on_kronecker_band() {
        // One non-zero coefficient per band → maximum tonality.
        let lm = 3;
        let x = build_normalised_shape_per_band(lm, |_band, i, _bw| if i == 0 { 1.0 } else { 0.0 });
        let s = spread_decision(&x, lm, 0, NB_EBANDS);
        assert!(
            s == SPREAD_NONE || s == SPREAD_LIGHT,
            "Kronecker bands should suppress spreading (got {})",
            s
        );
    }

    #[test]
    fn spread_picks_normal_on_mixed_content() {
        // Each band: one large coefficient + uniform background → mid-
        // range tonality.
        let lm = 3;
        let x = build_normalised_shape_per_band(lm, |_band, i, bw| {
            if i == 0 {
                3.0
            } else if i < bw / 2 {
                1.0
            } else {
                0.5
            }
        });
        let s = spread_decision(&x, lm, 0, NB_EBANDS);
        assert!(
            s == SPREAD_LIGHT || s == SPREAD_NORMAL,
            "mixed content should land in mid-range (got {})",
            s
        );
    }

    #[test]
    fn dynalloc_boosts_outlier_band() {
        // band_log_e is mean-removed (after E_MEANS subtraction), so a
        // single peak above the rest is the typical pattern.
        let mut log_e = [0f32; NB_EBANDS];
        // Inject a ~12 dB outlier at band 10. Threshold is ~6 dB so this
        // comfortably trips the boost.
        log_e[10] = 2.0; // log2(linear) ~ 12 dB
        let pick = pick_dynalloc_boost_band(&log_e, 0, NB_EBANDS);
        assert_eq!(pick, Some(10), "outlier band must be picked");
    }

    #[test]
    fn dynalloc_skips_when_no_outlier() {
        // Flat band energies → no boost.
        let log_e = [0.1f32; NB_EBANDS];
        let pick = pick_dynalloc_boost_band(&log_e, 0, NB_EBANDS);
        assert_eq!(pick, None, "flat energies must produce no boost");
    }

    #[test]
    fn dynalloc_respects_min_excess_threshold() {
        // Sub-threshold excess (~3 dB) → no boost (we require ~6 dB).
        let mut log_e = [0f32; NB_EBANDS];
        log_e[5] = 0.5; // ~3 dB — below the 6 dB threshold
        let pick = pick_dynalloc_boost_band(&log_e, 0, NB_EBANDS);
        assert_eq!(pick, None, "sub-threshold excess must not boost");
    }
}
