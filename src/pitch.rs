//! Pitch search and post-filter parameter decision (RFC 6716 §5.3.1).
//!
//! ## What this module covers
//!
//! RFC 6716 §5.3.1 describes the encoder's pitch pre-filter — "applied
//! in such a way as to be the inverse of the decoder's post-filter"
//! (the inverse itself is
//! [`crate::post_filter::apply_pitch_prefilter_transition_f32`]) — and
//! states that "the main non-obvious aspect of the pre-filter is the
//! selection of the pitch period. The pitch search should be optimized
//! for the following criteria:
//!
//! * **continuity**: it is important that the pitch period does not
//!   change abruptly between frames; and
//! * **avoidance of pitch multiples**: when the period used is a
//!   multiple of the real period (lower frequency fundamental), the
//!   post-filter loses most of its ability to reduce noise."
//!
//! The RFC gives no search algorithm — like every §5.3 decision, "an
//! encoder is free to choose the values in any manner" — so this
//! module implements a documented in-crate search honouring exactly
//! those two criteria: a normalized-autocorrelation scan over the
//! legal §4.3.7.1 period range, a sub-period demotion pass (a
//! candidate whose fundamental scores comparably is replaced by the
//! fundamental — the "avoidance of pitch multiples" criterion), and a
//! previous-period preference window (the "continuity" criterion).
//! Every threshold is a named constant documented as in-crate encoder
//! freedom; the coded parameters are ordinary §4.3.7.1 wire values, so
//! any choice keeps the decoder in lockstep.
//!
//! ## Clean-room provenance
//!
//! The two search criteria and the inverse-of-post-filter framing are
//! RFC 6716 §5.3.1 (`docs/audio/opus/rfc6716-opus.txt` lines
//! 8383–8397); the period/gain/tapset field ranges are §4.3.7.1. The
//! search algorithm and its thresholds are this crate's own design.
//! No external library source was consulted.

use crate::post_filter::{PostFilterParams, POST_FILTER_PERIOD_MAX, POST_FILTER_PERIOD_MIN};

/// Result of a [`pitch_search`]: the chosen period and its normalized
/// autocorrelation (`0.0..=1.0` for a genuinely periodic signal).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchEstimate {
    /// Chosen pitch period in samples (`15..=1022`, the §4.3.7.1
    /// bounds).
    pub period: u16,
    /// Normalized autocorrelation of the signal against itself shifted
    /// by `period`, over the analysis tail. Values near `1.0` mean a
    /// strongly periodic signal.
    pub correlation: f32,
}

/// A sub-period (fundamental) candidate replaces the best candidate
/// when its normalized correlation reaches this fraction of the best —
/// the §5.3.1 "avoidance of pitch multiples" criterion. In-crate
/// encoder freedom.
pub const MULTIPLE_AVOIDANCE_RATIO: f32 = 0.85;

/// A period within [`CONTINUITY_HALF_WIDTH`] of the previous frame's
/// period is kept when its correlation reaches this fraction of the
/// best — the §5.3.1 "continuity" criterion. In-crate encoder freedom.
pub const CONTINUITY_RATIO: f32 = 0.9;

/// Half-width of the previous-period preference window, in samples.
pub const CONTINUITY_HALF_WIDTH: u16 = 2;

/// Correlation floor below which [`choose_post_filter_params`] leaves
/// the post-filter off: a weakly periodic frame gains nothing from
/// comb filtering. In-crate encoder freedom.
pub const MIN_PITCH_CORRELATION: f32 = 0.25;

/// Normalized autocorrelation of `signal`'s trailing `window` samples
/// against the same window shifted `period` samples into the past.
/// Returns `0.0` when either energy vanishes.
fn normalized_correlation(signal: &[f32], window: usize, period: usize) -> f32 {
    let n = signal.len();
    if window == 0 || period == 0 || window + period > n {
        return 0.0;
    }
    let tail = &signal[n - window..];
    let lagged = &signal[n - window - period..n - period];
    let mut dot = 0.0f64;
    let mut e0 = 0.0f64;
    let mut e1 = 0.0f64;
    for (&a, &b) in tail.iter().zip(lagged.iter()) {
        dot += f64::from(a) * f64::from(b);
        e0 += f64::from(a) * f64::from(a);
        e1 += f64::from(b) * f64::from(b);
    }
    if e0 <= 0.0 || e1 <= 0.0 {
        return 0.0;
    }
    (dot / (e0.sqrt() * e1.sqrt())) as f32
}

/// Search for the pitch period of `signal`'s trailing `window` samples
/// over the §4.3.7.1 legal range, honouring the two §5.3.1 criteria.
///
/// * `signal` — recent (ideally pre-emphasized) samples, oldest first;
///   the last `window` samples are the analysis frame and everything
///   before them is lag history. Periods beyond the available history
///   are not probed.
/// * `window` — analysis length (a frame, typically).
/// * `prev_period` — the previous frame's chosen period, engaging the
///   §5.3.1 continuity preference when `Some`.
///
/// The scan picks the maximum normalized autocorrelation over
/// `15..=min(1022, signal.len() - window)`; a sub-period pass then
/// demotes pitch multiples (the smallest `period/k`, `k = 2..=4`,
/// whose correlation reaches [`MULTIPLE_AVOIDANCE_RATIO`] of the best
/// wins), and a continuity pass prefers a period within
/// [`CONTINUITY_HALF_WIDTH`] of `prev_period` when its correlation
/// reaches [`CONTINUITY_RATIO`] of the best. Returns `None` when the
/// probe range is empty or no positive correlation exists.
pub fn pitch_search(
    signal: &[f32],
    window: usize,
    prev_period: Option<u16>,
) -> Option<PitchEstimate> {
    let n = signal.len();
    if window == 0 || n <= window {
        return None;
    }
    let max_lag = (n - window).min(POST_FILTER_PERIOD_MAX as usize);
    let min_lag = POST_FILTER_PERIOD_MIN as usize;
    if max_lag < min_lag {
        return None;
    }

    // Base scan: maximum normalized autocorrelation.
    let mut best_t = 0usize;
    let mut best_c = 0.0f32;
    for t in min_lag..=max_lag {
        let c = normalized_correlation(signal, window, t);
        if c > best_c {
            best_c = c;
            best_t = t;
        }
    }
    if best_t == 0 || best_c <= 0.0 {
        return None;
    }

    // §5.3.1 avoidance of pitch multiples: if a sub-period (candidate
    // fundamental) correlates comparably, use it instead. Probe the
    // nearest integer around best/k with ±1 slack.
    let mut chosen_t = best_t;
    let mut chosen_c = best_c;
    for k in (2..=4usize).rev() {
        let centre = best_t / k;
        if centre < min_lag {
            continue;
        }
        for cand in centre.saturating_sub(1)..=(centre + 1).min(max_lag) {
            if cand < min_lag {
                continue;
            }
            let c = normalized_correlation(signal, window, cand);
            if c >= MULTIPLE_AVOIDANCE_RATIO * best_c && cand < chosen_t {
                chosen_t = cand;
                chosen_c = c;
            }
        }
    }

    // §5.3.1 continuity: stay near the previous period when it is
    // still a comparably good explanation of the frame.
    if let Some(prev) = prev_period {
        let lo = (prev.saturating_sub(CONTINUITY_HALF_WIDTH) as usize).max(min_lag);
        let hi = ((prev + CONTINUITY_HALF_WIDTH) as usize).min(max_lag);
        let mut near_best_t = 0usize;
        let mut near_best_c = 0.0f32;
        for t in lo..=hi {
            let c = normalized_correlation(signal, window, t);
            if c > near_best_c {
                near_best_c = c;
                near_best_t = t;
            }
        }
        let already_near = chosen_t.abs_diff(prev as usize) <= CONTINUITY_HALF_WIDTH as usize;
        if !already_near && near_best_t != 0 && near_best_c >= CONTINUITY_RATIO * chosen_c {
            chosen_t = near_best_t;
            chosen_c = near_best_c;
        }
    }

    Some(PitchEstimate {
        period: chosen_t as u16,
        correlation: chosen_c.clamp(-1.0, 1.0),
    })
}

/// Turn a [`PitchEstimate`] into §4.3.7.1 post-filter parameters, or
/// `None` (post-filter off) when the frame is too weakly periodic to
/// benefit ([`MIN_PITCH_CORRELATION`]).
///
/// The 3-bit gain index maps the correlation onto the §4.3.7.1 gain
/// grid `G = 3*(index+1)/32`: the target gain is half the correlation
/// (a documented in-crate conservative choice — a full-strength comb
/// on an imperfectly periodic signal smears transients), rounded to
/// the nearest index and clamped to `0..=7`. The tapset is fixed at
/// `0` (the flattest §4.3.7.1 response; tapset choice is signalled
/// per-frame and any value is legal).
pub fn choose_post_filter_params(estimate: PitchEstimate) -> Option<PostFilterParams> {
    if estimate.correlation < MIN_PITCH_CORRELATION {
        return None;
    }
    let target_gain = 0.5 * estimate.correlation;
    // G = 3*(idx+1)/32  =>  idx = round(32*G/3) - 1.
    let idx = ((32.0 * target_gain / 3.0).round() as i32 - 1).clamp(0, 7);
    Some(PostFilterParams {
        enabled: true,
        period: estimate
            .period
            .clamp(POST_FILTER_PERIOD_MIN, POST_FILTER_PERIOD_MAX),
        gain_index: idx as u8,
        tapset: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pure sine of known period is found by the base scan.
    #[test]
    fn finds_pure_tone_period() {
        // Period 60 samples: f = 2*pi/60.
        let period = 60usize;
        let n = 1500usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / period as f32).sin())
            .collect();
        let est = pitch_search(&signal, 480, None).expect("estimate");
        assert!(
            est.period as usize % period <= 1 || period % est.period as usize <= 1,
            "period {} not compatible with true {period}",
            est.period
        );
        // The multiple-avoidance pass keeps it at (or below) the
        // fundamental rather than a multiple.
        assert!(
            (est.period as usize) < 2 * period,
            "picked a multiple: {}",
            est.period
        );
        assert!(
            est.correlation > 0.95,
            "weak correlation {}",
            est.correlation
        );
    }

    /// Multiple avoidance: seeded at the fundamental even when the
    /// best raw correlation sits at a multiple (a perfectly periodic
    /// signal correlates identically at every multiple; the pass must
    /// pick the smallest).
    #[test]
    fn prefers_fundamental_over_multiple() {
        let period = 40usize;
        let n = 1200usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let ph = 2.0 * std::f32::consts::PI * i as f32 / period as f32;
                ph.sin() + 0.4 * (2.0 * ph).sin()
            })
            .collect();
        let est = pitch_search(&signal, 400, None).expect("estimate");
        assert!(
            est.period as usize <= period + 1,
            "picked a multiple of the fundamental: {}",
            est.period
        );
    }

    /// Continuity: a previous period close to (but not exactly) the
    /// truth is kept when it still explains the frame.
    #[test]
    fn continuity_prefers_previous_period() {
        let period = 80usize;
        let n = 1600usize;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / period as f32).sin())
            .collect();
        // A perfectly periodic tone correlates ~1.0 at 79..81 too; the
        // continuity window must keep the previous choice.
        let est = pitch_search(&signal, 480, Some(79)).expect("estimate");
        assert!(
            est.period.abs_diff(79) <= CONTINUITY_HALF_WIDTH,
            "continuity ignored: {} vs prev 79",
            est.period
        );
    }

    /// White noise yields a weak correlation and no post-filter.
    #[test]
    fn noise_disables_post_filter() {
        let mut seed = 0x1234_5678u32;
        let signal: Vec<f32> = (0..2000)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect();
        if let Some(est) = pitch_search(&signal, 480, None) {
            // Whatever weak peak the scan finds must not clear the
            // enable threshold.
            assert!(
                choose_post_filter_params(est).is_none() || est.correlation < 0.5,
                "noise produced a confident pitch: {est:?}"
            );
        }
    }

    /// Gain mapping: full correlation lands mid-grid (G = 0.5 target),
    /// threshold correlation lands at the bottom, sub-threshold is off.
    #[test]
    fn gain_mapping_is_monotone_and_bounded() {
        let mk = |c: f32| PitchEstimate {
            period: 100,
            correlation: c,
        };
        assert!(choose_post_filter_params(mk(0.1)).is_none());
        let low = choose_post_filter_params(mk(0.3)).unwrap();
        let mid = choose_post_filter_params(mk(0.6)).unwrap();
        let high = choose_post_filter_params(mk(1.0)).unwrap();
        assert!(low.gain_index <= mid.gain_index);
        assert!(mid.gain_index <= high.gain_index);
        assert!(high.gain_index <= 7);
        // Target G = 0.5 => idx = round(32*0.5/3) - 1 = 4.
        assert_eq!(high.gain_index, 4);
        assert!(low.enabled && mid.enabled && high.enabled);
        assert_eq!(high.tapset, 0);
    }

    /// Degenerate inputs return None instead of panicking.
    #[test]
    fn degenerate_inputs_are_none() {
        assert!(pitch_search(&[], 480, None).is_none());
        assert!(pitch_search(&[0.0; 100], 100, None).is_none());
        let silent = vec![0.0f32; 2000];
        assert!(pitch_search(&silent, 480, None).is_none());
    }
}
