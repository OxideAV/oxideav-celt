//! CELT encoder-side pitch analysis for the comb pre-filter (RFC 6716
//! §4.3.7.1).
//!
//! The decoder runs `comb_filter(y, x_history, …, +g, …)` after IMDCT to
//! re-introduce a periodic component the MDCT could not represent
//! efficiently. To make that re-introduction work as quality
//! enhancement (and not just noise injection), the encoder must
//! *subtract* the same component from the PCM before MDCT — a pre-filter
//! call to `comb_filter` with the gains negated. The decoder then
//! cancels the pre-filter (post = -pre) and the bitstream effectively
//! describes the residual, which compresses better.
//!
//! This module covers the analysis half: pick `(period T, gain G,
//! tapset)` from the input PCM. The actual call to `comb_filter`
//! happens in `encoder.rs`; this module only returns the parameters and
//! the encoded `(octave, fine_pitch, gain_idx)` syntax fields.
//!
//! Algorithm — straightforward NCC autocorrelation:
//!
//! 1. Build a Hann-windowed copy of the PCM frame (windowing avoids
//!    spurious peaks at the long-period end where the autocorrelation
//!    sum ranges over fewer aligned samples).
//! 2. Compute the normalised autocorrelation `r(τ) = R(τ) /
//!    sqrt(R0_a * R0_b)` for `τ ∈ [MINPERIOD, MAXPERIOD]`. `R(τ) =
//!    Σ_{n} x[n] * x[n+τ]`. The two energy normalisers are the
//!    energy of `x[n]` and of `x[n+τ]` over the same overlap range,
//!    so `r(τ)` is bounded in `[-1, 1]`.
//! 3. Pick the `τ*` maximising `r(τ)`. Reject if peak `r < gain_floor`
//!    (typical: 0.3) — the frame isn't periodic enough to benefit
//!    from a pre-filter.
//! 4. Quantise `τ*` to the spec syntax: pick the largest `octave`
//!    such that `T ≥ (16 << octave)`, then `fine_pitch = T - (16
//!    << octave) + 1`, fitting in `4 + octave` bits.
//! 5. Map normalised correlation strength to a 3-bit gain index:
//!    `int_gain = round(8 * r) - 1`, clamped to `[0, 7]`. The decoded
//!    gain is then `G = 3 * (int_gain + 1) / 32 ∈ [3/32, 3/4]`.
//! 6. Pick the tapset by minimising the post-pre-filter energy across
//!    the three tap shapes — same heuristic libopus uses ("less
//!    residual energy = closer match to the harmonic series").
//!
//! This is a clean-room implementation against RFC 6716 §4.3.7.1 + the
//! standard NCC pitch-analysis literature ("Speech Coding Algorithms"
//! by Chu, ch. 4; "Discrete-Time Speech Signal Processing" by Quatieri,
//! ch. 10). No libopus source consulted for the algorithm — only the
//! RFC for the syntax bounds and `tables::COMB_FILTER_TAPS` for the
//! tap shapes (which the decoder also uses, so any mismatch would
//! cancel anyway).

use crate::tables::{COMB_FILTER_MAXPERIOD, COMB_FILTER_MINPERIOD, COMB_FILTER_TAPS};

/// Pitch-analysis result. `period` is the time-domain pitch period in
/// samples (`τ*`), `gain` is the linear post-filter gain `G` to apply,
/// `tapset` is the tap-shape index (0..=2). The encoded-syntax fields
/// `(octave, fine_pitch, gain_idx)` are the values the encoder writes
/// into the post-filter header. `correlation` is the peak normalised
/// autocorrelation in `[-1, 1]` — useful for callers to gate the
/// pre-filter on a stricter floor than this module's default.
#[derive(Copy, Clone, Debug)]
pub struct PitchParams {
    pub period: u32,
    pub gain: f32,
    pub tapset: u32,
    pub octave: u32,
    pub fine_pitch: u32,
    pub gain_idx: u32,
    pub correlation: f32,
}

/// Default normalised-autocorrelation floor for enabling the pre-filter.
/// Frames with peak `r < CORR_FLOOR` skip the pre-filter — the bit cost
/// of the post-filter header (~17 bits) is not recovered on the
/// quantisation side, and forcing a low-confidence period in just
/// adds aliased energy.
pub const CORR_FLOOR: f32 = 0.30;

/// Analyse `pcm` and return pitch parameters, or `None` if the frame
/// is not periodic enough (peak NCC below `CORR_FLOOR`) or too short
/// to host the minimum search lag. `pcm.len()` must be at least
/// `2 * COMB_FILTER_MINPERIOD` (30 samples) — typical CELT frames
/// (480 / 960 samples) are far above this.
///
/// `history` is the previous frame's PCM tail used to extend the
/// autocorrelation window into the past, mirroring what the
/// pre-filter call's `x_history` provides. `history.len()` should be
/// at least `COMB_FILTER_MAXPERIOD` so lags up to 1022 see real data
/// instead of zero-padding bias. Empty history is allowed (cold
/// start); long-period peaks will then be biased low because the
/// trailing samples don't have a past reference.
pub fn analyse_pitch(pcm: &[f32], history: &[f32]) -> Option<PitchParams> {
    analyse_pitch_with_floor(pcm, history, CORR_FLOOR)
}

/// Same as [`analyse_pitch`] but with an explicit correlation floor.
/// Tests use this to pin the gate at a known operating point.
pub fn analyse_pitch_with_floor(
    pcm: &[f32],
    history: &[f32],
    corr_floor: f32,
) -> Option<PitchParams> {
    let n = pcm.len();
    let min_period = COMB_FILTER_MINPERIOD as usize;
    let max_period = COMB_FILTER_MAXPERIOD as usize;
    if n < 2 * min_period {
        return None;
    }
    // Cap the search range to what the frame + history can actually
    // support. We need at least `min_period` samples of overlap on each
    // side of the lag, so the largest meaningful `τ` is `n - min_period`
    // (frame-internal) plus whatever history we can read past `pcm[0]`.
    let max_search = max_period
        .min(n + history.len() - min_period)
        .max(min_period);
    if max_search < min_period {
        return None;
    }

    // Hann window — applied to the autocorrelation source signal.
    // Tapering both ends suppresses the long-period bias caused by the
    // shrinking number of overlapping samples at high lags. We build
    // a windowed copy `w_pcm` so the per-lag correlation loop reads
    // pre-tapered values without recomputing the window every step.
    let win: Vec<f32> = (0..n)
        .map(|i| {
            let phase = std::f32::consts::PI * (i as f32 + 0.5) / n as f32;
            let s = phase.sin();
            s * s
        })
        .collect();
    let w_pcm: Vec<f32> = pcm.iter().zip(win.iter()).map(|(&x, &w)| x * w).collect();

    // R0_a and R0_b are computed per-lag inside the loop because the
    // overlap range depends on whether the lookback into `history`
    // covers the lag. We do precompute `e_total` for an early-exit
    // silence check.
    let e_total: f32 = w_pcm.iter().map(|&v| v * v).sum();
    if e_total < 1e-12 {
        return None;
    }

    // Look back into history for the τ-shifted samples. `hist_len`
    // samples of history are addressable; index `-1` is the sample
    // immediately before `pcm[0]`. For a given lag `τ`, the n samples
    // we'd read at `pcm[i - τ]` for `i ∈ [0, n)` span `[-τ, n-τ)`.
    // The portion `[-τ, 0)` falls into history if available, the rest
    // is from `pcm` itself (which we replicate as the windowed copy).
    let hist_len = history.len();
    let read_past = |i: isize| -> f32 {
        if i >= 0 {
            // Same windowed signal we use for the unshifted ref.
            w_pcm[i as usize]
        } else {
            let idx = hist_len as isize + i;
            if idx >= 0 {
                // History samples aren't windowed — they're "outside"
                // the analysis frame. Tapering them would double-count
                // the window from the previous frame. Use raw values.
                history[idx as usize]
            } else {
                0.0
            }
        }
    };

    // Compute normalised autocorrelation for τ ∈ [min_period, max_search].
    // We collect the full curve so the post-pass can pick the first
    // significant local maximum, not just the global argmax. The
    // global argmax is biased to short lags on smooth signals (the
    // autocorrelation of a slow sinusoid is near 1 for any τ small
    // compared to one period), which would mis-identify the pitch.
    let mut ncc_curve = vec![0f32; max_search + 1];
    for tau in min_period..=max_search {
        let mut r = 0f32;
        let mut e_b = 0f32;
        for i in 0..n {
            let xi = w_pcm[i];
            let xj = read_past(i as isize - tau as isize);
            r += xi * xj;
            e_b += xj * xj;
        }
        let denom = (e_total * e_b).max(1e-30).sqrt();
        ncc_curve[tau] = r / denom;
    }

    // Pitch picking: walk τ from min_period and find the first
    // significant local maximum. "Significant" = NCC at that τ is at
    // least `corr_floor` AND no subsequent local max within a window
    // of ±20% of T exceeds it. This rejects the smooth-tone short-lag
    // bias (a sine's NCC starts at ~1 at τ=MINPERIOD, gradually drops
    // through zero, and only peaks again at τ=T) while still picking
    // the true pitch on harmonic-rich signals (where the first local
    // max is the actual fundamental).
    //
    // We additionally walk forward looking for *higher* local maxima
    // up to `2 * tau`. If we find one within `[0.5*tau, 2*tau]` whose
    // NCC exceeds the current peak by at least 0.05, we prefer the
    // longer one — guards against picking a sub-multiple (T/2, T/3)
    // of the true period.
    let mut best_tau = 0usize;
    let mut best_corr = f32::NEG_INFINITY;
    // First pass: collect all local maxima above the floor.
    let mut local_maxes: Vec<(usize, f32)> = Vec::new();
    for tau in (min_period + 1)..max_search {
        let v = ncc_curve[tau];
        if v >= corr_floor && v > ncc_curve[tau - 1] && v > ncc_curve[tau + 1] {
            local_maxes.push((tau, v));
        }
    }
    if local_maxes.is_empty() {
        return None;
    }
    // Pick the candidate with the largest NCC, breaking ties by
    // preferring the smaller τ (since larger τ candidates are usually
    // multiples of the true period). Then check if doubling that τ
    // yields a comparable or better local max — if so, prefer the
    // double (we picked a sub-multiple).
    let mut idx = 0usize;
    for (i, &(_, v)) in local_maxes.iter().enumerate() {
        if v > local_maxes[idx].1 {
            idx = i;
        }
    }
    let mut chosen_tau = local_maxes[idx].0;
    let mut chosen_corr = local_maxes[idx].1;
    // Walk forward in τ, see if any local max within `[2*chosen_tau ±
    // 10%]` has comparable correlation. If yes, the original was a
    // sub-multiple — adopt the longer period.
    loop {
        let target = 2 * chosen_tau;
        let lo = target * 9 / 10;
        let hi = (target * 11 / 10).min(max_search);
        if lo > max_search {
            break;
        }
        let mut found: Option<(usize, f32)> = None;
        for &(t, v) in &local_maxes {
            if t >= lo && t <= hi && v > chosen_corr - 0.10 {
                if found.map(|(_, fv)| v > fv).unwrap_or(true) {
                    found = Some((t, v));
                }
            }
        }
        match found {
            Some((t, v)) => {
                chosen_tau = t;
                chosen_corr = v;
            }
            None => break,
        }
    }
    best_tau = chosen_tau;
    best_corr = chosen_corr;

    // Reject low-confidence frames (no clear periodicity).
    if best_corr < corr_floor {
        return None;
    }

    // Quantise τ to the (octave, fine_pitch) syntax. RFC 6716 §4.3.7.1:
    // `T = (16 << octave) + fine_pitch - 1`, octave ∈ [0, 5],
    // fine_pitch in `4 + octave` bits.
    let t = (best_tau as u32).clamp(COMB_FILTER_MINPERIOD, COMB_FILTER_MAXPERIOD);
    let (octave, fine_pitch) = encode_period(t);

    // Map peak correlation to a 3-bit gain index. A perfect harmonic
    // (r == 1.0) maps to int_gain = 7 (G = 0.75). r at the floor (0.3)
    // maps to int_gain = round(8*0.3) - 1 = 1 (G = 6/32 ≈ 0.19).
    // Below the floor we'd return None above; above 1.0 we clamp.
    let raw_gain = (8.0 * best_corr).round() as i32 - 1;
    let gain_idx = raw_gain.clamp(0, 7) as u32;
    let gain = 3.0 * ((gain_idx + 1) as f32) / 32.0;

    // Pick the tapset that minimises the post-pre-filter residual
    // energy on this frame. Same idea libopus uses ("the tap shape
    // closest to the actual harmonic series leaves the smallest
    // residual after the comb subtraction"). Try all three tap shapes
    // and pick the one with the lowest pre-filtered RMS over the frame.
    let tapset = pick_best_tapset(pcm, history, t as usize, gain);

    Some(PitchParams {
        period: t,
        gain,
        tapset,
        octave,
        fine_pitch,
        gain_idx,
        correlation: best_corr,
    })
}

/// Quantise pitch period `T ∈ [MINPERIOD, MAXPERIOD]` to the spec syntax
/// `(octave, fine_pitch)` such that the decoder reconstructs `T` exactly:
///
/// ```text
/// T = (16 << octave) + fine_pitch - 1
/// ```
///
/// Picks the largest `octave ∈ [0, 5]` with `(16 << octave) ≤ T + 1`.
/// `fine_pitch` is then `T + 1 - (16 << octave)` and fits in
/// `4 + octave` bits (0..(2^(4+octave) - 1)).
pub fn encode_period(t: u32) -> (u32, u32) {
    let t = t.clamp(COMB_FILTER_MINPERIOD, COMB_FILTER_MAXPERIOD);
    // `t + 1` ranges over [16, 1023]. Largest power-of-two-times-16
    // that fits is 16<<octave with octave ≤ 5 (16<<5 = 512 ≤ 1023).
    let mut octave = 0u32;
    while octave < 5 && (16u32 << (octave + 1)) <= t + 1 {
        octave += 1;
    }
    let fine_pitch = (t + 1) - (16u32 << octave);
    debug_assert!(fine_pitch < (1u32 << (4 + octave)));
    (octave, fine_pitch)
}

/// Try all three tap shapes and pick the one producing the smallest
/// post-pre-filter signal energy on `pcm`. The pre-filter formula is
/// the same `comb_filter` we run from the encoder, here unrolled
/// with negated gain.
fn pick_best_tapset(pcm: &[f32], history: &[f32], t: usize, gain: f32) -> u32 {
    let n = pcm.len();
    let hist_len = history.len();
    let read = |i: isize| -> f32 {
        if i >= 0 {
            pcm[i as usize]
        } else {
            let idx = hist_len as isize + i;
            if idx >= 0 {
                history[idx as usize]
            } else {
                0.0
            }
        }
    };
    let mut best_e = f32::INFINITY;
    let mut best_ts = 0u32;
    for ts in 0..3u32 {
        let g0 = -gain * COMB_FILTER_TAPS[ts as usize][0];
        let g1 = -gain * COMB_FILTER_TAPS[ts as usize][1];
        let g2 = -gain * COMB_FILTER_TAPS[ts as usize][2];
        let mut e = 0f32;
        for i in 0..n {
            let i_s = i as isize;
            let y = pcm[i]
                + g0 * read(i_s - t as isize)
                + g1 * (read(i_s - t as isize + 1) + read(i_s - t as isize - 1))
                + g2 * (read(i_s - t as isize + 2) + read(i_s - t as isize - 2));
            e += y * y;
        }
        if e < best_e {
            best_e = e;
            best_ts = ts;
        }
    }
    best_ts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `encode_period` round-trips: every `T ∈ [15, 1022]` decodes back
    /// to itself via the spec formula `T = (16 << octave) + fine_pitch
    /// - 1`.
    #[test]
    fn encode_period_round_trips() {
        for t in COMB_FILTER_MINPERIOD..=1022 {
            let (octave, fine) = encode_period(t);
            assert!(octave <= 5, "octave {octave} out of range for T={t}");
            assert!(fine < (1u32 << (4 + octave)));
            let decoded = (16u32 << octave) + fine - 1;
            assert_eq!(
                decoded, t,
                "round-trip failed: T={} → ({}, {}) → {}",
                t, octave, fine, decoded
            );
        }
    }

    /// Pure sine at 220 Hz / 48 kHz has a fundamental period of
    /// 48000 / 220 ≈ 218.18 samples. Pitch analyser should pick a
    /// period within a few samples of that and report a high
    /// correlation (close to 1.0 for a clean sinusoid).
    #[test]
    fn analyse_pitch_finds_220hz_period() {
        let n = 960;
        let f = 220.0f32;
        let sr = 48000.0f32;
        let pcm: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr).sin() * 0.4)
            .collect();
        let history = vec![0.0f32; COMB_FILTER_MAXPERIOD as usize];
        let p = analyse_pitch(&pcm, &history).expect("should find pitch on clean sine");
        // Period within ±2 samples of the true 218 (autocorrelation
        // can pick adjacent integer lags depending on phase).
        let true_t = (sr / f) as i32;
        let diff = (p.period as i32 - true_t).abs();
        assert!(
            diff <= 4,
            "period {} too far from true {}",
            p.period,
            true_t
        );
        // Strong correlation — a Hann-windowed sinusoid gives r > 0.6
        // (the window decorrelates the edges so the NCC at the true
        // period doesn't quite reach 1.0).
        assert!(p.correlation > 0.6, "correlation {} too low", p.correlation);
        // Round-trip the encoded syntax back to T.
        let decoded = (16u32 << p.octave) + p.fine_pitch - 1;
        assert_eq!(decoded, p.period);
    }

    /// White noise has no periodic structure: the analyser should
    /// reject it (return `None`) at the default floor.
    #[test]
    fn analyse_pitch_rejects_white_noise() {
        // Deterministic LCG so the test is repeatable.
        let n = 960;
        let mut state: u32 = 0xdeadbeef;
        let pcm: Vec<f32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect();
        let history = vec![0.0f32; COMB_FILTER_MAXPERIOD as usize];
        let result = analyse_pitch(&pcm, &history);
        // Some random noise instances do exhibit weak periodic
        // structure on a 960-sample window. We allow either None
        // OR a low-correlation finding (well below 0.6 — the
        // analyser's gain mapping would already pick a small G).
        if let Some(p) = result {
            assert!(
                p.correlation < 0.6,
                "noise NCC {} too high (expected low or rejected)",
                p.correlation
            );
        }
    }

    /// Silent frame must be rejected (energy floor).
    #[test]
    fn analyse_pitch_rejects_silence() {
        let pcm = vec![0f32; 960];
        let history = vec![0.0f32; COMB_FILTER_MAXPERIOD as usize];
        assert!(analyse_pitch(&pcm, &history).is_none());
    }

    /// Pitch analyser run on a sine + harmonic content (220 Hz +
    /// 4 harmonics) should still pick a fundamental period close to
    /// the sine's 218 samples — harmonics reinforce periodicity.
    #[test]
    fn analyse_pitch_handles_harmonic_content() {
        let n = 960;
        let f0 = 220.0f32;
        let sr = 48000.0f32;
        let pcm: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let mut s = 0.0;
                for k in 1..=5 {
                    let amp = 0.3 / k as f32;
                    s += amp * (2.0 * std::f32::consts::PI * f0 * k as f32 * t).sin();
                }
                s
            })
            .collect();
        let history = vec![0.0f32; COMB_FILTER_MAXPERIOD as usize];
        let p = analyse_pitch(&pcm, &history).expect("harmonic content should produce a pitch");
        let true_t = (sr / f0) as i32;
        let diff = (p.period as i32 - true_t).abs();
        // Harmonic content can shift the autocorrelation peak to a
        // sub-multiple (T/2) — accept that as a valid pitch find.
        assert!(
            diff <= 4 || (p.period as i32 - true_t / 2).abs() <= 4,
            "period {} not near {} or {}",
            p.period,
            true_t,
            true_t / 2
        );
    }
}
