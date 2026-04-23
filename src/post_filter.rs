//! CELT pitch post-filter / comb filter (RFC 6716 §4.3.7.1, libopus
//! `comb_filter` in celt.c).
//!
//! Applied to the IMDCT+OLA output. The filter is IIR:
//!
//! ```text
//! y[n] = x[n] + G*(g0*y[n-T] + g1*(y[n-T+1]+y[n-T-1]) + g2*(y[n-T+2]+y[n-T-2]))
//! ```
//!
//! In libopus the call is in-place (`x` == `y`), so all `y[n-T]` reads with
//! `n < T` fall back to the caller-supplied `x_history` buffer (the previous
//! frame's already-post-filtered tail). Within the current frame we read from
//! `y` at the same indices (data written earlier in this invocation).
//!
//! The first `overlap` samples cross-fade the **old** filter (`T0`, `g0`,
//! `tapset0` — parameters from the *previous* frame) into the **new** filter
//! (`T1`, `g1`, `tapset1` — parameters from the *current* frame) using
//! `window[i]^2` as the crossfade weight. After `overlap`, only the new
//! filter applies. If both frames share the same parameters the overlap
//! region is skipped (pure new filter over the whole frame).

use crate::tables::{COMB_FILTER_MINPERIOD, COMB_FILTER_TAPS};

const Q15ONE: f32 = 1.0;

/// In-place comb post-filter. `y` is both input and output. `x_history` holds
/// the last `max(T0, T1) + 2` (at least `COMB_FILTER_MAXPERIOD + 2`) samples
/// of post-filtered output from the previous frame — indexed so that
/// `x_history[len - 1]` is the sample immediately before `y[0]`.
///
/// `t0`, `g0`, `tapset0` are the previous-frame post-filter parameters (used
/// during the `overlap`-sample cross-fade). `t1`, `g1`, `tapset1` are the
/// current-frame parameters.
///
/// Matches libopus `comb_filter` (celt.c) to the extent the signs and
/// cross-fade shape allow: callers flip the sign of the gains externally if
/// their pitch search is negated.
#[allow(clippy::too_many_arguments)]
pub fn comb_filter(
    y: &mut [f32],
    x_history: &[f32],
    t0: i32,
    t1: i32,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: usize,
    tapset1: usize,
    window: &[f32],
    overlap: usize,
) {
    if g0 == 0.0 && g1 == 0.0 {
        // Zero gain on both sides: identity. Leave y untouched.
        return;
    }
    let t0 = t0.max(COMB_FILTER_MINPERIOD as i32) as usize;
    let t1 = t1.max(COMB_FILTER_MINPERIOD as i32) as usize;
    let g00 = g0 * COMB_FILTER_TAPS[tapset0][0];
    let g01 = g0 * COMB_FILTER_TAPS[tapset0][1];
    let g02 = g0 * COMB_FILTER_TAPS[tapset0][2];
    let g10 = g1 * COMB_FILTER_TAPS[tapset1][0];
    let g11 = g1 * COMB_FILTER_TAPS[tapset1][1];
    let g12 = g1 * COMB_FILTER_TAPS[tapset1][2];
    let same = g0 == g1 && t0 == t1 && tapset0 == tapset1;
    let overlap = if same { 0 } else { overlap };

    // Read y[i] with negative i falling back to x_history. When i >= 0 we
    // read from y itself; the caller runs the filter in-place, and since T
    // is >= COMB_FILTER_MINPERIOD > 0, any y[i - T] read inside the loop
    // touches earlier-written samples (already post-filtered).
    let hist_len = x_history.len();
    let read = |y: &[f32], i: isize| -> f32 {
        if i >= 0 {
            y[i as usize]
        } else {
            let idx = hist_len as isize + i;
            if idx >= 0 {
                x_history[idx as usize]
            } else {
                0.0
            }
        }
    };

    // Crossfade region: blend old and new filter.
    for i in 0..overlap {
        let f = window[i] * window[i];
        let one_minus_f = Q15ONE - f;
        let i_s = i as isize;
        let val = read(y, i_s)
            + (one_minus_f * g00) * read(y, i_s - t0 as isize)
            + (one_minus_f * g01) * (read(y, i_s - t0 as isize + 1) + read(y, i_s - t0 as isize - 1))
            + (one_minus_f * g02) * (read(y, i_s - t0 as isize + 2) + read(y, i_s - t0 as isize - 2))
            + (f * g10) * read(y, i_s - t1 as isize)
            + (f * g11) * (read(y, i_s - t1 as isize + 1) + read(y, i_s - t1 as isize - 1))
            + (f * g12) * (read(y, i_s - t1 as isize + 2) + read(y, i_s - t1 as isize - 2));
        y[i] = val;
    }

    if g1 == 0.0 {
        // New-frame gain is zero: nothing more to do past the overlap.
        return;
    }
    // Main region: only new filter.
    for i in overlap..n {
        let i_s = i as isize;
        let val = read(y, i_s)
            + g10 * read(y, i_s - t1 as isize)
            + g11 * (read(y, i_s - t1 as isize + 1) + read(y, i_s - t1 as isize - 1))
            + g12 * (read(y, i_s - t1 as isize + 2) + read(y, i_s - t1 as isize - 2));
        y[i] = val;
    }
}

/// Decode an RFC 6716 §4.3.7.1 post-filter gain symbol (3 raw bits) into the
/// `G = 3*(int_gain + 1) / 32` linear gain.
#[inline]
pub fn decode_pitch_gain(gain_code: u32) -> f32 {
    3.0 * ((gain_code + 1) as f32) / 32.0
}

/// Decode the RFC 6716 §4.3.7.1 pitch period from `(octave, fine_pitch)`:
/// `T = (16 << octave) + fine_pitch - 1`, bounded to [MINPERIOD, MAXPERIOD].
#[inline]
pub fn decode_pitch_period(octave: u32, fine_pitch: u32) -> u32 {
    let raw = (16u32 << octave) + fine_pitch - 1;
    raw.clamp(
        crate::tables::COMB_FILTER_MINPERIOD,
        crate::tables::COMB_FILTER_MAXPERIOD,
    )
}

/// Single-pole IIR de-emphasis filter (RFC 6716 §4.3.7.2). Applies
/// `y[n] = x[n] + alpha_p * y[n-1]` in-place on `samples`, returning the
/// final state `y[n-1]` to be passed in on the next call. `alpha_p` is
/// fixed at `DEEMPHASIS_COEF` per the RFC.
#[inline]
pub fn deemphasis(samples: &mut [f32], mut state: f32) -> f32 {
    let alpha = crate::tables::DEEMPHASIS_COEF;
    for v in samples.iter_mut() {
        let y = *v + alpha * state;
        *v = y;
        state = y;
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comb_filter_zero_gain_is_identity() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = x.clone();
        let hist = vec![0.0; 32];
        let win = vec![1.0; 0];
        comb_filter(&mut y, &hist, 16, 16, 4, 0.0, 0.0, 0, 0, &win, 0);
        assert_eq!(y, x);
    }

    #[test]
    fn decode_pitch_gain_boundaries() {
        // gain_code = 0 → G = 3/32 = 0.09375
        assert!((decode_pitch_gain(0) - 0.09375).abs() < 1e-6);
        // gain_code = 7 → G = 3*8/32 = 0.75
        assert!((decode_pitch_gain(7) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn decode_pitch_period_bounds() {
        // octave=0, fine_pitch=0 → (16 << 0) - 1 = 15 → MINPERIOD.
        assert_eq!(decode_pitch_period(0, 0), 15);
        // octave=5, fine_pitch=(1<<9)-1 = 511 → (16<<5) + 510 = 1022 → MAXPERIOD clamp is 1024.
        assert_eq!(decode_pitch_period(5, 511), 1022);
    }

    #[test]
    fn deemphasis_state_preserves_across_calls() {
        // Unit impulse: single-pole response is alpha^n.
        let mut x = vec![0.0f32; 8];
        x[0] = 1.0;
        let s1 = deemphasis(&mut x, 0.0);
        let alpha = crate::tables::DEEMPHASIS_COEF;
        let mut expected = 1.0f32;
        for (i, &v) in x.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "sample {i}: got {} expected {}",
                v,
                expected
            );
            expected *= alpha;
        }
        // Continuing the filter in a second call must pick up the state.
        let mut x2 = vec![0.0f32; 4];
        let _s2 = deemphasis(&mut x2, s1);
        // First sample of x2 = alpha * s1 = alpha^8.
        let mut e = s1 * alpha;
        for &v in &x2 {
            assert!((v - e).abs() < 1e-5);
            e *= alpha;
        }
    }

    #[test]
    fn comb_filter_in_place_periodic_identity() {
        // With a constant-valued history + signal, an IIR post-filter with
        // non-zero g just adds a constant bias per tap. Verify the function
        // runs (doesn't panic) and produces a bounded result.
        let mut y = vec![1.0f32; 200];
        let hist = vec![1.0f32; 1026];
        let win = vec![0.0f32; 0];
        comb_filter(&mut y, &hist, 20, 20, 200, 0.5, 0.5, 0, 0, &win, 0);
        // y[0] = 1 + 0.5 * (g0*1 + g1*(1+1) + g2*(1+1))
        //      = 1 + 0.5 * (0.3066 + 0.2188*2 + 0)    [tapset 0]
        let tap = &COMB_FILTER_TAPS[0];
        let expected0 = 1.0 + 0.5 * (tap[0] + 2.0 * tap[1] + 2.0 * tap[2]);
        assert!(
            (y[0] - expected0).abs() < 1e-5,
            "y[0]={} expected={}",
            y[0],
            expected0
        );
    }
}
