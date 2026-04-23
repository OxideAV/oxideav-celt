//! Edge-case tests for the CELT post-filter (RFC 6716 §4.3.7.1).
//!
//! These tests pin the contract that `comb_filter` tolerates short
//! sub-frames (`n < overlap`) without panicking. The Opus wrapper
//! splits a CELT frame into a 120-sample head and an `n - 120`-sample
//! tail; for 5 ms CELT frames (N=200) the tail is only 80 samples, and
//! a naive implementation that runs the 120-sample crossfade window
//! over an 80-sample buffer reads past the end of `y`.
//!
//! RFC 6716 §4.3.7.1 specifies the crossfade region as "the square of
//! the MDCT window", without saying what happens when the window is
//! longer than the available data. The only sensible interpretation
//! (and the one libopus implements) is to truncate the crossfade to
//! the available data — there is nothing to blend past the end of the
//! sub-frame.

use oxideav_celt::post_filter::comb_filter;

/// Regression: calling `comb_filter` with `overlap > n` on a sub-frame
/// shorter than the crossfade window must not panic. Reproduces the
/// RFC 6716 Appendix A `testvector08/09/10` panic.
#[test]
fn comb_filter_short_subframe_no_panic() {
    // 5 ms CELT frame tail: N = 200 - 120 = 80 samples.
    let mut y = vec![0.1f32; 80];
    let history = vec![0.05f32; 1026];
    // 120-sample MDCT window (squared crossfade weights).
    let window: Vec<f32> = (0..120)
        .map(|i| {
            let x = (i as f32 + 0.5) / 120.0 * std::f32::consts::FRAC_PI_2;
            x.sin()
        })
        .collect();

    // Different period and tapset so the crossfade actually runs.
    comb_filter(
        &mut y, &history, 20, 33, 80, 0.3, 0.5, 0, 1, &window, 120,
    );
    // No panic == pass. Just assert the buffer was touched (non-all-zero).
    assert!(y.iter().any(|&v| v != 0.0));
}

/// With `n` exactly equal to the minimum pitch period MINPERIOD (15),
/// the IIR must read from `x_history` for all lookbacks and produce a
/// finite result.
#[test]
fn comb_filter_n_equals_minperiod_no_panic() {
    let mut y = vec![0.2f32; 15];
    let history = vec![0.1f32; 1026];
    let window = vec![0.0f32; 0];
    comb_filter(&mut y, &history, 15, 15, 15, 0.5, 0.5, 0, 0, &window, 0);
    assert!(y.iter().all(|v| v.is_finite()));
}

/// `n < overlap` with identical old/new parameters: overlap should be
/// collapsed to zero internally (the `same` branch skips the crossfade)
/// and the main loop must still run without reading past the buffer.
#[test]
fn comb_filter_short_subframe_same_params() {
    let mut y = vec![1.0f32; 60];
    let history = vec![0.0f32; 1026];
    let window = vec![0.0f32; 120];
    comb_filter(&mut y, &history, 20, 20, 60, 0.5, 0.5, 0, 0, &window, 120);
    assert!(y.iter().all(|v| v.is_finite()));
}

/// A caller passing a period above `COMB_FILTER_MAXPERIOD` (e.g. from
/// a buggy decode) must not index past `x_history`. The clamp in
/// `comb_filter` pins the effective period to the maximum.
#[test]
fn comb_filter_clamps_out_of_range_period() {
    let mut y = vec![0.1f32; 120];
    let history = vec![0.05f32; 1026];
    let window = vec![0.0f32; 0];
    // Pass periods well outside the RFC range. Should not panic.
    comb_filter(
        &mut y,
        &history,
        5_000, // way above MAXPERIOD = 1024
        -50,   // below MINPERIOD = 15 (also negative)
        120,
        0.4,
        0.4,
        0,
        0,
        &window,
        0,
    );
    assert!(y.iter().all(|v| v.is_finite()));
}
