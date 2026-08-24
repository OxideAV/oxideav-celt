//! Packet-loss-concealment helpers: the LPC / autocorrelation /
//! pitch-estimation chain the RFC 6716 §4.3 decoder runs when a
//! frame is lost (`celt_decode_lost`).
//!
//! ## Provenance
//!
//! Transcribed from the **normative RFC 6716 Appendix A reference
//! listing** (`celt_lpc.c`: `_celt_autocorr`, `_celt_lpc`,
//! `celt_fir`, `celt_iir`; `pitch.c`: `pitch_downsample`,
//! `pitch_search`, `find_best_pitch`), extracted from the staged
//! `docs/audio/opus/rfc6716-opus.txt` per §A.1 and SHA-1-verified
//! against the §A.1-printed value
//! (`86a927223e73d2476646a1b933fcd3fffb6ecc8c`); float-build
//! semantics throughout (the fixed-point shifts are identities in
//! the float build).

/// The concealment LPC order (`LPC_ORDER`).
pub(crate) const LPC_ORDER: usize = 24;

/// Windowed autocorrelation (`_celt_autocorr`, float build): the
/// first and last `overlap` samples are shaped by `window` before
/// the correlation, and the zero-lag term gets the listing's `+10`
/// noise floor.
pub(crate) fn celt_autocorr(x: &[f32], ac: &mut [f32], window: Option<&[f32]>, overlap: usize) {
    let n = x.len();
    let mut xx = x.to_vec();
    if let Some(w) = window {
        for i in 0..overlap {
            xx[i] = x[i] * w[i];
            xx[n - i - 1] = x[n - i - 1] * w[i];
        }
    }
    for lag in (0..ac.len()).rev() {
        let mut d = 0f32;
        for i in lag..n {
            d += xx[i] * xx[i - lag];
        }
        ac[lag] = d;
    }
    ac[0] += 10.0;
}

/// Levinson-Durbin recursion (`_celt_lpc`, float build): `ac` holds
/// `lpc.len() + 1` autocorrelation values; bails out at 30 dB
/// prediction gain.
pub(crate) fn celt_lpc(lpc: &mut [f32], ac: &[f32]) {
    let p = lpc.len();
    let mut error = ac[0];
    lpc.fill(0.0);
    if ac[0] == 0.0 {
        return;
    }
    for i in 0..p {
        // This iteration's reflection coefficient.
        let mut rr = 0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        // Update the coefficients and the total error.
        lpc[i] = r;
        for j in 0..((i + 1) >> 1) {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }
        error -= r * r * error;
        // Bail out once we get 30 dB gain.
        if error < 0.001 * ac[0] {
            break;
        }
    }
}

/// All-zero filter (`celt_fir`, float build), in place: `y[i] =
/// x[i] + Σ num[j]·x[i-1-j]`, with `mem` carrying the pre-filter
/// input history (most recent first) across calls.
pub(crate) fn celt_fir_inplace(buf: &mut [f32], num: &[f32], mem: &mut [f32]) {
    let ord = num.len();
    for v in buf.iter_mut() {
        let x = *v;
        let mut sum = x;
        for j in 0..ord {
            sum += num[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = x;
        *v = sum;
    }
}

/// All-pole filter (`celt_iir`, float build), in place: `y[i] =
/// x[i] - Σ den[j]·y[i-1-j]`, with `mem` carrying the output
/// history (most recent first).
pub(crate) fn celt_iir_inplace(buf: &mut [f32], den: &[f32], mem: &mut [f32]) {
    let ord = den.len();
    for v in buf.iter_mut() {
        let mut sum = *v;
        for j in 0..ord {
            sum -= den[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = sum;
        *v = sum;
    }
}

/// 2:1 pitch-analysis downsampling (`pitch_downsample`, float
/// build): a 1-2-1 smoother onto the half-rate grid (channels
/// summed), followed by a damped 4th-order LPC whitener and a fixed
/// one-tap 0.8 filter. `x` holds one slice per channel (equal
/// lengths); the output has `len / 2` samples.
pub(crate) fn pitch_downsample(x: &[&[f32]], x_lp: &mut [f32]) {
    let len = x[0].len();
    let half = len >> 1;
    for (i, o) in x_lp.iter_mut().enumerate().take(half).skip(1) {
        *o = 0.5 * (0.5 * (x[0][2 * i - 1] + x[0][2 * i + 1]) + x[0][2 * i]);
    }
    x_lp[0] = 0.5 * (0.5 * x[0][1] + x[0][0]);
    if x.len() == 2 {
        for (i, o) in x_lp.iter_mut().enumerate().take(half).skip(1) {
            *o += 0.5 * (0.5 * (x[1][2 * i - 1] + x[1][2 * i + 1]) + x[1][2 * i]);
        }
        x_lp[0] += 0.5 * (0.5 * x[1][1] + x[1][0]);
    }

    let mut ac = [0f32; 5];
    celt_autocorr(&x_lp[..half], &mut ac, None, 0);

    // Noise floor -40 dB.
    ac[0] *= 1.0001;
    // Lag windowing.
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        *a -= *a * (0.008 * i as f32) * (0.008 * i as f32);
    }
    let mut lpc = [0f32; 4];
    celt_lpc(&mut lpc, &ac);
    let mut tmp = 1.0f32;
    for l in lpc.iter_mut() {
        tmp *= 0.9;
        *l *= tmp;
    }
    let mut mem = [0f32; 4];
    celt_fir_inplace(&mut x_lp[..half], &lpc, &mut mem);

    let mut mem1 = [0f32; 1];
    celt_fir_inplace(&mut x_lp[..half], &[0.8], &mut mem1);
}

/// The two-best tracker over a cross-correlation sweep
/// (`find_best_pitch`, float build).
fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [usize; 2] {
    let mut best_num = [-1f32; 2];
    let mut best_den = [0f32; 2];
    let mut best_pitch = [0usize, 1];
    let mut syy = 1f32;
    for &v in y.iter().take(len) {
        syy += v * v;
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let num = xcorr[i] * xcorr[i];
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        syy = syy.max(1.0);
    }
    best_pitch
}

/// Coarse-to-fine pitch search (`pitch_search`, float build) over an
/// already 2:1-downsampled signal: a 4x-decimated coarse sweep, a
/// 2x refinement around the two best candidates, and a
/// pseudo-interpolation step. `x_lp` is the current window (`len/2`
/// samples), `y` the full lagged history (`(len+max_pitch)/2`
/// samples, `x_lp` being its tail); `len` and `max_pitch` count
/// full-rate samples. Returns the full-rate pitch lag.
pub(crate) fn pitch_search(x_lp: &[f32], y: &[f32], len: usize, max_pitch: usize) -> usize {
    let lag = len + max_pitch;
    // Downsample by 2 again.
    let x_lp4: Vec<f32> = (0..len >> 2).map(|j| x_lp[2 * j]).collect();
    let y_lp4: Vec<f32> = (0..lag >> 2).map(|j| y[2 * j]).collect();

    // Coarse search with 4x decimation.
    let mut xcorr = vec![0f32; max_pitch >> 1];
    for i in 0..max_pitch >> 2 {
        let mut sum = 0f32;
        for j in 0..len >> 2 {
            sum += x_lp4[j] * y_lp4[i + j];
        }
        xcorr[i] = sum.max(-1.0);
    }
    let best = find_best_pitch(&xcorr, &y_lp4, len >> 2, max_pitch >> 2);

    // Finer search with 2x decimation.
    for i in 0..max_pitch >> 1 {
        xcorr[i] = 0.0;
        if (i as i32 - 2 * best[0] as i32).abs() > 2 && (i as i32 - 2 * best[1] as i32).abs() > 2 {
            continue;
        }
        let mut sum = 0f32;
        for j in 0..len >> 1 {
            sum += x_lp[j] * y[i + j];
        }
        xcorr[i] = sum.max(-1.0);
    }
    let best = find_best_pitch(&xcorr, y, len >> 1, max_pitch >> 1);

    // Refine by pseudo-interpolation.
    let offset: i32 = if best[0] > 0 && best[0] < (max_pitch >> 1) - 1 {
        let a = xcorr[best[0] - 1];
        let b = xcorr[best[0]];
        let c = xcorr[best[0] + 1];
        if c - a > 0.7 * (b - a) {
            1
        } else if a - c > 0.7 * (b - c) {
            -1
        } else {
            0
        }
    } else {
        0
    };
    (2 * best[0] as i32 - offset) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Levinson recursion inverts a known AR(2) process: filter
    /// white noise through 1/(1 - a1 z^-1 - a2 z^-2), estimate, and
    /// recover the pole polynomial.
    #[test]
    fn lpc_recovers_ar2() {
        let (a1, a2) = (1.2f32, -0.7f32);
        let mut x = vec![0f32; 4096];
        let mut seed = 12345u32;
        let mut prev = (0f32, 0f32);
        for v in x.iter_mut() {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let w = ((seed >> 16) as i32 - 32_768) as f32 / 32_768.0;
            let s = w + a1 * prev.0 + a2 * prev.1;
            prev = (s, prev.0);
            *v = s;
        }
        let mut ac = [0f32; 3];
        celt_autocorr(&x, &mut ac, None, 0);
        let mut lpc = [0f32; 2];
        celt_lpc(&mut lpc, &ac);
        // celt_fir adds num[j]·x[i-1-j]: the whitener's taps are the
        // negated AR coefficients.
        assert!(
            (lpc[0] + a1).abs() < 0.05 && (lpc[1] + a2).abs() < 0.05,
            "estimated {lpc:?}"
        );
    }

    /// FIR then IIR with the same taps is the identity (the
    /// excitation/synthesis pair used by the concealment).
    #[test]
    fn fir_iir_roundtrip() {
        let num = [0.5f32, -0.25, 0.125];
        let x: Vec<f32> = (0..64).map(|i| ((i * 37) % 17) as f32 - 8.0).collect();
        let mut y = x.clone();
        let mut mem = [0f32; 3];
        celt_fir_inplace(&mut y, &num, &mut mem);
        let mut mem = [0f32; 3];
        // FIR is Y = X·(1 + Σ num_j z^-(j+1)); IIR is Y = X / (1 +
        // Σ den_j z^-(j+1)): the same taps undo each other exactly
        // (the concealment pair passes `lpc` to both).
        celt_iir_inplace(&mut y, &num, &mut mem);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    /// The pitch search finds a planted period on a periodic signal
    /// (the decode-side 67-480 Hz window).
    #[test]
    fn pitch_search_finds_period() {
        let n = 2048usize;
        let period = 240usize; // 200 Hz at 48 kHz
        let sig: Vec<f32> = (0..n)
            .map(|t| {
                let ph = (t % period) as f32 / period as f32;
                (2.0 * std::f32::consts::PI * ph).sin()
                    + 0.3 * (6.0 * std::f32::consts::PI * ph).sin()
            })
            .collect();
        let mut x_lp = vec![0f32; n >> 1];
        pitch_downsample(&[&sig], &mut x_lp);
        let poffset = 720usize;
        let found = pitch_search(&x_lp[poffset >> 1..], &x_lp, n - poffset, poffset - 100);
        let pitch_index = poffset - found;
        assert!(
            (pitch_index as i32 - period as i32).abs() <= 4
                || (pitch_index as i32 - (period / 2) as i32).abs() <= 4,
            "found lag {pitch_index}, planted {period}"
        );
    }
}
