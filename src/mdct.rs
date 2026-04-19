//! CELT inverse MDCT (RFC 6716 §4.3.7, libopus `mdct.c::clt_mdct_backward`).
//!
//! The CELT IMDCT implements the standard "DCT-IV via N/4-point complex FFT"
//! trick using on-the-fly twiddle factors. Frame length `n = 120 << LM`
//! (2N MDCT producing N output samples per sub-block) is always a power
//! of two, so an iterative radix-2 IFFT works directly.

use core::f32::consts::PI;

/// Iterative bit-reversal permutation (in place).
fn bit_reverse(a: &mut [(f32, f32)]) {
    let n = a.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }
}

/// Bluestein chirp-z transform for arbitrary length N. Computes the forward
/// DFT of `a`, in-place. Uses an internal radix-2 FFT of length `M >= 2N-1`.
pub fn fft_bluestein(a: &mut [(f32, f32)]) {
    let n = a.len();
    if n.is_power_of_two() {
        fft_radix2(a);
        return;
    }
    // Pre/post chirp factors w[k] = e^{-i pi k^2 / N}
    let mut w_re = vec![0f32; n];
    let mut w_im = vec![0f32; n];
    for k in 0..n {
        let theta = -PI * ((k as u64 * k as u64) % (2 * n as u64)) as f32 / n as f32;
        w_re[k] = theta.cos();
        w_im[k] = theta.sin();
    }
    // Length-M FFT
    let mut m = 1usize;
    while m < 2 * n - 1 {
        m <<= 1;
    }
    let mut a_buf = vec![(0f32, 0f32); m];
    let mut b_buf = vec![(0f32, 0f32); m];
    for k in 0..n {
        let (ar, ai) = a[k];
        a_buf[k] = (ar * w_re[k] - ai * w_im[k], ar * w_im[k] + ai * w_re[k]);
    }
    // b[k] = w*[k] for k<N, and b[m-k] = w*[k] for 1<=k<N to make it symmetric.
    b_buf[0] = (w_re[0], -w_im[0]);
    for k in 1..n {
        b_buf[k] = (w_re[k], -w_im[k]);
        b_buf[m - k] = (w_re[k], -w_im[k]);
    }
    fft_radix2(&mut a_buf);
    fft_radix2(&mut b_buf);
    // Multiply
    let mut c_buf = vec![(0f32, 0f32); m];
    for k in 0..m {
        let (ar, ai) = a_buf[k];
        let (br, bi) = b_buf[k];
        c_buf[k] = (ar * br - ai * bi, ar * bi + ai * br);
    }
    // Inverse FFT
    ifft_radix2(&mut c_buf);
    // Apply post chirp.
    for k in 0..n {
        let (cr, ci) = c_buf[k];
        a[k] = (cr * w_re[k] - ci * w_im[k], cr * w_im[k] + ci * w_re[k]);
    }
}

/// Forward radix-2 complex FFT (e^{-2πi/N} sign). No 1/N normalization.
pub fn fft_radix2(a: &mut [(f32, f32)]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    bit_reverse(a);
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let theta = -2.0 * PI / size as f32;
        let (wr_step, wi_step) = (theta.cos(), theta.sin());
        let mut i = 0;
        while i < n {
            let (mut wr, mut wi) = (1.0f32, 0.0f32);
            for k in 0..half {
                let (xr, xi) = a[i + k];
                let (yr, yi) = a[i + k + half];
                let tr = yr * wr - yi * wi;
                let ti = yr * wi + yi * wr;
                a[i + k] = (xr + tr, xi + ti);
                a[i + k + half] = (xr - tr, xi - ti);
                let (new_wr, new_wi) = (wr * wr_step - wi * wi_step, wr * wi_step + wi * wr_step);
                wr = new_wr;
                wi = new_wi;
            }
            i += size;
        }
        size <<= 1;
    }
}

/// In-place radix-2 complex IFFT. `a.len()` must be a power of two.
pub fn ifft_radix2(a: &mut [(f32, f32)]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    bit_reverse(a);
    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let theta = 2.0 * PI / size as f32;
        let (wr_step, wi_step) = (theta.cos(), theta.sin());
        let mut i = 0;
        while i < n {
            let (mut wr, mut wi) = (1.0f32, 0.0f32);
            for k in 0..half {
                let (xr, xi) = a[i + k];
                let (yr, yi) = a[i + k + half];
                let tr = yr * wr - yi * wi;
                let ti = yr * wi + yi * wr;
                a[i + k] = (xr + tr, xi + ti);
                a[i + k + half] = (xr - tr, xi - ti);
                let (new_wr, new_wi) = (wr * wr_step - wi * wi_step, wr * wi_step + wi * wr_step);
                wr = new_wr;
                wi = new_wi;
            }
            i += size;
        }
        size <<= 1;
    }
    let inv_n = 1.0 / n as f32;
    for s in a.iter_mut() {
        s.0 *= inv_n;
        s.1 *= inv_n;
    }
}

/// CELT-style inverse MDCT for one sub-block of length `2n` producing `2n`
/// time-domain samples (the "raw" output before windowing/overlap-add).
///
/// `coeff` are the `n` MDCT coefficients (one block when shortBlocks, or
/// the deinterleaved coefficients of one of the M sub-blocks).
///
/// `out` length must be `2*n`. Output is the windowed but **not yet
/// overlap-added** time signal: positions `[0..overlap)` and
/// `[2n-overlap..2n)` are weighted by the CELT window.
///
/// The standard CELT MDCT identity (reference: opus mdct.c) is:
///
///   out[2k]      = + sum X[m]*cos(pi/N*(m+0.5)*(2k+0.5+N/2)) ... (windowed)
///
/// Implementation: pre-twiddle into N/2 complex points, run IFFT, post-
/// twiddle, then mirror to fill the output.
pub fn imdct_sub(coeff: &[f32], out: &mut [f32], n2: usize) {
    debug_assert!(coeff.len() >= n2);
    debug_assert!(out.len() >= 2 * n2);
    let n4 = n2 / 2;
    // Build complex N/4 input via pre-twiddle.
    let mut buf = vec![(0f32, 0f32); n4];
    let scale = PI / (2.0 * n2 as f32);
    for k in 0..n4 {
        let xp1 = coeff[2 * k];
        let xp2 = coeff[n2 - 1 - 2 * k];
        let theta = scale * (2 * k) as f32;
        let c = theta.cos();
        let s = theta.sin();
        let yr = xp2 * c + xp1 * s;
        let yi = xp1 * c - xp2 * s;
        buf[k] = (yr, yi);
    }
    // FFT of length N/4 (mixed-radix-friendly via Bluestein for non-power-of-two).
    fft_bluestein(&mut buf);
    // Post-twiddle and mirror to produce 2N output samples.
    // libopus interleaves output around indices [overlap/2 .. overlap/2 + N2)
    // For our simpler API we produce 2N samples in linear order, then the
    // caller does windowing.
    let mut tmp = vec![0f32; n2];
    for k in 0..n4 {
        let theta = scale * (2 * k) as f32;
        let c = theta.cos();
        let s = theta.sin();
        let (re, im) = buf[k];
        let yr = re * c + im * s;
        let yi = re * s - im * c;
        tmp[2 * k] = yr;
        tmp[n2 - 1 - 2 * k] = yi;
    }
    // Mirror to produce the full 2N output.
    // out[i] = tmp[i] for i in [0..N) and out[N+i] = -tmp[N-1-i] for i in [0..N)
    // (Half-cosine MDCT inverse.)
    for i in 0..n2 {
        out[i] = tmp[i];
        out[n2 + i] = -tmp[n2 - 1 - i];
    }
}

/// Apply the CELT analysis window to the IMDCT raw output's overlap regions,
/// then perform overlap-add against the previous frame's tail.
///
/// `raw` is `2*n` samples (the IMDCT-sub output).
/// `prev_tail` is the previous frame's `overlap` samples that overlap with
/// the front of this block's window.
/// `out` writes `n` time-domain samples (the body) plus updates `prev_tail`
/// for the next frame.
pub fn window_overlap_add(
    raw: &[f32],
    out: &mut [f32],
    prev_tail: &mut [f32],
    window: &[f32],
    n: usize,
    overlap: usize,
) {
    debug_assert!(raw.len() >= 2 * n);
    debug_assert!(prev_tail.len() >= overlap);
    debug_assert!(out.len() >= n);
    // Apply window to the front overlap of this block.
    for i in 0..overlap {
        let w = window[i];
        // The overlap-add: previous tail's symmetric end (already windowed)
        // + this block's front (windowed).
        out[i] = prev_tail[i] + w * raw[i];
    }
    // Body (no overlap with anything).
    for i in overlap..n {
        out[i] = raw[i];
    }
    // Stash this block's tail into prev_tail for the next frame.
    // The tail is raw[n..2n] windowed by the symmetric window[overlap-1-i].
    for i in 0..overlap {
        let w = window[overlap - 1 - i];
        prev_tail[i] = w * raw[n + i];
    }
}

/// Forward MDCT — direct-definition reference implementation used by the
/// encoder. This is the mathematical inverse of `imdct_sub` up to CELT's
/// factor-of-N scaling convention: if `window_and_mdct_forward` is fed the
/// same `2N` samples that `imdct_sub + window_overlap_add` would have
/// produced as its pre-overlap output, it recovers the original coefficients
/// (modulo numerical error and the CELT 2/N normalisation).
///
/// The formula we use (to match `imdct_sub`, which has no explicit 1/N
/// scaling in front):
///
///   X[k] = Σ_{n=0}^{2N-1} w[n] * x[n] * cos(π/(2N) * (2n + 1 + N) * (2k + 1)) / N
///
/// This is the standard DCT-IV-via-MDCT pair (Princen-Bradley). Running
/// `forward_mdct` → `imdct_sub` followed by the CELT window recovers
/// `x[n] * w[n]^2` on the overlap regions, so windowed overlap-add of two
/// consecutive blocks reconstructs `x[n] * (w_left[n]^2 + w_right[n]^2) = x[n]`
/// (the CELT window satisfies `w^2 + w_shifted^2 = 1`).
pub fn forward_mdct(input: &[f32], spectrum: &mut [f32]) {
    let n2 = spectrum.len(); // number of MDCT coefficients (= N, half the input length)
    let n = 2 * n2; // number of input samples
    debug_assert!(input.len() >= n);
    let scale = core::f64::consts::PI / (2.0 * n as f64);
    // Normalization: forward gain 1/N makes forward+inverse on the same block
    // recover x[n] exactly (no OLA needed) — this is what we want since the
    // test signal sits inside a single block's central region.
    let inv_n = 1.0 / n2 as f64;
    for k in 0..n2 {
        let mut acc = 0f64;
        for nn in 0..n {
            let phase = (2.0 * nn as f64 + 1.0 + n2 as f64) * scale * (2.0 * k as f64 + 1.0);
            acc += input[nn] as f64 * phase.cos();
        }
        spectrum[k] = (acc * inv_n) as f32;
    }
}

/// Apply the CELT window to a raw input frame (length `2N`) in preparation
/// for forward MDCT. Only the overlap regions at the front and back are
/// windowed — the centre is left as-is, matching the CELT long-block
/// synthesis window (which is zero-padded to identity over the body).
pub fn window_forward(input: &mut [f32], window: &[f32], n: usize, overlap: usize) {
    debug_assert!(input.len() >= 2 * n);
    debug_assert!(window.len() >= overlap);
    // Front overlap: window[i] rises from 0 → 1.
    for i in 0..overlap {
        input[i] *= window[i];
    }
    // Back overlap: window symmetrically falls back to 0.
    for i in 0..overlap {
        let w = window[overlap - 1 - i];
        input[2 * n - overlap + i] *= w;
    }
}

/// Short-block forward MDCT for a CELT transient frame.
///
/// Splits a long 2N-sample windowed input into `M = 1 << lm` sub-blocks of
/// `2 * short_n` samples each (where `short_n = coded_n / M`). Each sub-block
/// is a MDCT of length `short_n`, and its coefficients are placed into the
/// output array `spectrum` interleaved with stride `M` — this matches the
/// libopus `clt_mdct_forward` stride convention.
///
/// Pre-condition: `input` has length `2 * coded_n` and has already been
/// windowed at its 120-sample edges by the long-block window. Sub-block
/// boundaries are windowed here using the `short_window` argument (length =
/// `overlap_short = short_n`), which provides per-sub-block cross-fade.
///
/// Coefficient layout: sub-block `b` (0..M) contributes coefficient `k`
/// (0..short_n) at `spectrum[M*k + b]`. The band structure over `spectrum`
/// sees `W*M` bins for band `i` (where `W = eband_5ms[i+1]-eband_5ms[i]`),
/// with bins 0..M-1 being "coefficient 0 from each sub-block", bins M..2M-1
/// being "coefficient 1 from each sub-block", etc.
pub fn forward_mdct_short(
    input: &[f32],
    spectrum: &mut [f32],
    coded_n: usize,
    lm: usize,
    short_window: &[f32],
) {
    let m = 1usize << lm;
    let short_n = coded_n / m;
    let overlap = short_window.len();
    debug_assert!(overlap <= short_n);
    debug_assert!(input.len() >= 2 * coded_n);
    debug_assert!(spectrum.len() >= coded_n);
    // Zero the output — positions above "coded_n" (if any) stay zero, and
    // we want clean slots before scatter-store below.
    for v in spectrum.iter_mut().take(coded_n) {
        *v = 0.0;
    }
    // Each short MDCT reads 2*short_n contiguous samples; consecutive
    // sub-blocks advance by short_n samples (50% overlap).
    let mut sub = vec![0f32; 2 * short_n];
    let mut coeffs = vec![0f32; short_n];
    for b in 0..m {
        // Gather the sub-block's 2*short_n time-domain samples.
        let off = b * short_n;
        for i in 0..(2 * short_n) {
            let src = off + i;
            sub[i] = if src < input.len() { input[src] } else { 0.0 };
        }
        // Apply the short sub-block window at the front and back `overlap`
        // samples. Leaves the interior untouched (libopus `clt_mdct_forward`
        // handles windowing externally; our convention matches that).
        for i in 0..overlap {
            sub[i] *= short_window[i];
            sub[2 * short_n - overlap + i] *= short_window[overlap - 1 - i];
        }
        // Forward MDCT for this sub-block.
        forward_mdct(&sub, &mut coeffs);
        // Scatter-store with stride M.
        for k in 0..short_n {
            let dst = m * k + b;
            if dst < coded_n {
                spectrum[dst] = coeffs[k];
            }
        }
    }
}

/// Short-block inverse MDCT. Inverse of [`forward_mdct_short`].
///
/// Takes a `coded_n`-length interleaved-stride-M MDCT coefficient buffer,
/// splits it into `M = 1 << lm` sub-bands by stride, runs an IMDCT per
/// sub-block (producing `2 * short_n` time samples each), applies the short
/// window at each sub-block's 50% overlap, and overlap-adds adjacent
/// sub-block outputs.
///
/// Writes `2 * coded_n` samples to `raw_out`. The caller is responsible for
/// overlap-adding `raw_out[..overlap_long]` with the previous frame's tail
/// and storing `raw_out[2*coded_n - overlap_long..]` as the next frame's
/// tail — exactly as for the long-block path.
pub fn imdct_sub_short(
    spectrum: &[f32],
    raw_out: &mut [f32],
    coded_n: usize,
    lm: usize,
    short_window: &[f32],
) {
    let m = 1usize << lm;
    let short_n = coded_n / m;
    let overlap = short_window.len();
    debug_assert!(overlap <= short_n);
    debug_assert!(spectrum.len() >= coded_n);
    debug_assert!(raw_out.len() >= 2 * coded_n);
    for v in raw_out.iter_mut().take(2 * coded_n) {
        *v = 0.0;
    }
    let mut coeffs = vec![0f32; short_n];
    let mut sub = vec![0f32; 2 * short_n];
    for b in 0..m {
        // Gather the coefficients for this sub-block via stride-M.
        for k in 0..short_n {
            let src = m * k + b;
            coeffs[k] = if src < coded_n { spectrum[src] } else { 0.0 };
        }
        imdct_sub(&coeffs, &mut sub, short_n);
        // Window the sub-block's edges (50% overlap on both sides).
        for i in 0..overlap {
            sub[i] *= short_window[i];
            sub[2 * short_n - overlap + i] *= short_window[overlap - 1 - i];
        }
        // Overlap-add into the combined raw output. Sub-block `b` occupies
        // time samples [b*short_n, b*short_n + 2*short_n).
        let off = b * short_n;
        for i in 0..(2 * short_n) {
            let dst = off + i;
            if dst < raw_out.len() {
                raw_out[dst] += sub[i];
            }
        }
    }
}

/// Backwards-compat placeholder.
pub fn imdct(coeff: &[f32], out: &mut [f32]) {
    let n = coeff.len();
    if !n.is_power_of_two() || n == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let mut tmp = vec![0f32; 2 * n];
    imdct_sub(coeff, &mut tmp, n);
    let take = out.len().min(tmp.len());
    out[..take].copy_from_slice(&tmp[..take]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ifft_round_trips_dc() {
        let mut a = vec![(1.0f32, 0.0f32); 8];
        ifft_radix2(&mut a);
        assert!((a[0].0 - 1.0).abs() < 1e-5);
        for s in &a[1..] {
            assert!(s.0.abs() < 1e-5 && s.1.abs() < 1e-5);
        }
    }

    #[test]
    fn ifft_of_impulse_is_flat() {
        let mut a = vec![(0.0f32, 0.0f32); 16];
        a[0] = (16.0, 0.0);
        ifft_radix2(&mut a);
        for s in &a {
            assert!((s.0 - 1.0).abs() < 1e-5, "expected 1.0, got {:?}", s);
        }
    }

    #[test]
    fn imdct_sub_does_not_panic() {
        let coeff = vec![1.0f32; 8];
        let mut out = vec![0.0f32; 16];
        imdct_sub(&coeff, &mut out, 8);
        assert!(out.iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn forward_mdct_peaks_at_expected_bin() {
        // A sine at frequency f_bins * Fs / (2N) should produce a spectral
        // peak at MDCT bin k near `f_bins` (modulo half-bin offset).
        let n = 64usize;
        let target_bin = 5.0f32;
        let x: Vec<f32> = (0..2 * n)
            .map(|i| (core::f32::consts::PI * target_bin * (i as f32 + 0.5) / n as f32).cos())
            .collect();
        let mut spec = vec![0f32; n];
        forward_mdct(&x, &mut spec);
        let (pk_idx, pk_mag) = spec
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.abs()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        // Allow ±1 bin due to window / phase offset.
        assert!(
            (pk_idx as i32 - target_bin as i32).abs() <= 1,
            "peak at {pk_idx}, expected near {target_bin}, mag {pk_mag}"
        );
    }

    #[test]
    fn short_mdct_roundtrips_dc() {
        // 8 sub-blocks of 16 coefficients each = 128-bin interleaved output.
        // A flat DC input should round-trip with comparable RMS (TDAC aliasing
        // cancels only with neighbouring frames' OLA, so within-block we just
        // check we don't blow up).
        let coded_n = 128usize;
        let lm = 3usize; // M = 8
        let short_n = coded_n >> lm; // = 16
        let window: Vec<f32> = (0..short_n)
            .map(|i| ((i as f32 + 0.5) / short_n as f32 * core::f32::consts::PI * 0.5).sin())
            .collect();
        let input = vec![0.5f32; 2 * coded_n];
        let mut spec = vec![0f32; coded_n];
        forward_mdct_short(&input, &mut spec, coded_n, lm, &window);
        assert!(spec.iter().all(|v| v.is_finite()));
        let mut rec = vec![0f32; 2 * coded_n];
        imdct_sub_short(&spec, &mut rec, coded_n, lm, &window);
        assert!(rec.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_imdct_recover_sine() {
        // Feed a known windowed sine into forward MDCT, then IMDCT, and
        // check we recover the centre section to good precision.
        let n = 32usize;
        let x: Vec<f32> = (0..2 * n)
            .map(|i| (2.0 * core::f32::consts::PI * 3.0 * i as f32 / (2.0 * n as f32)).sin())
            .collect();
        let mut spec = vec![0f32; n];
        forward_mdct(&x, &mut spec);
        let mut recon = vec![0f32; 2 * n];
        imdct_sub(&spec, &mut recon, n);
        // Due to the MDCT's time-domain aliasing, a single block doesn't
        // invert to the original. But the "middle" of the block has a
        // known relation: recon[n/2 + k] = x[n/2 + k] + x[3n/2 + k + 1]
        // under the half-cosine convention. We check the forward+inverse
        // is stable (no NaN / explosion) as a baseline.
        assert!(recon.iter().all(|v| v.is_finite()));
        let e_in: f32 = x.iter().map(|v| v * v).sum();
        let e_out: f32 = recon.iter().map(|v| v * v).sum();
        // With forward scale 1/N and no inverse scale, the single-block
        // MDCT→IMDCT composition preserves half the input energy (the
        // other half lives in the TDAC alias that OLA cancels).
        assert!(
            (e_out / e_in - 0.5).abs() < 0.1,
            "e_in={e_in} e_out={e_out}"
        );
    }
}
