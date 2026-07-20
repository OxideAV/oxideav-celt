//! Encoder-side control decisions from RFC 6716 §5.3 — the §5.3.4.1
//! band-boost rule, the §5.3.4.2 allocation-trim decision, and the
//! §5.3.3 intra/inter coarse-mode selection.
//!
//! ## §5.3.4.1 Band Boost
//!
//! > "The reference encoder makes a decision to boost a band when the
//! > energy of that band is significantly higher than that of the
//! > neighboring bands. Let E_j be the log-energy of band j, we
//! > define
//! >
//! >   D_j = 2*E_j - E_j-1 - E_j+1
//! >
//! > The allocation of band j is boosted once if D_j > t1 and twice
//! > if D_j > t2. For LM>=1, t1=2 and t2=4, while for LM<1, t1=3 and
//! > t2=5."
//!
//! `E_j` is the per-band base-2 log-energy (1.0 = 6 dB — the same
//! axis the §4.3.2.1 coarse quantizer works on), so `t1 = 2` means "a
//! band poking 12 dB above the mean of its neighbors gets one boost
//! quantum". One boost is one §4.3.3 dynalloc quantum,
//! `min(8*N, max(48, N))` 1/8 bits for a band of `N` MDCT bins — the
//! step size of the decode-side boost loop
//! ([`decode_band_boosts`](crate::band_cap::decode_band_boosts)), so
//! a "once"/"twice" decision lands exactly on the wire grid.
//!
//! The `D_j` metric needs both neighbors, so the first and last coded
//! bands are never boosted by this rule (the RFC formula is undefined
//! there); band boosts are a free encoder choice, so the conservative
//! no-boost edge behaviour is a legal documented decision.
//!
//! Boost decisions are encoder freedom (§5.3.4: "the three mechanisms
//! that can be used by the encoder to adjust the bitrate ... are band
//! boost, allocation trim, and band skipping") — the decoder derives
//! everything from the bitstream, so any output of this rule stays in
//! lockstep automatically.
//!
//! ## §5.3.4.2 Allocation Trim
//!
//! > "The encoder starts with a safe 'default' of 5 and deviates
//! > from that default in two different ways. First, the trim can
//! > deviate by +/- 2 depending on the spectral tilt of the input
//! > signal. For signals with more low frequencies, the trim is
//! > increased by up to 2, while for signals with more high
//! > frequencies, the trim is decreased by up to 2. For stereo
//! > inputs, the trim value can be decreased by up to 4 when the
//! > inter-channel correlation at low frequency (first 8 bands) is
//! > high."
//!
//! The RFC pins the *envelope* of the decision — the default, both
//! deviation directions, and both bounds — but not the exact
//! tilt→deviation or correlation→deviation maps. Trim is pure
//! encoder freedom (any coded value in `0..=10` keeps the decoder in
//! lockstep — it reads the trim off the wire), so
//! [`choose_alloc_trim`] fills the two maps with documented in-crate
//! choices that stay inside the RFC's envelope:
//!
//! * **Tilt:** the least-squares slope of the per-band base-2
//!   log-energy across the coded window (log-energy units per band
//!   index). The deviation is `round(-slope * TRIM_TILT_GAIN)`
//!   clamped to `-2..=2` — a *negative* slope (low-heavy signal)
//!   raises the trim, a positive slope lowers it, saturating at
//!   `|slope| >= 0.5` (3 dB per band) with `TRIM_TILT_GAIN = 4`.
//! * **Stereo:** the deviation is `round(4 * r^2)` clamped to
//!   `0..=4`, subtracted from the trim, where `r ∈ [0, 1]` is the
//!   normalized inter-channel correlation over the first 8 coded
//!   bands' MDCT bins ([`low_band_stereo_correlation`]). The
//!   quadratic keeps weakly-correlated content from moving the trim;
//!   `r = 1` (dual-mono) hits the full −4.
//!
//! ## Clean-room provenance
//!
//! RFC 6716 §5.3.4/§5.3.4.1/§5.3.4.2
//! (`docs/audio/opus/rfc6716-opus.txt`) and the §4.3.3 dynalloc
//! quantum already transcribed in [`crate::band_cap`]. The tilt and
//! correlation map constants are in-crate encoder freedom inside the
//! §5.3.4.2 envelope, documented above. No external library source
//! was consulted.

use crate::band_minimums::BAND_BINS_LM;
use crate::coarse_energy::NUM_BANDS;

/// The §5.3.4.1 boost thresholds `(t1, t2)` in base-2 log-energy
/// units: `(2, 4)` for `LM >= 1`, `(3, 5)` for `LM < 1`.
#[inline]
pub fn boost_thresholds(lm: u32) -> (f32, f32) {
    if lm >= 1 {
        (2.0, 4.0)
    } else {
        (3.0, 5.0)
    }
}

/// The §4.3.3 dynalloc boost quantum for a band of `n` MDCT bins, in
/// 1/8 bits: `min(8*N, max(48, N))` — the per-"1"-bit step of the
/// decode-side boost loop.
#[inline]
pub fn boost_quanta_8th(n: u32) -> i32 {
    (8 * n).min(n.max(48)) as i32
}

/// The §5.3.4.1 band-boost decision over a coded-band window.
///
/// `window_log_energy` holds the per-band base-2 log-energies of the
/// coded bands `[start, end)` (window-relative indexing — the
/// [`analyze_bands_f32`](crate::band_analysis::analyze_bands_f32)
/// output order). For each *interior* band `j` the §5.3.4.1 contrast
/// `D_j = 2*E_j - E_{j-1} - E_{j+1}` is compared against the per-LM
/// thresholds; a band is boosted once (`D_j > t1`) or twice
/// (`D_j > t2`), each boost worth one §4.3.3 dynalloc quantum for
/// that band's width.
///
/// Returns the per-coded-band boost targets in 1/8 bits — exactly the
/// `target_boost` input
/// [`encode_band_boosts`](crate::band_cap::encode_band_boosts) and
/// the frame encoders take (the encode loop gate-truncates against
/// the live budget, so an over-budget decision degrades exactly the
/// way the wire format prescribes).
///
/// Returns `None` when `lm > 3`, the window is out of range, or the
/// energy slice length disagrees with the window.
pub fn choose_band_boosts(
    window_log_energy: &[f32],
    lm: u32,
    start: usize,
    end: usize,
) -> Option<Vec<i32>> {
    if lm > 3 || start > end || end > NUM_BANDS {
        return None;
    }
    let coded = end - start;
    if window_log_energy.len() != coded {
        return None;
    }
    let (t1, t2) = boost_thresholds(lm);
    let mut out = vec![0i32; coded];
    // Interior bands only: D_j needs both neighbors inside the window.
    for j in 1..coded.saturating_sub(1) {
        let d = 2.0 * window_log_energy[j] - window_log_energy[j - 1] - window_log_energy[j + 1];
        let times = if d > t2 {
            2
        } else if d > t1 {
            1
        } else {
            0
        };
        if times > 0 {
            let n = BAND_BINS_LM[lm as usize][start + j];
            out[j] = times * boost_quanta_8th(n);
        }
    }
    Some(out)
}

/// The tilt→deviation gain of the §5.3.4.2 trim map: the ±2 bound
/// saturates at a spectral tilt of `2 / TRIM_TILT_GAIN = 0.5`
/// log2-energy units per band (3 dB per band). An in-crate constant —
/// the RFC pins the direction and the bound, not the map (see the
/// module docs).
pub const TRIM_TILT_GAIN: f32 = 4.0;

/// The least-squares spectral-tilt slope of a per-band log-energy
/// window, in base-2 log-energy units per band index.
///
/// This is the "spectral tilt of the input signal" measure feeding
/// the §5.3.4.2 trim deviation: negative for low-heavy signals
/// (energy falling with frequency), positive for high-heavy ones.
/// Windows shorter than 2 bands have no defined tilt and return 0.
pub fn spectral_tilt_slope(window_log_energy: &[f32]) -> f32 {
    let n = window_log_energy.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    let mean_x = (nf - 1.0) / 2.0;
    let mean_y = window_log_energy.iter().sum::<f32>() / nf;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (j, &e) in window_log_energy.iter().enumerate() {
        let dx = j as f32 - mean_x;
        num += dx * (e - mean_y);
        den += dx * dx;
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// The normalized inter-channel correlation over the first 8 coded
/// bands' MDCT bins — the "inter-channel correlation at low
/// frequency (first 8 bands)" measure of the §5.3.4.2 stereo trim
/// deviation.
///
/// `left` / `right` are the two channels' coded-window spectra (the
/// band-contiguous layout the frame encoders consume, window starting
/// at absolute band `start`); the correlation is `|<L, R>| /
/// (||L|| * ||R||)` over the bins of absolute bands `0..8`, i.e. the
/// first `sum(N[0..8])` bins — only defined for a window with
/// `start == 0` (a Hybrid window codes no low bands, so the stereo
/// deviation does not apply there).
///
/// Returns `None` when `start != 0`, `lm > 3`, either spectrum is
/// shorter than the low-band span, or either channel has zero
/// low-band energy (no direction to correlate). The result is in
/// `[0, 1]`.
pub fn low_band_stereo_correlation(
    left: &[f32],
    right: &[f32],
    lm: u32,
    start: usize,
) -> Option<f32> {
    if start != 0 || lm > 3 {
        return None;
    }
    let span: u32 = BAND_BINS_LM[lm as usize][..8].iter().sum();
    let span = span as usize;
    if left.len() < span || right.len() < span {
        return None;
    }
    let mut dot = 0.0f64;
    let mut ll = 0.0f64;
    let mut rr = 0.0f64;
    for (l, r) in left[..span].iter().zip(&right[..span]) {
        dot += f64::from(*l) * f64::from(*r);
        ll += f64::from(*l) * f64::from(*l);
        rr += f64::from(*r) * f64::from(*r);
    }
    if ll <= 0.0 || rr <= 0.0 {
        return None;
    }
    Some((dot.abs() / (ll.sqrt() * rr.sqrt())).min(1.0) as f32)
}

/// The §5.3.4.2 allocation-trim decision.
///
/// Starts from the safe default of 5 and applies the two RFC-pinned
/// deviations (see the module docs for the in-crate maps):
///
/// * `+/-2` from the spectral tilt of `window_log_energy` (the coded
///   window's per-band base-2 log-energies —
///   [`spectral_tilt_slope`]): low-heavy signals raise the trim,
///   high-heavy signals lower it.
/// * up to `-4` from a high low-frequency inter-channel correlation
///   (`stereo_low_band_correlation`, the
///   [`low_band_stereo_correlation`] output; pass `None` for mono or
///   when the window codes no low bands).
///
/// The result is clamped to the legal `0..=10` field range. Trim is
/// encoder freedom: whatever this returns is coded on the wire and
/// read back by the decoder, so lockstep holds for any output.
pub fn choose_alloc_trim(
    window_log_energy: &[f32],
    stereo_low_band_correlation: Option<f32>,
) -> u8 {
    let slope = spectral_tilt_slope(window_log_energy);
    let tilt_dev = (-slope * TRIM_TILT_GAIN).round().clamp(-2.0, 2.0) as i32;
    let stereo_dev = match stereo_low_band_correlation {
        Some(r) => {
            let r = r.clamp(0.0, 1.0);
            (4.0 * r * r).round().clamp(0.0, 4.0) as i32
        }
        None => 0,
    };
    (5 + tilt_dev - stereo_dev).clamp(0, 10) as u8
}

/// The §5.3.4.2 allocation-trim decision at the reference listing's
/// exact operating point (transcribed from the §A.1 listing,
/// `alloc_trim_analysis`) — the trim map the RFC's §5.3.4.2 envelope
/// describes:
///
/// * **Stereo:** the low-frequency inter-channel correlation is the
///   mean over the first 8 bands of the per-band dot product of the
///   two channels' **unit-norm** band shapes (`x`/`y`, the coded
///   band-contiguous normalized spectra). The trim drops by 4/3/2/1
///   for correlations above `0.995 / 0.92 / 0.85 / 0.8`.
/// * **Tilt:** `diff = mean over channels and bands < end-1 of
///   E_i * (2 + 2*i - NUM_BANDS) / 2`, the first-moment spectral
///   tilt of the per-band base-2 log-energies (`band_log_e`,
///   eMeans-relative). The trim drops by 1 for `diff > 2` and again
///   for `diff > 8`, and rises by 1 for `diff < -4` and again for
///   `diff < -10`.
///
/// The result clamps to the legal `0..=10`. Trim is encoder freedom
/// (the coded value keeps the decoder in lockstep), so this is a
/// quality decision, not a wire requirement — but it is the map the
/// reference rate-distortion behaviour was tuned around.
pub fn alloc_trim_analysis(
    x: &[f32],
    y: Option<&[f32]>,
    band_log_e: &[[f32; NUM_BANDS]; 2],
    end: usize,
    lm: u32,
) -> u8 {
    use crate::band_layout::EBAND_EDGES_5MS;
    let mut trim = 5i32;
    let m = 1usize << lm;
    let eb = |i: usize| m * EBAND_EDGES_5MS[i] as usize;
    let channels = 1 + usize::from(y.is_some());
    if let Some(y) = y {
        // Inter-channel correlation over the first 8 bands' unit-norm
        // shapes: each band's dot product is its correlation, so the
        // mean over 8 bands is the listing's Q10 `sum`.
        let mut sum = 0.0f32;
        for i in 0..8 {
            let mut partial = 0.0f32;
            for (l, r) in x[eb(i)..eb(i + 1)].iter().zip(&y[eb(i)..eb(i + 1)]) {
                partial += l * r;
            }
            sum += partial;
        }
        sum *= 1.0 / 8.0;
        if sum > 0.995 {
            trim -= 4;
        } else if sum > 0.92 {
            trim -= 3;
        } else if sum > 0.85 {
            trim -= 2;
        } else if sum > 0.8 {
            trim -= 1;
        }
    }
    let mut diff = 0.0f32;
    for ch in band_log_e.iter().take(channels) {
        for (i, &e) in ch.iter().enumerate().take(end.saturating_sub(1)) {
            diff += e * (2 + 2 * i as i32 - NUM_BANDS as i32) as f32;
        }
    }
    diff /= (2 * channels * (end - 1)) as f32;
    if diff > 2.0 {
        trim -= 1;
    }
    if diff > 8.0 {
        trim -= 1;
    }
    if diff < -4.0 {
        trim += 1;
    }
    if diff < -10.0 {
        trim += 1;
    }
    trim.clamp(0, 10) as u8
}

/// The §5.3.5 dual-vs-mid/side stereo decision at the reference
/// listing's exact operating point (transcribed from the §A.1
/// listing, `stereo_analysis`): the L1 norms of the L/R pair and the
/// (unnormalized) M/S pair over the first 13 bands' coded bins model
/// the two codings' entropy, the M/S norm scaled by the listing's
/// literal `0.707107`, and the M/S side additionally charged
/// `thetas` extra degrees of freedom (13, minus 8 for `LM <= 1` —
/// the low bands that code no theta there). Returns `true` for
/// **dual** stereo. `x` / `y` are the coded-window unit-norm spectra
/// (band-contiguous), `m = 1 << LM`.
pub fn stereo_analysis(x: &[f32], y: &[f32], lm: u32) -> bool {
    use crate::band_layout::EBAND_EDGES_5MS;
    let m = 1usize << lm;
    let span = m * EBAND_EDGES_5MS[13] as usize;
    let mut sum_lr = 1e-15f64;
    let mut sum_ms = 1e-15f64;
    for (&l, &r) in x[..span].iter().zip(&y[..span]) {
        let (l, r) = (f64::from(l), f64::from(r));
        sum_lr += l.abs() + r.abs();
        sum_ms += (l + r).abs() + (l - r).abs();
    }
    #[allow(clippy::approx_constant)] // the listing's literal constant
    let sum_ms = 0.707107 * sum_ms;
    let mut thetas = 13i64;
    // No thetas for the lower bands at LM <= 1.
    if lm <= 1 {
        thetas -= 8;
    }
    let w = (EBAND_EDGES_5MS[13] as i64) << (lm + 1);
    (w + thetas) as f64 * sum_ms > w as f64 * sum_lr
}

/// The number of bands the §5.3.5 mid/side-vs-dual comparison runs
/// over: "comparing the estimated entropy with and without coupling
/// over the first 13 bands".
pub const MID_SIDE_DECISION_BANDS: usize = 13;

/// The §5.3.5 "extra degrees of freedom" term `E` for mid-side
/// coding: "For LM>1, E=13, otherwise E=5" (every band with more than
/// two MDCT bins needs one extra degree of freedom when coded in
/// mid-side).
#[inline]
pub fn mid_side_extra_dof(lm: u32) -> u32 {
    if lm > 1 {
        13
    } else {
        5
    }
}

/// The §5.3.5 mid/side-vs-dual stereo decision.
///
/// > "This decision is made by comparing the estimated entropy with
/// > and without coupling over the first 13 bands ... Let L1_ms and
/// > L1_lr be the L1-norm of the mid-side vector and the L1-norm of
/// > the left-right vector, respectively. The decision to use
/// > mid-side is made if and only if
/// > L1_ms / (bins + E) < L1_lr / bins"
///
/// `left` / `right` are the two channels' coded-window spectra
/// (band-contiguous, window starting at absolute band `start`); the
/// norms run over the bins of the first
/// [`MID_SIDE_DECISION_BANDS`] absolute bands, `bins` is that span,
/// and `E` is [`mid_side_extra_dof`]. The mid/side vector uses the
/// orthonormal rotation `m = (l + r)/sqrt(2)`, `s = (l - r)/sqrt(2)`
/// — the RFC does not pin the scaling convention, and the
/// norm-preserving rotation is the natural reading given §5.3.5's
/// framing ("CELT applies mid-side stereo coupling in the normalized
/// domain"); the convention is documented because it shifts the
/// decision boundary.
///
/// Returns `Some(true)` for mid-side, `Some(false)` for dual, `None`
/// when the decision is undefined (`start != 0` — the first 13 bands
/// are not coded — `lm > 3`, either spectrum shorter than the span,
/// or a zero `L1_lr`).
///
/// Since r417 the reference-compatible encoder
/// ([`crate::ref_encode::CeltRefEncoder`]) carries this verdict on
/// the wire: the exact §4.3.4 band loop codes the `itheta` mid/side
/// coupling, so a `Some(true)` frame is coupled and a `Some(false)`
/// frame signals dual stereo (§5.3.5 sanctions either choice on any
/// frame). The pre-r414 `pcm_encode` drivers still pin the uncoupled
/// path.
pub fn choose_mid_side_stereo(left: &[f32], right: &[f32], lm: u32, start: usize) -> Option<bool> {
    if start != 0 || lm > 3 {
        return None;
    }
    let span: u32 = BAND_BINS_LM[lm as usize][..MID_SIDE_DECISION_BANDS]
        .iter()
        .sum();
    let span = span as usize;
    if left.len() < span || right.len() < span {
        return None;
    }
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut l1_ms = 0.0f64;
    let mut l1_lr = 0.0f64;
    for (l, r) in left[..span].iter().zip(&right[..span]) {
        let l = f64::from(*l);
        let r = f64::from(*r);
        l1_ms += ((l + r) * inv_sqrt2).abs() + ((l - r) * inv_sqrt2).abs();
        l1_lr += l.abs() + r.abs();
    }
    if l1_lr <= 0.0 {
        return None;
    }
    let bins = span as f64;
    let e = f64::from(mid_side_extra_dof(lm));
    Some(l1_ms / (bins + e) < l1_lr / bins)
}

/// The §5.3.5 Table 66 intensity-stereo threshold: the first band
/// using intensity coding, decided "based on the bitrate alone.
/// After taking into account the frame size by subtracting 80 bits
/// per frame for coarse energy".
///
/// `frame_bits` is the coded frame size in bits (`frame_bytes * 8`);
/// `lm` gives the frame duration (`2.5 ms << lm`, i.e. `400 >> lm`
/// frames per second), so the compared rate is
/// `(frame_bits - 80) * (400 >> lm)` bits per second. Table 66 maps
/// it to the start band:
///
/// | rate (kbit/s) | start band |
/// |---------------|------------|
/// | < 35          | 8          |
/// | 35–50         | 12         |
/// | 50–68         | 16         |
/// | 68–84         | 18         |
/// | 84–102        | 19         |
/// | 102–130       | 20         |
/// | > 130         | disabled   |
///
/// (The RFC's printed table labels the fourth row "84-84"; the
/// monotone reading — it covers the 68–84 gap the printed rows leave —
/// is used here and noted as an apparent erratum in the RFC text.)
///
/// Returns `None` for "disabled" (rate above 130 kbit/s, and `lm > 3`
/// defensively). Since r417 the reference-compatible encoder
/// ([`crate::ref_encode::CeltRefEncoder`]) signals this threshold in
/// the exact §4.3.3 allocation walk, and the exact §4.3.4 band loop
/// codes the intensity path; the pre-r414 `pcm_encode` drivers still
/// pin "intensity never applies".
pub fn intensity_start_band(frame_bits: u32, lm: u32) -> Option<usize> {
    if lm > 3 {
        return None;
    }
    let frames_per_second = 400u64 >> lm;
    let rate_bps = u64::from(frame_bits.saturating_sub(80)) * frames_per_second;
    match rate_bps {
        r if r < 35_000 => Some(8),
        r if r < 50_000 => Some(12),
        r if r < 68_000 => Some(16),
        r if r < 84_000 => Some(18),
        r if r < 102_000 => Some(19),
        r if r <= 130_000 => Some(20),
        _ => None,
    }
}

/// The §5.3.3 coarse-energy prediction-mode decision: "it is best to
/// try encoding the coarse energy both with and without inter-frame
/// prediction such that the best prediction mode can be selected."
///
/// Runs the §4.3.2.1 coarse encode **twice** on scratch states — once
/// intra, once inter — against the same targets and budget, and
/// returns `true` (intra) iff the intra pass spends strictly fewer
/// `tell_frac` eighth-bits. Both passes quantize each band to the
/// nearest 6 dB step (the §5.3.3 per-value rule the coarse encoder
/// already applies), so the coded error is comparable and the rate is
/// the discriminating lever; ties go to **inter** (the §4.3.2.1
/// time-arm prediction reduces the next frame's cost for stationary
/// signals). §5.3.3 notes the optimal mode also depends on the
/// packet-loss rate — loss-aware weighting is transport policy and
/// stays with the caller (force `intra = true` for robustness).
///
/// `budget_bits` is the frame budget in **bits** (`frame_bytes * 8`),
/// the same value the frame encoders hand the coarse dispatch.
/// Returns `None` when the underlying coarse encode rejects the
/// window/`lm`/`channels`.
#[allow(clippy::too_many_arguments)]
pub fn choose_intra_mode(
    coarse: &crate::coarse_energy::CoarseEnergyState,
    energy_target: &[[f32; NUM_BANDS]; crate::coarse_energy::MAX_CHANNELS],
    lm: u32,
    start: usize,
    end: usize,
    channels: usize,
    budget_bits: u32,
) -> Option<bool> {
    let mut cost_8th = [0u32; 2];
    for (idx, intra) in [(0usize, false), (1usize, true)] {
        let mut enc = crate::range_encoder::RangeEncoder::new();
        // The §4.3.2.1 intra flag itself (Table 56, PDF `{7,1}/8`)
        // is part of the price of the mode — ~3 bits when set vs
        // ~0.19 when clear — so it is written into the scratch
        // encoder ahead of the coarse walk, exactly as the real
        // prefix does.
        enc.enc_bit_logp(u32::from(intra), 3).ok()?;
        let mut scratch = *coarse;
        crate::coarse_energy::encode_coarse_energy(
            &mut enc,
            &mut scratch,
            energy_target,
            intra,
            lm,
            start,
            end,
            channels,
            budget_bits,
        )
        .ok()?;
        cost_8th[idx] = enc.tell_frac();
    }
    Some(cost_8th[1] < cost_8th[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The threshold pairs are the literal §5.3.4.1 constants split at
    /// LM = 1.
    #[test]
    fn thresholds_split_at_lm_one() {
        assert_eq!(boost_thresholds(0), (3.0, 5.0));
        assert_eq!(boost_thresholds(1), (2.0, 4.0));
        assert_eq!(boost_thresholds(2), (2.0, 4.0));
        assert_eq!(boost_thresholds(3), (2.0, 4.0));
    }

    /// The quantum is the §4.3.3 `min(8*N, max(48, N))` decode-loop
    /// step.
    #[test]
    fn quanta_matches_dynalloc_step() {
        assert_eq!(boost_quanta_8th(1), 8); // 8*1 < max(48, 1)
        assert_eq!(boost_quanta_8th(4), 32); // 8*4 < 48
        assert_eq!(boost_quanta_8th(6), 48); // 8*6 == 48
        assert_eq!(boost_quanta_8th(8), 48); // 64 > 48 → 48
        assert_eq!(boost_quanta_8th(88), 88); // N > 48 → N
    }

    /// A flat envelope earns no boosts at any LM.
    #[test]
    fn flat_envelope_earns_no_boosts() {
        for lm in 0..=3u32 {
            let energy = vec![1.5f32; NUM_BANDS];
            let boosts = choose_band_boosts(&energy, lm, 0, NUM_BANDS).unwrap();
            assert!(boosts.iter().all(|&b| b == 0), "lm={lm}");
        }
    }

    /// An isolated peak crosses t1 (one quantum) and t2 (two quanta)
    /// exactly per the §5.3.4.1 rule, on both sides of the LM split.
    #[test]
    fn isolated_peak_boosts_once_then_twice() {
        for (lm, t1, t2) in [(0u32, 3.0f32, 5.0f32), (2, 2.0, 4.0)] {
            let mut energy = vec![0.0f32; NUM_BANDS];
            let peak = 7usize;
            // D_peak = 2*E with flat zero neighbors.
            // Just over t1 (but not t2): one quantum.
            energy[peak] = (t1 + 0.1) / 2.0;
            let boosts = choose_band_boosts(&energy, lm, 0, NUM_BANDS).unwrap();
            let n = BAND_BINS_LM[lm as usize][peak];
            assert_eq!(boosts[peak], boost_quanta_8th(n), "lm={lm} once");
            // Neighbors see D = -E_peak < 0: never boosted.
            assert_eq!(boosts[peak - 1], 0);
            assert_eq!(boosts[peak + 1], 0);
            // Just over t2: two quanta.
            energy[peak] = (t2 + 0.1) / 2.0;
            let boosts = choose_band_boosts(&energy, lm, 0, NUM_BANDS).unwrap();
            assert_eq!(boosts[peak], 2 * boost_quanta_8th(n), "lm={lm} twice");
            // At exactly t1 the strict `>` keeps the boost off.
            energy[peak] = t1 / 2.0;
            let boosts = choose_band_boosts(&energy, lm, 0, NUM_BANDS).unwrap();
            assert_eq!(boosts[peak], 0, "lm={lm} strict threshold");
        }
    }

    /// Edge bands are never boosted (D_j needs both neighbors), even
    /// with an extreme edge peak; a Hybrid window applies the same
    /// rule relative to its own edges and indexes the absolute band
    /// widths.
    #[test]
    fn edges_and_hybrid_window() {
        let mut energy = vec![0.0f32; NUM_BANDS];
        energy[0] = 100.0;
        energy[NUM_BANDS - 1] = 100.0;
        let boosts = choose_band_boosts(&energy, 2, 0, NUM_BANDS).unwrap();
        assert_eq!(boosts[0], 0);
        assert_eq!(boosts[NUM_BANDS - 1], 0);
        // Band 1 / band 19 see a huge *negative* contrast next to the
        // edge peaks: still zero.
        assert_eq!(boosts[1], 0);
        assert_eq!(boosts[NUM_BANDS - 2], 0);

        // Hybrid window 17..21: peak at absolute band 18 (window
        // index 1) uses band 18's width at LM=3.
        let window = vec![0.0f32, 3.0, 0.0, 0.0];
        let boosts = choose_band_boosts(&window, 3, 17, NUM_BANDS).unwrap();
        let n = BAND_BINS_LM[3][18];
        assert_eq!(boosts, vec![0, 2 * boost_quanta_8th(n), 0, 0]);
    }

    /// A flat envelope has zero tilt and lands on the §5.3.4.2 safe
    /// default of 5, mono and stereo-uncorrelated alike.
    #[test]
    fn trim_flat_envelope_is_default() {
        let flat = vec![1.5f32; NUM_BANDS];
        assert_eq!(spectral_tilt_slope(&flat), 0.0);
        assert_eq!(choose_alloc_trim(&flat, None), 5);
        assert_eq!(choose_alloc_trim(&flat, Some(0.0)), 5);
        // Degenerate windows: no defined tilt.
        assert_eq!(choose_alloc_trim(&[], None), 5);
        assert_eq!(choose_alloc_trim(&[3.0], None), 5);
    }

    /// The tilt deviation follows the §5.3.4.2 directions — low-heavy
    /// raises the trim, high-heavy lowers it — and saturates at ±2.
    #[test]
    fn trim_tilt_direction_and_bound() {
        // Gentle low-heavy tilt (slope -0.25): +1.
        let gentle_low: Vec<f32> = (0..NUM_BANDS).map(|b| 4.0 - 0.25 * b as f32).collect();
        assert_eq!(choose_alloc_trim(&gentle_low, None), 6);
        // Steep low-heavy tilt (slope -1.0): saturates at +2.
        let steep_low: Vec<f32> = (0..NUM_BANDS).map(|b| 10.0 - b as f32).collect();
        assert_eq!(choose_alloc_trim(&steep_low, None), 7);
        // Gentle high-heavy tilt: -1.
        let gentle_high: Vec<f32> = (0..NUM_BANDS).map(|b| 0.25 * b as f32).collect();
        assert_eq!(choose_alloc_trim(&gentle_high, None), 4);
        // Steep high-heavy tilt: saturates at -2.
        let steep_high: Vec<f32> = (0..NUM_BANDS).map(|b| b as f32).collect();
        assert_eq!(choose_alloc_trim(&steep_high, None), 3);
    }

    /// The stereo deviation subtracts up to 4 for fully-correlated
    /// low bands, is quadratic (weak correlation barely moves it),
    /// and the final trim clamps into `0..=10`.
    #[test]
    fn trim_stereo_deviation_and_clamp() {
        let flat = vec![1.0f32; NUM_BANDS];
        assert_eq!(choose_alloc_trim(&flat, Some(1.0)), 1); // 5 - 4
        assert_eq!(choose_alloc_trim(&flat, Some(0.5)), 4); // 4*0.25 = 1
        assert_eq!(choose_alloc_trim(&flat, Some(0.2)), 5); // ~0.16 rounds to 0
                                                            // Steep high tilt + full correlation: 5 - 2 - 4 = -1 → clamp 0.
        let steep_high: Vec<f32> = (0..NUM_BANDS).map(|b| b as f32).collect();
        assert_eq!(choose_alloc_trim(&steep_high, Some(1.0)), 0);
        // Out-of-range correlation input saturates safely.
        assert_eq!(choose_alloc_trim(&flat, Some(7.0)), 1);
    }

    /// `low_band_stereo_correlation` measures the first-8-bands span:
    /// identical channels score 1, orthogonal low bands score 0, a
    /// Hybrid window (start != 0) and degenerate inputs return None.
    #[test]
    fn low_band_correlation_measure() {
        let lm = 1u32;
        let span: usize = BAND_BINS_LM[lm as usize][..8].iter().sum::<u32>() as usize;
        let coded = 200usize; // 100 << lm
        let l: Vec<f32> = (0..coded)
            .map(|i| ((i * 7 + 1) % 13) as f32 - 6.0)
            .collect();
        let mut r = l.clone();
        // Identical channels: r = 1 (higher bands may differ freely).
        for v in r[span..].iter_mut() {
            *v = -*v;
        }
        let c = low_band_stereo_correlation(&l, &r, lm, 0).unwrap();
        assert!((c - 1.0).abs() < 1e-6, "identical low bands: {c}");
        // Anti-correlated low bands still score |r| = 1 (sign-blind).
        for i in 0..span {
            r[i] = -l[i];
        }
        let c = low_band_stereo_correlation(&l, &r, lm, 0).unwrap();
        assert!((c - 1.0).abs() < 1e-6, "sign-blind correlation: {c}");
        // Orthogonal low bands: r = 0. Build right as an exact
        // orthogonal permutation-with-signs of left over an even span.
        for i in 0..span {
            r[i] = if i % 2 == 0 { l[i + 1] } else { -l[i - 1] };
        }
        let c = low_band_stereo_correlation(&l, &r, lm, 0).unwrap();
        assert!(c.abs() < 1e-6, "orthogonal low bands: {c}");
        // Hybrid window / zero-energy channel / short slice: None.
        assert!(low_band_stereo_correlation(&l, &r, lm, 17).is_none());
        assert!(low_band_stereo_correlation(&l, &vec![0.0; coded], lm, 0).is_none());
        assert!(low_band_stereo_correlation(&l[..span - 1], &r, lm, 0).is_none());
        assert!(low_band_stereo_correlation(&l, &r, 4, 0).is_none());
    }

    /// The §5.3.5 mid/side decision: correlated (dual-mono) content
    /// picks mid-side (the side vector vanishes); hard-panned content
    /// at LM=1 picks dual (`sqrt(2)*bins > bins + 5` at 40 bins); and
    /// the decision is undefined off the full-band window.
    #[test]
    fn mid_side_decision_follows_l1_rule() {
        let lm = 1u32;
        let span: usize = BAND_BINS_LM[lm as usize][..MID_SIDE_DECISION_BANDS]
            .iter()
            .sum::<u32>() as usize;
        let coded = 200usize;
        let l: Vec<f32> = (0..coded)
            .map(|i| ((i * 5 + 3) % 11) as f32 - 5.0)
            .collect();

        // Dual-mono: side = 0 ⇒ L1_ms = L1_lr / sqrt(2) ⇒ mid-side.
        assert_eq!(choose_mid_side_stereo(&l, &l, lm, 0), Some(true));

        // Hard-panned (right silent in the decision span): at LM=1,
        // bins = 40 and E = 5, so sqrt(2)/45 > 1/40 ⇒ dual.
        assert_eq!(span, 40);
        let mut r = l.clone();
        for v in r[..span].iter_mut() {
            *v = 0.0;
        }
        assert_eq!(choose_mid_side_stereo(&l, &r, lm, 0), Some(false));

        // Manual L1 evaluation agrees with the verdict on a mixed
        // signal.
        let r2: Vec<f32> = (0..coded).map(|i| ((i * 3 + 7) % 9) as f32 - 4.0).collect();
        let mut l1_ms = 0.0f64;
        let mut l1_lr = 0.0f64;
        for i in 0..span {
            let (a, b) = (f64::from(l[i]), f64::from(r2[i]));
            l1_ms += ((a + b) / 2f64.sqrt()).abs() + ((a - b) / 2f64.sqrt()).abs();
            l1_lr += a.abs() + b.abs();
        }
        let expected = l1_ms / (span as f64 + 5.0) < l1_lr / span as f64;
        assert_eq!(choose_mid_side_stereo(&l, &r2, lm, 0), Some(expected));

        // Undefined cases.
        assert_eq!(choose_mid_side_stereo(&l, &r2, lm, 17), None);
        assert_eq!(choose_mid_side_stereo(&l, &r2, 4, 0), None);
        assert_eq!(choose_mid_side_stereo(&l[..span - 1], &r2, lm, 0), None);
        let zeros = vec![0.0f32; coded];
        assert_eq!(choose_mid_side_stereo(&zeros, &zeros, lm, 0), None);
    }

    /// The `E` term splits at LM=1/LM=2 per the §5.3.5 prose.
    #[test]
    fn mid_side_extra_dof_splits_at_lm_two() {
        assert_eq!(mid_side_extra_dof(0), 5);
        assert_eq!(mid_side_extra_dof(1), 5);
        assert_eq!(mid_side_extra_dof(2), 13);
        assert_eq!(mid_side_extra_dof(3), 13);
    }

    /// Table 66 rows, including the coarse-energy 80-bit subtraction,
    /// the frame-duration scaling, and the 68–84 gap reading of the
    /// RFC's "84-84" row.
    #[test]
    fn intensity_start_band_table_66() {
        // lm=2 → 10 ms → 100 frames/s: kbps = (bits - 80) / 10.
        let bits = |kbps: u32| 80 + kbps * 10;
        assert_eq!(intensity_start_band(bits(30), 2), Some(8));
        assert_eq!(intensity_start_band(bits(45), 2), Some(12));
        assert_eq!(intensity_start_band(bits(60), 2), Some(16));
        assert_eq!(intensity_start_band(bits(75), 2), Some(18));
        assert_eq!(intensity_start_band(bits(90), 2), Some(19));
        assert_eq!(intensity_start_band(bits(120), 2), Some(20));
        assert_eq!(intensity_start_band(bits(200), 2), None);
        // Exact boundaries: 35/50/68/84/102 land in the upper row,
        // 130 is the last enabled rate.
        assert_eq!(intensity_start_band(bits(35), 2), Some(12));
        assert_eq!(intensity_start_band(bits(50), 2), Some(16));
        assert_eq!(intensity_start_band(bits(130), 2), Some(20));
        // The 80-bit coarse subtraction: an 80-bit frame rates 0.
        assert_eq!(intensity_start_band(80, 2), Some(8));
        // Frame-duration scaling: the same byte size rates twice as
        // fast at half the frame duration.
        assert_eq!(intensity_start_band(bits(30), 2), Some(8));
        assert_eq!(intensity_start_band(bits(30), 1), Some(16)); // 60 kbps
        assert_eq!(intensity_start_band(400, 4), None);
    }

    /// `choose_intra_mode` equals a manual two-pass cost comparison
    /// (intra-flag bit + §4.3.2.1 coarse walk on scratch states) over
    /// a spread of states × targets × frame sizes.
    #[test]
    fn intra_mode_matches_manual_cost_comparison() {
        use crate::coarse_energy::{encode_coarse_energy, CoarseEnergyState, MAX_CHANNELS};
        use crate::range_encoder::RangeEncoder;

        let budget = 96u32 * 8;
        let mut checked_both = [false, false];
        for case in 0..6u32 {
            let lm = case % 4;
            let mut state = CoarseEnergyState::new();
            let mut target = [[0.0f32; NUM_BANDS]; MAX_CHANNELS];
            for (b, slot) in target[0].iter_mut().enumerate() {
                // A mix of stationary and adversarial cases.
                let sign = if b % 2 == 0 { 1.0 } else { -1.0 };
                state.energy[0][b] = sign * (case as f32);
                *slot = -sign * (case as f32) + 0.3;
            }
            let got = choose_intra_mode(&state, &target, lm, 0, NUM_BANDS, 1, budget).unwrap();

            let mut cost = [0u32; 2];
            for (idx, intra) in [(0usize, false), (1usize, true)] {
                let mut enc = RangeEncoder::new();
                enc.enc_bit_logp(u32::from(intra), 3).unwrap();
                let mut scratch = state;
                encode_coarse_energy(
                    &mut enc,
                    &mut scratch,
                    &target,
                    intra,
                    lm,
                    0,
                    NUM_BANDS,
                    1,
                    budget,
                )
                .unwrap();
                cost[idx] = enc.tell_frac();
            }
            assert_eq!(
                got,
                cost[1] < cost[0],
                "case {case}: decision != cost order"
            );
            checked_both[usize::from(got)] = true;
        }
        // The sweep must exercise both outcomes, or it proves nothing.
        assert!(
            checked_both[0] && checked_both[1],
            "sweep failed to produce both intra and inter decisions"
        );
    }

    /// A stationary stream prefers inter: after the state has locked
    /// onto the targets, re-coding the same envelope is near-free with
    /// inter prediction, while intra pays the full envelope (plus the
    /// 3-bit flag) again.
    #[test]
    fn stationary_stream_prefers_inter() {
        use crate::coarse_energy::{encode_coarse_energy, CoarseEnergyState, MAX_CHANNELS};
        use crate::range_encoder::RangeEncoder;

        let budget = 96u32 * 8;
        let mut target = [[0.0f32; NUM_BANDS]; MAX_CHANNELS];
        for (b, slot) in target[0].iter_mut().enumerate() {
            *slot = 5.0 - 0.2 * b as f32;
        }
        // Lock the state onto the targets with one intra pass.
        let mut state = CoarseEnergyState::new();
        let mut enc = RangeEncoder::new();
        encode_coarse_energy(
            &mut enc, &mut state, &target, true, 2, 0, NUM_BANDS, 1, budget,
        )
        .unwrap();
        // Same envelope again: inter must win.
        assert_eq!(
            choose_intra_mode(&state, &target, 2, 0, NUM_BANDS, 1, budget),
            Some(false)
        );
    }

    /// Bad parameters propagate as `None`.
    #[test]
    fn intra_mode_rejects_bad_params() {
        use crate::coarse_energy::{CoarseEnergyState, MAX_CHANNELS};
        let state = CoarseEnergyState::new();
        let target = [[0.0f32; NUM_BANDS]; MAX_CHANNELS];
        assert!(choose_intra_mode(&state, &target, 4, 0, NUM_BANDS, 1, 768).is_none());
        assert!(choose_intra_mode(&state, &target, 0, 5, 3, 1, 768).is_none());
        assert!(choose_intra_mode(&state, &target, 0, 0, NUM_BANDS, 0, 768).is_none());
    }

    /// Input validation: bad lm / window / length are rejected; tiny
    /// windows have no interior bands.
    #[test]
    fn validation_and_tiny_windows() {
        let e = vec![0.0f32; NUM_BANDS];
        assert!(choose_band_boosts(&e, 4, 0, NUM_BANDS).is_none());
        assert!(choose_band_boosts(&e, 0, 5, 3).is_none());
        assert!(choose_band_boosts(&e, 0, 0, NUM_BANDS + 1).is_none());
        assert!(choose_band_boosts(&e[..5], 0, 0, NUM_BANDS).is_none());
        // Windows of width 0/1/2 have no interior band: all zero.
        assert_eq!(choose_band_boosts(&[], 1, 3, 3).unwrap(), Vec::<i32>::new());
        assert_eq!(choose_band_boosts(&[9.0], 1, 3, 4).unwrap(), vec![0]);
        assert_eq!(
            choose_band_boosts(&[9.0, 9.0], 1, 3, 5).unwrap(),
            vec![0, 0]
        );
    }
}
