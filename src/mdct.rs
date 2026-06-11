//! Inverse MDCT and low-overlap window (RFC 6716 §4.3.7).
//!
//! Per §4.3.7, the inverse MDCT "has no special characteristics": the
//! input is `N` frequency-domain samples and the output is `2*N`
//! time-domain samples, while scaling by `1/2`. A "low-overlap"
//! window reduces the algorithmic delay; it is derived from a basic
//! (full-overlap) 240-sample version of the window used by the Vorbis
//! codec:
//!
//! ```text
//!                                          2
//!           /   /pi      /pi   n + 1/2\ \ \
//!    W(n) = |sin|-- * sin|-- * -------| | |
//!           \   \2       \2       L   / / /
//! ```
//!
//! i.e. `W(n) = sin((pi/2) * sin((pi/2) * (n + 1/2) / L)^2)` — the
//! square applies to the inner sine (the Vorbis power-of-sine
//! convention). This reading is pinned numerically: the staged
//! per-mode tables `docs/audio/opus/tables/window120.csv` /
//! `window240.csv` match it to full printed precision at every sampled
//! position, while the alternative reading (squaring the outer sine)
//! does not match any entry.
//!
//! The low-overlap window is created by zero-padding the basic window
//! and inserting ones in the middle, such that the resulting window
//! still satisfies power complementarity (§4.3.7 cites [PRINCEN86]).
//! With frame size `N` (the MDCT hop) and overlap `L`, the `2*N`-sample
//! synthesis window is
//!
//! ```text
//! [ 0 × (N-L)/2 | rise(0..L) | 1 × (N-L) | fall(0..L) | 0 × (N-L)/2 ]
//! ```
//!
//! where `rise(n) = W(n)` and `fall(n) = W(L-1-n)`. At hop distance
//! `N` this satisfies `w(n)^2 + w(n+N)^2 = 1` exactly (the rise/fall
//! halves complement by the identity `sin^2 + cos^2 = 1`; the ones
//! region lands on the zero padding). For `N = L` the construction
//! degenerates to the basic full-overlap window (240 samples at
//! `L = 120`).
//!
//! ## Transform convention
//!
//! §4.3.7 defines only the decoder side ("output is 2*N ... while
//! scaling by 1/2") and leaves the analysis normalization to the
//! encoder. This module fixes the self-consistent pair
//!
//! ```text
//! C(n, k)   = cos((pi/N) * (n + 1/2 + N/2) * (k + 1/2))
//! MDCT:  X(k) = (4/N) * sum_{n=0}^{2N-1} x(n) * C(n, k)
//! IMDCT: y(n) = (1/2) * sum_{k=0}^{N-1}  X(k) * C(n, k)
//! ```
//!
//! so that the inverse carries the literal §4.3.7 `1/2` scaling and
//! the windowed overlap-add round trip (analysis window → MDCT →
//! IMDCT → synthesis window → overlap-add at hop `N`) reconstructs
//! the input exactly in the steady state — the time-domain aliasing
//! cancels between adjacent frames per [PRINCEN86] given the power
//! complementarity above. Both directions run their accumulators in
//! `f64` and are direct-form `O(N^2)` (correctness-first; a fast
//! factorization can replace the inner loop without touching the API).
//!
//! ## Provenance
//!
//! The window formula, the 240-sample basic length, the low-overlap
//! construction recipe, the `2*N` output length, and the `1/2` inverse
//! scaling are all stated in the RFC 6716 §4.3.7 narrative. The MDCT
//! itself is a public textbook transform; no delegated source file was
//! consulted. The staged tables `window120.csv` / `window240.csv`
//! (data-only extractions) validate the formula reading.

/// Length of the basic (full-overlap) window §4.3.7 derives the
/// low-overlap window from: 240 samples.
pub const BASIC_WINDOW_LEN: usize = 240;

/// Rising-half length of the basic 240-sample window (= the overlap
/// length of the canonical 48 kHz CELT modes; the staged
/// `window120.csv` table holds exactly this half).
pub const BASIC_WINDOW_HALF: usize = BASIC_WINDOW_LEN / 2;

/// Single coefficient of the §4.3.7 window rising half:
/// `W(n) = sin((pi/2) * sin((pi/2) * (n + 1/2) / overlap)^2)`.
///
/// `n` must be below `overlap`; out-of-range `n` (or `overlap == 0`)
/// returns `0.0` defensively. `W` is strictly increasing on
/// `0..overlap` with `W(overlap - 1)` just below `1.0`.
#[inline]
pub fn celt_window_f32(n: usize, overlap: usize) -> f32 {
    if overlap == 0 || n >= overlap {
        return 0.0;
    }
    let inner = (std::f64::consts::FRAC_PI_2 * (n as f64 + 0.5) / overlap as f64).sin();
    (std::f64::consts::FRAC_PI_2 * inner * inner).sin() as f32
}

/// The full rising half `W(0..overlap)` of the §4.3.7 window — the
/// layout of the staged `window120.csv` / `window240.csv` tables.
pub fn build_window_half_f32(overlap: usize) -> Vec<f32> {
    (0..overlap).map(|n| celt_window_f32(n, overlap)).collect()
}

/// Build the `2*N`-sample low-overlap synthesis window for frame size
/// (MDCT hop) `n` and overlap `overlap` per the §4.3.7 construction:
/// zero-pad the basic window and insert ones in the middle so the
/// result still satisfies power complementarity at hop `N`.
///
/// Returns `None` when the geometry is invalid: `n == 0`,
/// `overlap == 0`, `overlap > n`, or `n - overlap` odd (the zero
/// padding must split evenly across both ends).
pub fn build_low_overlap_window_f32(n: usize, overlap: usize) -> Option<Vec<f32>> {
    if n == 0 || overlap == 0 || overlap > n || (n - overlap) % 2 != 0 {
        return None;
    }
    let pad = (n - overlap) / 2;
    let mut w = Vec::with_capacity(2 * n);
    w.resize(pad, 0.0);
    for i in 0..overlap {
        w.push(celt_window_f32(i, overlap));
    }
    w.resize(pad + overlap + (n - overlap), 1.0);
    for i in (0..overlap).rev() {
        w.push(celt_window_f32(i, overlap));
    }
    w.resize(2 * n, 0.0);
    Some(w)
}

/// Inverse MDCT (RFC 6716 §4.3.7), direct form.
///
/// `spectrum` holds `N` frequency-domain samples; `out` must hold
/// exactly `2*N` slots and receives the time-domain block
/// `y(n) = (1/2) * sum_k X(k) * cos((pi/N)(n + 1/2 + N/2)(k + 1/2))`
/// — the literal §4.3.7 "output is 2*N time-domain samples, while
/// scaling by 1/2". Accumulation runs in `f64`.
///
/// Returns `false` (leaving `out` untouched) when `spectrum` is empty
/// or `out.len() != 2 * spectrum.len()`.
pub fn imdct_naive_f32(spectrum: &[f32], out: &mut [f32]) -> bool {
    let n = spectrum.len();
    if n == 0 || out.len() != 2 * n {
        return false;
    }
    let nf = n as f64;
    for (t, slot) in out.iter_mut().enumerate() {
        let phase_n = t as f64 + 0.5 + nf / 2.0;
        let mut acc = 0.0f64;
        for (k, &x) in spectrum.iter().enumerate() {
            acc += x as f64 * (std::f64::consts::PI / nf * phase_n * (k as f64 + 0.5)).cos();
        }
        *slot = (0.5 * acc) as f32;
    }
    true
}

/// Forward MDCT companion to [`imdct_naive_f32`], direct form.
///
/// `time` holds a `2*N`-sample (windowed) block; `out` must hold
/// exactly `N` slots and receives
/// `X(k) = (4/N) * sum_n x(n) * cos((pi/N)(n + 1/2 + N/2)(k + 1/2))`.
/// The `4/N` analysis normalization is the unique choice making the
/// §4.3.7 `1/2`-scaled inverse a unit-gain weighted-overlap-add round
/// trip (see the module docs). §4.3.7 does not constrain the encoder
/// side; this companion exists for the future encoder and for
/// round-trip validation.
///
/// Returns `false` (leaving `out` untouched) when `out` is empty or
/// `time.len() != 2 * out.len()`.
pub fn mdct_naive_f32(time: &[f32], out: &mut [f32]) -> bool {
    let n = out.len();
    if n == 0 || time.len() != 2 * n {
        return false;
    }
    let nf = n as f64;
    for (k, slot) in out.iter_mut().enumerate() {
        let phase_k = k as f64 + 0.5;
        let mut acc = 0.0f64;
        for (t, &x) in time.iter().enumerate() {
            let phase_n = t as f64 + 0.5 + nf / 2.0;
            acc += x as f64 * (std::f64::consts::PI / nf * phase_n * phase_k).cos();
        }
        *slot = (4.0 / nf * acc) as f32;
    }
    true
}

/// Streaming weighted-overlap-add synthesis state (RFC 6716 §4.3.7 /
/// §4.3.7.1 "output of the inverse MDCT (after weighted overlap-add)").
///
/// Each call to [`MdctSynthesis::frame`] runs the §4.3.7 inverse MDCT,
/// applies the synthesis window, overlap-adds the first half against
/// the previous frame's saved tail, and emits `N` output samples; the
/// second half becomes the new tail. The first emitted frame folds
/// against a zero tail (stream-open condition).
#[derive(Debug, Clone)]
pub struct MdctSynthesis {
    /// Saved second half of the previous windowed IMDCT block
    /// (`N` samples).
    tail: Vec<f32>,
}

impl MdctSynthesis {
    /// Create a synthesis state for frame size (MDCT hop) `n`, with a
    /// zero tail.
    pub fn new(n: usize) -> Self {
        Self { tail: vec![0.0; n] }
    }

    /// Frame size `N` this state was created for.
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.tail.len()
    }

    /// Zero the carried tail (decoder reset per §4.5.2).
    pub fn reset(&mut self) {
        self.tail.iter_mut().for_each(|s| *s = 0.0);
    }

    /// Synthesize one frame: inverse MDCT of `spectrum` (`N` samples),
    /// multiply by the `2*N`-sample synthesis `window`, overlap-add at
    /// hop `N`, and write the `N` finished output samples into `out`.
    ///
    /// Returns `false` (state and `out` untouched) when
    /// `spectrum.len() != N`, `window.len() != 2 * N`, or
    /// `out.len() != N`.
    pub fn frame(&mut self, spectrum: &[f32], window: &[f32], out: &mut [f32]) -> bool {
        let n = self.tail.len();
        if n == 0 || spectrum.len() != n || window.len() != 2 * n || out.len() != n {
            return false;
        }
        let mut block = vec![0.0f32; 2 * n];
        if !imdct_naive_f32(spectrum, &mut block) {
            return false;
        }
        for (b, &w) in block.iter_mut().zip(window.iter()) {
            *b *= w;
        }
        for ((slot, &tail), &first_half) in out.iter_mut().zip(&self.tail).zip(&block[..n]) {
            *slot = tail + first_half;
        }
        self.tail.copy_from_slice(&block[n..]);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values sampled from the staged data-only extraction
    /// `docs/audio/opus/tables/window120.csv` (rows 1/30/60/90/120,
    /// i.e. `n = 0, 29, 59, 89, 119`).
    const WINDOW120_SAMPLES: [(usize, f32); 5] = [
        (0, 6.728_696_6e-5),
        (29, 0.220_976_82),
        // Staged CSV prints 0.69980010; the trailing zero exceeds f32
        // print precision, so the literal drops it (same value).
        (59, 0.699_800_1),
        (89, 0.971_963_34),
        (119, 1.000_000_0),
    ];

    /// Values sampled from `docs/audio/opus/tables/window240.csv`
    /// (rows 1/60/120/180/240, i.e. `n = 0, 59, 119, 179, 239`).
    const WINDOW240_SAMPLES: [(usize, f32); 5] = [
        (0, 1.682_192_2e-5),
        (59, 0.224_485_37),
        (119, 0.703_462_66),
        // Staged CSV prints 0.97281981; the nearest f32 is 0.9728198
        // (the 8th significant digit exceeds f32 print precision).
        (179, 0.972_819_8),
        (239, 1.000_000_0),
    ];

    #[test]
    fn window_formula_matches_staged_window120() {
        for &(n, expected) in &WINDOW120_SAMPLES {
            let got = celt_window_f32(n, BASIC_WINDOW_HALF);
            assert!(
                (got - expected).abs() <= 1e-7,
                "W({n}) = {got}, staged {expected}"
            );
        }
    }

    #[test]
    fn window_formula_matches_staged_window240() {
        for &(n, expected) in &WINDOW240_SAMPLES {
            let got = celt_window_f32(n, BASIC_WINDOW_LEN);
            assert!(
                (got - expected).abs() <= 1e-7,
                "W({n}) = {got}, staged {expected}"
            );
        }
    }

    #[test]
    fn window_half_is_monotone_in_unit_range() {
        let half = build_window_half_f32(BASIC_WINDOW_HALF);
        assert_eq!(half.len(), BASIC_WINDOW_HALF);
        for pair in half.windows(2) {
            assert!(pair[0] < pair[1], "window rising half must increase");
        }
        assert!(half[0] > 0.0);
        assert!(half[BASIC_WINDOW_HALF - 1] <= 1.0);
    }

    #[test]
    fn window_half_power_complementarity() {
        // W(n)^2 + W(L-1-n)^2 = 1 exactly: the reflected argument turns
        // the inner sin^2 into cos^2, and sin(pi/2 x)^2 + sin(pi/2 (1-x))^2
        // with x + (1-x) = 1 collapses via sin^2 + cos^2 = 1.
        let l = BASIC_WINDOW_HALF;
        for n in 0..l {
            let a = celt_window_f32(n, l) as f64;
            let b = celt_window_f32(l - 1 - n, l) as f64;
            assert!(
                (a * a + b * b - 1.0).abs() <= 1e-6,
                "power complementarity broken at n={n}"
            );
        }
    }

    #[test]
    fn low_overlap_window_structure() {
        // N = 480 (LM=2 at 48 kHz), overlap = 120: pad 180 zeros, rise
        // 120, ones 360, fall 120, pad 180 zeros.
        let w = build_low_overlap_window_f32(480, 120).expect("valid geometry");
        assert_eq!(w.len(), 960);
        assert!(w[..180].iter().all(|&s| s == 0.0));
        for i in 0..120 {
            assert_eq!(w[180 + i], celt_window_f32(i, 120));
        }
        assert!(w[300..660].iter().all(|&s| s == 1.0));
        for i in 0..120 {
            assert_eq!(w[660 + i], celt_window_f32(119 - i, 120));
        }
        assert!(w[780..].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn low_overlap_degenerates_to_basic_window_at_full_overlap() {
        // N = overlap = 120 reproduces the basic 240-sample window:
        // rising half then mirrored falling half, no padding, no ones.
        let w = build_low_overlap_window_f32(BASIC_WINDOW_HALF, BASIC_WINDOW_HALF)
            .expect("full overlap is valid");
        assert_eq!(w.len(), BASIC_WINDOW_LEN);
        let half = build_window_half_f32(BASIC_WINDOW_HALF);
        for i in 0..BASIC_WINDOW_HALF {
            assert_eq!(w[i], half[i]);
            assert_eq!(w[BASIC_WINDOW_LEN - 1 - i], half[i]);
        }
    }

    #[test]
    fn low_overlap_power_complementarity_at_hop_n() {
        for &(n, overlap) in &[(120usize, 120usize), (240, 120), (480, 120), (960, 120)] {
            let w = build_low_overlap_window_f32(n, overlap).expect("valid geometry");
            for i in 0..n {
                let a = w[i] as f64;
                let b = w[i + n] as f64;
                assert!(
                    (a * a + b * b - 1.0).abs() <= 1e-6,
                    "hop-N power complementarity broken at N={n} i={i}"
                );
            }
        }
    }

    #[test]
    fn low_overlap_rejects_invalid_geometry() {
        assert!(build_low_overlap_window_f32(0, 120).is_none());
        assert!(build_low_overlap_window_f32(120, 0).is_none());
        assert!(build_low_overlap_window_f32(100, 120).is_none()); // overlap > N
        assert!(build_low_overlap_window_f32(121, 120).is_none()); // odd padding
    }

    #[test]
    fn imdct_rejects_length_mismatch() {
        let spec = [1.0f32; 4];
        let mut out = [0.0f32; 7];
        assert!(!imdct_naive_f32(&spec, &mut out));
        assert!(out.iter().all(|&s| s == 0.0));
        let mut out2 = [0.0f32; 0];
        assert!(!imdct_naive_f32(&[], &mut out2));
    }

    #[test]
    fn mdct_rejects_length_mismatch() {
        let time = [1.0f32; 8];
        let mut out = [0.0f32; 3];
        assert!(!mdct_naive_f32(&time, &mut out));
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn imdct_zero_spectrum_is_zero() {
        let spec = [0.0f32; 16];
        let mut out = [1.0f32; 32];
        assert!(imdct_naive_f32(&spec, &mut out));
        assert!(out.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn imdct_of_basis_vector_is_half_scaled_cosine() {
        // X = e_k decodes to y(n) = (1/2) cos((pi/N)(n + 1/2 + N/2)(k + 1/2)).
        let n = 16usize;
        let k = 3usize;
        let mut spec = vec![0.0f32; n];
        spec[k] = 1.0;
        let mut out = vec![0.0f32; 2 * n];
        assert!(imdct_naive_f32(&spec, &mut out));
        for (t, &y) in out.iter().enumerate() {
            let expected = 0.5
                * (std::f64::consts::PI / n as f64
                    * (t as f64 + 0.5 + n as f64 / 2.0)
                    * (k as f64 + 0.5))
                    .cos();
            assert!(
                (y as f64 - expected).abs() <= 1e-6,
                "basis IMDCT mismatch at t={t}"
            );
        }
    }

    #[test]
    fn imdct_is_linear() {
        let n = 12usize;
        let mut rng = 0x9e3779b9u32;
        let mut next = move || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..n).map(|_| next()).collect();
        let b: Vec<f32> = (0..n).map(|_| next()).collect();
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let mut ya = vec![0.0f32; 2 * n];
        let mut yb = vec![0.0f32; 2 * n];
        let mut ys = vec![0.0f32; 2 * n];
        assert!(imdct_naive_f32(&a, &mut ya));
        assert!(imdct_naive_f32(&b, &mut yb));
        assert!(imdct_naive_f32(&sum, &mut ys));
        for i in 0..2 * n {
            assert!((ys[i] - (ya[i] + yb[i])).abs() <= 1e-5);
        }
    }

    /// Deterministic pseudo-random test signal in [-1, 1).
    fn test_signal(len: usize, mut seed: u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// Offline windowed-overlap-add round trip: analysis window →
    /// MDCT → IMDCT → synthesis window → overlap-add at hop N.
    fn wola_round_trip(x: &[f32], n: usize, window: &[f32]) -> Vec<f32> {
        let frames = x.len() / n - 1;
        let mut out = vec![0.0f32; x.len()];
        let mut spec = vec![0.0f32; n];
        let mut block = vec![0.0f32; 2 * n];
        for t in 0..frames {
            let windowed: Vec<f32> = (0..2 * n).map(|i| x[t * n + i] * window[i]).collect();
            assert!(mdct_naive_f32(&windowed, &mut spec));
            assert!(imdct_naive_f32(&spec, &mut block));
            for i in 0..2 * n {
                out[t * n + i] += block[i] * window[i];
            }
        }
        out
    }

    #[test]
    fn wola_perfect_reconstruction_full_overlap() {
        // Basic full-overlap window, N = 120: steady-state samples
        // (away from the first/last half-frames) reconstruct exactly.
        let n = BASIC_WINDOW_HALF;
        let w = build_low_overlap_window_f32(n, n).expect("valid");
        let x = test_signal(n * 6, 0xC0FFEE);
        let out = wola_round_trip(&x, n, &w);
        for i in n..n * 4 {
            assert!(
                (out[i] - x[i]).abs() <= 1e-4,
                "full-overlap PR broken at {i}: {} vs {}",
                out[i],
                x[i]
            );
        }
    }

    #[test]
    fn wola_perfect_reconstruction_low_overlap() {
        // Low-overlap window, N = 240 with overlap 120 (one step of
        // the §4.3.7 zero-pad + ones construction).
        let n = 240usize;
        let w = build_low_overlap_window_f32(n, 120).expect("valid");
        let x = test_signal(n * 5, 0xBADCAB);
        let out = wola_round_trip(&x, n, &w);
        for i in n..n * 3 {
            assert!(
                (out[i] - x[i]).abs() <= 1e-4,
                "low-overlap PR broken at {i}: {} vs {}",
                out[i],
                x[i]
            );
        }
    }

    #[test]
    fn streaming_synthesis_matches_offline_overlap_add() {
        let n = 120usize;
        let w = build_low_overlap_window_f32(n, n).expect("valid");
        // Three frames of pseudo-random spectra.
        let spectra: Vec<Vec<f32>> = (0..3)
            .map(|t| test_signal(n, 0xABCD17 + t as u32))
            .collect();
        // Offline: window each IMDCT block, overlap-add at hop N.
        let mut offline = vec![0.0f32; n * 4];
        let mut block = vec![0.0f32; 2 * n];
        for (t, spec) in spectra.iter().enumerate() {
            assert!(imdct_naive_f32(spec, &mut block));
            for i in 0..2 * n {
                offline[t * n + i] += block[i] * w[i];
            }
        }
        // Streaming: MdctSynthesis emits N samples per frame.
        let mut synth = MdctSynthesis::new(n);
        assert_eq!(synth.frame_size(), n);
        let mut streamed = Vec::new();
        let mut out = vec![0.0f32; n];
        for spec in &spectra {
            assert!(synth.frame(spec, &w, &mut out));
            streamed.extend_from_slice(&out);
        }
        for i in 0..3 * n {
            assert!(
                (streamed[i] - offline[i]).abs() <= 1e-6,
                "streaming/offline mismatch at {i}"
            );
        }
    }

    #[test]
    fn streaming_synthesis_rejects_bad_shapes_and_resets() {
        let n = 8usize;
        let w = build_low_overlap_window_f32(n, n).expect("valid");
        let mut synth = MdctSynthesis::new(n);
        let spec = vec![1.0f32; n];
        let mut out = vec![0.0f32; n];
        assert!(!synth.frame(&spec[..n - 1], &w, &mut out));
        assert!(!synth.frame(&spec, &w[..2 * n - 1], &mut out));
        assert!(!synth.frame(&spec, &w, &mut out[..n - 1]));
        // A good frame leaves a non-zero tail; reset() zeroes it so the
        // next frame folds against silence again (§4.5.2 decoder reset).
        assert!(synth.frame(&spec, &w, &mut out));
        let mut after_first = vec![0.0f32; n];
        assert!(synth.frame(&spec, &w, &mut after_first));
        synth.reset();
        let mut after_reset = vec![0.0f32; n];
        assert!(synth.frame(&spec, &w, &mut after_reset));
        for i in 0..n {
            assert!((after_reset[i] - out[i]).abs() <= 1e-7);
        }
    }
}
