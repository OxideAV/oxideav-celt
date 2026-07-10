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

/// Geometry of the §4.3.1 transient short-block layout inside one
/// hop-`N` frame: `nb_blocks` short MDCTs of `sb = n / nb_blocks` bins
/// each, placed at stride `sb` starting `pad = (n - sb) / 2` samples
/// into the frame.
///
/// Returns `(sb, pad)` on success, or `None` when the geometry is
/// invalid: `nb_blocks == 0`, `nb_blocks` does not divide `n`, or
/// `n - sb` is odd (the placement offset must be integral).
///
/// ## Why this placement (derivation, not a free choice)
///
/// RFC 6716 §4.3.1 states a transient frame's coefficients "represent
/// multiple short MDCTs in the frame" and §4.3.7 requires the
/// synthesis window to satisfy power complementarity \[PRINCEN86\],
/// but the RFC narrative does not spell out where the short blocks
/// sit within the frame. The placement is nevertheless **forced** by
/// requiring \[PRINCEN86\] time-domain aliasing cancellation to keep
/// holding across every long↔short frame transition in a stream:
///
/// * The long low-overlap window ([`build_low_overlap_window_f32`])
///   rises over frame positions `[pad_l, pad_l + overlap)` with
///   `pad_l = (n - overlap) / 2`, and its carried tail falls over the
///   same positions of the **next** frame (its aliasing folds about
///   the centre of that falling region).
/// * A short block's rising half folds about the centre of its first
///   `sb` samples. Cancellation against a preceding long frame's tail
///   therefore requires the first short block's rising half to occupy
///   exactly `[pad_l, pad_l + overlap)` — i.e. `sb = overlap` and the
///   block sequence starting at `pad = (n - sb) / 2`.
/// * Consecutive short blocks at stride `sb` with the full-overlap
///   basic window cancel pairwise (equal-size TDAC), and the last
///   block's falling half then lands on `[pad, pad + sb)` of the next
///   frame — the same positions a following long (or short) frame's
///   rising region occupies. Every transition is covered.
///
/// With the canonical CELT geometry (`n = 120 << lm`,
/// `nb_blocks = 1 << lm`) the short block size is always
/// `sb = 120 = ` [`BASIC_WINDOW_HALF`], so the short window is the
/// full-overlap 240-sample basic window and `pad = (n - 120) / 2`
/// matches the long window's zero padding.
#[inline]
pub fn short_block_geometry(n: usize, nb_blocks: usize) -> Option<(usize, usize)> {
    if nb_blocks == 0 || n == 0 || n % nb_blocks != 0 {
        return None;
    }
    let sb = n / nb_blocks;
    if (n - sb) % 2 != 0 {
        return None;
    }
    Some((sb, (n - sb) / 2))
}

impl MdctSynthesis {
    /// Synthesize one **transient** frame from `nb_blocks` interleaved
    /// short MDCTs (RFC 6716 §4.3.1 / §4.3.7): de-interleave the
    /// `N`-bin spectrum into `nb_blocks` short spectra of
    /// `sb = N / nb_blocks` bins (`spectrum[nb_blocks * f + s]` is bin
    /// `f` of short block `s` — the same per-block interleave the
    /// §4.3.4.3 spreading and §4.3.4.5 Hadamard stages use), run the
    /// §4.3.7 inverse MDCT per block, window each `2 * sb` output with
    /// `short_window`, and overlap-add the blocks at stride `sb`
    /// starting `pad = (N - sb) / 2` samples into the frame (the
    /// placement [`short_block_geometry`] derives). Emits the `N`
    /// finished output samples and carries the assembly's tail into
    /// the shared cross-frame overlap state, so long and short frames
    /// interleave freely in one stream.
    ///
    /// `short_window` must be the `2 * sb`-sample synthesis window for
    /// the short blocks (the full-overlap window
    /// `build_low_overlap_window_f32(sb, sb)` in the canonical
    /// `sb = 120` geometry).
    ///
    /// Returns `false` (state and `out` untouched) when the block
    /// geometry is invalid ([`short_block_geometry`]),
    /// `spectrum.len() != N`, `short_window.len() != 2 * sb`, or
    /// `out.len() != N`.
    pub fn frame_short(
        &mut self,
        spectrum: &[f32],
        short_window: &[f32],
        nb_blocks: usize,
        out: &mut [f32],
    ) -> bool {
        let n = self.tail.len();
        let Some((sb, pad)) = short_block_geometry(n, nb_blocks) else {
            return false;
        };
        if spectrum.len() != n || short_window.len() != 2 * sb || out.len() != n {
            return false;
        }
        // Assembly buffer: the last short block ends at
        // pad + (nb_blocks - 1) * sb + 2 * sb = n + pad + sb.
        let ext = n + pad + sb;
        let mut acc = vec![0.0f32; ext];
        acc[..n].copy_from_slice(&self.tail);

        let mut spec_s = vec![0.0f32; sb];
        let mut block = vec![0.0f32; 2 * sb];
        for s in 0..nb_blocks {
            for (f, slot) in spec_s.iter_mut().enumerate() {
                *slot = spectrum[nb_blocks * f + s];
            }
            // Lengths are consistent by construction; cannot fail.
            if !imdct_naive_f32(&spec_s, &mut block) {
                return false;
            }
            let base = pad + s * sb;
            for (i, (&b, &w)) in block.iter().zip(short_window.iter()).enumerate() {
                acc[base + i] += b * w;
            }
        }

        out.copy_from_slice(&acc[..n]);
        // New tail: the assembly's overhang past the emitted frame
        // (extent pad + sb — the same nonzero extent the long window's
        // second half has), zero beyond.
        self.tail[..pad + sb].copy_from_slice(&acc[n..]);
        self.tail[pad + sb..].iter_mut().for_each(|s| *s = 0.0);
        true
    }
}

/// Streaming windowed MDCT **analysis** state — the exact mirror of
/// [`MdctSynthesis`] (RFC 6716 §4.3.7, encoder direction per §5.3:
/// "the filters and rotations in the encoder are simply the inverse of
/// the operation performed by the decoder").
///
/// Each call to [`MdctAnalysis::frame`] consumes `N` new input
/// samples, forms the `2*N` transform block `[history | input]` (the
/// previous call's `N` samples followed by the new ones — the sliding
/// hop-`N` block layout the synthesis side overlap-adds at), applies
/// the analysis window (the same power-complementary §4.3.7 window the
/// synthesis side uses, per [PRINCEN86] weighted overlap-add), and
/// runs the forward MDCT ([`mdct_naive_f32`]), emitting `N` spectral
/// bins. The first call folds against a zero history (stream-open
/// condition, the mirror of [`MdctSynthesis`]'s zero tail).
///
/// ## Latency contract
///
/// Feeding each emitted spectrum straight into
/// [`MdctSynthesis::frame`] with the same window reconstructs the
/// input exactly with **one frame (`N` samples) of algorithmic
/// delay**: analysis call `t` transforms the block covering input
/// frames `t-1` and `t`, and synthesis call `t` finishes exactly that
/// block's first half — input frame `t-1`. §4.3.7 specifies only the
/// decoder side; the one-frame-history buffering is the alignment
/// under which the streaming pair realizes the [PRINCEN86]
/// time-domain aliasing cancellation with no extra buffering on
/// either side (pinned by the round-trip tests).
#[derive(Debug, Clone)]
pub struct MdctAnalysis {
    /// Saved previous input frame (`N` samples) — the first half of
    /// the next transform block.
    history: Vec<f32>,
}

impl MdctAnalysis {
    /// Create an analysis state for frame size (MDCT hop) `n`, with a
    /// zero history.
    pub fn new(n: usize) -> Self {
        Self {
            history: vec![0.0; n],
        }
    }

    /// Frame size `N` this state was created for.
    #[inline]
    pub fn frame_size(&self) -> usize {
        self.history.len()
    }

    /// Zero the carried history (encoder reset, the mirror of the
    /// §4.5.2 decoder reset).
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|s| *s = 0.0);
    }

    /// Analyze one frame: form the `2*N` block `[history | input]`,
    /// multiply by the `2*N`-sample analysis `window`, forward-MDCT it,
    /// and write the `N` spectral bins into `out`. `input` becomes the
    /// new history.
    ///
    /// Returns `false` (state and `out` untouched) when
    /// `input.len() != N`, `window.len() != 2 * N`, or
    /// `out.len() != N`.
    pub fn frame(&mut self, input: &[f32], window: &[f32], out: &mut [f32]) -> bool {
        let n = self.history.len();
        if n == 0 || input.len() != n || window.len() != 2 * n || out.len() != n {
            return false;
        }
        let mut block = vec![0.0f32; 2 * n];
        for ((b, &x), &w) in block[..n]
            .iter_mut()
            .zip(self.history.iter())
            .zip(window[..n].iter())
        {
            *b = x * w;
        }
        for ((b, &x), &w) in block[n..]
            .iter_mut()
            .zip(input.iter())
            .zip(window[n..].iter())
        {
            *b = x * w;
        }
        if !mdct_naive_f32(&block, out) {
            return false;
        }
        self.history.copy_from_slice(input);
        true
    }

    /// Analyze one **transient** frame into `nb_blocks` interleaved
    /// short MDCTs — the exact mirror of
    /// [`MdctSynthesis::frame_short`]. Forms the `2 * N` sliding block
    /// `[history | input]`, cuts the `nb_blocks` short segments of
    /// `2 * sb` samples at stride `sb` starting at
    /// `pad = (N - sb) / 2` (the [`short_block_geometry`] placement),
    /// windows each with `short_window`, forward-MDCTs it
    /// ([`mdct_naive_f32`]), and interleaves the per-block bins into
    /// `out` as `out[nb_blocks * f + s]` = bin `f` of block `s`.
    /// `input` becomes the new history.
    ///
    /// Feeding each emitted spectrum into
    /// [`MdctSynthesis::frame_short`] with the same window
    /// reconstructs the input with one frame of delay, exactly like
    /// the long-block pair — including across mixed long/short frame
    /// sequences (the placement derivation guarantees the transition
    /// cancellation; see [`short_block_geometry`]).
    ///
    /// Returns `false` (state and `out` untouched) when the geometry
    /// is invalid, `input.len() != N`, `short_window.len() != 2 * sb`,
    /// or `out.len() != N`.
    pub fn frame_short(
        &mut self,
        input: &[f32],
        short_window: &[f32],
        nb_blocks: usize,
        out: &mut [f32],
    ) -> bool {
        let n = self.history.len();
        let Some((sb, pad)) = short_block_geometry(n, nb_blocks) else {
            return false;
        };
        if input.len() != n || short_window.len() != 2 * sb || out.len() != n {
            return false;
        }
        let mut block2n = vec![0.0f32; 2 * n];
        block2n[..n].copy_from_slice(&self.history);
        block2n[n..].copy_from_slice(input);

        let mut seg = vec![0.0f32; 2 * sb];
        let mut bins = vec![0.0f32; sb];
        for s in 0..nb_blocks {
            let base = pad + s * sb;
            for (i, (slot, &w)) in seg.iter_mut().zip(short_window.iter()).enumerate() {
                *slot = block2n[base + i] * w;
            }
            if !mdct_naive_f32(&seg, &mut bins) {
                return false;
            }
            for (f, &x) in bins.iter().enumerate() {
                out[nb_blocks * f + s] = x;
            }
        }
        self.history.copy_from_slice(input);
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

    // ---- MdctAnalysis (encoder direction) ----

    /// The streaming analysis matches the offline windowed forward
    /// MDCT of the sliding `[history | input]` block, frame by frame.
    #[test]
    fn streaming_analysis_matches_offline_blocks() {
        let n = 120usize;
        let w = build_low_overlap_window_f32(n, n).expect("valid");
        let x = test_signal(n * 4, 0x5EEDED);
        let mut ana = MdctAnalysis::new(n);
        assert_eq!(ana.frame_size(), n);
        let mut spec = vec![0.0f32; n];
        // Zero-history first frame, then sliding blocks.
        let mut padded = vec![0.0f32; n];
        padded.extend_from_slice(&x);
        for t in 0..4 {
            assert!(ana.frame(&x[t * n..(t + 1) * n], &w, &mut spec));
            let windowed: Vec<f32> = (0..2 * n).map(|i| padded[t * n + i] * w[i]).collect();
            let mut expected = vec![0.0f32; n];
            assert!(mdct_naive_f32(&windowed, &mut expected));
            for k in 0..n {
                assert!(
                    (spec[k] - expected[k]).abs() <= 1e-5,
                    "frame {t} bin {k}: streaming {} vs offline {}",
                    spec[k],
                    expected[k]
                );
            }
        }
    }

    /// The streaming analysis → synthesis pair is the identity with
    /// exactly one frame (`N` samples) of delay — the [PRINCEN86]
    /// time-domain aliasing cancellation through the §4.3.7 window
    /// pair, full-overlap geometry.
    #[test]
    fn analysis_synthesis_round_trip_one_frame_delay_full_overlap() {
        let n = BASIC_WINDOW_HALF;
        let w = build_low_overlap_window_f32(n, n).expect("valid");
        let x = test_signal(n * 6, 0xFEED01);
        let mut ana = MdctAnalysis::new(n);
        let mut synth = MdctSynthesis::new(n);
        let mut spec = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let mut recovered = Vec::new();
        for t in 0..6 {
            assert!(ana.frame(&x[t * n..(t + 1) * n], &w, &mut spec));
            assert!(synth.frame(&spec, &w, &mut out));
            recovered.extend_from_slice(&out);
        }
        // Output frame t reconstructs input frame t-1: compare
        // recovered[N..] against x[..5N].
        for i in 0..5 * n {
            assert!(
                (recovered[n + i] - x[i]).abs() <= 1e-4,
                "one-frame-delay PR broken at {i}: {} vs {}",
                recovered[n + i],
                x[i]
            );
        }
    }

    /// Same round-trip through the **low-overlap** window geometry
    /// (N = 240, overlap 120 — one step of the §4.3.7 zero-pad + ones
    /// construction).
    #[test]
    fn analysis_synthesis_round_trip_low_overlap() {
        let n = 240usize;
        let w = build_low_overlap_window_f32(n, 120).expect("valid");
        let x = test_signal(n * 5, 0xA11CE5);
        let mut ana = MdctAnalysis::new(n);
        let mut synth = MdctSynthesis::new(n);
        let mut spec = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        let mut recovered = Vec::new();
        for t in 0..5 {
            assert!(ana.frame(&x[t * n..(t + 1) * n], &w, &mut spec));
            assert!(synth.frame(&spec, &w, &mut out));
            recovered.extend_from_slice(&out);
        }
        for i in 0..4 * n {
            assert!(
                (recovered[n + i] - x[i]).abs() <= 1e-4,
                "low-overlap analysis/synthesis PR broken at {i}"
            );
        }
    }

    /// Analysis of a stream synthesized from known spectra recovers
    /// those spectra with one frame of delay (the synthesis map is
    /// injective on the window support, so the time-domain PR pins the
    /// spectral round-trip too).
    #[test]
    fn analysis_recovers_synthesized_spectra() {
        let n = 120usize;
        let w = build_low_overlap_window_f32(n, n).expect("valid");
        let spectra: Vec<Vec<f32>> = (0..5)
            .map(|t| test_signal(n, 0xD1CE00 + t as u32))
            .collect();
        // Synthesize the stream.
        let mut synth = MdctSynthesis::new(n);
        let mut pcm = Vec::new();
        let mut out = vec![0.0f32; n];
        for spec in &spectra {
            assert!(synth.frame(spec, &w, &mut out));
            pcm.extend_from_slice(&out);
        }
        // Analyze it back: analysis frame t sees synthesis block t-1.
        let mut ana = MdctAnalysis::new(n);
        let mut recovered = vec![0.0f32; n];
        for t in 0..5 {
            assert!(ana.frame(&pcm[t * n..(t + 1) * n], &w, &mut recovered));
            if t >= 1 {
                // Steady state needs both halves of block t-1 present:
                // block t-1's first half overlaps block t-2's tail, so
                // from t = 2 the recovery is exact; t = 1 sees block 0
                // whose first half folded against a zero tail.
                if t >= 2 {
                    for k in 0..n {
                        assert!(
                            (recovered[k] - spectra[t - 1][k]).abs() <= 1e-4,
                            "spectral round-trip broken at frame {} bin {k}",
                            t - 1
                        );
                    }
                }
            }
        }
    }

    /// Bad shapes are rejected without touching the state; `reset()`
    /// restores the fresh zero-history behaviour.
    #[test]
    fn streaming_analysis_rejects_bad_shapes_and_resets() {
        let n = 16usize;
        let w = build_low_overlap_window_f32(n, n).expect("valid");
        let mut ana = MdctAnalysis::new(n);
        let x = test_signal(n, 0xBEEF);
        let mut spec = vec![0.0f32; n];
        assert!(!ana.frame(&x[..n - 1], &w, &mut spec));
        assert!(!ana.frame(&x, &w[..2 * n - 1], &mut spec));
        assert!(!ana.frame(&x, &w, &mut spec[..n - 1]));
        // First good frame from fresh state.
        assert!(ana.frame(&x, &w, &mut spec));
        let first = spec.clone();
        // Second frame differs (history is now x).
        assert!(ana.frame(&x, &w, &mut spec));
        // reset() reproduces the fresh-state output.
        ana.reset();
        assert!(ana.frame(&x, &w, &mut spec));
        for k in 0..n {
            assert!((spec[k] - first[k]).abs() <= 1e-7);
        }
    }

    // --- Transient short-block pair (§4.3.1 / §4.3.7) ---

    #[test]
    fn short_block_geometry_canonical_and_rejections() {
        // Canonical CELT geometry: sb is always 120, pad = (n - 120)/2.
        for lm in 0..=3u32 {
            let n = 120usize << lm;
            let b = 1usize << lm;
            assert_eq!(short_block_geometry(n, b), Some((120, (n - 120) / 2)));
        }
        // B = 1 degenerates to sb = n, pad = 0 (the full-overlap case).
        assert_eq!(short_block_geometry(120, 1), Some((120, 0)));
        // Rejections: zero blocks, non-dividing blocks, odd offset.
        assert_eq!(short_block_geometry(240, 0), None);
        assert_eq!(short_block_geometry(240, 7), None);
        assert_eq!(short_block_geometry(0, 1), None);
        // n=6, b=2 → sb=3, n-sb=3 odd → rejected.
        assert_eq!(short_block_geometry(6, 2), None);
    }

    /// The short analysis/synthesis pair reconstructs an all-short
    /// stream with one frame of delay, exactly like the long pair.
    #[test]
    fn short_pair_round_trip_one_frame_delay() {
        let n = 240usize; // lm = 1 geometry
        let b = 2usize;
        let sb = n / b;
        let w = build_low_overlap_window_f32(sb, sb).expect("valid");
        let frames = 5usize;
        let x = test_signal(n * frames, 0x51DE5);

        let mut ana = MdctAnalysis::new(n);
        let mut synth = MdctSynthesis::new(n);
        let mut spec = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        for t in 0..frames {
            assert!(ana.frame_short(&x[t * n..(t + 1) * n], &w, b, &mut spec));
            assert!(synth.frame_short(&spec, &w, b, &mut out));
            if t >= 1 {
                // Synthesis call t finishes input frame t-1.
                for k in 0..n {
                    assert!(
                        (out[k] - x[(t - 1) * n + k]).abs() <= 1e-4,
                        "short PR broken at frame {} sample {k}: {} vs {}",
                        t - 1,
                        out[k],
                        x[(t - 1) * n + k]
                    );
                }
            }
        }
    }

    /// Mixed long/short streams reconstruct across every transition
    /// (long→short, short→short, short→long): the short-block placement
    /// keeps the \[PRINCEN86\] cancellation with the long low-overlap
    /// window at the frame boundaries.
    #[test]
    fn mixed_long_short_stream_round_trip() {
        let n = 480usize; // lm = 2 geometry
        let b = 4usize;
        let sb = n / b;
        assert_eq!(sb, 120);
        let w_long = build_low_overlap_window_f32(n, sb).expect("valid");
        let w_short = build_low_overlap_window_f32(sb, sb).expect("valid");
        // Frame kinds: long, short, short, long, short, long.
        let kinds = [false, true, true, false, true, false];
        let frames = kinds.len();
        let x = test_signal(n * frames, 0xA11CE);

        let mut ana = MdctAnalysis::new(n);
        let mut synth = MdctSynthesis::new(n);
        let mut spec = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        for (t, &transient) in kinds.iter().enumerate() {
            let input = &x[t * n..(t + 1) * n];
            if transient {
                assert!(ana.frame_short(input, &w_short, b, &mut spec));
                assert!(synth.frame_short(&spec, &w_short, b, &mut out));
            } else {
                assert!(ana.frame(input, &w_long, &mut spec));
                assert!(synth.frame(&spec, &w_long, &mut out));
            }
            if t >= 1 {
                for k in 0..n {
                    assert!(
                        (out[k] - x[(t - 1) * n + k]).abs() <= 1e-4,
                        "mixed PR broken at frame {} (kind {}) sample {k}: {} vs {}",
                        t - 1,
                        kinds[t - 1],
                        out[k],
                        x[(t - 1) * n + k]
                    );
                }
            }
        }
    }

    /// `nb_blocks = 1` short synthesis with the full-overlap window of
    /// size N equals the long path with the same window (the degenerate
    /// lm = 0 transient case).
    #[test]
    fn single_block_short_equals_long_full_overlap() {
        let n = 120usize;
        let w = build_low_overlap_window_f32(n, n).expect("valid");
        let spec = test_signal(n, 0xF00D);
        let mut s1 = MdctSynthesis::new(n);
        let mut s2 = MdctSynthesis::new(n);
        let mut o1 = vec![0.0f32; n];
        let mut o2 = vec![0.0f32; n];
        for _ in 0..3 {
            assert!(s1.frame(&spec, &w, &mut o1));
            assert!(s2.frame_short(&spec, &w, 1, &mut o2));
            for k in 0..n {
                assert!((o1[k] - o2[k]).abs() <= 1e-6);
            }
        }
    }

    /// Bad shapes are rejected without touching the state on both
    /// sides of the short pair.
    #[test]
    fn short_pair_rejects_bad_shapes() {
        let n = 240usize;
        let b = 2usize;
        let sb = n / b;
        let w = build_low_overlap_window_f32(sb, sb).expect("valid");
        let x = test_signal(n, 0xDEAD1);
        let mut spec = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];

        let mut ana = MdctAnalysis::new(n);
        assert!(!ana.frame_short(&x[..n - 1], &w, b, &mut spec));
        assert!(!ana.frame_short(&x, &w[..2 * sb - 1], b, &mut spec));
        assert!(!ana.frame_short(&x, &w, 0, &mut spec));
        assert!(!ana.frame_short(&x, &w, 7, &mut spec));
        assert!(!ana.frame_short(&x, &w, b, &mut spec[..n - 1]));
        // A good call still works from the untouched state and matches
        // a fresh state's first call.
        assert!(ana.frame_short(&x, &w, b, &mut spec));
        let mut fresh = MdctAnalysis::new(n);
        let mut spec2 = vec![0.0f32; n];
        assert!(fresh.frame_short(&x, &w, b, &mut spec2));
        for k in 0..n {
            assert!((spec[k] - spec2[k]).abs() <= 1e-7);
        }

        let mut synth = MdctSynthesis::new(n);
        assert!(!synth.frame_short(&spec[..n - 1], &w, b, &mut out));
        assert!(!synth.frame_short(&spec, &w[..2 * sb - 1], b, &mut out));
        assert!(!synth.frame_short(&spec, &w, 0, &mut out));
        assert!(!synth.frame_short(&spec, &w, 7, &mut out));
        assert!(!synth.frame_short(&spec, &w, b, &mut out[..n - 1]));
        assert!(synth.frame_short(&spec, &w, b, &mut out));
    }

    /// The short-block interleave convention: a spectrum that is
    /// nonzero only at the positions of one block (`nb_blocks*f + s`)
    /// synthesizes energy concentrated in that block's time span.
    #[test]
    fn short_block_interleave_isolates_blocks() {
        let n = 480usize;
        let b = 4usize;
        let sb = n / b;
        let pad = (n - sb) / 2;
        let w = build_low_overlap_window_f32(sb, sb).expect("valid");

        for s in 0..b {
            let mut spec = vec![0.0f32; n];
            // Put a mid-band tone into block s only.
            spec[b * (sb / 3) + s] = 1.0;
            let mut synth = MdctSynthesis::new(n);
            let mut out = vec![0.0f32; n];
            assert!(synth.frame_short(&spec, &w, b, &mut out));
            // Energy inside the block's emitted span [pad + s*sb, ...)
            // clipped to the frame, vs total frame energy: everything
            // outside the block's own 2*sb extent must be zero. Late
            // blocks overhang into the carried tail, so clip both ends
            // to the emitted frame.
            let lo = (pad + s * sb).min(n);
            let hi = (pad + s * sb + 2 * sb).min(n);
            let inside: f64 = out[lo..hi].iter().map(|&v| (v as f64) * (v as f64)).sum();
            let total: f64 = out.iter().map(|&v| (v as f64) * (v as f64)).sum();
            if lo < n {
                assert!(total > 0.0, "block {s} produced no in-frame energy");
            }
            assert!(
                (total - inside).abs() <= total.max(1e-12) * 1e-9,
                "block {s} leaked outside its time span"
            );
        }
    }
}
