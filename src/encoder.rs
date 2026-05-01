//! CELT encoder — 48 kHz, full-band (FB), 20 ms (LM=3, 960 samples) or
//! 10 ms (LM=2, 480 samples) frames.
//!
//! Default constructor [`CeltEncoder::new`] uses LM=3 (the historical
//! single-frame-size mode); call [`CeltEncoder::new_with_frame_samples`]
//! with `480` to opt into the LM=2 path. Both share the same per-band
//! quantisation, PVQ, TF analysis and silence/transient/post-filter
//! header machinery — only the MDCT length, the per-frame coded bin
//! count `N = 100 * M`, and the default packet budget differ.
//!
//! Scope (per RFC 6716 §4.3):
//!
//! * §4.3 frame header: silence (silence-fast-path on near-zero input),
//!   post_filter (test-hook only), `transient` (per-frame, see
//!   [`detect_transient`] below) and `intra` (true on the first frame,
//!   false thereafter — see "Inter-frame energy prediction").
//! * §4.3.2.1 coarse energy: [`crate::quant_bands::quant_coarse_energy`] —
//!   Laplace-encoded, inter/intra prediction coefficients as per libopus.
//! * §4.3.2.2 fine energy: [`crate::quant_bands::quant_fine_energy`].
//! * §4.3.2.3 fine-energy finalise: [`crate::quant_bands::quant_energy_finalise`].
//! * §4.3.3 bit allocation: [`crate::encoder_rate::clt_compute_allocation_enc`] —
//!   matches the decoder's alloc table lookup, trim, skip, intensity and
//!   dual_stereo symbols.
//! * §4.3.4 PVQ shape encoding:
//!   - Mono: [`crate::encoder_bands::encode_all_bands_mono`] — per-band
//!     PVQ search, exp_rotation, canonical enumeration. Selects long
//!     (`big_b = 1`) or short (`big_b = M`) partitioning from the
//!     `transient` flag, matching the decoder.
//!   - Stereo: [`crate::encoder_bands::encode_all_bands_stereo_dual`] —
//!     dual-stereo only. Each channel is coded as an independent mono band
//!     inside one packet. Intensity stereo is not implemented; the encoder
//!     pins `intensity = coded_bands` so the decoder never applies the IS
//!     merge.
//! * §4.3.5 anti-collapse: 1 bit reserved on transient LM>=2 frames; the
//!   encoder emits `anti_collapse_on = 0` (no encoder-side pulse injection).
//! * §4.3.6 denormalisation: implicit in the decoder; encoder normalises
//!   the forward-MDCT coefficients before PVQ.
//! * §4.3.7 forward MDCT: long block uses [`crate::mdct::forward_mdct`]
//!   (direct definition). Transient frames use
//!   [`crate::mdct::forward_mdct_short`] — 8 × 200→100 sub-MDCTs with a
//!   100-tap sin² sub-block window, interleaved at stride M into the
//!   800-bin coefficient buffer.
//! * §4.3.8 comb post-filter: not applied (the test-only `force_post_filter`
//!   knob just sets the header flag for decoder smoke-tests).
//!
//! NOT implemented (fall back to long-block / no-boost path):
//!
//! * **Time-frequency change flags** — all `tf_res[i]` = 0. The transient
//!   path picks the TF deltas + tf_select that yield zero post-lookup
//!   for every band (libopus' nonzero TF decisions remain a TODO).
//! * **Dynalloc band-energy boosts** — no per-band boost is emitted.
//! * **Intensity stereo (M/S with theta)** — not implemented. Stereo uses
//!   dual-stereo only (L and R as two independent mono bands). Intensity
//!   stereo would buy extra HF bits on low bit-rate stereo; at 64 kbit/s
//!   the dual path produces well-separated L/R output.
//!
//! Inter-frame energy prediction IS enabled (RFC 6716 §4.3.2.1): the first
//! frame of the session is forced `intra=true` (no prior state), all
//! subsequent frames use `intra=false` so the Laplace-coded residuals
//! predict from the previous frame's quantised band energies. This saves
//! bits on steady-state content at the cost of state-drift exposure if the
//! decoder loses a packet — callers that need resync boundaries should
//! reinstantiate the encoder.

use std::collections::VecDeque;

use oxideav_core::Encoder;
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, Frame, MediaType, Packet, Result, SampleFormat,
    TimeBase,
};

use crate::encoder_bands::{encode_all_bands_mono, encode_all_bands_stereo_dual};
use crate::encoder_rate::clt_compute_allocation_enc;
use crate::header::encode_header;
use crate::mdct::forward_mdct;
use crate::quant_bands::{quant_coarse_energy, quant_energy_finalise, quant_fine_energy};
use crate::range_encoder::{RangeEncoder, BITRES};
use crate::tables::{
    init_caps, lm_for_frame_samples, EBAND_5MS, E_MEANS, NB_EBANDS, SPREAD_ICDF, SPREAD_NORMAL,
    TRIM_ICDF,
};

/// Default CELT frame length (LM=3): 960 samples = 20 ms at 48 kHz. The
/// encoder also supports LM=2 (480 samples = 10 ms) via
/// [`CeltEncoder::new_with_frame_samples`]; that path picks its own
/// per-instance frame count and the public `FRAME_SAMPLES` constant
/// remains the historical default for callers using [`CeltEncoder::new`].
pub const FRAME_SAMPLES: usize = 960;
pub const SAMPLE_RATE: u32 = 48_000;
const OVERLAP: usize = 120;

/// Internal MDCT-coded length at the default LM=3: `EBAND_5MS[21] * M = 800`.
/// Bins 800..960 after forward MDCT are not transmitted; the decoder
/// reconstructs them as zero. This matches the decoder's
/// `n = m * EBAND_5MS[NB_EBANDS]` = 800 (the output IMDCT time length is
/// 2N = 1600, the remaining 320 samples of the "true" 1920-sample long
/// block are the zero-padded tail that the next frame's overlap supplies).
///
/// At LM=2 the equivalent is `100 * 4 = 400` (computed per-instance).
pub const CODED_N: usize = 800;

/// Short-block sub-MDCT coefficient count at the default LM=3:
/// `CODED_N / M = 800 / 8`. The LM=2 path uses `400 / 4 = 100` (same value
/// — the short-window length is `coded_n / M` regardless of LM since both
/// LM=2 and LM=3 yield 100-sample sub-blocks).
const SHORT_N: usize = 100;

/// Fixed target bitrate for mono at 20 ms: 160 bytes/frame ≈ 64 kbit/s.
/// At 10 ms the per-frame overhead (header + coarse energy + allocation
/// symbols) doesn't shrink linearly with frame samples, so we don't
/// halve the byte count — see [`bytes_per_frame_for`] for the schedule.
const DEFAULT_BYTES_PER_FRAME: usize = 160;
/// Stereo needs more headroom for per-channel coarse/fine energy overhead
/// and the L/R shape bits. 256 bytes/frame ≈ 102 kbit/s at 20 ms.
const DEFAULT_BYTES_PER_FRAME_STEREO: usize = 256;

/// Default packet-size schedule per (frame_samples, channels). At LM=3
/// (960 samples / 20 ms) we use the historical 160 / 256 byte budgets
/// (≈ 64 / 102 kbit/s). At LM=2 (480 / 10 ms) the per-frame overhead
/// (header + coarse energy + allocation) doesn't shrink linearly with
/// frame size, so we keep ~75% of the 20 ms budget to leave enough room
/// for PVQ shape bits — 120 bytes mono / 192 bytes stereo (≈ 96 kbit/s
/// mono / 154 kbit/s stereo), well above the libopus minimum-viable
/// 10 ms budget for full-band coding.
fn bytes_per_frame_for(frame_samples: usize, channels: usize) -> usize {
    let base = if channels == 2 {
        DEFAULT_BYTES_PER_FRAME_STEREO
    } else {
        DEFAULT_BYTES_PER_FRAME
    };
    // Split into a fixed per-frame overhead and a per-coded-bin component.
    // Empirically the encoder needs ~80 bytes of overhead per frame plus
    // ~0.1 bytes per coded bin (mono) before it stops running into the
    // bit-budget exhaustion seen on LM=2 with a naively halved budget.
    if frame_samples == FRAME_SAMPLES {
        base
    } else {
        // For LM=2 (frame_samples = 480, half of 960), use 75% of the
        // 20 ms budget: 120 mono / 192 stereo.
        (base * 3) / 4
    }
}

/// Build the CELT-style short-block cosine window of length `len`. Kept in
/// sync with `decoder.rs::build_short_window` — the two must produce
/// bit-identical tables.
fn build_short_window(len: usize) -> Vec<f32> {
    let mut w = vec![0f32; len];
    for i in 0..len {
        let base = std::f32::consts::FRAC_PI_2 * (i as f32 + 0.5) / len as f32;
        let s = base.sin();
        w[i] = (std::f32::consts::FRAC_PI_2 * s * s).sin();
    }
    w
}

/// Default transient detection threshold, in dB. The detector flags a
/// frame as transient when the peak sub-block energy exceeds the median
/// sub-block energy by more than this many dB. Tuned empirically against
/// short test signals — see [`detect_transient`] for the rationale.
pub const DEFAULT_TRANSIENT_THRESHOLD_DB: f32 = 15.0;

/// Transient detection (libopus `transient_analysis` surrogate).
///
/// Splits `pcm` into 8 sub-blocks (matches LM=3 short-block count) and
/// compares the peak sub-block energy to the **median** sub-block energy.
/// If the ratio in dB exceeds `threshold_db`, the frame is flagged
/// transient.
///
/// We compare against the median (not the min) for robustness: a single
/// near-silent sub-block at the start or end of a frame would otherwise
/// trip a false positive on slow envelopes (fade-in / fade-out). The
/// median stays high through gradual envelope changes and only drops when
/// the actual majority of sub-blocks are quiet — which is exactly the
/// "burst-against-silence" pattern transient coding is meant to catch.
///
/// Returns `false` for near-silent input (peak energy below -80 dBFS²) so
/// that the silence fast-path runs unmolested. Callers using the public
/// encoder don't see this directly — a true return value flips the encoder
/// onto the [short-block path](`crate::mdct::forward_mdct_short`).
pub fn detect_transient(pcm: &[f32]) -> bool {
    detect_transient_with_threshold(pcm, DEFAULT_TRANSIENT_THRESHOLD_DB)
}

/// Same as [`detect_transient`] but with a caller-supplied threshold in
/// dB. Exposed for tests that want to pin the detector's behaviour at a
/// specific operating point. Production code should use [`detect_transient`].
pub fn detect_transient_with_threshold(pcm: &[f32], threshold_db: f32) -> bool {
    const SUB_BLOCKS: usize = 8;
    let block_len = pcm.len() / SUB_BLOCKS;
    if block_len == 0 {
        return false;
    }
    let mut energies = [0f32; SUB_BLOCKS];
    for b in 0..SUB_BLOCKS {
        let lo = b * block_len;
        let hi = lo + block_len;
        let mut e = 0f32;
        for &s in &pcm[lo..hi] {
            e += s * s;
        }
        energies[b] = e / block_len as f32;
    }
    // Peak energy across sub-blocks.
    let mut peak = energies[0];
    for &e in &energies[1..] {
        if e > peak {
            peak = e;
        }
    }
    // Guard against silent floors (avoid div-by-zero → spurious detection).
    // If the loudest sub-block is essentially zero the frame isn't transient
    // (silence-fast-path will handle it on the encode side).
    if peak < 1e-8 {
        return false;
    }
    // Median sub-block energy — robust against single-block outliers at
    // either end of the frame.
    let mut sorted = energies;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // SUB_BLOCKS=8 → median of two middle entries.
    let median = (sorted[SUB_BLOCKS / 2 - 1] + sorted[SUB_BLOCKS / 2]) * 0.5;
    let median = median.max(1e-12);
    let ratio_db = 10.0 * (peak / median).log10();
    ratio_db > threshold_db
}

/// CELT window — same as libopus and the decoder's post-filter (120 taps).
#[rustfmt::skip]
const WINDOW_120: [f32; 120] = [
    6.7286966e-05, 0.00060551348, 0.0016815970, 0.0032947962, 0.0054439943,
    0.0081276923, 0.011344001, 0.015090633, 0.019364886, 0.024163635,
    0.029483315, 0.035319905, 0.041668911, 0.048525347, 0.055883718,
    0.063737999, 0.072081616, 0.080907428, 0.090207705, 0.099974111,
    0.11019769, 0.12086883, 0.13197729, 0.14351214, 0.15546177,
    0.16781389, 0.18055550, 0.19367290, 0.20715171, 0.22097682,
    0.23513243, 0.24960208, 0.26436860, 0.27941419, 0.29472040,
    0.31026818, 0.32603788, 0.34200931, 0.35816177, 0.37447407,
    0.39092462, 0.40749142, 0.42415215, 0.44088423, 0.45766484,
    0.47447104, 0.49127978, 0.50806798, 0.52481261, 0.54149077,
    0.55807973, 0.57455701, 0.59090049, 0.60708841, 0.62309951,
    0.63891306, 0.65450896, 0.66986776, 0.68497077, 0.69980010,
    0.71433873, 0.72857055, 0.74248043, 0.75605424, 0.76927895,
    0.78214257, 0.79463430, 0.80674445, 0.81846456, 0.82978733,
    0.84070669, 0.85121779, 0.86131698, 0.87100183, 0.88027111,
    0.88912479, 0.89756398, 0.90559094, 0.91320904, 0.92042270,
    0.92723738, 0.93365955, 0.93969656, 0.94535671, 0.95064907,
    0.95558353, 0.96017067, 0.96442171, 0.96834849, 0.97196334,
    0.97527906, 0.97830883, 0.98106616, 0.98356480, 0.98581869,
    0.98784191, 0.98964856, 0.99125274, 0.99266849, 0.99390969,
    0.99499004, 0.99592297, 0.99672162, 0.99739874, 0.99796667,
    0.99843728, 0.99882195, 0.99913147, 0.99937606, 0.99956527,
    0.99970802, 0.99981248, 0.99988613, 0.99993565, 0.99996697,
    0.99998518, 0.99999457, 0.99999859, 0.99999982, 1.0000000,
];

pub struct CeltEncoder {
    params: CodecParameters,
    channels: usize,
    /// Per-instance CELT frame size in 48 kHz samples. 480 (LM=2, 10 ms)
    /// or 960 (LM=3, 20 ms). Drives the MDCT length, coded bin count and
    /// short-block sub-MDCT layout for this encoder.
    frame_samples: usize,
    /// Per-instance coded bin count = `100 * M = 100 * (1 << lm)`. 400 at
    /// LM=2, 800 at LM=3.
    coded_n: usize,
    /// Per-instance short-block sub-MDCT coefficient count = `coded_n / M`.
    /// 100 at both LM=2 (M=4) and LM=3 (M=8).
    short_n: usize,
    /// Per-instance LM = `log2(frame_samples / 120)`. 2 or 3.
    lm: i32,
    /// Pending interleaved PCM — L,R,L,R,... for stereo; length is a multiple
    /// of `channels`.
    pending: Vec<f32>,
    /// Previous frame's PCM tail for MDCT overlap (per channel).
    prev_tail: Vec<Vec<f32>>,
    /// Previous frame's quantised band energies (per-channel × NB_EBANDS).
    old_band_e: Vec<f32>,
    /// Whether the next encoded frame is the first frame of the session.
    /// The first frame MUST use `intra=true` because `old_band_e` is zero-
    /// initialised and would otherwise poison the decoder's prediction.
    first_frame: bool,
    output: VecDeque<Packet>,
    bytes_per_frame: usize,
    pts_counter: i64,
    /// Cached 100-tap cosine window for short sub-block MDCTs. Mirrors
    /// `CeltDecoder::short_window`.
    short_window: Vec<f32>,
    /// When true, suppress `detect_transient` and always emit long blocks.
    /// Intended for A/B testing short-block vs long-only reconstruction
    /// quality; production users shouldn't set this.
    force_long_only: bool,
    /// When true, force the per-band TF analyser
    /// ([`crate::tf_analysis::tf_analysis`]) into the no-op decision —
    /// every band gets `tf_change = 0` regardless of the masking model.
    /// Used by the round-trip A/B tests to measure the TF analyser's
    /// quality contribution against the baseline that pre-dated it.
    force_tf_off: bool,
    /// Pre-emphasis filter state (`x[n-1]` of the raw PCM, per channel).
    /// Applied as `y[n] = x[n] - alpha_p * x[n-1]` before MDCT. Pairs with
    /// the decoder's post-IMDCT de-emphasis so a full encode→decode chain
    /// recovers the original scale (the two are exact inverses).
    preemph_state: Vec<f32>,
    /// Test-only: when set, every emitted packet carries a post-filter
    /// header. The encoder does not run the encoder-side pitch pre-filter
    /// (libopus `celt_preemphasis`/`comb_filter_inplace` pair) — this knob
    /// simply fires the decoder's `comb_filter` + history + crossfade
    /// machinery on a stream where the pre-filter analysis was skipped, so
    /// the decoded output will not be the original signal. Purely a smoke
    /// test that the decoder's post-filter path does not panic / error.
    force_post_filter: Option<crate::header::PostFilter>,
}

impl CeltEncoder {
    /// Construct a default-frame-size (LM=3, 960-sample / 20 ms) CELT
    /// encoder. Equivalent to
    /// [`CeltEncoder::new_with_frame_samples(params, 960)`].
    pub fn new(params: &CodecParameters) -> Result<Self> {
        Self::new_with_frame_samples(params, FRAME_SAMPLES)
    }

    /// Construct a CELT encoder for an explicit frame size. Supported
    /// values are `960` (LM=3, 20 ms) and `480` (LM=2, 10 ms). All other
    /// sizes return [`Error::unsupported`]. The 10 ms path is the
    /// CELT-MDCT length used by the Opus 10 ms Hybrid configuration
    /// (RFC 6716 Table 2 configs 6/8/10) — the encoder otherwise behaves
    /// identically to the 20 ms path (same per-band quantisation, PVQ,
    /// TF analyser, silence/transient/post-filter machinery).
    ///
    /// Bit budget defaults scale linearly with `frame_samples` so the
    /// nominal mono ≈ 64 kbit/s / stereo ≈ 102 kbit/s targets stay the
    /// same across frame sizes.
    pub fn new_with_frame_samples(params: &CodecParameters, frame_samples: usize) -> Result<Self> {
        let channels = params.channels.unwrap_or(1) as usize;
        if channels != 1 && channels != 2 {
            return Err(Error::unsupported(
                "CELT encoder: only mono (1) and stereo (2) channels are supported",
            ));
        }
        let sr = params.sample_rate.unwrap_or(SAMPLE_RATE);
        if sr != SAMPLE_RATE {
            return Err(Error::unsupported("CELT encoder: only 48 kHz is supported"));
        }
        if frame_samples != 960 && frame_samples != 480 {
            return Err(Error::unsupported(
                "CELT encoder: frame_samples must be 480 (LM=2, 10 ms) or 960 (LM=3, 20 ms)",
            ));
        }
        let lm = lm_for_frame_samples(frame_samples as u32) as i32;
        // Coded bin count = EBAND_5MS[NB_EBANDS] * M = 100 * (1<<lm).
        let coded_n = 100usize << (lm as usize);
        // Short-block sub-MDCT coefficient count = coded_n / M = 100.
        let short_n = coded_n >> (lm as usize);
        let mut out_params = params.clone();
        out_params.channels = Some(channels as u16);
        out_params.sample_rate = Some(SAMPLE_RATE);
        // Default packet size — see [`bytes_per_frame_for`] for the
        // rationale on why LM=2 doesn't simply halve the LM=3 budget.
        let bytes_per_frame = bytes_per_frame_for(frame_samples, channels);
        Ok(Self {
            params: out_params,
            channels,
            frame_samples,
            coded_n,
            short_n,
            lm,
            pending: Vec::new(),
            prev_tail: vec![vec![0.0; OVERLAP]; channels],
            old_band_e: vec![0.0; NB_EBANDS * 2],
            first_frame: true,
            output: VecDeque::new(),
            bytes_per_frame,
            pts_counter: 0,
            short_window: build_short_window(short_n),
            force_long_only: false,
            force_tf_off: false,
            preemph_state: vec![0.0; channels],
            force_post_filter: None,
        })
    }

    /// Per-instance CELT frame length in 48 kHz samples (480 or 960).
    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// Test hook: force every emitted packet to carry a post-filter header
    /// with the given parameters. See the `force_post_filter` field for
    /// caveats — this does not run the encoder-side pitch pre-filter, so
    /// reconstruction quality is degraded. Used exclusively to exercise the
    /// decoder's post-filter code path in integration tests.
    #[doc(hidden)]
    pub fn set_force_post_filter(&mut self, pf: Option<crate::header::PostFilter>) {
        self.force_post_filter = pf;
    }

    /// Force the encoder to emit long blocks for every frame, bypassing
    /// transient detection. Used by the round-trip tests to A/B short-
    /// block coding against a long-only baseline on the same input.
    #[doc(hidden)]
    pub fn set_force_long_only(&mut self, value: bool) {
        self.force_long_only = value;
    }

    /// Force the encoder's per-band TF analyser
    /// ([`crate::tf_analysis::tf_analysis`]) into the no-op decision —
    /// every band ends up with `tf_change = 0`. Used by the round-trip
    /// tests to measure the TF picks' contribution to reconstruction
    /// quality against the pre-TF baseline. Production callers should
    /// leave this at the default (false) so the analyser drives the
    /// per-band decisions.
    #[doc(hidden)]
    pub fn set_force_tf_off(&mut self, value: bool) {
        self.force_tf_off = value;
    }

    /// Encode the CELT high-band portion of an Opus Hybrid frame (RFC
    /// 6716 §4.4) into the caller-provided `RangeEncoder`. This is the
    /// encoder-side mirror of `decode_celt_body` with `start_band=17`.
    ///
    /// * `pcm_in` — 960 samples (20 ms) of mono 48 kHz PCM. Only the
    ///   high-band content is meaningful — bands 0..17 (0..8 kHz) are
    ///   skipped so the caller is responsible for filtering / SILK-
    ///   covering that region. We still take the full-band PCM since
    ///   the MDCT operates on it before the band-energy split.
    /// * `enc` — in-flight range encoder shared with the SILK low band.
    ///   Must already contain the SILK frame body. The CELT body is
    ///   appended in place; callers call `enc.done()` after this returns.
    /// * `start_band` — 17 for both SWB and FB hybrid (RFC 6716 §4.4).
    /// * `end_band` — 19 (SWB) or 21 (FB).
    /// * `bytes_remaining_budget` — how many bytes the caller wants
    ///   the entire packet to fit in. The CELT bit allocator uses this
    ///   to decide pulse counts; pick `total_packet_bytes - silk_bytes_so_far`.
    ///
    /// State (`old_band_e`, `prev_tail`, `preemph_state`, `first_frame`)
    /// is updated as in [`Self::encode_frame`] — bands below `start_band`
    /// are kept at their previous value (we only touch `start_band..end_band`,
    /// matching the decoder's behaviour). Frame size is 20 ms only;
    /// 10 ms hybrid (configs 12, 14) is a tracked follow-up.
    pub fn encode_hybrid_body_mono(
        &mut self,
        pcm_in: &[f32],
        enc: &mut RangeEncoder,
        start_band: usize,
        end_band: usize,
        bytes_remaining_budget: usize,
    ) -> Result<()> {
        if pcm_in.len() != FRAME_SAMPLES {
            return Err(Error::invalid(
                "CELT hybrid encoder: only 20 ms (960 samples) frames are supported today",
            ));
        }
        if !(start_band < end_band && end_band <= NB_EBANDS) {
            return Err(Error::invalid(
                "CELT hybrid encoder: invalid start/end band range",
            ));
        }
        if self.channels != 1 {
            return Err(Error::unsupported(
                "CELT hybrid encoder: stereo CELT high-band not implemented",
            ));
        }
        let lm = lm_for_frame_samples(FRAME_SAMPLES as u32) as i32;

        // Pre-emphasis — same filter the CELT-only path uses, since the
        // decoder applies the same de-emphasis on the summed Hybrid output.
        let alpha = crate::tables::DEEMPHASIS_COEF;
        let mut pcm: Vec<f32> = Vec::with_capacity(FRAME_SAMPLES);
        {
            let mut prev = self.preemph_state[0];
            for &x in pcm_in {
                pcm.push(x - alpha * prev);
                prev = x;
            }
            self.preemph_state[0] = prev;
        }
        let pcm = pcm.as_slice();

        // Hybrid never emits the silence fast path — the SILK body has
        // already committed to a non-silent frame. Skip the peak gate.

        let n = CODED_N;
        // Hybrid frames never go transient — the SILK low band carries
        // the speech onset; CELT high band stays long. This keeps the
        // bit budget simple.
        let transient = false;

        let mut raw = vec![0f32; 2 * n];
        raw[..OVERLAP].copy_from_slice(&self.prev_tail[0]);
        let take = n.min(pcm.len());
        raw[OVERLAP..OVERLAP + take].copy_from_slice(&pcm[..take]);
        self.prev_tail[0].copy_from_slice(&pcm[FRAME_SAMPLES - OVERLAP..]);

        crate::mdct::window_forward(&mut raw, &WINDOW_120, n, OVERLAP);

        let mut coeffs = vec![0f32; n];
        forward_mdct(&raw, &mut coeffs);

        // Per-band log-energy and normalised shape — bands 0..start_band
        // are not coded and not used downstream, but we still compute
        // their energy so the prediction state stays in sync with how
        // the encoder thinks of "old_band_e".
        let m = 1i32 << lm;
        let mut band_log_e = vec![0f32; NB_EBANDS];
        let mut shape = vec![0f32; n];
        for i in 0..NB_EBANDS {
            let lo = (m * EBAND_5MS[i] as i32) as usize;
            let hi = (m * EBAND_5MS[i + 1] as i32) as usize;
            let mut e: f32 = 0.0;
            for &c in &coeffs[lo..hi] {
                e += c * c;
            }
            let e = e.max(1e-30).sqrt();
            band_log_e[i] = e.log2() - E_MEANS[i];
            for c in &mut shape[lo..hi] {
                *c /= e;
            }
        }

        // Intra/inter — Hybrid frames need an honest first-frame intra
        // for the same reason the CELT-only path does: old_band_e is all
        // zeros until we've encoded one frame.
        let intra = self.first_frame;
        let bytes = bytes_remaining_budget;

        // Header: silence=0, no post-filter, transient=0, intra=<flag>.
        encode_header(enc, false, None, transient, intra);

        // Coarse energy across [start_band, end_band).
        let mut new_log_e = vec![0f32; NB_EBANDS * 2];
        new_log_e[..NB_EBANDS].copy_from_slice(&band_log_e);
        let mut old_e_bands = self.old_band_e.clone();
        quant_coarse_energy(
            enc,
            &new_log_e,
            &mut old_e_bands,
            start_band,
            end_band,
            intra,
            1,
            lm as usize,
        );

        // tf_decode: emit raw=0 + tsel=0 → tf_res[i] = 0 everywhere
        // (long blocks). Same logic as encode_frame, restricted to
        // [start_band, end_band).
        let tf_delta_first = false;
        let tf_delta_rest = false;
        let tf_sel: bool = false;

        let budget = (bytes * 8) as u32;
        let mut tell_u = enc.tell() as u32;
        let mut logp: u32 = 4; // non-transient
        let tf_select_rsv = if lm > 0 && tell_u + logp + 1 <= budget {
            1
        } else {
            0
        };
        let budget_after = budget - tf_select_rsv;
        for band_i in start_band..end_band {
            if tell_u + logp <= budget_after {
                let bit = if band_i == start_band {
                    tf_delta_first
                } else {
                    tf_delta_rest
                };
                enc.encode_bit_logp(bit, logp);
                tell_u = enc.tell() as u32;
            }
            logp = 5;
        }
        if tf_select_rsv != 0
            && crate::tables::TF_SELECT_TABLE[lm as usize][0]
                != crate::tables::TF_SELECT_TABLE[lm as usize][2]
        {
            enc.encode_bit_logp(tf_sel, 1);
        }
        let tf_res = vec![0i32; NB_EBANDS];

        // Spread decision.
        let tell = enc.tell();
        let total_bits_check = (bytes * 8) as i32;
        if tell + 4 <= total_bits_check {
            enc.encode_icdf(SPREAD_NORMAL as usize, &SPREAD_ICDF, 5);
        }

        // Dynalloc — emit no boost per band. Loop runs over the coded
        // band range only, matching the decoder.
        let cap = init_caps(lm as usize, 1);
        let mut offsets = [0i32; NB_EBANDS];
        let dynalloc_logp = 6i32;
        let total_bits_frac = (bytes as i32) * 8 << BITRES;
        let mut tell = enc.tell_frac() as i32;
        for i in start_band..end_band {
            let _width = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32 * m;
            let dynalloc_loop_logp = dynalloc_logp;
            let boost = 0i32;
            if tell + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
                enc.encode_bit_logp(false, dynalloc_loop_logp as u32);
                tell = enc.tell_frac() as i32;
            }
            offsets[i] = boost;
        }
        let _ = total_bits_frac;

        // Allocation trim.
        if tell + (6 << BITRES) <= total_bits_frac {
            enc.encode_icdf(5, &TRIM_ICDF, 7);
        }

        let mut bits = ((bytes as i32) * 8 << BITRES) - enc.tell_frac() as i32 - 1;
        // Anti-collapse: never reserved on Hybrid (transient = false).
        let anti_collapse_rsv = 0i32;
        bits -= anti_collapse_rsv;

        let mut pulses = vec![0i32; NB_EBANDS];
        let mut fine_quant = vec![0i32; NB_EBANDS];
        let mut fine_priority = vec![0i32; NB_EBANDS];
        let mut balance = 0i32;
        let mut intensity = 0i32;
        let mut dual_stereo = 0i32;
        let coded_bands = clt_compute_allocation_enc(
            start_band,
            end_band,
            &offsets,
            &cap,
            5,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            1,
            lm,
            enc,
        );
        let _ = (intensity, dual_stereo);

        // Fine energy.
        quant_fine_energy(
            enc,
            &new_log_e,
            &mut old_e_bands,
            start_band,
            end_band,
            &fine_quant,
            1,
        );

        // PVQ shape. The shape buffer carries n = CODED_N normalised
        // coefficients; only bands [start_band, end_band) are coded.
        let total_pvq_bits = (bytes as i32) * (8 << BITRES) - anti_collapse_rsv;
        let mut collapse_masks = vec![0u8; NB_EBANDS];
        let mut rng_local = 0u32;
        encode_all_bands_mono(
            start_band,
            end_band,
            &mut shape,
            &mut collapse_masks,
            &pulses,
            transient,
            SPREAD_NORMAL,
            &tf_res,
            total_pvq_bits,
            balance,
            enc,
            lm,
            coded_bands,
            &mut rng_local,
        );

        // Anti-collapse not reserved → no flag bit emitted.
        if anti_collapse_rsv > 0 {
            enc.encode_bits(0, 1);
        }

        // Final fine-energy pass.
        let bits_left = (bytes as i32) * 8 - enc.tell();
        quant_energy_finalise(
            enc,
            &new_log_e,
            &mut old_e_bands,
            start_band,
            end_band,
            &fine_quant,
            &fine_priority,
            bits_left,
            1,
        );

        self.old_band_e = old_e_bands;
        self.first_frame = false;
        Ok(())
    }

    /// Stereo variant of [`Self::encode_hybrid_body_mono`] — encode the
    /// CELT high-band of an Opus Hybrid stereo frame (RFC 6716 §4.4) into
    /// the caller-provided `RangeEncoder`. Mirrors the `decode_celt_body`
    /// path with `channels = 2` and `start_band = 17`.
    ///
    /// `pcm_in` is **interleaved L,R,L,R...** (length = `2 * FRAME_SAMPLES`).
    /// The encoder must have been constructed with `channels = 2` so that
    /// `prev_tail` and `preemph_state` carry per-channel state.
    ///
    /// Layout of the emitted bits matches the decoder's hybrid stereo path:
    /// header → coarse energy (both channels) → tf_decode → spread → dynalloc
    /// → trim → allocation (`channels = 2`, dual-stereo coupling) → fine
    /// energy → PVQ shape via `encode_all_bands_stereo_dual` restricted to
    /// `[start_band, end_band)` → anti-collapse-on flag (always 0 since
    /// hybrid is non-transient) → final fine pass.
    ///
    /// Cross-frame state (`old_band_e`, `prev_tail`, `preemph_state`,
    /// `first_frame`) is updated for both channels.
    pub fn encode_hybrid_body_stereo(
        &mut self,
        pcm_in: &[f32],
        enc: &mut RangeEncoder,
        start_band: usize,
        end_band: usize,
        bytes_remaining_budget: usize,
    ) -> Result<()> {
        if pcm_in.len() != 2 * FRAME_SAMPLES {
            return Err(Error::invalid(
                "CELT hybrid encoder: stereo path needs 2 * 960 (interleaved) samples",
            ));
        }
        if !(start_band < end_band && end_band <= NB_EBANDS) {
            return Err(Error::invalid(
                "CELT hybrid encoder: invalid start/end band range",
            ));
        }
        if self.channels != 2 {
            return Err(Error::unsupported(
                "CELT hybrid encoder: stereo path requires channels=2 encoder",
            ));
        }
        let lm = lm_for_frame_samples(FRAME_SAMPLES as u32) as i32;
        let channels = 2usize;

        // De-interleave into L and R, applying per-channel pre-emphasis to
        // match the decoder's per-channel de-emphasis on the summed Hybrid
        // output.
        let mut l_pcm = vec![0f32; FRAME_SAMPLES];
        let mut r_pcm = vec![0f32; FRAME_SAMPLES];
        for i in 0..FRAME_SAMPLES {
            l_pcm[i] = pcm_in[2 * i];
            r_pcm[i] = pcm_in[2 * i + 1];
        }
        let alpha = crate::tables::DEEMPHASIS_COEF;
        {
            let mut prev = self.preemph_state[0];
            for v in l_pcm.iter_mut() {
                let x = *v;
                *v = x - alpha * prev;
                prev = x;
            }
            self.preemph_state[0] = prev;
        }
        {
            let mut prev = self.preemph_state[1];
            for v in r_pcm.iter_mut() {
                let x = *v;
                *v = x - alpha * prev;
                prev = x;
            }
            self.preemph_state[1] = prev;
        }

        // Hybrid never emits the silence fast path — the SILK body has
        // already committed to a non-silent frame. Skip the peak gate.

        let n = CODED_N;
        // Hybrid frames never go transient — long blocks only.
        let transient = false;

        // Forward MDCT per channel.
        let mut coeffs_l = vec![0f32; n];
        let mut coeffs_r = vec![0f32; n];
        for (ch, pcm_ch, coeffs) in [
            (0usize, &l_pcm, &mut coeffs_l),
            (1usize, &r_pcm, &mut coeffs_r),
        ] {
            let mut raw = vec![0f32; 2 * n];
            raw[..OVERLAP].copy_from_slice(&self.prev_tail[ch]);
            let take = n.min(pcm_ch.len());
            raw[OVERLAP..OVERLAP + take].copy_from_slice(&pcm_ch[..take]);
            self.prev_tail[ch].copy_from_slice(&pcm_ch[FRAME_SAMPLES - OVERLAP..]);
            crate::mdct::window_forward(&mut raw, &WINDOW_120, n, OVERLAP);
            forward_mdct(&raw, coeffs);
        }

        // Per-band log-energy and normalised shape, per channel.
        let m = 1i32 << lm;
        let mut band_log_e = vec![0f32; NB_EBANDS * channels];
        let mut shape_l = vec![0f32; n];
        let mut shape_r = vec![0f32; n];
        for i in 0..NB_EBANDS {
            let lo = (m * EBAND_5MS[i] as i32) as usize;
            let hi = (m * EBAND_5MS[i + 1] as i32) as usize;
            for (ch, coeffs, shape) in [
                (0usize, &coeffs_l, &mut shape_l),
                (1usize, &coeffs_r, &mut shape_r),
            ] {
                let mut e: f32 = 0.0;
                for &c in &coeffs[lo..hi] {
                    e += c * c;
                }
                let e = e.max(1e-30).sqrt();
                band_log_e[ch * NB_EBANDS + i] = e.log2() - E_MEANS[i];
                for c in &mut shape[lo..hi] {
                    *c /= e;
                }
            }
        }

        // First-frame intra so old_band_e isn't poisoned.
        let intra = self.first_frame;
        let bytes = bytes_remaining_budget;

        // Header: silence=0, no post-filter, transient=0, intra=<flag>.
        encode_header(enc, false, None, transient, intra);

        // Coarse energy (both channels, restricted to coded band range).
        let mut new_log_e = vec![0f32; NB_EBANDS * 2];
        new_log_e[..NB_EBANDS * channels].copy_from_slice(&band_log_e);
        let mut old_e_bands = self.old_band_e.clone();
        quant_coarse_energy(
            enc,
            &new_log_e,
            &mut old_e_bands,
            start_band,
            end_band,
            intra,
            channels,
            lm as usize,
        );

        // tf_decode: emit raw=0 + tsel=0 → tf_res[i] = 0 everywhere.
        let tf_delta_first = false;
        let tf_delta_rest = false;
        let tf_sel: bool = false;

        let budget = (bytes * 8) as u32;
        let mut tell_u = enc.tell() as u32;
        let mut logp: u32 = 4; // non-transient
        let tf_select_rsv = if lm > 0 && tell_u + logp + 1 <= budget {
            1
        } else {
            0
        };
        let budget_after = budget - tf_select_rsv;
        for band_i in start_band..end_band {
            if tell_u + logp <= budget_after {
                let bit = if band_i == start_band {
                    tf_delta_first
                } else {
                    tf_delta_rest
                };
                enc.encode_bit_logp(bit, logp);
                tell_u = enc.tell() as u32;
            }
            logp = 5;
        }
        if tf_select_rsv != 0
            && crate::tables::TF_SELECT_TABLE[lm as usize][0]
                != crate::tables::TF_SELECT_TABLE[lm as usize][2]
        {
            enc.encode_bit_logp(tf_sel, 1);
        }
        let tf_res = vec![0i32; NB_EBANDS];

        // Spread decision.
        let tell = enc.tell();
        let total_bits_check = (bytes * 8) as i32;
        if tell + 4 <= total_bits_check {
            enc.encode_icdf(SPREAD_NORMAL as usize, &SPREAD_ICDF, 5);
        }

        // Dynalloc — emit no boost per band.
        let cap = init_caps(lm as usize, channels);
        let mut offsets = [0i32; NB_EBANDS];
        let dynalloc_logp = 6i32;
        let total_bits_frac = (bytes as i32) * 8 << BITRES;
        let mut tell = enc.tell_frac() as i32;
        for i in start_band..end_band {
            let _width = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32 * m;
            let dynalloc_loop_logp = dynalloc_logp;
            let boost = 0i32;
            if tell + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
                enc.encode_bit_logp(false, dynalloc_loop_logp as u32);
                tell = enc.tell_frac() as i32;
            }
            offsets[i] = boost;
        }
        let _ = total_bits_frac;

        // Allocation trim.
        if tell + (6 << BITRES) <= total_bits_frac {
            enc.encode_icdf(5, &TRIM_ICDF, 7);
        }

        let mut bits = ((bytes as i32) * 8 << BITRES) - enc.tell_frac() as i32 - 1;
        // Anti-collapse: never reserved on Hybrid (transient = false).
        let anti_collapse_rsv = 0i32;
        bits -= anti_collapse_rsv;

        let mut pulses = vec![0i32; NB_EBANDS];
        let mut fine_quant = vec![0i32; NB_EBANDS];
        let mut fine_priority = vec![0i32; NB_EBANDS];
        let mut balance = 0i32;
        let mut intensity = 0i32;
        let mut dual_stereo = 0i32;
        let coded_bands = clt_compute_allocation_enc(
            start_band,
            end_band,
            &offsets,
            &cap,
            5,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            channels as i32,
            lm,
            enc,
        );
        let _ = (intensity, dual_stereo);

        // Fine energy (both channels).
        quant_fine_energy(
            enc,
            &new_log_e,
            &mut old_e_bands,
            start_band,
            end_band,
            &fine_quant,
            channels,
        );

        // PVQ shape — dual-stereo, restricted to [start_band, end_band).
        let total_pvq_bits = (bytes as i32) * (8 << BITRES) - anti_collapse_rsv;
        let mut collapse_masks = vec![0u8; NB_EBANDS * channels];
        let mut rng_local = 0u32;
        encode_all_bands_stereo_dual(
            start_band,
            end_band,
            &mut shape_l,
            &mut shape_r,
            &mut collapse_masks,
            &pulses,
            transient,
            SPREAD_NORMAL,
            &tf_res,
            total_pvq_bits,
            balance,
            enc,
            lm,
            coded_bands,
            &mut rng_local,
        );

        // Anti-collapse-on flag (anti_collapse_rsv == 0, so no bit emitted).
        if anti_collapse_rsv > 0 {
            enc.encode_bits(0, 1);
        }

        // Final fine-energy pass.
        let bits_left = (bytes as i32) * 8 - enc.tell();
        quant_energy_finalise(
            enc,
            &new_log_e,
            &mut old_e_bands,
            start_band,
            end_band,
            &fine_quant,
            &fine_priority,
            bits_left,
            channels,
        );

        self.old_band_e = old_e_bands;
        self.first_frame = false;
        Ok(())
    }

    fn drain_frames(&mut self) -> Result<()> {
        let needed = self.frame_samples * self.channels;
        while self.pending.len() >= needed {
            let frame: Vec<f32> = self.pending.drain(..needed).collect();
            let pkt = if self.channels == 1 {
                self.encode_frame(&frame)?
            } else {
                self.encode_frame_stereo(&frame)?
            };
            self.output.push_back(pkt);
        }
        Ok(())
    }

    fn encode_frame(&mut self, pcm_in: &[f32]) -> Result<Packet> {
        debug_assert_eq!(pcm_in.len(), self.frame_samples);
        let lm = self.lm;
        debug_assert!(lm == 2 || lm == 3);
        let frame_samples = self.frame_samples;

        // Pre-emphasis (RFC 6716 §4.3.7.2 inverse): `y[n] = x[n] - alpha_p *
        // x[n-1]`. Carried state is the previous raw input sample per
        // channel. Runs here so both the silence peak check (below) and all
        // downstream MDCT stages see the pre-emphasized signal — the
        // decoder's post-IMDCT de-emphasis is the exact inverse.
        let alpha = crate::tables::DEEMPHASIS_COEF;
        let mut pcm: Vec<f32> = Vec::with_capacity(frame_samples);
        {
            let mut prev = self.preemph_state[0];
            for &x in pcm_in {
                pcm.push(x - alpha * prev);
                prev = x;
            }
            self.preemph_state[0] = prev;
        }
        let pcm = pcm.as_slice();

        // Silence-flag fast path. If the input frame is below a hard -90 dBFS
        // floor (peak |s| < 1e-5), emit a silence-only packet: the decoder
        // sees the silence bit and skips every other §4.3.2-§4.3.7 symbol,
        // producing zero PCM. This is RFC 6716 §4.3 Table 56 silence, not the
        // same as DTX — the packet is still emitted, just tiny. Peak is
        // measured on the raw input to avoid pre-emphasis boosting HF noise
        // into the non-silent regime.
        let peak = pcm_in.iter().fold(0f32, |m, &s| m.max(s.abs()));
        if peak < 1e-5 {
            let bytes = self.bytes_per_frame;
            let mut rc = RangeEncoder::new(bytes as u32);
            encode_header(&mut rc, true, None, false, self.first_frame);
            // Silence wipes the band-energy prediction state: there's nothing
            // useful to carry forward.
            for v in &mut self.old_band_e {
                *v = 0.0;
            }
            // And also the OLA tail — the next frame's overlap region is zero.
            for v in &mut self.prev_tail[0] {
                *v = 0.0;
            }
            self.first_frame = false;
            let buf = rc.done()?;
            let tb = TimeBase::new(1, SAMPLE_RATE as i64);
            let pts = self.pts_counter;
            self.pts_counter += frame_samples as i64;
            return Ok(Packet::new(0, tb, buf)
                .with_pts(pts)
                .with_duration(frame_samples as i64));
        }

        // Build the 2N-point MDCT input frame where N = self.coded_n (the
        // coded-bin count = 100*M). We resample the PCM into N samples by
        // simple truncation: keep the first N samples of the frame +
        // previous tail. This loses the top ~2 kHz of bandwidth per frame
        // but keeps the encoder's MDCT coefficient count consistent with
        // what the decoder expects (`n = m * EBAND_5MS[NB_EBANDS]`).
        let n = self.coded_n;

        // Transient detection: run BEFORE assembling the windowed MDCT
        // input, on the raw PCM so the sub-block energies reflect actual
        // input dynamics (not post-window envelope decay).
        let transient = !self.force_long_only && detect_transient(pcm);

        let mut raw = vec![0f32; 2 * n];
        raw[..OVERLAP].copy_from_slice(&self.prev_tail[0]);
        // Place up to N PCM samples starting after the overlap.
        let take = n.min(pcm.len());
        raw[OVERLAP..OVERLAP + take].copy_from_slice(&pcm[..take]);
        // Stash the tail of THIS frame's PCM (last OVERLAP samples) for the
        // next frame — this uses the raw frame tail (`frame_samples`), not
        // the coded `n` tail, so the OLA at the decoder side lines up.
        self.prev_tail[0].copy_from_slice(&pcm[frame_samples - OVERLAP..]);

        // Apply CELT window (only the overlap regions) — this windowing
        // acts at the frame boundary for both long and short blocks, so the
        // decoder's matching OLA with the adjacent frame's tail lines up.
        crate::mdct::window_forward(&mut raw, &WINDOW_120, n, OVERLAP);

        // 2) Forward MDCT → N coefficients. Short blocks run 8 × 200→100
        // sub-MDCTs with their own 100-tap sin² window at sub-block
        // boundaries, interleaved into `coeffs` at stride M (matching the
        // decoder's `imdct_sub_short` coefficient layout).
        let mut coeffs = vec![0f32; n];
        if transient {
            crate::mdct::forward_mdct_short(&raw, &mut coeffs, n, lm as usize, &self.short_window);
        } else {
            forward_mdct(&raw, &mut coeffs);
        }

        // 3) Compute per-band log-energy and normalised shape.
        let m = 1i32 << lm;
        let mut band_log_e = vec![0f32; NB_EBANDS];
        let mut shape = vec![0f32; n];
        for i in 0..NB_EBANDS {
            let lo = (m * EBAND_5MS[i] as i32) as usize;
            let hi = (m * EBAND_5MS[i + 1] as i32) as usize;
            let mut e: f32 = 0.0;
            for &c in &coeffs[lo..hi] {
                e += c * c;
            }
            let e = e.max(1e-30).sqrt();
            band_log_e[i] = e.log2() - E_MEANS[i];
            for c in &mut shape[lo..hi] {
                *c /= e;
            }
        }

        // 4) Range-code the frame.
        // Intra-vs-inter decision: force intra on the first frame of the
        // session (old_band_e is all zeros — using it as prediction state
        // would poison the decoder). Subsequent frames use inter prediction
        // for tighter residuals on steady-state content.
        let intra = self.first_frame;
        let bytes = self.bytes_per_frame;
        let mut rc = RangeEncoder::new(bytes as u32);
        // Header: silence=0, optional post-filter (test-only hook),
        // transient=<flag>, intra=<flag>.
        encode_header(&mut rc, false, self.force_post_filter, transient, intra);

        // Coarse energy.
        let mut new_log_e = vec![0f32; NB_EBANDS * 2];
        new_log_e[..NB_EBANDS].copy_from_slice(&band_log_e);
        let old_before = self.old_band_e.clone();
        let mut old_e_bands = old_before.clone();
        quant_coarse_energy(
            &mut rc,
            &new_log_e,
            &mut old_e_bands,
            0,
            NB_EBANDS,
            intra,
            1,
            lm as usize,
        );

        // tf_decode (RFC 6716 §4.3.4.5): pick per-band TF resolution
        // adjustments via the L1-norm masking model in
        // [`crate::tf_analysis::tf_analysis`]. The analyser returns
        // - the per-band post-lookup `tf_res[i]` values,
        // - the matching raw delta bits + `tf_select` flag the decoder
        //   needs to reconstruct them via cumulative XOR through
        //   `TF_SELECT_TABLE`.
        //
        // The encoder pipeline currently honours `tf_res[i] != 0` only in
        // the mono long-block (non-transient) path via the haar1 wrapping
        // in `encoder_bands::encode_all_bands_mono`. Transient frames fall
        // back to the no-op decision so the encoder + decoder stay in
        // sync; see `tf_analysis::tf_decision_no_op` for the bit pattern.
        let tf_decision = if transient || self.force_tf_off {
            crate::tf_analysis::tf_decision_no_op(transient, 0, NB_EBANDS)
        } else {
            let raw_dec = crate::tf_analysis::tf_analysis(&coeffs, transient, lm, 0, NB_EBANDS);
            if raw_dec.apply_in_pipeline {
                raw_dec
            } else {
                crate::tf_analysis::tf_decision_no_op(transient, 0, NB_EBANDS)
            }
        };

        let budget = (bytes * 8) as u32;
        let mut tell_u = rc.tell() as u32;
        let mut logp: u32 = if transient { 2 } else { 4 };
        let tf_select_rsv = if lm > 0 && tell_u + logp + 1 <= budget {
            1
        } else {
            0
        };
        let budget_after = budget - tf_select_rsv;
        // Track which bands actually carried a delta bit (matches the
        // decoder's `tell_u + logp <= budget_after` gate). Bands skipped
        // due to budget exhaustion don't contribute to the cumulative XOR
        // on either side — but the analyser already includes them in the
        // raw stream so the cumulative XOR matches.
        let mut emitted_changed = 0i32;
        let mut curr_chk = 0i32;
        for band_i in 0..NB_EBANDS {
            if tell_u + logp <= budget_after {
                let bit = tf_decision.raw_per_band[band_i];
                rc.encode_bit_logp(bit, logp);
                tell_u = rc.tell() as u32;
                curr_chk ^= bit as i32;
                emitted_changed |= curr_chk;
            }
            logp = if transient { 4 } else { 5 };
        }
        // Match the decoder's `tf_select` reservation gate: it depends on
        // the cumulative-XOR `tf_changed` value (which can be 0 or 1)
        // produced while reading the raw delta bits, not just the seed.
        // `emitted_changed = OR of curr_chk over all coded bands`.
        if tf_select_rsv != 0
            && crate::tables::TF_SELECT_TABLE[lm as usize]
                [4 * transient as usize + emitted_changed as usize]
                != crate::tables::TF_SELECT_TABLE[lm as usize]
                    [4 * transient as usize + 2 + emitted_changed as usize]
        {
            rc.encode_bit_logp(tf_decision.tf_select, 1);
        }
        // Per-band post-lookup TF changes (RFC §4.3.4.5). Drives the
        // haar1 chain in `encoder_bands::encode_all_bands_mono` for
        // non-transient frames where the analyser opted for ≠ 0 entries.
        let tf_res = tf_decision.tf_res.clone();

        // Spread decision.
        let mut tell = rc.tell();
        let total_bits_check = (bytes * 8) as i32;
        if tell + 4 <= total_bits_check {
            rc.encode_icdf(SPREAD_NORMAL as usize, &SPREAD_ICDF, 5);
        }

        // dynalloc offsets: emit ALL zeros. Dec loop emits `decode_bit_logp(dynalloc_logp)`
        // until it gets back false. So we emit ONE false per band (no boosts).
        let cap = init_caps(lm as usize, 1);
        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_frac = (bytes as i32) * 8 << BITRES;
        tell = rc.tell_frac() as i32;
        for i in 0..NB_EBANDS {
            let width = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32 * m;
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            if tell + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
                // Emit `false` = no boost. Decoder breaks out on `!flag`, so
                // we only ever emit at most one.
                rc.encode_bit_logp(false, dynalloc_loop_logp as u32);
                tell = rc.tell_frac() as i32;
            }
            offsets[i] = boost;
            // dynalloc_logp stays at 6 since we added no boost.
        }
        let _ = total_bits_frac;

        // Allocation trim — emit default (5).
        if tell + (6 << BITRES) <= total_bits_frac {
            rc.encode_icdf(5, &TRIM_ICDF, 7);
        }

        let mut bits = ((bytes as i32) * 8 << BITRES) - rc.tell_frac() as i32 - 1;
        // Anti-collapse reservation (§4.3.5): 1 bit for transient LM>=2 frames
        // when budget is large enough. Mirrors the decoder gate exactly.
        let anti_collapse_rsv = if transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
            1 << BITRES
        } else {
            0i32
        };
        bits -= anti_collapse_rsv;

        let mut pulses = vec![0i32; NB_EBANDS];
        let mut fine_quant = vec![0i32; NB_EBANDS];
        let mut fine_priority = vec![0i32; NB_EBANDS];
        let mut balance = 0i32;
        let mut intensity = 0i32;
        let mut dual_stereo = 0i32;
        let coded_bands = clt_compute_allocation_enc(
            0,
            NB_EBANDS,
            &offsets,
            &cap,
            5,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            1,
            lm,
            &mut rc,
        );
        let _ = (intensity, dual_stereo);

        // Fine energy.
        quant_fine_energy(
            &mut rc,
            &new_log_e,
            &mut old_e_bands,
            0,
            NB_EBANDS,
            &fine_quant,
            1,
        );

        // PVQ shape.
        let total_pvq_bits = (bytes as i32) * (8 << BITRES) - anti_collapse_rsv;
        let mut collapse_masks = vec![0u8; NB_EBANDS];
        let mut rng_local = 0u32;
        encode_all_bands_mono(
            0,
            NB_EBANDS,
            &mut shape,
            &mut collapse_masks,
            &pulses,
            transient,
            SPREAD_NORMAL,
            &tf_res,
            total_pvq_bits,
            balance,
            &mut rc,
            lm,
            coded_bands,
            &mut rng_local,
        );

        // Anti-collapse ON flag. We choose `off`, so emit the reserved 1 bit
        // as zero — decoder skips the anti-collapse pass.
        if anti_collapse_rsv > 0 {
            rc.encode_bits(0, 1);
        }

        // Final fine-energy pass (bits left from total - rc.tell()).
        let bits_left = (bytes as i32) * 8 - rc.tell();
        quant_energy_finalise(
            &mut rc,
            &new_log_e,
            &mut old_e_bands,
            0,
            NB_EBANDS,
            &fine_quant,
            &fine_priority,
            bits_left,
            1,
        );

        // Commit energy state for next frame.
        self.old_band_e = old_e_bands;
        self.first_frame = false;

        let buf = rc.done()?;
        let tb = TimeBase::new(1, SAMPLE_RATE as i64);
        let pts = self.pts_counter;
        self.pts_counter += frame_samples as i64;
        Ok(Packet::new(0, tb, buf)
            .with_pts(pts)
            .with_duration(frame_samples as i64))
    }

    /// Stereo (dual-stereo) encode path. `pcm` is interleaved L,R,L,R,...
    /// length = 2 * `self.frame_samples`. L goes into the x channel, R into y.
    /// Intensity stereo is NOT applied (encoder pins `intensity = coded_bands`
    /// so the decoder never merges norm channels).
    fn encode_frame_stereo(&mut self, pcm: &[f32]) -> Result<Packet> {
        debug_assert_eq!(pcm.len(), 2 * self.frame_samples);
        let lm = self.lm;
        debug_assert!(lm == 2 || lm == 3);
        let frame_samples = self.frame_samples;
        let channels = 2usize;

        // Silence fast path — peak across both channels below -90 dBFS.
        let peak = pcm.iter().fold(0f32, |m, &s| m.max(s.abs()));
        if peak < 1e-5 {
            let bytes = self.bytes_per_frame;
            let mut rc = RangeEncoder::new(bytes as u32);
            encode_header(&mut rc, true, None, false, self.first_frame);
            for v in &mut self.old_band_e {
                *v = 0.0;
            }
            for ch in &mut self.prev_tail {
                for v in ch {
                    *v = 0.0;
                }
            }
            self.first_frame = false;
            let buf = rc.done()?;
            let tb = TimeBase::new(1, SAMPLE_RATE as i64);
            let pts = self.pts_counter;
            self.pts_counter += frame_samples as i64;
            return Ok(Packet::new(0, tb, buf)
                .with_pts(pts)
                .with_duration(frame_samples as i64));
        }

        // De-interleave into L and R PCM, then apply per-channel
        // pre-emphasis (RFC §4.3.7.2 inverse, matched by the decoder's
        // de-emphasis).
        let mut l_pcm = vec![0f32; frame_samples];
        let mut r_pcm = vec![0f32; frame_samples];
        for i in 0..frame_samples {
            l_pcm[i] = pcm[2 * i];
            r_pcm[i] = pcm[2 * i + 1];
        }
        let alpha = crate::tables::DEEMPHASIS_COEF;
        {
            let mut prev = self.preemph_state[0];
            for v in l_pcm.iter_mut() {
                let x = *v;
                *v = x - alpha * prev;
                prev = x;
            }
            self.preemph_state[0] = prev;
        }
        {
            let mut prev = self.preemph_state[1];
            for v in r_pcm.iter_mut() {
                let x = *v;
                *v = x - alpha * prev;
                prev = x;
            }
            self.preemph_state[1] = prev;
        }

        // Transient detection: flag if either channel has percussive onset.
        let transient =
            !self.force_long_only && (detect_transient(&l_pcm) || detect_transient(&r_pcm));

        let n = self.coded_n;
        let m = 1i32 << lm;

        // Forward MDCT per channel. Output: coeffs_l, coeffs_r (length n).
        let mut coeffs_l = vec![0f32; n];
        let mut coeffs_r = vec![0f32; n];
        for (ch, pcm_ch, coeffs) in [
            (0usize, &l_pcm, &mut coeffs_l),
            (1usize, &r_pcm, &mut coeffs_r),
        ] {
            let mut raw = vec![0f32; 2 * n];
            raw[..OVERLAP].copy_from_slice(&self.prev_tail[ch]);
            let take = n.min(pcm_ch.len());
            raw[OVERLAP..OVERLAP + take].copy_from_slice(&pcm_ch[..take]);
            self.prev_tail[ch].copy_from_slice(&pcm_ch[frame_samples - OVERLAP..]);
            crate::mdct::window_forward(&mut raw, &WINDOW_120, n, OVERLAP);
            if transient {
                crate::mdct::forward_mdct_short(&raw, coeffs, n, lm as usize, &self.short_window);
            } else {
                forward_mdct(&raw, coeffs);
            }
        }

        // Per-band log-energy and normalised shape, per channel.
        let mut band_log_e = vec![0f32; NB_EBANDS * channels];
        let mut shape_l = vec![0f32; n];
        let mut shape_r = vec![0f32; n];
        for i in 0..NB_EBANDS {
            let lo = (m * EBAND_5MS[i] as i32) as usize;
            let hi = (m * EBAND_5MS[i + 1] as i32) as usize;
            for (ch, coeffs, shape) in [
                (0usize, &coeffs_l, &mut shape_l),
                (1usize, &coeffs_r, &mut shape_r),
            ] {
                let mut e: f32 = 0.0;
                for &c in &coeffs[lo..hi] {
                    e += c * c;
                }
                let e = e.max(1e-30).sqrt();
                band_log_e[ch * NB_EBANDS + i] = e.log2() - E_MEANS[i];
                for c in &mut shape[lo..hi] {
                    *c /= e;
                }
            }
        }

        // Range-code the frame.
        // Intra-vs-inter decision: first frame forced intra (zero old state),
        // subsequent frames use inter prediction. See mono path for details.
        let intra = self.first_frame;
        let bytes = self.bytes_per_frame;
        let mut rc = RangeEncoder::new(bytes as u32);
        // Header: silence=0, optional post-filter (test-only hook),
        // transient=<flag>, intra=<flag>.
        encode_header(&mut rc, false, self.force_post_filter, transient, intra);

        // Coarse energy (both channels).
        let mut new_log_e = vec![0f32; NB_EBANDS * 2];
        new_log_e[..NB_EBANDS * channels].copy_from_slice(&band_log_e);
        let old_before = self.old_band_e.clone();
        let mut old_e_bands = old_before.clone();
        quant_coarse_energy(
            &mut rc,
            &new_log_e,
            &mut old_e_bands,
            0,
            NB_EBANDS,
            intra,
            channels,
            lm as usize,
        );

        // tf_decode: pick deltas + tf_select so the decoder's TF_SELECT_TABLE
        // lookup yields tf_res = 0 everywhere. See mono path for the logic.
        let (tf_delta_first, tf_delta_rest) = if transient {
            (true, false)
        } else {
            (false, false)
        };
        let tf_sel: bool = false;
        let budget = (bytes * 8) as u32;
        let mut tell_u = rc.tell() as u32;
        let mut logp: u32 = if transient { 2 } else { 4 };
        let tf_select_rsv = if lm > 0 && tell_u + logp + 1 <= budget {
            1
        } else {
            0
        };
        let budget_after = budget - tf_select_rsv;
        for band_i in 0..NB_EBANDS {
            if tell_u + logp <= budget_after {
                let bit = if band_i == 0 {
                    tf_delta_first
                } else {
                    tf_delta_rest
                };
                rc.encode_bit_logp(bit, logp);
                tell_u = rc.tell() as u32;
            }
            logp = if transient { 4 } else { 5 };
        }
        if tf_select_rsv != 0
            && crate::tables::TF_SELECT_TABLE[lm as usize][4 * transient as usize]
                != crate::tables::TF_SELECT_TABLE[lm as usize][4 * transient as usize + 2]
        {
            rc.encode_bit_logp(tf_sel, 1);
        }
        let tf_res = vec![0i32; NB_EBANDS];

        // Spread decision.
        let mut tell = rc.tell();
        let total_bits_check = (bytes * 8) as i32;
        if tell + 4 <= total_bits_check {
            rc.encode_icdf(SPREAD_NORMAL as usize, &SPREAD_ICDF, 5);
        }

        // Dynalloc: emit all zeros.
        let cap = init_caps(lm as usize, channels);
        let offsets = [0i32; NB_EBANDS];
        let dynalloc_logp = 6i32;
        let total_bits_frac = (bytes as i32) * 8 << BITRES;
        tell = rc.tell_frac() as i32;
        for i in 0..NB_EBANDS {
            let _width = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32 * m;
            let dynalloc_loop_logp = dynalloc_logp;
            let boost = 0i32;
            if tell + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
                rc.encode_bit_logp(false, dynalloc_loop_logp as u32);
                tell = rc.tell_frac() as i32;
            }
        }

        // Allocation trim.
        if tell + (6 << BITRES) <= total_bits_frac {
            rc.encode_icdf(5, &TRIM_ICDF, 7);
        }

        let mut bits = ((bytes as i32) * 8 << BITRES) - rc.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
            1 << BITRES
        } else {
            0i32
        };
        bits -= anti_collapse_rsv;

        let mut pulses = vec![0i32; NB_EBANDS];
        let mut fine_quant = vec![0i32; NB_EBANDS];
        let mut fine_priority = vec![0i32; NB_EBANDS];
        let mut balance = 0i32;
        let mut intensity = 0i32;
        let mut dual_stereo = 0i32;
        let coded_bands = clt_compute_allocation_enc(
            0,
            NB_EBANDS,
            &offsets,
            &cap,
            5,
            &mut intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            channels as i32,
            lm,
            &mut rc,
        );
        debug_assert_eq!(dual_stereo, 1);
        debug_assert_eq!(intensity as usize, coded_bands);

        // Fine energy (both channels).
        quant_fine_energy(
            &mut rc,
            &new_log_e,
            &mut old_e_bands,
            0,
            NB_EBANDS,
            &fine_quant,
            channels,
        );

        // PVQ shape — dual-stereo.
        let total_pvq_bits = (bytes as i32) * (8 << BITRES) - anti_collapse_rsv;
        let mut collapse_masks = vec![0u8; NB_EBANDS * channels];
        let mut rng_local = 0u32;
        encode_all_bands_stereo_dual(
            0,
            NB_EBANDS,
            &mut shape_l,
            &mut shape_r,
            &mut collapse_masks,
            &pulses,
            transient,
            SPREAD_NORMAL,
            &tf_res,
            total_pvq_bits,
            balance,
            &mut rc,
            lm,
            coded_bands,
            &mut rng_local,
        );

        // Anti-collapse-on flag = 0 (we don't run anti-collapse on encode).
        if anti_collapse_rsv > 0 {
            rc.encode_bits(0, 1);
        }

        // Final fine-energy pass.
        let bits_left = (bytes as i32) * 8 - rc.tell();
        quant_energy_finalise(
            &mut rc,
            &new_log_e,
            &mut old_e_bands,
            0,
            NB_EBANDS,
            &fine_quant,
            &fine_priority,
            bits_left,
            channels,
        );

        // Commit energy state for next frame.
        self.old_band_e = old_e_bands;
        self.first_frame = false;

        let buf = rc.done()?;
        let tb = TimeBase::new(1, SAMPLE_RATE as i64);
        let pts = self.pts_counter;
        self.pts_counter += frame_samples as i64;
        Ok(Packet::new(0, tb, buf)
            .with_pts(pts)
            .with_duration(frame_samples as i64))
    }
}

impl Encoder for CeltEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let audio = match frame {
            Frame::Audio(a) => a,
            _ => {
                return Err(Error::invalid(
                    "CELT encoder: expected audio frame, got video",
                ))
            }
        };
        let format = self.params.sample_format.ok_or_else(|| {
            Error::invalid("CELT encoder: stream sample_format missing on CodecParameters")
        })?;
        let samples = extract_interleaved_f32(audio, self.channels, format)?;
        self.pending.extend(samples);
        self.drain_frames()
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.output.pop_front() {
            Ok(p)
        } else {
            Err(Error::NeedMore)
        }
    }

    fn flush(&mut self) -> Result<()> {
        // Pad with zeros to a frame boundary, then drain.
        let frame_bytes = self.frame_samples * self.channels;
        if !self.pending.is_empty() {
            let rem = frame_bytes - (self.pending.len() % frame_bytes);
            if rem != frame_bytes {
                self.pending.extend(std::iter::repeat(0.0f32).take(rem));
            }
            self.drain_frames()?;
        }
        Ok(())
    }
}

/// Convert the `AudioFrame`'s samples to `Vec<f32>`, always interleaved
/// (L,R,L,R,... for stereo). Supports F32, F32P, S16, S16P.
/// `channels` is the expected channel count; planar formats build the
/// interleaved result by reading `channels` separate data planes.
/// `format` is the stream's sample format from `CodecParameters`.
fn extract_interleaved_f32(
    audio: &AudioFrame,
    channels: usize,
    format: SampleFormat,
) -> Result<Vec<f32>> {
    let n = audio.samples as usize;
    let mut out = vec![0f32; n * channels];
    match format {
        SampleFormat::F32 => {
            let bytes = &audio.data[0];
            if bytes.len() < n * channels * 4 {
                return Err(Error::invalid("CELT encoder: F32 input too short"));
            }
            for i in 0..n * channels {
                let b = &bytes[i * 4..i * 4 + 4];
                out[i] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        SampleFormat::F32P => {
            // Planar: one plane per channel. Interleave.
            if audio.data.len() < channels {
                return Err(Error::invalid("CELT encoder: F32P missing channel planes"));
            }
            for ch in 0..channels {
                let bytes = &audio.data[ch];
                if bytes.len() < n * 4 {
                    return Err(Error::invalid("CELT encoder: F32P input too short"));
                }
                for i in 0..n {
                    let b = &bytes[i * 4..i * 4 + 4];
                    out[i * channels + ch] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
        }
        SampleFormat::S16 => {
            let bytes = &audio.data[0];
            if bytes.len() < n * channels * 2 {
                return Err(Error::invalid("CELT encoder: S16 input too short"));
            }
            for i in 0..n * channels {
                let s = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                out[i] = s as f32 / 32768.0;
            }
        }
        SampleFormat::S16P => {
            if audio.data.len() < channels {
                return Err(Error::invalid("CELT encoder: S16P missing channel planes"));
            }
            for ch in 0..channels {
                let bytes = &audio.data[ch];
                if bytes.len() < n * 2 {
                    return Err(Error::invalid("CELT encoder: S16P input too short"));
                }
                for i in 0..n {
                    let s = i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    out[i * channels + ch] = s as f32 / 32768.0;
                }
            }
        }
        other => {
            return Err(Error::unsupported(format!(
                "CELT encoder: sample format {:?} not supported",
                other
            )));
        }
    }
    let _ = MediaType::Audio;
    Ok(out)
}

pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Ok(Box::new(CeltEncoder::new(params)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CELT encoder and drive it with one frame of pure silence — the
    /// encoder should emit the silence-flag fast path (RFC 6716 §4.3 Table 56),
    /// so the decoded header signals silence and skips every other symbol.
    #[test]
    fn silence_frame_produces_silence_flag() {
        let mut p = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        p.channels = Some(1);
        p.sample_rate = Some(SAMPLE_RATE);
        let mut enc = CeltEncoder::new(&p).unwrap();
        let pcm = vec![0.0f32; FRAME_SAMPLES];
        let pkt = enc.encode_frame(&pcm).unwrap();
        assert!(!pkt.data.is_empty());
        // Header should decode as silence (None).
        let mut rd = crate::range_decoder::RangeDecoder::new(&pkt.data);
        let h = crate::header::decode_header(&mut rd);
        assert!(
            h.is_none(),
            "silence frame must set the silence flag (header returns None)"
        );
    }

    /// A near-silent frame (all samples below -90 dBFS) must also trip the
    /// silence flag, not the full PVQ path.
    #[test]
    fn near_silence_frame_produces_silence_flag() {
        let mut p = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        p.channels = Some(1);
        p.sample_rate = Some(SAMPLE_RATE);
        let mut enc = CeltEncoder::new(&p).unwrap();
        let pcm = vec![1e-6f32; FRAME_SAMPLES];
        let pkt = enc.encode_frame(&pcm).unwrap();
        let mut rd = crate::range_decoder::RangeDecoder::new(&pkt.data);
        assert!(crate::header::decode_header(&mut rd).is_none());
    }

    /// A non-silent (sine) frame must follow the normal §4.3 pipeline —
    /// silence=false, transient=false, intra=true on first frame.
    #[test]
    fn sine_frame_skips_silence_flag() {
        let mut p = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        p.channels = Some(1);
        p.sample_rate = Some(SAMPLE_RATE);
        let mut enc = CeltEncoder::new(&p).unwrap();
        let pcm: Vec<f32> = (0..FRAME_SAMPLES)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.3
            })
            .collect();
        let pkt = enc.encode_frame(&pcm).unwrap();
        let mut rd = crate::range_decoder::RangeDecoder::new(&pkt.data);
        let h = crate::header::decode_header(&mut rd).expect("non-silent header should parse");
        assert!(!h.silence);
        assert!(!h.transient);
        assert!(h.intra, "first frame of a session must be intra");
    }

    /// Second frame onward must advertise `intra=false` — inter-frame energy
    /// prediction kicks in as soon as there's valid prior state.
    #[test]
    fn second_frame_uses_inter_prediction() {
        let mut p = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        p.channels = Some(1);
        p.sample_rate = Some(SAMPLE_RATE);
        let mut enc = CeltEncoder::new(&p).unwrap();
        // Frame 1: intra.
        let pcm1: Vec<f32> = (0..FRAME_SAMPLES)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.3
            })
            .collect();
        let pkt1 = enc.encode_frame(&pcm1).unwrap();
        // Frame 2: should now be inter.
        let pcm2: Vec<f32> = (FRAME_SAMPLES..2 * FRAME_SAMPLES)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.3
            })
            .collect();
        let pkt2 = enc.encode_frame(&pcm2).unwrap();

        let mut rd1 = crate::range_decoder::RangeDecoder::new(&pkt1.data);
        let h1 = crate::header::decode_header(&mut rd1).unwrap();
        assert!(h1.intra, "frame 1 must be intra");

        let mut rd2 = crate::range_decoder::RangeDecoder::new(&pkt2.data);
        let h2 = crate::header::decode_header(&mut rd2).unwrap();
        assert!(!h2.intra, "frame 2 must be inter");
    }

    // -----------------------------------------------------------------
    // detect_transient — signal-shape unit tests.
    //
    // Each test pins one expected behaviour of the detector:
    //   * silence stays long
    //   * pure sine stays long
    //   * gradual fade-in (slow envelope) stays long — this is the case
    //     the median-vs-min refactor fixed; under the old peak/min
    //     ratio a fade-in tripped the detector
    //   * sharp click against silence flips to short
    //   * white noise (uniform energy) stays long
    //   * threshold parameter actually does what the docstring claims
    // -----------------------------------------------------------------

    /// Pure silence must never be flagged transient — the encoder's
    /// silence-fast-path picks it up first anyway, but the detector
    /// itself should also bail gracefully without dividing by zero.
    #[test]
    fn detect_transient_silence_returns_false() {
        let pcm = vec![0f32; FRAME_SAMPLES];
        assert!(!detect_transient(&pcm));
    }

    /// A clean sine has uniform per-sub-block energy → detector must
    /// keep `transient=false`.
    #[test]
    fn detect_transient_sine_returns_false() {
        let pcm: Vec<f32> = (0..FRAME_SAMPLES)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.3
            })
            .collect();
        assert!(!detect_transient(&pcm));
    }

    /// A linear fade-in from silence to full scale is exactly the case
    /// the old peak/min detector tripped on — sub-block 0 is near-silent
    /// (peak/min → ∞ dB) but the actual signal is steady-state. The
    /// median-based replacement must recognise this is a slow envelope,
    /// not a transient.
    #[test]
    fn detect_transient_slow_fade_in_returns_false() {
        let pcm: Vec<f32> = (0..FRAME_SAMPLES)
            .map(|i| {
                let env = i as f32 / FRAME_SAMPLES as f32;
                env * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin()
                    * 0.3
            })
            .collect();
        assert!(
            !detect_transient(&pcm),
            "slow fade-in must not flip detector; this is the burst-vs-envelope distinction"
        );
    }

    /// A sharp click in the middle of an otherwise-silent frame is the
    /// canonical "transient" stimulus. Detector must flip to true.
    #[test]
    fn detect_transient_click_returns_true() {
        let mut pcm = vec![0f32; FRAME_SAMPLES];
        // ~2.5 ms burst at 0.7 amplitude in the middle of the frame.
        for i in FRAME_SAMPLES / 2..FRAME_SAMPLES / 2 + 120 {
            pcm[i] = 0.7;
        }
        assert!(
            detect_transient(&pcm),
            "click in the middle of a silent frame must trigger transient"
        );
    }

    /// White noise has roughly equal sub-block energy → no transient.
    #[test]
    fn detect_transient_white_noise_returns_false() {
        let mut seed: u32 = 0x42424242;
        let pcm: Vec<f32> = (0..FRAME_SAMPLES)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 16) as i32 - 32768) as f32 / 32768.0 * 0.1
            })
            .collect();
        assert!(!detect_transient(&pcm));
    }

    /// Lowering the threshold to 6 dB must allow even modest envelope
    /// modulation to trip the detector. This pins the
    /// `detect_transient_with_threshold` parameter actually doing its
    /// job — useful for callers that want a more aggressive detector.
    #[test]
    fn detect_transient_threshold_param_takes_effect() {
        // 6 dB amplitude bump in one sub-block.
        let mut pcm = vec![0.1f32; FRAME_SAMPLES];
        for i in 0..FRAME_SAMPLES / 8 {
            pcm[i] = 0.2; // sub-block 0 has 4x the energy of the rest
        }
        // Default threshold (15 dB) should not trip on a 6 dB delta.
        assert!(!detect_transient(&pcm));
        // Lowered threshold (3 dB) should.
        assert!(detect_transient_with_threshold(&pcm, 3.0));
    }

    /// End-to-end: feeding a sharp click through the encoder must
    /// produce at least one packet whose header carries `transient=1`.
    /// Mirrors the public_api_roundtrip test but at the encoder unit-
    /// test level, with no decoder dependency.
    #[test]
    fn encoder_emits_transient_flag_on_click_signal() {
        let mut p = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        p.channels = Some(1);
        p.sample_rate = Some(SAMPLE_RATE);
        let mut enc = CeltEncoder::new(&p).unwrap();
        let mut pcm = vec![0f32; FRAME_SAMPLES];
        for i in 200..320 {
            pcm[i] = 0.7;
        }
        let pkt = enc.encode_frame(&pcm).unwrap();
        let mut rd = crate::range_decoder::RangeDecoder::new(&pkt.data);
        let h = crate::header::decode_header(&mut rd).expect("non-silent header should parse");
        assert!(
            h.transient,
            "encoder must emit transient=1 on a clicky frame"
        );
    }

    /// Conversely, a clean steady-state sine through the encoder must
    /// always come out as `transient=0` — false positives waste short-
    /// block bits on signals that don't need them.
    #[test]
    fn encoder_keeps_transient_false_on_steady_sine() {
        let mut p = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        p.channels = Some(1);
        p.sample_rate = Some(SAMPLE_RATE);
        let mut enc = CeltEncoder::new(&p).unwrap();
        // 8 frames of 1 kHz sine.
        for f in 0..8 {
            let pcm: Vec<f32> = (0..FRAME_SAMPLES)
                .map(|i| {
                    let n = (f * FRAME_SAMPLES + i) as f32;
                    (2.0 * std::f32::consts::PI * 1000.0 * n / SAMPLE_RATE as f32).sin() * 0.3
                })
                .collect();
            let pkt = enc.encode_frame(&pcm).unwrap();
            let mut rd = crate::range_decoder::RangeDecoder::new(&pkt.data);
            let h = crate::header::decode_header(&mut rd).unwrap();
            assert!(
                !h.transient,
                "frame {f} of a steady sine must NOT be flagged transient"
            );
        }
    }

    /// `set_force_long_only(true)` must override even the strongest
    /// transient stimulus. Pins the A/B-test escape hatch.
    #[test]
    fn force_long_only_overrides_detector() {
        let mut p = CodecParameters::audio(CodecId::new(crate::CODEC_ID_STR));
        p.channels = Some(1);
        p.sample_rate = Some(SAMPLE_RATE);
        let mut enc = CeltEncoder::new(&p).unwrap();
        enc.set_force_long_only(true);
        let mut pcm = vec![0f32; FRAME_SAMPLES];
        for i in 200..320 {
            pcm[i] = 0.7;
        }
        let pkt = enc.encode_frame(&pcm).unwrap();
        let mut rd = crate::range_decoder::RangeDecoder::new(&pkt.data);
        let h = crate::header::decode_header(&mut rd).unwrap();
        assert!(
            !h.transient,
            "force_long_only must clamp transient=false even on click input"
        );
    }
}
