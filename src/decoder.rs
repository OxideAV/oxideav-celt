//! CELT decoder — 48 kHz, full-band, 20 ms (LM=3, 960 samples) or 10 ms
//! (LM=2, 480 samples) frames.
//!
//! Default constructor [`CeltDecoder::new`] selects the historical 20 ms
//! / LM=3 path. Pair it with the matching encoder constructor —
//! [`crate::encoder::CeltEncoder::new`]. For the 10 ms path used by
//! Opus 10 ms Hybrid (RFC 6716 Table 2 configs 6/8/10), use
//! [`CeltDecoder::new_with_frame_samples`] with `480` and
//! [`crate::encoder::CeltEncoder::new_with_frame_samples`] in lock-step.
//!
//! Scope matches what the in-crate encoder emits (RFC 6716 §4.3): mono or
//! dual-stereo, both long and short (transient) blocks, with comb-filter
//! post-filter parsing on packets that carry it. A libopus-produced
//! CELT packet may still exercise paths this decoder does not yet cover
//! (intensity stereo, dynalloc band boosts other than zero, sample rates
//! other than 48 kHz). Those paths take libopus' "decode whatever the
//! header says" approach but the dispatch assumes the encoder's profile;
//! mismatches surface as drifted band-energy state rather than a hard
//! error, which callers can detect via the `peer-was-libopus` flags
//! upstream.
//!
//! For the full Opus decoder (SILK + CELT + hybrid + range-coder framing),
//! use the `oxideav-opus` crate, which dispatches into these same modules.

use std::collections::VecDeque;

use oxideav_core::Decoder;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::bands::{anti_collapse, denormalise_bands, quant_all_bands};
use crate::header::decode_header;
use crate::mdct::{imdct_sub, imdct_sub_short};
use crate::post_filter::{comb_filter, decode_pitch_gain, decode_pitch_period, deemphasis};
use crate::quant_bands::{unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy};
use crate::range_decoder::{RangeDecoder, BITRES};
use crate::rate::clt_compute_allocation;
use crate::tables::{
    init_caps, lm_for_frame_samples, COMB_FILTER_MAXPERIOD, EBAND_5MS, NB_EBANDS, SPREAD_ICDF,
    SPREAD_NORMAL, TF_SELECT_TABLE, TRIM_ICDF,
};

/// Default CELT frame length (LM=3): 960 samples = 20 ms at 48 kHz.
/// 10 ms (LM=2, 480 samples) is also supported via
/// [`CeltDecoder::new_with_frame_samples`]; this constant remains the
/// historical default for callers using [`CeltDecoder::new`].
pub const FRAME_SAMPLES: usize = 960;
pub const SAMPLE_RATE: u32 = 48_000;
const OVERLAP: usize = 120;
/// Internal MDCT coefficient count at the default LM=3:
/// `EBAND_5MS[21] * M = 100 * 8 = 800`. The long-block IMDCT produces
/// 2N=1600 time-domain samples; the remaining `2*(FRAME_SAMPLES - CODED_N)
/// = 320` at the far edges of the true 1920-sample long window are
/// zero-filled and contribute nothing, matching the encoder's convention.
/// At LM=2 the equivalent is `100 * 4 = 400` (computed per-instance).
const CODED_N: usize = 800;
/// Short sub-block coefficient count at the default LM=3, i.e.
/// `CODED_N / M = 800 / 8`. Each short sub-block runs a 200→100 MDCT with
/// a 100-sample cosine window. LM=2 produces the same value
/// (`400 / 4 = 100`).
const SHORT_N: usize = 100;

/// Build the CELT-style short-block cosine window of length `len`. Uses the
/// Vorbis/CELT sin-sin window formula — the same shape as `WINDOW_120` but
/// scaled to an arbitrary length. Perfect reconstruction of adjacent 50%-
/// overlapped sub-blocks relies on `w[i]^2 + w[len-1-i]^2 = 1`.
fn build_short_window(len: usize) -> Vec<f32> {
    let mut w = vec![0f32; len];
    for i in 0..len {
        let base = std::f32::consts::FRAC_PI_2 * (i as f32 + 0.5) / len as f32;
        let s = base.sin();
        w[i] = (std::f32::consts::FRAC_PI_2 * s * s).sin();
    }
    w
}

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

pub struct CeltDecoder {
    params: CodecParameters,
    channels: usize,
    /// Per-instance CELT frame size in 48 kHz samples (480 or 960). Drives
    /// the IMDCT length, coded bin count and short-block sub-MDCT layout.
    frame_samples: usize,
    /// Per-instance coded bin count = `100 * M = 100 * (1 << lm)`. 400 at
    /// LM=2, 800 at LM=3.
    coded_n: usize,
    /// Per-instance short-block sub-MDCT coefficient count = `coded_n / M`
    /// = 100 at both supported LMs.
    short_n: usize,
    /// Per-instance LM = `log2(frame_samples / 120)`. 2 or 3.
    lm: i32,
    old_band_e: Vec<f32>,
    /// Two frames of prior quantised log-energy, used by §4.3.5 anti-collapse.
    /// Layout: `channels * NB_EBANDS` per slot.
    prev1_log_e: Vec<f32>,
    prev2_log_e: Vec<f32>,
    prev_tail: Vec<Vec<f32>>,
    rng: u32,
    output: VecDeque<Frame>,
    pts_counter: i64,
    short_window: Vec<f32>,
    /// Comb post-filter state, per RFC 6716 §4.3.7.1. One slot per channel.
    /// `pf_period_old[c] / pf_gain_old[c] / pf_tapset_old[c]` are the
    /// parameters from the *previous* decoded frame — they drive the
    /// crossfade at the start of the current frame. Fresh decoders start at
    /// (0, 0.0, 0) = "no post-filter in history".
    pf_period_old: Vec<i32>,
    pf_gain_old: Vec<f32>,
    pf_tapset_old: Vec<usize>,
    /// Per-channel comb-filter history buffer: the last
    /// `COMB_FILTER_MAXPERIOD + 2` post-filtered output samples from prior
    /// frames, used by `comb_filter` for the `y[n - T]` lookbacks at the
    /// start of the current frame.
    pf_history: Vec<Vec<f32>>,
    /// De-emphasis filter state (`y[n-1]`) per channel (RFC §4.3.7.2).
    deemph_state: Vec<f32>,
}

impl CeltDecoder {
    /// Construct a default-frame-size (LM=3, 960-sample / 20 ms) CELT
    /// decoder. Equivalent to
    /// [`CeltDecoder::new_with_frame_samples(params, 960)`].
    pub fn new(params: &CodecParameters) -> Result<Self> {
        Self::new_with_frame_samples(params, FRAME_SAMPLES)
    }

    /// Construct a CELT decoder for an explicit frame size. Supported
    /// values are `960` (LM=3, 20 ms) and `480` (LM=2, 10 ms). All other
    /// sizes return [`Error::unsupported`]. Pair this with the matching
    /// encoder constructor —
    /// [`crate::encoder::CeltEncoder::new_with_frame_samples`] — and feed
    /// the decoder packets emitted by an encoder of the same frame size.
    /// The 10 ms path is the CELT-MDCT length used by the Opus 10 ms
    /// Hybrid configuration (RFC 6716 Table 2 configs 6/8/10).
    pub fn new_with_frame_samples(params: &CodecParameters, frame_samples: usize) -> Result<Self> {
        let channels = params.channels.unwrap_or(1) as usize;
        if channels != 1 && channels != 2 {
            return Err(Error::unsupported(
                "CELT decoder: only mono (1) and stereo (2) channels are supported",
            ));
        }
        let sr = params.sample_rate.unwrap_or(SAMPLE_RATE);
        if sr != SAMPLE_RATE {
            return Err(Error::unsupported("CELT decoder: only 48 kHz is supported"));
        }
        if frame_samples != 960 && frame_samples != 480 {
            return Err(Error::unsupported(
                "CELT decoder: frame_samples must be 480 (LM=2, 10 ms) or 960 (LM=3, 20 ms)",
            ));
        }
        let lm = lm_for_frame_samples(frame_samples as u32) as i32;
        let coded_n = 100usize << (lm as usize);
        let short_n = coded_n >> (lm as usize);
        let mut out_params = params.clone();
        out_params.channels = Some(channels as u16);
        out_params.sample_rate = Some(SAMPLE_RATE);
        let pf_hist_len = COMB_FILTER_MAXPERIOD as usize + 2;
        Ok(Self {
            params: out_params,
            channels,
            frame_samples,
            coded_n,
            short_n,
            lm,
            old_band_e: vec![0.0; NB_EBANDS * 2],
            prev1_log_e: vec![0.0; NB_EBANDS * 2],
            prev2_log_e: vec![0.0; NB_EBANDS * 2],
            prev_tail: vec![vec![0.0; OVERLAP]; channels],
            rng: 0,
            output: VecDeque::new(),
            pts_counter: 0,
            short_window: build_short_window(short_n),
            pf_period_old: vec![0; channels],
            pf_gain_old: vec![0.0; channels],
            pf_tapset_old: vec![0; channels],
            pf_history: vec![vec![0.0; pf_hist_len]; channels],
            deemph_state: vec![0.0; channels],
        })
    }

    /// Per-instance CELT frame length in 48 kHz samples (480 or 960).
    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    fn decode_frame_mono(&mut self, bytes: &[u8]) -> Result<Vec<f32>> {
        let mut rc = RangeDecoder::new(bytes);
        let lm = self.lm;
        let frame_samples = self.frame_samples;
        let end_band = NB_EBANDS;
        let start_band = 0usize;

        let header = match decode_header(&mut rc) {
            Some(h) => h,
            None => {
                // silence flag was set — emit zeros, advance OLA state so the
                // next frame's overlap lands on actual zeros.
                for v in &mut self.prev_tail[0] {
                    *v = 0.0;
                }
                // Roll prev1/prev2 log-energy for §4.3.5 anti-collapse state.
                self.prev2_log_e.copy_from_slice(&self.prev1_log_e);
                self.prev1_log_e.copy_from_slice(&self.old_band_e);
                // Silence still ages the post-filter state: no new params,
                // next frame treats this frame's silent tail as history. We
                // keep `pf_*_old` (which govern this frame's crossfade) and
                // slide the comb-filter history forward by one frame
                // of zeros. Deemphasis state decays to zero naturally.
                let zeros = vec![0.0f32; frame_samples];
                let mut out = zeros.clone();
                self.run_postfilter_and_deemph(0, &mut out, 0, 0.0, 0);
                return Ok(out);
            }
        };
        let transient = header.transient;
        // Current-frame post-filter params (None → zero gain, no filter).
        let (pf_t1, pf_g1, pf_tapset1) = match header.post_filter {
            Some(pf) => (
                decode_pitch_period(pf.octave, pf.period) as i32,
                decode_pitch_gain(pf.gain),
                pf.tapset as usize,
            ),
            None => (0i32, 0.0f32, 0usize),
        };

        let m = 1i32 << lm;
        let n = (m * EBAND_5MS[NB_EBANDS] as i32) as usize;

        unquant_coarse_energy(
            &mut rc,
            &mut self.old_band_e,
            start_band,
            end_band,
            header.intra,
            1,
            lm as usize,
        );

        let budget = rc.storage() * 8;
        let mut tell_u = rc.tell() as u32;
        let mut logp: u32 = if transient { 2 } else { 4 };
        let tf_select_rsv = if lm > 0 && tell_u + logp < budget {
            1
        } else {
            0
        };
        let budget_after = budget - tf_select_rsv;
        let mut tf_res = vec![0i32; NB_EBANDS];
        let mut tf_changed = 0i32;
        let mut curr = 0i32;
        for i in start_band..end_band {
            if tell_u + logp <= budget_after {
                let bit = rc.decode_bit_logp(logp);
                curr ^= bit as i32;
                tell_u = rc.tell() as u32;
                tf_changed |= curr;
            }
            tf_res[i] = curr;
            logp = if transient { 4 } else { 5 };
        }
        let mut tf_select = 0i32;
        if tf_select_rsv != 0
            && TF_SELECT_TABLE[lm as usize][4 * transient as usize + tf_changed as usize]
                != TF_SELECT_TABLE[lm as usize][4 * transient as usize + 2 + tf_changed as usize]
        {
            tf_select = if rc.decode_bit_logp(1) { 1 } else { 0 };
        }
        for i in start_band..end_band {
            let idx = (4 * transient as i32 + 2 * tf_select + tf_res[i]) as usize;
            tf_res[i] = TF_SELECT_TABLE[lm as usize][idx] as i32;
        }

        let mut tell = rc.tell();
        let total_bits_check = (rc.storage() * 8) as i32;
        let spread = if tell + 4 <= total_bits_check {
            rc.decode_icdf(&SPREAD_ICDF, 5) as i32
        } else {
            SPREAD_NORMAL
        };

        let cap = init_caps(lm as usize, 1);
        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_frac = ((bytes.len() as i32) * 8) << BITRES;
        tell = rc.tell_frac() as i32;
        for i in start_band..end_band {
            let width = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32 * m;
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
                let flag = rc.decode_bit_logp(dynalloc_loop_logp as u32);
                tell = rc.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits_frac -= quanta;
                dynalloc_loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
        }

        let alloc_trim = if tell + (6 << BITRES) <= total_bits_frac {
            rc.decode_icdf(&TRIM_ICDF, 7) as i32
        } else {
            5
        };

        // Allocation residual (fractional bits). Anti-collapse reservation
        // (RFC §4.3.5) takes 1 bit when transient && LM>=2 && bits big enough.
        let mut bits = (((bytes.len() as i32) * 8) << BITRES) - rc.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
            1 << BITRES
        } else {
            0i32
        };
        bits -= anti_collapse_rsv;

        let mut pulses = vec![0i32; NB_EBANDS];
        let mut fine_quant = vec![0i32; NB_EBANDS];
        let mut fine_priority = vec![0i32; NB_EBANDS];
        let mut intensity = 0i32;
        let mut dual_stereo = 0i32;
        let mut balance = 0i32;
        let coded_bands = clt_compute_allocation(
            start_band,
            end_band,
            &offsets,
            &cap,
            alloc_trim,
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

        unquant_fine_energy(
            &mut rc,
            &mut self.old_band_e,
            start_band,
            end_band,
            &fine_quant,
            1,
        );

        let mut x_buf = vec![0f32; n];
        let mut collapse_masks = vec![0u8; NB_EBANDS];
        let total_pvq_bits = ((bytes.len() as i32) * 8 << BITRES) - anti_collapse_rsv;
        let band_e_snapshot = self.old_band_e.clone();
        quant_all_bands(
            start_band,
            end_band,
            &mut x_buf,
            None,
            &mut collapse_masks,
            &band_e_snapshot,
            &pulses,
            transient,
            spread,
            dual_stereo,
            intensity,
            &tf_res,
            total_pvq_bits,
            balance,
            &mut rc,
            lm,
            coded_bands,
            &mut self.rng,
            false,
        );

        // Anti-collapse ON flag (1 range-coded bit when reserved). §4.3.5.
        let anti_collapse_on = if anti_collapse_rsv > 0 {
            rc.decode_bits(1) != 0
        } else {
            false
        };

        let bits_left = (bytes.len() as i32) * 8 - rc.tell();
        unquant_energy_finalise(
            &mut rc,
            &mut self.old_band_e,
            start_band,
            end_band,
            &fine_quant,
            &fine_priority,
            bits_left,
            1,
        );

        if anti_collapse_on {
            anti_collapse(
                &mut x_buf,
                &collapse_masks,
                lm,
                1,
                n,
                start_band,
                end_band,
                &self.old_band_e,
                &self.prev1_log_e,
                &self.prev2_log_e,
                &pulses,
                self.rng,
            );
        }

        let mut freq = vec![0f32; n];
        denormalise_bands(
            &x_buf,
            &mut freq,
            &self.old_band_e[..NB_EBANDS],
            start_band,
            end_band,
            m as usize,
            false,
        );

        let sub_n = n;
        let mut raw = vec![0f32; 2 * frame_samples];
        if transient {
            // M × short_n-point IMDCTs interleaved into `freq`, then OLA
            // across the M sub-blocks into `raw`. The caller-side front
            // and back overlap with the previous/next frame uses the long
            // window (120 taps).
            let mut raw_coded = vec![0f32; 2 * sub_n];
            imdct_sub_short(
                &freq,
                &mut raw_coded,
                sub_n,
                lm as usize,
                &self.short_window,
            );
            raw[..2 * sub_n].copy_from_slice(&raw_coded);
        } else {
            let mut raw_coded = vec![0f32; 2 * sub_n];
            imdct_sub(&freq, &mut raw_coded, sub_n);
            raw[..2 * sub_n].copy_from_slice(&raw_coded);
        }

        let mut out = vec![0f32; frame_samples];
        for i in 0..OVERLAP {
            let w = WINDOW_120[i];
            out[i] = self.prev_tail[0][i] + w * raw[i];
        }
        for i in OVERLAP..frame_samples {
            out[i] = raw[i];
        }
        for i in 0..OVERLAP {
            let w = WINDOW_120[OVERLAP - 1 - i];
            self.prev_tail[0][i] = w * raw[frame_samples + i];
        }
        // Post-filter (RFC §4.3.7.1) + de-emphasis (§4.3.7.2), and roll
        // comb-filter history + deemph state.
        self.run_postfilter_and_deemph(0, &mut out, pf_t1, pf_g1, pf_tapset1);
        // Roll anti-collapse log-energy state.
        self.prev2_log_e.copy_from_slice(&self.prev1_log_e);
        self.prev1_log_e.copy_from_slice(&self.old_band_e);
        Ok(out)
    }

    fn decode_frame_stereo(&mut self, bytes: &[u8]) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut rc = RangeDecoder::new(bytes);
        let lm = self.lm;
        let frame_samples = self.frame_samples;
        let end_band = NB_EBANDS;
        let start_band = 0usize;
        let channels = 2usize;

        let header = match decode_header(&mut rc) {
            Some(h) => h,
            None => {
                for v in &mut self.prev_tail[0] {
                    *v = 0.0;
                }
                for v in &mut self.prev_tail[1] {
                    *v = 0.0;
                }
                self.prev2_log_e.copy_from_slice(&self.prev1_log_e);
                self.prev1_log_e.copy_from_slice(&self.old_band_e);
                // Silence path: still age the post-filter history + deemph
                // state by running an all-zero frame through them (see the
                // mono silence path for rationale).
                let mut zl = vec![0.0f32; frame_samples];
                let mut zr = vec![0.0f32; frame_samples];
                self.run_postfilter_and_deemph(0, &mut zl, 0, 0.0, 0);
                self.run_postfilter_and_deemph(1, &mut zr, 0, 0.0, 0);
                return Ok((zl, zr));
            }
        };
        let transient = header.transient;
        // Current-frame post-filter params (mono in Opus — stereo CELT
        // shares one set across L+R, per libopus).
        let (pf_t1, pf_g1, pf_tapset1) = match header.post_filter {
            Some(pf) => (
                decode_pitch_period(pf.octave, pf.period) as i32,
                decode_pitch_gain(pf.gain),
                pf.tapset as usize,
            ),
            None => (0i32, 0.0f32, 0usize),
        };

        let m = 1i32 << lm;
        let n = (m * EBAND_5MS[NB_EBANDS] as i32) as usize;

        unquant_coarse_energy(
            &mut rc,
            &mut self.old_band_e,
            start_band,
            end_band,
            header.intra,
            channels,
            lm as usize,
        );

        let budget = rc.storage() * 8;
        let mut tell_u = rc.tell() as u32;
        let mut logp: u32 = if transient { 2 } else { 4 };
        let tf_select_rsv = if lm > 0 && tell_u + logp < budget {
            1
        } else {
            0
        };
        let budget_after = budget - tf_select_rsv;
        let mut tf_res = vec![0i32; NB_EBANDS];
        let mut tf_changed = 0i32;
        let mut curr = 0i32;
        for i in start_band..end_band {
            if tell_u + logp <= budget_after {
                let bit = rc.decode_bit_logp(logp);
                curr ^= bit as i32;
                tell_u = rc.tell() as u32;
                tf_changed |= curr;
            }
            tf_res[i] = curr;
            logp = if transient { 4 } else { 5 };
        }
        let mut tf_select = 0i32;
        if tf_select_rsv != 0
            && TF_SELECT_TABLE[lm as usize][4 * transient as usize + tf_changed as usize]
                != TF_SELECT_TABLE[lm as usize][4 * transient as usize + 2 + tf_changed as usize]
        {
            tf_select = if rc.decode_bit_logp(1) { 1 } else { 0 };
        }
        for i in start_band..end_band {
            let idx = (4 * transient as i32 + 2 * tf_select + tf_res[i]) as usize;
            tf_res[i] = TF_SELECT_TABLE[lm as usize][idx] as i32;
        }

        let mut tell = rc.tell();
        let total_bits_check = (rc.storage() * 8) as i32;
        let spread = if tell + 4 <= total_bits_check {
            rc.decode_icdf(&SPREAD_ICDF, 5) as i32
        } else {
            SPREAD_NORMAL
        };

        let cap = init_caps(lm as usize, channels);
        let mut offsets = [0i32; NB_EBANDS];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_frac = ((bytes.len() as i32) * 8) << BITRES;
        tell = rc.tell_frac() as i32;
        for i in start_band..end_band {
            let width = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32 * m * channels as i32;
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
                let flag = rc.decode_bit_logp(dynalloc_loop_logp as u32);
                tell = rc.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits_frac -= quanta;
                dynalloc_loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
        }

        let alloc_trim = if tell + (6 << BITRES) <= total_bits_frac {
            rc.decode_icdf(&TRIM_ICDF, 7) as i32
        } else {
            5
        };

        let mut bits = (((bytes.len() as i32) * 8) << BITRES) - rc.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
            1 << BITRES
        } else {
            0i32
        };
        bits -= anti_collapse_rsv;

        let mut pulses = vec![0i32; NB_EBANDS];
        let mut fine_quant = vec![0i32; NB_EBANDS];
        let mut fine_priority = vec![0i32; NB_EBANDS];
        let mut intensity = 0i32;
        let mut dual_stereo = 0i32;
        let mut balance = 0i32;
        let coded_bands = clt_compute_allocation(
            start_band,
            end_band,
            &offsets,
            &cap,
            alloc_trim,
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

        unquant_fine_energy(
            &mut rc,
            &mut self.old_band_e,
            start_band,
            end_band,
            &fine_quant,
            channels,
        );

        let mut x_buf = vec![0f32; n];
        let mut y_buf = vec![0f32; n];
        let mut collapse_masks = vec![0u8; NB_EBANDS * channels];
        let total_pvq_bits = ((bytes.len() as i32) * 8 << BITRES) - anti_collapse_rsv;
        let band_e_snapshot = self.old_band_e.clone();
        quant_all_bands(
            start_band,
            end_band,
            &mut x_buf,
            Some(&mut y_buf),
            &mut collapse_masks,
            &band_e_snapshot,
            &pulses,
            transient,
            spread,
            dual_stereo,
            intensity,
            &tf_res,
            total_pvq_bits,
            balance,
            &mut rc,
            lm,
            coded_bands,
            &mut self.rng,
            false,
        );

        let anti_collapse_on = if anti_collapse_rsv > 0 {
            rc.decode_bits(1) != 0
        } else {
            false
        };

        let bits_left = (bytes.len() as i32) * 8 - rc.tell();
        unquant_energy_finalise(
            &mut rc,
            &mut self.old_band_e,
            start_band,
            end_band,
            &fine_quant,
            &fine_priority,
            bits_left,
            channels,
        );

        if anti_collapse_on {
            // `x_buf | y_buf` concatenated as a 2*N buffer for anti-collapse's
            // per-channel indexing (chan * size + ...).
            let mut xy = vec![0f32; 2 * n];
            xy[..n].copy_from_slice(&x_buf);
            xy[n..].copy_from_slice(&y_buf);
            anti_collapse(
                &mut xy,
                &collapse_masks,
                lm,
                channels,
                n,
                start_band,
                end_band,
                &self.old_band_e,
                &self.prev1_log_e,
                &self.prev2_log_e,
                &pulses,
                self.rng,
            );
            x_buf.copy_from_slice(&xy[..n]);
            y_buf.copy_from_slice(&xy[n..]);
        }

        let mut freq_l = vec![0f32; n];
        let mut freq_r = vec![0f32; n];
        denormalise_bands(
            &x_buf,
            &mut freq_l,
            &self.old_band_e[..NB_EBANDS],
            start_band,
            end_band,
            m as usize,
            false,
        );
        denormalise_bands(
            &y_buf,
            &mut freq_r,
            &self.old_band_e[NB_EBANDS..2 * NB_EBANDS],
            start_band,
            end_band,
            m as usize,
            false,
        );

        let sub_n = n;
        let mut out_l = vec![0f32; frame_samples];
        let mut out_r = vec![0f32; frame_samples];
        for (ch, freq, out) in [(0usize, &freq_l, &mut out_l), (1usize, &freq_r, &mut out_r)] {
            let mut raw = vec![0f32; 2 * frame_samples];
            if transient {
                let mut raw_coded = vec![0f32; 2 * sub_n];
                imdct_sub_short(freq, &mut raw_coded, sub_n, lm as usize, &self.short_window);
                raw[..2 * sub_n].copy_from_slice(&raw_coded);
            } else {
                let mut raw_coded = vec![0f32; 2 * sub_n];
                imdct_sub(freq, &mut raw_coded, sub_n);
                raw[..2 * sub_n].copy_from_slice(&raw_coded);
            }
            for i in 0..OVERLAP {
                let w = WINDOW_120[i];
                out[i] = self.prev_tail[ch][i] + w * raw[i];
            }
            for i in OVERLAP..frame_samples {
                out[i] = raw[i];
            }
            for i in 0..OVERLAP {
                let w = WINDOW_120[OVERLAP - 1 - i];
                self.prev_tail[ch][i] = w * raw[frame_samples + i];
            }
        }
        // Post-filter + de-emphasis on each channel. Both channels share the
        // single pitch / gain / tapset decoded from the header.
        self.run_postfilter_and_deemph(0, &mut out_l, pf_t1, pf_g1, pf_tapset1);
        self.run_postfilter_and_deemph(1, &mut out_r, pf_t1, pf_g1, pf_tapset1);
        self.prev2_log_e.copy_from_slice(&self.prev1_log_e);
        self.prev1_log_e.copy_from_slice(&self.old_band_e);
        Ok((out_l, out_r))
    }

    /// Run the RFC 6716 §4.3.7 tail of the decode pipeline on one channel of
    /// `FRAME_SAMPLES` post-IMDCT+OLA samples. Consumes the current-frame
    /// pitch parameters `(pf_t1, pf_g1, pf_tapset1)`, uses the stored
    /// previous-frame ones as the crossfade source, writes the filtered
    /// output back in place, then:
    ///
    /// 1. Updates `pf_history[ch]` with the last `COMB_FILTER_MAXPERIOD + 2`
    ///    samples of the *post-filtered but not yet de-emphasized* output.
    /// 2. Rotates `pf_*_old[ch]` to the current-frame values.
    /// 3. Applies in-place single-pole de-emphasis, updating
    ///    `deemph_state[ch]`.
    fn run_postfilter_and_deemph(
        &mut self,
        ch: usize,
        out: &mut [f32],
        pf_t1: i32,
        pf_g1: f32,
        pf_tapset1: usize,
    ) {
        let pf_t0 = self.pf_period_old[ch];
        let pf_g0 = self.pf_gain_old[ch];
        let pf_tapset0 = self.pf_tapset_old[ch];

        // The post-filter uses the 120-tap long MDCT window for its
        // crossfade (libopus passes `mode->window`, which at LM=3 is the
        // 120-sample sin-window; this matches `WINDOW_120`).
        comb_filter(
            out,
            &self.pf_history[ch],
            pf_t0,
            pf_t1,
            out.len(),
            pf_g0,
            pf_g1,
            pf_tapset0,
            pf_tapset1,
            &WINDOW_120,
            OVERLAP,
        );

        // Update the MAX_PERIOD + 2 tail history with the trailing samples
        // of the current frame's post-filtered output. When the frame is
        // shorter than the history buffer we slide existing contents left.
        let hist_len = self.pf_history[ch].len();
        if out.len() >= hist_len {
            let start = out.len() - hist_len;
            self.pf_history[ch].copy_from_slice(&out[start..]);
        } else {
            // Shift older history left, append current frame at the end.
            let keep = hist_len - out.len();
            self.pf_history[ch].copy_within(hist_len - keep.., 0);
            // above is a no-op when keep == 0; the next copy fills the tail
            self.pf_history[ch][keep..].copy_from_slice(out);
        }

        // Rotate post-filter params.
        self.pf_period_old[ch] = pf_t1;
        self.pf_gain_old[ch] = pf_g1;
        self.pf_tapset_old[ch] = pf_tapset1;

        // De-emphasis.
        self.deemph_state[ch] = deemphasis(out, self.deemph_state[ch]);
    }

    fn build_audio_frame(&mut self, pcm: Vec<f32>, samples: usize) -> Frame {
        let mut bytes = Vec::with_capacity(pcm.len() * 4);
        for s in pcm {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let pts = self.pts_counter;
        self.pts_counter += samples as i64;
        Frame::Audio(AudioFrame {
            samples: samples as u32,
            pts: Some(pts),
            data: vec![bytes],
        })
    }
}

impl Decoder for CeltDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.params.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let frame_samples = self.frame_samples;
        if self.channels == 1 {
            let pcm = self.decode_frame_mono(&packet.data)?;
            let frame = self.build_audio_frame(pcm, frame_samples);
            self.output.push_back(frame);
        } else {
            let (l, r) = self.decode_frame_stereo(&packet.data)?;
            let mut interleaved = Vec::with_capacity(frame_samples * 2);
            for i in 0..frame_samples {
                interleaved.push(l[i]);
                interleaved.push(r[i]);
            }
            let frame = self.build_audio_frame(interleaved, frame_samples);
            self.output.push_back(frame);
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        self.output.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.output.clear();
        self.old_band_e.iter_mut().for_each(|v| *v = 0.0);
        self.prev1_log_e.iter_mut().for_each(|v| *v = 0.0);
        self.prev2_log_e.iter_mut().for_each(|v| *v = 0.0);
        for ch in &mut self.prev_tail {
            ch.iter_mut().for_each(|v| *v = 0.0);
        }
        self.rng = 0;
        for v in &mut self.pf_period_old {
            *v = 0;
        }
        for v in &mut self.pf_gain_old {
            *v = 0.0;
        }
        for v in &mut self.pf_tapset_old {
            *v = 0;
        }
        for h in &mut self.pf_history {
            h.iter_mut().for_each(|v| *v = 0.0);
        }
        for v in &mut self.deemph_state {
            *v = 0.0;
        }
        Ok(())
    }
}

pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(CeltDecoder::new(params)?))
}
