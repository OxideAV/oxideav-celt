//! The reference-exact end-to-end CELT frame decoder (RFC 6716 §4.3)
//! — the complete Table-56 walk with every budget gate at its exact
//! position, the §4.3.3 exact allocation, the §4.3.4 exact band loop,
//! the §4.3.2 absolute energy scale (`eMeans`), the exact §4.3.5
//! anti-collapse, the two-stage §4.3.7.1 comb filter, and the §4.3.7.2
//! de-emphasis, for mono and stereo streams on one unified driver.
//!
//! ## Provenance
//!
//! Transcribed from the **normative RFC 6716 Appendix A reference
//! listing** (`celt.c` decode driver, `quant_bands.c` energy codec,
//! `bands.c` anti-collapse), extracted from the staged
//! `docs/audio/opus/rfc6716-opus.txt` per §A.1 and SHA-1-verified
//! against the §A.1-printed value
//! (`86a927223e73d2476646a1b933fcd3fffb6ecc8c`); float-build
//! semantics throughout. The `eMeans` values cross-check against the
//! staged `docs/audio/opus/tables/e-means.csv` (Q4). The §4.3.6/4.3.7
//! inverse transform reuses the crate's §4.3.7-prose synthesis spine
//! ([`crate::synthesis::LongMdctSynthesis`]); the output is scaled by
//! the reference signal scale (`1/32768`, the float-API output
//! convention).

use crate::alloc_exact::{compute_allocation_exact, AllocIo, BITRES, MAX_FINE_BITS};
use crate::band_cap::{compute_band_caps, decode_band_boosts};
use crate::band_layout::EBAND_EDGES_5MS;
use crate::band_minimums::BAND_BINS_LM;
use crate::band_quant::{celt_lcg_rand, quant_all_bands, renormalise_vector, QuantIo};
use crate::bit_allocation::decode_alloc_trim;
use crate::coarse_energy::{decode_coarse_energy, CoarseEnergyState, NUM_BANDS};
use crate::mdct::{build_low_overlap_window_f32, celt_window_f32, imdct_naive_f32};
use crate::range_decoder::RangeDecoder;
use crate::spread::Spread;
use crate::synthesis::mdct_size;
use crate::tf_change::tf_adjustment;
use crate::Error;

/// Mean band energy in base-2 log-amplitude units (`eMeans`,
/// Appendix A `quant_bands.c` float table; staged as
/// `docs/audio/opus/tables/e-means.csv` in Q4 — these are the Q4
/// values divided by 16). Only the first 21 entries apply to the
/// 48 kHz mode.
pub const E_MEANS: [f32; NUM_BANDS] = [
    6.437_5, 6.25, 5.75, 5.312_5, 5.062_5, 4.812_5, 4.5, 4.375, 4.875, 4.687_5, 4.562_5, 4.437_5,
    4.875, 4.625, 4.312_5, 4.5, 4.375, 4.625, 4.75, 4.437_5, 3.75,
];

/// §4.3.7.1 comb-filter tap shapes (Appendix A `celt.c` `gains`
/// table), rows indexed by tapset.
const COMB_GAINS: [[f32; 3]; 3] = [
    [0.306_640_62, 0.217_041_02, 0.129_638_67],
    [0.463_867_2, 0.268_066_4, 0.0],
    [0.799_804_7, 0.100_097_66, 0.0],
];

/// §4.3.7.1 minimum pitch period (`COMBFILTER_MINPERIOD`).
pub const COMBFILTER_MINPERIOD: usize = 15;

/// Maximum §4.3.7.1 pitch period the history must cover
/// (`MAX_PERIOD`).
pub const MAX_PERIOD: usize = 1024;

/// The §4.3.7 overlap length (fixed 120 for the 48 kHz mode).
const OVERLAP: usize = 120;

/// The tapset ICDF (`{2, 1, 1}/4`).
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// The de-emphasis coefficient (Appendix A mode data `preemph[0]` for
/// 48 kHz).
const PREEMPH_COEF: f32 = 0.850_006_1;

/// The float-API signal scale (`CELT_SIG_SCALE`).
const SIG_SCALE: f32 = 32768.0;

/// One channel's §4.3.7.1 comb-filter pass over `y[off..off+n]`
/// with a squared-window crossfade from the `(t0, g0_, tap0)`
/// parameter set to `(t1, g1_, tap1)` (Appendix A `celt.c`
/// `comb_filter`). `y` carries the filtered history before `off`, and
/// the recursion reads the filtered signal.
#[allow(clippy::too_many_arguments)]
fn comb_filter(
    y: &mut [f32],
    off: usize,
    n: usize,
    t0: usize,
    t1: usize,
    g0_: f32,
    g1_: f32,
    tap0: usize,
    tap1: usize,
    window: &[f32],
) {
    let g00 = g0_ * COMB_GAINS[tap0][0];
    let g01 = g0_ * COMB_GAINS[tap0][1];
    let g02 = g0_ * COMB_GAINS[tap0][2];
    let g10 = g1_ * COMB_GAINS[tap1][0];
    let g11 = g1_ * COMB_GAINS[tap1][1];
    let g12 = g1_ * COMB_GAINS[tap1][2];
    let overlap = window.len().min(n);
    for (i, w) in window.iter().enumerate().take(overlap) {
        let idx = off + i;
        let f = w * w;
        let a0 = idx - t0;
        let a1 = idx - t1;
        y[idx] += (1.0 - f) * g00 * y[a0]
            + (1.0 - f) * g01 * (y[a0 - 1] + y[a0 + 1])
            + (1.0 - f) * g02 * (y[a0 - 2] + y[a0 + 2])
            + f * g10 * y[a1]
            + f * g11 * (y[a1 - 1] + y[a1 + 1])
            + f * g12 * (y[a1 - 2] + y[a1 + 2]);
    }
    for i in overlap..n {
        let idx = off + i;
        let a1 = idx - t1;
        y[idx] += g10 * y[a1] + g11 * (y[a1 - 1] + y[a1 + 1]) + g12 * (y[a1 - 2] + y[a1 + 2]);
    }
}

/// The exact §4.3.4.5 tf-parameter decode (Appendix A `celt.c`
/// `tf_decode`): per-band gated toggle bits, the gated `tf_select`
/// bit, and the Tables-60–63 adjustment mapping.
fn tf_decode(
    dec: &mut RangeDecoder<'_>,
    start: usize,
    end: usize,
    is_transient: bool,
    lm: u32,
    tf_res: &mut [i32; NUM_BANDS],
) {
    let budget0 = dec.storage_bits();
    let mut tell = dec.tell();
    let mut logp: u32 = if is_transient { 2 } else { 4 };
    #[allow(clippy::int_plus_one)] // the listing's literal gate
    let tf_select_rsv = u32::from(lm > 0 && tell + logp + 1 <= budget0);
    let budget = budget0 - tf_select_rsv;
    let mut tf_changed = false;
    let mut curr = false;
    let mut raw = [false; NUM_BANDS];
    for r in raw.iter_mut().take(end).skip(start) {
        if tell + logp <= budget {
            curr ^= dec.dec_bit_logp(logp) == 1;
            tell = dec.tell();
            tf_changed |= curr;
        }
        *r = curr;
        logp = if is_transient { 4 } else { 5 };
    }
    let mut tf_select = 0u8;
    if tf_select_rsv == 1
        && tf_adjustment(is_transient, 0, lm as u8, tf_changed)
            != tf_adjustment(is_transient, 1, lm as u8, tf_changed)
    {
        tf_select = dec.dec_bit_logp(1) as u8;
    }
    for i in start..end {
        tf_res[i] = tf_adjustment(is_transient, tf_select, lm as u8, raw[i]) as i32;
    }
}

/// The exact §4.3.5 anti-collapse injection (Appendix A `bands.c`
/// `anti_collapse`, float build): for every collapsed short block of
/// every coded band, inject pseudo-random noise at a level derived
/// from the two-frame energy history, then renormalize.
#[allow(clippy::too_many_arguments)]
fn anti_collapse(
    x: &mut [f32],
    y: Option<&mut [f32]>,
    collapse_masks: &[u8],
    lm: u32,
    channels: usize,
    start: usize,
    end: usize,
    log_e: &[[f32; NUM_BANDS]; 2],
    prev1_log_e: &[[f32; NUM_BANDS]; 2],
    prev2_log_e: &[[f32; NUM_BANDS]; 2],
    pulses: &[i32; NUM_BANDS],
    mut seed: u32,
) {
    let eb = |i: usize| EBAND_EDGES_5MS[i] as usize;
    let chans: [Option<&mut [f32]>; 2] = [Some(x), y];
    let mut chans = chans;
    for i in start..end {
        let n0 = eb(i + 1) - eb(i);
        // Depth in 1/8 bits.
        let depth = (1 + pulses[i]) / ((n0 as i32) << lm);
        let thresh = 0.5 * (-0.125 * depth as f32).exp2();
        let sqrt_1 = 1.0 / (((n0 << lm) as f32).sqrt());

        for c in 0..channels {
            let xc = chans[c].as_deref_mut().expect("channel present");
            let mut prev1 = prev1_log_e[c][i];
            let mut prev2 = prev2_log_e[c][i];
            if channels == 1 {
                prev1 = prev1.max(prev1_log_e[1][i]);
                prev2 = prev2.max(prev2_log_e[1][i]);
            }
            let ediff = (log_e[c][i] - prev1.min(prev2)).max(0.0);
            // r is doubled (or x 2*sqrt(2) at LM 3) because short
            // blocks don't have the same energy as long ones.
            let mut r = 2.0 * (-ediff).exp2();
            if lm == 3 {
                // The listing's literal sqrt(2) truncation.
                #[allow(clippy::excessive_precision, clippy::approx_constant)]
                const SHORT_LM3_GAIN: f32 = 1.414_213_56;
                r *= SHORT_LM3_GAIN;
            }
            r = thresh.min(r);
            r *= sqrt_1;
            let base = eb(i) << lm;
            let mut renormalize = false;
            for k in 0..(1usize << lm) {
                // Detect collapse.
                if collapse_masks[i * channels + c] & (1 << k) == 0 {
                    // Fill with noise.
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        xc[base + (j << lm) + k] = if seed & 0x8000 != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            // Energy was added: renormalize.
            if renormalize {
                renormalise_vector(&mut xc[base..base + (n0 << lm)], 1.0);
            }
        }
    }
}

/// Streaming state of the reference-exact decoder: everything the
/// Appendix A decode driver carries across frames.
#[derive(Debug)]
pub struct CeltRefDecoder {
    lm: u32,
    channels: usize,
    /// §4.3.2.1 inter-frame energy prediction (`oldBandE`) — carries
    /// the fine/finalize-corrected values per the reference.
    pub coarse: CoarseEnergyState,
    old_log_e: [[f32; NUM_BANDS]; 2],
    old_log_e2: [[f32; NUM_BANDS]; 2],
    /// Per-channel §4.3.7 overlap memory (`overlap_mem`, 120 samples).
    overlap_mem: Vec<Vec<f32>>,
    /// The low-overlap long window over the `2 * frame` basis span.
    long_window: Vec<f32>,
    /// The full-overlap 240-sample short-block window.
    short_window: Vec<f32>,
    /// Per-channel filtered output history for the §4.3.7.1 comb
    /// filter (`MAX_PERIOD + 2` most recent samples, oldest first).
    pf_hist: Vec<Vec<f32>>,
    deemph_mem: [f32; 2],
    pf_period: usize,
    pf_gain: f32,
    pf_tapset: usize,
    pf_period_old: usize,
    pf_gain_old: f32,
    pf_tapset_old: usize,
    /// Per-frame §4.3.5 noise seed (the range coder's final `rng`).
    rng: u32,
    window: Vec<f32>,
}

impl CeltRefDecoder {
    /// Build a decoder for frame-size shift `lm` (`0..=3`) and 1 or 2
    /// channels.
    pub fn new(lm: u32, channels: usize) -> Result<Self, Error> {
        if lm > 3 || !(1..=2).contains(&channels) {
            return Err(Error::InvalidParameter);
        }
        let frame = mdct_size(lm).ok_or(Error::InvalidParameter)?;
        let long_window =
            build_low_overlap_window_f32(frame, OVERLAP).ok_or(Error::InvalidParameter)?;
        let short_window =
            build_low_overlap_window_f32(120, OVERLAP).ok_or(Error::InvalidParameter)?;
        let window: Vec<f32> = (0..OVERLAP).map(|i| celt_window_f32(i, OVERLAP)).collect();
        Ok(Self {
            lm,
            channels,
            coarse: CoarseEnergyState::new(),
            old_log_e: [[-28.0; NUM_BANDS]; 2],
            old_log_e2: [[-28.0; NUM_BANDS]; 2],
            overlap_mem: vec![vec![0.0; OVERLAP]; channels],
            long_window,
            short_window,
            pf_hist: vec![vec![0.0; MAX_PERIOD + 2]; channels],
            deemph_mem: [0.0; 2],
            pf_period: COMBFILTER_MINPERIOD,
            pf_gain: 0.0,
            pf_tapset: 0,
            pf_period_old: COMBFILTER_MINPERIOD,
            pf_gain_old: 0.0,
            pf_tapset_old: 0,
            rng: 0,
            window,
        })
    }

    /// The per-channel frame size in samples.
    pub fn frame_size(&self) -> usize {
        mdct_size(self.lm).expect("lm validated at construction")
    }

    /// Decode one CELT frame into interleaved f32 PCM in `[-1, 1]`
    /// (the reference float-API output scale).
    pub fn decode_frame(&mut self, bytes: &[u8]) -> Result<Vec<f32>, Error> {
        let lm = self.lm;
        let channels = self.channels;
        let frame = self.frame_size();
        let start = 0usize;
        let end = NUM_BANDS;
        let n_coded = (1usize << lm) * EBAND_EDGES_5MS[end] as usize;
        if bytes.is_empty() || bytes.len() > 1275 {
            return Err(Error::InvalidParameter);
        }
        let mut dec = RangeDecoder::new(bytes);
        let total_bits = (bytes.len() * 8) as u32;

        // A mono frame after a stereo one predicts from the max.
        if channels == 1 {
            for i in 0..NUM_BANDS {
                self.coarse.energy[0][i] = self.coarse.energy[0][i].max(self.coarse.energy[1][i]);
            }
        }

        let mut tell = dec.tell();
        let silence = if tell >= total_bits {
            true
        } else if tell == 1 {
            dec.dec_bit_logp(15) == 1
        } else {
            false
        };

        let mut x = vec![0f32; n_coded];
        let mut y = vec![0f32; n_coded];
        let mut band_e = [[0f32; NUM_BANDS]; 2];
        let mut is_transient = false;
        let mut pf_pitch = 0usize;
        let mut pf_gain = 0.0f32;
        let mut pf_tapset = 0usize;

        if !silence {
            // Post-filter parameters (only when the frame starts at
            // band 0 and the budget allows the full field).
            tell = dec.tell();
            if start == 0 && tell + 16 <= total_bits {
                if dec.dec_bit_logp(1) == 1 {
                    let octave = dec.dec_uint(6).map_err(|_| Error::InvalidParameter)?;
                    pf_pitch = ((16usize << octave) + dec.dec_bits(4 + octave) as usize) - 1;
                    let qg = dec.dec_bits(3);
                    if dec.tell() + 2 <= total_bits {
                        pf_tapset = dec.dec_icdf(&TAPSET_ICDF, 2) as usize;
                    }
                    pf_gain = 0.09375 * (qg + 1) as f32;
                }
                tell = dec.tell();
            }

            // Transient flag.
            if lm > 0 && tell + 3 <= total_bits {
                is_transient = dec.dec_bit_logp(3) == 1;
                tell = dec.tell();
            }

            // Intra flag + coarse energy.
            let intra = tell + 3 <= total_bits && dec.dec_bit_logp(3) == 1;
            decode_coarse_energy(&mut dec, &mut self.coarse, intra, lm, start, end, channels)?;

            // Time-frequency parameters.
            let mut tf_res = [0i32; NUM_BANDS];
            tf_decode(&mut dec, start, end, is_transient, lm, &mut tf_res);

            // Spread decision.
            tell = dec.tell();
            let spread = if tell + 4 <= total_bits {
                crate::spread::decode_spread(&mut dec)
            } else {
                Spread::Normal
            };

            // Per-band caps + dynalloc boosts.
            let bins: Vec<u32> = BAND_BINS_LM[lm as usize][start..end].to_vec();
            let mut caps16 = vec![0i16; end - start];
            if !compute_band_caps(lm, channels == 2, channels as u32, &bins, &mut caps16) {
                return Err(Error::InvalidParameter);
            }
            let frame_8th = (bytes.len() * 8 * 8) as i32;
            let boosts = decode_band_boosts(
                &mut dec,
                start as u32,
                end as u32,
                channels as u32,
                &bins,
                &caps16,
                frame_8th,
            )
            .ok_or(Error::InvalidParameter)?;
            let mut offsets = [0i32; NUM_BANDS];
            offsets[start..end].copy_from_slice(&boosts.boost);

            // Allocation trim.
            let trim_gated =
                dec.tell_frac() as i64 + 48 <= frame_8th as i64 - boosts.total_boost as i64;
            let alloc_trim = decode_alloc_trim(&mut dec, trim_gated).unwrap_or(5);

            // Anti-collapse reservation + the exact allocation walk.
            let mut bits = (bytes.len() as i32 * 8) * 8 - dec.tell_frac() as i32 - 1;
            let anti_collapse_rsv =
                if is_transient && lm >= 2 && bits >= ((lm as i32 + 2) << BITRES) {
                    1 << BITRES
                } else {
                    0
                };
            bits -= anti_collapse_rsv;
            let mut caps = [0i32; NUM_BANDS];
            for (c, &v) in caps[start..end].iter_mut().zip(caps16.iter()) {
                *c = v as i32;
            }
            let alloc = compute_allocation_exact(
                start,
                end,
                &offsets,
                &caps,
                alloc_trim as i32,
                bits,
                channels as i32,
                lm,
                AllocIo::Decode(&mut dec),
            )?;

            // Fine energy (band-major, channel-minor).
            for i in start..end {
                let fq = alloc.fine_bits[i];
                if fq <= 0 {
                    continue;
                }
                for c in 0..channels {
                    let q2 = dec.dec_bits(fq as u32) as f32;
                    let offset = (q2 + 0.5) * (1 << (14 - fq)) as f32 * (1.0 / 16384.0) - 0.5;
                    self.coarse.energy[c][i] += offset;
                }
            }

            // The §4.3.4 band loop.
            let mut seed = self.rng;
            let walk = quant_all_bands(
                QuantIo::Decode(&mut dec),
                start,
                end,
                &mut x,
                (channels == 2).then_some(&mut y[..]),
                &alloc.shape_bits,
                is_transient,
                spread,
                alloc.dual_stereo,
                alloc.intensity,
                &tf_res,
                (bytes.len() as i32) * (8 << BITRES) - anti_collapse_rsv,
                alloc.balance,
                lm,
                alloc.coded_bands,
                &mut seed,
                None,
                false,
            )?;

            // Anti-collapse bit (after the shape vectors).
            let anti_collapse_on = anti_collapse_rsv > 0 && dec.dec_bits(1) == 1;

            // Final fine-energy bits (§4.3.2.2 finalize).
            let mut bits_left = (bytes.len() * 8) as i32 - dec.tell() as i32;
            for prio in [false, true] {
                let mut i = start;
                while i < end && bits_left >= channels as i32 {
                    if alloc.fine_bits[i] >= MAX_FINE_BITS || alloc.fine_priority[i] != prio {
                        i += 1;
                        continue;
                    }
                    for c in 0..channels {
                        let q2 = dec.dec_bits(1) as f32;
                        let offset = (q2 - 0.5)
                            * (1 << (14 - alloc.fine_bits[i] - 1)) as f32
                            * (1.0 / 16384.0);
                        self.coarse.energy[c][i] += offset;
                        bits_left -= 1;
                    }
                    i += 1;
                }
            }

            if anti_collapse_on {
                anti_collapse(
                    &mut x,
                    (channels == 2).then_some(&mut y[..]),
                    &walk.collapse_masks,
                    lm,
                    channels,
                    start,
                    end,
                    &self.coarse.energy,
                    &self.old_log_e,
                    &self.old_log_e2,
                    &alloc.shape_bits,
                    seed,
                );
            }

            // log2Amp: the absolute amplitude scale (eMeans restored).
            for (be, ce) in band_e
                .iter_mut()
                .zip(self.coarse.energy.iter())
                .take(channels)
            {
                for i in start..end {
                    be[i] = (ce[i] + E_MEANS[i]).exp2();
                }
            }
        } else {
            // Silence: zero spectrum, floor energies.
            for c in 0..2 {
                for i in 0..NUM_BANDS {
                    self.coarse.energy[c][i] = -28.0;
                }
            }
        }

        // Denormalise + inverse MDCT per channel, then the two-stage
        // comb filter over the filtered history.
        let m = 1usize << lm;
        let eb = |i: usize| EBAND_EDGES_5MS[i] as usize;
        let mut pcm = vec![0f32; channels * frame];
        let short_size = frame / m;
        self.pf_period = self.pf_period.max(COMBFILTER_MINPERIOD);
        self.pf_period_old = self.pf_period_old.max(COMBFILTER_MINPERIOD);
        // Legal encoders bound the period to 1022 (§4.3.7.1); the
        // upper clamp is a defensive bound keeping the filter inside
        // the carried history on malformed streams.
        let pitch_clamped = pf_pitch.clamp(COMBFILTER_MINPERIOD, MAX_PERIOD - 2);

        for c in 0..channels {
            let spec = if c == 0 { &x } else { &y };
            // §4.3.6 denormalization onto the full MDCT span (the
            // coded top sits below `frame`; the rest stays zero).
            let mut freq = vec![0f32; frame];
            for (i, &g) in band_e[c].iter().enumerate().take(end).skip(start) {
                for j in m * eb(i)..m * eb(i + 1) {
                    freq[j] = spec[j] * g;
                }
            }
            // The inverse MDCT + overlap-add at the reference
            // emission alignment: the long basis spans `2*frame`
            // samples; the listing emits the window's support
            // `[P, P + frame + overlap)` with `P = (frame -
            // overlap)/2` directly (its low-overlap window is zero
            // outside), carrying the last `overlap` samples in
            // `overlap_mem`. The backward transform carries twice
            // the §4.3.7 half-scale (the listing folds that factor
            // into the window mixing).
            let mut xbuf = vec![0f32; frame + OVERLAP];
            if !is_transient {
                let p = (frame - OVERLAP) / 2;
                let mut u = vec![0f32; 2 * frame];
                if !imdct_naive_f32(&freq, &mut u) {
                    return Err(Error::InvalidParameter);
                }
                for (j, o) in xbuf.iter_mut().enumerate() {
                    *o = 2.0 * u[p + j] * self.long_window[p + j];
                }
            } else {
                // 2^lm interleaved short blocks at hop 120 with the
                // full-overlap 240-sample window (P = 0).
                let blocks = m;
                let mut block_spec = vec![0f32; short_size];
                let mut u = vec![0f32; 2 * short_size];
                for b in 0..blocks {
                    for (j, s) in block_spec.iter_mut().enumerate() {
                        *s = freq[b + j * blocks];
                    }
                    if !imdct_naive_f32(&block_spec, &mut u) {
                        return Err(Error::InvalidParameter);
                    }
                    for (j, &uv) in u.iter().enumerate() {
                        xbuf[b * short_size + j] += 2.0 * uv * self.short_window[j];
                    }
                }
            }
            let mut time = vec![0f32; frame];
            for (t, (&xv, &ov)) in time
                .iter_mut()
                .zip(xbuf.iter().zip(self.overlap_mem[c].iter()))
                .take(OVERLAP)
            {
                *t = xv + ov;
            }
            time[OVERLAP..frame].copy_from_slice(&xbuf[OVERLAP..frame]);
            self.overlap_mem[c].copy_from_slice(&xbuf[frame..frame + OVERLAP]);

            // Comb filter over [history | frame].
            let hist_len = self.pf_hist[c].len();
            let mut buf = Vec::with_capacity(hist_len + frame);
            buf.extend_from_slice(&self.pf_hist[c]);
            buf.extend_from_slice(&time);
            comb_filter(
                &mut buf,
                hist_len,
                short_size,
                self.pf_period_old,
                self.pf_period,
                self.pf_gain_old,
                self.pf_gain,
                self.pf_tapset_old,
                self.pf_tapset,
                &self.window,
            );
            if lm != 0 {
                comb_filter(
                    &mut buf,
                    hist_len + short_size,
                    frame - short_size,
                    self.pf_period,
                    pitch_clamped,
                    self.pf_gain,
                    pf_gain,
                    self.pf_tapset,
                    pf_tapset,
                    &self.window,
                );
            }
            // Keep the filtered tail as next frame's history.
            let keep = self.pf_hist[c].len();
            let tail = buf.len() - keep;
            self.pf_hist[c].copy_from_slice(&buf[tail..]);

            // De-emphasis + output scale.
            let mut mem = self.deemph_mem[c];
            for (j, &v) in buf[hist_len..].iter().enumerate() {
                let tmp = v + mem;
                mem = PREEMPH_COEF * tmp;
                pcm[j * channels + c] = tmp * (1.0 / SIG_SCALE);
            }
            self.deemph_mem[c] = mem;
        }

        // Post-filter parameter pipeline.
        self.pf_period_old = self.pf_period;
        self.pf_gain_old = self.pf_gain;
        self.pf_tapset_old = self.pf_tapset;
        self.pf_period = pitch_clamped;
        self.pf_gain = pf_gain;
        self.pf_tapset = pf_tapset;
        if lm != 0 {
            self.pf_period_old = self.pf_period;
            self.pf_gain_old = self.pf_gain;
            self.pf_tapset_old = self.pf_tapset;
        }

        // Mono duplicates its energy row.
        if channels == 1 {
            for i in 0..NUM_BANDS {
                self.coarse.energy[1][i] = self.coarse.energy[0][i];
            }
        }

        // Two-frame energy history for the §4.3.5 anti-collapse.
        if !is_transient {
            self.old_log_e2 = self.old_log_e;
            self.old_log_e = self.coarse.energy;
        } else {
            for c in 0..2 {
                for i in 0..NUM_BANDS {
                    self.old_log_e[c][i] = self.old_log_e[c][i].min(self.coarse.energy[c][i]);
                }
            }
        }

        self.rng = dec.range_state();
        Ok(pcm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A silence frame (all-zero payload after the silence flag)
    /// decodes to a decaying tail and floors the energy state.
    #[test]
    fn silence_frame_floors_energy() {
        let mut d = CeltRefDecoder::new(2, 1).expect("decoder");
        // A one-byte frame: tell() == 1 after init, the silence flag
        // is decoded from logp 15; an all-ones first byte yields the
        // low-probability "1".
        let bytes = [0xFFu8, 0xFF];
        let pcm = d.decode_frame(&bytes).expect("decode");
        assert_eq!(pcm.len(), 480);
        assert!(pcm.iter().all(|v| v.abs() < 1.0));
        assert!(d.coarse.energy[0].iter().all(|&e| e == -28.0));
    }

    /// Random payload bytes decode to finite PCM without panicking
    /// at every LM and channel count (robustness of the exact walk on
    /// arbitrary input).
    #[test]
    fn random_frames_decode_finite() {
        for &(lm, ch, len) in &[
            (0u32, 1usize, 30usize),
            (1, 1, 47),
            (2, 1, 80),
            (3, 1, 160),
            (1, 2, 96),
            (3, 2, 201),
        ] {
            let mut d = CeltRefDecoder::new(lm, ch).expect("decoder");
            let mut seed = 0x00C0_FFEEu32 ^ (lm << 8) ^ ch as u32;
            for _ in 0..6 {
                let bytes: Vec<u8> = (0..len)
                    .map(|_| {
                        seed = celt_lcg_rand(seed);
                        (seed >> 24) as u8
                    })
                    .collect();
                // Garbage may legitimately trip the §4.1.5
                // corrupt-frame path (surfaced as an error rather
                // than the reference's clamp-and-continue); a
                // successful decode must be finite and full-length.
                match d.decode_frame(&bytes) {
                    Ok(pcm) => {
                        assert_eq!(pcm.len(), ch * d.frame_size());
                        assert!(
                            pcm.iter().all(|v| v.is_finite()),
                            "non-finite PCM at lm={lm} ch={ch}"
                        );
                    }
                    Err(e) => {
                        assert!(
                            matches!(
                                e,
                                crate::Error::NotImplemented | crate::Error::InvalidParameter
                            ),
                            "unexpected error kind"
                        );
                    }
                }
            }
        }
    }
}
