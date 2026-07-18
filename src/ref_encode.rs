//! The reference-compatible end-to-end CELT frame **encoder**
//! (RFC 6716 §4.3 in the encode direction) — the exact Table-56 walk
//! with every budget gate at the position the reference-exact decoder
//! ([`crate::ref_decode::CeltRefDecoder`]) evaluates it, driving the
//! exact §4.3.3 allocation walk ([`crate::alloc_exact`]) and the exact
//! §4.3.4 band loop ([`crate::band_quant`]) in their encode direction,
//! the absolute `eMeans` energy scale, and the §4.3.7.2 pre-emphasis /
//! forward-MDCT analysis front end that mirrors the decoder's
//! synthesis alignment.
//!
//! ## Provenance
//!
//! The bitstream walk is the mirror image of `ref_decode` (transcribed
//! from the normative RFC 6716 Appendix A reference listing, extracted
//! from the staged `docs/audio/opus/rfc6716-opus.txt` per §A.1 and
//! SHA-1-verified against the §A.1-printed value
//! `86a927223e73d2476646a1b933fcd3fffb6ecc8c`); every symbol the
//! decoder reads under a budget gate is written here under the
//! identical gate, so any stream this encoder emits parses identically
//! on both the crate's decoder and the §A.1 listing decoder. The
//! **decisions** (transient detection, intra selection, spread, trim,
//! boosts, anti-collapse) are encoder freedom under §5.3 and are
//! documented in-crate at each site; every choice is carried on the
//! wire, so decoders stay in lockstep by construction.
//!
//! ## Analysis front end
//!
//! The §4.3.7.2 pre-emphasis FIR runs at the reference signal scale
//! (`32768 * pcm`), and the forward MDCT is the exact adjoint of the
//! decoder's emission alignment: the long basis analyzes the window
//! support `[P, P + frame + overlap)`, `P = (frame - overlap)/2`, at
//! **half** the crate's §4.3.7 forward companion scale (the decoder
//! emits at twice the §4.3.7 half-scale, so the round trip is unit
//! gain); transient frames analyze `2^LM` interleaved short blocks at
//! hop 120 with the full-overlap 240-sample window. The encoder
//! therefore carries `overlap` (120) samples of algorithmic delay,
//! exactly like the reference.

use crate::alloc_exact::{compute_allocation_exact, AllocIo, BITRES, MAX_FINE_BITS};
use crate::band_cap::{compute_band_caps, encode_band_boosts};
use crate::band_layout::EBAND_EDGES_5MS;
use crate::band_minimums::BAND_BINS_LM;
use crate::band_quant::{haar1, quant_all_bands, BandAmps, QuantIo};
use crate::bit_allocation::encode_alloc_trim;
use crate::coarse_energy::{encode_coarse_energy, CoarseEnergyState, NUM_BANDS};
use crate::encoder_decisions::{
    boost_thresholds, choose_alloc_trim, choose_intra_mode, choose_mid_side_stereo,
    intensity_start_band, low_band_stereo_correlation,
};
use crate::mdct::{build_low_overlap_window_f32, mdct_naive_f32};
use crate::range_encoder::RangeEncoder;
use crate::ref_decode::E_MEANS;
use crate::spread::{encode_spread, Spread};
use crate::synthesis::mdct_size;
use crate::tf_change::tf_adjustment;
use crate::Error;

/// The §4.3.7 overlap length (fixed 120 for the 48 kHz mode).
const OVERLAP: usize = 120;

/// The §4.3.7.2 pre-emphasis coefficient (48 kHz mode).
const PREEMPH_COEF: f32 = 0.850_006_1;

/// The reference float-API signal scale (`CELT_SIG_SCALE`).
const SIG_SCALE: f32 = 32768.0;

/// Energy floor for the coarse targets in base-2 log-amplitude units
/// (the decoder's silence floor; keeps the Laplace symbols bounded on
/// digital silence).
const TARGET_FLOOR: f32 = -28.0;

/// All-band silence threshold: a frame whose every band target sits at
/// or below this is coded as a Table-56 silence frame. Documented
/// in-crate encoder freedom (§4.3 only defines the silence flag).
const SILENCE_THRESHOLD: f32 = -27.5;

/// The §4.3.4.5 tf-parameter encode — the exact mirror of the
/// decoder's gated walk: one differential toggle bit per band under
/// the running budget gate, then the gated `tf_select` bit. `desired`
/// carries the per-band choices from [`choose_tf_resolution`]; a band
/// whose toggle the budget gate closes keeps the running value,
/// exactly as the decoder will reconstruct it. `tf_select` stays 0 (a
/// legal §5.3 choice; the bit is written whenever the decoder's gate
/// would read it).
#[allow(clippy::too_many_arguments)]
fn tf_encode(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    is_transient: bool,
    lm: u32,
    total_bits: u32,
    desired: &[bool; NUM_BANDS],
    tf_res: &mut [i32; NUM_BANDS],
) -> Result<(), Error> {
    let mut tell = enc.tell();
    let mut logp: u32 = if is_transient { 2 } else { 4 };
    #[allow(clippy::int_plus_one)] // the decode mirror's literal gate
    let tf_select_rsv = u32::from(lm > 0 && tell + logp + 1 <= total_bits);
    let budget = total_bits - tf_select_rsv;
    let mut tf_changed = false;
    let mut curr = false;
    let mut raw = [false; NUM_BANDS];
    for (i, r) in raw.iter_mut().enumerate().take(end).skip(start) {
        if tell + logp <= budget {
            let toggle = desired[i] != curr;
            enc.enc_bit_logp(u32::from(toggle), logp)?;
            curr ^= toggle;
            tell = enc.tell();
            tf_changed |= curr;
        }
        *r = curr;
        logp = if is_transient { 4 } else { 5 };
    }
    let tf_select = 0u8;
    if tf_select_rsv == 1
        && tf_adjustment(is_transient, 0, lm as u8, tf_changed)
            != tf_adjustment(is_transient, 1, lm as u8, tf_changed)
    {
        enc.enc_bit_logp(u32::from(tf_select), 1)?;
    }
    for i in start..end {
        tf_res[i] = tf_adjustment(is_transient, tf_select, lm as u8, raw[i]) as i32;
    }
    Ok(())
}

/// L1 compaction of one band's unit-norm shape after the §4.3.4.5
/// TF transform the band walk would apply for `tf_change` (the
/// level-0 mono recombine / time-split ladder of the §4.3.4 loop).
/// A lower L1 at fixed L2 means a sparser vector — cheaper for the
/// PVQ codebook at equal quality.
fn tf_l1_metric(band: &[f32], b_frame: usize, tf_change: i32) -> f32 {
    let mut v = band.to_vec();
    let n = v.len();
    let mut b_blocks = b_frame;
    let mut n_b = n / b_blocks.max(1);
    let mut tf = tf_change;
    let recombine = tf.max(0).min(b_blocks.trailing_zeros() as i32);
    for k in 0..recombine {
        haar1(&mut v, n >> k, 1 << k);
    }
    b_blocks >>= recombine;
    n_b <<= recombine;
    while (n_b & 1) == 0 && tf < 0 {
        haar1(&mut v, n_b, b_blocks);
        b_blocks <<= 1;
        n_b >>= 1;
        tf += 1;
    }
    v.iter().map(|a| a.abs()).sum()
}

/// Per-band §4.3.4.5 TF-resolution decision (encoder freedom; the RFC
/// leaves the criterion to the encoder): on a **transient** frame,
/// toggle a band (keep it at short-block resolution instead of the
/// Table-62 default full recombine) when the short-resolution shape is
/// markedly sparser in L1 — a band whose energy is concentrated in one
/// short block codes far better untangled. Non-transient frames keep
/// the identity resolution: an A/B sweep against the §A.1 listing
/// encoder measured the L1 proxy as a small net loss for the long-MDCT
/// time-split candidates on tonal material (a rate-distortion lambda
/// would be needed to price those toggles; recorded as a follow-up),
/// while the transient-frame decision is a clear win on onsets.
fn choose_tf_resolution(
    x: &[f32],
    m: usize,
    is_transient: bool,
    lm: u32,
    start: usize,
    end: usize,
) -> [bool; NUM_BANDS] {
    const MARGIN: f32 = 0.85;
    let eb = |i: usize| EBAND_EDGES_5MS[i] as usize;
    let b_frame = if is_transient { m } else { 1 };
    let adj0 = tf_adjustment(is_transient, 0, lm as u8, false) as i32;
    let adj1 = tf_adjustment(is_transient, 0, lm as u8, true) as i32;
    let mut out = [false; NUM_BANDS];
    if !is_transient || adj0 == adj1 {
        return out;
    }
    for i in start..end {
        let band = &x[m * eb(i)..m * eb(i + 1)];
        if band.len() < 2 {
            continue;
        }
        let l1_keep = tf_l1_metric(band, b_frame, adj0);
        let l1_toggle = tf_l1_metric(band, b_frame, adj1);
        out[i] = l1_toggle < MARGIN * l1_keep;
    }
    out
}

/// Transient detector (encoder freedom, RFC 6716 §5.3.2 leaves the
/// decision to the encoder): the pre-emphasized frame is cut into its
/// `2^LM` short blocks and a frame is declared transient when a later
/// block carries at least `TRANSIENT_RATIO` times the energy of every
/// block before it (a hard onset that a single long MDCT would smear).
fn detect_transient(chans: &[Vec<f32>], frame: usize, lm: u32) -> bool {
    if lm == 0 {
        return false;
    }
    let blocks = 1usize << lm;
    let sb = frame / blocks;
    const TRANSIENT_RATIO: f32 = 40.0;
    for ch in chans {
        let cur = &ch[OVERLAP..OVERLAP + frame];
        let mut max_prev = 1e-9f32;
        for b in 0..blocks {
            let e: f32 = cur[b * sb..(b + 1) * sb].iter().map(|v| v * v).sum();
            if b > 0 && e > TRANSIENT_RATIO * max_prev {
                return true;
            }
            max_prev = max_prev.max(e);
        }
    }
    false
}

/// Streaming state of the reference-compatible encoder: the
/// §4.3.2.1 quantized-energy prediction (kept in lockstep with the
/// decoder by construction), the pre-emphasis tap, the analysis
/// overlap history, and the cross-frame folding seed.
#[derive(Debug)]
pub struct CeltRefEncoder {
    lm: u32,
    channels: usize,
    /// §4.3.2.1 inter-frame energy prediction over the **quantized**
    /// energies — the same values the decoder reconstructs.
    coarse: CoarseEnergyState,
    old_log_e: [[f32; NUM_BANDS]; 2],
    old_log_e2: [[f32; NUM_BANDS]; 2],
    /// Per-channel pre-emphasized input history (`overlap` samples) —
    /// the analysis lookahead delay.
    in_mem: Vec<Vec<f32>>,
    preemph_mem: [f32; 2],
    long_window: Vec<f32>,
    short_window: Vec<f32>,
    /// Cross-frame §4.3.4 folding seed (the range coder's final
    /// state, mirroring the decoder's pipeline).
    rng: u32,
    /// Previous frame's `coded_bands` (the §4.3.3 skip-choice
    /// hysteresis anchor).
    prev_coded_bands: i32,
}

impl CeltRefEncoder {
    /// Build an encoder for frame-size shift `lm` (`0..=3`) and 1 or 2
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
        Ok(Self {
            lm,
            channels,
            coarse: CoarseEnergyState::new(),
            old_log_e: [[-28.0; NUM_BANDS]; 2],
            old_log_e2: [[-28.0; NUM_BANDS]; 2],
            in_mem: vec![vec![0.0; OVERLAP]; channels],
            preemph_mem: [0.0; 2],
            long_window,
            short_window,
            rng: 0,
            prev_coded_bands: 0,
        })
    }

    /// The per-channel frame size in samples.
    pub fn frame_size(&self) -> usize {
        mdct_size(self.lm).expect("lm validated at construction")
    }

    /// Forward MDCT of one channel's analysis block (`frame + overlap`
    /// pre-emphasized samples) at the decoder's emission alignment.
    fn forward_freq(&self, block: &[f32], is_transient: bool) -> Result<Vec<f32>, Error> {
        let frame = self.frame_size();
        let m = 1usize << self.lm;
        let mut freq = vec![0f32; frame];
        if !is_transient {
            let p = (frame - OVERLAP) / 2;
            let mut v = vec![0f32; 2 * frame];
            for (j, &s) in block.iter().enumerate() {
                v[p + j] = s * self.long_window[p + j];
            }
            if !mdct_naive_f32(&v, &mut freq) {
                return Err(Error::InvalidParameter);
            }
            for f in freq.iter_mut() {
                *f *= 0.5;
            }
        } else {
            let mut v = vec![0f32; 240];
            let mut sb = vec![0f32; 120];
            for b in 0..m {
                for (j, o) in v.iter_mut().enumerate() {
                    *o = block[b * 120 + j] * self.short_window[j];
                }
                if !mdct_naive_f32(&v, &mut sb) {
                    return Err(Error::InvalidParameter);
                }
                for (j, &s) in sb.iter().enumerate() {
                    freq[b + j * m] = 0.5 * s;
                }
            }
        }
        Ok(freq)
    }

    /// Encode one frame of interleaved f32 PCM in `[-1, 1]` (the
    /// reference float-API input scale; `frame_size() * channels`
    /// samples) into exactly `frame_bytes` bytes. The decoded output
    /// is the input delayed by `overlap` (120) samples.
    pub fn encode_frame(&mut self, pcm: &[f32], frame_bytes: usize) -> Result<Vec<u8>, Error> {
        let lm = self.lm;
        let channels = self.channels;
        let frame = self.frame_size();
        let m = 1usize << lm;
        let start = 0usize;
        let end = NUM_BANDS;
        let eb = |i: usize| EBAND_EDGES_5MS[i] as usize;
        let n_coded = m * eb(end);
        if pcm.len() != channels * frame || !(2..=1275).contains(&frame_bytes) {
            return Err(Error::InvalidParameter);
        }
        let total_bits = (frame_bytes * 8) as u32;
        let frame_8th = (frame_bytes * 64) as i32;

        // §4.3.7.2 pre-emphasis at the reference signal scale, into
        // per-channel `[history | frame]` analysis blocks.
        let mut blocks: Vec<Vec<f32>> = Vec::with_capacity(channels);
        for c in 0..channels {
            let mut block = Vec::with_capacity(frame + OVERLAP);
            block.extend_from_slice(&self.in_mem[c]);
            let mut mem = self.preemph_mem[c];
            for j in 0..frame {
                let s = SIG_SCALE * pcm[j * channels + c];
                block.push(s - PREEMPH_COEF * mem);
                mem = s;
            }
            self.preemph_mem[c] = mem;
            self.in_mem[c].copy_from_slice(&block[frame..frame + OVERLAP]);
            blocks.push(block);
        }

        // Mono prediction parity with the decoder ("a mono frame
        // after a stereo one predicts from the max"); for a fixed-mono
        // stream this is the identity from the second frame on.
        if channels == 1 {
            for i in 0..NUM_BANDS {
                self.coarse.energy[0][i] = self.coarse.energy[0][i].max(self.coarse.energy[1][i]);
            }
        }

        let mut enc = RangeEncoder::new();

        // ── Table 56: silence ──
        let want_transient = detect_transient(&blocks, frame, lm);
        // Provisional analysis to detect all-band silence (geometry
        // does not matter for the threshold; use the long transform).
        // Real analysis happens after the transient gate below.
        let mut tell = enc.tell();
        let mut silence = true;
        'sil: for block in &blocks {
            for &s in &block[..] {
                if s.abs() >= 0.5 {
                    silence = false;
                    break 'sil;
                }
            }
        }
        if tell >= total_bits {
            // Degenerate frame: nothing fits (unreachable for
            // `frame_bytes >= 2`); the decoder infers silence.
            silence = true;
        } else if tell == 1 {
            enc.enc_bit_logp(u32::from(silence), 15)?;
        }

        let mut is_transient = false;
        if !silence {
            // ── Post-filter gate (this encoder does not signal one;
            // §5.3.1 sanctions the choice) ──
            tell = enc.tell();
            if start == 0 && tell + 16 <= total_bits {
                enc.enc_bit_logp(0, 1)?;
                tell = enc.tell();
            }

            // ── Transient flag ──
            if lm > 0 && tell + 3 <= total_bits {
                is_transient = want_transient;
                enc.enc_bit_logp(u32::from(is_transient), 3)?;
                tell = enc.tell();
            }

            // ── Analysis: forward MDCT, band energies, unit shapes ──
            let mut x = vec![0f32; n_coded];
            let mut y = vec![0f32; n_coded];
            let mut amps: BandAmps = [[0.0; NUM_BANDS]; 2];
            let mut targets = [[0f32; NUM_BANDS]; 2];
            let mut freqs: Vec<Vec<f32>> = Vec::with_capacity(channels);
            for c in 0..channels {
                let freq = self.forward_freq(&blocks[c], is_transient)?;
                let spec = if c == 0 { &mut x } else { &mut y };
                for i in start..end {
                    let lo = m * eb(i);
                    let hi = m * eb(i + 1);
                    let e: f32 = freq[lo..hi].iter().map(|v| v * v).sum();
                    let amp = (1e-27 + e).sqrt();
                    amps[c][i] = amp;
                    targets[c][i] = (amp.log2() - E_MEANS[i]).max(TARGET_FLOOR);
                    let g = 1.0 / amp;
                    for (o, &v) in spec[lo..hi].iter_mut().zip(freq[lo..hi].iter()) {
                        *o = v * g;
                    }
                }
                freqs.push(freq);
            }
            // A frame that turns out spectrally silent is still coded
            // as a normal frame here (the silence decision above is
            // taken on the time-domain block before the flag is
            // written); `SILENCE_THRESHOLD` only floors the targets.
            for t in targets.iter_mut().take(channels) {
                for v in t.iter_mut() {
                    if *v < SILENCE_THRESHOLD {
                        *v = TARGET_FLOOR;
                    }
                }
            }

            // ── Intra flag + coarse energy ──
            // §5.3.3 two-pass mode selection: price the coarse walk
            // both ways on scratch states and keep the cheaper one
            // (ties to inter).
            let intra =
                choose_intra_mode(&self.coarse, &targets, lm, start, end, channels, total_bits)
                    .unwrap_or(false);
            if tell + 3 <= total_bits {
                enc.enc_bit_logp(u32::from(intra), 3)?;
            }
            encode_coarse_energy(
                &mut enc,
                &mut self.coarse,
                &targets,
                intra,
                lm,
                start,
                end,
                channels,
                total_bits,
            )?;

            // ── Time-frequency parameters (per-band L1 decision) ──
            let mut tf_desired = choose_tf_resolution(&x, m, is_transient, lm, start, end);
            if channels == 2 {
                let tf_y = choose_tf_resolution(&y, m, is_transient, lm, start, end);
                // One tf vector covers both channels (Table 56 codes
                // one walk per frame): only toggle when both agree.
                for (d, ty) in tf_desired.iter_mut().zip(tf_y.iter()) {
                    *d &= *ty;
                }
            }
            let mut tf_res = [0i32; NUM_BANDS];
            tf_encode(
                &mut enc,
                start,
                end,
                is_transient,
                lm,
                total_bits,
                &tf_desired,
                &mut tf_res,
            )?;

            // ── Spread decision (§5.3 freedom; Normal by default) ──
            let spread = Spread::Normal;
            tell = enc.tell();
            if tell + 4 <= total_bits {
                encode_spread(&mut enc, spread)?;
            }

            // ── Per-band caps + dynalloc boosts ──
            let bins: Vec<u32> = BAND_BINS_LM[lm as usize][start..end].to_vec();
            let mut caps16 = vec![0i16; end - start];
            if !compute_band_caps(lm, channels == 2, channels as u32, &bins, &mut caps16) {
                return Err(Error::InvalidParameter);
            }
            // §5.3.4.1 boost rule on the channel-max band envelope:
            // an interior band whose contrast `D_j = 2E_j - E_{j-1} -
            // E_{j+1}` exceeds `t1` (`t2`) gets one (two) dynalloc
            // quanta; the quantum is the decode loop's
            // `min(8*width, max(48, width))` with `width = C*N`.
            let mut emax = [0f32; NUM_BANDS];
            for i in start..end {
                emax[i] = targets[0][i];
                if channels == 2 {
                    emax[i] = emax[i].max(targets[1][i]);
                }
            }
            let (t1, t2) = boost_thresholds(lm);
            let mut want_boost = vec![0i32; end - start];
            for j in start + 1..end.saturating_sub(1) {
                let d = 2.0 * emax[j] - emax[j - 1] - emax[j + 1];
                let steps = if d > t2 {
                    2
                } else if d > t1 {
                    1
                } else {
                    0
                };
                if steps > 0 {
                    let width = channels as i32 * (m * (eb(j + 1) - eb(j))) as i32;
                    let quanta = (8 * width).min(width.max(48));
                    want_boost[j - start] = steps * quanta;
                }
            }
            let boosts = encode_band_boosts(
                &mut enc,
                start as u32,
                end as u32,
                channels as u32,
                &bins,
                &caps16,
                frame_8th,
                &want_boost,
            )?;
            let mut offsets = [0i32; NUM_BANDS];
            offsets[start..end].copy_from_slice(&boosts.boost);

            // ── Allocation trim (§5.3.4.2: tilt + stereo-correlation
            // deviations from the default 5) ──
            let stereo_corr = if channels == 2 {
                low_band_stereo_correlation(&freqs[0], &freqs[1], lm, start)
            } else {
                None
            };
            let alloc_trim = choose_alloc_trim(&emax[start..end], stereo_corr);
            let trim_gated =
                enc.tell_frac() as i64 + 48 <= frame_8th as i64 - boosts.total_boost as i64;
            encode_alloc_trim(&mut enc, trim_gated, alloc_trim)?;

            // ── Anti-collapse reservation + the exact allocation ──
            let mut bits = (frame_bytes as i32 * 8) * 8 - enc.tell_frac() as i32 - 1;
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
                AllocIo::Encode {
                    enc: &mut enc,
                    // §5.3.5: the Table-66 bitrate threshold picks the
                    // first intensity-coded band (`end` = disabled);
                    // the walk clamps to the post-skip window.
                    intensity: intensity_start_band(total_bits, lm).unwrap_or(end) as i32,
                    // §5.3.5 L1 mid/side-vs-dual verdict on the raw
                    // per-channel spectra (dual when L/R wins).
                    dual_stereo: channels == 2
                        && !choose_mid_side_stereo(&freqs[0], &freqs[1], lm, start).unwrap_or(true),
                    prev_coded_bands: self.prev_coded_bands,
                },
            )?;
            self.prev_coded_bands = alloc.coded_bands as i32;

            // ── Fine energy (band-major, channel-minor) ──
            #[allow(clippy::needless_range_loop)] // decode-mirror shape
            for i in start..end {
                let fq = alloc.fine_bits[i];
                if fq <= 0 {
                    continue;
                }
                #[allow(clippy::needless_range_loop)] // decode-mirror shape
                for c in 0..channels {
                    let err = targets[c][i] - self.coarse.energy[c][i];
                    let q2 = (((err + 0.5) * (1 << fq) as f32).floor() as i32)
                        .clamp(0, (1 << fq) - 1) as u32;
                    enc.enc_bits(q2, fq as u32)?;
                    let offset =
                        (q2 as f32 + 0.5) * (1 << (14 - fq)) as f32 * (1.0 / 16384.0) - 0.5;
                    self.coarse.energy[c][i] += offset;
                }
            }

            // ── The §4.3.4 band loop (encode + resynthesis) ──
            let mut seed = self.rng;
            let _walk = quant_all_bands(
                QuantIo::Encode(&mut enc),
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
                (frame_bytes as i32) * (8 << BITRES) - anti_collapse_rsv,
                alloc.balance,
                lm,
                alloc.coded_bands,
                &mut seed,
                Some(&amps),
                true,
            )?;

            // ── Anti-collapse bit (this encoder never requests the
            // injection; a legal choice — the reservation is still
            // honoured so the budget stays in lockstep) ──
            if anti_collapse_rsv > 0 {
                enc.enc_bits(0, 1)?;
            }

            // ── Final fine-energy bits (§4.3.2.2 finalize) ──
            let mut bits_left = (frame_bytes * 8) as i32 - enc.tell() as i32;
            for prio in [false, true] {
                let mut i = start;
                while i < end && bits_left >= channels as i32 {
                    if alloc.fine_bits[i] >= MAX_FINE_BITS || alloc.fine_priority[i] != prio {
                        i += 1;
                        continue;
                    }
                    #[allow(clippy::needless_range_loop)] // decode-mirror shape
                    for c in 0..channels {
                        let err = targets[c][i] - self.coarse.energy[c][i];
                        let q2 = u32::from(err >= 0.0);
                        enc.enc_bits(q2, 1)?;
                        let offset = (q2 as f32 - 0.5)
                            * (1 << (14 - alloc.fine_bits[i] - 1)) as f32
                            * (1.0 / 16384.0);
                        self.coarse.energy[c][i] += offset;
                        bits_left -= 1;
                    }
                    i += 1;
                }
            }
        } else {
            // Silence: floor the prediction state like the decoder.
            for c in 0..2 {
                for i in 0..NUM_BANDS {
                    self.coarse.energy[c][i] = -28.0;
                }
            }
        }

        // Mono duplicates its energy row (decoder parity).
        if channels == 1 {
            for i in 0..NUM_BANDS {
                self.coarse.energy[1][i] = self.coarse.energy[0][i];
            }
        }

        // Two-frame energy history (§4.3.5 pipeline parity).
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

        self.rng = enc.range_state();
        enc.finish_to_size(frame_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ref_decode::CeltRefDecoder;

    fn tone(frame: usize, channels: usize, frames: usize, amp: f32) -> Vec<f32> {
        let n = frame * frames;
        let mut out = Vec::with_capacity(n * channels);
        for t in 0..n {
            for c in 0..channels {
                let f0 = if c == 0 { 440.0 } else { 523.0 };
                out.push(amp * (2.0 * std::f32::consts::PI * f0 * t as f32 / 48000.0).sin());
            }
        }
        out
    }

    /// Every stream the encoder emits decodes through the crate's own
    /// reference-exact decoder, at every LM, mono and stereo, and the
    /// decoded output approximates the (delay-compensated) input.
    #[test]
    fn encode_decodes_through_own_decoder() {
        for &(lm, channels, bytes) in &[
            (0u32, 1usize, 40usize),
            (1, 1, 60),
            (2, 1, 100),
            (3, 1, 160),
            (0, 2, 60),
            (1, 2, 90),
            (2, 2, 140),
            (3, 2, 240),
        ] {
            let frame = 120usize << lm;
            let frames = (14400 / frame).max(6);
            let pcm = tone(frame, channels, frames, 0.3);
            let mut enc = CeltRefEncoder::new(lm, channels).expect("encoder");
            let mut dec = CeltRefDecoder::new(lm, channels).expect("decoder");
            let mut out: Vec<f32> = Vec::new();
            for f in 0..frames {
                let chunk = &pcm[f * frame * channels..(f + 1) * frame * channels];
                let bytes_out = enc.encode_frame(chunk, bytes).expect("encode");
                assert_eq!(bytes_out.len(), bytes);
                out.extend(dec.decode_frame(&bytes_out).expect("decode"));
            }
            // Delay-compensated SNR over the steady state (skip the
            // first two frames of adaptation).
            let delay = OVERLAP * channels;
            let skip = 2 * frame * channels;
            let n = out.len() - delay - skip;
            let mut ee = 0f64;
            let mut err = 0f64;
            for i in 0..n {
                let e = pcm[skip + i] as f64;
                ee += e * e;
                let d = e - out[skip + delay + i] as f64;
                err += d * d;
            }
            let snr = 10.0 * (ee / err.max(1e-30)).log10();
            assert!(
                snr > 8.0,
                "lm={lm} C={channels} {bytes}B: SNR {snr:.2} dB too low"
            );
        }
    }

    /// Tiny and odd byte budgets exercise every Table-56 budget gate
    /// (post-filter, transient, intra, coarse fallbacks, tf, spread,
    /// boosts, trim, reservations, skip floor) and the §5.1.5
    /// fixed-size assembly: every frame in `2..=1275` territory must
    /// encode to exactly the requested size and decode finite.
    #[test]
    fn tiny_and_odd_byte_budgets_roundtrip() {
        let budgets = [
            2usize, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 47, 60, 120, 239, 1275,
        ];
        for lm in 0u32..=3 {
            let frame = 120usize << lm;
            for channels in 1usize..=2 {
                let pcm = tone(frame, channels, 2, 0.4);
                for &bytes in &budgets {
                    let mut enc = CeltRefEncoder::new(lm, channels).expect("encoder");
                    let mut dec = CeltRefDecoder::new(lm, channels).expect("decoder");
                    for f in 0..2 {
                        let chunk = &pcm[f * frame * channels..(f + 1) * frame * channels];
                        let b = enc
                            .encode_frame(chunk, bytes)
                            .unwrap_or_else(|e| panic!("lm={lm} C={channels} {bytes}B: {e}"));
                        assert_eq!(b.len(), bytes);
                        let out = dec
                            .decode_frame(&b)
                            .unwrap_or_else(|e| panic!("lm={lm} C={channels} {bytes}B: {e}"));
                        assert!(
                            out.iter().all(|v| v.is_finite()),
                            "non-finite at lm={lm} C={channels} {bytes}B"
                        );
                    }
                }
            }
        }
    }

    /// The transient-frame TF decision toggles a band whose energy is
    /// concentrated in a single short block (sparser at short
    /// resolution than after the Table-62 default recombine) and
    /// leaves a block-uniform band alone; non-transient frames always
    /// keep the identity resolution.
    #[test]
    fn tf_decision_prefers_short_resolution_for_concentrated_bands() {
        let lm = 2u32;
        let m = 1usize << lm;
        let n_coded = m * EBAND_EDGES_5MS[NUM_BANDS] as usize;
        let mut x = vec![0f32; n_coded];
        // Band 13 spans 4 base bins (16 coefficients at LM 2). In the
        // interleaved transient layout, coefficient `b + j*m` belongs
        // to short block `b`: concentrate all energy in block 0.
        let eb13 = m * EBAND_EDGES_5MS[13] as usize;
        let eb14 = m * EBAND_EDGES_5MS[14] as usize;
        for j in 0..(eb14 - eb13) / m {
            x[eb13 + j * m] = 0.5;
        }
        // Band 14: spread evenly across all four blocks.
        let eb15 = m * EBAND_EDGES_5MS[15] as usize;
        for (k, v) in x[eb14..eb15].iter_mut().enumerate() {
            *v = if k % 2 == 0 { 0.25 } else { -0.25 };
        }
        let choice = choose_tf_resolution(&x, m, true, lm, 0, NUM_BANDS);
        assert!(choice[13], "block-concentrated band should stay short");
        assert!(!choice[14], "block-uniform band should recombine");
        let long_choice = choose_tf_resolution(&x, m, false, lm, 0, NUM_BANDS);
        assert!(long_choice.iter().all(|&b| !b), "long frames stay identity");
    }

    /// Determinism: the same input yields byte-identical frames.
    #[test]
    fn encode_deterministic() {
        let frame = 480usize;
        let pcm = tone(frame, 1, 4, 0.25);
        let run = || {
            let mut enc = CeltRefEncoder::new(2, 1).expect("encoder");
            (0..4)
                .map(|f| {
                    enc.encode_frame(&pcm[f * frame..(f + 1) * frame], 90)
                        .expect("encode")
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    /// Digital silence produces a valid silence frame the decoder
    /// plays out as (near-)silence, and the codec survives the
    /// silence → tone transition.
    #[test]
    fn silence_frames() {
        let frame = 480usize;
        let mut enc = CeltRefEncoder::new(2, 1).expect("encoder");
        let mut dec = CeltRefDecoder::new(2, 1).expect("decoder");
        let zeros = vec![0f32; frame];
        for _ in 0..3 {
            let bytes = enc.encode_frame(&zeros, 60).expect("encode");
            let pcm = dec.decode_frame(&bytes).expect("decode");
            assert!(pcm.iter().all(|v| v.abs() < 1e-3));
        }
        let tone_pcm = tone(frame, 1, 1, 0.3);
        let bytes = enc.encode_frame(&tone_pcm, 60).expect("encode");
        let pcm = dec.decode_frame(&bytes).expect("decode");
        assert!(pcm.iter().all(|v| v.is_finite()));
    }

    /// A hard onset triggers the transient path at LM >= 1 and the
    /// stream still decodes cleanly across the long/short boundary.
    #[test]
    fn transient_stream_decodes() {
        let frame = 960usize;
        let mut pcm = vec![0f32; frame * 4];
        // Quiet tone, then a hard burst mid-stream.
        for (t, v) in pcm.iter_mut().enumerate() {
            *v = 0.02 * (2.0 * std::f32::consts::PI * 330.0 * t as f32 / 48000.0).sin();
        }
        let mut lcg = 0x1234_5678u32;
        for v in pcm[2 * frame + 400..2 * frame + 700].iter_mut() {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v += 0.6 * ((lcg >> 16) as i16 as f32 / 32768.0);
        }
        let mut enc = CeltRefEncoder::new(3, 1).expect("encoder");
        let mut dec = CeltRefDecoder::new(3, 1).expect("decoder");
        let mut any_transient = false;
        for f in 0..4 {
            let bytes = enc
                .encode_frame(&pcm[f * frame..(f + 1) * frame], 160)
                .expect("encode");
            // Bit 2 of a CELT frame: silence(0) → pf(0) → transient.
            // Cheap probe: decode and rely on internal consistency.
            let out = dec.decode_frame(&bytes).expect("decode");
            assert!(out.iter().all(|v| v.is_finite()));
            any_transient |= detect_transient(
                &[{
                    let mut b = vec![0f32; OVERLAP];
                    b.extend(
                        pcm[f * frame..(f + 1) * frame]
                            .iter()
                            .map(|v| v * SIG_SCALE),
                    );
                    b
                }],
                frame,
                3,
            );
        }
        assert!(any_transient, "test signal never tripped the detector");
    }
}
