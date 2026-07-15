//! The reference-exact §4.3.4 band-quantization loop (RFC 6716) —
//! per-band bit targeting with the running balance, the §4.3.4.4
//! band splitting with the quantized split angle `itheta`, the
//! §4.3.4.5 time-frequency pre/post transforms, the §4.3.4.3
//! spreading rotation, spectral folding with the LCG noise fill, the
//! stereo mid/side and intensity paths, and the per-band collapse
//! masks the §4.3.5 anti-collapse pass consumes.
//!
//! ## Provenance
//!
//! Transcribed from the **normative RFC 6716 Appendix A reference
//! listing** (`bands.c`, `vq.c`, `rate.h`, `mathops.c`), extracted
//! from the staged `docs/audio/opus/rfc6716-opus.txt` per §A.1 and
//! SHA-1-verified against the §A.1-printed value
//! (`86a927223e73d2476646a1b933fcd3fffb6ecc8c`). RFC 6716 §6 makes
//! the decoder side of that listing normative; §§4.3.4.1–4.3.4.5 are
//! the prose narrative. All arithmetic follows the listing's
//! float-build semantics (the crate's native `f32` path); the
//! bit-exact integer helpers (`bitexact_cos`, `bitexact_log2tan`,
//! `isqrt32`, `compute_qn`) are integer-exact on both builds. The
//! `ordery`, `bit_interleave`, `bit_deinterleave`, and `exp2` tables
//! cross-check against the staged extractions under
//! `docs/audio/opus/tables/` (`ordery-table.csv`,
//! `bit-interleave-table.csv`, `bit-deinterleave-table.csv`,
//! `exp2-table8.csv`).

use crate::alloc_exact::{
    bits2pulses, cache_row, get_pulses, pulses2bits, BITRES, LOG_N400, QTHETA_OFFSET,
    QTHETA_OFFSET_TWOPHASE,
};
use crate::band_layout::EBAND_EDGES_5MS;
use crate::coarse_energy::NUM_BANDS;
use crate::pvq::{decode_pulses, encode_pulses, pvq_search};
use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::spread::Spread;
use crate::Error;

/// The §4.3.5 linear congruential generator (Appendix A `bands.c`
/// `celt_lcg_rand`).
#[inline]
pub fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// `FRAC_MUL16` (Appendix A `mathops.h`): rounded Q15 product on
/// 16-bit-truncated operands — `(16384 + (i16)a * (i16)b) >> 15`.
#[inline]
fn frac_mul16(a: i32, b: i32) -> i32 {
    (16384 + (a as i16 as i32) * (b as i16 as i32)) >> 15
}

/// The bit-exact cosine approximation the split-angle mid/side gains
/// are derived from (Appendix A `bands.c` `bitexact_cos`).
pub fn bitexact_cos(x: i32) -> i32 {
    let tmp = (4096 + x * x) >> 13;
    let x2 = tmp;
    let x2 = (32767 - x2) + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2)));
    1 + x2
}

/// The bit-exact `log2(tan)` approximation feeding the §4.3.4.4
/// mid/side allocation offset (Appendix A `bands.c`
/// `bitexact_log2tan`).
pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = ec_ilog(icos as u32) as i32;
    let ls = ec_ilog(isin as u32) as i32;
    let icos = icos << (15 - lc);
    let isin = isin << (15 - ls);
    (ls - lc) * (1 << 11) + frac_mul16(isin, frac_mul16(isin, -2597) + 7932)
        - frac_mul16(icos, frac_mul16(icos, -2597) + 7932)
}

/// `EC_ILOG`: index of the highest set bit plus one (`0` for `0`).
#[inline]
fn ec_ilog(v: u32) -> u32 {
    32 - v.leading_zeros()
}

/// Exact integer `floor(sqrt(v))` (Appendix A `mathops.c` `isqrt32`).
pub fn isqrt32(mut val: u32) -> u32 {
    if val == 0 {
        return 0;
    }
    let mut g: u32 = 0;
    let mut bshift: i32 = ((ec_ilog(val) - 1) >> 1) as i32;
    let mut b: u32 = 1 << bshift;
    loop {
        let t = ((g << 1) + b) << bshift;
        if t <= val {
            g += b;
            val -= t;
        }
        b >>= 1;
        bshift -= 1;
        if bshift < 0 {
            break;
        }
    }
    g
}

/// The natural→"ordery" Hadamard reindexing table (Appendix A
/// `bands.c`; staged as `docs/audio/opus/tables/ordery-table.csv`).
/// Rows for strides 2, 4, 8, 16, concatenated.
const ORDERY_TABLE: [usize; 30] = [
    1, 0, // stride 2
    3, 0, 2, 1, // stride 4
    7, 0, 4, 3, 6, 1, 5, 2, // stride 8
    15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5, // stride 16
];

fn ordery(stride: usize) -> &'static [usize] {
    let off = stride - 2;
    &ORDERY_TABLE[off..off + stride]
}

/// The §4.3.4.5 Haar transform on interleaved blocks (Appendix A
/// `bands.c` `haar1`): orthonormal butterflies at the given stride.
pub fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    const SQRT_HALF: f32 = core::f32::consts::FRAC_1_SQRT_2;
    let n0 = n0 >> 1;
    for i in 0..stride {
        for j in 0..n0 {
            let a = stride * 2 * j + i;
            let b = stride * (2 * j + 1) + i;
            let tmp1 = SQRT_HALF * x[a];
            let tmp2 = SQRT_HALF * x[b];
            x[a] = tmp1 + tmp2;
            x[b] = tmp1 - tmp2;
        }
    }
}

/// Reorganize interleaved samples into time order (Appendix A
/// `bands.c` `deinterleave_hadamard`).
fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0f32; n];
    if hadamard {
        let ord = ordery(stride);
        for (i, &oi) in ord.iter().enumerate() {
            for j in 0..n0 {
                tmp[oi * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// Undo [`deinterleave_hadamard`] (Appendix A `bands.c`
/// `interleave_hadamard`).
fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0f32; n];
    if hadamard {
        let ord = ordery(stride);
        for (i, &oi) in ord.iter().enumerate() {
            for j in 0..n0 {
                tmp[j * stride + i] = x[oi * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

/// The split-angle resolution (Appendix A `bands.c` `compute_qn`).
/// `exp2_table8` cross-checks against the staged
/// `docs/audio/opus/tables/exp2-table8.csv`.
pub fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    const EXP2_TABLE8: [i32; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    // The upper limit ensures a stereo split with itheta == 16384 has
    // enough bits left to code at least one side pulse; otherwise the
    // side would collapse (it is not folded).
    let mut qb = (b - pulse_cap - (4 << BITRES)).min((b + n2 * offset) / n2);
    qb = (8 << BITRES).min(qb);
    let qn = if qb < (1 << BITRES >> 1) {
        1
    } else {
        let q = EXP2_TABLE8[(qb & 0x7) as usize] >> (14 - (qb >> BITRES));
        ((q + 1) >> 1) << 1
    };
    debug_assert!(qn <= 256);
    qn
}

/// One §4.3.4.3 rotation pass (Appendix A `vq.c` `exp_rotation1`).
fn exp_rotation1(x: &mut [f32], stride: usize, c: f32, s: f32) {
    let len = x.len();
    if len < stride {
        return;
    }
    for i in 0..len - stride {
        let x1 = x[i];
        let x2 = x[i + stride];
        x[i + stride] = c * x2 + s * x1;
        x[i] = c * x1 - s * x2;
    }
    if len > 2 * stride {
        for i in (0..len - 2 * stride).rev() {
            let x1 = x[i];
            let x2 = x[i + stride];
            x[i + stride] = c * x2 + s * x1;
            x[i] = c * x1 - s * x2;
        }
    }
}

/// The §4.3.4.3 spreading rotation (Appendix A `vq.c`
/// `exp_rotation`): `dir = 1` before the encoder's PVQ search,
/// `dir = -1` after the decoder's PVQ reconstruction.
pub fn exp_rotation(x: &mut [f32], dir: i32, stride: usize, k: i32, spread: Spread) {
    let len = x.len() as i32;
    let factor = match spread {
        Spread::None => return,
        Spread::Light => 15,
        Spread::Normal => 10,
        Spread::Aggressive => 5,
    };
    if 2 * k >= len || k <= 0 {
        return;
    }
    let gain = len as f32 / (len + factor * k) as f32;
    let theta = 0.5 * gain * gain;
    let c = (0.5 * core::f32::consts::PI * theta).cos();
    let s = (0.5 * core::f32::consts::PI * (1.0 - theta)).cos();

    let mut stride2: usize = 0;
    if len >= 8 * stride as i32 {
        stride2 = 1;
        // Equivalent to sqrt(len/stride) with rounding: increment
        // while (stride2 + 0.5)^2 < len/stride.
        while ((stride2 * stride2 + stride2) * stride + (stride >> 2)) < len as usize {
            stride2 += 1;
        }
    }
    let sub = (len as usize) / stride;
    for i in 0..stride {
        let blk = &mut x[i * sub..(i + 1) * sub];
        if dir < 0 {
            if stride2 != 0 {
                exp_rotation1(blk, stride2, s, c);
            }
            exp_rotation1(blk, 1, c, s);
        } else {
            exp_rotation1(blk, 1, c, -s);
            if stride2 != 0 {
                exp_rotation1(blk, stride2, s, -c);
            }
        }
    }
}

/// Mix a decoded pulse vector to the unit-gain band shape (Appendix A
/// `vq.c` `normalise_residual`, float build).
fn normalise_residual(iy: &[i32], x: &mut [f32], ryy: f32, gain: f32) {
    let g = gain / ryy.sqrt();
    for (o, &p) in x.iter_mut().zip(iy.iter()) {
        *o = g * p as f32;
    }
}

/// Per-block non-zero mask of the pulse vector (Appendix A `vq.c`
/// `extract_collapse_mask`).
fn extract_collapse_mask(iy: &[i32], b: usize) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = iy.len() / b;
    let mut mask = 0u32;
    for i in 0..b {
        for j in 0..n0 {
            if iy[i * n0 + j] != 0 {
                mask |= 1 << i;
                break;
            }
        }
    }
    mask
}

/// Renormalize a band to the target gain (Appendix A `vq.c`
/// `renormalise_vector`, float build).
pub fn renormalise_vector(x: &mut [f32], gain: f32) {
    let mut e = 1e-15f64;
    for &v in x.iter() {
        e += f64::from(v) * f64::from(v);
    }
    let g = gain / (e as f32).sqrt();
    for v in x.iter_mut() {
        *v *= g;
    }
}

/// The encoder's split-angle measurement (Appendix A `vq.c`
/// `stereo_itheta`, float build). The `0.63662` scale is the
/// listing's own literal (a truncation of 2/pi — kept verbatim, not
/// replaced with the exact constant, to stay faithful).
#[allow(clippy::approx_constant)]
fn stereo_itheta(x: &[f32], y: &[f32], stereo: bool) -> i32 {
    let mut emid = 1e-15f32;
    let mut eside = 1e-15f32;
    if stereo {
        for (&l, &r) in x.iter().zip(y.iter()) {
            let m = 0.5 * l + 0.5 * r;
            let s = 0.5 * l - 0.5 * r;
            emid += m * m;
            eside += s * s;
        }
    } else {
        for &v in x.iter() {
            emid += v * v;
        }
        for &v in y.iter() {
            eside += v * v;
        }
    }
    let mid = emid.sqrt();
    let side = eside.sqrt();
    (0.5f64 + 16384.0f64 * 0.63662f64 * f64::from(side.atan2(mid))).floor() as i32
}

/// Orthonormal mid/side rotation applied by the stereo encoder
/// (Appendix A `bands.c` `stereo_split`).
fn stereo_split(x: &mut [f32], y: &mut [f32]) {
    const SQRT_HALF: f32 = core::f32::consts::FRAC_1_SQRT_2;
    for (l0, r0) in x.iter_mut().zip(y.iter_mut()) {
        let l = SQRT_HALF * *l0;
        let r = SQRT_HALF * *r0;
        *l0 = l + r;
        *r0 = r - l;
    }
}

/// Undo the mid/side coupling into unit-norm left/right (Appendix A
/// `bands.c` `stereo_merge`, float build).
fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32) {
    let mut xp = 0f32;
    let mut side = 0f32;
    for (&l, &r) in x.iter().zip(y.iter()) {
        xp += l * r;
        side += r * r;
    }
    // Compensating for the mid normalization.
    xp *= mid;
    let el = mid * mid + side - 2.0 * xp;
    let er = mid * mid + side + 2.0 * xp;
    if er < 6e-4 || el < 6e-4 {
        y.copy_from_slice(x);
        return;
    }
    let lgain = 1.0 / el.sqrt();
    let rgain = 1.0 / er.sqrt();
    for (l0, r0) in x.iter_mut().zip(y.iter_mut()) {
        // Apply mid scaling (side is already scaled).
        let l = mid * *l0;
        let r = *r0;
        *l0 = lgain * (l - r);
        *r0 = rgain * (l + r);
    }
}

/// Encoder-side intensity collapse: replace the mid with the
/// energy-weighted mono downmix (Appendix A `bands.c`
/// `intensity_stereo`, float build).
fn intensity_stereo(x: &mut [f32], y: &[f32], left_amp: f32, right_amp: f32) {
    let norm = 1e-15 + (1e-15 + left_amp * left_amp + right_amp * right_amp).sqrt();
    let a1 = left_amp / norm;
    let a2 = right_amp / norm;
    for (l, &r) in x.iter_mut().zip(y.iter()) {
        *l = a1 * *l + a2 * r;
    }
}

/// The recombine-loop collapse-mask interleave map (Appendix A
/// `bands.c`; staged as
/// `docs/audio/opus/tables/bit-interleave-table.csv`).
const BIT_INTERLEAVE_TABLE: [u32; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];

/// The inverse recombine collapse-mask spread map (Appendix A
/// `bands.c`; staged as
/// `docs/audio/opus/tables/bit-deinterleave-table.csv`).
const BIT_DEINTERLEAVE_TABLE: [u32; 16] = [
    0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3, 0xFC, 0xFF,
];

/// Range-coder attachment for the band loop: decode reads symbols,
/// encode writes the encoder's choices at identical positions.
#[allow(missing_debug_implementations)] // carries raw coder state
pub enum QuantIo<'a, 'b> {
    /// Decode side.
    Decode(&'a mut RangeDecoder<'b>),
    /// Encode side.
    Encode(&'a mut RangeEncoder),
}

impl QuantIo<'_, '_> {
    fn is_encode(&self) -> bool {
        matches!(self, QuantIo::Encode(_))
    }

    fn tell_frac(&self) -> u32 {
        match self {
            QuantIo::Decode(d) => d.tell_frac(),
            QuantIo::Encode(e) => e.tell_frac(),
        }
    }

    /// Raw bits (§4.1.4 / §5.1.3). Encode writes `val`; decode
    /// ignores it. Returns the coded value.
    fn bits(&mut self, val: u32, n: u32) -> Result<u32, Error> {
        match self {
            QuantIo::Decode(d) => Ok(d.dec_bits(n)),
            QuantIo::Encode(e) => {
                e.enc_bits(val, n)?;
                Ok(val)
            }
        }
    }

    /// One binary symbol with probability `2^-logp` of a `1`.
    fn bit_logp(&mut self, val: bool, logp: u32) -> Result<bool, Error> {
        match self {
            QuantIo::Decode(d) => Ok(d.dec_bit_logp(logp) == 1),
            QuantIo::Encode(e) => {
                e.enc_bit_logp(u32::from(val), logp)?;
                Ok(val)
            }
        }
    }

    /// Uniform integer in `0..ft`.
    fn uint(&mut self, val: u32, ft: u32) -> Result<u32, Error> {
        match self {
            QuantIo::Decode(d) => d.dec_uint(ft),
            QuantIo::Encode(e) => {
                e.enc_uint(val, ft)?;
                Ok(val)
            }
        }
    }
}

/// Per-channel band amplitudes (`bandE`) the encoder's intensity
/// collapse consults; the decoder never needs them.
pub type BandAmps = [[f32; NUM_BANDS]; 2];

/// Everything that stays constant across one frame's band walk.
struct FrameCtx<'c> {
    spread: Spread,
    intensity: i32,
    band_amps: Option<&'c BandAmps>,
    resynth: bool,
}

/// The §4.3.4.2 leaf PVQ codec (Appendix A `vq.c` `alg_quant` /
/// `alg_unquant`). Returns the collapse mask.
fn alg_pvq(
    io: &mut QuantIo<'_, '_>,
    ctx: &FrameCtx<'_>,
    x: &mut [f32],
    k: i32,
    spread: Spread,
    b: usize,
    gain: f32,
) -> Result<u32, Error> {
    let n = x.len() as u32;
    match io {
        QuantIo::Encode(enc) => {
            exp_rotation(x, 1, b, k, spread);
            let iy = pvq_search(x, n, k as u32).ok_or(Error::NotImplemented)?;
            encode_pulses(enc, &iy, n, k as u32)?;
            if ctx.resynth {
                let ryy: f32 = iy.iter().map(|&v| (v * v) as f32).sum();
                normalise_residual(&iy, x, ryy, gain);
                exp_rotation(x, -1, b, k, spread);
            }
            Ok(extract_collapse_mask(&iy, b))
        }
        QuantIo::Decode(dec) => {
            let iy = decode_pulses(dec, n, k as u32).ok_or(Error::NotImplemented)?;
            let ryy: f32 = iy.iter().map(|&v| (v * v) as f32).sum();
            normalise_residual(&iy, x, ryy, gain);
            exp_rotation(x, -1, b, k, spread);
            Ok(extract_collapse_mask(&iy, b))
        }
    }
}

/// Arguments that flow through the recursive band walk.
#[allow(clippy::too_many_arguments)]
fn quant_band(
    io: &mut QuantIo<'_, '_>,
    ctx: &FrameCtx<'_>,
    band: usize,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    mut b: i32,
    mut b_blocks: usize, // B
    tf_change: i32,
    mut lowband: Option<Vec<f32>>,
    remaining_bits: &mut i32,
    lm: i32,
    mut lowband_out: Option<&mut [f32]>,
    level: i32,
    seed: &mut u32,
    gain: f32,
    mut fill: u32,
) -> Result<u32, Error> {
    let n0 = x.len();
    let mut n = n0;
    let stereo = y.is_some();
    let encode = io.is_encode();
    let resynth = ctx.resynth || !encode;
    let long_blocks = b_blocks == 1;
    let mut n_b = n / b_blocks;
    let mut b0 = b_blocks;
    let n_b0;
    let mut time_divide = 0usize;
    let mut recombine = 0i32;
    let mut tf_change = tf_change;

    // Special case for one sample.
    if n == 1 {
        let mut cm = 1u32;
        {
            let mut do_one =
                |io: &mut QuantIo<'_, '_>, xs: &mut [f32], b: &mut i32| -> Result<(), Error> {
                    let mut sign = 0u32;
                    if *remaining_bits >= 1 << BITRES {
                        if encode {
                            sign = u32::from(xs[0] < 0.0);
                        }
                        sign = io.bits(sign, 1)?;
                        *remaining_bits -= 1 << BITRES;
                        *b -= 1 << BITRES;
                    }
                    if resynth {
                        xs[0] = if sign == 1 { -1.0 } else { 1.0 };
                    }
                    Ok(())
                };
            do_one(io, x, &mut b)?;
            if let Some(ys) = y.as_deref_mut() {
                do_one(io, ys, &mut b)?;
            }
        }
        if let Some(out) = lowband_out.as_deref_mut() {
            out[0] = x[0];
        }
        let _ = cm;
        cm = 1;
        return Ok(cm);
    }

    if !stereo && level == 0 {
        if tf_change > 0 {
            recombine = tf_change;
        }
        // Band recombining to increase frequency resolution.
        for k in 0..recombine {
            if encode {
                haar1(x, n >> k, 1 << k);
            }
            if let Some(lb) = lowband.as_deref_mut() {
                haar1(lb, n >> k, 1 << k);
            }
            fill = BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize]
                | (BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] << 2);
        }
        b_blocks >>= recombine;
        n_b <<= recombine;

        // Increasing the time resolution.
        while (n_b & 1) == 0 && tf_change < 0 {
            if encode {
                haar1(x, n_b, b_blocks);
            }
            if let Some(lb) = lowband.as_deref_mut() {
                haar1(lb, n_b, b_blocks);
            }
            fill |= fill << b_blocks;
            b_blocks <<= 1;
            n_b >>= 1;
            time_divide += 1;
            tf_change += 1;
        }
        b0 = b_blocks;
        n_b0 = n_b;

        // Reorganize the samples in time order instead of frequency
        // order.
        if b0 > 1 {
            if encode {
                deinterleave_hadamard(x, n_b >> recombine, b0 << recombine, long_blocks);
            }
            if let Some(lb) = lowband.as_deref_mut() {
                deinterleave_hadamard(lb, n_b >> recombine, b0 << recombine, long_blocks);
            }
        }
    } else {
        n_b0 = n_b;
    }

    // If we need 1.5 more bits than the band can produce, split it.
    let mut split = stereo;
    let mut split_lm = lm;
    if !stereo && lm != -1 && n > 2 {
        if let Some(cache) = cache_row(band, lm) {
            let max_cost = cache[cache[0] as usize] as i32;
            if b > max_cost + 12 {
                n >>= 1;
                split = true;
                split_lm = lm - 1;
                if b_blocks == 1 {
                    fill = (fill & 1) | (fill << 1);
                }
                b_blocks = b_blocks.div_ceil(2);
            }
        }
    }

    let mut cm: u32;
    if split {
        let lm = split_lm;
        // Decide on the resolution of the split parameter theta.
        let pulse_cap = LOG_N400[band] as i32 + lm * (1 << BITRES);
        let offset = (pulse_cap >> 1)
            - if stereo && n == 2 {
                QTHETA_OFFSET_TWOPHASE
            } else {
                QTHETA_OFFSET
            };
        let mut qn = compute_qn(n as i32, b, offset, pulse_cap, stereo);
        if stereo && band as i32 >= ctx.intensity {
            qn = 1;
        }

        // Split the two operand vectors: for stereo they are the two
        // channels; for a mono split they are the band's two halves.
        let (xs, ys): (&mut [f32], &mut [f32]) = if stereo {
            (&mut *x, y.take().unwrap())
        } else {
            let (a, bb) = x.split_at_mut(n);
            (a, bb)
        };

        let mut itheta: i32 = 0;
        if encode {
            itheta = stereo_itheta(xs, ys, stereo);
        }
        let tell = io.tell_frac() as i32;
        let mut inv = false;
        if qn != 1 {
            if encode {
                itheta = (itheta * qn + 8192) >> 14;
            }
            // Entropy coding of the angle: a step PDF for stereo, a
            // uniform PDF for the time split, a triangular one
            // otherwise.
            if stereo && n > 2 {
                let p0 = 3i32;
                let x0 = qn / 2;
                let ft = (p0 * (x0 + 1) + x0) as u32;
                match io {
                    QuantIo::Encode(enc) => {
                        let xv = itheta;
                        let (fl, fh) = if xv <= x0 {
                            ((p0 * xv) as u32, (p0 * (xv + 1)) as u32)
                        } else {
                            (
                                ((xv - 1 - x0) + (x0 + 1) * p0) as u32,
                                ((xv - x0) + (x0 + 1) * p0) as u32,
                            )
                        };
                        enc.encode(fl, fh, ft)?;
                    }
                    QuantIo::Decode(dec) => {
                        let fs = dec.decode(ft) as i32;
                        let xv = if fs < (x0 + 1) * p0 {
                            fs / p0
                        } else {
                            x0 + 1 + (fs - (x0 + 1) * p0)
                        };
                        let (fl, fh) = if xv <= x0 {
                            ((p0 * xv) as u32, (p0 * (xv + 1)) as u32)
                        } else {
                            (
                                ((xv - 1 - x0) + (x0 + 1) * p0) as u32,
                                ((xv - x0) + (x0 + 1) * p0) as u32,
                            )
                        };
                        dec.dec_update(fl, fh, ft);
                        itheta = xv;
                    }
                }
            } else if b0 > 1 || stereo {
                // Uniform PDF.
                itheta = io.uint(itheta as u32, qn as u32 + 1)? as i32;
            } else {
                // Triangular PDF.
                let half = qn >> 1;
                let ft = ((half + 1) * (half + 1)) as u32;
                match io {
                    QuantIo::Encode(enc) => {
                        let (fs, fl) = if itheta <= half {
                            (itheta + 1, (itheta * (itheta + 1)) >> 1)
                        } else {
                            (
                                qn + 1 - itheta,
                                ft as i32 - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1),
                            )
                        };
                        enc.encode(fl as u32, (fl + fs) as u32, ft)?;
                    }
                    QuantIo::Decode(dec) => {
                        let fm = dec.decode(ft) as i32;
                        let (fs, fl);
                        if fm < (half * (half + 1)) >> 1 {
                            itheta = ((isqrt32(8 * fm as u32 + 1) as i32) - 1) >> 1;
                            fs = itheta + 1;
                            fl = (itheta * (itheta + 1)) >> 1;
                        } else {
                            itheta =
                                (2 * (qn + 1) - isqrt32(8 * (ft - fm as u32 - 1) + 1) as i32) >> 1;
                            fs = qn + 1 - itheta;
                            fl = ft as i32 - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                        }
                        dec.dec_update(fl as u32, (fl + fs) as u32, ft);
                    }
                }
            }
            debug_assert!(itheta >= 0);
            itheta = itheta * 16384 / qn;
            if encode && stereo {
                if itheta == 0 {
                    let amps = ctx.band_amps.ok_or(Error::InvalidParameter)?;
                    intensity_stereo(xs, ys, amps[0][band], amps[1][band]);
                } else {
                    stereo_split(xs, ys);
                }
            }
        } else if stereo {
            // qn == 1: only the inversion flag may be coded.
            if encode {
                inv = itheta > 8192;
                if inv {
                    for v in ys.iter_mut() {
                        *v = -*v;
                    }
                }
                let amps = ctx.band_amps.ok_or(Error::InvalidParameter)?;
                intensity_stereo(xs, ys, amps[0][band], amps[1][band]);
            }
            if b > 2 << BITRES && *remaining_bits > 2 << BITRES {
                inv = io.bit_logp(inv, 2)?;
            } else {
                inv = false;
            }
            itheta = 0;
        }
        let qalloc = io.tell_frac() as i32 - tell;
        b -= qalloc;

        let orig_fill = fill;
        let (imid, iside, mut delta);
        if itheta == 0 {
            imid = 32767;
            iside = 0;
            fill &= (1u32 << b_blocks) - 1;
            delta = -16384;
        } else if itheta == 16384 {
            imid = 0;
            iside = 32767;
            fill &= ((1u32 << b_blocks) - 1) << b_blocks;
            delta = 16384;
        } else {
            imid = bitexact_cos(itheta);
            iside = bitexact_cos(16384 - itheta);
            // The mid vs side allocation that minimizes squared error
            // in the band.
            delta = frac_mul16(((n - 1) << 7) as i32, bitexact_log2tan(iside, imid));
        }
        let mid = (1.0 / 32768.0) * imid as f32;
        let side = (1.0 / 32768.0) * iside as f32;

        if n == 2 && stereo {
            // Special stereo N=2 case: mid and side are orthogonal, so
            // one sign bit codes the side.
            let mut sbits = 0i32;
            if itheta != 0 && itheta != 16384 {
                sbits = 1 << BITRES;
            }
            let mbits = b - sbits;
            let c = itheta > 8192;
            *remaining_bits -= qalloc + sbits;

            // Work on an owned copy of the coded channel to keep the
            // two-channel reconstruction below straightforward.
            let mut x2: Vec<f32> = if c { ys.to_vec() } else { xs.to_vec() };
            let mut sign = 0u32;
            if sbits != 0 {
                if encode {
                    let y2 = if c { &*xs } else { &*ys };
                    sign = u32::from(x2[0] * y2[1] - x2[1] * y2[0] < 0.0);
                }
                sign = io.bits(sign, 1)?;
            }
            let signf = 1.0 - 2.0 * sign as f32;
            // orig_fill folds the side even when itheta cleared fill.
            cm = quant_band(
                io,
                ctx,
                band,
                &mut x2,
                None,
                mbits,
                b_blocks,
                tf_change,
                lowband.take(),
                remaining_bits,
                lm,
                lowband_out.take(),
                level,
                seed,
                gain,
                orig_fill,
            )?;
            // The side is the mid rotated by 90 degrees.
            let y2 = [-signf * x2[1], signf * x2[0]];
            if resynth {
                // Map (x2, y2) back onto the mid/side pair: X is the
                // mid channel unless itheta selected the swap.
                let (xv, yv): (&[f32], [f32; 2]) =
                    if c { (&y2, [x2[0], x2[1]]) } else { (&x2, y2) };
                let xm = [mid * xv[0], mid * xv[1]];
                let ysc = [side * yv[0], side * yv[1]];
                xs[0] = xm[0] - ysc[0];
                ys[0] = xm[0] + ysc[0];
                xs[1] = xm[1] - ysc[1];
                ys[1] = xm[1] + ysc[1];
                if inv {
                    for v in ys.iter_mut() {
                        *v = -*v;
                    }
                }
            } else if c {
                ys.copy_from_slice(&x2);
            } else {
                xs.copy_from_slice(&x2);
            }
        } else {
            // "Normal" split code.
            // Give more bits to low-energy MDCTs than they would
            // otherwise deserve.
            if b0 > 1 && !stereo && (itheta & 0x3fff) != 0 {
                if itheta > 8192 {
                    // Rough approximation for pre-echo masking.
                    delta -= delta >> (4 - lm);
                } else {
                    // A forward-masking slope of 1.5 dB per 10 ms.
                    delta = 0.min(delta + ((n as i32) << BITRES >> (5 - lm)));
                }
            }
            let mbits = 0.max(b.min((b - delta) / 2));
            let mut sbits = b - mbits;
            let mut mbits = mbits;
            *remaining_bits -= qalloc;

            // Split the folding source for the two halves (mono);
            // stereo passes the whole lowband to the mid and no
            // folding to the side.
            let (lowband_mid, lowband_side) = match (&lowband, stereo) {
                (Some(lb), false) if lb.len() >= 2 * n => {
                    (Some(lb[..n].to_vec()), Some(lb[n..2 * n].to_vec()))
                }
                (Some(lb), false) => (Some(lb.clone()), None),
                (Some(lb), true) => (Some(lb.clone()), None),
                (None, _) => (None, None),
            };

            let next_level = if stereo { level } else { level + 1 };
            let side_shift = if stereo { 0 } else { b0 >> 1 };

            let rebalance_start = *remaining_bits;
            if mbits >= sbits {
                // In stereo mode the mid gets no gain scaling: the
                // normalized mid is needed for folding later.
                let mid_gain = if stereo { 1.0 } else { gain * mid };
                cm = quant_band(
                    io,
                    ctx,
                    band,
                    xs,
                    None,
                    mbits,
                    b_blocks,
                    tf_change,
                    lowband_mid,
                    remaining_bits,
                    lm,
                    if stereo { lowband_out.take() } else { None },
                    next_level,
                    seed,
                    mid_gain,
                    fill,
                )?;
                let rebalance = mbits - (rebalance_start - *remaining_bits);
                if rebalance > 3 << BITRES && itheta != 0 {
                    sbits += rebalance - (3 << BITRES);
                }
                // For a stereo split the high fill bits are zero, so
                // the side never folds.
                cm |= quant_band(
                    io,
                    ctx,
                    band,
                    ys,
                    None,
                    sbits,
                    b_blocks,
                    tf_change,
                    lowband_side,
                    remaining_bits,
                    lm,
                    None,
                    next_level,
                    seed,
                    gain * side,
                    fill >> b_blocks,
                )? << side_shift;
            } else {
                cm = quant_band(
                    io,
                    ctx,
                    band,
                    ys,
                    None,
                    sbits,
                    b_blocks,
                    tf_change,
                    lowband_side,
                    remaining_bits,
                    lm,
                    None,
                    next_level,
                    seed,
                    gain * side,
                    fill >> b_blocks,
                )? << side_shift;
                let rebalance = sbits - (rebalance_start - *remaining_bits);
                if rebalance > 3 << BITRES && itheta != 16384 {
                    mbits += rebalance - (3 << BITRES);
                }
                let mid_gain = if stereo { 1.0 } else { gain * mid };
                cm |= quant_band(
                    io,
                    ctx,
                    band,
                    xs,
                    None,
                    mbits,
                    b_blocks,
                    tf_change,
                    lowband_mid,
                    remaining_bits,
                    lm,
                    if stereo { lowband_out.take() } else { None },
                    next_level,
                    seed,
                    mid_gain,
                    fill,
                )?;
            }

            // Resynthesis: undo the stereo coupling.
            if resynth && stereo {
                if n != 2 {
                    stereo_merge(xs, ys, mid);
                }
                if inv {
                    for v in ys.iter_mut() {
                        *v = -*v;
                    }
                }
            }
        }
    } else {
        // The basic no-split case.
        let q0 = bits2pulses(band, lm, b.min(16383)).ok_or(Error::NotImplemented)?;
        let mut q = q0;
        let mut curr_bits = pulses2bits(band, lm, q).ok_or(Error::NotImplemented)?;
        *remaining_bits -= curr_bits;
        // Never bust the budget.
        while *remaining_bits < 0 && q > 0 {
            *remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(band, lm, q).ok_or(Error::NotImplemented)?;
            *remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q);
            cm = alg_pvq(io, ctx, x, k, ctx.spread, b_blocks, gain)?;
        } else {
            // No pulses: fill the band anyway.
            cm = 0;
            if resynth {
                let cm_mask = (1u32 << b_blocks) - 1;
                fill &= cm_mask;
                if fill == 0 {
                    for v in x.iter_mut() {
                        *v = 0.0;
                    }
                } else {
                    match lowband.as_deref() {
                        None => {
                            // Noise-fill the band.
                            for v in x.iter_mut() {
                                *seed = celt_lcg_rand(*seed);
                                *v = ((*seed as i32) >> 20) as f32;
                            }
                            cm = cm_mask;
                        }
                        Some(lb) => {
                            // Folded spectrum, dithered ~48 dB below
                            // the folding level.
                            for (v, &l) in x.iter_mut().zip(lb.iter()) {
                                *seed = celt_lcg_rand(*seed);
                                let tmp = if *seed & 0x8000 != 0 {
                                    1.0 / 256.0
                                } else {
                                    -1.0 / 256.0
                                };
                                *v = l + tmp;
                            }
                            cm = fill;
                        }
                    }
                    renormalise_vector(x, gain);
                }
            }
        }
    }

    // Decoder-side (and resynthesising-encoder) post pass.
    if resynth && !stereo && level == 0 {
        // Undo the time-order reorganization.
        if b0 > 1 {
            interleave_hadamard(x, n_b >> recombine, b0 << recombine, long_blocks);
        }
        // Undo the time-resolution increases.
        let mut n_b = n_b0;
        let mut b_cur = b0;
        for _ in 0..time_divide {
            b_cur >>= 1;
            n_b <<= 1;
            cm |= cm >> b_cur;
            haar1(x, n_b, b_cur);
        }
        // Undo the recombining.
        for k in 0..recombine {
            cm = BIT_DEINTERLEAVE_TABLE[(cm & 0xF) as usize];
            haar1(x, n0 >> k, 1 << k);
        }
        let b_final = b_cur << recombine;

        // Scale the output for later folding.
        if let Some(out) = lowband_out {
            let nscale = (n0 as f32).sqrt();
            for (o, &v) in out.iter_mut().zip(x.iter()) {
                *o = nscale * v;
            }
        }
        cm &= (1u32 << b_final) - 1;
    }
    Ok(cm)
}

/// Outputs of the frame band walk.
#[derive(Debug, Clone)]
pub struct BandWalkResult {
    /// Per-band per-channel collapse masks
    /// (`collapse_masks[band * channels + channel]`).
    pub collapse_masks: Vec<u8>,
}

/// The reference-exact §4.3.4 frame band loop (Appendix A `bands.c`
/// `quant_all_bands`).
///
/// * `x` / `y` — the per-channel normalized spectra over the full
///   coded range (length `100 << lm` each; band `i` occupies
///   `[M*eb(i), M*eb(i+1))`). On decode they are outputs; on encode
///   they carry the analyzed shapes (mutated in place by the
///   resynthesis).
/// * `shape_bits` — the per-band 1/8-bit shape allocation
///   (`pulses[]` from
///   [`crate::alloc_exact::compute_allocation_exact`]).
/// * `tf_res` — per-band §4.3.4.5 Hadamard adjustment (the Tables
///   60–63 lookup of the decoded `tf_change` bits).
/// * `total_bits` — `frame_bytes * 64 - anti_collapse_rsv` in 1/8
///   bits; `balance` — the allocation walk's leftover.
/// * `band_amps` — encoder-side per-channel band amplitudes for the
///   intensity collapse (decode passes `None`).
/// * `resynth` — force resynthesis on the encode side (the decode
///   side always resynthesizes).
///
/// Returns the per-band collapse masks (the §4.3.5 input).
#[allow(clippy::too_many_arguments)]
pub fn quant_all_bands(
    mut io: QuantIo<'_, '_>,
    start: usize,
    end: usize,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    shape_bits: &[i32; NUM_BANDS],
    short_blocks: bool,
    spread: Spread,
    dual_stereo: bool,
    intensity: i32,
    tf_res: &[i32; NUM_BANDS],
    total_bits: i32,
    mut balance: i32,
    lm: u32,
    coded_bands: usize,
    seed: &mut u32,
    band_amps: Option<&BandAmps>,
    resynth: bool,
) -> Result<BandWalkResult, Error> {
    if start >= end || end > NUM_BANDS || lm > 3 {
        return Err(Error::InvalidParameter);
    }
    let channels = if y.is_some() { 2 } else { 1 };
    let m = 1usize << lm;
    let eb = |i: usize| EBAND_EDGES_5MS[i] as usize;
    if x.len() < m * eb(end) || y.as_deref().is_some_and(|yy| yy.len() < m * eb(end)) {
        return Err(Error::InvalidParameter);
    }
    let b_frame = if short_blocks { m } else { 1 };
    let norm_len = m * eb(NUM_BANDS);
    let mut norm = vec![0f32; channels * norm_len];
    let mut collapse_masks = vec![0u8; channels * NUM_BANDS];
    let mut dual_stereo = dual_stereo;

    let mut lowband_offset = 0usize;
    let mut update_lowband = true;

    let ctx = FrameCtx {
        spread,
        intensity,
        band_amps,
        resynth: resynth || !io.is_encode(),
    };

    for i in start..end {
        let tell = io.tell_frac() as i32;
        let band_lo = m * eb(i);
        let band_hi = m * eb(i + 1);
        let n = band_hi - band_lo;

        // Compute how many bits to allocate to this band.
        if i != start {
            balance -= tell;
        }
        let remaining = total_bits - tell - 1;
        let mut remaining_bits = remaining;
        let b = if i < coded_bands {
            let curr_balance = balance / (3.min(coded_bands - i) as i32);
            0.max(16383.min((remaining + 1).min(shape_bits[i] + curr_balance)))
        } else {
            0
        };

        if ctx.resynth
            && band_lo as i64 - n as i64 >= (m * eb(start)) as i64
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }

        let tf_change = tf_res[i];
        // Conservative estimate of the collapse masks of the bands
        // that supply the folding source.
        let mut x_cm: u32;
        let mut y_cm: u32;
        let mut effective_lowband: Option<usize> = None;
        if lowband_offset != 0 && (spread != Spread::Aggressive || b_frame > 1 || tf_change < 0) {
            // Never repeat spectral content within one band.
            let ell =
                ((m * eb(start)) as i64).max((m * eb(lowband_offset)) as i64 - n as i64) as usize;
            effective_lowband = Some(ell);
            let mut fold_start = lowband_offset;
            while m * eb(fold_start - 1) > ell {
                fold_start -= 1;
            }
            fold_start -= 1;
            let mut fold_end = lowband_offset - 1;
            while m * eb(fold_end + 1) < ell + n {
                fold_end += 1;
            }
            fold_end += 1;
            x_cm = 0;
            y_cm = 0;
            for fold_i in fold_start..fold_end {
                x_cm |= collapse_masks[fold_i * channels] as u32;
                y_cm |= collapse_masks[fold_i * channels + channels - 1] as u32;
            }
        } else {
            // Fold via the LCG: all blocks are (almost surely)
            // non-zero.
            x_cm = (1u32 << b_frame) - 1;
            y_cm = x_cm;
        }

        if dual_stereo && i as i32 == intensity {
            // Switch off dual stereo to do intensity.
            dual_stereo = false;
            let (n0c, n1c) = norm.split_at_mut(norm_len);
            for j in m * eb(start)..band_lo {
                n0c[j] = 0.5 * (n0c[j] + n1c[j]);
            }
        }

        // Owned folding-source copies (per channel).
        let lb_copy = |norm: &[f32], ch: usize| -> Option<Vec<f32>> {
            effective_lowband.map(|off| norm[ch * norm_len + off..ch * norm_len + off + n].to_vec())
        };

        if dual_stereo {
            let lbx = lb_copy(&norm, 0);
            let lby = lb_copy(&norm, 1);
            let mut out_x = vec![0f32; n];
            let mut out_y = vec![0f32; n];
            {
                let xs = &mut x[band_lo..band_hi];
                x_cm = quant_band(
                    &mut io,
                    &ctx,
                    i,
                    xs,
                    None,
                    b / 2,
                    b_frame,
                    tf_change,
                    lbx,
                    &mut remaining_bits,
                    lm as i32,
                    Some(&mut out_x),
                    0,
                    seed,
                    1.0,
                    x_cm,
                )?;
            }
            {
                let ys = &mut y.as_deref_mut().unwrap()[band_lo..band_hi];
                y_cm = quant_band(
                    &mut io,
                    &ctx,
                    i,
                    ys,
                    None,
                    b / 2,
                    b_frame,
                    tf_change,
                    lby,
                    &mut remaining_bits,
                    lm as i32,
                    Some(&mut out_y),
                    0,
                    seed,
                    1.0,
                    y_cm,
                )?;
            }
            norm[band_lo..band_lo + n].copy_from_slice(&out_x);
            norm[norm_len + band_lo..norm_len + band_lo + n].copy_from_slice(&out_y);
        } else {
            let lbx = lb_copy(&norm, 0);
            let mut out_x = vec![0f32; n];
            {
                let xs = &mut x[band_lo..band_hi];
                let cm = quant_band(
                    &mut io,
                    &ctx,
                    i,
                    xs,
                    y.as_deref_mut().map(|yy| &mut yy[band_lo..band_hi]),
                    b,
                    b_frame,
                    tf_change,
                    lbx,
                    &mut remaining_bits,
                    lm as i32,
                    Some(&mut out_x),
                    0,
                    seed,
                    1.0,
                    x_cm | y_cm,
                )?;
                x_cm = cm;
                y_cm = cm;
            }
            norm[band_lo..band_lo + n].copy_from_slice(&out_x);
        }
        collapse_masks[i * channels] = x_cm as u8;
        collapse_masks[i * channels + channels - 1] = y_cm as u8;
        balance += shape_bits[i] + tell;

        // Keep the folding position as long as there is 1 bit/sample.
        update_lowband = b > (n as i32) << BITRES;
    }

    Ok(BandWalkResult { collapse_masks })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The LCG and the bit-exact trigonometric helpers match their
    /// published integer contracts.
    #[test]
    fn integer_helpers() {
        assert_eq!(celt_lcg_rand(0), 1_013_904_223);
        assert_eq!(celt_lcg_rand(1_013_904_223), 1_196_435_762);
        // bitexact_cos endpoints: cos(0) = 32767 + 1; the split code
        // only evaluates interior angles (0 and 16384 take the
        // special-cased mid/side constants), so probe the interior.
        assert_eq!(bitexact_cos(0), 32768);
        assert!(bitexact_cos(16383) <= 32);
        let mid = bitexact_cos(8192) as f64 / 32768.0;
        assert!((mid - (core::f64::consts::PI / 4.0).cos()).abs() < 0.01);
        // isqrt32 is floor(sqrt(x)) over a sweep.
        for v in [0u32, 1, 2, 3, 4, 8, 15, 16, 17, 999, 1_000_000, u32::MAX] {
            let g = isqrt32(v);
            assert!(u64::from(g) * u64::from(g) <= u64::from(v));
            assert!((u64::from(g) + 1) * (u64::from(g) + 1) > u64::from(v));
        }
    }

    /// haar1 is orthonormal: applying it twice at the same geometry
    /// returns the input (it is its own inverse up to f32 rounding).
    #[test]
    fn haar1_involution() {
        let orig: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut x = orig.clone();
        haar1(&mut x, 16, 1);
        haar1(&mut x, 16, 1);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// interleave/deinterleave are exact inverses in both the
    /// Hadamard and the sequential orderings.
    #[test]
    fn hadamard_reorder_roundtrip() {
        for &(n0, stride, had) in &[
            (4usize, 4usize, true),
            (4, 4, false),
            (2, 8, true),
            (3, 2, false),
        ] {
            let orig: Vec<f32> = (0..n0 * stride).map(|i| i as f32).collect();
            let mut x = orig.clone();
            deinterleave_hadamard(&mut x, n0, stride, had);
            interleave_hadamard(&mut x, n0, stride, had);
            assert_eq!(x, orig, "n0={n0} stride={stride} had={had}");
        }
    }

    /// exp_rotation preserves the L2 norm (orthonormal) and inverts
    /// itself with the opposite direction.
    #[test]
    fn exp_rotation_inverse() {
        let orig: Vec<f32> = (0..24).map(|i| ((i * 7 + 3) as f32 * 0.11).cos()).collect();
        for spread in [Spread::Light, Spread::Normal, Spread::Aggressive] {
            let mut x = orig.clone();
            exp_rotation(&mut x, 1, 1, 4, spread);
            let norm_after: f32 = x.iter().map(|v| v * v).sum();
            let norm_orig: f32 = orig.iter().map(|v| v * v).sum();
            assert!((norm_after - norm_orig).abs() < 1e-4);
            exp_rotation(&mut x, -1, 1, 4, spread);
            for (a, b) in x.iter().zip(orig.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    /// compute_qn spot values: qn is even (or 1) and capped at 256.
    #[test]
    fn compute_qn_contract() {
        for n in [2i32, 4, 8, 16] {
            for b in [0i32, 32, 64, 128, 500, 3000] {
                for offset in [-16i32, 0, 16] {
                    let qn = compute_qn(n, b, offset, 100, false);
                    assert!(qn == 1 || qn % 2 == 0);
                    assert!(qn <= 256);
                }
            }
        }
    }

    /// Encode the full §4.3.3 walk + §4.3.4 band loop, then decode the
    /// produced bytes: the decoded allocation and the decoded spectra
    /// must match the encoder's resynthesis bit-for-bit (both sides
    /// run the identical arithmetic, RFC 6716 §4.3.3 lines 6113–6118).
    #[test]
    fn band_walk_encode_decode_lockstep() {
        use crate::alloc_exact::{compute_allocation_exact, AllocIo};
        use crate::band_cap::compute_band_caps;
        use crate::band_minimums::BAND_BINS_LM;
        use crate::coarse_energy::NUM_BANDS;
        use crate::tf_change::tf_adjustment;

        for &(channels, lm, frame_bytes, transient) in &[
            (1usize, 0u32, 40usize, false),
            (1, 2, 80, false),
            (1, 3, 160, false),
            (1, 2, 60, true),
            (2, 1, 96, false),
            (2, 3, 200, true),
        ] {
            let m = 1usize << lm;
            let n_coded = m * 100;
            let stereo = channels == 2;
            // Deterministic pseudo-random input spectra.
            let mut rng = 0x1234_5678u32;
            let mut gen = |len: usize| -> Vec<f32> {
                (0..len)
                    .map(|_| {
                        rng = celt_lcg_rand(rng);
                        ((rng as i32) >> 16) as f32 / 32768.0
                    })
                    .collect()
            };
            let mut x_enc = gen(n_coded);
            let mut y_enc_v = gen(n_coded);

            let bins: Vec<u32> = BAND_BINS_LM[lm as usize].to_vec();
            let mut caps16 = vec![0i16; NUM_BANDS];
            assert!(compute_band_caps(
                lm,
                stereo,
                channels as u32,
                &bins,
                &mut caps16
            ));
            let mut caps = [0i32; NUM_BANDS];
            for (c, &v) in caps.iter_mut().zip(caps16.iter()) {
                *c = v as i32;
            }
            let offsets = [0i32; NUM_BANDS];
            let mut tf_res = [0i32; NUM_BANDS];
            for (i, t) in tf_res.iter_mut().enumerate() {
                *t = tf_adjustment(transient, 0, lm as u8, i % 5 == 3) as i32;
            }
            let band_amps: BandAmps = [[1.0; NUM_BANDS]; 2];

            // ---- encode ----
            let mut enc = RangeEncoder::new();
            let total = (frame_bytes as i32) * 64 - enc.tell_frac() as i32 - 1;
            let alloc_enc = compute_allocation_exact(
                0,
                NUM_BANDS,
                &offsets,
                &caps,
                5,
                total,
                channels as i32,
                lm,
                AllocIo::Encode {
                    enc: &mut enc,
                    intensity: 21,
                    dual_stereo: false,
                    prev_coded_bands: 0,
                },
            )
            .expect("encode allocation");
            let mut seed_enc = 42u32;
            let walk_enc = quant_all_bands(
                QuantIo::Encode(&mut enc),
                0,
                NUM_BANDS,
                &mut x_enc,
                stereo.then_some(&mut y_enc_v[..]),
                &alloc_enc.shape_bits,
                transient,
                Spread::Normal,
                alloc_enc.dual_stereo,
                alloc_enc.intensity,
                &tf_res,
                (frame_bytes as i32) * 64,
                alloc_enc.balance,
                lm,
                alloc_enc.coded_bands,
                &mut seed_enc,
                Some(&band_amps),
                true,
            )
            .expect("encode walk");
            let bytes = enc.finish_to_size(frame_bytes).expect("finish");

            // ---- decode ----
            let mut dec = RangeDecoder::new(&bytes);
            let total_dec = (frame_bytes as i32) * 64 - dec.tell_frac() as i32 - 1;
            assert_eq!(total, total_dec);
            let alloc_dec = compute_allocation_exact(
                0,
                NUM_BANDS,
                &offsets,
                &caps,
                5,
                total_dec,
                channels as i32,
                lm,
                AllocIo::Decode(&mut dec),
            )
            .expect("decode allocation");
            assert_eq!(alloc_enc, alloc_dec, "C={channels} LM={lm}");

            let mut x_dec = vec![0f32; n_coded];
            let mut y_dec_v = vec![0f32; n_coded];
            let mut seed_dec = 42u32;
            let walk_dec = quant_all_bands(
                QuantIo::Decode(&mut dec),
                0,
                NUM_BANDS,
                &mut x_dec,
                stereo.then_some(&mut y_dec_v[..]),
                &alloc_dec.shape_bits,
                transient,
                Spread::Normal,
                alloc_dec.dual_stereo,
                alloc_dec.intensity,
                &tf_res,
                (frame_bytes as i32) * 64,
                alloc_dec.balance,
                lm,
                alloc_dec.coded_bands,
                &mut seed_dec,
                None,
                false,
            )
            .expect("decode walk");

            assert_eq!(
                walk_enc.collapse_masks, walk_dec.collapse_masks,
                "collapse masks C={channels} LM={lm} tr={transient}"
            );
            assert_eq!(seed_enc, seed_dec, "LCG lockstep");
            for (j, (a, b)) in x_enc.iter().zip(x_dec.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-6 * a.abs().max(1.0),
                    "ch0 bin {j}: enc {a} dec {b} (C={channels} LM={lm} tr={transient})"
                );
            }
            if stereo {
                for (j, (a, b)) in y_enc_v.iter().zip(y_dec_v.iter()).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-6 * a.abs().max(1.0),
                        "ch1 bin {j}: enc {a} dec {b} (C={channels} LM={lm} tr={transient})"
                    );
                }
            }
        }
    }
}
