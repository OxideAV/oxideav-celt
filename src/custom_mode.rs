//! Custom-mode construction (RFC 6716 §4.3 / Appendix A `opus_custom`):
//! the complete CELT mode geometry derived from an **arbitrary sample
//! rate and frame size** — band layout, allocation-table
//! interpolation, `logN`, window, pre-emphasis coefficients, pulse
//! cache, and per-band caps.
//!
//! ## What a "mode" is
//!
//! Every table the §4.3 frame walk consumes is a pure function of the
//! `(sample rate, frame size)` pair: the band edges follow the Bark
//! critical-band scale at the mode's spectral resolution, the static
//! allocation matrix is interpolated from the 48 kHz table over the
//! band center frequencies, the pulse-cost cache is rebuilt from the
//! §4.3.4.2 codebook sizes for the mode's actual band widths, and the
//! window/MDCT sizing follows the derived short-block size. The
//! standard 48 kHz configuration is nothing special: feeding
//! `(48000, 960)` through this construction reproduces the staged
//! 48 kHz tables bit-exactly (pinned by tests against
//! [`crate::band_layout::EBAND_EDGES_5MS`],
//! [`crate::static_alloc::STATIC_ALLOC`],
//! [`crate::alloc_exact::LOG_N400`],
//! [`crate::pulse_cache::CACHE_INDEX50`] /
//! [`crate::pulse_cache::CACHE_BITS50`], and
//! [`crate::band_cap::CACHE_CAPS50`]).
//!
//! ## Provenance
//!
//! Transcribed from the **normative RFC 6716 Appendix A reference
//! listing** (`modes.c` mode construction + `eband5ms` /
//! `band_allocation` / `bark_freq` data, `rate.c`
//! `compute_pulse_cache` / `fits_in32`, `cwrs.c` `log2_frac` /
//! `get_required_bits`), extracted from the staged
//! `docs/audio/opus/rfc6716-opus.txt` per §A.1 and SHA-1-verified
//! against the §A.1-printed value
//! (`86a927223e73d2476646a1b933fcd3fffb6ecc8c`); float-build
//! semantics. The §4.3.4.2 codebook sizes come from this crate's own
//! `V(N, K)` recursion ([`crate::pvq::v_count`]).

use crate::alloc_exact::{
    get_pulses, BITRES, FINE_OFFSET, MAX_FINE_BITS, QTHETA_OFFSET, QTHETA_OFFSET_TWOPHASE,
};
use crate::mdct::celt_window_f32;
use crate::pvq::v_count;
use crate::Error;

/// Upper bound on the band count any legal custom mode can produce
/// (the Appendix A `eMeans` / prob-model tables are sized for it; an
/// exhaustive scan of the legal `(rate, frame)` space measures 23).
pub const MAX_BANDS: usize = 25;

/// The 25 Bark critical-band edges in Hz (Appendix A `modes.c`
/// `bark_freq`).
const BARK_FREQ: [i32; 26] = [
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
    3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000,
];

/// The 2.5 ms band edges every 400·frame-rate mode shares (Appendix A
/// `modes.c` `eband5ms`; identical to
/// [`crate::band_layout::EBAND_EDGES_5MS`]).
const EBAND_5MS: [i16; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// Quality rows of the allocation table (Appendix A `modes.c`
/// `BITALLOC_SIZE`; RFC 6716 Table 57).
pub const NB_ALLOC_VECTORS: usize = 11;

/// The 48 kHz per-critical-band allocation matrix in 1/32 bit/sample
/// (Appendix A `modes.c` `band_allocation`, row-major
/// `[quality][band]`; RFC 6716 Table 57 carries the same data).
const BAND_ALLOCATION: [[u8; 21]; NB_ALLOC_VECTORS] = [
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        90, 80, 75, 69, 63, 56, 49, 40, 34, 29, 20, 18, 10, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    [
        110, 100, 90, 84, 78, 71, 65, 58, 51, 45, 39, 32, 26, 20, 12, 0, 0, 0, 0, 0, 0,
    ],
    [
        118, 110, 103, 93, 86, 80, 75, 70, 65, 59, 53, 47, 40, 31, 23, 15, 4, 0, 0, 0, 0,
    ],
    [
        126, 119, 112, 104, 95, 89, 83, 78, 72, 66, 60, 54, 47, 39, 32, 25, 17, 12, 1, 0, 0,
    ],
    [
        134, 127, 120, 114, 103, 97, 91, 85, 78, 72, 66, 60, 54, 47, 41, 35, 29, 23, 16, 10, 1,
    ],
    [
        144, 137, 130, 124, 113, 107, 101, 95, 88, 82, 76, 70, 64, 57, 51, 45, 39, 33, 26, 15, 1,
    ],
    [
        152, 145, 138, 132, 123, 117, 111, 105, 98, 92, 86, 80, 74, 67, 61, 55, 49, 43, 36, 20, 1,
    ],
    [
        162, 155, 148, 142, 133, 127, 121, 115, 108, 102, 96, 90, 84, 77, 71, 65, 59, 53, 46, 30, 1,
    ],
    [
        172, 165, 158, 152, 143, 137, 131, 125, 118, 112, 106, 100, 94, 87, 81, 75, 69, 63, 56, 45,
        20,
    ],
    [
        200, 200, 200, 200, 200, 200, 200, 200, 198, 193, 188, 183, 178, 173, 168, 163, 158, 153,
        148, 129, 104,
    ],
];

/// Maximum pseudo-pulse index a cache run covers (Appendix A `rate.h`
/// `MAX_PSEUDO`).
const MAX_PSEUDO: i32 = 40;

/// `fits_in32` (Appendix A `rate.c`): whether `V(n, k)` fits an
/// unsigned 32-bit integer.
fn fits_in32(n: i32, k: i32) -> bool {
    const MAX_N: [i32; 15] = [
        32767, 32767, 32767, 1476, 283, 109, 60, 40, 29, 24, 20, 18, 16, 14, 13,
    ];
    const MAX_K: [i32; 15] = [
        32767, 32767, 32767, 32767, 1172, 238, 95, 53, 36, 27, 22, 18, 16, 15, 13,
    ];
    if n >= 14 {
        if k >= 14 {
            false
        } else {
            n <= MAX_N[k as usize]
        }
    } else {
        k <= MAX_K[n as usize]
    }
}

/// `log2_frac` (Appendix A `cwrs.c`): a conservatively-large binary
/// logarithm with `frac` fractional bits.
pub fn log2_frac(val: u32, frac: u32) -> i32 {
    let mut val = val;
    let l = 32 - val.leading_zeros() as i32; // EC_ILOG
    if val & (val - 1) != 0 {
        // Round the mantissa up into 16 bits (bias-free even at the
        // top of the range).
        if l > 16 {
            let sh = (l - 16) as u32;
            val = (val >> sh) + (((val & ((1 << sh) - 1)) + (1 << sh) - 1) >> sh);
        } else {
            val <<= (16 - l) as u32;
        }
        let mut l = (l - 1) << frac;
        let mut fr = frac as i32;
        // One iteration is always needed: the rounding above can bump
        // the integer part.
        loop {
            let b = (val >> 16) as i32;
            l += b << fr;
            val = (val + b as u32) >> b as u32;
            val = (val * val + 0x7FFF) >> 15;
            fr -= 1;
            if fr < 0 {
                break;
            }
        }
        // A mantissa away from exactly 0x8000 rounds the remainder up.
        l + i32::from(val > 0x8000)
    } else {
        (l - 1) << frac
    }
}

/// `compute_ebands` (Appendix A `modes.c`): the Bark-scaled band-edge
/// layout at spectral resolution `res` Hz/bin, in short-MDCT bins.
fn compute_ebands(fs: i32, frame_size: i32, res: i32) -> Vec<i16> {
    // All modes with 2.5 ms short blocks share the eband5ms layout.
    if fs == 400 * frame_size {
        return EBAND_5MS.to_vec();
    }
    // Number of critical bands the sampling rate supports.
    let mut n_bark = 1usize;
    while n_bark < BARK_FREQ.len() - 1 {
        if BARK_FREQ[n_bark + 1] * 2 >= fs {
            break;
        }
        n_bark += 1;
    }
    // Where the linear spacing ends (spacing >= the resolution).
    let mut lin = 0usize;
    while lin < n_bark {
        if BARK_FREQ[lin + 1] - BARK_FREQ[lin] >= res {
            break;
        }
        lin += 1;
    }
    let low = ((BARK_FREQ[lin] + res / 2) / res) as usize;
    let high = n_bark - lin;
    let mut nb = low + high;
    let mut e = vec![0i32; nb + 2];
    let mut offset = 0i32;
    // Linear spacing at the resolution.
    for (i, v) in e.iter_mut().enumerate().take(low) {
        *v = i as i32;
    }
    if low > 0 {
        offset = e[low - 1] * res - BARK_FREQ[lin - 1];
    }
    // Spacing follows the critical bands.
    for i in 0..high {
        let target = BARK_FREQ[lin + i];
        // Round to an even value.
        e[i + low] = (target + offset / 2 + res) / (2 * res) * 2;
        offset = e[i + low] * res - target;
    }
    // Enforce the minimum spacing at the boundary.
    for (i, v) in e.iter_mut().enumerate().take(nb) {
        if *v < i as i32 {
            *v = i as i32;
        }
    }
    // Round to an even value; clamp to the frame.
    e[nb] = (BARK_FREQ[n_bark] + res) / (2 * res) * 2;
    if e[nb] > frame_size {
        e[nb] = frame_size;
    }
    for i in 1..nb.saturating_sub(1) {
        if e[i + 1] - e[i] < e[i] - e[i - 1] {
            e[i] -= (2 * e[i] - e[i - 1] - e[i + 1]) / 2;
        }
    }
    // Remove any empty bands.
    let mut j = 0usize;
    for i in 0..nb {
        if e[i + 1] > e[j] {
            j += 1;
            e[j] = e[i + 1];
        }
    }
    nb = j;
    e.truncate(nb + 1);
    e.iter().map(|&v| v as i16).collect()
}

/// A CELT operating mode: every geometry-derived table the §4.3 frame
/// walk consumes, for one `(sample rate, frame size)` pair.
///
/// [`CeltCustomMode::new`] runs the full Appendix A construction;
/// [`CeltCustomMode::standard`] is the 48 kHz / 960-sample mode every
/// RFC 6716 Opus stream uses (the same construction — its output is
/// pinned bit-exact against the staged 48 kHz tables).
#[derive(Debug, Clone)]
pub struct CeltCustomMode {
    /// Sample rate in Hz (8000..=96000).
    pub fs: u32,
    /// Largest frame-size shift the mode supports (frames are
    /// `short_mdct_size << lm` samples for `lm ∈ 0..=max_lm`).
    pub max_lm: u32,
    /// Short-block MDCT size in samples.
    pub short_mdct_size: usize,
    /// Window overlap in samples (`(short_mdct_size >> 2) << 2`).
    pub overlap: usize,
    /// Number of energy bands.
    pub nb_ebands: usize,
    /// One-past-last band the frame walk codes (`<= nb_ebands`; below
    /// it only when the shared 2.5 ms layout overshoots a small
    /// short-MDCT size).
    pub eff_ebands: usize,
    /// Band edges in short-MDCT bins (`nb_ebands + 1` entries).
    pub e_bands: Vec<i16>,
    /// Interpolated allocation matrix, row-major
    /// `[quality][band]` over [`NB_ALLOC_VECTORS`] quality rows.
    pub alloc_vectors: Vec<u8>,
    /// Per-band `log2(width)` in 1/8-bit units.
    pub log_n: Vec<i16>,
    /// The rising window half `W(0..overlap)`.
    pub window: Vec<f32>,
    /// Pre/de-emphasis coefficients (float-build values; index 0/1 are
    /// the filter taps, 2/3 the input/output scale pair).
    pub preemph: [f32; 4],
    /// Pulse-cache run offsets, row-major `[lm + 1][band]` over
    /// `max_lm + 2` rows (`-1` marks a never-dereferenced sentinel).
    pub cache_index: Vec<i16>,
    /// Concatenated pulse-cost runs (`run[0]` = max pseudo-pulse
    /// index, `run[k]` = cost of `k` pseudo-pulses minus one).
    pub cache_bits: Vec<u8>,
    /// Per-band allocation caps, row-major `[lm][stereo][band]`.
    pub cache_caps: Vec<u8>,
}

impl CeltCustomMode {
    /// Build the mode for `fs` Hz and `frame_size` samples per frame.
    ///
    /// The legality envelope is the Appendix A one: `fs` in
    /// 8000..=96000, `frame_size` even in 40..=1024, frames of at
    /// least 1 ms, short blocks of at most 3.3 ms, and the band-growth
    /// invariants the construction asserts (every band no wider than
    /// the last, no band more than twice its predecessor). Violations
    /// return [`Error::InvalidParameter`].
    pub fn new(fs: u32, frame_size: usize) -> Result<Self, Error> {
        let fs_i = fs as i32;
        let frame = frame_size as i32;
        if !(8000..=96000).contains(&fs_i) {
            return Err(Error::InvalidParameter);
        }
        if !(40..=1024).contains(&frame) || frame % 2 != 0 {
            return Err(Error::InvalidParameter);
        }
        // Frames of less than 1 ms are not supported.
        if frame * 1000 < fs_i {
            return Err(Error::InvalidParameter);
        }

        let max_lm: u32 = if frame * 75 >= fs_i && frame % 16 == 0 {
            3
        } else if frame * 150 >= fs_i && frame % 8 == 0 {
            2
        } else if frame * 300 >= fs_i && frame % 4 == 0 {
            1
        } else {
            0
        };
        // Short blocks longer than 3.3 ms are not supported.
        if (frame >> max_lm) * 300 > fs_i {
            return Err(Error::InvalidParameter);
        }

        // Pre/de-emphasis approximates the 48 kHz A(z) = 1 - 0.85/z
        // at the mode rate (float-build coefficient values).
        let preemph: [f32; 4] = if fs_i < 12000 {
            [0.350_006_1, -0.179_992_68, 0.271_996_8, 3.676_513_7]
        } else if fs_i < 24000 {
            [0.600_006_1, -0.179_992_68, 0.442_499_87, 2.259_887_7]
        } else if fs_i < 40000 {
            [0.779_998_8, -0.100_006_1, 0.749_977_1, 1.333_374]
        } else {
            [0.850_006_1, 0.0, 1.0, 1.0]
        };

        let nb_short_mdcts = 1usize << max_lm;
        let short_mdct_size = frame_size / nb_short_mdcts;
        let short = short_mdct_size as i32;
        let res = (fs_i + short) / (2 * short);

        let e_bands = compute_ebands(fs_i, short, res);
        let nb_ebands = e_bands.len() - 1;
        if nb_ebands == 0 || nb_ebands > MAX_BANDS {
            return Err(Error::InvalidParameter);
        }
        // The construction's own invariants (asserted by the listing):
        // every band no wider than the last, and no band more than
        // twice as wide as its predecessor.
        for i in 1..nb_ebands {
            if e_bands[i] - e_bands[i - 1] > e_bands[nb_ebands] - e_bands[nb_ebands - 1] {
                return Err(Error::InvalidParameter);
            }
            if e_bands[i + 1] - e_bands[i] > 2 * (e_bands[i] - e_bands[i - 1]) {
                return Err(Error::InvalidParameter);
            }
        }

        let mut eff_ebands = nb_ebands;
        while e_bands[eff_ebands] as usize > short_mdct_size {
            eff_ebands -= 1;
        }

        // Overlap must be divisible by 4.
        let overlap = (short_mdct_size >> 2) << 2;
        if overlap == 0 {
            return Err(Error::InvalidParameter);
        }

        let alloc_vectors = compute_allocation_table(fs_i, short, &e_bands, nb_ebands);
        let window: Vec<f32> = (0..overlap).map(|i| celt_window_f32(i, overlap)).collect();
        let log_n: Vec<i16> = (0..nb_ebands)
            .map(|i| log2_frac((e_bands[i + 1] - e_bands[i]) as u32, BITRES) as i16)
            .collect();

        let mut mode = Self {
            fs,
            max_lm,
            short_mdct_size,
            overlap,
            nb_ebands,
            eff_ebands,
            e_bands,
            alloc_vectors,
            log_n,
            window,
            preemph,
            cache_index: Vec::new(),
            cache_bits: Vec::new(),
            cache_caps: Vec::new(),
        };
        mode.compute_pulse_cache()?;
        Ok(mode)
    }

    /// The standard 48 kHz / 960-sample mode (the one every RFC 6716
    /// Opus stream uses), built by the same construction.
    pub fn standard() -> &'static CeltCustomMode {
        use std::sync::OnceLock;
        static STANDARD: OnceLock<CeltCustomMode> = OnceLock::new();
        STANDARD.get_or_init(|| {
            CeltCustomMode::new(48_000, 960).expect("the standard mode always constructs")
        })
    }

    /// The frame size in samples at shift `lm` (`None` above
    /// [`Self::max_lm`]).
    pub fn frame_size(&self, lm: u32) -> Option<usize> {
        (lm <= self.max_lm).then_some(self.short_mdct_size << lm)
    }

    /// The width of `band` in per-channel MDCT bins at shift `lm`.
    pub(crate) fn band_bins(&self, band: usize, lm: u32) -> usize {
        ((self.e_bands[band + 1] - self.e_bands[band]) as usize) << lm
    }

    /// The pulse-cost cache run for `(band, lm)` where `lm` is the
    /// split-adjusted shift in `-1..=max_lm` (callers index row
    /// `lm + 1`). `None` on a sentinel run or out-of-range arguments.
    pub(crate) fn cache_row(&self, band: usize, lm: i32) -> Option<&[u8]> {
        if band >= self.nb_ebands || lm < -1 || lm > self.max_lm as i32 {
            return None;
        }
        let idx = self.cache_index[((lm + 1) as usize) * self.nb_ebands + band];
        if idx < 0 {
            return None;
        }
        let off = idx as usize;
        let max_pseudo = self.cache_bits[off] as usize;
        Some(&self.cache_bits[off..=off + max_pseudo])
    }

    /// The §4.3.4.1 bits → pseudo-pulses inversion over this mode's
    /// cache (Appendix A `rate.h` `bits2pulses`).
    pub(crate) fn bits2pulses(&self, band: usize, lm: i32, bits: i32) -> Option<i32> {
        let cache = self.cache_row(band, lm)?;
        let mut lo: i32 = 0;
        let mut hi: i32 = cache[0] as i32;
        let bits = bits - 1;
        for _ in 0..crate::alloc_exact::LOG_MAX_PSEUDO {
            let mid = (lo + hi + 1) >> 1;
            if cache[mid as usize] as i32 >= bits {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let lo_cost = if lo == 0 {
            -1
        } else {
            cache[lo as usize] as i32
        };
        if bits - lo_cost <= cache[hi as usize] as i32 - bits {
            Some(lo)
        } else {
            Some(hi)
        }
    }

    /// The §4.3.4.1 pseudo-pulses → bits cost over this mode's cache
    /// (Appendix A `rate.h` `pulses2bits`).
    pub(crate) fn pulses2bits(&self, band: usize, lm: i32, pulses: i32) -> Option<i32> {
        if pulses == 0 {
            return Some(0);
        }
        let cache = self.cache_row(band, lm)?;
        if pulses < 0 || pulses as usize >= cache.len() {
            return None;
        }
        Some(cache[pulses as usize] as i32 + 1)
    }

    /// The per-band allocation caps in 1/8 bits (Appendix A `celt.c`
    /// `init_caps`): `cap[j] = (caps_row[j] + 64) * C * N / 4`.
    pub(crate) fn init_caps(&self, lm: u32, channels: usize, cap: &mut [i32]) {
        let row = (2 * lm as usize + (channels - 1)) * self.nb_ebands;
        for (j, c) in cap.iter_mut().enumerate().take(self.nb_ebands) {
            let n = self.band_bins(j, lm) as i32;
            *c = (self.cache_caps[row + j] as i32 + 64) * channels as i32 * n / 4;
        }
    }

    /// `compute_pulse_cache` (Appendix A `rate.c`): scan for unique
    /// band sizes across the split ladder, price each unique size from
    /// the §4.3.4.2 codebook sizes, and derive the per-band caps.
    fn compute_pulse_cache(&mut self) -> Result<(), Error> {
        let lm = self.max_lm as i32;
        let nb = self.nb_ebands;
        let e = &self.e_bands;
        let mut cindex = vec![0i16; nb * (lm as usize + 2)];
        let mut entry_n = Vec::new();
        let mut entry_k = Vec::new();
        let mut entry_i = Vec::new();
        let mut curr: i32 = 0;

        // Scan for all unique band sizes (rows are lm = -1..=max_lm).
        for i in 0..=(lm + 1) as usize {
            for j in 0..nb {
                let n = ((e[j + 1] - e[j]) as i32) << i >> 1;
                cindex[i * nb + j] = -1;
                // Find an earlier band with the same size.
                'outer: for k in 0..=i {
                    let n_bound = if k == i { j } else { nb };
                    for m in 0..n_bound {
                        if n == ((e[m + 1] - e[m]) as i32) << k >> 1 {
                            cindex[i * nb + j] = cindex[k * nb + m];
                            break 'outer;
                        }
                    }
                }
                if cindex[i * nb + j] == -1 && n != 0 {
                    let mut k = 0i32;
                    while fits_in32(n, get_pulses(k + 1)) && k < MAX_PSEUDO {
                        k += 1;
                    }
                    entry_n.push(n);
                    entry_k.push(k);
                    entry_i.push(curr as usize);
                    cindex[i * nb + j] =
                        i16::try_from(curr).map_err(|_| Error::InvalidParameter)?;
                    curr += k + 1;
                }
            }
        }

        // Price each unique size: run[k] = ceil(8*log2 V(N, pulses(k))) - 1.
        let mut bits = vec![0u8; curr as usize];
        for (i, (&n, &k_max)) in entry_n.iter().zip(entry_k.iter()).enumerate() {
            let off = entry_i[i];
            bits[off] = u8::try_from(k_max).map_err(|_| Error::InvalidParameter)?;
            for j in 1..=k_max {
                let k = get_pulses(j);
                let v = v_count(n as u32, k as u32);
                if v == crate::pvq::V_COUNT_SATURATION {
                    return Err(Error::InvalidParameter);
                }
                let cost = log2_frac(v, BITRES) - 1;
                bits[off + j as usize] = u8::try_from(cost).map_err(|_| Error::InvalidParameter)?;
            }
        }

        // Per-band caps: the maximum rate at which the band reliably
        // uses as many bits as asked for.
        let mut caps = vec![0u8; (lm as usize + 1) * 2 * nb];
        let mut cap_at = 0usize;
        for i in 0..=lm {
            for c in 1..=2i32 {
                for j in 0..nb {
                    let mut n0 = (e[j + 1] - e[j]) as i32;
                    let max_bits;
                    if n0 << i == 1 {
                        // N=1 bands only have a sign bit and fine bits.
                        max_bits = (c * (1 + MAX_FINE_BITS)) << BITRES;
                    } else {
                        let mut lm0: i32 = 0;
                        // Even-sized bands bigger than N=2 can be
                        // split one more time; N0=1 bands can't be
                        // split below N=2.
                        if n0 > 2 {
                            n0 >>= 1;
                            lm0 -= 1;
                        } else if n0 <= 1 {
                            lm0 = i.min(1);
                            n0 <<= lm0;
                        }
                        // Cost of the lowest-level PVQ of a fully
                        // split band.
                        let idx = cindex[((lm0 + 1) as usize) * nb + j];
                        debug_assert!(idx >= 0);
                        let run_off = idx as usize;
                        let run_max = bits[run_off] as usize;
                        let mut mb = bits[run_off + run_max] as i32 + 1;
                        // Cost of coding the regular splits.
                        let mut n = n0;
                        for k in 0..(i - lm0) {
                            mb <<= 1;
                            // Offset qtheta bits by log2(N)/2 +
                            // QTHETA_OFFSET vs their fair share.
                            let offset = ((self.log_n[j] as i32 + ((lm0 + k) << BITRES)) >> 1)
                                - QTHETA_OFFSET;
                            // Average measured qtheta cost ~ 459/512.
                            let num = 459 * ((2 * n - 1) * offset + mb);
                            let den = ((2 * n - 1) << 9) - 459;
                            let qb = ((num + (den >> 1)) / den).min(57);
                            debug_assert!(qb >= 0);
                            mb += qb;
                            n <<= 1;
                        }
                        // Cost of a stereo split, if necessary.
                        if c == 2 {
                            mb <<= 1;
                            let offset = ((self.log_n[j] as i32 + (i << BITRES)) >> 1)
                                - if n == 2 {
                                    QTHETA_OFFSET_TWOPHASE
                                } else {
                                    QTHETA_OFFSET
                                };
                            let ndof = 2 * n - 1 - i32::from(n == 2);
                            // Step-PDF theta cost ~ 487/512.
                            let f = if n == 2 { 512 } else { 487 };
                            let num = f * (mb + ndof * offset);
                            let den = (ndof << 9) - f;
                            let qb = ((num + (den >> 1)) / den).min(if n == 2 { 64 } else { 61 });
                            debug_assert!(qb >= 0);
                            mb += qb;
                        }
                        // Add the fine bits (extra stereo DoF above
                        // N=2), offset by log2(N)/2 + FINE_OFFSET.
                        let ndof = c * n + i32::from(c == 2 && n > 2);
                        let mut offset =
                            ((self.log_n[j] as i32 + (i << BITRES)) >> 1) - FINE_OFFSET;
                        if n == 2 {
                            offset += (1 << BITRES) >> 2;
                        }
                        let num = mb + ndof * offset;
                        let den = (ndof - 1) << BITRES;
                        let qb = ((num + (den >> 1)) / den).min(MAX_FINE_BITS);
                        debug_assert!(qb >= 0);
                        mb += (c * qb) << BITRES;
                        max_bits = mb;
                    }
                    let width = c * (((e[j + 1] - e[j]) as i32) << i);
                    let cap = 4 * max_bits / width - 64;
                    if !(0..256).contains(&cap) {
                        return Err(Error::InvalidParameter);
                    }
                    caps[cap_at] = cap as u8;
                    cap_at += 1;
                }
            }
        }

        self.cache_index = cindex;
        self.cache_bits = bits;
        self.cache_caps = caps;
        Ok(())
    }
}

/// `compute_allocation_table` (Appendix A `modes.c`): the 48 kHz
/// per-critical-band matrix, interpolated onto this mode's band
/// centers over the 400·`eband5ms` Hz grid.
fn compute_allocation_table(fs: i32, short_mdct_size: i32, e_bands: &[i16], nb: usize) -> Vec<u8> {
    let max_bands = EBAND_5MS.len() - 1;
    let mut out = vec![0u8; NB_ALLOC_VECTORS * nb];
    // The standard 2.5 ms layout keeps the matrix verbatim.
    if fs == 400 * short_mdct_size {
        for (i, row) in BAND_ALLOCATION.iter().enumerate() {
            out[i * nb..(i + 1) * nb].copy_from_slice(&row[..nb]);
        }
        return out;
    }
    for i in 0..NB_ALLOC_VECTORS {
        for j in 0..nb {
            // This band's center on the 48 kHz table's Hz grid.
            let freq = e_bands[j] as i32 * fs / short_mdct_size;
            let mut k = 0usize;
            while k < max_bands {
                if 400 * EBAND_5MS[k] as i32 > freq {
                    break;
                }
                k += 1;
            }
            out[i * nb + j] = if k > max_bands - 1 {
                BAND_ALLOCATION[i][max_bands - 1]
            } else {
                let a1 = freq - 400 * EBAND_5MS[k - 1] as i32;
                let a0 = 400 * EBAND_5MS[k] as i32 - freq;
                ((a0 * BAND_ALLOCATION[i][k - 1] as i32 + a1 * BAND_ALLOCATION[i][k] as i32)
                    / (a0 + a1)) as u8
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc_exact::LOG_N400;
    use crate::band_cap::CACHE_CAPS50;
    use crate::band_layout::EBAND_EDGES_5MS;
    use crate::coarse_energy::NUM_BANDS;
    use crate::pulse_cache::{CACHE_BITS50, CACHE_INDEX50};
    use crate::static_alloc::{NUM_Q, STATIC_ALLOC};

    /// The construction, fed the standard 48 kHz configuration,
    /// reproduces every staged 48 kHz table bit-exactly.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn standard_mode_reproduces_staged_tables() {
        let m = CeltCustomMode::new(48_000, 960).expect("standard mode");
        assert_eq!(m.max_lm, 3);
        assert_eq!(m.short_mdct_size, 120);
        assert_eq!(m.overlap, 120);
        assert_eq!(m.nb_ebands, NUM_BANDS);
        assert_eq!(m.eff_ebands, NUM_BANDS);
        // Band edges == Table 55.
        for (i, &e) in m.e_bands.iter().enumerate() {
            assert_eq!(e as u32, EBAND_EDGES_5MS[i], "edge {i}");
        }
        // Allocation matrix == Table 57 (staged column-major).
        assert_eq!(NB_ALLOC_VECTORS, NUM_Q);
        for q in 0..NUM_Q {
            for j in 0..NUM_BANDS {
                assert_eq!(
                    m.alloc_vectors[q * NUM_BANDS + j],
                    STATIC_ALLOC[j][q],
                    "alloc[{q}][{j}]"
                );
            }
        }
        // logN == the staged log-n400 table.
        assert_eq!(&m.log_n[..], &LOG_N400[..]);
        // Pulse cache == the staged cache_index50 / cache_bits50.
        assert_eq!(&m.cache_index[..], &CACHE_INDEX50[..]);
        assert_eq!(&m.cache_bits[..], &CACHE_BITS50[..]);
        // Caps == the staged cache_caps50 (row-major [2*lm+stereo]).
        for r in 0..8 {
            for j in 0..NUM_BANDS {
                assert_eq!(
                    m.cache_caps[r * NUM_BANDS + j],
                    CACHE_CAPS50[r][j],
                    "caps[{r}][{j}]"
                );
            }
        }
        // Window == the §4.3.7 rising half at overlap 120.
        for (i, &w) in m.window.iter().enumerate() {
            assert_eq!(w, celt_window_f32(i, 120), "window[{i}]");
        }
        assert_eq!(m.preemph, [0.850_006_1, 0.0, 1.0, 1.0]);
    }

    /// Every frame size that reduces to the standard short size builds
    /// the identical geometry.
    #[test]
    fn standard_family_shares_geometry() {
        for frame in [120usize, 240, 480, 960] {
            let m = CeltCustomMode::new(48_000, frame).expect("mode");
            assert_eq!(m.short_mdct_size * (1 << m.max_lm), frame);
            assert_eq!(m.nb_ebands, NUM_BANDS);
            // NB: max_lm shrinks with the frame (the mode covers
            // frames up to the requested size).
            assert_eq!(m.frame_size(m.max_lm), Some(frame));
        }
    }

    /// The Appendix A legality envelope: rate, frame size, parity,
    /// the 1 ms frame floor and the 3.3 ms short-block ceiling.
    #[test]
    fn illegal_configurations_are_rejected() {
        assert!(CeltCustomMode::new(7999, 160).is_err());
        assert!(CeltCustomMode::new(96001, 960).is_err());
        assert!(CeltCustomMode::new(48_000, 38).is_err());
        assert!(CeltCustomMode::new(48_000, 1026).is_err());
        assert!(CeltCustomMode::new(48_000, 481).is_err());
        // Frame under 1 ms: 40 samples at 48 kHz is 0.83 ms.
        assert!(CeltCustomMode::new(48_000, 40).is_err());
        // Short block over 3.3 ms: 22050 Hz with a 220-sample frame
        // yields a 110-sample (5 ms) short block.
        assert!(CeltCustomMode::new(22_050, 220).is_err());
    }

    /// Structural invariants across a broad legal grid: edges start at
    /// zero, stay strictly increasing, end at or below the short-MDCT
    /// size (except the shared 2.5 ms layout), the band count stays
    /// within [`MAX_BANDS`], every cost run is monotone, and the caps
    /// row layout is complete.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn constructed_modes_hold_structural_invariants() {
        let configs: &[(u32, usize)] = &[
            (8_000, 160),
            (8_000, 64),
            (11_025, 44),
            (12_000, 240),
            (16_000, 320),
            (16_000, 40),
            (22_050, 88),
            (24_000, 480),
            (32_000, 640),
            (44_100, 440),
            (44_100, 880),
            (44_100, 1024),
            (48_000, 960),
            (48_000, 1024),
            (88_200, 272),
            (96_000, 960),
            (96_000, 320),
        ];
        for &(fs, frame) in configs {
            let m = CeltCustomMode::new(fs, frame)
                .unwrap_or_else(|_| panic!("mode ({fs}, {frame}) must construct"));
            assert!(m.nb_ebands <= MAX_BANDS, "({fs}, {frame}) band count");
            assert!(m.eff_ebands >= 1 && m.eff_ebands <= m.nb_ebands);
            assert_eq!(m.e_bands.len(), m.nb_ebands + 1);
            assert_eq!(m.e_bands[0], 0);
            for w in m.e_bands.windows(2) {
                assert!(w[1] > w[0], "({fs}, {frame}) edges increase");
            }
            assert!(m.e_bands[m.eff_ebands] as usize <= m.short_mdct_size);
            assert_eq!(m.overlap % 4, 0);
            assert!(m.overlap <= m.short_mdct_size && m.short_mdct_size - m.overlap <= 2);
            assert_eq!(m.log_n.len(), m.nb_ebands);
            assert_eq!(m.alloc_vectors.len(), NB_ALLOC_VECTORS * m.nb_ebands);
            assert_eq!(
                m.cache_caps.len(),
                (m.max_lm as usize + 1) * 2 * m.nb_ebands
            );
            assert_eq!(m.cache_index.len(), (m.max_lm as usize + 2) * m.nb_ebands);
            // Every non-sentinel cache run is monotone non-decreasing
            // and matches this crate's V(N, K) pricing.
            for lm in -1..=(m.max_lm as i32) {
                for j in 0..m.nb_ebands {
                    let n = ((m.e_bands[j + 1] - m.e_bands[j]) as i32) << (lm + 1) >> 1;
                    let Some(run) = m.cache_row(j, lm) else {
                        assert_eq!(n, 0, "({fs}, {frame}) sentinel only for empty size");
                        continue;
                    };
                    assert!(n > 0);
                    let max_pseudo = run[0] as usize;
                    assert_eq!(run.len(), max_pseudo + 1);
                    for k in 2..=max_pseudo {
                        assert!(run[k] >= run[k - 1], "({fs}, {frame}) run monotone");
                    }
                    for k in 1..=max_pseudo {
                        let v = v_count(n as u32, get_pulses(k as i32) as u32);
                        assert_eq!(
                            run[k] as i32,
                            log2_frac(v, BITRES) - 1,
                            "({fs}, {frame}) band {j} lm {lm} k {k}"
                        );
                    }
                }
            }
            // Rows quality 0 is all-zero; higher rows are monotone in
            // quality for every band (the interpolation preserves the
            // source matrix's column monotonicity).
            for j in 0..m.nb_ebands {
                assert_eq!(m.alloc_vectors[j], 0);
                for q in 1..NB_ALLOC_VECTORS {
                    assert!(
                        m.alloc_vectors[q * m.nb_ebands + j]
                            >= m.alloc_vectors[(q - 1) * m.nb_ebands + j],
                        "({fs}, {frame}) alloc monotone in quality"
                    );
                }
            }
        }
    }

    /// The exhaustive-scan band bound: sweep a coarse rate grid plus
    /// the known worst region and confirm the [`MAX_BANDS`] bound and
    /// the invariant rejections stay consistent.
    #[test]
    fn band_count_stays_bounded_over_rate_grid() {
        let mut max_nb = 0usize;
        for fs in (8_000..=96_000u32).step_by(501) {
            for frame in (40..=1024usize).step_by(2) {
                if let Ok(m) = CeltCustomMode::new(fs, frame) {
                    max_nb = max_nb.max(m.nb_ebands);
                }
            }
        }
        // The worst case an exhaustive scan finds is 23 bands (around
        // 40 kHz with ~163 Hz resolution).
        for fs in 39_990..=40_010u32 {
            if let Ok(m) = CeltCustomMode::new(fs, 976) {
                max_nb = max_nb.max(m.nb_ebands);
            }
        }
        assert!(max_nb <= MAX_BANDS, "measured {max_nb}");
        assert!(max_nb >= NUM_BANDS, "grid must reach rich layouts");
    }

    /// `log2_frac` against the whole-bit logarithm on exact powers of
    /// two and its conservative-overestimate contract elsewhere.
    #[test]
    fn log2_frac_contract() {
        for p in 0..31u32 {
            assert_eq!(log2_frac(1 << p, BITRES), (p as i32) << BITRES);
        }
        for v in 2..2000u32 {
            let exact = (v as f64).log2();
            let got = log2_frac(v, BITRES) as f64 / 8.0;
            assert!(got >= exact - 1e-9, "under-estimate at {v}");
            assert!(got <= exact + 0.13, "loose over-estimate at {v}");
        }
    }
}
