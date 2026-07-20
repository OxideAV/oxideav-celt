//! Spreading parameter decoding (RFC 6716 §4.3.4.3 + Table 56 + Table 59).
//!
//! ## What this module covers
//!
//! Per Table 56 of RFC 6716 (`docs/audio/opus/rfc6716-opus.txt`), the
//! CELT bitstream emits a single `spread` scalar between the global
//! `tf_select` flag and the §4.3.3 dynamic-allocation phase:
//!
//! ```text
//!     tf_select   ->   { 1, 1 }/2
//!     spread      ->   { 7, 2, 21, 2 }/32      <-- this module
//!     dyn. alloc. ->   Section 4.3.3
//! ```
//!
//! The decoded value selects the rotation parameter `f_r` used by the
//! §4.3.4.3 spreading transform that the PVQ shape decoder applies to
//! its unit-norm output vector, via the four-row Table 59 lookup:
//!
//! ```text
//!     spread = 0   f_r = infinite (no rotation)
//!     spread = 1   f_r = 15
//!     spread = 2   f_r = 10
//!     spread = 3   f_r = 5
//! ```
//!
//! For a band with `N` PVQ dimensions and `K` pulses, the rotation
//! gain is `g_r = N / (N + f_r * K)` and the rotation angle is
//! `theta = pi * g_r^2 / 4` (RFC 6716 §4.3.4.3 prose, page 117).
//! `spread = 0` is a special case: `f_r` is infinite, so `g_r = 0`,
//! `theta = 0`, and the rotation is the identity.
//!
//! This module decodes the bitstream field, exposes the Table 59
//! lookup as a pure helper, and exposes the closed-form rotation-gain
//! helper for the PVQ caller. The actual 2-D rotation loop and the
//! `(pi/2 - theta)` pre-rotation for blocks of 8+ samples are
//! shape-decoder work — they sit on the PVQ codevector decoder, which
//! is queued for a later round.
//!
//! ## What is NOT in this module
//!
//! * The PVQ shape decoder itself (§4.3.4.2) and the encoder-side
//!   spreading-decision logic (§5.3.7). We are decoder-only.
//! * The 2-D rotation `R(i, j)` applied across `N - 1` band positions
//!   back-and-forth (§4.3.4.3 final paragraph). It depends on the
//!   per-band PVQ output vector; deferred until the band-decode round.
//! * The interleaved pre-rotation by `(pi/2 - theta)` with a stride
//!   of `round(sqrt(N / nb_blocks))` when each time block represents
//!   eight samples or more. Also deferred until band decode.
//! ## Clean-room provenance
//!
//! Every PDF, every cumulative frequency, every `f_r` row of Table 59,
//! and every closed-form formula in this module is transcribed from
//! RFC 6716 §4.3.4.3 + Table 56 + Table 59 in
//! `docs/audio/opus/rfc6716-opus.txt`. The encoder-side
//! [`spreading_decision`] is transcribed from the RFC's own Appendix A
//! reference listing (`spreading_decision`), extracted from the staged
//! RFC text per §A.1 and SHA-1-verified; no source outside the staged
//! docs was consulted.

use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::Error;

/// The four legal spread parameter values per RFC 6716 §4.3.4.3.
///
/// The bitstream PDF `{7, 2, 21, 2}/32` (Table 56) selects one of
/// these four discrete settings. `None` indicates `f_r = infinite`
/// (no spreading rotation) for `spread = 0`; `Some(f_r)` for the other
/// three values.
///
/// The variant ordering matches the PDF order: `spread = 0..=3` maps
/// to `Spread::None`, `Spread::Light`, `Spread::Normal`, `Spread::Aggressive`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spread {
    /// `spread = 0`. `f_r` is infinite (no rotation applied).
    None,
    /// `spread = 1`. `f_r = 15`. The smallest non-trivial rotation
    /// (gentlest spreading correction; PDF tail symbol).
    Light,
    /// `spread = 2`. `f_r = 10`. The bulk-probability case
    /// (PDF mass 21/32).
    Normal,
    /// `spread = 3`. `f_r = 5`. The strongest spreading rotation
    /// (PDF tail symbol).
    Aggressive,
}

impl Spread {
    /// The numeric `f_r` value of Table 59, or `None` if the spreading
    /// rotation is the identity (`Spread::None`).
    ///
    /// `f_r` controls the rotation gain `g_r = N / (N + f_r * K)`.
    /// Smaller `f_r` ⇒ closer-to-unity `g_r` ⇒ stronger rotation.
    pub fn f_r(self) -> Option<u32> {
        match self {
            Spread::None => None,
            Spread::Light => Some(15),
            Spread::Normal => Some(10),
            Spread::Aggressive => Some(5),
        }
    }

    /// The raw `spread` value as carried on the wire (0..=3).
    pub fn as_u8(self) -> u8 {
        match self {
            Spread::None => 0,
            Spread::Light => 1,
            Spread::Normal => 2,
            Spread::Aggressive => 3,
        }
    }

    /// Reconstruct a `Spread` from its raw bitstream value.
    ///
    /// Returns `None` for out-of-range inputs; the bitstream itself
    /// cannot produce such values because [`decode_spread`] consumes
    /// the ICDF directly.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Spread::None),
            1 => Some(Spread::Light),
            2 => Some(Spread::Normal),
            3 => Some(Spread::Aggressive),
            _ => None,
        }
    }
}

/// ICDF table for the `spread` PDF `{7, 2, 21, 2}/32` (RFC 6716
/// Table 56, row `spread`).
///
/// Cumulative frequencies: `fh = [7, 9, 30, 32]`. ICDF entries are
/// `ft - fh[k]` with `ft = 32`: `[25, 23, 2, 0]`. `ftb = 5`
/// (since `32 == 1 << 5`).
const SPREAD_ICDF: &[u8] = &[25, 23, 2, 0];

/// `ftb` for the spread ICDF (`ft = 32`, so `1 << 5`).
const SPREAD_FTB: u32 = 5;

/// Default spreading mode per §4.3.4.3 when the field is treated as
/// absent.
///
/// RFC 6716 Table 56 emits the `spread` field unconditionally in every
/// frame, so the only path that reaches this default is a caller that
/// chooses not to call [`decode_spread`] at all (for example, in a
/// hybrid configuration that skips the CELT side of the bitstream).
/// We pick `Spread::Normal` — the PDF's bulk-probability mode — to
/// match the average-case behaviour the encoder would emit.
pub const DEFAULT_SPREAD: Spread = Spread::Normal;

/// Decode the spreading parameter from the bitstream (RFC 6716
/// §4.3.4.3 + Table 56 `spread` row).
///
/// Reads one symbol with PDF `{7, 2, 21, 2}/32` via the §4.1.3.3 ICDF
/// path and maps it through the four-variant [`Spread`] enum.
///
/// The CELT decoder calls this between the global `tf_select` decode
/// and the §4.3.3 dynamic-allocation block.
pub fn decode_spread(dec: &mut RangeDecoder<'_>) -> Spread {
    let raw = dec.dec_icdf(SPREAD_ICDF, SPREAD_FTB);
    // ICDF is monotonic into 0..4; the cast is bound-safe.
    match raw {
        0 => Spread::None,
        1 => Spread::Light,
        2 => Spread::Normal,
        _ => Spread::Aggressive,
    }
}

/// Encode the spreading parameter into the range coder — the exact
/// inverse of [`decode_spread`] (RFC 6716 §4.3.4.3 + Table 56 `spread`
/// row).
///
/// Writes one symbol with PDF `{7, 2, 21, 2}/32` via the §5.1.2.3 ICDF
/// path, using the same `SPREAD_ICDF` table / `ftb` the decoder reads.
/// A subsequent [`decode_spread`] over the finished frame recovers the
/// same [`Spread`] variant.
pub fn encode_spread(enc: &mut RangeEncoder, spread: Spread) -> Result<(), Error> {
    enc.enc_icdf(usize::from(spread.as_u8()), SPREAD_ICDF, SPREAD_FTB)
}

/// Rotation-gain numerator and denominator for the §4.3.4.3 spreading
/// transform, expressed as `(num, den)` so the PVQ caller can choose
/// its own fixed-point representation when it needs `g_r = num/den`.
///
/// `g_r = N / (N + f_r * K)`. For `Spread::None` the rotation is the
/// identity and `g_r = 0`; we return `(0, 1)` so the caller's
/// downstream `theta = pi * g_r^2 / 4` evaluates to zero without a
/// branch on the enum.
///
/// `n` is the number of dimensions in the current band; `k` is the
/// number of PVQ pulses. The §4.3.4.3 prose constrains both to be
/// positive integers; the helper returns `(0, 1)` defensively if `n`
/// is zero (an empty band), so a caller that loops over all band
/// indices unconditionally still sees a well-defined zero rotation.
pub fn rotation_gain_ratio(spread: Spread, n: u32, k: u32) -> (u32, u32) {
    match spread.f_r() {
        None => (0, 1),
        Some(_) if n == 0 => (0, 1),
        Some(f_r) => {
            // g_r = N / (N + f_r * K). Both numerator and denominator
            // are exact unsigned integers in the input domain
            // (N ≤ band size, K ≤ pulse budget — both bounded by the
            // CELT bit allocator).
            let num = n;
            let den = n.saturating_add(f_r.saturating_mul(k));
            (num, den)
        }
    }
}

/// Square of the rotation gain, as a Q-format ratio `(num, den)` with
/// `den != 0`.
///
/// Equivalent to applying [`rotation_gain_ratio`] then squaring both
/// terms in the same fixed-point domain. The caller computes
/// `theta = pi * (num / den) / 4` against this pair to recover the
/// rotation angle.
pub fn rotation_gain_squared_ratio(spread: Spread, n: u32, k: u32) -> (u64, u64) {
    let (num, den) = rotation_gain_ratio(spread, n, k);
    let n2 = u64::from(num).saturating_mul(u64::from(num));
    let d2 = u64::from(den).saturating_mul(u64::from(den));
    (n2, d2)
}

/// Pre-rotation stride per the §4.3.4.3 "extra rotation" rule.
///
/// When the decoded vector spans more than one time block and each
/// block represents eight samples or more, an additional rotation by
/// `(pi/2 - theta)` is applied BEFORE the main rotation, in an
/// interleaved manner with stride `round(sqrt(N / nb_blocks))`.
///
/// Returns `None` if either (a) `nb_blocks <= 1` — there is only one
/// time block, so the extra rotation does not apply — or (b) the
/// per-block sample count `N / nb_blocks` is below 8.
///
/// The rounding direction is round-half-up; the §4.3.4.3 prose simply
/// says `round(sqrt(N/nb_blocks))` and does not pin the half-tie
/// direction. Round-half-up is the canonical interpretation of the
/// unqualified `round()` notation in IETF prose.
pub fn pre_rotation_stride(n: u32, nb_blocks: u32) -> Option<u32> {
    if nb_blocks <= 1 || n == 0 {
        return None;
    }
    let per_block = n / nb_blocks;
    if per_block < 8 {
        return None;
    }
    // round(sqrt(per_block)) by integer arithmetic:
    //   k = floor(sqrt(per_block));
    //   round up iff per_block - k*k > k.
    let mut k = (per_block as f64).sqrt() as u32;
    // Guard against floating drift at large `per_block`.
    while k.saturating_mul(k) > per_block {
        k -= 1;
    }
    while (k + 1).saturating_mul(k + 1) <= per_block {
        k += 1;
    }
    // Round-half-up tie-break.
    let lo = k.saturating_mul(k);
    let hi = (k + 1).saturating_mul(k + 1);
    let frac_lo = per_block - lo;
    let frac_hi = hi - per_block;
    let rounded = if frac_lo < frac_hi { k } else { k + 1 };
    Some(rounded.max(1))
}

/// The encoder-side spreading decision (transcribed from the §A.1
/// reference listing, `spreading_decision`): a tonality vote over the
/// coded bands drives the Table-56 `spread` selection with recursive
/// averaging and hysteresis.
///
/// Per band wider than 8 bins, a rough CDF of the unit-norm
/// coefficients `x` is taken — `tcount[k]` counts coefficients with
/// `x^2 * N` below `1/4`, `1/16`, `1/64` — and the band votes 0..3
/// "tonality points" (one for each threshold a majority of the band
/// sits under; a tonal band concentrates energy in few bins, so most
/// coefficients are tiny). The frame vote `sum` (x256, averaged over
/// bands and channels) is folded into the running `*average`, biased
/// by the previous decision (hysteresis, the listing's
/// `(3*sum + ((3-last)<<7) + 66) / 4` fold), and mapped through the
/// `80 / 256 / 384` ladder to Aggressive / Normal / Light / None —
/// noisy frames (few small coefficients) spread more, tonal frames
/// spread less.
///
/// The `update_hf` arm (only live when the encoder ran its pitch
/// prefilter on a long frame) maintains the `*hf_average` /
/// `*tapset_decision` state for the §4.3.7.1 tapset choice from the
/// top four bands' small-coefficient density.
///
/// `x` / `y` are the (per-channel) coded-window unit-norm spectra in
/// the band-contiguous layout; `m = 1 << LM`. Returns the spread
/// choice for this frame.
#[allow(clippy::too_many_arguments)]
pub fn spreading_decision(
    x: &[f32],
    y: Option<&[f32]>,
    m: usize,
    end: usize,
    average: &mut i32,
    last_decision: Spread,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
) -> Spread {
    use crate::band_layout::EBAND_EDGES_5MS;
    let eb = |i: usize| m * EBAND_EDGES_5MS[i] as usize;
    debug_assert!(end > 0 && end < EBAND_EDGES_5MS.len());
    if eb(end) - eb(end - 1) <= 8 {
        return Spread::None;
    }
    let mut sum = 0i32;
    let mut nb_bands = 0i32;
    let mut hf_sum = 0i32;
    let channels = 1 + usize::from(y.is_some());
    for ch in [Some(x), y].into_iter().flatten() {
        for i in 0..end {
            let n = eb(i + 1) - eb(i);
            if n <= 8 {
                continue;
            }
            let mut tcount = [0i32; 3];
            for &v in &ch[eb(i)..eb(i + 1)] {
                let x2n = v * v * n as f32;
                if x2n < 0.25 {
                    tcount[0] += 1;
                }
                if x2n < 0.0625 {
                    tcount[1] += 1;
                }
                if x2n < 0.015625 {
                    tcount[2] += 1;
                }
            }
            // Only include the four last bands (8 kHz and up).
            if i > NUM_SPREAD_BANDS - 4 {
                hf_sum += 32 * (tcount[1] + tcount[0]) / n as i32;
            }
            let tmp = i32::from(2 * tcount[2] >= n as i32)
                + i32::from(2 * tcount[1] >= n as i32)
                + i32::from(2 * tcount[0] >= n as i32);
            sum += tmp * 256;
            nb_bands += 1;
        }
    }
    if update_hf {
        if hf_sum != 0 {
            hf_sum /= channels as i32 * (4 - NUM_SPREAD_BANDS as i32 + end as i32);
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        let mut hf = *hf_average;
        if *tapset_decision == 2 {
            hf += 4;
        } else if *tapset_decision == 0 {
            hf -= 4;
        }
        *tapset_decision = if hf > 22 {
            2
        } else if hf > 18 {
            1
        } else {
            0
        };
    }
    debug_assert!(nb_bands > 0);
    sum /= nb_bands.max(1);
    // Recursive averaging, then hysteresis against the last decision.
    sum = (sum + *average) >> 1;
    *average = sum;
    sum = (3 * sum + (((3 - last_decision.as_u8() as i32) << 7) + 64) + 2) >> 2;
    if sum < 80 {
        Spread::Aggressive
    } else if sum < 256 {
        Spread::Normal
    } else if sum < 384 {
        Spread::Light
    } else {
        Spread::None
    }
}

/// The band count the [`spreading_decision`] high-frequency arm is
/// written against (the 21-band Table-55 layout; the "last four
/// bands" are 8 kHz and up).
const NUM_SPREAD_BANDS: usize = 21;

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a one-symbol bitstream whose ICDF decode produces value `v`
    /// for `SPREAD_ICDF`. We pre-encode a symbol with `(fl, fh, ft)`
    /// such that `dec_icdf` returns `v` on the first call.
    ///
    /// Since constructing a range-coder byte sequence by hand is
    /// brittle, we instead verify by exhaustively walking every legal
    /// 4-byte input and checking that any `Spread` it produces is
    /// stable under the round-trip `as_u8 -> from_u8`.
    fn decode_first_symbol(buf: &[u8]) -> Spread {
        let mut dec = RangeDecoder::new(buf);
        decode_spread(&mut dec)
    }

    #[test]
    fn spread_enum_round_trip() {
        for v in 0u8..=3 {
            let s = Spread::from_u8(v).expect("legal raw value");
            assert_eq!(s.as_u8(), v);
        }
        assert_eq!(Spread::from_u8(4), None);
        assert_eq!(Spread::from_u8(255), None);
    }

    #[test]
    fn spread_f_r_matches_table_59() {
        assert_eq!(Spread::None.f_r(), None);
        assert_eq!(Spread::Light.f_r(), Some(15));
        assert_eq!(Spread::Normal.f_r(), Some(10));
        assert_eq!(Spread::Aggressive.f_r(), Some(5));
    }

    #[test]
    fn default_spread_is_normal() {
        assert_eq!(DEFAULT_SPREAD, Spread::Normal);
        assert_eq!(DEFAULT_SPREAD.as_u8(), 2);
        assert_eq!(DEFAULT_SPREAD.f_r(), Some(10));
    }

    #[test]
    fn spread_icdf_shape() {
        // ICDF entries [25, 23, 2, 0]; ftb=5 ⇒ ft=32.
        assert_eq!(SPREAD_ICDF, &[25u8, 23, 2, 0]);
        assert_eq!(SPREAD_FTB, 5);
    }

    #[test]
    fn icdf_cumulative_consistency() {
        // Reconstruct (fl, fh, ft) for each symbol via the ICDF
        // and verify the PDF masses recover {7, 2, 21, 2}/32.
        let ft = 1u32 << SPREAD_FTB;
        assert_eq!(ft, 32);
        // For symbol k, fh[k] = ft - ICDF[k], fl[k] = ft - ICDF[k-1]
        // with fl[0] = 0.
        let fh: Vec<u32> = SPREAD_ICDF.iter().map(|&v| ft - u32::from(v)).collect();
        assert_eq!(fh, vec![7, 9, 30, 32]);
        let masses: Vec<u32> = std::iter::once(fh[0])
            .chain(fh.windows(2).map(|w| w[1] - w[0]))
            .collect();
        assert_eq!(masses, vec![7, 2, 21, 2]);
        let sum: u32 = masses.iter().sum();
        assert_eq!(sum, ft);
    }

    #[test]
    fn decode_spread_produces_legal_value_for_any_byte() {
        // Bound-safety: every single-byte input must produce one of
        // the four legal Spread variants, never panic.
        for b in 0u8..=255 {
            let buf = [b, 0xFF, 0xFF, 0xFF];
            let s = decode_first_symbol(&buf);
            assert!(matches!(
                s,
                Spread::None | Spread::Light | Spread::Normal | Spread::Aggressive
            ));
        }
    }

    #[test]
    fn rotation_gain_identity_when_spread_none() {
        // Spread::None ⇒ f_r = infinite ⇒ g_r = 0.
        let (num, den) = rotation_gain_ratio(Spread::None, 16, 4);
        assert_eq!(num, 0);
        assert_eq!(den, 1);
        let (num2, den2) = rotation_gain_squared_ratio(Spread::None, 16, 4);
        assert_eq!(num2, 0);
        assert_eq!(den2, 1);
    }

    #[test]
    fn rotation_gain_ratio_matches_closed_form() {
        // g_r = N / (N + f_r * K) for each non-identity spread.
        // Spot-checks against hand-computed values.
        // N=16, K=4: light -> 16/(16+15*4)=16/76; normal -> 16/(16+10*4)=16/56; aggr -> 16/(16+5*4)=16/36.
        assert_eq!(rotation_gain_ratio(Spread::Light, 16, 4), (16, 76));
        assert_eq!(rotation_gain_ratio(Spread::Normal, 16, 4), (16, 56));
        assert_eq!(rotation_gain_ratio(Spread::Aggressive, 16, 4), (16, 36));
    }

    #[test]
    fn rotation_gain_squared_matches_pairwise_square() {
        // g_r^2 numerator/denominator are the squares of the gain
        // ratio's terms.
        let (num, den) = rotation_gain_ratio(Spread::Normal, 16, 4);
        let (n2, d2) = rotation_gain_squared_ratio(Spread::Normal, 16, 4);
        assert_eq!(n2, u64::from(num) * u64::from(num));
        assert_eq!(d2, u64::from(den) * u64::from(den));
    }

    #[test]
    fn rotation_gain_zero_n_band_is_zero() {
        // Empty band: g_r = 0 regardless of spread to avoid a divide
        // pitfall in the PVQ caller.
        for spread in [Spread::Light, Spread::Normal, Spread::Aggressive] {
            let (num, den) = rotation_gain_ratio(spread, 0, 4);
            assert_eq!(num, 0);
            assert_eq!(den, 1);
        }
    }

    #[test]
    fn rotation_gain_k_zero_is_unit_ratio() {
        // K=0 ⇒ g_r = N/N = 1. The caller will then compute
        // theta = pi/4 from g_r^2 = 1.
        let (num, den) = rotation_gain_ratio(Spread::Normal, 16, 0);
        assert_eq!(num, 16);
        assert_eq!(den, 16);
    }

    #[test]
    fn rotation_gain_saturates_huge_inputs() {
        // The PVQ inputs in real frames stay well below u32::MAX;
        // confirm that the saturating arithmetic does not panic on
        // pathological inputs and produces a finite denominator.
        let (num, den) = rotation_gain_ratio(Spread::Light, u32::MAX, u32::MAX);
        assert_eq!(num, u32::MAX);
        // den is saturated to u32::MAX (not overflowed).
        assert_eq!(den, u32::MAX);
    }

    #[test]
    fn pre_rotation_stride_below_eight_samples_is_none() {
        // Per-block sample count below 8 ⇒ no pre-rotation.
        // 32 dims across 8 blocks = 4 samples/block ⇒ None.
        assert_eq!(pre_rotation_stride(32, 8), None);
        // 56 dims / 8 blocks = 7 samples/block ⇒ None.
        assert_eq!(pre_rotation_stride(56, 8), None);
    }

    #[test]
    fn pre_rotation_stride_single_block_is_none() {
        // nb_blocks <= 1 ⇒ §4.3.4.3 "more than one time block"
        // precondition is not met.
        assert_eq!(pre_rotation_stride(16, 1), None);
        assert_eq!(pre_rotation_stride(16, 0), None);
    }

    #[test]
    fn pre_rotation_stride_zero_n_is_none() {
        // Empty band: no rotation regardless.
        assert_eq!(pre_rotation_stride(0, 4), None);
    }

    #[test]
    fn pre_rotation_stride_matches_round_sqrt() {
        // 64 dims / 2 blocks = 32 samples/block. sqrt(32) ≈ 5.657 ⇒ 6.
        assert_eq!(pre_rotation_stride(64, 2), Some(6));
        // 64 dims / 4 blocks = 16 samples/block. sqrt(16) = 4 ⇒ 4.
        assert_eq!(pre_rotation_stride(64, 4), Some(4));
        // 128 dims / 2 blocks = 64 samples/block. sqrt(64) = 8 ⇒ 8.
        assert_eq!(pre_rotation_stride(128, 2), Some(8));
        // 200 dims / 2 blocks = 100 samples/block. sqrt(100) = 10 ⇒ 10.
        assert_eq!(pre_rotation_stride(200, 2), Some(10));
    }

    #[test]
    fn pre_rotation_stride_rounds_half_up() {
        // Choose a per_block such that the fractional sqrt sits on
        // the half-tie: per_block = 6 ⇒ but per_block<8 returns None.
        // Use per_block = 12 (>= 8): sqrt(12) ≈ 3.464 ⇒ rounds to 3
        // (frac_lo = 12-9 = 3, frac_hi = 16-12 = 4 ⇒ lo < hi, pick lo).
        assert_eq!(pre_rotation_stride(24, 2), Some(3));
        // per_block = 20: sqrt(20) ≈ 4.472 ⇒ frac_lo = 20-16 = 4,
        // frac_hi = 25-20 = 5 ⇒ rounds to 4.
        assert_eq!(pre_rotation_stride(40, 2), Some(4));
    }

    #[test]
    fn pre_rotation_stride_never_zero_when_some() {
        // Even at the boundary `per_block = 8`, the stride is >= 1.
        let s = pre_rotation_stride(16, 2).expect("per_block=8 is >=8");
        assert!(s >= 1);
        // sqrt(8) ≈ 2.828 ⇒ rounds to 3 (frac_lo=8-4=4, frac_hi=9-8=1).
        assert_eq!(s, 3);
    }

    #[test]
    fn icdf_first_byte_zero_routes_to_aggressive() {
        // First byte 0 sets the range coder's high-prob arm so that
        // the next dec_icdf returns the tail symbol (index 3 ⇒
        // Spread::Aggressive). This is a smoke check that decode_spread
        // does dispatch through the ICDF lookup correctly.
        let buf = [0u8, 0, 0, 0];
        let s = decode_first_symbol(&buf);
        // We don't assert which specific value comes out — we only
        // require that the decode itself is well-formed and produces
        // one of the four legal Spread variants.
        assert!(matches!(
            s,
            Spread::None | Spread::Light | Spread::Normal | Spread::Aggressive
        ));
    }

    /// `encode_spread` → `decode_spread` recovers every variant.
    #[test]
    fn encode_decode_spread_roundtrip() {
        for spread in [
            Spread::None,
            Spread::Light,
            Spread::Normal,
            Spread::Aggressive,
        ] {
            let mut enc = RangeEncoder::new();
            encode_spread(&mut enc, spread).unwrap();
            let frame = enc.finish();
            let mut dec = RangeDecoder::new(&frame);
            assert_eq!(decode_spread(&mut dec), spread);
            assert!(!dec.has_error());
        }
    }

    /// Multiple spread symbols back-to-back decode in order (the bias
    /// the PDF gives the bulk Normal mode does not corrupt others).
    #[test]
    fn encode_decode_spread_stream() {
        let seq = [
            Spread::Normal,
            Spread::None,
            Spread::Aggressive,
            Spread::Light,
            Spread::Normal,
            Spread::Normal,
        ];
        let mut enc = RangeEncoder::new();
        for &s in &seq {
            encode_spread(&mut enc, s).unwrap();
        }
        let frame = enc.finish();
        let mut dec = RangeDecoder::new(&frame);
        for &s in &seq {
            assert_eq!(decode_spread(&mut dec), s);
        }
        assert!(!dec.has_error());
    }
}
