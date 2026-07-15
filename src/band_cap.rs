//! Per-band cap[] computation + band-boost decoding (RFC 6716 §4.3.3).
//!
//! ## What this module covers
//!
//! Two pieces of the §4.3.3 bit-allocation machinery:
//!
//! 1. The static [`CACHE_CAPS50`] table (`u8` per band per
//!    `(LM, stereo)` row), converted to a per-band cap vector via
//!    [`compute_band_caps`]. The cap controls how many 1/8-bit
//!    boosts a band is allowed to absorb in the band-boost loop
//!    (RFC 6716 §4.3.3 lines 6290–6316).
//! 2. The band-boost decode loop in [`decode_band_boosts`], which
//!    walks each coded band running a per-band dynalloc-logp loop
//!    that emits as many `quanta`-sized boosts as the running
//!    `(total_bits, total_boost, tell, cap[band])` budget permits
//!    (RFC 6716 §4.3.3 lines 6318–6360).
//!
//! Both pieces drop in ahead of the §4.3.3 alloc-trim gate, which
//! tests `ec_tell_frac() + 48 <= total_bits - total_boost` —
//! [`BoostResult::total_boost`] is exactly the `total_boost` value
//! the trim gate consults.
//!
//! ## Clean-room provenance
//!
//! Every numeric value, every formula, and every field comment in
//! this file is transcribed from RFC 6716 §4.3.3
//! (`docs/audio/opus/rfc6716-opus.txt` lines 6290–6360) and the
//! clean-room narrative at
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md`
//! §§2.2–2.3. The [`CACHE_CAPS50`] numeric values come from the
//! Feist-facts numeric extract at
//! `docs/audio/celt/tables/cache_caps50.csv` (metadata in
//! `cache_caps50.meta`). No external library source was consulted.

use crate::coarse_energy::NUM_BANDS;
use crate::range_decoder::RangeDecoder;
use crate::range_encoder::RangeEncoder;
use crate::Error;

/// CELT cache.caps cache, 8 rows × 21 bands per RFC 6716 §4.3.3 +
/// `docs/audio/celt/tables/cache_caps50.csv`.
///
/// The table is indexed as `CACHE_CAPS50[2*LM + stereo][band]`:
///
/// * `LM ∈ {0,1,2,3}` selects the frame size (`120/240/480/960`
///   samples per channel).
/// * `stereo ∈ {0,1}` selects mono (`0`) or stereo (`1`).
/// * `band ∈ 0..21` selects the CELT band per Table 55.
///
/// Each entry is the cached unsigned-byte component of the per-band
/// maximum-allocation vector. [`compute_band_caps`] converts it into
/// the actual cap[] vector via
/// `cap = (CACHE_CAPS50[row][band] + 64) * channels * N / 4`
/// (RFC 6716 §4.3.3 line 6310).
///
/// Numeric values are reproduced from the staged CSV; the row layout
/// (`r => LM = r/2, stereo = r%2`) matches the §4.3.3 prose
/// `i = nbBands * (2*LM + stereo)`.
pub const CACHE_CAPS50: [[u8; NUM_BANDS]; 8] = [
    // LM=0, mono
    [
        224, 224, 224, 224, 224, 224, 224, 224, 160, 160, 160, 160, 185, 185, 185, 178, 178, 168,
        134, 61, 37,
    ],
    // LM=0, stereo
    [
        224, 224, 224, 224, 224, 224, 224, 224, 240, 240, 240, 240, 207, 207, 207, 198, 198, 183,
        144, 66, 40,
    ],
    // LM=1, mono
    [
        160, 160, 160, 160, 160, 160, 160, 160, 185, 185, 185, 185, 193, 193, 193, 183, 183, 172,
        138, 64, 38,
    ],
    // LM=1, stereo
    [
        240, 240, 240, 240, 240, 240, 240, 240, 207, 207, 207, 207, 204, 204, 204, 193, 193, 180,
        143, 66, 40,
    ],
    // LM=2, mono
    [
        185, 185, 185, 185, 185, 185, 185, 185, 193, 193, 193, 193, 193, 193, 193, 183, 183, 172,
        138, 65, 39,
    ],
    // LM=2, stereo
    [
        207, 207, 207, 207, 207, 207, 207, 207, 204, 204, 204, 204, 201, 201, 201, 188, 188, 176,
        141, 66, 40,
    ],
    // LM=3, mono
    [
        193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 193, 194, 194, 194, 184, 184, 173,
        139, 65, 39,
    ],
    // LM=3, stereo
    [
        204, 204, 204, 204, 204, 204, 204, 204, 201, 201, 201, 201, 198, 198, 198, 187, 187, 175,
        140, 66, 40,
    ],
];

/// Convert one row of [`CACHE_CAPS50`] into the actual per-band cap[]
/// vector for a given mode (RFC 6716 §4.3.3 lines 6305–6313).
///
/// The §4.3.3 prose specifies the conversion:
///
/// ```text
/// i = nbBands * (2 * LM + stereo)
/// cap[band] = (cache.caps[i + band] + 64) * channels * N[band] / 4
/// ```
///
/// where:
///
/// * `lm` ∈ `0..=3` selects the frame size (`120 << LM` samples per
///   channel per frame).
/// * `stereo` is `false` for one channel, `true` for two.
/// * `channels` is `1` or `2` and equals `stereo as u32 + 1` in
///   practice; the §4.3.3 prose treats it as an independent input so
///   callers can supply it directly.
/// * `bins_per_band[band]` is the number of MDCT bins covered by
///   `band` in one channel — Table 55 layout dependent (RFC 6716
///   §4.3.3 line 6308 "N to the number of MDCT bins covered by the
///   band (for one channel)").
///
/// The output `caps` slice receives `cap[band]` for every band the
/// caller wants populated; values that exceed `i16::MAX` saturate to
/// `i16::MAX` defensively (the §4.3.3 prose notes the values "fit in
/// signed 16-bit integers but do not fit in 8 bits"). The integer
/// division is RFC-prose `divide ... by 4 using integer division`.
///
/// Returns `false` and leaves `caps` unchanged when:
///
/// * `lm > 3` (outside the §4.3.3 frame-size range).
/// * `channels == 0` or `channels > 2`.
/// * `caps.len() != bins_per_band.len()` (band-count mismatch).
/// * `caps.len() > NUM_BANDS` (Table 55 caps the band count).
///
/// Returns `true` on success.
pub fn compute_band_caps(
    lm: u32,
    stereo: bool,
    channels: u32,
    bins_per_band: &[u32],
    caps: &mut [i16],
) -> bool {
    if lm > 3 {
        return false;
    }
    if !(1..=2).contains(&channels) {
        return false;
    }
    if caps.len() != bins_per_band.len() || caps.len() > NUM_BANDS {
        return false;
    }
    let row = (2 * lm as usize) + stereo as usize;
    let cache_row = &CACHE_CAPS50[row];
    for (b, (&n, out)) in bins_per_band.iter().zip(caps.iter_mut()).enumerate() {
        // RFC 6716 §4.3.3: cap = (cache.caps[i] + 64) * channels * N / 4.
        // u32 arithmetic with saturation into i16; the worst case at
        // §4.3.3 LM=3 stereo (channels=2, N up to ~120 per band) is
        // (240+64)*2*120/4 = 18_240 which fits in i16 (max 32_767).
        let num = (cache_row[b] as u32 + 64) * channels * n;
        let val = (num / 4) as i32;
        *out = val.min(i16::MAX as i32) as i16;
    }
    true
}

/// Per-band §4.3.3 band-boost decode result.
///
/// Carries the per-band `boost` vector (in 1/8 bits, one entry per
/// coded band starting at the §4.3.3 `start` band) plus the running
/// `total_boost` accumulator the §4.3.3 trim gate consumes.
///
/// `total_bits_remaining` records the §4.3.3 `total_bits` field at
/// the end of the loop. The §4.3.3 prose specifies that the loop
/// subtracts `quanta` from `total_bits` every time a boost lands;
/// the caller's running budget tracks this value going into the
/// alloc-trim / skip / intensity reservation steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoostResult {
    /// Per-band boost in 1/8 bits, indexed from the coding-start
    /// band (`0` for pure CELT, `17` for Hybrid mode) up through
    /// `end - 1`. `boost[b - start]` is band `b`'s allocation
    /// boost.
    pub boost: Vec<i32>,
    /// Sum of all per-band boosts in 1/8 bits — the §4.3.3
    /// `total_boost` accumulator that the alloc-trim gate
    /// consumes (`ec_tell_frac() + 48 <= total_bits - total_boost`).
    pub total_boost: i32,
    /// `total_bits` at loop exit, in 1/8 bits. The §4.3.3 prose
    /// subtracts `quanta` from `total_bits` for every accepted
    /// boost; the running value at exit is the budget the
    /// downstream allocation steps inherit.
    pub total_bits_remaining: i32,
}

/// Decode the §4.3.3 band-boost loop.
///
/// Implements the loop per the normative RFC 6716 Appendix A listing
/// (`celt.c`, the decode-side dynalloc walk; the §4.3.3 prose at
/// lines 6339–6360 narrates the same loop with two slips the listing
/// resolves — the inner-loop budget comparison is against the
/// *diminishing* `total_bits` with the symbol cost scaled to 1/8
/// bits, and the quanta width counts both channels):
///
/// 1. Initialise `dynalloc_logp = 6` (the 6-bit / `p = 1/64`
///    starting cost for a boost), `total_bits` from the caller, and
///    `total_boost = 0`.
/// 2. For each band `b` in `start..end`:
///    * `width = channels * N[b]`,
///      `quanta = min(8*width, max(48, width))` — a 6-bit step size,
///      floored at 1/8 bit/sample and capped at 1 bit/sample.
///    * `boost = 0`, `dynalloc_loop_logp = dynalloc_logp`.
///    * While `tell + (dynalloc_loop_logp << 3) < total_bits`
///      AND `boost < cap[b]`: decode one bit with cost
///      `dynalloc_loop_logp` ([`RangeDecoder::dec_bit_logp`]),
///      update `tell` (via `tell_frac`). On `0` break the inner
///      loop. On `1`, add `quanta` to `boost` and `total_boost`,
///      subtract `quanta` from `total_bits`, and set
///      `dynalloc_loop_logp = 1`.
///    * After the inner loop: if `boost != 0` and `dynalloc_logp > 2`,
///      decrement `dynalloc_logp`.
///
/// Inputs:
///
/// * `dec` — range decoder, advanced by every emitted boost bit.
/// * `start` / `end` — `start <= end <= 21`; the band range covered
///   by this frame. `end - start` boost values are produced.
/// * `channels` — 1 or 2; folded into the per-band quanta width.
/// * `bins_per_band` — `N[b]` (MDCT bins for one channel) for
///   `b ∈ start..end`. Must have length `end - start`.
/// * `caps` — `cap[b]` (output of [`compute_band_caps`]) for
///   `b ∈ start..end`. Must have length `end - start`.
/// * `total_bits` — initial value, in 1/8 bits. Per §4.3.3 prose:
///   "the size of the frame in 8th bits".
///
/// Returns `None` when the inputs are inconsistent (slice lengths
/// don't agree, or `end > 21`, or `start > end`). Otherwise returns a
/// [`BoostResult`] with the per-band boosts, accumulated
/// `total_boost`, and final `total_bits`.
///
/// The §4.3.3 prose explicitly notes "at very low rates ... the inner
/// loop may not run even once" — the function tolerates this and
/// simply emits zero boosts. Per-band zero-boost behaviour is the
/// no-bit-emission path: the loop exits before reading anything from
/// the decoder.
pub fn decode_band_boosts(
    dec: &mut RangeDecoder<'_>,
    start: u32,
    end: u32,
    channels: u32,
    bins_per_band: &[u32],
    caps: &[i16],
    total_bits: i32,
) -> Option<BoostResult> {
    if !(1..=2).contains(&channels) {
        return None;
    }
    if start > end || end > NUM_BANDS as u32 {
        return None;
    }
    let coded_bands = (end - start) as usize;
    if bins_per_band.len() != coded_bands || caps.len() != coded_bands {
        return None;
    }
    let mut boost = vec![0i32; coded_bands];
    let mut total_boost: i32 = 0;
    let mut total_bits = total_bits;
    // RFC §4.3.3 line 6339: dynalloc_logp starts at 6 (6-bit cost of
    // the first boost in a band, p = 1/64).
    let mut dynalloc_logp: u32 = 6;

    for (idx, (&n, &cap_b)) in bins_per_band.iter().zip(caps.iter()).enumerate() {
        // Appendix A: width = C*N (both channels); quanta =
        // min(8*width, max(48, width)) — 6-bit step floored at
        // 1/8 bit/sample, capped at 1 bit/sample.
        let width = (channels * n) as i32;
        let quanta = (8 * width).min(width.max(48));
        let mut band_boost: i32 = 0;
        let mut dynalloc_loop_logp = dynalloc_logp;

        loop {
            // tell() in 1/8 bits comes from ec_tell_frac (§4.1.6.2).
            let tell = dec.tell_frac() as i32;
            // Appendix A loop guard: tell + (logp << 3) < total_bits
            // (which diminishes by quanta per boost) AND
            // boost < cap[band].
            if tell + ((dynalloc_loop_logp as i32) << 3) >= total_bits {
                break;
            }
            if band_boost >= cap_b as i32 {
                break;
            }
            let bit = dec.dec_bit_logp(dynalloc_loop_logp);
            if bit == 0 {
                // §4.3.3: "If the decoded value is zero break the loop."
                break;
            }
            // §4.3.3: accept boost — quanta units land in this band.
            band_boost += quanta;
            total_boost += quanta;
            total_bits -= quanta;
            // §4.3.3: "subsequent boosts in a band cost only a single
            // bit": dynalloc_loop_logp drops to 1 after the first
            // accepted boost in this band.
            dynalloc_loop_logp = 1;
        }

        boost[idx] = band_boost;
        // §4.3.3: "If boost is non-zero and dynalloc_logp is greater
        // than 2, decrease dynalloc_logp." The starting cost for the
        // NEXT band drops by one for every band that landed a boost,
        // down to the §4.3.3 floor of 2 (p = 1/4).
        if band_boost != 0 && dynalloc_logp > 2 {
            dynalloc_logp -= 1;
        }
    }

    Some(BoostResult {
        boost,
        total_boost,
        total_bits_remaining: total_bits,
    })
}

/// Encode the §4.3.3 band-boost loop — the exact inverse of
/// [`decode_band_boosts`].
///
/// Walks the same per-band dynalloc-logp loop the decoder runs, but
/// instead of reading each gated bit it **writes** one: a `1` while the
/// band still owes boost quanta from `target_boost`, then a terminating
/// `0` — emitted only when the decoder would go on to read it (the
/// §4.3.3 gate `tell + dynalloc_loop_logp < total_bits + total_boost`
/// AND `boost < cap[band]` still open). The encoder's
/// [`RangeEncoder::tell_frac`] locksteps with the decoder's after the
/// same symbols (§5.1.6 / §4.1.6), so the two sides evaluate every gate
/// identically.
///
/// `target_boost[i]` is the desired boost for coded band `start + i`,
/// in 1/8 bits. A target that is not a multiple of the band's `quanta`
/// (`min(8*N, max(48, N))`) is floored to the nearest multiple; a
/// target the running budget or the band cap cannot absorb is truncated
/// at the point the gate closes — the decoder can never reconstruct
/// more than the gates admit, so the encoder never writes more.
///
/// Returns the [`BoostResult`] that a matching [`decode_band_boosts`]
/// over the finished frame reconstructs (per-band boosts actually
/// encoded, `total_boost`, and the exit `total_bits`). Callers that
/// need the request honoured exactly compare `result.boost` with
/// `target_boost`.
///
/// Returns [`Error::InvalidParameter`] on inconsistent slice lengths /
/// band window, or propagates a [`RangeEncoder`] rejection.
#[allow(clippy::too_many_arguments)] // mirrors the §4.3.3 loop inputs
pub fn encode_band_boosts(
    enc: &mut RangeEncoder,
    start: u32,
    end: u32,
    channels: u32,
    bins_per_band: &[u32],
    caps: &[i16],
    total_bits: i32,
    target_boost: &[i32],
) -> Result<BoostResult, Error> {
    if start > end || end > NUM_BANDS as u32 || !(1..=2).contains(&channels) {
        return Err(Error::InvalidParameter);
    }
    let coded_bands = (end - start) as usize;
    if bins_per_band.len() != coded_bands
        || caps.len() != coded_bands
        || target_boost.len() != coded_bands
    {
        return Err(Error::InvalidParameter);
    }
    let mut boost = vec![0i32; coded_bands];
    let mut total_boost: i32 = 0;
    let mut total_bits = total_bits;
    // §4.3.3: dynalloc_logp starts at 6 (p = 1/64 for the first boost).
    let mut dynalloc_logp: u32 = 6;

    for (idx, ((&n, &cap_b), &want)) in bins_per_band
        .iter()
        .zip(caps.iter())
        .zip(target_boost.iter())
        .enumerate()
    {
        let width = (channels * n) as i32;
        let quanta = (8 * width).min(width.max(48));
        // Floor the request to a whole number of quanta (the §4.3.3
        // boost step is quantized); a negative request is zero.
        let mut want_steps = if quanta > 0 { want.max(0) / quanta } else { 0 };
        let mut band_boost: i32 = 0;
        let mut dynalloc_loop_logp = dynalloc_logp;

        loop {
            let tell = enc.tell_frac() as i32;
            // Same gates as the decode loop: when either closes, the
            // decoder stops without reading, so the encoder stops
            // without writing.
            if tell + ((dynalloc_loop_logp as i32) << 3) >= total_bits {
                break;
            }
            if band_boost >= cap_b as i32 {
                break;
            }
            if want_steps > 0 {
                // One more boost for this band.
                enc.enc_bit_logp(1, dynalloc_loop_logp)?;
                want_steps -= 1;
                band_boost += quanta;
                total_boost += quanta;
                total_bits -= quanta;
                dynalloc_loop_logp = 1;
            } else {
                // Terminate the band: the decoder reads this 0 and
                // breaks.
                enc.enc_bit_logp(0, dynalloc_loop_logp)?;
                break;
            }
        }

        boost[idx] = band_boost;
        if band_boost != 0 && dynalloc_logp > 2 {
            dynalloc_logp -= 1;
        }
    }

    Ok(BoostResult {
        boost,
        total_boost,
        total_bits_remaining: total_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `encode_band_boosts` → `decode_band_boosts` round-trips: the
    /// decoder reconstructs exactly the boosts the encoder reported it
    /// wrote, and the two `BoostResult`s agree field for field.
    #[test]
    fn encode_decode_boost_roundtrip() {
        // 6 bands with mixed bin counts; ample budget.
        let bins = [4u32, 8, 8, 16, 24, 32];
        let mut caps = [0i16; 6];
        assert!(compute_band_caps(2, false, 1, &bins, &mut caps));
        let total_bits = 8000i32;

        let targets: &[[i32; 6]] = &[
            [0, 0, 0, 0, 0, 0],
            // quanta per band: min(8N, max(48, N)) = [32, 48, 48, 48, 48, 48]... N=4 → 32.
            [32, 0, 48, 0, 96, 48],
            [64, 48, 0, 144, 0, 0],
            // Oversized requests truncate at the cap / budget.
            [30000, 0, 0, 0, 0, 30000],
        ];
        for target in targets {
            let mut enc = RangeEncoder::new();
            let enc_result =
                encode_band_boosts(&mut enc, 0, 6, 1, &bins, &caps, total_bits, target).unwrap();
            let mut frame = enc.finish();
            // Pad so the decoder's tell_frac gates see the same running
            // budget arithmetic (the gates only use tell_frac, which is
            // frame-length independent, but keep the frame non-empty).
            frame.resize(frame.len().max(8), 0);
            let mut dec = RangeDecoder::new(&frame);
            let dec_result =
                decode_band_boosts(&mut dec, 0, 6, 1, &bins, &caps, total_bits).unwrap();
            assert_eq!(dec_result, enc_result, "target {target:?}");
            assert!(!dec.has_error());
        }
    }

    /// A target that is a whole number of quanta and inside every gate
    /// is honoured exactly.
    #[test]
    fn encode_boosts_honours_feasible_target() {
        let bins = [8u32, 8, 8];
        let mut caps = [0i16; 3];
        assert!(compute_band_caps(1, false, 1, &bins, &mut caps));
        // quanta = min(64, max(48, 8)) = 48 for every band.
        let target = [48i32, 96, 48];
        let mut enc = RangeEncoder::new();
        let result = encode_band_boosts(&mut enc, 0, 3, 1, &bins, &caps, 8000, &target).unwrap();
        assert_eq!(result.boost, target.to_vec());
        assert_eq!(result.total_boost, 192);
        // Non-multiple targets floor to the quanta grid.
        let mut enc2 = RangeEncoder::new();
        let result2 =
            encode_band_boosts(&mut enc2, 0, 3, 1, &bins, &caps, 8000, &[50, 47, -3]).unwrap();
        assert_eq!(result2.boost, vec![48, 0, 0]);
    }

    /// A starved budget closes the outer gate immediately: no bits are
    /// written, all boosts are zero — matching the decoder's "inner
    /// loop may not run even once" path.
    #[test]
    fn encode_boosts_starved_budget_writes_nothing() {
        let bins = [8u32, 8];
        let mut caps = [0i16; 2];
        assert!(compute_band_caps(0, false, 1, &bins, &mut caps));
        let mut enc = RangeEncoder::new();
        let tell_before = enc.tell_frac();
        let result = encode_band_boosts(&mut enc, 0, 2, 1, &bins, &caps, 0, &[48, 48]).unwrap();
        assert_eq!(result.boost, vec![0, 0]);
        assert_eq!(result.total_boost, 0);
        assert_eq!(
            enc.tell_frac(),
            tell_before,
            "bits written on a closed gate"
        );
    }

    /// Inconsistent inputs are rejected.
    #[test]
    fn encode_boosts_rejects_bad_inputs() {
        let mut enc = RangeEncoder::new();
        let bins = [8u32; 3];
        let caps = [100i16; 3];
        // target length mismatch.
        assert_eq!(
            encode_band_boosts(&mut enc, 0, 3, 1, &bins, &caps, 1000, &[0, 0]),
            Err(Error::InvalidParameter)
        );
        // band window out of range.
        assert_eq!(
            encode_band_boosts(&mut enc, 0, 22, 1, &bins, &caps, 1000, &[0, 0, 0]),
            Err(Error::InvalidParameter)
        );
    }

    /// CACHE_CAPS50 has 8 rows × 21 bands (RFC 6716 §4.3.3 +
    /// `docs/audio/celt/tables/cache_caps50.meta`).
    #[test]
    fn cache_caps_layout_matches_csv() {
        assert_eq!(CACHE_CAPS50.len(), 8);
        for row in &CACHE_CAPS50 {
            assert_eq!(row.len(), NUM_BANDS);
        }
        // Spot-check a few §4.3.3 cells against the CSV at
        // docs/audio/celt/tables/cache_caps50.csv.
        // Row 0 = LM=0 mono: starts with 224 across the low bands,
        // ends with [37].
        assert_eq!(CACHE_CAPS50[0][0], 224);
        assert_eq!(CACHE_CAPS50[0][20], 37);
        // Row 7 = LM=3 stereo: starts with 204, ends with 40.
        assert_eq!(CACHE_CAPS50[7][0], 204);
        assert_eq!(CACHE_CAPS50[7][20], 40);
        // Row 3 = LM=1 stereo: band 8 = 207 per CSV.
        assert_eq!(CACHE_CAPS50[3][8], 207);
    }

    /// `compute_band_caps` applies the §4.3.3 formula
    /// `cap = (cache.caps[i] + 64) * channels * N / 4` per band.
    #[test]
    fn compute_band_caps_applies_rfc_formula() {
        // LM=2 mono (row index 2*2+0 = 4); pick 5 bands with N=8 each
        // for an easy hand-check.
        let bins = [8u32; 5];
        let mut caps = [0i16; 5];
        let ok = compute_band_caps(2, false, 1, &bins, &mut caps);
        assert!(ok);
        // Cache row 4 band 0 = 185 per CSV; (185+64)*1*8/4 = 498.
        assert_eq!(caps[0], 498);
        // Bands 1..=4 are also 185 → all 498.
        for &c in &caps[1..] {
            assert_eq!(c, 498);
        }
    }

    /// Stereo doubles the cap (channels=2) per §4.3.3 line 6311.
    #[test]
    fn compute_band_caps_stereo_doubles() {
        // LM=0 over all 21 bands so we can index band 8 (where the
        // §4.3.3 cache row diverges interestingly between mono and
        // stereo). Row index for mono is 2*0+0 = 0 (band 8 = 160),
        // for stereo 2*0+1 = 1 (band 8 = 240).
        let bins = [4u32; 21];
        let mut caps_mono = [0i16; 21];
        let mut caps_stereo = [0i16; 21];
        compute_band_caps(0, false, 1, &bins, &mut caps_mono);
        compute_band_caps(0, true, 2, &bins, &mut caps_stereo);
        // Mono cap band 8: (160+64)*1*4/4 = 224.
        assert_eq!(caps_mono[8], 224);
        // Stereo cap band 8: (240+64)*2*4/4 = 608.
        assert_eq!(caps_stereo[8], 608);
        // Mono cap band 0 (cache=224): (224+64)*1*4/4 = 288.
        assert_eq!(caps_mono[0], 288);
        // Stereo cap band 0 (cache=224): (224+64)*2*4/4 = 576.
        assert_eq!(caps_stereo[0], 576);
    }

    /// Reject malformed inputs: out-of-range LM, wrong channel count,
    /// length mismatch.
    #[test]
    fn compute_band_caps_rejects_bad_inputs() {
        let bins = [4u32; 2];
        let mut caps = [0i16; 2];
        // LM > 3 forbidden.
        assert!(!compute_band_caps(4, false, 1, &bins, &mut caps));
        // channels==0 forbidden.
        assert!(!compute_band_caps(0, false, 0, &bins, &mut caps));
        // channels==3 forbidden.
        assert!(!compute_band_caps(0, false, 3, &bins, &mut caps));
        // length mismatch.
        let bins_short = [4u32];
        assert!(!compute_band_caps(0, false, 1, &bins_short, &mut caps));
        // caps too long (>NUM_BANDS).
        let bins_big = [4u32; 22];
        let mut caps_big = [0i16; 22];
        assert!(!compute_band_caps(0, false, 1, &bins_big, &mut caps_big));
    }

    /// At very low rates the §4.3.3 inner loop "may not run even
    /// once": with a `total_bits` smaller than the initial
    /// dynalloc_logp + ec_tell_frac, every band records a zero boost
    /// and the range decoder is never consulted for boost bits.
    #[test]
    fn band_boost_low_rate_no_decode() {
        let buf = [0u8; 64];
        let mut dec = RangeDecoder::new(&buf);
        let tell_before = dec.tell_frac();
        // Set total_bits well below the §4.3.3 loop guard:
        // tell + dynalloc_logp >= total_bits + 0 must be true even
        // on the first iteration of the first band. The dec starts
        // at tell_frac() = 8 (1 bit termination reserve). With
        // dynalloc_logp = 6 the guard fails at total_bits <= 14.
        let bins = [4u32; 21];
        let caps = [100i16; 21];
        let result = decode_band_boosts(&mut dec, 0, 21, 1, &bins, &caps, 8).expect("decoder ok");
        assert_eq!(result.boost, vec![0i32; 21]);
        assert_eq!(result.total_boost, 0);
        assert_eq!(result.total_bits_remaining, 8);
        // No bits should have been consumed beyond the initial state.
        assert_eq!(dec.tell_frac(), tell_before);
    }

    /// A buffer of all-`0xFF` bytes biases `dec_bit_logp` toward "1",
    /// so several bands will get one accepted boost. Verify the
    /// total_boost accumulator + per-band quanta arithmetic.
    #[test]
    fn band_boost_accepts_boosts_against_one_biased_stream() {
        let buf = [0xFFu8; 64];
        let mut dec = RangeDecoder::new(&buf);
        // Generous budget so the loop guard never fires.
        let bins = [8u32; 21];
        let caps = [200i16; 21];
        let result =
            decode_band_boosts(&mut dec, 0, 21, 1, &bins, &caps, 4096).expect("decoder ok");
        // At least one boost should have landed somewhere.
        let nonzero = result.boost.iter().filter(|&&b| b > 0).count();
        assert!(
            nonzero > 0,
            "expected at least one band-boost on biased stream; got boosts {:?}",
            result.boost
        );
        // total_boost must equal sum of per-band boosts.
        let sum: i32 = result.boost.iter().sum();
        assert_eq!(sum, result.total_boost);
        // §4.3.3 quanta = min(8*N, max(48, N)) with N=8 → min(64, 48) = 48.
        // Every per-band boost must be a multiple of 48.
        for &b in &result.boost {
            assert_eq!(b % 48, 0, "per-band boost not a quanta multiple: {b}");
        }
        // total_bits subtracted by total_boost (the §4.3.3 prose).
        assert_eq!(result.total_bits_remaining, 4096 - result.total_boost);
    }

    /// A band's boost stops at cap[band] (§4.3.3 guard `boost <
    /// cap[]`). Set cap to one quanta and confirm the loop emits at
    /// most that much for the saturated band.
    #[test]
    fn band_boost_respects_cap_ceiling() {
        let buf = [0xFFu8; 64];
        let mut dec = RangeDecoder::new(&buf);
        let bins = [8u32]; // single band, quanta = 48
        let caps = [48i16]; // cap exactly one quanta
        let result = decode_band_boosts(&mut dec, 0, 1, 1, &bins, &caps, 4096).expect("decoder ok");
        // Boost is bounded by cap; with cap = quanta = 48, at most
        // one accepted boost iteration: band_boost == 0 or 48, never
        // 96. (After one accept band_boost = 48 = cap, the
        // `boost < cap` guard fails and the loop exits without
        // reading another bit.)
        assert!(
            result.boost[0] == 0 || result.boost[0] == 48,
            "boost not bounded by cap=48: got {}",
            result.boost[0]
        );
    }

    /// Bins length mismatch and band-range errors return None.
    #[test]
    fn band_boost_rejects_bad_inputs() {
        let buf = [0u8; 16];
        let mut dec = RangeDecoder::new(&buf);
        // bins length != end - start.
        let bins = [8u32, 8];
        let caps = [100i16];
        assert!(decode_band_boosts(&mut dec, 0, 1, 1, &bins, &caps, 1000).is_none());
        // start > end.
        let bins_ok = [8u32];
        assert!(decode_band_boosts(&mut dec, 5, 3, 1, &bins_ok, &caps, 1000).is_none());
        // end > NUM_BANDS.
        let bins_big = [8u32; 22];
        let caps_big = [100i16; 22];
        assert!(decode_band_boosts(&mut dec, 0, 22, 1, &bins_big, &caps_big, 1000).is_none());
    }

    /// Quanta computation per §4.3.3 line 6345: `min(8*N, max(48, N))`.
    /// Verify by feeding bands with N = {1, 4, 6, 8, 16, 32, 64}.
    /// quanta = floor of the function:
    /// * N=1  → min(8, max(48,1))  = min(8, 48)  = 8
    /// * N=4  → min(32, max(48,4)) = min(32, 48) = 32
    /// * N=6  → min(48, max(48,6)) = min(48, 48) = 48
    /// * N=8  → min(64, max(48,8)) = min(64, 48) = 48
    /// * N=16 → min(128, max(48,16)) = min(128, 48) = 48
    /// * N=32 → min(256, max(48,32)) = min(256, 48) = 48
    /// * N=64 → min(512, max(48,64)) = min(512, 64) = 64
    #[test]
    fn band_boost_quanta_formula_holds() {
        // We can't observe quanta directly except by counting per-band
        // boost increments. Run the loop with a one-biased stream and
        // an effectively-infinite cap, then check that every per-band
        // boost is a multiple of the expected quanta for that band.
        let buf = [0xFFu8; 256];
        let bins = [1u32, 4, 6, 8, 16, 32, 64];
        let caps = [10_000i16; 7];
        let expected_q = [8, 32, 48, 48, 48, 48, 64];
        let mut dec = RangeDecoder::new(&buf);
        let result =
            decode_band_boosts(&mut dec, 0, 7, 1, &bins, &caps, 16_384).expect("decoder ok");
        for (b, (&boost_v, &q)) in result.boost.iter().zip(expected_q.iter()).enumerate() {
            assert_eq!(
                boost_v % q,
                0,
                "band {b}: boost {boost_v} not a multiple of quanta {q}"
            );
        }
    }
}
