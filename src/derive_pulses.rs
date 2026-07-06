//! Derive per-band PVQ pulse counts from a decoded frame prefix — the
//! documented half of the §4.3.3 → §4.3.4.1 allocation seam (RFC 6716).
//!
//! ## What this module covers
//!
//! [`crate::frame_synthesis::decode_celt_frame`] takes the per-band
//! pulse counts `band_k` (and the per-band fine-bit counts) as **caller
//! inputs**. Those inputs are the output of the §4.3.3 allocation pass.
//! Two parts of that pass make different demands on provenance:
//!
//! * The parts RFC 6716 §4.3.3 / §4.3.4.1 fully specify — the
//!   interpolated static allocation, the boost/trim/cap combination
//!   (the column search [`crate::alloc_combine::find_combined_alloc`]),
//!   and the §4.3.4.1 bits-to-pulses search with its balance accumulator
//!   ([`crate::bits_to_pulses::bits_to_pulses_band_loop`]) — are
//!   implemented in this crate.
//! * The parts RFC 6716 §4.3.3 defers to the reference implementation —
//!   the `interp_bits2pulses` per-band reallocation bisection with
//!   concurrent skip decoding and the fine-energy-vs-shape split — are a
//!   documented docs gap.
//!
//! ## Cost model: the bit-exact pulse cache (corrected LM-major trace)
//!
//! The bits-to-pulses search here runs on the **bit-exact
//! `cache_index50` / `cache_bits50` cost curves**
//! ([`bits_to_pulses_band_loop_cached`]) under the corrected LM-major
//! index mapping (`docs/audio/opus/pulse-cache-format-trace.md`,
//! corrected per issue #184) with the trace §2.3 `qbits[K] + 1`
//! retrieval convention. This replaces the r389 §4.1.5 worst-case
//! `dec_uint` estimator, which the earlier (band-major) reading of the
//! trace had forced: under that reading the cache priced some symbols
//! far below their wire cost (a 2-bin band's `K = 40` index at 7
//! eighth-bits vs ~59 monolithic) and mixed band sizes onto single
//! cost curves. The corrected mapping resolves both defects, and the
//! budget-safety property transfers: the retrieved cost
//! `qbits[K] + 1` never under-prices the monolithic
//! `ceil(8*log2 V(N, K))` index cost (exact for `K <= 16`; the
//! high-`K` surplus is the splitting-aware accounting) — proven
//! against this crate's own `V(N, K)` combinatorics in the
//! `pulse_cache` validation tests — while pricing up to 7 eighth-bits
//! *tighter* than the whole-bit estimator, so the same budget buys
//! more pulses without overrunning the frame. The cached walk also
//! caps each band's `K` at the run's `maxK` (the cached pulse
//! ceiling), which keeps every derived codebook `u32`-representable
//! by construction.
//!
//! [`derive_band_pulses`] composes the **documented** parts into the
//! single derivation the residual loop's `band_k` input wants: from a
//! decoded [`FramePrefix`] it runs the column search over the
//! post-boost budget, then the §4.3.4.1 bits-to-pulses loop (threading
//! the balance accumulator across the coded-band window in spec order),
//! and returns the per-band pulse counts. It treats the whole combined
//! column-search allocation as the **shape** budget — i.e. it makes no
//! fine-energy reservation, which is the maximal documented
//! approximation given the deferred fine/shape split. Each pulse count
//! is clamped to the largest `K` whose PVQ codebook `V(N, K)` is still
//! representable in the documented single-block decode (a `K` that
//! overflows is exactly the regime the deferred §4.3.4.4 band split
//! exists for).
//!
//! [`decode_celt_frame_auto`] chains the whole thing: it decodes the
//! prefix, derives the pulse counts, and runs the documented mono
//! synthesis ([`decode_celt_frame`]) with no fine refinement — a
//! caller-input-free end-to-end mono decode of the documented seam.
//!
//! ## Why this is not the full decode
//!
//! The deferred fine/shape split means the derived `band_k` is **not**
//! guaranteed bit-exact against the reference allocator for a given
//! frame; it is the documented arithmetic composing coherently. When the
//! `interp_bits2pulses` docs gap (the precise per-band bisection + skip
//! decoding) closes, the split can be folded in here and the derivation
//! becomes exact. Until then, this is the deepest caller-input-free
//! decode the wall permits, and it is deterministic: identical bytes
//! produce identical pulse counts and identical PCM.
//!
//! ## Clean-room provenance
//!
//! The composition order is RFC 6716 §4.3.3 (the allocation search,
//! lines 6111–6229) → §4.3.4.1 (bits-to-pulses + balance, lines
//! 6476–6492). Every step delegates to an existing RFC-grounded module
//! whose own provenance is recorded in that module. No external library
//! source was consulted.

use crate::alloc_combine::find_combined_alloc;
use crate::band_minimums::compute_thresh;
use crate::band_minimums::BAND_BINS_LM;
use crate::bits_to_pulses::bits_to_pulses_band_loop_cached_thresh;
use crate::coarse_energy::{CoarseEnergyState, NUM_BANDS};
use crate::frame_decode::{decode_frame_prefix, FramePrefix};
use crate::frame_synthesis::{decode_celt_frame, CeltDecodeState, DecodedFrame};
use crate::pvq::{v_count, V_COUNT_SATURATION};
use crate::range_decoder::RangeDecoder;
use crate::Error;

/// Largest `K' <= K` such that the PVQ codebook size `V(N, K')` does not
/// saturate (i.e. it fits in `u32`). `v_count` is monotone non-decreasing
/// in `K`, so a descent from `K` finds the cap; in practice it engages
/// only on wide bands at high `K`, where the deferred §4.3.4.4 band split
/// would otherwise apply.
fn clamp_k_to_decodable(n: u32, k: u32) -> u32 {
    let mut k = k;
    while k > 0 && v_count(n, k) == V_COUNT_SATURATION {
        k -= 1;
    }
    k
}

/// The shape budget the §4.3.3 column search drives toward, in 1/8
/// bits: the band-boost loop's arithmetic `total_bits` at exit,
/// **capped by the re-measured wire remainder**
/// (`frame_bytes * 64 - tell_frac_after_prefix - 1`, the §4.3.3
/// "remaining available 8th bits" form re-read at the post-prefix
/// coder position). The arithmetic budget never pays for the
/// boost-flag / trim / skip / intensity / dual symbols themselves, so
/// without the cap a stereo frame (two PVQ indices per band, twice
/// the per-symbol overhead) can overrun its byte budget. Both sides
/// compute the identical value from the bit-identical prefix, so the
/// encoder/decoder lockstep is preserved.
fn shape_budget_1_8th(prefix: &FramePrefix) -> i64 {
    let arithmetic = i64::from(prefix.boosts.total_bits_remaining.max(0));
    let wire = i64::from(prefix.frame_bytes) * 64 - i64::from(prefix.tell_frac_after_prefix) - 1;
    arithmetic.min(wire.max(0))
}

/// Per-coded-PVQ-symbol budget provision in 1/8 bits.
///
/// The bit-exact cache prices a `K`-pulse index at
/// `ceil(8*log2 V(N, K))` in its exact-identity region, but the §5.1.4
/// `enc_uint` wire cost measured through this crate's own range
/// encoder can exceed that by **one** eighth-bit (`tell_frac` interval
/// rounding; e.g. `ft = 279`), never more. One provisioned eighth-bit
/// per coded PVQ symbol therefore makes the budget rigorous:
/// `sum(actual) <= sum(priced) + symbols <= (budget - symbols) +
/// symbols = budget`. Both sides derive the identical provision from
/// the bit-identical prefix, so lockstep is preserved.
const PER_SYMBOL_PROVISION_8TH: i64 = 1;

/// Derive the per-band PVQ pulse counts `band_k` from a decoded
/// [`FramePrefix`], using the documented §4.3.3 / §4.3.4.1 allocation
/// arithmetic.
///
/// Runs the §4.3.3 combined column search over the prefix's post-boost
/// budget, then the §4.3.4.1 bits-to-pulses loop (with the balance
/// accumulator threaded across the coded-band window), and returns one
/// pulse count per coded band in `prefix.start..prefix.end` order. Each
/// count is clamped so its PVQ codebook size is representable in the
/// documented single-block decode (see [`clamp_k_to_decodable`]).
///
/// The whole combined allocation is treated as the **shape** budget (no
/// fine reservation); this is the maximal documented approximation given
/// the deferred fine/shape split (see the module docs).
///
/// ## Parameters
///
/// * `prefix` — a decoded [`FramePrefix`] (from
///   [`decode_frame_prefix`]). Its `start` / `end` window, decoded
///   boosts, and trim drive the search.
/// * `lm` — the frame-size shift `log2(frame_size / 120)` ∈ `{0,1,2,3}`.
/// * `channels` — `1` (mono) or `2` (stereo); selects the static-alloc
///   scaling and the `cap[]` table row.
/// * `stereo` — whether the frame is stereo (selects the `cap[]` table
///   row; for mono pass `false`).
///
/// ## Returns
///
/// `Some(band_k)` with `band_k.len() == prefix.end - prefix.start` on
/// success, or `None` when the band window / `lm` / `channels` is out of
/// range, or the column search rejects the inputs (the same
/// input-validation paths [`find_combined_alloc`] documents).
pub fn derive_band_pulses(
    prefix: &FramePrefix,
    lm: u32,
    channels: u32,
    stereo: bool,
) -> Option<Vec<u32>> {
    let start = prefix.start;
    let end = prefix.end;
    if lm > 3 || start > end || end > NUM_BANDS || !(1..=2).contains(&channels) {
        return None;
    }
    if prefix.boosts.boost.len() != end - start {
        return None;
    }

    let bins: Vec<u32> = BAND_BINS_LM[lm as usize][start..end].to_vec();

    // The remaining budget the §4.3.3 search drives toward: the
    // band-boost loop's `total_bits` at exit (in 1/8 bits), the budget
    // the §4.3.3 prose says the downstream allocation steps inherit after
    // every accepted boost has been subtracted — less one provisioned
    // eighth-bit per coded band (one PVQ symbol each; see
    // `PER_SYMBOL_PROVISION_8TH`).
    let budget_1_8th =
        (shape_budget_1_8th(prefix) - PER_SYMBOL_PROVISION_8TH * (end - start) as i64).max(0);

    let search = find_combined_alloc(
        start,
        &bins,
        &prefix.boosts.boost,
        i32::from(prefix.allocation.alloc_trim),
        channels,
        stereo,
        lm,
        budget_1_8th,
    )?;

    // The combined per-band shape allocation (1/8 bits) drives the
    // §4.3.4.1 bits-to-pulses loop. Targets are non-negative by
    // construction (the combine step floors at zero).
    let targets: Vec<u32> = search.alloc.bits.iter().map(|&b| b.max(0) as u32).collect();

    // §4.3.3 hard-minimum skip floor: `thresh[band] = max((24*N)/16,
    // 8*channels)` — a band whose adjusted target falls below it is
    // better served by no allocation at all (see
    // `bits_to_pulses_band_loop_cached_thresh`).
    let mut thresh_i32 = vec![0i32; bins.len()];
    if !compute_thresh(channels, &bins, &mut thresh_i32) {
        return None;
    }
    let thresh: Vec<u32> = thresh_i32.iter().map(|&t| t.max(0) as u32).collect();

    let (per_band, _balance) =
        bits_to_pulses_band_loop_cached_thresh(lm as usize, start, &bins, &targets, Some(&thresh))?;

    Some(
        per_band
            .iter()
            .zip(bins.iter())
            .map(|(r, &n)| clamp_k_to_decodable(n, r.k))
            .collect(),
    )
}

/// Derive the shared per-band pulse counts for a **dual-stereo**
/// frame from a decoded [`FramePrefix`] — the two-channel counterpart
/// of [`derive_band_pulses`].
///
/// Runs the §4.3.3 combined column search with the stereo scaling
/// (`channels = 2`, the stereo `cap[]` row, the stereo reservations
/// already folded into the prefix's budget), then splits each band's
/// combined allocation evenly between the two channels (each channel's
/// shape budget is `bits[band] / 2`, flooring the odd 1/8 bit), and
/// runs the §4.3.4.1 bits-to-pulses loop **once** over the per-channel
/// targets — so both channels share one `K` per band, matching the
/// dual-coded walk
/// ([`decode_stereo_residual_bands`](crate::residual::decode_stereo_residual_bands)),
/// which codes one PVQ index per channel at that `K`.
///
/// The even split is an in-crate decision inside the documented
/// §4.3.3 gap: the RFC specifies the *combined* per-band allocation
/// (static + boost + trim, clamped by `cap[]`) but defers the stereo
/// per-channel division to the reference's `interp_bits2pulses`. An
/// even split is the natural uncoupled-path reading (dual stereo
/// "encodes the left and right channels separately" with no coded
/// imbalance parameter on the uncoupled path), and both sides of the
/// in-crate loop derive it from the bit-identical prefix, so lockstep
/// holds by construction.
///
/// Returns `Some(band_k)` with one shared `K` per coded band, or
/// `None` on the same failure paths as [`derive_band_pulses`].
pub fn derive_band_pulses_dual(prefix: &FramePrefix, lm: u32) -> Option<Vec<u32>> {
    let start = prefix.start;
    let end = prefix.end;
    if lm > 3 || start > end || end > NUM_BANDS {
        return None;
    }
    if prefix.boosts.boost.len() != end - start {
        return None;
    }

    let bins: Vec<u32> = BAND_BINS_LM[lm as usize][start..end].to_vec();
    // Two PVQ symbols per band on the dual wire, so two provisioned
    // eighth-bits per band (see `PER_SYMBOL_PROVISION_8TH`).
    let budget_1_8th =
        (shape_budget_1_8th(prefix) - 2 * PER_SYMBOL_PROVISION_8TH * (end - start) as i64).max(0);

    let search = find_combined_alloc(
        start,
        &bins,
        &prefix.boosts.boost,
        i32::from(prefix.allocation.alloc_trim),
        2,
        true,
        lm,
        budget_1_8th,
    )?;

    // Per-channel share: half the band's combined allocation (the odd
    // 1/8 bit floors away — conservative on both sides).
    let targets: Vec<u32> = search
        .alloc
        .bits
        .iter()
        .map(|&b| (b.max(0) as u32) / 2)
        .collect();

    // §4.3.3 hard-minimum skip floor, halved onto the per-channel
    // axis the dual targets live on (thresh is computed for the
    // 2-channel frame; each channel's share carries half of it).
    let mut thresh_i32 = vec![0i32; bins.len()];
    if !compute_thresh(2, &bins, &mut thresh_i32) {
        return None;
    }
    let thresh: Vec<u32> = thresh_i32.iter().map(|&t| (t.max(0) as u32) / 2).collect();

    let (per_band, _balance) =
        bits_to_pulses_band_loop_cached_thresh(lm as usize, start, &bins, &targets, Some(&thresh))?;

    Some(
        per_band
            .iter()
            .zip(bins.iter())
            .map(|(r, &n)| clamp_k_to_decodable(n, r.k))
            .collect(),
    )
}

/// Decode one mono, non-transient CELT frame to time-domain PCM,
/// deriving the per-band pulse counts from the documented §4.3.3 /
/// §4.3.4.1 allocation arithmetic rather than taking them as a caller
/// input.
///
/// This is [`decode_celt_frame`] with the `band_k` (and `fine_bits`)
/// inputs supplied internally: it decodes the prefix, runs
/// [`derive_band_pulses`] over it, and feeds the result back into the
/// documented synthesis chain with no fine refinement
/// (`fine_bits = 0`). It is the deepest caller-input-free mono decode
/// the wall permits — every step is RFC-specified except the deferred
/// fine/shape split, which is approximated by treating the whole
/// combined allocation as shape (see the module docs).
///
/// ## Parameters
///
/// * `state` — the streaming decoder state (mutated in place exactly as
///   [`decode_celt_frame`] mutates it).
/// * `frame_bytes` — the CELT range-coded payload for this frame.
/// * `start` / `end` — the coded-band window, `start <= end <= 21`.
///
/// ## Returns
///
/// The [`DecodedFrame`] on success, or:
///
/// * [`Error::InvalidParameter`] when the band window or `lm` is out of
///   range, or the documented derivation rejects the inputs.
/// * [`Error::NotImplemented`] when the decoded prefix signals a
///   transient frame (the §4.3.1 / §4.3.7 short-block reassembly gap) or
///   a band's codebook saturates (the §4.3.4.4 split gap), exactly as
///   [`decode_celt_frame`] surfaces them.
pub fn decode_celt_frame_auto(
    state: &mut CeltDecodeState,
    frame_bytes: &[u8],
    start: usize,
    end: usize,
) -> Result<DecodedFrame, Error> {
    if start > end || end > NUM_BANDS {
        return Err(Error::InvalidParameter);
    }
    let lm = state.lm();

    // Decode the prefix on a throwaway coarse-energy state so the real
    // streaming state is mutated exactly once, by decode_celt_frame
    // below (which re-walks the prefix). Deriving band_k only needs the
    // prefix's decoded boosts / trim / window, not the carried coarse
    // prediction, so a fresh coarse state is sufficient here.
    let mut probe_coarse = CoarseEnergyState::new();
    let mut probe_dec = RangeDecoder::new(frame_bytes);
    let prefix = decode_frame_prefix(
        &mut probe_dec,
        &mut probe_coarse,
        lm,
        frame_bytes.len() as u32,
        false,
        start,
        end,
    )?;

    if prefix.header.transient {
        return Err(Error::NotImplemented);
    }

    // A silence-flagged frame (§4.3 Table 56 `silence`) carries no
    // shape symbols: the residual is the zero spectrum, so the
    // synthesis plays out the overlap tail and the output decays to
    // silence. The Table-56 walk (including coarse energy) still ran
    // above, keeping the §4.3.2.1 prediction in encoder/decoder
    // lockstep — the wire contract `encode_celt_frame_auto` writes.
    let band_k = if prefix.header.silence {
        vec![0u32; end - start]
    } else {
        derive_band_pulses(&prefix, lm, 1, false).ok_or(Error::InvalidParameter)?
    };

    let fine_bits = [0u32; NUM_BANDS];
    decode_celt_frame(state, frame_bytes, start, end, &fine_bits, &band_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic, well-formed CELT range-coded payload. The
    /// control-symbol decoders all have defined low-budget fallbacks, so
    /// any byte pattern parses; a mixing recurrence keeps the stream from
    /// being trivially uniform.
    fn payload(len: usize, seed: u8) -> Vec<u8> {
        (0..len as u32)
            .map(|i| (i.wrapping_mul(89).wrapping_add(u32::from(seed)) & 0xff) as u8)
            .collect()
    }

    /// Decode the prefix and report whether the frame is transient.
    fn is_transient(buf: &[u8], lm: u32, start: usize, end: usize) -> bool {
        let mut coarse = CoarseEnergyState::new();
        let mut dec = RangeDecoder::new(buf);
        decode_frame_prefix(
            &mut dec,
            &mut coarse,
            lm,
            buf.len() as u32,
            false,
            start,
            end,
        )
        .expect("prefix decode")
        .header
        .transient
    }

    /// Scan seeds for a non-transient (long-MDCT) payload.
    fn nontransient_payload(len: usize, lm: u32, start: usize, end: usize) -> Vec<u8> {
        for seed in 0..=255u32 {
            let buf = payload(len, seed as u8);
            if !is_transient(&buf, lm, start, end) {
                return buf;
            }
        }
        panic!("no non-transient payload found (len={len})");
    }

    /// Scan seeds for a payload whose derived-pulse decode lands on the
    /// success path (not the documented `NotImplemented` gap): a
    /// non-transient frame whose every coded band's TF / split shape
    /// constraints are satisfiable in the single-block long-MDCT decode.
    /// Some non-transient frames signal a per-band `tf_change` that, with
    /// a single long MDCT (`nb_blocks == 1`), requests Hadamard levels
    /// that do not exist — exactly the §4.3.4.5 / §4.3.4.4 territory the
    /// auto driver correctly surfaces as `NotImplemented`. This helper
    /// finds a frame on the documented success path so the composition
    /// tests exercise it; the gate itself is covered by
    /// [`auto_rejects_transient`] and the §4.3.4.4 split surfacing.
    fn decodable_payload(len: usize, lm: u32, start: usize, end: usize) -> Vec<u8> {
        for seed in 0..=255u32 {
            let buf = payload(len, seed as u8);
            if is_transient(&buf, lm, start, end) {
                continue;
            }
            let mut state = CeltDecodeState::new(lm).expect("state");
            if decode_celt_frame_auto(&mut state, &buf, start, end).is_ok() {
                return buf;
            }
        }
        panic!("no documented-success payload found (len={len})");
    }

    /// Decode just the prefix of `buf` for the derivation tests.
    fn prefix_of(buf: &[u8], lm: u32, start: usize, end: usize) -> FramePrefix {
        let mut coarse = CoarseEnergyState::new();
        let mut dec = RangeDecoder::new(buf);
        decode_frame_prefix(
            &mut dec,
            &mut coarse,
            lm,
            buf.len() as u32,
            false,
            start,
            end,
        )
        .expect("prefix decode")
    }

    /// `derive_band_pulses` returns one pulse count per coded band, each
    /// within the documented decodable-codebook cap.
    #[test]
    fn derives_one_k_per_band() {
        let lm = 3u32;
        let buf = nontransient_payload(48, lm, 0, NUM_BANDS);
        let pfx = prefix_of(&buf, lm, 0, NUM_BANDS);

        let band_k = derive_band_pulses(&pfx, lm, 1, false).expect("derive");
        assert_eq!(band_k.len(), NUM_BANDS, "one K per coded band");

        // Every derived K leaves V(N, K) representable (no saturation),
        // i.e. clamp_k_to_decodable held it inside the single-block decode.
        for (b, &k) in band_k.iter().enumerate() {
            let n = BAND_BINS_LM[lm as usize][b];
            assert_ne!(
                v_count(n, k),
                V_COUNT_SATURATION,
                "band {b} K={k} (N={n}) overflows the codebook"
            );
        }
    }

    /// The derivation is deterministic: identical bytes yield identical
    /// pulse counts.
    #[test]
    fn derivation_is_deterministic() {
        let lm = 2u32;
        let buf = nontransient_payload(40, lm, 0, NUM_BANDS);
        let pfx = prefix_of(&buf, lm, 0, NUM_BANDS);

        let a = derive_band_pulses(&pfx, lm, 1, false).expect("derive a");
        let b = derive_band_pulses(&pfx, lm, 1, false).expect("derive b");
        assert_eq!(a, b, "derivation is non-deterministic");
    }

    /// A Hybrid-mode window (bands 17..=20) derives a 4-band pulse vector
    /// indexed from the window origin.
    #[test]
    fn derives_hybrid_window() {
        let lm = 2u32;
        let buf = nontransient_payload(32, lm, 17, NUM_BANDS);
        let pfx = prefix_of(&buf, lm, 17, NUM_BANDS);
        assert_eq!(pfx.start, 17);
        assert_eq!(pfx.end, NUM_BANDS);

        let band_k = derive_band_pulses(&pfx, lm, 1, false).expect("derive");
        assert_eq!(band_k.len(), NUM_BANDS - 17);
    }

    /// `decode_celt_frame_auto` composes the whole chain to finite PCM
    /// with no caller-supplied pulse counts.
    #[test]
    fn auto_decode_composes_to_finite_pcm() {
        let lm = 3u32;
        let buf = decodable_payload(48, lm, 0, NUM_BANDS);

        let mut state = CeltDecodeState::new(lm).expect("state");
        let frame = decode_celt_frame_auto(&mut state, &buf, 0, NUM_BANDS).expect("auto decode");

        assert_eq!(frame.pcm.len(), state.frame_size());
        for (i, &s) in frame.pcm.iter().enumerate() {
            assert!(s.is_finite(), "non-finite PCM sample at index {i}");
        }
    }

    /// `decode_celt_frame_auto` is deterministic across two fresh states.
    #[test]
    fn auto_decode_is_deterministic() {
        let lm = 2u32;
        let buf = decodable_payload(40, lm, 0, NUM_BANDS);

        let mut s_a = CeltDecodeState::new(lm).expect("state a");
        let mut s_b = CeltDecodeState::new(lm).expect("state b");
        let f_a = decode_celt_frame_auto(&mut s_a, &buf, 0, NUM_BANDS).unwrap();
        let f_b = decode_celt_frame_auto(&mut s_b, &buf, 0, NUM_BANDS).unwrap();
        assert_eq!(f_a.pcm, f_b.pcm, "auto decode is non-deterministic");
    }

    /// `decode_celt_frame_auto` matches the manual compose:
    /// `decode_frame_prefix` → `derive_band_pulses` → `decode_celt_frame`
    /// with `fine_bits = 0` produces the same PCM the auto driver does.
    #[test]
    fn auto_matches_manual_compose() {
        let lm = 3u32;
        let buf = decodable_payload(48, lm, 0, NUM_BANDS);

        // Manual compose.
        let pfx = prefix_of(&buf, lm, 0, NUM_BANDS);
        let band_k = derive_band_pulses(&pfx, lm, 1, false).expect("derive");
        let mut s_manual = CeltDecodeState::new(lm).expect("state");
        let fine_bits = [0u32; NUM_BANDS];
        let manual =
            decode_celt_frame(&mut s_manual, &buf, 0, NUM_BANDS, &fine_bits, &band_k).unwrap();

        // Auto compose.
        let mut s_auto = CeltDecodeState::new(lm).expect("state");
        let auto = decode_celt_frame_auto(&mut s_auto, &buf, 0, NUM_BANDS).unwrap();

        assert_eq!(auto.pcm, manual.pcm, "auto / manual PCM diverge");
    }

    /// A transient frame is rejected with `NotImplemented`, not silently
    /// mis-decoded (the §4.3.1 / §4.3.7 short-block reassembly gap).
    #[test]
    fn auto_rejects_transient() {
        // Scan for a transient seed at lm=1 (the {7,1}/8 "1" branch).
        let lm = 1u32;
        let mut saw_transient = false;
        for seed in 0..=255u32 {
            let buf = payload(24, seed as u8);
            if !is_transient(&buf, lm, 0, NUM_BANDS) {
                continue;
            }
            saw_transient = true;
            let mut state = CeltDecodeState::new(lm).expect("state");
            let got = decode_celt_frame_auto(&mut state, &buf, 0, NUM_BANDS);
            assert!(
                matches!(got, Err(Error::NotImplemented)),
                "transient frame (seed {seed}) not rejected: {got:?}"
            );
        }
        assert!(
            saw_transient,
            "no transient seed found to exercise the gate"
        );
    }

    /// Decode a stereo prefix of `buf` for the dual-derivation tests.
    fn stereo_prefix_of(buf: &[u8], lm: u32, start: usize, end: usize) -> FramePrefix {
        let mut coarse = CoarseEnergyState::new();
        let mut dec = RangeDecoder::new(buf);
        decode_frame_prefix(
            &mut dec,
            &mut coarse,
            lm,
            buf.len() as u32,
            true,
            start,
            end,
        )
        .expect("stereo prefix decode")
    }

    /// `derive_band_pulses_dual` returns one shared K per coded band,
    /// each within the decodable-codebook cap, deterministically.
    #[test]
    fn dual_derivation_one_shared_k_per_band() {
        let lm = 2u32;
        // Any seed works for the derivation (it only reads the decoded
        // prefix); use a couple to cover different boost/trim mixes.
        for seed in [3u8, 57, 190] {
            let buf = payload(72, seed);
            let pfx = stereo_prefix_of(&buf, lm, 0, NUM_BANDS);
            let a = derive_band_pulses_dual(&pfx, lm).expect("derive dual");
            assert_eq!(a.len(), NUM_BANDS);
            for (b, &k) in a.iter().enumerate() {
                let n = BAND_BINS_LM[lm as usize][b];
                assert_ne!(
                    v_count(n, k),
                    V_COUNT_SATURATION,
                    "band {b} K={k} overflows the codebook"
                );
            }
            let b = derive_band_pulses_dual(&pfx, lm).expect("derive dual again");
            assert_eq!(a, b, "dual derivation is non-deterministic");
        }
    }

    /// The per-channel share is half the combined stereo allocation:
    /// the dual K never exceeds the K a single channel would get from
    /// the *whole* stereo-scaled allocation.
    #[test]
    fn dual_k_never_exceeds_full_stereo_alloc_k() {
        let lm = 3u32;
        let buf = payload(96, 41);
        let pfx = stereo_prefix_of(&buf, lm, 0, NUM_BANDS);
        let dual = derive_band_pulses_dual(&pfx, lm).expect("dual");
        // Full stereo-scaled allocation driven through the same loop
        // (what a channel would get if it kept every 1/8 bit).
        let full = derive_band_pulses(&pfx, lm, 2, true).expect("full");
        for (b, (&d, &f)) in dual.iter().zip(&full).enumerate() {
            assert!(d <= f, "band {b}: dual K {d} > full-alloc K {f}");
        }
    }

    /// Bad windows / mismatched boosts propagate as `None`.
    #[test]
    fn dual_derivation_rejects_bad_inputs() {
        let lm = 2u32;
        let buf = payload(64, 9);
        let mut pfx = stereo_prefix_of(&buf, lm, 0, NUM_BANDS);
        assert!(derive_band_pulses_dual(&pfx, 4).is_none());
        pfx.boosts.boost.pop();
        assert!(derive_band_pulses_dual(&pfx, lm).is_none());
    }

    /// An out-of-range band window is rejected without panicking.
    #[test]
    fn auto_rejects_bad_window() {
        let lm = 3u32;
        let buf = payload(32, 5);
        let mut state = CeltDecodeState::new(lm).expect("state");
        assert!(matches!(
            decode_celt_frame_auto(&mut state, &buf, 0, 22),
            Err(Error::InvalidParameter)
        ));
        assert!(matches!(
            decode_celt_frame_auto(&mut state, &buf, 10, 5),
            Err(Error::InvalidParameter)
        ));
    }
}
