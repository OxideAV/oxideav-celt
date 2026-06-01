//! Initial bit-allocation budget walk (RFC 6716 §4.3.3).
//!
//! ## What this module covers
//!
//! The §4.3.3 bit allocator opens with a deterministic budget walk
//! that initializes the running 1/8-bit `total`, then takes four
//! reservations in fixed order:
//!
//! 1. **`anti_collapse_rsv`** — 8 1/8-bits reserved iff the frame
//!    is transient, `LM > 1`, and `total >= (LM+2)*8`.
//! 2. **`skip_rsv`** — 8 1/8-bits reserved iff `total > 8` after the
//!    anti-collapse step.
//! 3. **`intensity_rsv`** — conservative `log2(coded_bands)` in
//!    1/8 bits, via [`crate::bit_allocation::LOG2_FRAC_TABLE`]; only
//!    fires for stereo frames and is dropped if it would exceed the
//!    running budget.
//! 4. **`dual_stereo_rsv`** — 8 1/8-bits reserved iff stereo and the
//!    running budget after the intensity step is greater than 8.
//!
//! The four reservations together cap the budget the band-boost loop
//! (already exposed as [`crate::band_cap::decode_band_boosts`]) and
//! the band-allocation orchestrator
//! ([`crate::bit_allocation::decode_band_allocation`]) consume.
//!
//! All numbers in this module are in **1/8 bit units** because the
//! §4.3.3 prose runs the budget arithmetic at 1/8-bit precision
//! (`ec_tell_frac()`, [`LOG2_FRAC_TABLE`](crate::bit_allocation::LOG2_FRAC_TABLE),
//! and the binary reservations are all 1/8-bit quanta).
//!
//! ## What this module does NOT cover
//!
//! * The band-boost loop itself ([`crate::band_cap::decode_band_boosts`]
//!   is a separate entry point, queued in Table 56 order between this
//!   initial walk and the alloc-trim decode). The boost loop consumes
//!   the `total` field this module emits.
//! * Decoding `alloc.trim`, `skip`, `intensity`, and `dual_stereo`
//!   (already in [`crate::bit_allocation::decode_band_allocation`]).
//!   This module emits a [`BandAllocationGates`](crate::bit_allocation::BandAllocationGates)
//!   ready to feed that orchestrator once the boost loop has run.
//! * The Table-57 static-allocation search, the per-band minimum
//!   shape allocation, the trim-offset computation, the reallocation
//!   loop, and the fine-energy / shape split. Those come after the
//!   band-allocation orchestrator and are queued for later rounds.
//!
//! ## Clean-room provenance
//!
//! Every step is transcribed from the clean-room narrative at
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.5
//! (which paraphrases RFC 6716 §4.3.3) and from RFC 6716 §4.3.3
//! itself at `docs/audio/opus/rfc6716-opus.txt`. No external library
//! source was consulted.

use crate::bit_allocation::{intensity_rsv, BandAllocationGates};

/// One quantum of 1/8-bit budget for the binary reservations
/// (`anti_collapse_rsv`, `skip_rsv`, `dual_stereo_rsv`).
///
/// RFC 6716 §4.3.3 expresses each of the three binary reservations as
/// "1 bit" but runs the budget walk in 1/8-bit units, so each binary
/// reservation contributes 8 1/8-bits when it fires.
pub const RSV_BIT_8TH: i32 = 8;

/// The §4.3.3 conservative slack subtracted from the raw frame size
/// at the start of the budget walk to keep the allocator from
/// landing one 1/8-bit past the end of the range-coded segment.
///
/// Per the clean-room narrative §2.5 step 1: `total = (coded frame
/// size × 8) − ec_tell_frac()`, then **minus 1** (1/8 bit). That
/// trailing `-1` is this constant.
pub const RSV_INITIAL_SLACK_8TH: i32 = 1;

/// Result of the §4.3.3 initial budget walk.
///
/// Holds the running `total` (in 1/8 bits) at the point where the
/// caller should start the band-boost loop, alongside the four
/// reservations the walk took. The reservation fields are exposed in
/// the order the §4.3.3 prose takes them so a caller doing its own
/// step-by-step accounting can cross-check at each substep.
///
/// `total_initial` records the §4.3.3 "starting budget" before any
/// of the four reservations were subtracted (useful for the
/// `BandAllocationGates::trim_gated` test below, which references
/// the original frame-size budget less the running `total_boost`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitialReservations {
    /// Initial 1/8-bit budget before any reservation, computed as
    /// `frame_bytes * 64 - ec_tell_frac - 1`. The `* 64` factor
    /// expresses the frame size in 1/8-bit units (`frame_bytes * 8`
    /// converts bytes to bits; another `* 8` converts whole bits to
    /// 1/8 bit). The trailing `- 1` is [`RSV_INITIAL_SLACK_8TH`] per
    /// the §2.5 narrative.
    pub total_initial: i32,
    /// Anti-collapse reservation in 1/8 bits: [`RSV_BIT_8TH`] (= 8)
    /// when the §4.3.3 / §2.5 step 2 conditions fire (transient frame,
    /// `LM > 1`, and `total >= (LM+2)*8`), zero otherwise.
    pub anti_collapse_rsv: i32,
    /// Skip reservation in 1/8 bits: [`RSV_BIT_8TH`] (= 8) iff
    /// `total > 8` after the anti-collapse subtraction. The §4.3.3
    /// skip-bit covers the high-frequency band-skipping logic at
    /// the tail of the allocation pass.
    pub skip_rsv: i32,
    /// Intensity-stereo reservation in 1/8 bits. For stereo frames
    /// this is [`LOG2_FRAC_TABLE`](crate::bit_allocation::LOG2_FRAC_TABLE)
    /// looked up by `coded_bands` (clamped to zero outside the legal
    /// table range and zero when the reservation would exceed the
    /// running budget). For mono frames this is always zero.
    pub intensity_rsv: i32,
    /// Dual-stereo reservation in 1/8 bits: [`RSV_BIT_8TH`] (= 8) iff
    /// stereo AND the running budget after the intensity step is
    /// greater than 8. Mono frames always emit zero here.
    pub dual_stereo_rsv: i32,
    /// Running 1/8-bit budget after all four reservations have been
    /// taken. This is what the band-boost loop's `total_bits`
    /// argument should be initialized to — see
    /// [`crate::band_cap::decode_band_boosts`].
    pub total: i32,
    /// Number of coded bands (`end - start`) that this walk
    /// considered. Threaded through so the caller can pass it
    /// straight to [`BandAllocationGates`].
    pub coded_bands: u32,
    /// Whether the frame is stereo. Threaded through so the caller
    /// can pass it to follow-on decoders.
    pub stereo: bool,
}

impl InitialReservations {
    /// Sum of all four reservations in 1/8 bits.
    ///
    /// Equal to `total_initial - total` — a useful identity for
    /// cross-checking the running budget against the original
    /// frame-size budget.
    pub fn total_reserved(&self) -> i32 {
        self.anti_collapse_rsv + self.skip_rsv + self.intensity_rsv + self.dual_stereo_rsv
    }

    /// Build the [`BandAllocationGates`] for the §4.3.3 band-
    /// allocation orchestrator, given the `total_boost` accumulated
    /// by the band-boost loop.
    ///
    /// Per §4.3.3:
    ///
    /// * `trim_gated = ec_tell_frac() + 48 <= total_initial - total_boost`
    ///   (the trim-PDF gate; 48 1/8-bits = 6 whole bits of headroom
    ///   for the Table 58 ICDF walk).
    /// * `skip_gated = skip_rsv == 8` (the skip-bit was reserved).
    /// * `intensity_gated = stereo AND intensity_rsv > 0`.
    /// * `dual_gated = stereo AND dual_stereo_rsv == 8`.
    ///
    /// `ec_tell_frac_now` is the caller's `ec_tell_frac()` value at
    /// the moment the trim is about to be decoded (i.e. after the
    /// band-boost loop has run). `total_boost` is the 1/8-bit
    /// accumulator emitted by the band-boost loop.
    pub fn gates_for_band_allocation(
        &self,
        ec_tell_frac_now: u32,
        total_boost: i32,
    ) -> BandAllocationGates {
        let trim_lhs = ec_tell_frac_now as i32 + 48;
        let trim_rhs = self.total_initial - total_boost;
        let trim_gated = trim_lhs <= trim_rhs;
        BandAllocationGates {
            trim_gated,
            skip_gated: self.skip_rsv == RSV_BIT_8TH,
            intensity_gated: self.stereo && self.intensity_rsv > 0,
            dual_gated: self.stereo && self.dual_stereo_rsv == RSV_BIT_8TH,
            coded_bands: self.coded_bands,
        }
    }
}

/// Run the §4.3.3 initial budget walk and emit the four reservations
/// plus the running `total` budget the band-boost loop should start
/// from.
///
/// Inputs:
///
/// * `frame_bytes` — coded frame size in BYTES. The walk multiplies
///   by 8 (to bits) and again by 8 (to 1/8 bits = `* 64`) to obtain
///   the §4.3.3 budget. Pass the size of the CELT range-coded
///   segment for this frame (i.e. the byte count the
///   [`RangeDecoder`](crate::range_decoder::RangeDecoder) was
///   initialized with).
/// * `ec_tell_frac` — current `ec_tell_frac()` value in 1/8 bits,
///   taken AFTER the §4.3 frame-header prefix + §4.3.2.1 coarse
///   energy + §4.3.2.2 fine-energy refinement have been decoded.
///   This is the §4.3.3 prose's "1/8 bits decoded so far".
/// * `is_transient` — `true` when the §4.3.1 transient flag was set
///   on this frame. Drives the anti-collapse reservation gate.
/// * `lm` — frame-size shift `log2(frame_size / 120)` for this
///   stream's mode. Legal values 0..=3 corresponding to 2.5 / 5 / 10
///   / 20 ms frames. Drives the anti-collapse gate (`LM > 1`).
/// * `stereo` — `true` for two-channel CELT, `false` for mono.
///   Drives the intensity and dual reservations.
/// * `coded_bands` — `end - start`, the number of CELT bands this
///   frame codes. Indexes [`LOG2_FRAC_TABLE`](crate::bit_allocation::LOG2_FRAC_TABLE)
///   for the intensity reservation.
///
/// The walk never reads the range decoder — it is purely arithmetic
/// over the inputs. The §4.3.3 reservations only fire if their gates
/// hold, and the resulting [`InitialReservations::total`] is clamped
/// to `>= 0` after the anti-collapse step per the §2.5 narrative.
///
/// Defensive corners: `ec_tell_frac` larger than `frame_bytes * 64`
/// yields a negative `total_initial` and zero reservations; an out-
/// of-range `lm` (`> 3`) is treated as the maximum supported value
/// (3 = 20 ms) for the anti-collapse gate, but the §4.3.3 prose only
/// ever passes `lm ∈ 0..=3` in practice.
pub fn compute_initial_reservations(
    frame_bytes: u32,
    ec_tell_frac: u32,
    is_transient: bool,
    lm: u32,
    stereo: bool,
    coded_bands: u32,
) -> InitialReservations {
    // §2.5 step 1: total = frame_bytes * 8 (bits) * 8 (1/8 bits)
    //                    - ec_tell_frac
    //                    - RSV_INITIAL_SLACK_8TH.
    // Use i64 for the multiply so a malicious or pathological
    // `frame_bytes` near u32::MAX doesn't wrap; the §4.3.3 budget
    // fits comfortably in i32 once the frame size is realistic.
    let frame_8th = (frame_bytes as i64) * 64;
    let total_initial_i64 = frame_8th - ec_tell_frac as i64 - RSV_INITIAL_SLACK_8TH as i64;
    let total_initial = i32::try_from(total_initial_i64).unwrap_or(i32::MAX);

    let mut total = total_initial;

    // §2.5 step 2: anti_collapse_rsv = 8 iff transient, LM > 1, and
    // total >= (LM+2)*8 1/8 bits. Then clamp total >= 0.
    let lm_eff = lm.min(3); // defensive clamp to the legal CELT range.
    let anti_collapse_threshold = ((lm_eff as i32).saturating_add(2)).saturating_mul(RSV_BIT_8TH);
    let anti_collapse_rsv = if is_transient && lm_eff > 1 && total >= anti_collapse_threshold {
        RSV_BIT_8TH
    } else {
        0
    };
    total -= anti_collapse_rsv;
    if total < 0 {
        total = 0;
    }

    // §2.5 step 3: skip_rsv = 8 if total > 8, else 0. Decrement.
    let skip_rsv = if total > RSV_BIT_8TH { RSV_BIT_8TH } else { 0 };
    total -= skip_rsv;

    // §2.5 step 4: stereo intensity + dual reservations.
    // `intensity_rsv(coded_bands, stereo, total)` returns the
    // LOG2_FRAC_TABLE lookup gated by the budget; zero for mono /
    // empty band ranges / out-of-table values / over-budget cases.
    let intensity_rsv_8th = intensity_rsv(coded_bands, stereo, total) as i32;
    total -= intensity_rsv_8th;

    let dual_stereo_rsv = if stereo && total > RSV_BIT_8TH {
        RSV_BIT_8TH
    } else {
        0
    };
    total -= dual_stereo_rsv;

    InitialReservations {
        total_initial,
        anti_collapse_rsv,
        skip_rsv,
        intensity_rsv: intensity_rsv_8th,
        dual_stereo_rsv,
        total,
        coded_bands,
        stereo,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RSV_BIT_8TH = 8` (1 bit = 8 1/8-bits) per §4.3.3 prose.
    #[test]
    fn rsv_bit_8th_is_one_bit() {
        assert_eq!(RSV_BIT_8TH, 8);
    }

    /// `RSV_INITIAL_SLACK_8TH = 1` (1/8 bit) per §2.5 narrative step 1.
    #[test]
    fn rsv_initial_slack_is_one_8th() {
        assert_eq!(RSV_INITIAL_SLACK_8TH, 1);
    }

    /// A high-rate stereo 20 ms transient frame should fire every
    /// reservation: anti-collapse (LM=3 > 1, transient, budget
    /// covers), skip (budget covers), intensity (stereo + budget),
    /// dual (stereo + budget).
    #[test]
    fn high_rate_stereo_transient_fires_every_reservation() {
        // 256-byte frame (e.g. ~100 kbps at 20 ms) — plenty of budget.
        // ec_tell_frac = 100 (~12.5 bits decoded so far).
        let r = compute_initial_reservations(
            256,  // frame_bytes
            100,  // ec_tell_frac
            true, // is_transient
            3,    // lm = 20 ms
            true, // stereo
            21,   // coded_bands = full CELT
        );
        // total_initial = 256 * 64 - 100 - 1 = 16384 - 101 = 16283.
        assert_eq!(r.total_initial, 16283);
        assert_eq!(r.anti_collapse_rsv, 8);
        assert_eq!(r.skip_rsv, 8);
        // intensity = LOG2_FRAC_TABLE[21] = 36.
        assert_eq!(r.intensity_rsv, 36);
        assert_eq!(r.dual_stereo_rsv, 8);
        // Running total after every reservation.
        assert_eq!(r.total, 16283 - 8 - 8 - 36 - 8);
        // Identity: total = total_initial - sum of reservations.
        assert_eq!(r.total, r.total_initial - r.total_reserved());
    }

    /// Mono 20 ms transient: anti-collapse + skip fire; intensity +
    /// dual stay zero.
    #[test]
    fn mono_transient_no_stereo_reservations() {
        let r = compute_initial_reservations(64, 50, true, 3, false, 21);
        // total_initial = 64*64 - 50 - 1 = 4096 - 51 = 4045.
        assert_eq!(r.total_initial, 4045);
        assert_eq!(r.anti_collapse_rsv, 8);
        assert_eq!(r.skip_rsv, 8);
        assert_eq!(r.intensity_rsv, 0);
        assert_eq!(r.dual_stereo_rsv, 0);
        assert_eq!(r.total, 4045 - 16);
    }

    /// Non-transient frame: anti-collapse gate fails regardless of
    /// LM or budget. The skip, intensity, dual reservations still
    /// proceed.
    #[test]
    fn non_transient_skips_anti_collapse() {
        let r = compute_initial_reservations(128, 100, false, 3, true, 21);
        // total_initial = 128 * 64 - 100 - 1 = 8192 - 101 = 8091.
        assert_eq!(r.total_initial, 8091);
        // is_transient=false ⇒ anti_collapse_rsv = 0.
        assert_eq!(r.anti_collapse_rsv, 0);
        // Skip + intensity + dual still fire on a generous budget.
        assert_eq!(r.skip_rsv, 8);
        assert_eq!(r.intensity_rsv, 36);
        assert_eq!(r.dual_stereo_rsv, 8);
    }

    /// LM ∈ {0, 1} blocks the anti-collapse gate even on a transient.
    #[test]
    fn small_lm_blocks_anti_collapse() {
        for lm in [0u32, 1] {
            let r = compute_initial_reservations(128, 50, true, lm, true, 21);
            assert_eq!(
                r.anti_collapse_rsv, 0,
                "anti-collapse fired at LM={lm} (should require LM>1)"
            );
        }
        // LM = 2 should fire (transient + LM > 1 + budget covers).
        let r2 = compute_initial_reservations(128, 50, true, 2, true, 21);
        assert_eq!(r2.anti_collapse_rsv, 8);
        // LM = 3 should fire.
        let r3 = compute_initial_reservations(128, 50, true, 3, true, 21);
        assert_eq!(r3.anti_collapse_rsv, 8);
    }

    /// Anti-collapse budget gate: `total >= (LM+2)*8` must hold.
    /// At LM=2 the threshold is (2+2)*8 = 32 1/8-bits.
    #[test]
    fn anti_collapse_budget_gate() {
        // Choose ec_tell_frac so that total_initial sits at the
        // §2.5 threshold exactly. (LM+2)*8 at LM=2 is 32.
        // total_initial = frame*64 - ec_tell - 1. Pick frame = 1 byte
        // and ec_tell so total_initial = 32. 1*64 - 31 - 1 = 32.
        let r_eq = compute_initial_reservations(1, 31, true, 2, false, 21);
        assert_eq!(r_eq.total_initial, 32);
        assert_eq!(r_eq.anti_collapse_rsv, 8);
        // total_initial = 31 (one 1/8-bit below the threshold) ⇒ no
        // reservation. 1*64 - 32 - 1 = 31.
        let r_lt = compute_initial_reservations(1, 32, true, 2, false, 21);
        assert_eq!(r_lt.total_initial, 31);
        assert_eq!(r_lt.anti_collapse_rsv, 0);
    }

    /// LM clamp: passing LM > 3 saturates to LM=3 instead of panicking.
    #[test]
    fn lm_out_of_range_clamps_to_three() {
        let r_max = compute_initial_reservations(128, 50, true, 100, true, 21);
        let r_three = compute_initial_reservations(128, 50, true, 3, true, 21);
        assert_eq!(r_max, r_three);
    }

    /// Skip reservation requires `total > 8` after anti-collapse.
    /// At the boundary `total == 8` it does NOT fire.
    #[test]
    fn skip_reservation_boundary() {
        // Pick ec_tell so total_initial = 8 exactly (well below any
        // anti-collapse threshold so anti-collapse won't fire and
        // perturb the boundary). frame_bytes = 1, transient = false
        // ⇒ anti-collapse skipped; 1*64 - 55 - 1 = 8.
        let r_eq = compute_initial_reservations(1, 55, false, 0, false, 21);
        assert_eq!(r_eq.total_initial, 8);
        assert_eq!(r_eq.skip_rsv, 0); // 8 is NOT > 8.
                                      // 1*64 - 54 - 1 = 9 → skip fires.
        let r_gt = compute_initial_reservations(1, 54, false, 0, false, 21);
        assert_eq!(r_gt.total_initial, 9);
        assert_eq!(r_gt.skip_rsv, 8);
    }

    /// Very low rate: total_initial fits in `<= 8`, so neither skip
    /// nor intensity/dual fire. Anti-collapse can still fire iff its
    /// own gate holds.
    #[test]
    fn very_low_rate_drops_skip_intensity_dual() {
        // total_initial = 5: no skip, no intensity (mono), no dual.
        let r = compute_initial_reservations(1, 58, false, 0, false, 21);
        assert_eq!(r.total_initial, 5);
        assert_eq!(r.skip_rsv, 0);
        assert_eq!(r.intensity_rsv, 0);
        assert_eq!(r.dual_stereo_rsv, 0);
        assert_eq!(r.total, 5);
    }

    /// Intensity reservation drops when budget cannot cover it.
    /// At 21 bands, LOG2_FRAC_TABLE[21] = 36 — pick a budget tight
    /// enough that anti-collapse + skip leave less than 36.
    #[test]
    fn intensity_reservation_budget_drop() {
        // Pick frame so that total_initial after anti-collapse + skip
        // is exactly 35 (one 1/8-bit short of LOG2_FRAC_TABLE[21]).
        // No transient ⇒ no anti-collapse. Skip needs total > 8, so
        // it fires for any reasonable budget. total = total_initial - 8.
        // Need total_initial - 8 = 35, so total_initial = 43.
        // 1*64 - 20 - 1 = 43.
        let r = compute_initial_reservations(1, 20, false, 0, true, 21);
        assert_eq!(r.total_initial, 43);
        assert_eq!(r.anti_collapse_rsv, 0);
        assert_eq!(r.skip_rsv, 8);
        // intensity at 21 bands wants 36, budget 35 ⇒ drop.
        assert_eq!(r.intensity_rsv, 0);
        // Dual still fires: 35 > 8.
        assert_eq!(r.dual_stereo_rsv, 8);
        assert_eq!(r.total, 35 - 8);
    }

    /// Dual stereo gate requires `total > 8` after intensity. At
    /// equality (total == 8) it does NOT fire.
    #[test]
    fn dual_stereo_reservation_boundary() {
        // After anti-collapse + skip + intensity, want total = 8.
        // No transient ⇒ no anti-collapse. Skip fires for total > 8.
        // Choose 4 coded_bands ⇒ intensity = LOG2_FRAC_TABLE[4] = 19.
        // total_after_intensity = total_initial - 8 - 19 = 8.
        // total_initial = 35. 1*64 - 28 - 1 = 35.
        let r_eq = compute_initial_reservations(1, 28, false, 0, true, 4);
        assert_eq!(r_eq.total_initial, 35);
        assert_eq!(r_eq.skip_rsv, 8);
        assert_eq!(r_eq.intensity_rsv, 19);
        assert_eq!(r_eq.dual_stereo_rsv, 0); // 8 is NOT > 8.
        assert_eq!(r_eq.total, 8);
        // Add 1 → dual fires. total_initial = 36, total_after = 9 > 8.
        let r_gt = compute_initial_reservations(1, 27, false, 0, true, 4);
        assert_eq!(r_gt.total_initial, 36);
        assert_eq!(r_gt.dual_stereo_rsv, 8);
        assert_eq!(r_gt.total, 1);
    }

    /// Total identity: total = total_initial - sum of reservations
    /// across an ensemble of (frame_bytes, ec_tell, transient, lm,
    /// stereo, coded_bands) inputs.
    #[test]
    fn total_identity_holds_across_grid() {
        for &frame in &[16u32, 32, 64, 128, 256] {
            for &tell in &[0u32, 50, 200, 1000] {
                for &is_t in &[false, true] {
                    for lm in 0u32..=3 {
                        for &st in &[false, true] {
                            for &cb in &[1u32, 4, 17, 21] {
                                let r = compute_initial_reservations(frame, tell, is_t, lm, st, cb);
                                // After the anti-collapse clamp the
                                // identity holds only when no clamp
                                // fired (total >= 0 after anti-collapse
                                // subtraction). Skip cases where the
                                // clamp would have engaged.
                                let post_ac = r.total_initial - r.anti_collapse_rsv;
                                if post_ac < 0 {
                                    continue;
                                }
                                assert_eq!(
                                    r.total,
                                    r.total_initial - r.total_reserved(),
                                    "identity broken at frame={frame} tell={tell} \
                                     transient={is_t} lm={lm} stereo={st} cb={cb}: r={r:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// `ec_tell_frac` larger than the raw frame budget yields a
    /// negative `total_initial` and zero reservations — every gate
    /// fails defensively.
    #[test]
    fn over_consumed_decoder_is_defensive() {
        // ec_tell_frac larger than frame_bytes*64 ⇒ total_initial < 0.
        let r = compute_initial_reservations(1, 1000, true, 3, true, 21);
        assert!(r.total_initial < 0);
        // Anti-collapse threshold at LM=3 is 40; total < 40 ⇒ no rsv.
        assert_eq!(r.anti_collapse_rsv, 0);
        // total after anti-collapse subtraction is < 0 ⇒ clamped to 0;
        // skip needs total > 8 ⇒ no skip.
        assert_eq!(r.skip_rsv, 0);
        // intensity_rsv with negative budget returns 0.
        assert_eq!(r.intensity_rsv, 0);
        // Dual needs total > 8 after intensity ⇒ no.
        assert_eq!(r.dual_stereo_rsv, 0);
        assert_eq!(r.total, 0);
    }

    /// `gates_for_band_allocation`: with `total_boost = 0` and an
    /// `ec_tell_frac_now` well below the budget, the trim gate fires.
    #[test]
    fn gates_trim_fires_at_low_tell() {
        let r = compute_initial_reservations(256, 100, true, 3, true, 21);
        let gates = r.gates_for_band_allocation(100, 0);
        // total_initial = 16283; 100 + 48 = 148 <= 16283 ⇒ trim_gated.
        assert!(gates.trim_gated);
        assert!(gates.skip_gated);
        assert!(gates.intensity_gated);
        assert!(gates.dual_gated);
        assert_eq!(gates.coded_bands, 21);
    }

    /// `gates_for_band_allocation`: a huge `total_boost` can push the
    /// trim gate's RHS below `ec_tell_frac + 48`, gating it off.
    #[test]
    fn gates_trim_disabled_by_large_total_boost() {
        let r = compute_initial_reservations(256, 100, true, 3, true, 21);
        // total_initial = 16283. Need ec_tell_frac + 48 > 16283 -
        // total_boost. ec_tell_frac = 100, so total_boost must exceed
        // 16283 - 148 = 16135.
        let gates = r.gates_for_band_allocation(100, 16200);
        assert!(!gates.trim_gated);
        // Skip / intensity / dual still gated by the reservations
        // taken during the budget walk, not by total_boost.
        assert!(gates.skip_gated);
        assert!(gates.intensity_gated);
        assert!(gates.dual_gated);
    }

    /// `gates_for_band_allocation`: mono frame ⇒ intensity and dual
    /// gates are always off, regardless of `total_boost`.
    #[test]
    fn gates_mono_stereo_gates_off() {
        let r = compute_initial_reservations(256, 100, false, 3, false, 21);
        let gates = r.gates_for_band_allocation(100, 0);
        assert!(gates.trim_gated);
        assert!(gates.skip_gated);
        assert!(!gates.intensity_gated);
        assert!(!gates.dual_gated);
    }

    /// `gates_for_band_allocation`: an intensity reservation that
    /// was dropped during the walk (budget too tight) gates the
    /// intensity decode off even on stereo.
    #[test]
    fn gates_intensity_off_when_walk_dropped_it() {
        // Same setup as `intensity_reservation_budget_drop` above:
        // budget after skip is 35, intensity at 21 bands wants 36,
        // so the walk drops intensity to 0.
        let r = compute_initial_reservations(1, 20, false, 0, true, 21);
        assert_eq!(r.intensity_rsv, 0);
        let gates = r.gates_for_band_allocation(50, 0);
        assert!(!gates.intensity_gated);
    }

    /// `gates_for_band_allocation`: a dual reservation that was
    /// dropped (boundary case) gates dual off even on stereo.
    #[test]
    fn gates_dual_off_when_walk_dropped_it() {
        // Same setup as `dual_stereo_reservation_boundary` above:
        // budget after intensity is exactly 8 ⇒ no dual.
        let r = compute_initial_reservations(1, 28, false, 0, true, 4);
        assert_eq!(r.dual_stereo_rsv, 0);
        let gates = r.gates_for_band_allocation(50, 0);
        assert!(!gates.dual_gated);
    }

    /// Composition with [`crate::range_decoder::RangeDecoder`]: after
    /// the §4.3 prefix decoder has run, the reservations walk uses
    /// the decoder's `tell_frac()` correctly. This is a smoke test
    /// over the public API surface, not a bit-exact test.
    #[test]
    fn smoke_composes_with_range_decoder_tell_frac() {
        use crate::range_decoder::RangeDecoder;
        let buf = [0x55u8; 64];
        let mut dec = RangeDecoder::new(&buf);
        // Pretend the §4.3 prefix decoder has consumed some bits by
        // reading a single symbol.
        dec.dec_bit_logp(2);
        let tell = dec.tell_frac();
        let r = compute_initial_reservations(
            buf.len() as u32, // frame_bytes
            tell,             // ec_tell_frac after prefix
            false,            // is_transient
            3,                // lm = 20ms
            true,             // stereo
            21,               // coded_bands
        );
        // Sanity: budget is positive and the reservations follow the
        // standard §4.3.3 pattern (skip + intensity + dual fire on a
        // 64-byte frame; non-transient ⇒ no anti-collapse).
        assert!(r.total_initial > 0);
        assert_eq!(r.anti_collapse_rsv, 0);
        assert_eq!(r.skip_rsv, 8);
        assert_eq!(r.intensity_rsv, 36);
        assert_eq!(r.dual_stereo_rsv, 8);
        // Identity holds.
        assert_eq!(r.total, r.total_initial - r.total_reserved());
    }
}
