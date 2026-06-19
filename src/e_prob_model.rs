//! Coarse-energy Laplace probability model — `e_prob_model` table
//! (RFC 6716 §4.3.2.1).
//!
//! ## Scope
//!
//! RFC 6716 §4.3.2.1 normatively describes the coarse-energy decode
//! as a Laplace-distributed prediction-error coder with "separate
//! parameters for each frame size in intra- and inter-frame modes".
//! The RFC text states the parameters are held in a table named
//! `e_prob_model` and delegates the numeric values to a source file
//! that sits outside the workspace's clean-room allow-list.
//!
//! As of 2026-06-08 the numeric values are staged inside the
//! workspace clean-room corpus at
//! `docs/audio/celt/tables/e_prob_model.csv`, with the structural
//! description at `docs/audio/celt/tables/e_prob_model.meta` and the
//! prose narrative at
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §1.2.
//! That moves the table itself from "docs-gap blocked" to
//! "transcribable from in-tree clean-room material". This module
//! does the transcription and provides a typed accessor.
//!
//! The Laplace decoder ALGORITHM (`ec_laplace_decode`) is a separate
//! piece, implemented in [`crate::laplace`] from RFC 6716 Appendix A
//! `laplace.c` (the reference listing embedded in the RFC's own
//! text). The [`crate::coarse_energy`] decoder feeds it this table's
//! `{prob, decay}` pairs (`prob << 7` to Q15, `decay << 6` to Q14).
//!
//! ## Table shape
//!
//! Per `docs/audio/celt/tables/e_prob_model.meta`:
//!
//! * Outer axis: frame size `LM = log2(frame_size / 120)`, range
//!   `0..=3` selecting the 120 / 240 / 480 / 960-sample frame sizes
//!   (2.5 / 5 / 10 / 20 ms at 48 kHz).
//! * Middle axis: prediction type — `0` = inter (uses time
//!   prediction), `1` = intra. This matches the staged CSV's
//!   `intra` column and the meta's `intra_axis_note`.
//! * Inner axis: 21 bands per RFC Table 55, each contributing a
//!   `(prob, decay)` pair. Both values are unsigned bytes in Q8
//!   precision per the meta.
//!
//! Total: 4 × 2 × 21 = 168 pairs = 336 bytes.
//!
//! ## Clean-room provenance
//!
//! The 336 numeric values transcribed below come from the staged
//! clean-room corpus at `docs/audio/celt/tables/e_prob_model.csv`;
//! the CSV's provenance metadata is recorded alongside it in
//! `docs/audio/celt/tables/e_prob_model.meta`. The RFC's narrative
//! shape (the Laplace model's `{prob, decay}` per-band
//! parameterisation) is the structural backbone.

use crate::coarse_energy::NUM_BANDS;

/// Number of frame-size selectors covered by `e_prob_model`. The RFC's
/// `LM = log2(frame_size / 120)` ranges over `0..=3`, covering the
/// 120 / 240 / 480 / 960-sample CELT frame sizes.
pub const NUM_LM_FRAME_SIZES: usize = 4;

/// Number of prediction-type selectors covered by `e_prob_model`. Two:
/// `0` = inter (uses time prediction against the previous frame's
/// final fine-quantised energies), `1` = intra (no temporal arm).
pub const NUM_PREDICTION_TYPES: usize = 2;

/// Inner-axis index for the inter (non-intra) prediction mode. Matches
/// the staged CSV's `intra` column value `0`.
pub const PRED_INTER: usize = 0;

/// Inner-axis index for the intra prediction mode. Matches the staged
/// CSV's `intra` column value `1`. The RFC §4.3.2.1 prose names this
/// the "intra frame" case where the temporal-arm coefficient `α = 0`.
pub const PRED_INTRA: usize = 1;

/// A single `(prob, decay)` parameter pair for one `(LM, intra,
/// band)` cell of the Laplace model.
///
/// * `prob` is the Q8 probability of the prediction error being zero,
///   per the §4.3.2.1 prose ("probability of 0" in the staged
///   meta's `intra_axis_note`).
/// * `decay` is the Q8 decay rate of the Laplace distribution.
///
/// Both values fit in `u8` as the upstream extraction records them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbDecay {
    /// Q8 probability of zero prediction error.
    pub prob: u8,
    /// Q8 Laplace decay rate.
    pub decay: u8,
}

impl ProbDecay {
    /// Construct from a raw `(prob, decay)` byte pair as it appears in
    /// the staged CSV.
    #[inline]
    pub const fn new(prob: u8, decay: u8) -> Self {
        Self { prob, decay }
    }
}

/// The full Laplace coarse-energy probability model, indexed
/// `[lm][intra][band]`.
///
/// The values below are transcribed line-for-line from
/// `docs/audio/celt/tables/e_prob_model.csv`. Layout:
///
/// * `E_PROB_MODEL[lm][intra][band]` selects one `(prob, decay)` pair.
/// * `lm` is `0..=3` (`NUM_LM_FRAME_SIZES`); `intra` is `0` (inter) or
///   `1` (intra) (`NUM_PREDICTION_TYPES`); `band` is `0..=20`
///   (`NUM_BANDS`).
///
/// Reading order across each row of the CSV is
/// `prob0, decay0, prob1, decay1, ..., prob20, decay20`, so the
/// per-band `ProbDecay` entries here are paired byte-for-byte with the
/// CSV's 42-byte data rows.
pub const E_PROB_MODEL: [[[ProbDecay; NUM_BANDS]; NUM_PREDICTION_TYPES]; NUM_LM_FRAME_SIZES] = [
    // LM = 0 (120-sample / 2.5 ms frames).
    [
        // intra = 0 (inter prediction).
        [
            ProbDecay::new(72, 127),
            ProbDecay::new(65, 129),
            ProbDecay::new(66, 128),
            ProbDecay::new(65, 128),
            ProbDecay::new(64, 128),
            ProbDecay::new(62, 128),
            ProbDecay::new(64, 128),
            ProbDecay::new(64, 128),
            ProbDecay::new(92, 78),
            ProbDecay::new(92, 79),
            ProbDecay::new(92, 78),
            ProbDecay::new(90, 79),
            ProbDecay::new(116, 41),
            ProbDecay::new(115, 40),
            ProbDecay::new(114, 40),
            ProbDecay::new(132, 26),
            ProbDecay::new(132, 26),
            ProbDecay::new(145, 17),
            ProbDecay::new(161, 12),
            ProbDecay::new(176, 10),
            ProbDecay::new(177, 11),
        ],
        // intra = 1 (intra prediction).
        [
            ProbDecay::new(24, 179),
            ProbDecay::new(48, 138),
            ProbDecay::new(54, 135),
            ProbDecay::new(54, 132),
            ProbDecay::new(53, 134),
            ProbDecay::new(56, 133),
            ProbDecay::new(55, 132),
            ProbDecay::new(55, 132),
            ProbDecay::new(61, 114),
            ProbDecay::new(70, 96),
            ProbDecay::new(74, 88),
            ProbDecay::new(75, 88),
            ProbDecay::new(87, 74),
            ProbDecay::new(89, 66),
            ProbDecay::new(91, 67),
            ProbDecay::new(100, 59),
            ProbDecay::new(108, 50),
            ProbDecay::new(120, 40),
            ProbDecay::new(122, 37),
            ProbDecay::new(97, 43),
            ProbDecay::new(78, 50),
        ],
    ],
    // LM = 1 (240-sample / 5 ms frames).
    [
        // intra = 0.
        [
            ProbDecay::new(83, 78),
            ProbDecay::new(84, 81),
            ProbDecay::new(88, 75),
            ProbDecay::new(86, 74),
            ProbDecay::new(87, 71),
            ProbDecay::new(90, 73),
            ProbDecay::new(93, 74),
            ProbDecay::new(93, 74),
            ProbDecay::new(109, 40),
            ProbDecay::new(114, 36),
            ProbDecay::new(117, 34),
            ProbDecay::new(117, 34),
            ProbDecay::new(143, 17),
            ProbDecay::new(145, 18),
            ProbDecay::new(146, 19),
            ProbDecay::new(162, 12),
            ProbDecay::new(165, 10),
            ProbDecay::new(178, 7),
            ProbDecay::new(189, 6),
            ProbDecay::new(190, 8),
            ProbDecay::new(177, 9),
        ],
        // intra = 1.
        [
            ProbDecay::new(23, 178),
            ProbDecay::new(54, 115),
            ProbDecay::new(63, 102),
            ProbDecay::new(66, 98),
            ProbDecay::new(69, 99),
            ProbDecay::new(74, 89),
            ProbDecay::new(71, 91),
            ProbDecay::new(73, 91),
            ProbDecay::new(78, 89),
            ProbDecay::new(86, 80),
            ProbDecay::new(92, 66),
            ProbDecay::new(93, 64),
            ProbDecay::new(102, 59),
            ProbDecay::new(103, 60),
            ProbDecay::new(104, 60),
            ProbDecay::new(117, 52),
            ProbDecay::new(123, 44),
            ProbDecay::new(138, 35),
            ProbDecay::new(133, 31),
            ProbDecay::new(97, 38),
            ProbDecay::new(77, 45),
        ],
    ],
    // LM = 2 (480-sample / 10 ms frames).
    [
        // intra = 0.
        [
            ProbDecay::new(61, 90),
            ProbDecay::new(93, 60),
            ProbDecay::new(105, 42),
            ProbDecay::new(107, 41),
            ProbDecay::new(110, 45),
            ProbDecay::new(116, 38),
            ProbDecay::new(113, 38),
            ProbDecay::new(112, 38),
            ProbDecay::new(124, 26),
            ProbDecay::new(132, 27),
            ProbDecay::new(136, 19),
            ProbDecay::new(140, 20),
            ProbDecay::new(155, 14),
            ProbDecay::new(159, 16),
            ProbDecay::new(158, 18),
            ProbDecay::new(170, 13),
            ProbDecay::new(177, 10),
            ProbDecay::new(187, 8),
            ProbDecay::new(192, 6),
            ProbDecay::new(175, 9),
            ProbDecay::new(159, 10),
        ],
        // intra = 1.
        [
            ProbDecay::new(21, 178),
            ProbDecay::new(59, 110),
            ProbDecay::new(71, 86),
            ProbDecay::new(75, 85),
            ProbDecay::new(84, 83),
            ProbDecay::new(91, 66),
            ProbDecay::new(88, 73),
            ProbDecay::new(87, 72),
            ProbDecay::new(92, 75),
            ProbDecay::new(98, 72),
            ProbDecay::new(105, 58),
            ProbDecay::new(107, 54),
            ProbDecay::new(115, 52),
            ProbDecay::new(114, 55),
            ProbDecay::new(112, 56),
            ProbDecay::new(129, 51),
            ProbDecay::new(132, 40),
            ProbDecay::new(150, 33),
            ProbDecay::new(140, 29),
            ProbDecay::new(98, 35),
            ProbDecay::new(77, 42),
        ],
    ],
    // LM = 3 (960-sample / 20 ms frames).
    [
        // intra = 0.
        [
            ProbDecay::new(42, 121),
            ProbDecay::new(96, 66),
            ProbDecay::new(108, 43),
            ProbDecay::new(111, 40),
            ProbDecay::new(117, 44),
            ProbDecay::new(123, 32),
            ProbDecay::new(120, 36),
            ProbDecay::new(119, 33),
            ProbDecay::new(127, 33),
            ProbDecay::new(134, 34),
            ProbDecay::new(139, 21),
            ProbDecay::new(147, 23),
            ProbDecay::new(152, 20),
            ProbDecay::new(158, 25),
            ProbDecay::new(154, 26),
            ProbDecay::new(166, 21),
            ProbDecay::new(173, 16),
            ProbDecay::new(184, 13),
            ProbDecay::new(184, 10),
            ProbDecay::new(150, 13),
            ProbDecay::new(139, 15),
        ],
        // intra = 1.
        [
            ProbDecay::new(22, 178),
            ProbDecay::new(63, 114),
            ProbDecay::new(74, 82),
            ProbDecay::new(84, 83),
            ProbDecay::new(92, 82),
            ProbDecay::new(103, 62),
            ProbDecay::new(96, 72),
            ProbDecay::new(96, 67),
            ProbDecay::new(101, 73),
            ProbDecay::new(107, 72),
            ProbDecay::new(113, 55),
            ProbDecay::new(118, 52),
            ProbDecay::new(125, 52),
            ProbDecay::new(118, 52),
            ProbDecay::new(117, 55),
            ProbDecay::new(135, 49),
            ProbDecay::new(137, 39),
            ProbDecay::new(157, 32),
            ProbDecay::new(145, 29),
            ProbDecay::new(97, 33),
            ProbDecay::new(77, 40),
        ],
    ],
];

/// Look up the Q8 `(prob, decay)` Laplace parameter pair for one
/// `(LM, intra, band)` cell.
///
/// * `lm` is `log2(frame_size / 120)`, range `0..=3`. Out-of-range
///   values return `None`.
/// * `intra` is `false` for inter (time-predicted) frames and `true`
///   for intra frames. Maps onto the table's middle axis as
///   `false -> PRED_INTER`, `true -> PRED_INTRA`.
/// * `band` is the CELT band index, range `0..=20`. Out-of-range
///   values return `None`.
///
/// Pure-CELT decoders call this for every band. Hybrid-mode decoders
/// call it for bands `17..=20` only (the SILK layer covers bands
/// `0..=16` per RFC 6716 §4.3).
#[inline]
pub fn prob_decay(lm: u32, intra: bool, band: usize) -> Option<ProbDecay> {
    let lm = lm as usize;
    if lm >= NUM_LM_FRAME_SIZES || band >= NUM_BANDS {
        return None;
    }
    let pred = if intra { PRED_INTRA } else { PRED_INTER };
    Some(E_PROB_MODEL[lm][pred][band])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table shape must match the staged
    /// `docs/audio/celt/tables/e_prob_model.meta` `raw_dims: [4][2][42]`
    /// (which resolves to `[4 frame-sizes][2 prediction-types][21 bands
    /// x 2]` after pairing the `(prob, decay)` bytes).
    #[test]
    fn table_shape_matches_meta() {
        assert_eq!(NUM_LM_FRAME_SIZES, 4);
        assert_eq!(NUM_PREDICTION_TYPES, 2);
        assert_eq!(NUM_BANDS, 21);
        // 4 * 2 * 21 = 168 pairs = 336 bytes, matching the meta's
        // `element_count: 336`.
        let pairs: usize = E_PROB_MODEL.len() * E_PROB_MODEL[0].len() * E_PROB_MODEL[0][0].len();
        assert_eq!(pairs, 168);
        assert_eq!(pairs * 2, 336);
    }

    /// Spot-check the corners of the staged CSV against the transcribed
    /// table. Catches accidental row/column swaps without re-listing
    /// every value.
    #[test]
    fn transcription_corners_match_csv() {
        // CSV row 2 (LM=0, intra=0), first pair: 72,127.
        assert_eq!(E_PROB_MODEL[0][PRED_INTER][0], ProbDecay::new(72, 127));
        // CSV row 2, last pair (band 20): 177,11.
        assert_eq!(E_PROB_MODEL[0][PRED_INTER][20], ProbDecay::new(177, 11));
        // CSV row 3 (LM=0, intra=1), first pair: 24,179.
        assert_eq!(E_PROB_MODEL[0][PRED_INTRA][0], ProbDecay::new(24, 179));
        // CSV row 9 (LM=3, intra=1), last pair: 77,40.
        assert_eq!(E_PROB_MODEL[3][PRED_INTRA][20], ProbDecay::new(77, 40));
        // CSV row 8 (LM=3, intra=0), first pair: 42,121.
        assert_eq!(E_PROB_MODEL[3][PRED_INTER][0], ProbDecay::new(42, 121));
    }

    /// The accessor must fold the `bool intra` selector onto the right
    /// inner-axis index per the meta's `intra_axis_note`
    /// (`0 = inter`, `1 = intra`).
    #[test]
    fn accessor_maps_intra_flag_to_inner_axis() {
        // LM=2, band 10. CSV row 6 (LM=2, intra=0) band 10 pair: 136,19.
        assert_eq!(prob_decay(2, false, 10), Some(ProbDecay::new(136, 19)));
        // CSV row 7 (LM=2, intra=1) band 10 pair: 105,58.
        assert_eq!(prob_decay(2, true, 10), Some(ProbDecay::new(105, 58)));
    }

    /// Out-of-range queries must return `None` rather than panic.
    #[test]
    fn accessor_returns_none_out_of_range() {
        assert_eq!(prob_decay(4, false, 0), None); // lm too large
        assert_eq!(prob_decay(0, false, 21), None); // band too large
        assert_eq!(prob_decay(0, true, 21), None);
        assert_eq!(prob_decay(5, true, 25), None);
    }

    /// Every transcribed `prob` byte fits in 8 bits by construction
    /// (`u8`), but verify that no value sits at the degenerate `0`
    /// boundary (which would correspond to "zero probability of zero"
    /// — meaningless for the Laplace model). The staged CSV inspection
    /// shows the smallest `prob` value is `21` (LM=2, intra=1, band 0).
    #[test]
    fn no_zero_prob_entries() {
        let mut min_prob = u8::MAX;
        let mut max_prob = 0u8;
        for (lm, lm_row) in E_PROB_MODEL.iter().enumerate() {
            for (intra, intra_row) in lm_row.iter().enumerate() {
                for (band, cell) in intra_row.iter().enumerate() {
                    let p = cell.prob;
                    assert!(p > 0, "zero prob at lm={lm} intra={intra} band={band}");
                    if p < min_prob {
                        min_prob = p;
                    }
                    if p > max_prob {
                        max_prob = p;
                    }
                }
            }
        }
        assert_eq!(min_prob, 21);
        assert_eq!(max_prob, 192);
    }

    /// Every transcribed `decay` byte fits in 8 bits by construction;
    /// also confirm none is zero (a zero decay would collapse the
    /// Laplace tail to a delta at 0).
    #[test]
    fn no_zero_decay_entries() {
        let mut min_decay = u8::MAX;
        let mut max_decay = 0u8;
        for (lm, lm_row) in E_PROB_MODEL.iter().enumerate() {
            for (intra, intra_row) in lm_row.iter().enumerate() {
                for (band, cell) in intra_row.iter().enumerate() {
                    let d = cell.decay;
                    assert!(d > 0, "zero decay at lm={lm} intra={intra} band={band}");
                    if d < min_decay {
                        min_decay = d;
                    }
                    if d > max_decay {
                        max_decay = d;
                    }
                }
            }
        }
        // From CSV inspection: smallest decay is 6 (LM=1 intra=0 band 18,
        // and LM=2 intra=0 band 18); largest is 179 (LM=0 intra=1 band 0).
        assert_eq!(min_decay, 6);
        assert_eq!(max_decay, 179);
    }

    /// Sanity: the intra-mode `prob` for band 0 monotonically drops as
    /// `LM` increases from 0 to 2 (24 → 23 → 21), reflecting the
    /// staged CSV's shape. This is a regression sentinel against
    /// accidental row swaps between `LM` levels.
    #[test]
    fn intra_band0_prob_decreases_through_lm_2() {
        assert_eq!(E_PROB_MODEL[0][PRED_INTRA][0].prob, 24);
        assert_eq!(E_PROB_MODEL[1][PRED_INTRA][0].prob, 23);
        assert_eq!(E_PROB_MODEL[2][PRED_INTRA][0].prob, 21);
        // LM=3 intra band 0 bounces back up to 22 per the CSV; not
        // monotone over all four rows.
        assert_eq!(E_PROB_MODEL[3][PRED_INTRA][0].prob, 22);
    }

    /// `ProbDecay::new` is `const fn` so it can be used in the static
    /// table without runtime overhead. Verify by constructing an
    /// instance in a const context.
    #[test]
    fn prob_decay_is_const_constructible() {
        const SAMPLE: ProbDecay = ProbDecay::new(64, 128);
        assert_eq!(SAMPLE.prob, 64);
        assert_eq!(SAMPLE.decay, 128);
    }
}
