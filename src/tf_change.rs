//! Time-frequency change parameter decoding (RFC 6716 §4.3.1 + §4.3.4.5).
//!
//! ## What this module covers
//!
//! Per Table 56 of RFC 6716 (`docs/audio/opus/rfc6716-opus.txt`), the
//! CELT bitstream emits TWO related groups of TF (time-frequency)
//! resolution parameters between the coarse-energy block and the
//! §4.3.3 bit-allocation scalars:
//!
//! 1. `tf_change` — one bit per coded band, the per-band TF choice.
//!    For the first band: PDF `{3, 1}/4` (transient frames) OR
//!    `{15, 1}/16` (non-transient frames). For subsequent bands the
//!    bit is differentially coded — it toggles the running TF
//!    choice when set, so the PDF is biased against toggling:
//!    `{15, 1}/16` (transient) OR `{31, 1}/32` (non-transient).
//! 2. `tf_select` — one global flag, PDF `{1, 1}/2`. Decoded LAST
//!    (after all per-band tf_change bits), and ONLY when it could
//!    actually change the result for the per-band TF choices that
//!    were just decoded. If decoding `tf_select` would be a no-op
//!    (every per-band choice maps to the same TF adjustment under
//!    `tf_select=0` and `tf_select=1`), the encoder skips it and
//!    the decoder defaults it to `0`.
//!
//! The mapping from the `(tf_select, tf_change)` pair to the actual
//! TF adjustment per band lives in Tables 60–63 (the four `tf_select_table`
//! sub-tables), indexed by `(is_transient, tf_select, LM, tf_change)`.
//! A negative adjustment increases temporal resolution; a positive
//! adjustment increases frequency resolution (only the transient
//! tables ever produce positive entries).
//!
//! ## What is NOT in this module
//!
//! * §4.3.4.1 transient-detection / encoder-side derivation of the
//!   `transient` flag itself. We are decoder-only here; the caller
//!   passes `is_transient` in from the [`crate::frame_header`]
//!   prefix walk.
//! * The actual Hadamard transform that implements the TF resolution
//!   change in the shape decode path (RFC 6716 §4.3.4.5 final
//!   paragraph). That is band-decode work for a later round.
//! * Band budget reservation accounting. The §4.3.4.5 prose does
//!   not itself describe a bit-budget gate for `tf_change`; the
//!   per-band bits are spent unconditionally within the §4.3.3
//!   budget allocation. The "can this `tf_select` bit have any
//!   impact on the result" check is performed against the decoded
//!   per-band sequence in [`tf_select_matters`], so the caller does
//!   not need to thread budget state through this module.
//!
//! ## Clean-room provenance
//!
//! Every PDF, every adjustment value, every prose paragraph below is
//! transcribed from RFC 6716 §4.3.1 and §4.3.4.5 + Tables 60–63
//! (`docs/audio/opus/rfc6716-opus.txt`). No external implementation
//! was consulted.

use crate::range_decoder::RangeDecoder;

/// Number of `(LM, tf_change)` columns in each TF adjustment sub-table.
///
/// The four `tf_select_table` sub-tables of RFC 6716 §4.3.4.5
/// (Tables 60, 61, 62, 63) each have one row per frame-size index
/// (LM = 0..=3 ↔ 2.5 / 5 / 10 / 20 ms) and TWO columns indexed by
/// `tf_change ∈ {0, 1}`. The same shape applies to all four tables.
pub const TF_CHANGE_VALUES: usize = 2;

/// Number of LM (frame-size log) rows in each TF adjustment sub-table.
///
/// `LM = log2(frame_size_samples / 120)`, ranging over `{0, 1, 2, 3}`
/// for the four legal CELT frame sizes (2.5, 5, 10, 20 ms). RFC 6716
/// §4.3.4.5 publishes one row per LM in each of Tables 60–63.
pub const LM_VALUES: usize = 4;

/// Table 60 of RFC 6716 §4.3.4.5 — TF adjustments for **non-transient**
/// frames with `tf_select = 0`.
///
/// Indexed as `TABLE_60_NON_TRANSIENT_SEL0[LM][tf_change]`, where
/// `LM` ∈ `0..=3` (2.5 / 5 / 10 / 20 ms) and `tf_change` ∈ `{0, 1}`.
///
/// ```text
///   +-----------------+---+----+
///   | Frame size (ms) | 0 |  1 |
///   +-----------------+---+----+
///   |       2.5       | 0 | -1 |
///   |        5        | 0 | -1 |
///   |       10        | 0 | -2 |
///   |       20        | 0 | -2 |
///   +-----------------+---+----+
/// ```
pub const TABLE_60_NON_TRANSIENT_SEL0: [[i8; TF_CHANGE_VALUES]; LM_VALUES] = [
    [0, -1], // LM=0, 2.5 ms
    [0, -1], // LM=1, 5 ms
    [0, -2], // LM=2, 10 ms
    [0, -2], // LM=3, 20 ms
];

/// Table 61 of RFC 6716 §4.3.4.5 — TF adjustments for **non-transient**
/// frames with `tf_select = 1`.
///
/// ```text
///   +-----------------+---+----+
///   | Frame size (ms) | 0 |  1 |
///   +-----------------+---+----+
///   |       2.5       | 0 | -1 |
///   |        5        | 0 | -2 |
///   |       10        | 0 | -3 |
///   |       20        | 0 | -3 |
///   +-----------------+---+----+
/// ```
pub const TABLE_61_NON_TRANSIENT_SEL1: [[i8; TF_CHANGE_VALUES]; LM_VALUES] = [
    [0, -1], // LM=0, 2.5 ms
    [0, -2], // LM=1, 5 ms
    [0, -3], // LM=2, 10 ms
    [0, -3], // LM=3, 20 ms
];

/// Table 62 of RFC 6716 §4.3.4.5 — TF adjustments for **transient**
/// frames with `tf_select = 0`.
///
/// ```text
///   +-----------------+---+----+
///   | Frame size (ms) | 0 |  1 |
///   +-----------------+---+----+
///   |       2.5       | 0 | -1 |
///   |        5        | 1 |  0 |
///   |       10        | 2 |  0 |
///   |       20        | 3 |  0 |
///   +-----------------+---+----+
/// ```
pub const TABLE_62_TRANSIENT_SEL0: [[i8; TF_CHANGE_VALUES]; LM_VALUES] = [
    [0, -1], // LM=0, 2.5 ms
    [1, 0],  // LM=1, 5 ms
    [2, 0],  // LM=2, 10 ms
    [3, 0],  // LM=3, 20 ms
];

/// Table 63 of RFC 6716 §4.3.4.5 — TF adjustments for **transient**
/// frames with `tf_select = 1`.
///
/// ```text
///   +-----------------+---+----+
///   | Frame size (ms) | 0 |  1 |
///   +-----------------+---+----+
///   |       2.5       | 0 | -1 |
///   |        5        | 1 | -1 |
///   |       10        | 1 | -1 |
///   |       20        | 1 | -1 |
///   +-----------------+---+----+
/// ```
pub const TABLE_63_TRANSIENT_SEL1: [[i8; TF_CHANGE_VALUES]; LM_VALUES] = [
    [0, -1], // LM=0, 2.5 ms
    [1, -1], // LM=1, 5 ms
    [1, -1], // LM=2, 10 ms
    [1, -1], // LM=3, 20 ms
];

/// Look up the TF adjustment for a single band per RFC 6716 §4.3.4.5
/// Tables 60–63.
///
/// Parameters:
///
/// * `is_transient` — the frame-level transient flag from
///   [`crate::frame_header::CeltFrameHeader::transient`].
/// * `tf_select` — the global `tf_select` flag. `0` when the bit was
///   not decoded (gated off because both tables yield the same value
///   for every observed `tf_change`).
/// * `lm` — the frame-size log, `LM = log2(frame_size_samples / 120)`,
///   in `0..=3`. Out-of-range LM values saturate to `LM = 3` so this
///   never panics on a corrupt frame.
/// * `tf_change` — the per-band TF choice as decoded by
///   [`decode_tf_changes`].
///
/// Returns the per-band adjustment in samples-of-frequency-resolution
/// (signed): negative increases temporal resolution, positive increases
/// frequency resolution. A return value of `0` means no TF change is
/// applied to that band.
pub fn tf_adjustment(is_transient: bool, tf_select: u8, lm: u8, tf_change: bool) -> i8 {
    let lm = lm.min((LM_VALUES - 1) as u8) as usize;
    let col = tf_change as usize;
    match (is_transient, tf_select) {
        (false, 0) => TABLE_60_NON_TRANSIENT_SEL0[lm][col],
        (false, _) => TABLE_61_NON_TRANSIENT_SEL1[lm][col],
        (true, 0) => TABLE_62_TRANSIENT_SEL0[lm][col],
        (true, _) => TABLE_63_TRANSIENT_SEL1[lm][col],
    }
}

/// Per-band `tf_change` decoded result + the optional global
/// `tf_select` (RFC 6716 §4.3.4.5).
///
/// `tf_changes[k]` corresponds to the `(start + k)`-th band; the
/// vector's length equals the number of coded bands `end - start`.
/// Each entry is the raw decoder output (the per-band bit). Combine
/// with `tf_select`, `is_transient`, and the frame's `lm` via
/// [`tf_adjustment`] to obtain the actual sample adjustment.
///
/// `tf_select_decoded` is `true` when the bitstream actually carried
/// the `tf_select` bit (i.e. the §4.3.4.5 "could it have an impact"
/// test passed). When `false`, `tf_select` defaults to `0` and the
/// range decoder was not advanced for that field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TfParameters {
    /// One per-band `tf_change` bit, in band-order from `start` to
    /// `end-1` of the coded band range. Length is `end - start`.
    pub tf_changes: Vec<bool>,
    /// Global `tf_select` value in `{0, 1}`. Defaults to `0` if the
    /// bit was not decoded.
    pub tf_select: u8,
    /// `true` iff the `tf_select` bit was actually consumed from the
    /// range decoder. `false` means the field defaulted to `0` per
    /// the §4.3.4.5 "no impact" gate.
    pub tf_select_decoded: bool,
}

impl TfParameters {
    /// All-zero TF parameters across `num_bands` bands (no per-band
    /// TF change, `tf_select = 0`, not decoded). This is the silent /
    /// low-rate fallback shape.
    pub fn zeros(num_bands: usize) -> Self {
        Self {
            tf_changes: vec![false; num_bands],
            tf_select: 0,
            tf_select_decoded: false,
        }
    }
}

/// `logp` for the first-band tf_change PDF on a transient frame
/// (`{3, 1}/4` ⇒ `ft = 4`, so `logp = 2`).
const FIRST_BAND_LOGP_TRANSIENT: u32 = 2;

/// `logp` for the first-band tf_change PDF on a non-transient frame
/// (`{15, 1}/16` ⇒ `ft = 16`, so `logp = 4`).
const FIRST_BAND_LOGP_NON_TRANSIENT: u32 = 4;

/// `logp` for the subsequent-band tf_change PDF on a transient frame
/// (`{15, 1}/16` ⇒ `ft = 16`, so `logp = 4`).
const CONT_BAND_LOGP_TRANSIENT: u32 = 4;

/// `logp` for the subsequent-band tf_change PDF on a non-transient
/// frame (`{31, 1}/32` ⇒ `ft = 32`, so `logp = 5`).
const CONT_BAND_LOGP_NON_TRANSIENT: u32 = 5;

/// `logp` for the global `tf_select` PDF (`{1, 1}/2` ⇒ `logp = 1`).
const TF_SELECT_LOGP: u32 = 1;

/// Decode the per-band `tf_change` bits for the coded band range
/// `start..end` (RFC 6716 §4.3.4.5, first paragraph).
///
/// Returns one `bool` per coded band. The first bit is decoded with
/// PDF `{3, 1}/4` (transient) or `{15, 1}/16` (non-transient). The
/// PDF says a "1" is the **rare** symbol. The bitstream value is
/// XOR'd into a running TF-choice accumulator initialised to `false`;
/// subsequent bands repeat with PDF `{15, 1}/16` (transient) or
/// `{31, 1}/32` (non-transient), again with "1" being the rare
/// toggle symbol.
///
/// Returns an empty `Vec` if `start >= end` (no coded bands). The
/// caller is responsible for the `start..end` range — for pure CELT
/// it is `0..NUM_BANDS`, for hybrid mode it is `17..NUM_BANDS`.
pub fn decode_tf_changes(
    dec: &mut RangeDecoder<'_>,
    start: usize,
    end: usize,
    is_transient: bool,
) -> Vec<bool> {
    if start >= end {
        return Vec::new();
    }
    let n = end - start;
    let mut out = Vec::with_capacity(n);

    let first_logp = if is_transient {
        FIRST_BAND_LOGP_TRANSIENT
    } else {
        FIRST_BAND_LOGP_NON_TRANSIENT
    };
    let cont_logp = if is_transient {
        CONT_BAND_LOGP_TRANSIENT
    } else {
        CONT_BAND_LOGP_NON_TRANSIENT
    };

    // Running per-band TF choice. The §4.3.4.5 first paragraph says
    // "For subsequent bands, the TF choice is coded relative to the
    // previous TF choice" — that is, the rare-"1" symbol toggles the
    // running value.
    let mut curr = false;
    for k in 0..n {
        let logp = if k == 0 { first_logp } else { cont_logp };
        let toggle = dec.dec_bit_logp(logp) != 0;
        if toggle {
            curr = !curr;
        }
        out.push(curr);
    }
    out
}

/// Does the `tf_select` bit affect the per-band TF adjustments for
/// the given `tf_changes` sequence? (RFC 6716 §4.3.4.5 prose: "The
/// tf_select flag … is only decoded if it can have an impact on the
/// result knowing the value of all per-band tf_change flags.")
///
/// For each coded band, compute the TF adjustment under both
/// `tf_select = 0` and `tf_select = 1` for the band's `tf_change`
/// value. If any band's pair differs, `tf_select` matters and must
/// be decoded. If every pair is equal, `tf_select` is gated off.
///
/// Returns `true` iff the bit must be decoded.
pub fn tf_select_matters(is_transient: bool, lm: u8, tf_changes: &[bool]) -> bool {
    for &chg in tf_changes {
        let a = tf_adjustment(is_transient, 0, lm, chg);
        let b = tf_adjustment(is_transient, 1, lm, chg);
        if a != b {
            return true;
        }
    }
    false
}

/// Decode the global `tf_select` flag iff the §4.3.4.5 "can have an
/// impact" gate passes (RFC 6716 §4.3.4.5 prose).
///
/// Returns `(tf_select_value, was_decoded)`. When the gate fails,
/// the bit is not consumed from the range decoder and the value
/// defaults to `0`.
pub fn decode_tf_select(
    dec: &mut RangeDecoder<'_>,
    is_transient: bool,
    lm: u8,
    tf_changes: &[bool],
) -> (u8, bool) {
    if !tf_select_matters(is_transient, lm, tf_changes) {
        return (0, false);
    }
    let bit = dec.dec_bit_logp(TF_SELECT_LOGP) as u8;
    (bit, true)
}

/// Decode the full §4.3.4.5 TF-parameter group in Table 56 order
/// (`tf_change` per band, then `tf_select` if it would matter).
///
/// * `start..end` is the coded band range (`0..NUM_BANDS` for pure
///   CELT, `17..NUM_BANDS` for hybrid).
/// * `is_transient` is the §4.3.1 transient flag from the frame
///   header prefix.
/// * `lm` is `log2(frame_size_samples / 120)` in `0..=3`.
///
/// On `start >= end` returns [`TfParameters::zeros(0)`] and does
/// not touch the range decoder.
pub fn decode_tf_parameters(
    dec: &mut RangeDecoder<'_>,
    start: usize,
    end: usize,
    is_transient: bool,
    lm: u8,
) -> TfParameters {
    if start >= end {
        return TfParameters::zeros(0);
    }
    let tf_changes = decode_tf_changes(dec, start, end, is_transient);
    let (tf_select, tf_select_decoded) = decode_tf_select(dec, is_transient, lm, &tf_changes);
    TfParameters {
        tf_changes,
        tf_select,
        tf_select_decoded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: the four published TF tables have the right shape
    /// (LM=4 rows × tf_change=2 columns) and the documented entries
    /// agree column-for-column with the RFC.
    #[test]
    fn tables_60_through_63_match_rfc() {
        // Table 60 — non-transient, tf_select = 0.
        let expected_60: [[i8; 2]; 4] = [[0, -1], [0, -1], [0, -2], [0, -2]];
        assert_eq!(TABLE_60_NON_TRANSIENT_SEL0, expected_60);

        // Table 61 — non-transient, tf_select = 1.
        let expected_61: [[i8; 2]; 4] = [[0, -1], [0, -2], [0, -3], [0, -3]];
        assert_eq!(TABLE_61_NON_TRANSIENT_SEL1, expected_61);

        // Table 62 — transient, tf_select = 0.
        let expected_62: [[i8; 2]; 4] = [[0, -1], [1, 0], [2, 0], [3, 0]];
        assert_eq!(TABLE_62_TRANSIENT_SEL0, expected_62);

        // Table 63 — transient, tf_select = 1.
        let expected_63: [[i8; 2]; 4] = [[0, -1], [1, -1], [1, -1], [1, -1]];
        assert_eq!(TABLE_63_TRANSIENT_SEL1, expected_63);
    }

    /// Tables 60–61 (non-transient) have a `tf_change=0` column of
    /// all zeros: a "no change" choice never increases frequency
    /// resolution on a non-transient frame.
    #[test]
    fn non_transient_tf_change_zero_column_is_zeros() {
        for lm in 0..LM_VALUES {
            assert_eq!(TABLE_60_NON_TRANSIENT_SEL0[lm][0], 0);
            assert_eq!(TABLE_61_NON_TRANSIENT_SEL1[lm][0], 0);
        }
    }

    /// Tables 60–61 (non-transient) only ever decrease (or hold)
    /// the time resolution: every entry is <= 0.
    #[test]
    fn non_transient_entries_non_positive() {
        for lm in 0..LM_VALUES {
            for ch in 0..TF_CHANGE_VALUES {
                assert!(TABLE_60_NON_TRANSIENT_SEL0[lm][ch] <= 0);
                assert!(TABLE_61_NON_TRANSIENT_SEL1[lm][ch] <= 0);
            }
        }
    }

    /// tf_adjustment dispatch picks the right table per
    /// `(is_transient, tf_select)`.
    #[test]
    fn tf_adjustment_dispatch_matches_tables() {
        for lm_idx in 0..LM_VALUES {
            let lm = lm_idx as u8;
            for ch in 0..TF_CHANGE_VALUES {
                let chg = ch == 1;
                assert_eq!(
                    tf_adjustment(false, 0, lm, chg),
                    TABLE_60_NON_TRANSIENT_SEL0[lm_idx][ch]
                );
                assert_eq!(
                    tf_adjustment(false, 1, lm, chg),
                    TABLE_61_NON_TRANSIENT_SEL1[lm_idx][ch]
                );
                assert_eq!(
                    tf_adjustment(true, 0, lm, chg),
                    TABLE_62_TRANSIENT_SEL0[lm_idx][ch]
                );
                assert_eq!(
                    tf_adjustment(true, 1, lm, chg),
                    TABLE_63_TRANSIENT_SEL1[lm_idx][ch]
                );
            }
        }
    }

    /// `tf_adjustment` clamps an out-of-range `lm` to `LM = 3`
    /// instead of panicking on a corrupt frame.
    #[test]
    fn tf_adjustment_clamps_oversized_lm() {
        for bad_lm in 4u8..=255u8 {
            // Should not panic.
            let v0 = tf_adjustment(false, 0, bad_lm, false);
            let v1 = tf_adjustment(true, 1, bad_lm, true);
            // Both should match the LM=3 row.
            assert_eq!(v0, TABLE_60_NON_TRANSIENT_SEL0[3][0]);
            assert_eq!(v1, TABLE_63_TRANSIENT_SEL1[3][1]);
        }
    }

    /// `tf_adjustment` accepts any non-zero `tf_select` as "select=1"
    /// (clean coercion for the decoder's `u8` carrier).
    #[test]
    fn tf_adjustment_non_zero_select_is_select_one() {
        for sel in 1u8..=255u8 {
            assert_eq!(
                tf_adjustment(false, sel, 2, true),
                TABLE_61_NON_TRANSIENT_SEL1[2][1]
            );
            assert_eq!(
                tf_adjustment(true, sel, 3, false),
                TABLE_63_TRANSIENT_SEL1[3][0]
            );
        }
    }

    /// `TfParameters::zeros` matches the empty / low-rate default
    /// shape.
    #[test]
    fn zeros_shape() {
        let z = TfParameters::zeros(21);
        assert_eq!(z.tf_changes.len(), 21);
        assert!(z.tf_changes.iter().all(|&b| !b));
        assert_eq!(z.tf_select, 0);
        assert!(!z.tf_select_decoded);

        let empty = TfParameters::zeros(0);
        assert!(empty.tf_changes.is_empty());
    }

    /// `decode_tf_changes` with an empty band range returns an empty
    /// vector and does not touch the range decoder.
    #[test]
    fn decode_tf_changes_empty_range_is_noop() {
        let mut dec = RangeDecoder::new(&[0xFFu8; 8]);
        let before = dec.tell();
        let v = decode_tf_changes(&mut dec, 5, 5, false);
        assert!(v.is_empty());
        assert_eq!(dec.tell(), before);
        assert!(!dec.has_error());

        // start > end is also a no-op (clean degenerate).
        let mut dec = RangeDecoder::new(&[0x00u8; 8]);
        let before = dec.tell();
        let v = decode_tf_changes(&mut dec, 10, 5, true);
        assert!(v.is_empty());
        assert_eq!(dec.tell(), before);
    }

    /// `decode_tf_changes` over the full 21-band CELT range advances
    /// the decoder and returns 21 entries.
    #[test]
    fn decode_tf_changes_full_celt_range_returns_21() {
        let buf: Vec<u8> = (0u8..32).cycle().take(64).collect();
        let mut dec = RangeDecoder::new(&buf);
        let before = dec.tell();
        let v = decode_tf_changes(&mut dec, 0, 21, false);
        assert_eq!(v.len(), 21);
        assert!(dec.tell() > before);
        assert!(!dec.has_error());
    }

    /// `decode_tf_changes` over the hybrid 17..21 range yields 4
    /// bits (CELT layer of hybrid mode covers only bands 17..=20).
    #[test]
    fn decode_tf_changes_hybrid_range_returns_4() {
        let buf: Vec<u8> = (0u8..32).rev().take(48).collect();
        let mut dec = RangeDecoder::new(&buf);
        let v = decode_tf_changes(&mut dec, 17, 21, true);
        assert_eq!(v.len(), 4);
        assert!(!dec.has_error());
    }

    /// On a uniform-low (`0x00`) buffer the §4.1.1 init gives the
    /// decoder state `val = 127, rng = 128`, biasing every
    /// `dec_bit_logp(logp)` decode toward the dominant "0" symbol
    /// (since `val=127` is NOT less than `rng>>logp`). The running
    /// per-band toggle therefore never fires and every `tf_change`
    /// is `false`.
    #[test]
    fn decode_tf_changes_low_byte_buffer_all_false() {
        for is_transient in [false, true] {
            let mut dec = RangeDecoder::new(&[0x00u8; 32]);
            let v = decode_tf_changes(&mut dec, 0, 21, is_transient);
            assert!(
                v.iter().all(|&b| !b),
                "got at least one toggle on 0x00 buffer, transient={is_transient}: {v:?}"
            );
        }
    }

    /// On a uniform-high (`0xFF`) buffer the §4.1.1 init gives
    /// `val = 127 - (0xFF >> 1) = 0`, which is less than every
    /// `rng>>logp` shift the §4.3.4.5 PDFs use. The first per-band
    /// decode therefore draws the rare "1" symbol and the running
    /// TF choice flips to `true`.
    #[test]
    fn decode_tf_changes_high_byte_buffer_first_band_toggles() {
        for is_transient in [false, true] {
            let mut dec = RangeDecoder::new(&[0xFFu8; 32]);
            let v = decode_tf_changes(&mut dec, 0, 1, is_transient);
            assert_eq!(v.len(), 1);
            assert!(
                v[0],
                "first-band tf_change should be true on 0xFF buffer, transient={is_transient}"
            );
        }
    }

    /// `tf_select_matters` returns `false` when ALL `tf_change` bits
    /// are `false` AND the column-0 of the two relevant tables is
    /// equal. For non-transient at LM=2, Tables 60[2][0]=0 and
    /// 61[2][0]=0, so an all-false sequence makes tf_select moot.
    #[test]
    fn tf_select_matters_non_transient_all_false_no_impact() {
        let changes = vec![false; 21];
        // Non-transient: column 0 is (0, 0) for every LM, so no
        // impact.
        for lm in 0..=3u8 {
            assert!(
                !tf_select_matters(false, lm, &changes),
                "expected NO impact for non-transient lm={lm} all-false"
            );
        }
    }

    /// `tf_select_matters` returns `true` for non-transient when at
    /// least one band has `tf_change=1` AND the column-1 entries
    /// differ between Table 60 and Table 61 at that LM.
    ///
    /// Specifically: Table 60[1][1] = -1, Table 61[1][1] = -2 ⇒
    /// LM=1 with at least one true band must require tf_select.
    /// Same at LM=2 (-2 vs -3) and LM=3 (-2 vs -3).
    #[test]
    fn tf_select_matters_non_transient_true_at_lm1_to_3() {
        let mut changes = vec![false; 5];
        changes[2] = true;
        for lm in 1..=3u8 {
            assert!(
                tf_select_matters(false, lm, &changes),
                "expected impact for non-transient lm={lm} with one true"
            );
        }
        // At LM=0, Table 60[0][1] = -1 == Table 61[0][1] = -1, so
        // even with a true entry the bit is moot.
        assert!(!tf_select_matters(false, 0, &changes));
    }

    /// `tf_select_matters` for transient frames depends on Tables 62
    /// vs 63. At LM=0 the column-0 and column-1 entries are both
    /// (0, -1) — identical — so no impact.
    #[test]
    fn tf_select_matters_transient_lm0_no_impact() {
        let changes = vec![true, false, true, false];
        assert!(!tf_select_matters(true, 0, &changes));
    }

    /// `tf_select_matters` for transient: at LM=2 Table 62 column-0
    /// is `2` and Table 63 column-0 is `1` — differ ⇒ a single false
    /// band already makes tf_select matter.
    #[test]
    fn tf_select_matters_transient_lm2_one_false_has_impact() {
        let changes = vec![false];
        assert!(tf_select_matters(true, 2, &changes));
    }

    /// Empty `tf_changes` — tf_select cannot have any impact (no
    /// band to apply it to). The function should return `false`.
    #[test]
    fn tf_select_matters_empty_changes_no_impact() {
        for is_transient in [false, true] {
            for lm in 0..=3u8 {
                assert!(
                    !tf_select_matters(is_transient, lm, &[]),
                    "transient={is_transient} lm={lm}"
                );
            }
        }
    }

    /// `decode_tf_select` gates off cleanly: an empty `tf_changes`
    /// returns `(0, false)` and does not touch the range decoder.
    #[test]
    fn decode_tf_select_empty_changes_is_noop() {
        let mut dec = RangeDecoder::new(&[0xAAu8; 16]);
        let before = dec.tell();
        let (sel, was) = decode_tf_select(&mut dec, false, 2, &[]);
        assert_eq!(sel, 0);
        assert!(!was);
        assert_eq!(dec.tell(), before);
    }

    /// `decode_tf_select` advances the range decoder when the gate
    /// is open. (LM=2, transient, one false band: column-0 of T62 vs
    /// T63 is 2 vs 1 — differ.)
    #[test]
    fn decode_tf_select_gate_open_advances_decoder() {
        let mut dec = RangeDecoder::new(&[0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]);
        let before = dec.tell();
        let (sel, was) = decode_tf_select(&mut dec, true, 2, &[false]);
        assert!(was);
        assert!(sel <= 1);
        assert!(dec.tell() > before);
    }

    /// `decode_tf_parameters` end-to-end on the full CELT range:
    /// produces 21 tf_change bits, and `tf_select_decoded` is
    /// consistent with `tf_select_matters` over the decoded
    /// sequence.
    #[test]
    fn decode_tf_parameters_full_pipeline_consistency() {
        // Try several distinct buffers + (transient, LM) combos.
        for &buf_seed in &[0x10u8, 0x55, 0xA5, 0xCC] {
            for is_transient in [false, true] {
                for lm in 0..=3u8 {
                    let buf: Vec<u8> = (0u8..64).map(|i| i.wrapping_add(buf_seed)).collect();
                    let mut dec = RangeDecoder::new(&buf);
                    let p = decode_tf_parameters(&mut dec, 0, 21, is_transient, lm);
                    assert_eq!(p.tf_changes.len(), 21);
                    assert!(p.tf_select <= 1);
                    // The gate decision is a pure function of the
                    // decoded sequence + (is_transient, LM).
                    assert_eq!(
                        p.tf_select_decoded,
                        tf_select_matters(is_transient, lm, &p.tf_changes),
                        "tf_select_decoded out of sync with gate logic for seed=0x{buf_seed:02X} transient={is_transient} lm={lm}"
                    );
                    assert!(!dec.has_error());
                }
            }
        }
    }

    /// `decode_tf_parameters` matches a hand-stitched decode (a per-
    /// band loop followed by the gated select) on the same buffer.
    /// This is the Table-56-order locality check: callers can
    /// invoke `decode_tf_changes` + `decode_tf_select` directly and
    /// get the same byte-level behaviour.
    #[test]
    fn decode_tf_parameters_matches_hand_stitched_order() {
        let buf: Vec<u8> = (0u8..32).cycle().take(64).collect();

        // Hand-stitched path.
        let mut hand = RangeDecoder::new(&buf);
        let tf_changes = decode_tf_changes(&mut hand, 0, 21, true);
        let (sel_hand, was_hand) = decode_tf_select(&mut hand, true, 3, &tf_changes);

        // Orchestrated path.
        let mut orch = RangeDecoder::new(&buf);
        let p = decode_tf_parameters(&mut orch, 0, 21, true, 3);

        assert_eq!(p.tf_changes, tf_changes);
        assert_eq!(p.tf_select, sel_hand);
        assert_eq!(p.tf_select_decoded, was_hand);
        assert_eq!(orch.tell(), hand.tell());
    }

    /// `decode_tf_parameters` with `start == end` is a clean no-op
    /// returning an empty parameter set and not touching the range
    /// decoder.
    #[test]
    fn decode_tf_parameters_empty_range_is_noop() {
        let mut dec = RangeDecoder::new(&[0x77u8; 16]);
        let before = dec.tell();
        let p = decode_tf_parameters(&mut dec, 7, 7, true, 1);
        assert!(p.tf_changes.is_empty());
        assert_eq!(p.tf_select, 0);
        assert!(!p.tf_select_decoded);
        assert_eq!(dec.tell(), before);
    }

    /// Round-trip: applying `tf_adjustment` to each band of a hand-
    /// crafted `tf_changes` sequence reproduces the published-table
    /// entries column-for-column.
    #[test]
    fn tf_adjustment_table_round_trip() {
        let changes = [false, true, false, true];
        for is_transient in [false, true] {
            for sel in 0..=1u8 {
                for lm in 0..=3u8 {
                    for &chg in &changes {
                        let v = tf_adjustment(is_transient, sel, lm, chg);
                        let lm_idx = lm as usize;
                        let col = chg as usize;
                        let expected = match (is_transient, sel) {
                            (false, 0) => TABLE_60_NON_TRANSIENT_SEL0[lm_idx][col],
                            (false, _) => TABLE_61_NON_TRANSIENT_SEL1[lm_idx][col],
                            (true, 0) => TABLE_62_TRANSIENT_SEL0[lm_idx][col],
                            (true, _) => TABLE_63_TRANSIENT_SEL1[lm_idx][col],
                        };
                        assert_eq!(
                            v, expected,
                            "transient={is_transient} sel={sel} lm={lm} chg={chg}"
                        );
                    }
                }
            }
        }
    }
}
