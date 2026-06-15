//! CELT MDCT band layout — band edges and per-band bin ranges
//! (RFC 6716 §4.3, Table 55).
//!
//! ## What this module covers
//!
//! Every band-walking step of the CELT decode chain — the §4.3.6
//! denormalization across 21 bands, the §4.3.7 inverse-MDCT input
//! layout, the §4.3.4 per-band PVQ shape walk, the §4.3.4.4 band-split
//! recursion, the §4.3.5 anti-collapse pass — needs to map a band index
//! to the half-open range of MDCT bins `[start, end)` that the band
//! occupies in the (one-channel) spectrum. RFC 6716 §4.3 Table 55 gives
//! the per-band bin *count* `N[band]` for each of the four CELT frame
//! sizes; the band **edges** (the running cumulative bin offsets) are
//! the natural companion that those walkers index.
//!
//! This module exposes the canonical band-edge layout and the
//! derived range / count accessors:
//!
//! * [`EBAND_EDGES_5MS`] — the 22 band edges at the 2.5 ms (LM = 0)
//!   frame size, i.e. the cumulative MDCT-bin offsets `edge[band]` with
//!   `edge[0] = 0` and `edge[21] = 100` (the 0–20 kHz coded range). The
//!   per-band bin count at LM = 0 is `edge[band + 1] - edge[band]`, and
//!   the count at frame-size shift `lm` is that difference scaled by
//!   `1 << lm` — exactly the columns of RFC 6716 Table 55.
//! * [`band_edge`] / [`band_bin_range`] / [`band_bins`] /
//!   [`coded_total_bins`] — accessors that compute a band's bin offset,
//!   half-open bin range, bin count, or the window's total bin count at
//!   a given `lm`.
//!
//! The bin counts produced here are bit-identical to
//! [`crate::band_minimums::BAND_BINS_LM`]; this module is the
//! edge-form companion of that count table, and a unit test pins the
//! two against each other so neither transcription can drift.
//!
//! ## Why edges and not just counts
//!
//! `BAND_BINS_LM` answers "how wide is band `b`"; the band-decode
//! walkers also need "where does band `b` start in the spectrum", which
//! is the running sum of the widths below it. Computing that prefix sum
//! at every call site is error-prone (off-by-one at the Hybrid coding
//! start, double-counting under the LM scaling), so the offset is
//! derived once here from the canonical edge layout.
//!
//! ## Clean-room provenance
//!
//! Every numeric value is RFC 6716 §4.3 Table 55
//! (`docs/audio/opus/rfc6716-opus.txt` lines 5813–5870): the LM = 0
//! "Bins" column is `[1,1,1,1,1,1,1,1,2,2,2,2,4,4,4,6,6,8,12,18,22]`,
//! whose running prefix sum is [`EBAND_EDGES_5MS`]. The `1 << lm`
//! frame-size scaling is the table's own published doubling across the
//! four frame-size columns (2.5 / 5 / 10 / 20 ms). No external library
//! source was consulted.

use crate::band_minimums::NUM_LM;
use crate::coarse_energy::NUM_BANDS;

/// Number of band edges — one more than the band count, since `N`
/// bands have `N + 1` boundaries.
pub const NUM_BAND_EDGES: usize = NUM_BANDS + 1;

/// The CELT MDCT band edges at the 2.5 ms (LM = 0) frame size, in
/// per-channel MDCT bins — RFC 6716 §4.3 Table 55.
///
/// `EBAND_EDGES_5MS[band]` is the cumulative bin offset of the start of
/// `band`; `EBAND_EDGES_5MS[band + 1]` is its (exclusive) end. The
/// final entry `EBAND_EDGES_5MS[21] = 100` is the total coded bin count
/// at LM = 0 (the 0–20 kHz range); the underlying MDCT has `120 << lm`
/// bins per channel, so the 20-bin gap above 20 kHz at LM = 0 is not
/// band-coded.
///
/// The per-band bin count at LM = 0 is the consecutive difference
/// `EBAND_EDGES_5MS[band + 1] - EBAND_EDGES_5MS[band]`, reproducing the
/// "2.5 ms / Bins" column of Table 55
/// (`[1,1,1,1,1,1,1,1,2,2,2,2,4,4,4,6,6,8,12,18,22]`); at frame-size
/// shift `lm` the count and the edges scale by `1 << lm`.
pub const EBAND_EDGES_5MS: [u32; NUM_BAND_EDGES] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// The MDCT-bin offset (one channel) of the start of `band` at
/// frame-size shift `lm`, i.e. `EBAND_EDGES_5MS[band] << lm`.
///
/// `band` may be `0..=NUM_BANDS` — passing `NUM_BANDS` (= 21) returns
/// the total coded bin count `100 << lm`, the exclusive end of the last
/// band. Returns `None` if `band > NUM_BANDS` or `lm >= NUM_LM`.
pub fn band_edge(band: usize, lm: u32) -> Option<u32> {
    if band >= NUM_BAND_EDGES || lm as usize >= NUM_LM {
        return None;
    }
    Some(EBAND_EDGES_5MS[band] << lm)
}

/// The per-channel MDCT-bin count `N[band]` of `band` at frame-size
/// shift `lm` (RFC 6716 §4.3 Table 55), i.e. the consecutive band-edge
/// difference scaled by `1 << lm`.
///
/// Returns `None` if `band >= NUM_BANDS` or `lm >= NUM_LM`. The result
/// is bit-identical to `crate::band_minimums::BAND_BINS_LM[lm][band]`.
pub fn band_bins(band: usize, lm: u32) -> Option<u32> {
    if band >= NUM_BANDS || lm as usize >= NUM_LM {
        return None;
    }
    Some((EBAND_EDGES_5MS[band + 1] - EBAND_EDGES_5MS[band]) << lm)
}

/// The half-open MDCT-bin range `[start, end)` (one channel) that
/// `band` occupies at frame-size shift `lm`.
///
/// `start = band_edge(band, lm)` and `end = band_edge(band + 1, lm)`,
/// so `end - start == band_bins(band, lm)`. Returns `None` if
/// `band >= NUM_BANDS` or `lm >= NUM_LM`.
pub fn band_bin_range(band: usize, lm: u32) -> Option<(u32, u32)> {
    if band >= NUM_BANDS || lm as usize >= NUM_LM {
        return None;
    }
    Some((EBAND_EDGES_5MS[band] << lm, EBAND_EDGES_5MS[band + 1] << lm))
}

/// The total per-channel MDCT-bin count over the coded-band window
/// `[start, end)` at frame-size shift `lm`.
///
/// This is `band_edge(end, lm) - band_edge(start, lm)` — the bin span
/// the window covers. Pure CELT uses `start = 0`, Hybrid mode uses
/// `start = 17`; `end` is the signaled coding end (at most `NUM_BANDS`).
/// The full-band case `(0, NUM_BANDS)` returns `100 << lm`.
///
/// Returns `None` if `start > end`, `end > NUM_BANDS`, or
/// `lm >= NUM_LM`.
pub fn coded_total_bins(start: usize, end: usize, lm: u32) -> Option<u32> {
    if start > end || end > NUM_BANDS || lm as usize >= NUM_LM {
        return None;
    }
    Some((EBAND_EDGES_5MS[end] - EBAND_EDGES_5MS[start]) << lm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band_minimums::{BAND_BINS_LM, SHORT_FRAME_BAND_BINS};

    #[test]
    fn edges_have_band_count_plus_one_entries() {
        assert_eq!(EBAND_EDGES_5MS.len(), NUM_BAND_EDGES);
        assert_eq!(NUM_BAND_EDGES, NUM_BANDS + 1);
    }

    #[test]
    fn edges_start_at_zero_end_at_hundred() {
        assert_eq!(EBAND_EDGES_5MS[0], 0);
        assert_eq!(EBAND_EDGES_5MS[NUM_BANDS], 100);
    }

    #[test]
    fn edges_are_strictly_increasing() {
        for w in EBAND_EDGES_5MS.windows(2) {
            assert!(w[1] > w[0], "edges must be strictly increasing: {w:?}");
        }
    }

    #[test]
    fn lm0_band_widths_match_table_55_bins_column() {
        // RFC 6716 Table 55, the "2.5 ms / Bins" column.
        let expected: [u32; NUM_BANDS] = [
            1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 6, 6, 8, 12, 18, 22,
        ];
        for band in 0..NUM_BANDS {
            assert_eq!(
                EBAND_EDGES_5MS[band + 1] - EBAND_EDGES_5MS[band],
                expected[band],
                "band {band} LM=0 width"
            );
        }
    }

    #[test]
    fn band_bins_match_band_bins_lm_table_for_every_lm() {
        // The edge-form layout and the count-form BAND_BINS_LM table
        // are two transcriptions of the same Table 55; pin them so
        // neither can drift away from the canonical layout.
        for lm in 0..NUM_LM as u32 {
            for (band, &expected) in BAND_BINS_LM[lm as usize].iter().enumerate() {
                assert_eq!(
                    band_bins(band, lm).unwrap(),
                    expected,
                    "band_bins disagrees with BAND_BINS_LM at lm={lm}, band={band}"
                );
            }
        }
    }

    #[test]
    fn band_bins_match_short_frame_band_bins_at_lm0() {
        for (band, &expected) in SHORT_FRAME_BAND_BINS.iter().enumerate() {
            assert_eq!(band_bins(band, 0).unwrap(), expected);
        }
    }

    #[test]
    fn band_edge_is_running_prefix_sum_of_widths() {
        for lm in 0..NUM_LM as u32 {
            let mut acc = 0u32;
            for band in 0..NUM_BANDS {
                assert_eq!(band_edge(band, lm).unwrap(), acc, "lm={lm} band={band}");
                acc += band_bins(band, lm).unwrap();
            }
            // The edge past the last band is the total coded bin count.
            assert_eq!(band_edge(NUM_BANDS, lm).unwrap(), acc);
            assert_eq!(acc, 100 << lm);
        }
    }

    #[test]
    fn band_bin_range_endpoints_match_edges_and_width() {
        for lm in 0..NUM_LM as u32 {
            for band in 0..NUM_BANDS {
                let (start, end) = band_bin_range(band, lm).unwrap();
                assert_eq!(start, band_edge(band, lm).unwrap());
                assert_eq!(end, band_edge(band + 1, lm).unwrap());
                assert_eq!(end - start, band_bins(band, lm).unwrap());
            }
        }
    }

    #[test]
    fn band_bin_ranges_tile_the_spectrum_without_gaps_or_overlap() {
        for lm in 0..NUM_LM as u32 {
            let mut cursor = 0u32;
            for band in 0..NUM_BANDS {
                let (start, end) = band_bin_range(band, lm).unwrap();
                assert_eq!(start, cursor, "gap/overlap before band {band} at lm={lm}");
                cursor = end;
            }
            assert_eq!(cursor, 100 << lm);
        }
    }

    #[test]
    fn coded_total_bins_full_band_window() {
        for lm in 0..NUM_LM as u32 {
            assert_eq!(coded_total_bins(0, NUM_BANDS, lm).unwrap(), 100 << lm);
        }
    }

    #[test]
    fn coded_total_bins_hybrid_window() {
        // Hybrid mode codes bands 17..=20 (start = 17). The LM=0 span is
        // edge[21] - edge[17] = 100 - 40 = 60 bins.
        assert_eq!(coded_total_bins(17, NUM_BANDS, 0).unwrap(), 60);
        assert_eq!(coded_total_bins(17, NUM_BANDS, 3).unwrap(), 60 << 3);
    }

    #[test]
    fn coded_total_bins_decomposes_into_per_band_bins() {
        for lm in 0..NUM_LM as u32 {
            for start in 0..=NUM_BANDS {
                for end in start..=NUM_BANDS {
                    let total = coded_total_bins(start, end, lm).unwrap();
                    let sum: u32 = (start..end).map(|b| band_bins(b, lm).unwrap()).sum();
                    assert_eq!(total, sum, "lm={lm} window=[{start},{end})");
                }
            }
        }
    }

    #[test]
    fn coded_total_bins_empty_window_is_zero() {
        for lm in 0..NUM_LM as u32 {
            for b in 0..=NUM_BANDS {
                assert_eq!(coded_total_bins(b, b, lm).unwrap(), 0);
            }
        }
    }

    #[test]
    fn band_edge_accepts_the_terminal_edge_index() {
        // band_edge tolerates band == NUM_BANDS (the exclusive end of
        // the last band); band_bins / band_bin_range do not (they index
        // a band, which must be < NUM_BANDS).
        assert_eq!(band_edge(NUM_BANDS, 0).unwrap(), 100);
        assert!(band_bins(NUM_BANDS, 0).is_none());
        assert!(band_bin_range(NUM_BANDS, 0).is_none());
    }

    #[test]
    fn out_of_range_band_and_lm_return_none() {
        assert!(band_edge(NUM_BAND_EDGES, 0).is_none());
        assert!(band_edge(0, NUM_LM as u32).is_none());
        assert!(band_bins(NUM_BANDS, 0).is_none());
        assert!(band_bins(0, NUM_LM as u32).is_none());
        assert!(band_bin_range(NUM_BANDS, 0).is_none());
        assert!(band_bin_range(0, NUM_LM as u32).is_none());
        assert!(coded_total_bins(0, NUM_BANDS + 1, 0).is_none());
        assert!(coded_total_bins(0, 0, NUM_LM as u32).is_none());
    }

    #[test]
    fn coded_total_bins_rejects_inverted_window() {
        assert!(coded_total_bins(5, 3, 0).is_none());
    }
}
