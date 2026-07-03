//! Encoder-side control decisions RFC 6716 §5.3.4 spells out in
//! closed form — currently the §5.3.4.1 band-boost rule.
//!
//! ## §5.3.4.1 Band Boost
//!
//! > "The reference encoder makes a decision to boost a band when the
//! > energy of that band is significantly higher than that of the
//! > neighboring bands. Let E_j be the log-energy of band j, we
//! > define
//! >
//! >   D_j = 2*E_j - E_j-1 - E_j+1
//! >
//! > The allocation of band j is boosted once if D_j > t1 and twice
//! > if D_j > t2. For LM>=1, t1=2 and t2=4, while for LM<1, t1=3 and
//! > t2=5."
//!
//! `E_j` is the per-band base-2 log-energy (1.0 = 6 dB — the same
//! axis the §4.3.2.1 coarse quantizer works on), so `t1 = 2` means "a
//! band poking 12 dB above the mean of its neighbors gets one boost
//! quantum". One boost is one §4.3.3 dynalloc quantum,
//! `min(8*N, max(48, N))` 1/8 bits for a band of `N` MDCT bins — the
//! step size of the decode-side boost loop
//! ([`decode_band_boosts`](crate::band_cap::decode_band_boosts)), so
//! a "once"/"twice" decision lands exactly on the wire grid.
//!
//! The `D_j` metric needs both neighbors, so the first and last coded
//! bands are never boosted by this rule (the RFC formula is undefined
//! there); band boosts are a free encoder choice, so the conservative
//! no-boost edge behaviour is a legal documented decision.
//!
//! Boost decisions are encoder freedom (§5.3.4: "the three mechanisms
//! that can be used by the encoder to adjust the bitrate ... are band
//! boost, allocation trim, and band skipping") — the decoder derives
//! everything from the bitstream, so any output of this rule stays in
//! lockstep automatically.
//!
//! The §5.3.4.2 allocation-trim deviation ("the trim can deviate by
//! +/- 2 depending on the spectral tilt") is **not** implemented: the
//! RFC gives the direction and the bound but not the tilt→deviation
//! map, so a closed-form transcription is not possible from the
//! staged docs (documented docs gap; the safe default of 5 stands).
//!
//! ## Clean-room provenance
//!
//! RFC 6716 §5.3.4/§5.3.4.1 (`docs/audio/opus/rfc6716-opus.txt`) and
//! the §4.3.3 dynalloc quantum already transcribed in
//! [`crate::band_cap`]. No external library source was consulted.

use crate::band_minimums::BAND_BINS_LM;
use crate::coarse_energy::NUM_BANDS;

/// The §5.3.4.1 boost thresholds `(t1, t2)` in base-2 log-energy
/// units: `(2, 4)` for `LM >= 1`, `(3, 5)` for `LM < 1`.
#[inline]
pub fn boost_thresholds(lm: u32) -> (f32, f32) {
    if lm >= 1 {
        (2.0, 4.0)
    } else {
        (3.0, 5.0)
    }
}

/// The §4.3.3 dynalloc boost quantum for a band of `n` MDCT bins, in
/// 1/8 bits: `min(8*N, max(48, N))` — the per-"1"-bit step of the
/// decode-side boost loop.
#[inline]
pub fn boost_quanta_8th(n: u32) -> i32 {
    (8 * n).min(n.max(48)) as i32
}

/// The §5.3.4.1 band-boost decision over a coded-band window.
///
/// `window_log_energy` holds the per-band base-2 log-energies of the
/// coded bands `[start, end)` (window-relative indexing — the
/// [`analyze_bands_f32`](crate::band_analysis::analyze_bands_f32)
/// output order). For each *interior* band `j` the §5.3.4.1 contrast
/// `D_j = 2*E_j - E_{j-1} - E_{j+1}` is compared against the per-LM
/// thresholds; a band is boosted once (`D_j > t1`) or twice
/// (`D_j > t2`), each boost worth one §4.3.3 dynalloc quantum for
/// that band's width.
///
/// Returns the per-coded-band boost targets in 1/8 bits — exactly the
/// `target_boost` input
/// [`encode_band_boosts`](crate::band_cap::encode_band_boosts) and
/// the frame encoders take (the encode loop gate-truncates against
/// the live budget, so an over-budget decision degrades exactly the
/// way the wire format prescribes).
///
/// Returns `None` when `lm > 3`, the window is out of range, or the
/// energy slice length disagrees with the window.
pub fn choose_band_boosts(
    window_log_energy: &[f32],
    lm: u32,
    start: usize,
    end: usize,
) -> Option<Vec<i32>> {
    if lm > 3 || start > end || end > NUM_BANDS {
        return None;
    }
    let coded = end - start;
    if window_log_energy.len() != coded {
        return None;
    }
    let (t1, t2) = boost_thresholds(lm);
    let mut out = vec![0i32; coded];
    // Interior bands only: D_j needs both neighbors inside the window.
    for j in 1..coded.saturating_sub(1) {
        let d = 2.0 * window_log_energy[j] - window_log_energy[j - 1] - window_log_energy[j + 1];
        let times = if d > t2 {
            2
        } else if d > t1 {
            1
        } else {
            0
        };
        if times > 0 {
            let n = BAND_BINS_LM[lm as usize][start + j];
            out[j] = times * boost_quanta_8th(n);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The threshold pairs are the literal §5.3.4.1 constants split at
    /// LM = 1.
    #[test]
    fn thresholds_split_at_lm_one() {
        assert_eq!(boost_thresholds(0), (3.0, 5.0));
        assert_eq!(boost_thresholds(1), (2.0, 4.0));
        assert_eq!(boost_thresholds(2), (2.0, 4.0));
        assert_eq!(boost_thresholds(3), (2.0, 4.0));
    }

    /// The quantum is the §4.3.3 `min(8*N, max(48, N))` decode-loop
    /// step.
    #[test]
    fn quanta_matches_dynalloc_step() {
        assert_eq!(boost_quanta_8th(1), 8); // 8*1 < max(48, 1)
        assert_eq!(boost_quanta_8th(4), 32); // 8*4 < 48
        assert_eq!(boost_quanta_8th(6), 48); // 8*6 == 48
        assert_eq!(boost_quanta_8th(8), 48); // 64 > 48 → 48
        assert_eq!(boost_quanta_8th(88), 88); // N > 48 → N
    }

    /// A flat envelope earns no boosts at any LM.
    #[test]
    fn flat_envelope_earns_no_boosts() {
        for lm in 0..=3u32 {
            let energy = vec![1.5f32; NUM_BANDS];
            let boosts = choose_band_boosts(&energy, lm, 0, NUM_BANDS).unwrap();
            assert!(boosts.iter().all(|&b| b == 0), "lm={lm}");
        }
    }

    /// An isolated peak crosses t1 (one quantum) and t2 (two quanta)
    /// exactly per the §5.3.4.1 rule, on both sides of the LM split.
    #[test]
    fn isolated_peak_boosts_once_then_twice() {
        for (lm, t1, t2) in [(0u32, 3.0f32, 5.0f32), (2, 2.0, 4.0)] {
            let mut energy = vec![0.0f32; NUM_BANDS];
            let peak = 7usize;
            // D_peak = 2*E with flat zero neighbors.
            // Just over t1 (but not t2): one quantum.
            energy[peak] = (t1 + 0.1) / 2.0;
            let boosts = choose_band_boosts(&energy, lm, 0, NUM_BANDS).unwrap();
            let n = BAND_BINS_LM[lm as usize][peak];
            assert_eq!(boosts[peak], boost_quanta_8th(n), "lm={lm} once");
            // Neighbors see D = -E_peak < 0: never boosted.
            assert_eq!(boosts[peak - 1], 0);
            assert_eq!(boosts[peak + 1], 0);
            // Just over t2: two quanta.
            energy[peak] = (t2 + 0.1) / 2.0;
            let boosts = choose_band_boosts(&energy, lm, 0, NUM_BANDS).unwrap();
            assert_eq!(boosts[peak], 2 * boost_quanta_8th(n), "lm={lm} twice");
            // At exactly t1 the strict `>` keeps the boost off.
            energy[peak] = t1 / 2.0;
            let boosts = choose_band_boosts(&energy, lm, 0, NUM_BANDS).unwrap();
            assert_eq!(boosts[peak], 0, "lm={lm} strict threshold");
        }
    }

    /// Edge bands are never boosted (D_j needs both neighbors), even
    /// with an extreme edge peak; a Hybrid window applies the same
    /// rule relative to its own edges and indexes the absolute band
    /// widths.
    #[test]
    fn edges_and_hybrid_window() {
        let mut energy = vec![0.0f32; NUM_BANDS];
        energy[0] = 100.0;
        energy[NUM_BANDS - 1] = 100.0;
        let boosts = choose_band_boosts(&energy, 2, 0, NUM_BANDS).unwrap();
        assert_eq!(boosts[0], 0);
        assert_eq!(boosts[NUM_BANDS - 1], 0);
        // Band 1 / band 19 see a huge *negative* contrast next to the
        // edge peaks: still zero.
        assert_eq!(boosts[1], 0);
        assert_eq!(boosts[NUM_BANDS - 2], 0);

        // Hybrid window 17..21: peak at absolute band 18 (window
        // index 1) uses band 18's width at LM=3.
        let window = vec![0.0f32, 3.0, 0.0, 0.0];
        let boosts = choose_band_boosts(&window, 3, 17, NUM_BANDS).unwrap();
        let n = BAND_BINS_LM[3][18];
        assert_eq!(boosts, vec![0, 2 * boost_quanta_8th(n), 0, 0]);
    }

    /// Input validation: bad lm / window / length are rejected; tiny
    /// windows have no interior bands.
    #[test]
    fn validation_and_tiny_windows() {
        let e = vec![0.0f32; NUM_BANDS];
        assert!(choose_band_boosts(&e, 4, 0, NUM_BANDS).is_none());
        assert!(choose_band_boosts(&e, 0, 5, 3).is_none());
        assert!(choose_band_boosts(&e, 0, 0, NUM_BANDS + 1).is_none());
        assert!(choose_band_boosts(&e[..5], 0, 0, NUM_BANDS).is_none());
        // Windows of width 0/1/2 have no interior band: all zero.
        assert_eq!(choose_band_boosts(&[], 1, 3, 3).unwrap(), Vec::<i32>::new());
        assert_eq!(choose_band_boosts(&[9.0], 1, 3, 4).unwrap(), vec![0]);
        assert_eq!(
            choose_band_boosts(&[9.0, 9.0], 1, 3, 5).unwrap(),
            vec![0, 0]
        );
    }
}
