//! §4.3.5 anti-collapse processing (RFC 6716).
//!
//! ## What the RFC specifies
//!
//! RFC 6716 §4.3.5 (`docs/audio/opus/rfc6716-opus.txt` lines
//! 6710–6729) specifies, for transient frames whose anti-collapse bit
//! is set:
//!
//! > the energy in each small MDCT is prevented from collapsing to
//! > zero. For each band of each MDCT where a collapse is detected, a
//! > pseudo-random signal is inserted with an energy corresponding to
//! > the minimum energy over the two previous frames. A
//! > renormalization step is then required to ensure that the
//! > anti-collapse step did not alter the energy preservation
//! > property.
//!
//! The bit's position (after the band shape vectors — Table 56) and
//! its `{1, 1}/2` PDF are specified, as is its §4.3.3 reservation gate
//! (transient, `LM > 1`, budget ≥ `(LM+2)*8` eighth-bits). The
//! **collapse-detection criterion, the pseudo-random generator, and
//! the exact injection/renormalization arithmetic are not** — they
//! are documented in-crate decoder decisions chosen to satisfy every
//! property the prose does pin:
//!
//! * **Detection**: a `(band, short-MDCT)` pair has collapsed iff
//!   every one of that block's decoded samples is exactly zero. The
//!   per-block samples are the interleaved lanes
//!   `samples[block + nb_blocks * j]` — the same lanes the §4.3.7
//!   short-block synthesis de-interleaves into time-domain blocks, so
//!   "the energy in each small MDCT" is measured on exactly the
//!   samples that become that short MDCT.
//! * **Injection energy**: the flat per-band energy
//!   `2^min(E_prev1, E_prev2)` ("the minimum energy over the two
//!   previous frames", base-2 log domain), capped at the band's
//!   current coded energy `2^E_cur` so a formerly-loud band cannot
//!   inject above its own envelope. Each collapsed block receives its
//!   `1/nb_blocks` share, spread flat over the block's samples.
//! * **Pseudo-random signal**: a sign sequence `±r` driven by a
//!   32-bit linear congruential generator
//!   (`seed = seed * 1664525 + 1013904223`, sign from the top bit) —
//!   an in-crate choice; any full-period generator satisfies the
//!   prose. The seed threads through the caller's decoder state so
//!   the output is deterministic for a given stream.
//! * **Renormalization**: after injection the whole band is rescaled
//!   so its L2 energy equals the coded envelope `2^E_cur` again — the
//!   "energy preservation property" (§4.3.6 denormalization pins each
//!   band's energy to its envelope; injection must not change that).
//!
//! ## Clean-room provenance
//!
//! The trigger condition, injection intent, minimum-of-two-frames
//! energy, and renormalization requirement are RFC 6716 §4.3.5; the
//! detection/PRNG/arithmetic details are documented in-crate decisions
//! within that prose (flagged above). No external library source was
//! consulted.

use crate::band_layout::band_bins;
use crate::coarse_energy::NUM_BANDS;

/// Log-2 energy floor used to initialize the per-band energy history
/// on a fresh / reset decoder state: low enough (−28 log-2 steps =
/// −168 dB) that a stream-opening transient injects effectively
/// nothing, per the §4.5.2 "decoder starts from silence" reading.
pub const ENERGY_HISTORY_FLOOR_LOG2: f32 = -28.0;

/// Advance the anti-collapse LCG and return the next sign (`+1.0` /
/// `-1.0`). The generator is the in-crate documented choice (see the
/// module docs); callers thread the seed through their decoder state.
#[inline]
fn next_sign(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    if *seed & 0x8000_0000 != 0 {
        -1.0
    } else {
        1.0
    }
}

/// Apply the §4.3.5 anti-collapse injection to a decoded (denormalized)
/// coded-window residual spectrum, in place.
///
/// Call only when the frame is transient **and** the decoded
/// anti-collapse bit is set. For each coded band, every short-MDCT
/// block whose samples are all zero receives a flat pseudo-random
/// `±r` fill at the minimum-of-two-previous-frames energy share, and
/// any band that received an injection is renormalized back to its
/// coded envelope energy (see the module docs for the documented
/// in-crate arithmetic).
///
/// Parameters:
///
/// * `samples` — the coded-window residual (the
///   [`ResidualSpectrum::samples`](crate::residual::ResidualSpectrum::samples)
///   layout: band-contiguous, in-band interleaved across the
///   `1 << lm` short blocks).
/// * `lm` — frame-size shift, `0..=3`; the short-block count is
///   `1 << lm`.
/// * `start` / `end` — the coded-band window.
/// * `env_q8` — the current frame's final per-band Q8 log-energies,
///   window-relative (`end - start` entries).
/// * `prev1` / `prev2` — the previous / before-previous frames' final
///   per-band log-2 energies on the absolute band axis (the history
///   the decoder state carries; fresh states hold
///   [`ENERGY_HISTORY_FLOOR_LOG2`]).
/// * `seed` — the pseudo-random state, advanced once per injected
///   sample.
///
/// Returns `false` (leaving `samples` untouched) on a geometry
/// mismatch: `lm > 3`, a bad window, or `samples` / `env_q8` lengths
/// disagreeing with the window.
#[allow(clippy::too_many_arguments)]
pub fn apply_anti_collapse(
    samples: &mut [f32],
    lm: u32,
    start: usize,
    end: usize,
    env_q8: &[i32],
    prev1: &[f32; NUM_BANDS],
    prev2: &[f32; NUM_BANDS],
    seed: &mut u32,
) -> bool {
    if lm > 3 || start > end || end > NUM_BANDS || env_q8.len() != end - start {
        return false;
    }
    match crate::band_layout::coded_total_bins(start, end, lm) {
        Some(total) if total as usize == samples.len() => {}
        _ => return false,
    }
    let nb_blocks = 1usize << lm;

    let mut offset = 0usize;
    for (i, band) in (start..end).enumerate() {
        let Some(n) = band_bins(band, lm) else {
            return false;
        };
        let n = n as usize;
        if offset + n > samples.len() {
            return false;
        }
        let band_samples = &mut samples[offset..offset + n];
        offset += n;

        let sub = n / nb_blocks;
        if sub == 0 || n % nb_blocks != 0 {
            return false;
        }

        // Current coded band energy (linear L2²) from the Q8 envelope:
        // 2^(E_q8 / 256).
        let e_cur = env_q8[i] as f32 / 256.0;
        // Injection energy: min over the two previous frames, capped at
        // the current envelope.
        let e_inj = prev1[band].min(prev2[band]).min(e_cur);
        // Flat per-sample amplitude for a band-total energy of 2^e_inj
        // spread over all n samples (each collapsed block gets its
        // 1/nb_blocks share automatically).
        let r = (0.5 * e_inj).exp2() / (n as f32).sqrt();

        let mut injected = false;
        for block in 0..nb_blocks {
            // The block's samples are the interleaved lane
            // `block + nb_blocks * j` (the lane the §4.3.7 short-block
            // synthesis turns into this short MDCT).
            let collapsed = (0..sub).all(|j| band_samples[block + nb_blocks * j] == 0.0);
            if !collapsed {
                continue;
            }
            for j in 0..sub {
                band_samples[block + nb_blocks * j] = r * next_sign(seed);
            }
            injected = true;
        }

        if injected {
            // Renormalize the band back to its coded envelope energy
            // (the §4.3.5 energy preservation property).
            let e2: f64 = band_samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
            if e2 > 0.0 {
                let target = (e_cur as f64).exp2();
                let scale = (target / e2).sqrt() as f32;
                for x in band_samples.iter_mut() {
                    *x *= scale;
                }
            }
        }
    }
    debug_assert_eq!(offset, samples.len());
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band_layout::coded_total_bins;

    fn band_slice(samples: &[f32], band: usize, lm: u32, start: usize) -> &[f32] {
        let mut offset = 0usize;
        for b in start..band {
            offset += band_bins(b, lm).unwrap() as usize;
        }
        let n = band_bins(band, lm).unwrap() as usize;
        &samples[offset..offset + n]
    }

    fn l2sq(s: &[f32]) -> f64 {
        s.iter().map(|&x| (x as f64) * (x as f64)).sum()
    }

    /// A fully-zero band (every block collapsed) is filled and lands
    /// exactly on its coded envelope energy.
    #[test]
    fn fully_collapsed_band_renormalizes_to_envelope() {
        let lm = 2u32;
        let (start, end) = (0usize, NUM_BANDS);
        let total = coded_total_bins(start, end, lm).unwrap() as usize;
        let mut samples = vec![0.0f32; total];
        let env: Vec<i32> = (0..end - start)
            .map(|b| 256 + (b as i32 % 3) * 128)
            .collect();
        let prev = [1.0f32; NUM_BANDS];
        let mut seed = 0x1234_5678u32;

        assert!(apply_anti_collapse(
            &mut samples,
            lm,
            start,
            end,
            &env,
            &prev,
            &prev,
            &mut seed
        ));

        for band in start..end {
            let slice = band_slice(&samples, band, lm, start);
            let target = (env[band - start] as f64 / 256.0).exp2();
            let got = l2sq(slice);
            assert!(
                (got - target).abs() <= target * 1e-4,
                "band {band}: energy {got} != envelope {target}"
            );
            // Every sample was filled (no zeros survive a full fill).
            assert!(slice.iter().all(|&x| x != 0.0), "band {band} kept zeros");
        }
    }

    /// A band with live content is left bit-identical when no block
    /// collapsed.
    #[test]
    fn live_band_untouched() {
        let lm = 1u32;
        let (start, end) = (0usize, NUM_BANDS);
        let total = coded_total_bins(start, end, lm).unwrap() as usize;
        // Nonzero everywhere ⇒ no block collapses anywhere.
        let mut samples: Vec<f32> = (0..total).map(|i| 0.01 + (i % 7) as f32 * 0.1).collect();
        let before = samples.clone();
        let env = vec![0i32; end - start];
        let prev = [0.0f32; NUM_BANDS];
        let mut seed = 42u32;

        assert!(apply_anti_collapse(
            &mut samples,
            lm,
            start,
            end,
            &env,
            &prev,
            &prev,
            &mut seed
        ));
        assert_eq!(samples, before);
        assert_eq!(seed, 42, "seed advanced with no injection");
    }

    /// Partial collapse: one block zeroed in a two-block band gets
    /// filled; the band's total energy is preserved (renormalized back
    /// to the envelope).
    #[test]
    fn partial_collapse_preserves_band_energy() {
        let lm = 1u32; // 2 blocks
        let (start, end) = (20usize, NUM_BANDS); // band 20 only (widest)
        let n = band_bins(20, lm).unwrap() as usize;
        // Block 0 (even interleave lanes) live at unit-norm-ish values,
        // block 1 (odd lanes) all zero.
        let mut samples = vec![0.0f32; n];
        let sub = n / 2;
        let amp = 1.0f32 / (sub as f32).sqrt();
        for j in 0..sub {
            samples[2 * j] = amp; // block 0 lane
        }
        // Envelope 0 ⇒ target band energy 1.0; live content already
        // carries 1.0.
        let env = vec![0i32; end - start];
        let mut prev = [ENERGY_HISTORY_FLOOR_LOG2; NUM_BANDS];
        prev[20] = -2.0; // prior band energy 2^-2 = 0.25
        let mut seed = 7u32;

        assert!(apply_anti_collapse(
            &mut samples,
            lm,
            start,
            end,
            &env,
            &prev,
            &prev,
            &mut seed
        ));

        // Block 1 lanes are now nonzero.
        assert!((0..sub).all(|j| samples[2 * j + 1] != 0.0));
        // Band energy is back at the envelope (1.0).
        let got = l2sq(&samples);
        assert!((got - 1.0).abs() <= 1e-4, "band energy {got} != 1.0");
    }

    /// The injection level follows the MINIMUM of the two previous
    /// frames: with renormalization, a quieter history means the
    /// injected block carries a smaller share of the band energy.
    #[test]
    fn quieter_history_injects_smaller_share() {
        let lm = 1u32;
        let (start, end) = (20usize, NUM_BANDS);
        let n = band_bins(20, lm).unwrap() as usize;
        let sub = n / 2;
        let amp = 1.0f32 / (sub as f32).sqrt();

        let run = |hist: f32| -> f64 {
            let mut samples = vec![0.0f32; n];
            for j in 0..sub {
                samples[2 * j] = amp;
            }
            let env = vec![0i32; end - start];
            let mut p1 = [ENERGY_HISTORY_FLOOR_LOG2; NUM_BANDS];
            let mut p2 = [ENERGY_HISTORY_FLOOR_LOG2; NUM_BANDS];
            p1[20] = 0.0; // one loud previous frame …
            p2[20] = hist; // … the min comes from the other.
            let mut seed = 99u32;
            assert!(apply_anti_collapse(
                &mut samples,
                lm,
                start,
                end,
                &env,
                &p1,
                &p2,
                &mut seed
            ));
            // Energy of the injected block (odd lanes).
            (0..sub)
                .map(|j| {
                    let x = samples[2 * j + 1] as f64;
                    x * x
                })
                .sum()
        };

        let loud = run(-1.0);
        let quiet = run(-6.0);
        assert!(
            quiet < loud,
            "quieter history should inject less: quiet {quiet} vs loud {loud}"
        );
    }

    /// Determinism: identical inputs and seed produce identical output;
    /// the seed advances so successive bands draw fresh signs.
    #[test]
    fn deterministic_for_fixed_seed() {
        let lm = 2u32;
        let (start, end) = (0usize, NUM_BANDS);
        let total = coded_total_bins(start, end, lm).unwrap() as usize;
        let env = vec![128i32; end - start];
        let prev = [0.5f32; NUM_BANDS];

        let mut a = vec![0.0f32; total];
        let mut b = vec![0.0f32; total];
        let mut seed_a = 0xDEAD_BEEFu32;
        let mut seed_b = 0xDEAD_BEEFu32;
        assert!(apply_anti_collapse(
            &mut a,
            lm,
            start,
            end,
            &env,
            &prev,
            &prev,
            &mut seed_a
        ));
        assert!(apply_anti_collapse(
            &mut b,
            lm,
            start,
            end,
            &env,
            &prev,
            &prev,
            &mut seed_b
        ));
        assert_eq!(a, b);
        assert_eq!(seed_a, seed_b);
        assert_ne!(seed_a, 0xDEAD_BEEF, "seed did not advance");
    }

    /// Geometry rejection: bad window / wrong envelope length leave the
    /// samples untouched.
    #[test]
    fn rejects_bad_geometry() {
        let mut samples = vec![0.0f32; 16];
        let prev = [0.0f32; NUM_BANDS];
        let mut seed = 1u32;
        // lm out of range.
        assert!(!apply_anti_collapse(
            &mut samples,
            4,
            0,
            NUM_BANDS,
            &[0i32; NUM_BANDS],
            &prev,
            &prev,
            &mut seed
        ));
        // Envelope length mismatch.
        assert!(!apply_anti_collapse(
            &mut samples,
            1,
            0,
            NUM_BANDS,
            &[0i32; 5],
            &prev,
            &prev,
            &mut seed
        ));
        // Samples shorter than the window needs.
        assert!(!apply_anti_collapse(
            &mut samples,
            1,
            0,
            NUM_BANDS,
            &[0i32; NUM_BANDS],
            &prev,
            &prev,
            &mut seed
        ));
    }
}
