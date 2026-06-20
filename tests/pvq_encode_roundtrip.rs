//! Integration test for the CELT **PVQ encode → decode** round-trip
//! (RFC 6716 §5.3.8.1 codeword search → §4.3.4.2 index encode → §4.3.4.2
//! index decode → unit-norm reconstruction).
//!
//! ## Why this test exists
//!
//! The per-band PVQ shape is the one place CELT carries the actual
//! spectral detail. The decode direction (index → integer codeword →
//! unit vector) is exercised throughout the crate's decode path. This
//! file pins the encode direction composing through the public API:
//!
//! 1. [`pvq_search`] quantizes a unit input vector `x` onto the §4.3.4.2
//!    codebook (every integer vector with `sum(|y|) == K`) using the
//!    §5.3.8.1 projection-plus-greedy method.
//! 2. [`encode_pulses_to_index`] maps that integer codeword `y` to its
//!    unique index `i ∈ [0, V(N, K))`.
//! 3. [`decode_index_to_pulses`] decodes `i` back to a codeword, which
//!    must equal `y` exactly (the §4.3.4.2 codeword↔index bijection).
//! 4. [`normalize_to_unit_l2`] places `y` on the unit hypersphere — the
//!    reconstruction the decoder feeds to the §4.3.4.3 spreading stage.
//!
//! The whole chain lives inside fully-specified §4.3.4.2 / §5.3.8.1
//! territory — the RFC names the search method and states implementers
//! MAY use any search yielding a valid codebook vector, and the index
//! arithmetic is a closed-form bijection.

use oxideav_celt::{
    decode_index_to_pulses, encode_pulses_to_index, encode_unit_shape, normalize_to_unit_l2,
    pvq_search, v_count,
};

/// L1 norm (pulse count) of an integer codeword.
fn l1(v: &[i32]) -> u32 {
    v.iter().map(|&x| x.unsigned_abs()).sum()
}

/// A small deterministic pseudo-random generator so the test sweep is
/// reproducible without a third-party RNG crate.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_f32(&mut self) -> f32 {
        // Numerical-Recipes-style LCG; we only need a deterministic
        // spread, not statistical quality.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map the top 24 bits to [-1, 1).
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

#[test]
fn encode_search_then_index_then_decode_recovers_codeword() {
    // For a sweep of random unit-ish input vectors and (N, K), the
    // search → index → decode chain must recover the SAME integer
    // codeword the search produced, and that codeword must have exactly
    // K pulses.
    let mut rng = Lcg::new(0x0123_4567_89ab_cdef);
    for n in 1..=12u32 {
        for k in 1..=8u32 {
            for _trial in 0..16 {
                let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
                let y = pvq_search(&x, n, k).expect("search must produce a codeword");
                assert_eq!(y.len(), n as usize);
                assert_eq!(l1(&y), k, "sum|y| != K at N={} K={}", n, k);

                let index = encode_pulses_to_index(&y, n, k)
                    .expect("a valid codeword must encode to an index");
                let v = v_count(n, k);
                assert!(
                    index < v,
                    "index {} out of range V({},{})={}",
                    index,
                    n,
                    k,
                    v
                );

                let decoded =
                    decode_index_to_pulses(index, n, k).expect("a valid index must decode");
                assert_eq!(
                    decoded, y,
                    "decode(encode(search(x))) != search(x) at N={} K={}",
                    n, k
                );
            }
        }
    }
}

#[test]
fn encode_unit_shape_matches_manual_composition() {
    // encode_unit_shape must equal the manual pvq_search +
    // encode_pulses_to_index composition for every input.
    let mut rng = Lcg::new(0xfeed_face_dead_beef);
    for n in 1..=10u32 {
        for k in 1..=6u32 {
            for _trial in 0..8 {
                let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
                let (index, pulses) = encode_unit_shape(&x, n, k).unwrap();
                let manual_y = pvq_search(&x, n, k).unwrap();
                let manual_index = encode_pulses_to_index(&manual_y, n, k).unwrap();
                assert_eq!(pulses, manual_y);
                assert_eq!(index, manual_index);
            }
        }
    }
}

#[test]
fn reconstructed_unit_vector_is_normalized_and_sign_consistent() {
    // The decoder normalizes the recovered codeword to unit L2 norm.
    // Confirm the encode → decode → normalize round-trip yields a unit
    // vector whose signs match the integer codeword (the shape the
    // §4.3.4.3 spreading stage consumes).
    let mut rng = Lcg::new(0x00c0_ffee_1234_5678);
    for n in 2..=10u32 {
        for k in 1..=6u32 {
            let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let (index, pulses) = encode_unit_shape(&x, n, k).unwrap();
            let decoded = decode_index_to_pulses(index, n, k).unwrap();
            assert_eq!(decoded, pulses);
            let unit = normalize_to_unit_l2(&decoded);
            // Unit L2 norm.
            let energy: f64 = unit.iter().map(|&v| (v as f64) * (v as f64)).sum();
            assert!(
                (energy - 1.0).abs() < 1e-5,
                "L2 norm not 1 at N={} K={}: {}",
                n,
                k,
                energy
            );
            // Sign consistency with the integer codeword.
            for (p, u) in pulses.iter().zip(unit.iter()) {
                let ps = p.signum();
                let us = if *u > 0.0 {
                    1
                } else if *u < 0.0 {
                    -1
                } else {
                    0
                };
                assert_eq!(ps, us, "sign mismatch on {:?} -> {:?}", p, u);
            }
        }
    }
}

#[test]
fn search_increases_alignment_over_zero_codeword() {
    // The quantized codeword should be at least as aligned with the
    // input as the trivial all-zero choice — i.e. the search yields a
    // positive (or zero) correlation when the input has any structure.
    let mut rng = Lcg::new(0x9e37_79b9_7f4a_7c15);
    for n in 2..=12u32 {
        for k in 1..=8u32 {
            let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
            let y = pvq_search(&x, n, k).unwrap();
            let xy: f64 = x
                .iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| xi as f64 * yi as f64)
                .sum();
            let yy: f64 = y.iter().map(|&yi| (yi * yi) as f64).sum();
            // A valid codeword always has K pulses, so yy > 0.
            assert!(yy > 0.0);
            // The greedy search never settles on a net-negative
            // correlation: aligning with the input is always at least as
            // good as the zero-correlation baseline.
            assert!(
                xy >= -1e-6,
                "search produced a net-negative correlation {} at N={} K={}",
                xy,
                n,
                k
            );
        }
    }
}
