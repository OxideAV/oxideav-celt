//! The cached-basis IMDCT (`imdct_cached_f32`, r454) must be
//! **bit-identical** to the direct form on every size — cached
//! (standard-mode allowlist) and fallback alike. The cache stores
//! exactly the `f64` cosine factors the direct form computes and
//! keeps the accumulation order, so any bit difference here is a
//! real defect, not float noise.

use oxideav_celt::mdct::{imdct_cached_f32, imdct_naive_f32};

#[test]
fn cached_imdct_is_bit_identical() {
    let mut lcg = 0x1234_5678u32;
    // 120/240/480/960 take the cached path; 60 exercises the
    // fallback arm on a non-allowlisted geometry.
    for n in [120usize, 240, 480, 960, 60] {
        let mut spec = vec![0f32; n];
        for v in spec.iter_mut() {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((lcg >> 8) as i32 as f32) / (1 << 24) as f32;
        }
        let mut direct = vec![0f32; 2 * n];
        let mut cached = vec![0f32; 2 * n];
        assert!(imdct_naive_f32(&spec, &mut direct));
        assert!(imdct_cached_f32(&spec, &mut cached));
        for (i, (a, b)) in direct.iter().zip(&cached).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "n={n} sample {i}: {a} != {b} (bitwise)"
            );
        }
    }
}

#[test]
fn cached_imdct_rejects_bad_dimensions() {
    let spec = vec![0f32; 120];
    let mut bad = vec![0f32; 120];
    assert!(!imdct_cached_f32(&spec, &mut bad));
    let mut empty_out = vec![0f32; 0];
    assert!(!imdct_cached_f32(&[], &mut empty_out));
}
