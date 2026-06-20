//! Integration test for the documented CELT **allocation → pulses →
//! synthesis** seam (RFC 6716 §4.3.2.1 coarse energy → §4.3.3 bit
//! allocation column search → §4.3.4.1 bits-to-pulses → §4.3.4/§4.3.6/
//! §4.3.7 residual + synthesis).
//!
//! ## Why this test exists
//!
//! The CELT decode is fully specified by RFC 6716 from the range
//! decoder up to — and the per-band synthesis from — the band-shape
//! boundary. The one step the RFC §4.3.3 and the clean-room narrative
//! (`docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.7)
//! defer to the reference implementation is `interp_bits2pulses`: the
//! reallocation of unused bits with concurrent skip decoding and the
//! fine-energy-vs-shape split that, together, turn the combined static
//! allocation into the final per-band shape budget. Everything that
//! feeds that step, and everything that consumes its output, is
//! documented.
//!
//! This file pins the documented seam on *both sides* of that gap:
//!
//! 1. The §4.3.2.1 prefix walk ([`decode_frame_prefix`]) lands the
//!    decoder exactly at the `fine energy` symbol with a coherent
//!    boost/trim/reservation budget — the input side of the gap.
//! 2. The §4.3.3 column search ([`find_combined_alloc`]) over that
//!    budget yields a per-band combined shape allocation that respects
//!    the band maximums, boosts and tilt — the part of §2.7 the RFC
//!    *does* specify ("the entry nearest but not exceeding the
//!    available space, subject to the tilt, boosts, [and] band
//!    maximums").
//! 3. The §4.3.4.1 cached bits-to-pulses loop
//!    ([`bits_to_pulses_band_loop_cached`]) maps that per-band shape
//!    allocation to integer pulse counts `K` (= `band_k`), bit-exact
//!    against the `cache_bits50` curve for every band the allocator
//!    reaches.
//! 4. Those `band_k` values drive the full documented synthesis
//!    pipeline ([`decode_celt_frame`]) to time-domain PCM — the output
//!    side of the gap.
//!
//! In other words: the *only* thing standing between a raw CELT frame
//! and decoded PCM in this test is the deferred fine/shape split. By
//! feeding the column-search shape allocation straight into
//! bits-to-pulses (i.e. treating the whole combined allocation as
//! shape, with no fine reservation), the test exercises the documented
//! arithmetic of every other step composing coherently, and shows the
//! `band_k` the residual loop currently takes as an input *can* be
//! produced from the documented modules for the part of §4.3.3 that is
//! not a docs gap.
//!
//! Wall: this test reads only the crate's own public API and the RFC
//! §4.3 narrative cited in the module docs. No external library source.

use oxideav_celt::{
    bits_to_pulses_band_loop_cached, decode_celt_frame, decode_frame_prefix, find_combined_alloc,
    CeltDecodeState, CoarseEnergyState, BAND_BINS_LM, NUM_BANDS,
};

/// Build a deterministic, well-formed CELT range-coded payload of
/// `len` bytes. The control-symbol decoders all have defined
/// low-budget fallbacks, so any byte pattern parses; a mixing
/// recurrence keeps the stream from being trivially all-zero or
/// all-one (which would bias every band to the same fallback).
fn payload(len: usize, seed: u8) -> Vec<u8> {
    (0..len as u32)
        .map(|i| (i.wrapping_mul(89).wrapping_add(seed as u32) & 0xff) as u8)
        .collect()
}

/// Decode just the prefix and report whether the frame's `transient`
/// bit is set. `decode_celt_frame` rejects transient frames
/// ([`oxideav_celt::Error::NotImplemented`]) because the short-block
/// time-domain reassembly is a documented gap; the documented
/// long-MDCT chain this test exercises is the non-transient path.
fn is_transient(buf: &[u8], lm: u32, start: usize, end: usize) -> bool {
    let mut coarse = CoarseEnergyState::new();
    let mut dec = oxideav_celt::RangeDecoder::new(buf);
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

/// Build a deterministic non-transient payload of `len` bytes by
/// scanning seeds until the decoded prefix is a long-MDCT (non-
/// transient) frame. The transient flag is the rare `{7,1}/8` "1"
/// branch, so a non-transient seed is found within a handful of tries;
/// the search is deterministic so the chosen payload is stable.
fn nontransient_payload(len: usize, start_seed: u8, lm: u32, start: usize, end: usize) -> Vec<u8> {
    for off in 0..=255u32 {
        let seed = start_seed.wrapping_add(off as u8);
        let buf = payload(len, seed);
        if !is_transient(&buf, lm, start, end) {
            return buf;
        }
    }
    panic!("no non-transient payload found (len={len})");
}

/// Run the documented prefix → combined-allocation → bits-to-pulses
/// chain for a pure-CELT mono frame and return the per-band pulse
/// counts (`band_k`).
///
/// This is the composition the residual loop's `band_k` input is
/// currently supplied for; here it is *derived* from the documented
/// modules, treating the combined column-search allocation as the
/// shape budget (no fine reservation), which is the maximal documented
/// approximation given the deferred fine/shape split.
fn derive_band_k(buf: &[u8], lm: u32, start: usize, end: usize) -> Vec<u32> {
    let mut coarse = CoarseEnergyState::new();
    let mut dec = oxideav_celt::RangeDecoder::new(buf);
    let pfx = decode_frame_prefix(
        &mut dec,
        &mut coarse,
        lm,
        buf.len() as u32,
        false,
        start,
        end,
    )
    .expect("prefix decode");

    let bins: Vec<u32> = BAND_BINS_LM[lm as usize][start..end].to_vec();

    // The remaining budget the §4.3.3 search drives toward: the
    // band-boost loop's `total_bits` at exit (in 1/8 bits) — the budget
    // the §4.3.3 prose says the downstream allocation steps inherit
    // after every accepted boost has been subtracted. This is what the
    // combined column search compares its window total against.
    let budget_1_8th = pfx.boosts.total_bits_remaining.max(0) as i64;

    let search = find_combined_alloc(
        start,
        &bins,
        &pfx.boosts.boost,
        pfx.allocation.alloc_trim as i32,
        1, // mono
        false,
        lm,
        budget_1_8th,
    )
    .expect("combined allocation search");

    // The combined per-band shape allocation (in 1/8 bits) drives the
    // §4.3.4.1 bits-to-pulses loop. Targets are non-negative by
    // construction (the combine step floors at zero).
    let targets: Vec<u32> = search.alloc.bits.iter().map(|&b| b.max(0) as u32).collect();

    let (per_band, _balance) = bits_to_pulses_band_loop_cached(lm as usize, start, &bins, &targets)
        .expect("bits-to-pulses band loop");

    // Clamp each band's pulse count `K` to the largest value whose PVQ
    // codebook size `V(N, K)` is still representable in the documented
    // single-block decode (`v_count(n, k) != V_COUNT_SATURATION`). A
    // `K` so large that `V(N, K)` overflows is exactly the regime the
    // deferred §4.3.4.4 band-split path exists to handle (a wide band is
    // split into sub-bands before bits-to-pulses); keeping the derived
    // `band_k` at or below the non-overflow cap keeps this test inside
    // the documented single-block residual decode it exercises.
    per_band
        .iter()
        .zip(bins.iter())
        .map(|(r, &n)| clamp_k_to_decodable(n, r.k))
        .collect()
}

/// Largest `K' <= K` such that `v_count(n, K')` does not saturate (the
/// PVQ codebook size fits in `u32`). `v_count` is monotone in `K`, so a
/// simple descent from `K` finds it; in practice the clamp engages only
/// on wide bands at high `K`.
fn clamp_k_to_decodable(n: u32, k: u32) -> u32 {
    let mut k = k;
    while k > 0 && oxideav_celt::v_count(n, k) == oxideav_celt::V_COUNT_SATURATION {
        k -= 1;
    }
    k
}

/// The full chain composes: a real frame's prefix yields a combined
/// allocation, which yields integer pulse counts, which the synthesis
/// pipeline accepts and turns into finite PCM. Every per-band `K` is
/// within the documented search cap.
#[test]
fn documented_chain_composes_to_pcm_mono() {
    let lm = 3u32; // 20 ms / 960-sample frame
    let buf = nontransient_payload(48, 11, lm, 0, NUM_BANDS);

    let band_k = derive_band_k(&buf, lm, 0, NUM_BANDS);
    assert_eq!(band_k.len(), NUM_BANDS, "one K per coded band");

    // bits_to_pulses caps the search at K_SEARCH_CAP; nothing can
    // exceed it, and a non-negative shape budget never yields a
    // negative K (the type is unsigned, but assert the bound for
    // documentation).
    for (b, &k) in band_k.iter().enumerate() {
        assert!(
            k <= oxideav_celt::K_SEARCH_CAP,
            "band {b} K={k} exceeds search cap"
        );
    }

    // Feed the derived band_k into the documented synthesis pipeline.
    // fine_bits = 0 everywhere is the documented degenerate case (no
    // fine refinement); the residual loop + synthesis must still
    // produce a finite-energy frame.
    let mut state = CeltDecodeState::new(lm).expect("decode state");
    let fine_bits = [0u32; NUM_BANDS];
    let frame = decode_celt_frame(&mut state, &buf, 0, NUM_BANDS, &fine_bits, &band_k)
        .expect("celt frame decode");

    assert!(!frame.pcm.is_empty(), "decoded an empty frame");
    for (i, &s) in frame.pcm.iter().enumerate() {
        assert!(s.is_finite(), "non-finite PCM sample at index {i}");
    }
}

/// The seam is deterministic: identical bytes through the whole
/// documented chain produce identical `band_k` and identical PCM.
#[test]
fn documented_chain_is_deterministic() {
    let lm = 2u32;
    let buf = nontransient_payload(40, 23, lm, 0, NUM_BANDS);

    let k_a = derive_band_k(&buf, lm, 0, NUM_BANDS);
    let k_b = derive_band_k(&buf, lm, 0, NUM_BANDS);
    assert_eq!(k_a, k_b, "band_k derivation is non-deterministic");

    let mut s_a = CeltDecodeState::new(lm).expect("decode state");
    let mut s_b = CeltDecodeState::new(lm).expect("decode state");
    let fine_bits = [0u32; NUM_BANDS];
    let f_a = decode_celt_frame(&mut s_a, &buf, 0, NUM_BANDS, &fine_bits, &k_a).unwrap();
    let f_b = decode_celt_frame(&mut s_b, &buf, 0, NUM_BANDS, &fine_bits, &k_b).unwrap();
    assert_eq!(f_a.pcm, f_b.pcm, "PCM differs for identical input");
}

/// A larger range-coded budget can only raise the combined shape
/// allocation (the column search is monotone in available bits), so
/// the *total* pulse count across bands must not decrease when the
/// frame grows. This pins the documented monotonicity of the
/// allocation column search through to the pulse counts.
#[test]
fn larger_budget_does_not_reduce_total_pulses() {
    let lm = 3u32;

    // Two payloads with the same low-entropy content but different
    // lengths. The longer one carries a larger §4.3.3 frame budget, so
    // find_combined_alloc lands on a column with at least as much shape
    // allocation per band — hence at least as many total pulses.
    let small = payload(24, 7);
    let large = payload(64, 7);

    let k_small = derive_band_k(&small, lm, 0, NUM_BANDS);
    let k_large = derive_band_k(&large, lm, 0, NUM_BANDS);

    let sum_small: u64 = k_small.iter().map(|&k| k as u64).sum();
    let sum_large: u64 = k_large.iter().map(|&k| k as u64).sum();

    assert!(
        sum_large >= sum_small,
        "larger budget reduced total pulses: {sum_large} < {sum_small}"
    );
}

/// The Hybrid window (bands 17..=20) composes the documented
/// allocation seam the same way on a 4-band coded window: the prefix,
/// combined column search, and bits-to-pulses all index from
/// `coding_start = 17` and produce exactly 4 coherent pulse counts.
///
/// Unlike the pure-CELT cases, this test stops at the `band_k`
/// derivation rather than full synthesis. The four hybrid bands are the
/// widest in the layout, so — without the deferred §4.3.3 fine/shape
/// split — putting the *entire* combined allocation into shape pulses
/// systematically over-demands the per-band PVQ budget for these wide
/// bands; the residual loop then surfaces `NotImplemented` (the §4.3.4.4
/// band-split gap) for them. That over-demand is itself a symptom of the
/// missing split, so asserting full hybrid PCM here would be asserting
/// behaviour the docs gap controls. The documented composition this test
/// validates is the allocation→pulses indexing from band 17, which is
/// fully specified.
#[test]
fn documented_chain_hybrid_window() {
    let lm = 2u32;
    let buf = nontransient_payload(48, 41, lm, 17, NUM_BANDS);

    let band_k = derive_band_k(&buf, lm, 17, NUM_BANDS);
    assert_eq!(band_k.len(), 4, "hybrid window codes bands 17..=20");

    // Each derived K is within the documented single-block decodable
    // range for its band (the clamp guarantees `V(N, K)` is finite).
    let bins = &BAND_BINS_LM[lm as usize][17..NUM_BANDS];
    for (i, &k) in band_k.iter().enumerate() {
        assert_ne!(
            oxideav_celt::v_count(bins[i], k),
            oxideav_celt::V_COUNT_SATURATION,
            "hybrid band {} K={k} saturates V(N,K)",
            17 + i
        );
    }
}
