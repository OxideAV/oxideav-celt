//! Roundtrip + anchor tests for the §4.3.2.1 Laplace decoder and the
//! coarse-energy decode walk.
//!
//! The strongest validation of a decode recurrence is inverting the
//! matching encoder bit-for-bit. This file therefore carries a
//! TEST-ONLY range encoder transcribed from RFC 6716 Appendix A
//! `entenc.c` plus the `ec_laplace_encode` routine from RFC 6716
//! Appendix A `laplace.c` (both part of the staged spec at
//! `docs/audio/opus/rfc6716-opus.txt`, extracted per §A.1). Every
//! encoded stream must decode back to the encoder-committed values
//! through the crate's public decode API, and the §4.1.6 bit-usage
//! accounting must agree between the two sides at every symbol.
//!
//! The encoder here is deliberately NOT part of the crate's public
//! API: the crate is decoder-first, and the encoder subset below
//! implements only what the §4.3.2.1 coarse-energy path needs
//! (`ec_encode_bin`, `ec_enc_bit_logp`, `ec_enc_icdf`,
//! `ec_enc_done`).

use oxideav_celt::{
    decode_coarse_energy, ec_laplace_decode, CoarseEnergyState, RangeDecoder, BETA_COEF_Q15,
    E_PROB_MODEL, INTRA_BETA_Q15, LAPLACE_LOG_MINP, LAPLACE_MINP, LAPLACE_NMIN, MAX_CHANNELS,
    NUM_BANDS, PRED_COEF_Q15, PRED_INTER, PRED_INTRA, SMALL_ENERGY_ICDF,
};

// ---- test-only range encoder (RFC 6716 Appendix A entenc.c) ----

const EC_SYM_BITS: u32 = 8;
const EC_CODE_BITS: u32 = 32;
const EC_SYM_MAX: u32 = (1 << EC_SYM_BITS) - 1;
const EC_CODE_SHIFT: u32 = EC_CODE_BITS - EC_SYM_BITS - 1;
const EC_CODE_TOP: u32 = 1u32 << (EC_CODE_BITS - 1);
const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;

struct RangeEncoder {
    buf: Vec<u8>,
    offs: usize,
    end_offs: usize,
    nbits_total: u32,
    rng: u32,
    val: u32,
    ext: u32,
    rem: i32,
    error: bool,
}

impl RangeEncoder {
    fn new(size: usize) -> Self {
        Self {
            buf: vec![0u8; size],
            offs: 0,
            end_offs: 0,
            // ec_tell() subtracts partial bits from this offset.
            nbits_total: EC_CODE_BITS + 1,
            rng: EC_CODE_TOP,
            val: 0,
            ext: 0,
            rem: -1,
            error: false,
        }
    }

    fn write_byte(&mut self, value: u32) {
        if self.offs + self.end_offs >= self.buf.len() {
            self.error = true;
            return;
        }
        self.buf[self.offs] = value as u8;
        self.offs += 1;
    }

    /// Output a symbol with carry propagation; 0xFF symbols are
    /// buffered until the carry can be resolved.
    fn carry_out(&mut self, c: i32) {
        if c as u32 != EC_SYM_MAX {
            let carry = c >> EC_SYM_BITS;
            if self.rem >= 0 {
                let v = (self.rem + carry) as u32;
                self.write_byte(v);
            }
            if self.ext > 0 {
                let sym = (EC_SYM_MAX + carry as u32) & EC_SYM_MAX;
                while self.ext > 0 {
                    self.write_byte(sym);
                    self.ext -= 1;
                }
            }
            self.rem = (c as u32 & EC_SYM_MAX) as i32;
        } else {
            self.ext += 1;
        }
    }

    fn normalize(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.carry_out((self.val >> EC_CODE_SHIFT) as i32);
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total += EC_SYM_BITS;
        }
    }

    /// `ec_encode_bin(fl, fh, bits)` for a power-of-two total
    /// frequency `1 << bits`.
    fn encode_bin(&mut self, fl: u32, fh: u32, bits: u32) {
        let r = self.rng >> bits;
        if fl > 0 {
            self.val += self.rng - r * ((1u32 << bits) - fl);
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * ((1u32 << bits) - fh);
        }
        self.normalize();
    }

    /// `ec_enc_bit_logp(val, logp)`: probability of a "one" is
    /// `1 / (1 << logp)`.
    fn enc_bit_logp(&mut self, bit: bool, logp: u32) {
        let r = self.rng;
        let l = self.val;
        let s = r >> logp;
        let r = r - s;
        if bit {
            self.val = l + r;
        }
        self.rng = if bit { s } else { r };
        self.normalize();
    }

    /// `ec_enc_icdf(s, icdf, ftb)`.
    fn enc_icdf(&mut self, s: usize, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            self.val += self.rng - r * (icdf[s - 1] as u32);
            self.rng = r * ((icdf[s - 1] - icdf[s]) as u32);
        } else {
            self.rng -= r * (icdf[s] as u32);
        }
        self.normalize();
    }

    /// `ec_tell()`: `nbits_total - ilog(rng)`, identical on both
    /// sides of the coder per §4.1.6.1.
    fn tell(&self) -> u32 {
        self.nbits_total - (32 - self.rng.leading_zeros())
    }

    /// `ec_enc_done()`: flush the minimum number of bits that
    /// disambiguates the encoded symbols, then return the frame.
    fn done(mut self) -> Vec<u8> {
        let mut l: i32 = EC_CODE_BITS as i32 - (32 - self.rng.leading_zeros()) as i32;
        let mut msk: u32 = (EC_CODE_TOP - 1) >> l;
        let mut end = self.val.wrapping_add(msk) & !msk;
        if ((end | msk) as u64) >= self.val as u64 + self.rng as u64 {
            l += 1;
            msk >>= 1;
            end = self.val.wrapping_add(msk) & !msk;
        }
        while l > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1);
            l -= EC_SYM_BITS as i32;
        }
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }
        assert!(!self.error, "test encoder overran its buffer");
        self.buf
    }
}

// ---- test-only Laplace encoder (RFC 6716 Appendix A laplace.c) ----

/// Frequency span of the magnitude-1 pair (same formula the decoder
/// uses; reproduced here because the crate keeps it private).
fn laplace_get_freq1(fs0: u32, decay: u32) -> u32 {
    let ft = 32768 - LAPLACE_MINP * (2 * LAPLACE_NMIN) - fs0;
    (ft * (16384 - decay)) >> 15
}

/// `ec_laplace_encode`. May clamp `*value` when the magnitude falls
/// beyond the representable range; the committed value is what the
/// decoder must reproduce.
fn ec_laplace_encode(enc: &mut RangeEncoder, value: &mut i32, fs0: u32, decay: u32) {
    let mut fl: u32 = 0;
    let mut fs = fs0;
    let mut val = *value;
    if val != 0 {
        let s: i32 = if val < 0 { -1 } else { 0 };
        val = (val + s) ^ s;
        fl = fs;
        fs = laplace_get_freq1(fs0, decay);
        // Search the decaying part of the PDF.
        let mut i: i32 = 1;
        while fs > 0 && i < val {
            fs *= 2;
            fl += fs + 2 * LAPLACE_MINP;
            fs = (fs * decay) >> 15;
            i += 1;
        }
        // Everything beyond that has probability LAPLACE_MINP.
        if fs == 0 {
            let ndi_max: i32 = ((32768 - fl + LAPLACE_MINP - 1) >> LAPLACE_LOG_MINP) as i32;
            let ndi_max = (ndi_max - s) >> 1;
            let di = (val - i).min(ndi_max - 1);
            fl += ((2 * di + 1 + s) as u32) * LAPLACE_MINP;
            fs = LAPLACE_MINP.min(32768 - fl);
            *value = (i + di + s) ^ s;
        } else {
            fs += LAPLACE_MINP;
            if s == 0 {
                fl += fs;
            }
        }
        assert!(fl + fs <= 32768);
        assert!(fs > 0);
    }
    enc.encode_bin(fl, fl + fs, 15);
}

// ---- coarse-energy encode mirror (RFC 6716 Appendix A quant_bands.c) ----

/// Encode one frame's coarse prediction errors with the same
/// budget-keyed dispatch `decode_coarse_energy` uses, returning the
/// frame bytes plus the per-slot values actually committed (after
/// any clamping by the fallbacks or the Laplace tail).
#[allow(clippy::too_many_arguments)]
fn encode_coarse(
    qis: &[[i32; NUM_BANDS]; MAX_CHANNELS],
    intra: bool,
    lm: usize,
    start: usize,
    end: usize,
    channels: usize,
    storage: usize,
) -> (Vec<u8>, [[i32; NUM_BANDS]; MAX_CHANNELS]) {
    let pred = if intra { PRED_INTRA } else { PRED_INTER };
    let mut enc = RangeEncoder::new(storage);
    let budget = (storage as u32) * 8;
    let mut committed = [[-1i32; NUM_BANDS]; MAX_CHANNELS];
    for band in start..end {
        for c in 0..channels {
            let bits_left = budget.saturating_sub(enc.tell());
            let mut qi = qis[c][band];
            if bits_left >= 15 {
                let pd = E_PROB_MODEL[lm][pred][band];
                ec_laplace_encode(
                    &mut enc,
                    &mut qi,
                    (pd.prob as u32) << 7,
                    (pd.decay as u32) << 6,
                );
            } else if bits_left >= 2 {
                qi = qi.clamp(-1, 1);
                let s = (2 * qi) ^ -((qi < 0) as i32);
                enc.enc_icdf(s as usize, &SMALL_ENERGY_ICDF, 2);
            } else if bits_left >= 1 {
                qi = qi.clamp(-1, 0);
                enc.enc_bit_logp(qi == -1, 1);
            } else {
                qi = -1;
            }
            committed[c][band] = qi;
        }
    }
    (enc.done(), committed)
}

/// Independent evaluation of the Appendix A reconstruction recursion,
/// mirroring `decode_coarse_energy`'s f32 arithmetic operation for
/// operation.
fn reconstruct(
    prev: &[[f32; NUM_BANDS]; MAX_CHANNELS],
    qis: &[[i32; NUM_BANDS]; MAX_CHANNELS],
    intra: bool,
    lm: usize,
    start: usize,
    end: usize,
    channels: usize,
) -> [[f32; NUM_BANDS]; MAX_CHANNELS] {
    let (coef, beta) = if intra {
        (0.0_f32, INTRA_BETA_Q15 as f32 / 32768.0)
    } else {
        (
            PRED_COEF_Q15[lm] as f32 / 32768.0,
            BETA_COEF_Q15[lm] as f32 / 32768.0,
        )
    };
    let mut out = *prev;
    let mut running = [0.0_f32; MAX_CHANNELS];
    for band in start..end {
        for c in 0..channels {
            let q = qis[c][band] as f32;
            let old = out[c][band].max(-9.0);
            out[c][band] = coef * old + running[c] + q;
            running[c] += q - beta * q;
        }
    }
    out
}

// ---- tests ----

/// Roundtrip every `(prob, decay)` cell of `E_PROB_MODEL`: encode a
/// fixed pattern of small signed values (one per band, with the
/// band's own parameters), then decode and compare. Also pins the
/// §4.1.6 invariant that encoder and decoder report identical
/// `tell()` trajectories over the same symbol sequence.
#[test]
fn laplace_roundtrip_every_prob_model_cell() {
    let pattern: [i32; NUM_BANDS] = [
        0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 0, 1, -2, 3, -4, 5, -6, 0,
    ];
    for (lm, lm_row) in E_PROB_MODEL.iter().enumerate() {
        for pred in [PRED_INTER, PRED_INTRA] {
            let mut enc = RangeEncoder::new(128);
            let mut committed = [0i32; NUM_BANDS];
            let mut enc_tells = Vec::with_capacity(NUM_BANDS);
            for (band, pd) in lm_row[pred].iter().enumerate() {
                let mut v = pattern[band];
                ec_laplace_encode(
                    &mut enc,
                    &mut v,
                    (pd.prob as u32) << 7,
                    (pd.decay as u32) << 6,
                );
                committed[band] = v;
                enc_tells.push(enc.tell());
            }
            // Small values never hit the uniform-tail clamp.
            assert_eq!(committed, pattern, "lm={lm} pred={pred} clamped");
            let buf = enc.done();
            let mut dec = RangeDecoder::new(&buf);
            for (band, pd) in lm_row[pred].iter().enumerate() {
                let got =
                    ec_laplace_decode(&mut dec, (pd.prob as u32) << 7, (pd.decay as u32) << 6);
                assert_eq!(
                    got, committed[band],
                    "lm={lm} pred={pred} band={band} roundtrip mismatch"
                );
                assert_eq!(
                    dec.tell(),
                    enc_tells[band],
                    "lm={lm} pred={pred} band={band} tell() diverged"
                );
            }
            assert!(!dec.has_error());
        }
    }
}

/// Large magnitudes fall into the uniform `LAPLACE_MINP` tail; the
/// encoder clamps the committed value and the decoder must land on
/// exactly the committed (clamped) value, in both directions.
#[test]
fn laplace_roundtrip_uniform_tail_and_clamp() {
    // High decay → long geometric tail; low decay → quick handoff to
    // the uniform region. Cover both with extreme inputs.
    let params: [(u32, u32); 3] = [
        ((24u32) << 7, (179u32) << 6), // slow decay
        ((177u32) << 7, (11u32) << 6), // fast decay, heavy zero
        ((90u32) << 7, (79u32) << 6),  // middling
    ];
    let values: [i32; 8] = [40, -40, 400, -400, 20_000, -20_000, 7, -7];
    for &(fs0, decay) in &params {
        let mut enc = RangeEncoder::new(256);
        let mut committed = [0i32; 8];
        for (k, &v0) in values.iter().enumerate() {
            let mut v = v0;
            ec_laplace_encode(&mut enc, &mut v, fs0, decay);
            committed[k] = v;
            // The clamp never changes the sign and never grows the
            // magnitude.
            assert_eq!(v.signum(), v0.signum());
            assert!(v.abs() <= v0.abs());
        }
        let buf = enc.done();
        let mut dec = RangeDecoder::new(&buf);
        for (k, &want) in committed.iter().enumerate() {
            let got = ec_laplace_decode(&mut dec, fs0, decay);
            assert_eq!(got, want, "symbol {k} fs0={fs0} decay={decay}");
        }
        assert!(!dec.has_error());
    }
}

/// Anchor: a fixed Laplace-coded byte stream and its decoded values,
/// pinned against regressions in either the range decoder or the
/// Laplace recurrence. The stream was produced by the Appendix A
/// `entenc.c`/`laplace.c` transcriptions in this file encoding
/// `[0, 1, -1, 3, -7, 12, 0, -2]` with `(fs0, decay) =
/// (9216, 8128)` (= `E_PROB_MODEL[0][inter][0]` scaled to
/// Q15/Q14).
#[test]
fn laplace_anchor_stream() {
    const FS0: u32 = 9216;
    const DECAY: u32 = 8128;
    const VALUES: [i32; 8] = [0, 1, -1, 3, -7, 12, 0, -2];
    let mut enc = RangeEncoder::new(32);
    for &v0 in &VALUES {
        let mut v = v0;
        ec_laplace_encode(&mut enc, &mut v, FS0, DECAY);
        assert_eq!(v, v0);
    }
    let buf = enc.done();
    // ANCHOR_BYTES pins the deterministic output of the encode above
    // (regenerated by the very encode that precedes this assertion,
    // so a failure means the coder pair drifted from this pinned
    // history, not that the constant is stale).
    const ANCHOR_BYTES: [u8; 5] = [0x27, 0x14, 0x5D, 0xDA, 0xEE];
    assert_eq!(
        &buf[..ANCHOR_BYTES.len()],
        &ANCHOR_BYTES[..],
        "encoded prefix drifted"
    );
    assert!(buf[ANCHOR_BYTES.len()..].iter().all(|&b| b == 0));
    let mut dec = RangeDecoder::new(&buf);
    for &want in &VALUES {
        assert_eq!(ec_laplace_decode(&mut dec, FS0, DECAY), want);
    }
    assert!(!dec.has_error());
}

/// Full coarse-energy roundtrip, mono, all four frame sizes, both
/// prediction modes: encode a per-band error pattern, decode through
/// the public `decode_coarse_energy`, and compare the reconstructed
/// envelope bit-exactly against the independent recursion.
#[test]
fn coarse_energy_roundtrip_mono_all_modes() {
    let mut qis = [[0i32; NUM_BANDS]; MAX_CHANNELS];
    qis[0] = std::array::from_fn(|band| ((band as i32 * 5) % 7) - 3); // -3..=3 pattern
    for lm in 0..4usize {
        for intra in [false, true] {
            let (buf, committed) = encode_coarse(&qis, intra, lm, 0, NUM_BANDS, 1, 64);
            // Only channel 0 is coded in mono; the generous budget
            // means no value was clamped.
            assert_eq!(committed[0], qis[0], "lm={lm} intra={intra} clamped");
            let mut state = CoarseEnergyState::new();
            // Seed a previous-frame envelope so the inter time arm
            // has something to predict from.
            state.energy[0] = std::array::from_fn(|band| band as f32 * 0.5 - 4.0);
            let seeded = state.energy;
            let mut dec = RangeDecoder::new(&buf);
            decode_coarse_energy(&mut dec, &mut state, intra, lm as u32, 0, NUM_BANDS, 1).unwrap();
            assert!(!dec.has_error());
            let expected = reconstruct(&seeded, &committed, intra, lm, 0, NUM_BANDS, 1);
            assert_eq!(
                state.energy[0], expected[0],
                "lm={lm} intra={intra} envelope mismatch"
            );
        }
    }
}

/// Stereo coarse-energy roundtrip with distinct per-channel errors:
/// the band-major / channel-minor interleave and the per-channel
/// frequency-arm predictors must both line up with the encoder.
#[test]
fn coarse_energy_roundtrip_stereo() {
    let qis: [[i32; NUM_BANDS]; MAX_CHANNELS] = [
        std::array::from_fn(|band| ((band as i32) % 5) - 2),
        std::array::from_fn(|band| 2 - ((band as i32) % 4)),
    ];
    let lm = 2usize;
    let (buf, committed) = encode_coarse(&qis, false, lm, 0, NUM_BANDS, 2, 96);
    assert_eq!(committed, qis);
    let mut state = CoarseEnergyState::new();
    state.energy[0][4] = 3.0;
    state.energy[1][4] = -20.0; // exercises the -9.0 floor
    let seeded = state.energy;
    let mut dec = RangeDecoder::new(&buf);
    decode_coarse_energy(&mut dec, &mut state, false, lm as u32, 0, NUM_BANDS, 2).unwrap();
    assert!(!dec.has_error());
    let expected = reconstruct(&seeded, &committed, false, lm, 0, NUM_BANDS, 2);
    assert_eq!(state.energy, expected);
}

/// Hybrid-window roundtrip: only bands 17..21 are coded, the rest of
/// the envelope stays untouched.
#[test]
fn coarse_energy_roundtrip_hybrid_window() {
    let mut qis = [[0i32; NUM_BANDS]; MAX_CHANNELS];
    qis[0][17] = 2;
    qis[0][18] = -1;
    qis[0][19] = 0;
    qis[0][20] = 1;
    let (buf, committed) = encode_coarse(&qis, true, 3, 17, NUM_BANDS, 1, 32);
    let mut state = CoarseEnergyState::new();
    state.energy[0] = [7.0; NUM_BANDS];
    let seeded = state.energy;
    let mut dec = RangeDecoder::new(&buf);
    decode_coarse_energy(&mut dec, &mut state, true, 3, 17, NUM_BANDS, 1).unwrap();
    let expected = reconstruct(&seeded, &committed, true, 3, 17, NUM_BANDS, 1);
    assert_eq!(state.energy, expected);
    for (band, &e) in state.energy[0].iter().take(17).enumerate() {
        assert_eq!(e, 7.0, "band {band} clobbered");
    }
}

/// Budget-starved roundtrip: a 4-byte frame forces the decode through
/// every fallback tier (full Laplace while ≥ 15 bits remain, then the
/// 2-bit zig-zag, then the 1-bit flag, then the implicit -1). The
/// encoder mirrors the same dispatch, so the committed values it
/// reports are exactly what the decoder must reconstruct.
#[test]
fn coarse_energy_budget_fallback_tiers() {
    let mut qis = [[0i32; NUM_BANDS]; MAX_CHANNELS];
    qis[0] = std::array::from_fn(|band| [0, 1, -1][band % 3]);
    let storage = 4usize;
    let (buf, committed) = encode_coarse(&qis, true, 0, 0, NUM_BANDS, 1, storage);
    assert_eq!(buf.len(), storage);
    // The tiny budget must actually have exhausted: at least one
    // band fell back to the implicit -1 with input != -1.
    let clamped = (0..NUM_BANDS).any(|b| committed[0][b] != qis[0][b]);
    assert!(clamped, "budget never exhausted — test premise broken");
    let mut state = CoarseEnergyState::new();
    let seeded = state.energy;
    let mut dec = RangeDecoder::new(&buf);
    decode_coarse_energy(&mut dec, &mut state, true, 0, 0, NUM_BANDS, 1).unwrap();
    let expected = reconstruct(&seeded, &committed, true, 0, 0, NUM_BANDS, 1);
    assert_eq!(state.energy[0], expected[0]);
}

/// Sweep the storage size from 0 to 24 bytes so the budget boundary
/// crosses every dispatch tier at every possible offset; encoder and
/// decoder must agree on the committed values throughout.
#[test]
fn coarse_energy_budget_boundary_sweep() {
    let qis: [[i32; NUM_BANDS]; MAX_CHANNELS] = [
        std::array::from_fn(|band| ((band as i32) % 3) - 1),
        std::array::from_fn(|band| 1 - ((band as i32) % 3)),
    ];
    for storage in 0..=24usize {
        let (buf, committed) = encode_coarse(&qis, false, 1, 0, NUM_BANDS, 2, storage);
        let mut state = CoarseEnergyState::new();
        let seeded = state.energy;
        let mut dec = RangeDecoder::new(&buf);
        decode_coarse_energy(&mut dec, &mut state, false, 1, 0, NUM_BANDS, 2).unwrap();
        let expected = reconstruct(&seeded, &committed, false, 1, 0, NUM_BANDS, 2);
        assert_eq!(state.energy, expected, "storage={storage} mismatch");
    }
}
