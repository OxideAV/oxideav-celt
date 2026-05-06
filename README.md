# oxideav-celt

Pure-Rust **CELT** audio codec — the MDCT (music) path of Opus
(RFC 6716 §4.3). Encoder and decoder, bit-exact range coder, PVQ shape
coding, inverse MDCT, inter-frame energy prediction. Zero C
dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone. The full Opus stack (SILK + CELT +
hybrid, Ogg/Opus framing) lives in `oxideav-opus` and dispatches into
these same modules.

## Installation

```toml
[dependencies]
oxideav-core = "0.1"
oxideav-codec = "0.1"
oxideav-celt = "0.0"
```

## Quick use

```rust
use oxideav_celt::{CODEC_ID_STR, decoder::CeltDecoder, encoder::{CeltEncoder, FRAME_SAMPLES, SAMPLE_RATE}};
use oxideav_codec::{Decoder, Encoder};
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, SampleFormat, TimeBase};

let mut params = CodecParameters::audio(CodecId::new(CODEC_ID_STR));
params.channels = Some(1);
params.sample_rate = Some(SAMPLE_RATE);

let mut enc = CeltEncoder::new(&params)?;
let mut dec = CeltDecoder::new(&params)?;

let pcm = vec![0.0f32; FRAME_SAMPLES];
let mut bytes = Vec::with_capacity(pcm.len() * 4);
for s in &pcm { bytes.extend_from_slice(&s.to_le_bytes()); }
let frame = Frame::Audio(AudioFrame {
    format: SampleFormat::F32,
    channels: 1,
    sample_rate: SAMPLE_RATE,
    samples: FRAME_SAMPLES as u32,
    pts: None,
    time_base: TimeBase::new(1, SAMPLE_RATE as i64),
    data: vec![bytes],
});
enc.send_frame(&frame)?;
enc.flush()?;
while let Ok(pkt) = enc.receive_packet() {
    dec.send_packet(&pkt)?;
    while let Ok(decoded) = dec.receive_frame() {
        // `decoded` is Frame::Audio, SampleFormat::F32, 48 kHz, 960 samples/frame.
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

The codec is also discoverable through the `RuntimeContext` via
`oxideav_celt::register(&mut ctx)` (or, if you only have a bare
`CodecRegistry`, `oxideav_celt::register_codecs(&mut registry)`).

## Supported configuration

This crate targets the fullband subset of CELT — the same profile an
Opus encoder picks for "music" content. All four RFC-valid frame sizes
(LM=0..3) ship. Long-block and short-block (transient) coding are
supported at each frame size. Use `new_auto_lm(params, high_transient)`
to let the crate pick the optimal LM for your content.

| Parameter      | Value                                                     |
|----------------|-----------------------------------------------------------|
| Sample rate    | 48 kHz only                                               |
| Channels       | 1 (mono) or 2 (stereo, dual-stereo)                       |
| Frame size     | 960 (LM=3, 20 ms, default), 480 (LM=2, 10 ms), 240 (LM=1, 5 ms), 120 (LM=0, 2.5 ms) |
| Sample format  | `F32`, `F32P`, `S16`, `S16P` on input                    |
| Output format  | `F32`, interleaved                                        |
| Bandwidth      | Fullband (NB_EBANDS=21 bands)                             |

### Frame-size / bitrate guide

| LM | Frame   | Default mono | Default stereo | Use case                          |
|----|---------|--------------|----------------|-----------------------------------|
| 3  | 20 ms   | 64 kbit/s    | 102 kbit/s     | Music, speech — best efficiency   |
| 2  | 10 ms   | 96 kbit/s    | 154 kbit/s     | Low-latency voice / Opus Hybrid   |
| 1  |  5 ms   | 154 kbit/s   | 246 kbit/s     | Percussion, highly transient      |
| 0  | 2.5 ms  | 320 kbit/s   | 512 kbit/s     | Ultra-low-latency, live effects   |

The decoder accepts any packet that matches the encoder's profile, plus
the RFC 6716 §4.3 silence flag (packets with the silence bit set decode
to zeros) and §4.3 transient frames (decoded via 8 × 200→100 short
IMDCTs with stride-M coefficient interleaving and §4.3.5 anti-collapse
reservation + flag parsing). Frames carrying comb-post-filter parameters
still return `Error::Unsupported`.

### Intra vs inter frames

The encoder emits the first frame of a session as `intra=true` (no
valid prior state for energy prediction) and every subsequent frame as
`intra=false`, using the previous frame's quantised band energies as the
prediction base. This is the RFC 6716 §4.3.2.1 inter-frame path — it
saves bits on steady-state content, at the cost of state-drift exposure
if a packet is lost. Callers that need resync boundaries should
reinstantiate the encoder.

### Silence fast path

Input frames whose peak sample is below -90 dBFS (|s| < 1e-5) emit a
silence-only packet — just the range-coded silence bit — and wipe the
encoder's prediction state so the next audible frame re-learns from
zero. The decoder mirrors this: a silence header returns an all-zero
PCM frame and clears its OLA tail.

### Bitrate

The encoder uses a fixed per-frame byte budget: 160 bytes/frame
(~64 kbit/s) for mono and 256 bytes/frame (~102 kbit/s) for stereo.
These are the defaults libopus picks for the CELT music path at
20 ms/48 kHz; they leave enough headroom for the per-channel coarse/
fine energy overhead plus the PVQ shape bits.

## What's implemented

Following the RFC 6716 §4.3 section numbers:

- §4.1 range coder — bit-exact port of libopus `entdec.c` / `entenc.c`.
- §4.3 frame header — silence, post-filter, transient, intra symbols.
- §4.3.2.1 coarse band energy — Laplace-coded, inter/intra prediction.
- §4.3.2.2 fine band energy.
- §4.3.2.3 fine-energy finalise.
- §4.3.3 bit allocation — alloc table, trim, dynalloc boost, skip,
  intensity / dual-stereo flags, coded-bands split, fine/PVQ partition.
- §4.3.4 PVQ shape — tf_decode, spreading flag, theta-split band
  recursion, mono / dual-stereo, canonical codeword enumeration
  (`cwrs`), `exp_rotation`, collapse-mask extraction. Long AND short
  (`big_b = M`) block partitioning via Hadamard deinterleave / interleave.
- §4.3.5 anti-collapse — reserved and parsed on transient frames;
  the encoder inspects the per-band `collapse_masks` after PVQ shape
  coding and emits `anti_collapse_on = 1` when at least one coded band
  has a partial or full collapse (some short-block sub-windows received
  no pulses). The decoder runs the noise-floor injection when the flag
  is set.
- §4.3.6 denormalisation.
- §4.3.7 MDCT — forward (encoder) and inverse (decoder) via pre-twiddle
  + length-N/4 Bluestein FFT + post-twiddle. Transient frames run
  8 × 200→100 short sub-MDCTs with a 100-tap sin² window, interleaved
  at stride M into the 800-bin coefficient buffer. The encoder ships a
  `detect_transient` surrogate for libopus' `transient_analysis`: it
  splits the frame into 8 sub-blocks (one per LM=3 short-block window),
  computes per-block energy, and flips `transient=true` when the peak
  exceeds the **median** by `DEFAULT_TRANSIENT_THRESHOLD_DB` (15 dB).
  The peak/median ratio (rather than peak/min) is robust against
  fade-in / fade-out envelopes that would otherwise false-positive a
  silent edge sub-block as a transient.
- §4.3.7.1 comb pitch pre-/post-filter — both sides:
  - decoder runs `comb_filter` on the IMDCT+OLA output (with the
    crossfade between previous-frame and current-frame parameters);
  - encoder runs the matching pre-filter analyser (NCC autocorrelation
    pitch search + best-of-three tapset selection) and applies the
    in-place `comb_filter` with negated gains on the pre-emphasized
    PCM before MDCT, emitting the post-filter header for the decoder
    to invert. See `pitch_analysis::analyse_pitch`.
- §4.3.4.4 spread (rotation) parameter encoder
  (`encoder_decisions::spread_decision`): per-frame peak-to-RMS tonality
  score on the normalised shape picks `SPREAD_NONE` / `LIGHT` / `NORMAL`
  / `AGGRESSIVE`. Tonal content suppresses rotation, noise-like content
  (white noise, dense partials) maxes it out. Wired into both mono and
  stereo `encode_frame` paths.
- §4.3.3 dynalloc band boost (`encoder_decisions::pick_dynalloc_boost_band`):
  per-frame outlier detector picks at most one band whose log-energy
  exceeds the median by ~6 dB and emits one quanta of extra pulse budget
  via the `decode_bit_logp` boost loop. Mono only — stereo dynalloc is
  reserved (the picker still runs but stays unused) until the allocator
  bisection accounts for the dual-stereo offset doubling.

Static tables (transcribed from libopus `static_modes_float.h`):
`EBAND_5MS`, `E_PROB_MODEL`, `PRED_COEF` / `BETA_COEF` / `BETA_INTRA`,
`BAND_ALLOCATION`, `LOG2_FRAC_TABLE`, `LOGN400`, `CACHE_INDEX50`,
`CACHE_BITS50`, `CACHE_CAPS50`, `E_MEANS`, `SPREAD_ICDF`, `TRIM_ICDF`,
`TF_SELECT_TABLE`, `COMB_FILTER_TAPS`.

## What's not implemented

These paths are partially or fully missing. The codec runs end-to-end
without them, but bit-for-bit parity with libopus requires closing
every gap below.

- **Time-frequency change flags.** Per-band TF analysis (RFC §4.3.4.5 +
  §5.3.6) is implemented in `tf_analysis::tf_analysis`: a Viterbi-style
  search picks per-band `tf_change` values to minimise the L1-norm
  distortion proxy from the masking model, and emits the matching raw
  delta + `tf_select` bits the decoder reconstructs through
  `TF_SELECT_TABLE`. The non-transient mono long-block path applies the
  picked values end-to-end via the haar1 wrapping in
  `encoder_bands::encode_all_bands_mono`. **Scope still limited:**
  transient frames (short blocks), stereo, and the hybrid path emit the
  no-op decision (every band gets `tf_change = 0`) until the
  recombine + Hadamard interaction in those modes is wired through
  `quant_partition_enc`.
- **Dynalloc stereo path** still emits all-zero offsets; the mono path
  picks one outlier band per frame.
- **Intensity stereo.** Stereo uses dual-stereo only — L and R coded as
  two independent mono bands in one packet. Intensity stereo would buy
  extra HF bits on low bit-rate stereo; at 102 kbit/s the dual path
  produces well-separated L/R output.
- **IMDCT bit-exactness.** The current N/4 FFT uses Bluestein instead
  of libopus' bespoke mixed-radix kiss_fft (15·8 split at LM=3). The
  output has comparable RMS but the spectral peak is not yet bit-exact
  with libopus.
- **Sample rates other than 48 kHz.** CELT itself supports 8/12/16/24/48
  kHz; this crate pins 48 kHz. All four RFC-valid LM values (0..3) now
  ship.

## Codec id

- `"celt"`. Audio. `SampleFormat::F32` output, 48 kHz, 1 or 2 channels.
  Default frame size 960 samples (LM=3); use `new_with_frame_samples` for
  120 / 240 / 480 / 960 frame sizes.

## License

MIT — see [LICENSE](LICENSE).
