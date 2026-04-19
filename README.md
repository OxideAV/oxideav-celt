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

The codec is also discoverable through the `CodecRegistry` via
`oxideav_celt::register(&mut registry)`.

## Supported configuration

This crate targets the fullband long-block subset of CELT — the same
profile an Opus encoder picks for "music" content at the default
20 ms frame size.

| Parameter      | Value                                 |
|----------------|---------------------------------------|
| Sample rate    | 48 kHz only                           |
| Channels       | 1 (mono) or 2 (stereo, dual-stereo)   |
| Frame size     | 960 samples = 20 ms (LM=3, fullband)  |
| Sample format  | `F32`, `F32P`, `S16`, `S16P` on input |
| Output format  | `F32`, interleaved                    |
| Bandwidth      | Fullband (NB_EBANDS=21 bands)         |

The decoder accepts any packet that matches the encoder's profile, plus
the RFC 6716 §4.3 silence flag (packets with the silence bit set decode
to zeros). Frames marked `transient` or carrying comb-post-filter
parameters return `Error::Unsupported` — the decoder does not yet cover
those paths.

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
  (`cwrs`), `exp_rotation`, collapse-mask extraction.
- §4.3.5 anti-collapse — pulse-injection on transient long blocks
  (decoder path; the encoder currently does not emit transients).
- §4.3.6 denormalisation.
- §4.3.7 inverse MDCT — pre-twiddle, length-N/4 complex FFT via
  Bluestein, post-twiddle, window + overlap-add.
- §4.3.8 comb pitch post-filter — decoder path (the encoder does not
  emit post-filter taps).

Static tables (transcribed from libopus `static_modes_float.h`):
`EBAND_5MS`, `E_PROB_MODEL`, `PRED_COEF` / `BETA_COEF` / `BETA_INTRA`,
`BAND_ALLOCATION`, `LOG2_FRAC_TABLE`, `LOGN400`, `CACHE_INDEX50`,
`CACHE_BITS50`, `CACHE_CAPS50`, `E_MEANS`, `SPREAD_ICDF`, `TRIM_ICDF`,
`TF_SELECT_TABLE`, `COMB_FILTER_TAPS`.

## What's not implemented

These paths are partially or fully missing. The codec runs end-to-end
without them, but bit-for-bit parity with libopus requires closing
every gap below.

- **Transient detection / short blocks.** The encoder always emits
  `transient=false` and a single 960-sample long block. Percussive
  content will suffer pre-echo artefacts. The decoder rejects
  `transient=true` packets with `Error::Unsupported`.
- **Time-frequency change flags.** `tf_res[i] = 0` on the encode side;
  the decoder reads the symbols faithfully but the encoder never
  chooses a non-zero TF resolution.
- **Dynalloc band-energy boosts.** No per-band boost is emitted.
- **Intensity stereo.** Stereo uses dual-stereo only — L and R coded as
  two independent mono bands in one packet. Intensity stereo would buy
  extra HF bits on low bit-rate stereo; at 102 kbit/s the dual path
  produces well-separated L/R output.
- **Comb post-filter on the encoder.** The §4.3.8 pitch post-filter is
  implemented on the decoder side but the encoder never emits a
  non-zero post-filter flag.
- **IMDCT bit-exactness.** The current N/4 FFT uses Bluestein instead
  of libopus' bespoke mixed-radix kiss_fft (15·8 split at LM=3). The
  output has comparable RMS but the spectral peak is not yet bit-exact
  with libopus.
- **Sample rates other than 48 kHz and frame sizes other than 20 ms.**
  CELT itself supports 8/12/16/24/48 kHz and LM=0..3; this crate
  currently pins LM=3 / 48 kHz.

## Codec id

- `"celt"`. Audio. `SampleFormat::F32` output, 48 kHz, 1 or 2 channels,
  960 samples per frame.

## License

MIT — see [LICENSE](LICENSE).
