//! oxideav-core codec registration: the `celt` [`Decoder`] /
//! [`Encoder`] wrappers over [`CeltRefDecoder`] / [`CeltRefEncoder`],
//! the direct [`make_decoder`] / [`make_encoder`] factories, and the
//! [`register`] entry point (the workspace dual-API convention: the
//! registry path and the direct factories share the same functions).
//!
//! ## Stream model
//!
//! Raw CELT frames are not self-describing: the frame-size shift and
//! the channel count are stream-level parameters, exactly as in the
//! RFC 6716 §4.3 operating modes. The wrappers read them from
//! [`CodecParameters`]: `channels` (1 or 2, default 1),
//! `sample_rate` (must be 48 000 when set — the only rate the crate's
//! reference-exact chain drives today), and the `frame_size` codec
//! option (120 / 240 / 480 / 960 samples, default 960). Each
//! [`Packet`] carries exactly one raw CELT frame; decoded /
//! consumed audio is interleaved f32 ([`SampleFormat::F32`]).
//!
//! The encoder derives its fixed per-frame byte budget from
//! `CodecParameters::bit_rate` (default 96 000 bit/s), clamped to the
//! legal 2..=1275-byte frame range.

use std::collections::VecDeque;

use oxideav_core::{
    parse_options, CodecCapabilities, CodecId, CodecInfo, CodecOptionsStruct, CodecParameters,
    Error as CoreError, Frame, OptionField, OptionKind, OptionValue, Packet, Result as CoreResult,
    RuntimeContext, SampleFormat, TimeBase,
};

use crate::ref_decode::CeltRefDecoder;
use crate::ref_encode::CeltRefEncoder;

/// Registry identifier of this codec.
pub const CODEC_ID: &str = "celt";

/// The only sample rate the reference-exact chain operates at.
const SAMPLE_RATE: u32 = 48_000;

/// Default encoder bit rate (bit/s) when the caller sets none.
const DEFAULT_BIT_RATE: u64 = 96_000;

/// Typed options shared by the decoder and encoder factories.
#[derive(Debug, Clone)]
pub struct CeltCodecOptions {
    /// CELT frame size in samples at 48 kHz: 120, 240, 480, or 960.
    pub frame_size: u32,
}

impl Default for CeltCodecOptions {
    fn default() -> Self {
        Self { frame_size: 960 }
    }
}

impl CodecOptionsStruct for CeltCodecOptions {
    const SCHEMA: &'static [OptionField] = &[OptionField {
        name: "frame_size",
        kind: OptionKind::U32,
        default: OptionValue::U32(960),
        help: "CELT frame size in samples at 48 kHz (120, 240, 480, or 960)",
    }];

    fn apply(&mut self, key: &str, value: &OptionValue) -> CoreResult<()> {
        match key {
            "frame_size" => self.frame_size = value.as_u32()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// Shared parameter validation: `(lm, channels, frame_size)`.
fn stream_config(params: &CodecParameters) -> CoreResult<(u32, usize, usize)> {
    if let Some(sr) = params.sample_rate {
        if sr != SAMPLE_RATE {
            return Err(CoreError::unsupported(format!(
                "celt: only 48 kHz is supported (got {sr})"
            )));
        }
    }
    let channels = params.channels.unwrap_or(1);
    if !(1..=2).contains(&channels) {
        return Err(CoreError::unsupported(format!(
            "celt: 1 or 2 channels supported (got {channels})"
        )));
    }
    let opts: CeltCodecOptions = parse_options(&params.options)?;
    let lm = match opts.frame_size {
        120 => 0u32,
        240 => 1,
        480 => 2,
        960 => 3,
        other => {
            return Err(CoreError::invalid(format!(
                "celt: frame_size must be 120/240/480/960 (got {other})"
            )))
        }
    };
    Ok((lm, channels as usize, opts.frame_size as usize))
}

fn map_err(e: crate::Error) -> CoreError {
    CoreError::invalid(format!("celt: {e}"))
}

// ───────────────────────── decoder ─────────────────────────

/// Packet-to-frame decoder over [`CeltRefDecoder`] (one raw CELT
/// frame per packet, interleaved f32 output).
pub struct CeltDecoder {
    id: CodecId,
    inner: CeltRefDecoder,
    lm: u32,
    channels: usize,
    pending: VecDeque<Frame>,
    /// Running sample position for synthesized pts (1/48000 base),
    /// used when packets carry no pts of their own.
    next_pts: i64,
    eof: bool,
}

impl std::fmt::Debug for CeltDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CeltDecoder")
            .field("lm", &self.lm)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

impl oxideav_core::Decoder for CeltDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.id
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        if self.eof {
            return Err(CoreError::invalid("celt: send_packet after flush"));
        }
        let pcm = self.inner.decode_frame(&packet.data).map_err(map_err)?;
        let samples = (pcm.len() / self.channels) as u32;
        let mut bytes = Vec::with_capacity(pcm.len() * 4);
        for v in &pcm {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let pts = packet.pts.or(Some(self.next_pts));
        self.next_pts = pts.unwrap_or(self.next_pts) + i64::from(samples);
        self.pending
            .push_back(Frame::Audio(oxideav_core::AudioFrame {
                samples,
                pts,
                data: vec![bytes],
            }));
        Ok(())
    }

    fn receive_frame(&mut self) -> CoreResult<Frame> {
        match self.pending.pop_front() {
            Some(f) => Ok(f),
            None if self.eof => Err(CoreError::Eof),
            None => Err(CoreError::NeedMore),
        }
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> CoreResult<()> {
        self.inner = CeltRefDecoder::new(self.lm, self.channels).map_err(map_err)?;
        self.pending.clear();
        self.next_pts = 0;
        self.eof = false;
        Ok(())
    }
}

/// Direct decoder factory (the dual-API entry point; also installed
/// into the registry by [`register`]).
pub fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn oxideav_core::Decoder>> {
    let (lm, channels, _frame_size) = stream_config(params)?;
    Ok(Box::new(CeltDecoder {
        id: CodecId::new(CODEC_ID),
        inner: CeltRefDecoder::new(lm, channels).map_err(map_err)?,
        lm,
        channels,
        pending: VecDeque::new(),
        next_pts: 0,
        eof: false,
    }))
}

// ───────────────────────── encoder ─────────────────────────

/// Frame-to-packet encoder over [`CeltRefEncoder`]: buffers
/// interleaved f32 input and emits one fixed-size raw CELT frame per
/// `frame_size` samples. `flush` zero-pads the final partial frame.
pub struct CeltEncoder {
    id: CodecId,
    inner: CeltRefEncoder,
    output_params: CodecParameters,
    channels: usize,
    frame_size: usize,
    frame_bytes: usize,
    /// Interleaved sample FIFO awaiting a full frame.
    buffer: Vec<f32>,
    ready: VecDeque<Packet>,
    /// Samples-per-channel already emitted (pts accounting).
    position: i64,
    flushed: bool,
}

impl std::fmt::Debug for CeltEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CeltEncoder")
            .field("channels", &self.channels)
            .field("frame_size", &self.frame_size)
            .field("frame_bytes", &self.frame_bytes)
            .finish_non_exhaustive()
    }
}

impl CeltEncoder {
    fn encode_buffered(&mut self) -> CoreResult<()> {
        let span = self.frame_size * self.channels;
        while self.buffer.len() >= span {
            let chunk: Vec<f32> = self.buffer.drain(..span).collect();
            let data = self
                .inner
                .encode_frame(&chunk, self.frame_bytes)
                .map_err(map_err)?;
            let mut packet = Packet::new(0, TimeBase::from_rate(SAMPLE_RATE), data);
            packet.pts = Some(self.position);
            packet.dts = packet.pts;
            packet.duration = Some(self.frame_size as i64);
            packet.flags.keyframe = true;
            self.position += self.frame_size as i64;
            self.ready.push_back(packet);
        }
        Ok(())
    }
}

impl oxideav_core::Encoder for CeltEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> CoreResult<()> {
        if self.flushed {
            return Err(CoreError::invalid("celt: send_frame after flush"));
        }
        let audio = match frame {
            Frame::Audio(a) => a,
            _ => return Err(CoreError::invalid("celt: expected an audio frame")),
        };
        if audio.data.len() != 1 {
            return Err(CoreError::invalid(
                "celt: expected one interleaved f32 plane",
            ));
        }
        let plane = &audio.data[0];
        let expect = audio.samples as usize * self.channels * 4;
        if plane.len() != expect {
            return Err(CoreError::invalid(format!(
                "celt: plane holds {} bytes, expected {expect} \
                 ({} samples x {} channels x f32)",
                plane.len(),
                audio.samples,
                self.channels
            )));
        }
        self.buffer.extend(
            plane
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        self.encode_buffered()
    }

    fn receive_packet(&mut self) -> CoreResult<Packet> {
        match self.ready.pop_front() {
            Some(p) => Ok(p),
            None if self.flushed => Err(CoreError::Eof),
            None => Err(CoreError::NeedMore),
        }
    }

    fn flush(&mut self) -> CoreResult<()> {
        if !self.buffer.is_empty() {
            let span = self.frame_size * self.channels;
            self.buffer.resize(span, 0.0);
            self.encode_buffered()?;
        }
        self.flushed = true;
        Ok(())
    }
}

/// Direct encoder factory (the dual-API entry point; also installed
/// into the registry by [`register`]).
pub fn make_encoder(params: &CodecParameters) -> CoreResult<Box<dyn oxideav_core::Encoder>> {
    let (lm, channels, frame_size) = stream_config(params)?;
    let bit_rate = params.bit_rate.unwrap_or(DEFAULT_BIT_RATE);
    let frame_bytes =
        ((bit_rate as u128 * frame_size as u128) / (u128::from(SAMPLE_RATE) * 8)) as usize;
    let frame_bytes = frame_bytes.clamp(2, 1275);

    let mut output_params = CodecParameters::audio(CodecId::new(CODEC_ID));
    output_params.sample_rate = Some(SAMPLE_RATE);
    output_params.channels = Some(channels as u16);
    output_params.sample_format = Some(SampleFormat::F32);
    output_params.bit_rate =
        Some(frame_bytes as u64 * 8 * u64::from(SAMPLE_RATE) / frame_size as u64);
    output_params
        .options
        .insert("frame_size", frame_size.to_string());

    Ok(Box::new(CeltEncoder {
        id: CodecId::new(CODEC_ID),
        inner: CeltRefEncoder::new(lm, channels).map_err(map_err)?,
        output_params,
        channels,
        frame_size,
        frame_bytes,
        buffer: Vec::new(),
        ready: VecDeque::new(),
        position: 0,
        flushed: false,
    }))
}

// ───────────────────────── registration ─────────────────────────

/// Install the `celt` codec (decoder + encoder) into the runtime
/// context's codec registry.
pub fn register(ctx: &mut RuntimeContext) {
    ctx.codecs.register(
        CodecInfo::new(CodecId::new(CODEC_ID))
            .capabilities(
                CodecCapabilities::audio("celt_sw")
                    .with_lossy(true)
                    .with_max_sample_rate(SAMPLE_RATE)
                    .with_max_channels(2),
            )
            .decoder(make_decoder)
            .encoder(make_encoder)
            .decoder_options::<CeltCodecOptions>()
            .encoder_options::<CeltCodecOptions>(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(channels: u16, frame_size: u32, bit_rate: Option<u64>) -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new(CODEC_ID));
        p.sample_rate = Some(48_000);
        p.channels = Some(channels);
        p.sample_format = Some(SampleFormat::F32);
        p.bit_rate = bit_rate;
        p.options.insert("frame_size", frame_size.to_string());
        p
    }

    fn tone_frame(samples: usize, channels: usize, offset: usize) -> Frame {
        let mut bytes = Vec::with_capacity(samples * channels * 4);
        for t in 0..samples {
            for c in 0..channels {
                let f0 = if c == 0 { 440.0 } else { 523.0 };
                let v =
                    0.3 * (2.0 * std::f32::consts::PI * f0 * (offset + t) as f32 / 48_000.0).sin();
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        Frame::Audio(oxideav_core::AudioFrame {
            samples: samples as u32,
            pts: Some(offset as i64),
            data: vec![bytes],
        })
    }

    /// Registry round trip: register, resolve both factories, encode a
    /// tone through the Encoder trait, decode it back through the
    /// Decoder trait, and check the audio survives.
    #[test]
    fn registry_encode_decode_roundtrip() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(CODEC_ID);
        assert!(ctx.codecs.has_decoder(&id));
        assert!(ctx.codecs.has_encoder(&id));

        let p = params(1, 480, Some(96_000));
        let mut enc = ctx.codecs.first_encoder(&p).expect("encoder");
        assert_eq!(enc.output_params().sample_rate, Some(48_000));

        let frames_n = 10usize;
        for f in 0..frames_n {
            enc.send_frame(&tone_frame(480, 1, f * 480)).expect("send");
        }
        enc.flush().expect("flush");
        let mut packets = Vec::new();
        loop {
            match enc.receive_packet() {
                Ok(p) => packets.push(p),
                Err(CoreError::Eof) => break,
                Err(e) => panic!("unexpected encoder error: {e:?}"),
            }
        }
        assert_eq!(packets.len(), frames_n);
        // 96 kb/s at 10 ms = 120 bytes/frame.
        assert!(packets.iter().all(|p| p.data.len() == 120));

        let mut dec = ctx.codecs.first_decoder(&p).expect("decoder");
        let mut pcm: Vec<f32> = Vec::new();
        for packet in &packets {
            dec.send_packet(packet).expect("send_packet");
            loop {
                match dec.receive_frame() {
                    Ok(Frame::Audio(a)) => {
                        assert_eq!(a.samples, 480);
                        pcm.extend(
                            a.data[0]
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                        );
                    }
                    Ok(_) => panic!("expected audio"),
                    Err(CoreError::NeedMore) => break,
                    Err(e) => panic!("unexpected decoder error: {e:?}"),
                }
            }
        }
        assert_eq!(pcm.len(), frames_n * 480);

        // Delay-compensated fidelity on the steady state.
        let delay = 120usize;
        let skip = 2 * 480;
        let mut ee = 0f64;
        let mut err = 0f64;
        for i in 0..(pcm.len() - delay - skip) {
            let t = (skip + i) as f32 / 48_000.0;
            let e = (0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()) as f64;
            ee += e * e;
            let d = e - pcm[skip + delay + i] as f64;
            err += d * d;
        }
        let snr = 10.0 * (ee / err.max(1e-30)).log10();
        assert!(snr > 15.0, "registry loop SNR {snr:.2} dB too low");
    }

    /// The direct factories reject invalid parameter sets.
    #[test]
    fn factories_validate_parameters() {
        let mut p = params(1, 960, None);
        p.sample_rate = Some(44_100);
        assert!(make_decoder(&p).is_err());
        assert!(make_encoder(&p).is_err());

        let p = params(3, 960, None);
        assert!(make_decoder(&p).is_err());

        let p = params(1, 300, None);
        assert!(make_decoder(&p).is_err());
        assert!(make_encoder(&p).is_err());

        let mut p = params(1, 960, None);
        p.options.insert("bogus", "1");
        assert!(make_decoder(&p).is_err());
    }

    /// Decoder reset wipes cross-frame state: a stream decoded after
    /// reset matches a stream decoded on a fresh instance.
    #[test]
    fn decoder_reset_matches_fresh() {
        let p = params(1, 240, Some(64_000));
        let mut enc = make_encoder(&p).expect("encoder");
        for f in 0..4 {
            enc.send_frame(&tone_frame(240, 1, f * 240)).expect("send");
        }
        enc.flush().expect("flush");
        let mut packets = Vec::new();
        while let Ok(pk) = enc.receive_packet() {
            packets.push(pk);
        }

        let decode_all = |dec: &mut Box<dyn oxideav_core::Decoder>| -> Vec<u8> {
            let mut out = Vec::new();
            for pk in &packets {
                dec.send_packet(pk).expect("send");
                while let Ok(Frame::Audio(a)) = dec.receive_frame() {
                    out.extend_from_slice(&a.data[0]);
                }
            }
            out
        };

        let mut dec = make_decoder(&p).expect("decoder");
        let first = decode_all(&mut dec);
        oxideav_core::Decoder::reset(dec.as_mut()).expect("reset");
        let second = decode_all(&mut dec);
        let mut fresh = make_decoder(&p).expect("decoder");
        let fresh_out = decode_all(&mut fresh);
        assert_eq!(first, fresh_out);
        assert_eq!(second, fresh_out);
    }

    /// Encoder flush pads the trailing partial frame instead of
    /// dropping it.
    #[test]
    fn encoder_flush_pads_partial_frame() {
        let p = params(2, 120, Some(128_000));
        let mut enc = make_encoder(&p).expect("encoder");
        enc.send_frame(&tone_frame(100, 2, 0)).expect("send");
        assert!(matches!(enc.receive_packet(), Err(CoreError::NeedMore)));
        enc.flush().expect("flush");
        let pk = enc.receive_packet().expect("padded final packet");
        assert!(!pk.data.is_empty());
        assert!(matches!(enc.receive_packet(), Err(CoreError::Eof)));
    }
}
