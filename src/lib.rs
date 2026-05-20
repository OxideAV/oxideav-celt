//! # oxideav-celt
//!
//! Pure-Rust CELT layer of the Opus codec (RFC 6716).
//!
//! **Status (2026-05-20):** round-1 bootstrap. The crate carries the
//! bit-exact CELT/SILK range decoder (RFC 6716 §4.1) and the no-op
//! `register` glue; no MDCT, band decoding, or PVQ is wired up yet.
//!
//! Every other public API path returns [`Error::NotImplemented`].
//!
//! The range decoder is the foundational primitive: every CELT and
//! SILK symbol passes through it. Subsequent rounds will layer the
//! band-decode, PVQ, and MDCT machinery on top.
//!
//! ## Clean-room provenance
//!
//! All routines in this crate are transcribed from RFC 6716 (the IETF
//! standards-track definition of Opus) and RFC 8251 (the Opus update).
//! No external library source — libopus, the Opus reference encoder /
//! decoder, etc. — was consulted. Black-box invocations of `opusdec`
//! / `opusenc` are permitted under the workspace clean-room policy as
//! opaque validators.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod range_decoder;

pub use range_decoder::RangeDecoder;

/// Crate-local error type. The encoder, frame-level decoder, and
/// higher-level codec entry points are not yet wired up; calling them
/// returns [`Error::NotImplemented`]. The range decoder primitives in
/// [`range_decoder`] do not use this type for their hot paths — they
/// latch a sticky error flag instead, mirroring the behaviour
/// recommended by RFC 6716 §4.1.5 for corrupt frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A higher-level CELT entry point that has not been implemented
    /// yet (everything beyond the range decoder, today).
    NotImplemented,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NotImplemented => write!(
                f,
                "oxideav-celt: requested entry point is not yet implemented"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// No-op codec registration. The range decoder is a leaf primitive
/// and does not need to advertise a codec ID; once the band/MDCT
/// path lands, this hook will register the `celt` decoder with the
/// runtime.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("celt", register);
