//! # oxideav-celt
//!
//! Pure-Rust CELT layer of the Opus codec (RFC 6716).
//!
//! **Status (2026-05-29):** round-8. The bit-exact CELT/SILK range
//! decoder (RFC 6716 §4.1) is complete; the CELT frame-header prefix
//! (silence / post-filter / transient / intra per §4.3, plus the
//! deferred anti-collapse bit per §4.3.5) is wired up. The §4.3.2.1
//! coarse-energy scaffolding (21-band layout from Table 55 + intra
//! prediction filter with `α=0, β=4915/32768`) is in place; the
//! Laplace decoder + `e_prob_model` table are docs-gap-blocked until
//! a clean-room derivation lands. The §4.3.2.2 fine-energy refinement
//! decoder + finalize step is bit-exact. The §4.3.3 bit-allocation
//! field decoders (alloc.trim, skip, intensity-band, dual-stereo)
//! are exposed standalone, gated on caller-supplied reservation
//! booleans. The §4.3.4.5 time-frequency change parameters
//! (per-band `tf_change` + the gated global `tf_select` + the four
//! TF-adjustment tables 60–63) are wired up. The §4.3.4.3 spreading
//! parameter (PDF `{7, 2, 21, 2}/32`) + Table 59 `f_r` lookup +
//! closed-form rotation-gain helpers are now wired up. The band-boost
//! loop, full budget walk, band decode, PVQ, and MDCT machinery
//! still come later.
//!
//! Every other public API path returns [`Error::NotImplemented`].
//!
//! ## Clean-room provenance
//!
//! All routines in this crate are transcribed from RFC 6716 (the IETF
//! standards-track definition of Opus) and RFC 8251 (the Opus update).
//! Source files the RFC delegates to for normative numeric tables and
//! algorithms are off-limits under the workspace clean-room policy
//! and were not consulted. Black-box invocations of `opusdec` /
//! `opusenc` are permitted under the workspace clean-room policy as
//! opaque validators.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod bit_allocation;
pub mod coarse_energy;
pub mod fine_energy;
pub mod frame_header;
pub mod range_decoder;
pub mod spread;
pub mod tf_change;

pub use bit_allocation::{
    decode_alloc_trim, decode_band_allocation, decode_dual_stereo, decode_intensity_band,
    decode_skip_flag, BandAllocation, BandAllocationGates, DEFAULT_ALLOC_TRIM,
};
pub use coarse_energy::{
    apply_intra_prediction, decode_coarse_energy, CoarseEnergyState, INTRA_ALPHA_Q15,
    INTRA_BETA_Q15, NUM_BANDS,
};
pub use fine_energy::{
    decode_fine_energy, decode_fine_energy_band, finalize_extra_bits, fine_correction_q14,
    fine_correction_qn, FinalizePriority, FinalizeResult, MAX_FINE_BITS,
};
pub use frame_header::{decode_anti_collapse_flag, CeltFrameHeader, PostFilter};
pub use range_decoder::RangeDecoder;
pub use spread::{
    decode_spread, pre_rotation_stride, rotation_gain_ratio, rotation_gain_squared_ratio, Spread,
    DEFAULT_SPREAD,
};
pub use tf_change::{
    decode_tf_changes, decode_tf_parameters, decode_tf_select, tf_adjustment, tf_select_matters,
    TfParameters, LM_VALUES, TABLE_60_NON_TRANSIENT_SEL0, TABLE_61_NON_TRANSIENT_SEL1,
    TABLE_62_TRANSIENT_SEL0, TABLE_63_TRANSIENT_SEL1, TF_CHANGE_VALUES,
};

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
