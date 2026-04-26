# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2](https://github.com/OxideAV/oxideav-celt/compare/v0.1.1...v0.1.2) - 2026-04-26

### Other

- adopt slim VideoFrame/AudioFrame shape

## [0.2.0](https://github.com/OxideAV/oxideav-celt/compare/v0.1.0...v0.2.0) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- clamp post-filter period to avoid OOB on short subframes
- guard post-filter tapset on remaining bit budget
- use live range-coder rng for per-band PVQ + anti-collapse seed
- add post-filter smoke test + preemph/deemph inverse test
- wire post-filter + pre/de-emphasis through decoder & encoder
- add BSD-3-Clause attribution for libopus-derived code
- release v0.1.0

## [0.1.0](https://github.com/OxideAV/oxideav-celt/compare/v0.0.4...v0.1.0) - 2026-04-19

### Other

- promote to 0.1.0

## [0.0.4](https://github.com/OxideAV/oxideav-celt/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- transient test A/Bs short-block vs long-only pre-echo
- README + lib-doc: transient/short-blocks are implemented
- emit transient / short-block packets from encoder
- wire short-block IMDCT into decoder (transient path)
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- rewrite README + update Cargo description
- add public-API encode -> decode roundtrip tests
- emit silence flag (§4.3 Table 56) for near-silent input
- add standalone CeltDecoder on the CodecRegistry path
