# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
