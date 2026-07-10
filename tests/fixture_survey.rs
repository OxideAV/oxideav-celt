//! Real-stream survey: walk the CELT frames of the workspace-staged
//! Ogg-Opus fixtures (reference-encoder output, used strictly as
//! black-box byte streams) through the prefix + §4.3.3 reallocation
//! walk, and sanity-check what the clean-room decoder recovers.
//!
//! Runtime-gated: the fixtures live in the umbrella workspace's
//! `docs/` staging area, which the standalone crate CI does not carry
//! — the test passes with a note when the directory is absent.

use oxideav_celt::{
    decode_frame_prefix, run_prefix_walk, CoarseEnergyState, RangeDecoder, WalkIo, NUM_BANDS,
};
use std::path::PathBuf;

fn fixture_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("../../docs/audio/opus/fixtures"),
        PathBuf::from("docs/audio/opus/fixtures"),
    ];
    candidates.into_iter().find(|p| p.is_dir())
}

/// Minimal Ogg packet extractor (RFC 3533): concatenates page segments
/// into packets, following continuation lacing across pages.
fn ogg_packets(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= bytes.len() {
        assert_eq!(&bytes[pos..pos + 4], b"OggS", "lost page sync");
        let nsegs = bytes[pos + 26] as usize;
        let lacing = &bytes[pos + 27..pos + 27 + nsegs];
        let mut data = pos + 27 + nsegs;
        for &l in lacing {
            current.extend_from_slice(&bytes[data..data + l as usize]);
            data += l as usize;
            if l < 255 {
                packets.push(std::mem::take(&mut current));
            }
        }
        pos = data;
    }
    packets
}

/// Survey one mono CELT fixture: prefix + walk on every frame.
fn survey_mono_celt(path: &std::path::Path, expect_config: u8, lm: u32) -> String {
    let bytes = std::fs::read(path).unwrap();
    let packets = ogg_packets(&bytes);
    assert!(packets.len() > 2, "no audio packets");
    assert_eq!(&packets[0][0..8], b"OpusHead");
    assert_eq!(&packets[1][0..8], b"OpusTags");

    let mut coarse = CoarseEnergyState::new();
    let mut report = String::new();
    let mut n_frames = 0usize;
    let mut n_walk_ok = 0usize;
    for pkt in &packets[2..] {
        if pkt.is_empty() {
            continue;
        }
        let toc = pkt[0];
        let config = toc >> 3;
        assert_eq!(config, expect_config, "unexpected TOC config");
        assert_eq!(toc & 0x04, 0, "fixture must be mono");
        assert_eq!(toc & 0x03, 0, "fixture must be code 0");
        let frame = &pkt[1..];

        let mut dec = RangeDecoder::new(frame);
        let prefix = decode_frame_prefix(
            &mut dec,
            &mut coarse,
            lm,
            frame.len() as u32,
            false,
            0,
            NUM_BANDS,
        )
        .unwrap();
        n_frames += 1;
        let walk = run_prefix_walk(WalkIo::Decode(&mut dec), &prefix, lm, 1);
        if n_frames <= 3 {
            let e: Vec<i32> = coarse.energy[0].iter().map(|&x| x as i32).collect();
            report.push_str(&format!(
                "frame {}: bytes={} silence={} pf={} transient={} intra={} trim={} \
                 boost_total={} rsv_total={} energy[0..8]={:?}\n",
                n_frames,
                frame.len(),
                prefix.header.silence,
                prefix.header.post_filter.is_some(),
                prefix.header.transient,
                prefix.header.intra,
                prefix.allocation.alloc_trim,
                prefix.boosts.total_boost,
                prefix.reservations.total,
                &e[0..8],
            ));
            if let Ok(w) = &walk {
                report.push_str(&format!(
                    "         walk: qlo={} frac={} coded_bands={} skip_bits={} balance={} \
                     fine={:?}\n",
                    w.qlo, w.frac, w.coded_bands, w.skip_bits_used, w.balance, w.fine_bits
                ));
            }
        }
        if walk.is_ok() {
            n_walk_ok += 1;
        }
    }
    report.push_str(&format!(
        "frames={n_frames} walk_ok={n_walk_ok} (survey only; reference bit-exactness \
         is tracked separately)\n"
    ));
    assert_eq!(n_walk_ok, n_frames, "walk errored on a real frame");
    report
}

#[test]
fn survey_reference_celt_fixture_frames() {
    let Some(dir) = fixture_dir() else {
        eprintln!("fixtures not staged here; survey skipped");
        return;
    };
    let mono = dir.join("pair-mono-48k-64kbps/input.opus");
    if !mono.is_file() {
        eprintln!("mono fixture missing; survey skipped");
        return;
    }
    // config 31 = CELT FB 20 ms (LM 3).
    let report = survey_mono_celt(&mono, 31, 3);
    eprintln!("--- pair-mono-48k-64kbps survey ---\n{report}");
}
