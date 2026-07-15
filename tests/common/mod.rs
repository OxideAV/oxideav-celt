//! Shared black-box stream plumbing for the reference-exactness
//! tests: minimal Ogg packet extraction (RFC 3533), RFC 6716 §3.2
//! packet framing, and WAV loading (s16 and float32).

/// Minimal Ogg packet extractor (RFC 3533): concatenates page segments
/// into packets, following continuation lacing across pages.
pub fn ogg_packets(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut packets = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut pos = 0usize;
    while pos + 27 <= bytes.len() {
        assert_eq!(&bytes[pos..pos + 4], b"OggS", "lost page sync");
        let nsegs = bytes[pos + 26] as usize;
        let lacing = &bytes[pos + 27..pos + 27 + nsegs];
        let mut off = pos + 27 + nsegs;
        for &l in lacing {
            current.extend_from_slice(&bytes[off..off + l as usize]);
            off += l as usize;
            if l != 255 {
                packets.push(std::mem::take(&mut current));
            }
        }
        pos = off;
    }
    packets
}

/// RFC 6716 §3.2 packet parsing: split an Opus packet into its CELT
/// frames per the TOC code.
pub fn opus_frames(packet: &[u8]) -> Option<(u8, Vec<Vec<u8>>)> {
    if packet.is_empty() {
        return None;
    }
    let toc = packet[0];
    let code = toc & 3;
    let body = &packet[1..];
    let frames: Vec<Vec<u8>> = match code {
        0 => vec![body.to_vec()],
        1 => {
            let half = body.len() / 2;
            vec![body[..half].to_vec(), body[half..].to_vec()]
        }
        2 => {
            let (len1, used) = if body[0] < 252 {
                (body[0] as usize, 1)
            } else {
                (body[0] as usize + 4 * body[1] as usize, 2)
            };
            vec![
                body[used..used + len1].to_vec(),
                body[used + len1..].to_vec(),
            ]
        }
        _ => {
            // Code 3: count byte, optional padding, CBR or VBR.
            let count_byte = body[0];
            let m = (count_byte & 0x3F) as usize;
            let vbr = count_byte & 0x80 != 0;
            let padded = count_byte & 0x40 != 0;
            let mut off = 1usize;
            let mut pad_len = 0usize;
            if padded {
                loop {
                    let p = body[off] as usize;
                    off += 1;
                    if p == 255 {
                        pad_len += 254;
                    } else {
                        pad_len += p;
                        break;
                    }
                }
            }
            let payload_end = body.len() - pad_len;
            let mut frames = Vec::with_capacity(m);
            if vbr {
                let mut lens = Vec::with_capacity(m - 1);
                for _ in 0..m - 1 {
                    let (l, used) = if body[off] < 252 {
                        (body[off] as usize, 1)
                    } else {
                        (body[off] as usize + 4 * body[off + 1] as usize, 2)
                    };
                    off += used;
                    lens.push(l);
                }
                for &l in &lens {
                    frames.push(body[off..off + l].to_vec());
                    off += l;
                }
                frames.push(body[off..payload_end].to_vec());
            } else {
                let per = (payload_end - off) / m;
                for k in 0..m {
                    frames.push(body[off + k * per..off + (k + 1) * per].to_vec());
                }
            }
            frames
        }
    };
    Some((toc, frames))
}

/// Locate the `data` chunk of a WAV file and return interleaved f32
/// samples; handles PCM s16 (format 1) and IEEE float32 (format 3).
pub fn wav_pcm_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut pos = 12usize;
    let mut format: u16 = 1;
    let mut bits: u16 = 16;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if id == b"fmt " {
            format = u16::from_le_bytes([bytes[pos + 8], bytes[pos + 9]]);
            bits = u16::from_le_bytes([bytes[pos + 22], bytes[pos + 23]]);
            // WAVE_FORMAT_EXTENSIBLE: the effective tag is the first
            // word of the SubFormat GUID.
            if format == 0xFFFE && sz >= 40 {
                format = u16::from_le_bytes([bytes[pos + 32], bytes[pos + 33]]);
            }
        }
        if id == b"data" {
            let data = &bytes[pos + 8..(pos + 8 + sz).min(bytes.len())];
            return match (format, bits) {
                (1, 16) => data
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect(),
                (3, 32) => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                other => panic!("unsupported WAV format {other:?}"),
            };
        }
        pos += 8 + sz + (sz & 1);
    }
    panic!("no data chunk");
}

/// Write a mono/stereo s16 WAV at 48 kHz.
pub fn write_wav_s16(path: &std::path::Path, channels: u16, samples: &[i16]) {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&48000u32.to_le_bytes());
    out.extend_from_slice(&(48000u32 * channels as u32 * 2).to_le_bytes());
    out.extend_from_slice(&(channels * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, out).expect("write wav");
}
