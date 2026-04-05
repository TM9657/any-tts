use std::io::{Read, Seek};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

use super::AudioSamples;
use crate::TtsError;

pub(super) fn decode_wav_bytes(bytes: &[u8]) -> Result<AudioSamples, TtsError> {
    let (format, data) = parse_wav_chunks(bytes)?;
    let decoded = decode_wav_data(format, data)?;
    Ok(AudioSamples::new(downmix_to_mono(decoded, format.channels), format.sample_rate))
}

pub(super) fn decode_audio_stream<R>(stream: R) -> Result<AudioSamples, TtsError>
where
    R: Read + Seek + Send + Sync + 'static,
{
    decode_with_symphonia(StreamMediaSource(stream))
}

pub(super) fn decode_audio_bytes(bytes: &[u8]) -> Result<AudioSamples, TtsError> {
    if is_wav_bytes(bytes) {
        return decode_wav_bytes(bytes);
    }

    decode_audio_stream(std::io::Cursor::new(bytes.to_vec()))
}

fn is_wav_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

fn parse_wav_chunks(bytes: &[u8]) -> Result<(WavFormat, &[u8]), TtsError> {
    if !is_wav_bytes(bytes) {
        return Err(TtsError::AudioError("Invalid WAV header".into()));
    }

    let mut offset = 12usize;
    let mut format = None;
    let mut data = None;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start.saturating_add(chunk_size);

        if chunk_end > bytes.len() {
            return Err(TtsError::AudioError("Malformed WAV chunk size".into()));
        }

        match chunk_id {
            b"fmt " => format = Some(parse_wav_format(bytes, chunk_start, chunk_size)?),
            b"data" => data = Some(&bytes[chunk_start..chunk_end]),
            _ => {}
        }

        offset = chunk_end;
        if chunk_size % 2 == 1 {
            offset = offset.saturating_add(1);
        }
    }

    let format = format.ok_or_else(|| TtsError::AudioError("Missing WAV fmt chunk".into()))?;
    let data = data.ok_or_else(|| TtsError::AudioError("Missing WAV data chunk".into()))?;
    Ok((format, data))
}

fn parse_wav_format(bytes: &[u8], chunk_start: usize, chunk_size: usize) -> Result<WavFormat, TtsError> {
    if chunk_size < 16 {
        return Err(TtsError::AudioError("WAV fmt chunk is too small".into()));
    }

    let format = WavFormat {
        audio_format: u16::from_le_bytes([bytes[chunk_start], bytes[chunk_start + 1]]),
        channels: u16::from_le_bytes([bytes[chunk_start + 2], bytes[chunk_start + 3]]),
        sample_rate: u32::from_le_bytes([
            bytes[chunk_start + 4],
            bytes[chunk_start + 5],
            bytes[chunk_start + 6],
            bytes[chunk_start + 7],
        ]),
        bits_per_sample: u16::from_le_bytes([bytes[chunk_start + 14], bytes[chunk_start + 15]]),
    };
    if format.channels == 0 {
        return Err(TtsError::AudioError("WAV file declares zero channels".into()));
    }

    Ok(format)
}

fn decode_wav_data(format: WavFormat, data: &[u8]) -> Result<Vec<f32>, TtsError> {
    match (format.audio_format, format.bits_per_sample) {
        (1, 8) => Ok(data
            .iter()
            .map(|&sample| (sample as f32 - 128.0) / 127.0)
            .collect()),
        (1, 16) => Ok(data
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32)
            .collect()),
        (1, 24) => Ok(data
            .chunks_exact(3)
            .map(|chunk| {
                let value = ((chunk[2] as i32) << 24 >> 8) | ((chunk[1] as i32) << 8) | chunk[0] as i32;
                value as f32 / 8_388_607.0
            })
            .collect()),
        (1, 32) => Ok(data
            .chunks_exact(4)
            .map(|chunk| {
                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
                    / i32::MAX as f32
            })
            .collect()),
        (3, 32) => Ok(data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        _ => Err(TtsError::AudioError(format!(
            "Unsupported WAV format: format={}, bits={}",
            format.audio_format, format.bits_per_sample
        ))),
    }
}

fn downmix_to_mono(samples: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples;
    }

    samples
        .chunks(channels as usize)
        .map(|frame| frame.iter().copied().sum::<f32>() / frame.len() as f32)
        .collect()
}

fn decode_with_symphonia<S>(source: S) -> Result<AudioSamples, TtsError>
where
    S: MediaSource + Send + Sync + 'static,
{
    let stream = MediaSourceStream::new(Box::new(source), MediaSourceStreamOptions::default());
    let probed = get_probe()
        .format(
            &Hint::new(),
            stream,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|err| TtsError::AudioError(format!("Unsupported audio stream: {err}")))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| TtsError::AudioError("Audio stream has no default track".into()))?;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| TtsError::AudioError("Audio stream is missing a sample rate".into()))?;
    let track_id = track.id;
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|err| TtsError::AudioError(format!("Failed to create decoder: {err}")))?;

    let mut samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(TtsError::AudioError(format!("Failed to read audio packet: {err}"))),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(TtsError::AudioError(format!("Failed to decode audio packet: {err}"))),
        };

        append_decoded_packet(&mut samples, decoded);
    }

    Ok(AudioSamples::new(samples, sample_rate))
}

fn append_decoded_packet(samples: &mut Vec<f32>, decoded: symphonia::core::audio::AudioBufferRef<'_>) {
    let spec = *decoded.spec();
    let channels = spec.channels.count();
    let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
    buffer.copy_interleaved_ref(decoded);

    if channels == 1 {
        samples.extend_from_slice(buffer.samples());
        return;
    }

    for frame in buffer.samples().chunks(channels) {
        samples.push(frame.iter().copied().sum::<f32>() / frame.len() as f32);
    }
}

#[derive(Debug, Clone, Copy)]
struct WavFormat {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

struct StreamMediaSource<R>(R);

impl<R: Read + Seek> Read for StreamMediaSource<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

impl<R: Read + Seek> Seek for StreamMediaSource<R> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.seek(pos)
    }
}

impl<R: Read + Seek + Send + Sync> MediaSource for StreamMediaSource<R> {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        None
    }
}