use std::io::{Read, Seek};

use nanomp3::{Channels as Mp3Channels, Decoder as Mp3Decoder, MAX_SAMPLES_PER_FRAME};

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
    let mut bytes = Vec::new();
    let mut stream = stream;
    stream
        .read_to_end(&mut bytes)
        .map_err(|err| TtsError::AudioError(format!("Failed to read audio stream: {err}")))?;

    decode_audio_bytes(&bytes)
}

pub(super) fn decode_audio_bytes(bytes: &[u8]) -> Result<AudioSamples, TtsError> {
    if is_wav_bytes(bytes) {
        return decode_wav_bytes(bytes);
    }

    decode_mp3_bytes(bytes)
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

fn decode_mp3_bytes(bytes: &[u8]) -> Result<AudioSamples, TtsError> {
    let mut decoder = Mp3Decoder::new();
    let mut pcm = [0.0f32; MAX_SAMPLES_PER_FRAME];
    let mut sample_rate = None;
    let mut samples = Vec::new();
    let mut offset = 0usize;

    while offset < bytes.len() {
        let (consumed, frame) = decoder.decode(&bytes[offset..], &mut pcm);

        if consumed == 0 && frame.is_none() {
            break;
        }

        offset = offset.saturating_add(consumed);

        let Some(frame) = frame else {
            continue;
        };

        if frame.sample_rate == 0 {
            return Err(TtsError::AudioError("MP3 stream is missing a sample rate".into()));
        }

        match sample_rate {
            Some(existing) if existing != frame.sample_rate => {
                return Err(TtsError::AudioError(format!(
                    "MP3 stream changed sample rate from {existing} Hz to {} Hz",
                    frame.sample_rate
                )));
            }
            None => sample_rate = Some(frame.sample_rate),
            _ => {}
        }

        append_decoded_mp3_frame(&mut samples, &pcm, frame.samples_produced, frame.channels)?;
    }

    let sample_rate = sample_rate.ok_or_else(|| {
        TtsError::AudioError("Unsupported audio stream: expected WAV or MP3 data".into())
    })?;

    if samples.is_empty() {
        return Err(TtsError::AudioError(
            "Unsupported audio stream: expected WAV or MP3 data".into(),
        ));
    }

    Ok(AudioSamples::new(samples, sample_rate))
}

fn append_decoded_mp3_frame(
    samples: &mut Vec<f32>,
    pcm: &[f32],
    samples_produced: usize,
    channels: Mp3Channels,
) -> Result<(), TtsError> {
    let channel_count = channels.num() as usize;
    let total_samples = samples_produced.checked_mul(channel_count).ok_or_else(|| {
        TtsError::AudioError("MP3 frame sample count overflowed while decoding".into())
    })?;
    let frame = pcm.get(..total_samples).ok_or_else(|| {
        TtsError::AudioError("MP3 decoder returned an invalid frame length".into())
    })?;

    if channel_count == 1 {
        samples.extend_from_slice(frame);
        return Ok(());
    }

    for frame in frame.chunks_exact(channel_count) {
        samples.push(frame.iter().copied().sum::<f32>() / channel_count as f32);
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct WavFormat {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}