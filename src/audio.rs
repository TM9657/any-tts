//! Audio output types and utilities.

use std::io::{Read, Seek};

mod decode;
mod denoise;

use decode::{decode_audio_bytes, decode_audio_stream, decode_wav_bytes};
use denoise::denoise_audio_samples;
pub use denoise::DenoiseOptions;

/// Raw audio samples produced by TTS synthesis.
#[derive(Debug, Clone)]
pub struct AudioSamples {
    /// Raw PCM samples as f32 in range [-1.0, 1.0].
    pub samples: Vec<f32>,
    /// Sample rate in Hz (e.g. 24000).
    pub sample_rate: u32,
    /// Number of audio channels (always 1 for mono).
    pub channels: u16,
}

impl AudioSamples {
    /// Create a new `AudioSamples` instance.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    /// Duration of the audio in seconds.
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the audio is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Decode a WAV file from bytes.
    ///
    /// Supports RIFF/WAVE PCM integer data (8/16/24/32-bit) and 32-bit float.
    /// Multi-channel audio is downmixed to mono to match the library's output
    /// convention.
    pub fn from_wav_bytes(bytes: &[u8]) -> Result<Self, crate::TtsError> {
        decode_wav_bytes(bytes)
    }

    /// Decode a WAV file from disk.
    pub fn from_wav_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::TtsError> {
        let data = std::fs::read(path)?;
        Self::from_wav_bytes(&data)
    }

    /// Decode a WAV or MP3 stream into mono PCM samples.
    ///
    /// The input format is auto-detected. WAV is decoded directly and MP3 is
    /// decoded with the built-in MP3 decoder.
    pub fn from_audio_stream<R>(stream: R) -> Result<Self, crate::TtsError>
    where
        R: Read + Seek + Send + Sync + 'static,
    {
        decode_audio_stream(stream)
    }

    /// Decode a WAV or MP3 byte buffer into mono PCM samples.
    pub fn from_audio_bytes(bytes: &[u8]) -> Result<Self, crate::TtsError> {
        decode_audio_bytes(bytes)
    }

    /// Decode a WAV or MP3 file from disk.
    pub fn from_audio_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::TtsError> {
        let data = std::fs::read(path)?;
        Self::from_audio_bytes(&data)
    }

    /// Decode a WAV or MP3 stream and apply speech-focused denoising.
    pub fn denoise_audio_stream<R>(
        stream: R,
        options: DenoiseOptions,
    ) -> Result<Self, crate::TtsError>
    where
        R: Read + Seek + Send + Sync + 'static,
    {
        Ok(Self::from_audio_stream(stream)?.denoise_speech(options))
    }

    /// Decode a WAV or MP3 byte buffer and apply speech-focused denoising.
    pub fn denoise_audio_bytes(
        bytes: &[u8],
        options: DenoiseOptions,
    ) -> Result<Self, crate::TtsError> {
        Ok(Self::from_audio_bytes(bytes)?.denoise_speech(options))
    }

    /// Apply speech-focused denoising to the current audio samples.
    ///
    /// This is a classical DSP pass, not a learned source-separation model.
    /// It works best on mono spoken audio with steady background noise or music.
    pub fn denoise_speech(&self, options: DenoiseOptions) -> Self {
        denoise_audio_samples(self, options)
    }

    /// Convert samples to i16 PCM (for WAV output).
    pub fn to_i16(&self) -> Vec<i16> {
        self.samples
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                (clamped * i16::MAX as f32) as i16
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // WAV
    // -----------------------------------------------------------------------

    /// Encode the audio as a 16-bit PCM WAV and return the raw bytes.
    ///
    /// The returned `Vec<u8>` contains a complete RIFF WAV file that can be
    /// written to disk, sent over the network, or played back directly.
    pub fn get_wav(&self) -> Vec<u8> {
        let pcm = self.to_i16();
        let data_len = (pcm.len() * 2) as u32;
        let file_len = 36 + data_len;
        let byte_rate = self.sample_rate * self.channels as u32 * 2;
        let block_align = self.channels * 2;

        // Pre-allocate: 44 bytes header + PCM data
        let mut buf = Vec::with_capacity(44 + data_len as usize);

        // RIFF header
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&file_len.to_le_bytes());
        buf.extend_from_slice(b"WAVE");

        // fmt chunk
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        buf.extend_from_slice(&self.channels.to_le_bytes());
        buf.extend_from_slice(&self.sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // data chunk
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_len.to_le_bytes());
        for sample in &pcm {
            buf.extend_from_slice(&sample.to_le_bytes());
        }

        buf
    }

    /// Save the audio to a WAV file (16-bit PCM).
    ///
    /// Creates parent directories automatically.
    pub fn save_wav(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.get_wav())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use std::f32::consts::PI;
    use std::io::Cursor;

    const MP3_FIXTURE_BASE64: &str = "SUQzBAAAAAAAIlRTU0UAAAAOAAADTGF2ZjYxLjcuMTAwAAAAAAAAAAAAAAD/86TEAAWgBuJhQQABkQMiLzhh4efgHh55+AAAGe2Hh5//gIUXgBEZiAUDAUCAQBgSX26mpsGnjrCGMjzZfQYxMiAUGUJo1P2AU0EJBOfw5QWoJ8O3/EZC6juGGGG/8xLpImReL34lCQNN/KhIGgsCytAADosjVUE6KILUMMPJqgsA7cE8gj4UH7g5CxDs06FQV0J0jeIMIP3a+DMVJWtq2CCJ0AAOX8IbfFCA0OI7z4wAwAsGoGkFgIBAAAbZsDnojFGdFf5b8hEmQjmET/5fiiAS0Liv8CAXkcbcaSq/5aFvVafhTsOf/53jm8iAqSPZAPA/+TSQTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/8zTE/BKA8rr5mmkAVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/8yTE7QSAPuMBzQAAVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/8xTE/gKwPt6AAFgFVVVVVVVVVVVVVVX/8xTE/gKoQtqAA1gIVVVVVVVVVVVVVVX/8xTE/gJQQuaAApgIVVVVVVVVVVVVVVX/8xTE/wP4OuMBSwABVVVVVVVVVVVVVVX/8zTE+hHAsssZmnkAVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVX/8xTE7gAAA0gBwAAAVVVVVVVVVVVVVVU=";

    #[test]
    fn test_duration_calculation() {
        let audio = AudioSamples::new(vec![0.0; 24000], 24000);
        assert!((audio.duration_secs() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_duration_zero_sample_rate() {
        let audio = AudioSamples {
            samples: vec![0.0; 100],
            sample_rate: 0,
            channels: 1,
        };
        assert!((audio.duration_secs()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_to_i16_conversion() {
        let audio = AudioSamples::new(vec![0.0, 1.0, -1.0, 0.5], 24000);
        let pcm = audio.to_i16();
        assert_eq!(pcm[0], 0);
        assert_eq!(pcm[1], i16::MAX);
        assert_eq!(pcm[2], -i16::MAX);
        // 0.5 * 32767 = 16383 (truncated)
        assert_eq!(pcm[3], 16383);
    }

    #[test]
    fn test_to_i16_clamping() {
        let audio = AudioSamples::new(vec![2.0, -2.0], 24000);
        let pcm = audio.to_i16();
        assert_eq!(pcm[0], i16::MAX);
        assert_eq!(pcm[1], -i16::MAX);
    }

    #[test]
    fn test_empty_audio() {
        let audio = AudioSamples::new(vec![], 24000);
        assert!(audio.is_empty());
        assert_eq!(audio.len(), 0);
        assert!((audio.duration_secs()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_wav_roundtrip() {
        let original = AudioSamples::new(vec![0.0, 0.25, -0.25, 1.0, -1.0], 24000);
        let decoded = AudioSamples::from_wav_bytes(&original.get_wav()).unwrap();

        assert_eq!(decoded.sample_rate, 24000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples.len(), original.samples.len());
        assert!((decoded.samples[1] - original.samples[1]).abs() < 1e-3);
        assert!((decoded.samples[2] - original.samples[2]).abs() < 1e-3);
    }

    #[test]
    fn test_invalid_wav_rejected() {
        let err = AudioSamples::from_wav_bytes(b"not a wav").unwrap_err();
        assert!(err.to_string().contains("Invalid WAV header"));
    }

    #[test]
    fn test_from_audio_bytes_auto_detects_wav() {
        let original = AudioSamples::new(vec![0.0, 0.2, -0.2, 0.5, -0.5], 16_000);
        let decoded = AudioSamples::from_audio_bytes(&original.get_wav()).unwrap();

        assert_eq!(decoded.sample_rate, original.sample_rate);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples.len(), original.samples.len());
    }

    #[test]
    fn test_denoise_audio_stream_decodes_wav() {
        let original = AudioSamples::new(synthetic_voice_like_signal(16_000, 1.0), 16_000);
        let cleaned = AudioSamples::denoise_audio_stream(
            Cursor::new(original.get_wav()),
            DenoiseOptions::default(),
        )
        .unwrap();

        assert_eq!(cleaned.sample_rate, original.sample_rate);
        assert_eq!(cleaned.channels, 1);
        assert_eq!(cleaned.samples.len(), original.samples.len());
    }

    #[test]
    fn test_denoise_speech_improves_snr_on_synthetic_mix() {
        let sample_rate = 16_000;
        let clean = synthetic_voice_like_signal(sample_rate, 2.0);
        let noisy = mix_background_music(&clean, sample_rate);
        let audio = AudioSamples::new(noisy.clone(), sample_rate);
        let reference = AudioSamples::new(clean, sample_rate).denoise_speech(DenoiseOptions {
            noise_reduction: 0.0,
            residual_floor: 1.0,
            wet_mix: 1.0,
            ..DenoiseOptions::default()
        });
        let band_limited_noisy =
            AudioSamples::new(noisy.clone(), sample_rate).denoise_speech(DenoiseOptions {
                noise_reduction: 0.0,
                residual_floor: 1.0,
                wet_mix: 1.0,
                ..DenoiseOptions::default()
            });
        let cleaned = audio.denoise_speech(DenoiseOptions::default());

        let snr_before = snr_db(&reference.samples, &band_limited_noisy.samples);
        let snr_after = snr_db(&reference.samples, &cleaned.samples);

        assert!(
            snr_after > snr_before + 0.5,
            "Expected denoiser to improve SNR, before={snr_before:.2} dB after={snr_after:.2} dB"
        );
    }

    #[test]
    fn test_from_audio_stream_decodes_mp3() {
        let mp3 = base64::engine::general_purpose::STANDARD
            .decode(MP3_FIXTURE_BASE64)
            .unwrap();
        let decoded = AudioSamples::from_audio_stream(Cursor::new(mp3)).unwrap();

        assert_eq!(decoded.sample_rate, 24_000);
        assert!(!decoded.samples.is_empty());
        assert!(decoded.samples.iter().any(|sample| sample.abs() > 1e-3));
    }

    fn synthetic_voice_like_signal(sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let sample_count = (sample_rate as f32 * duration_secs) as usize;
        (0..sample_count)
            .map(|index| {
                let time = index as f32 / sample_rate as f32;
                let phrase = (2.0 * PI * 1.15 * time).sin().max(0.0).powf(1.8);
                let syllable = (2.0 * PI * 2.6 * time).sin().abs().powf(0.8);
                let clean = 0.45 * (2.0 * PI * 180.0 * time).sin()
                    + 0.25 * (2.0 * PI * 360.0 * time).sin()
                    + 0.08 * (2.0 * PI * 1_200.0 * time).sin();
                clean * phrase * (0.2 + 0.8 * syllable)
            })
            .collect()
    }

    fn mix_background_music(clean: &[f32], sample_rate: u32) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        clean
            .iter()
            .enumerate()
            .map(|(index, &sample)| {
                let time = index as f32 / sample_rate as f32;
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let pseudo_noise = ((state >> 8) as f32 / (u32::MAX >> 8) as f32) * 2.0 - 1.0;
                let music = 0.18 * (2.0 * PI * 110.0 * time).sin()
                    + 0.12 * (2.0 * PI * 220.0 * time).sin()
                    + 0.08 * (2.0 * PI * 3_600.0 * time).sin()
                    + 0.04 * pseudo_noise;
                (sample + music).clamp(-1.0, 1.0)
            })
            .collect()
    }

    fn snr_db(reference: &[f32], observed: &[f32]) -> f32 {
        let signal_power =
            reference.iter().map(|sample| sample * sample).sum::<f32>() / reference.len() as f32;
        let noise_power = reference
            .iter()
            .zip(observed)
            .map(|(reference, observed)| {
                let error = observed - reference;
                error * error
            })
            .sum::<f32>()
            / reference.len() as f32;

        10.0 * (signal_power / noise_power.max(1e-9)).log10()
    }
}
