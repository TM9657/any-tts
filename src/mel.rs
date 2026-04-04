//! Mel spectrogram extraction for voice cloning and audio analysis.
//!
//! Provides a pure-candle implementation of Short-Time Fourier Transform (STFT)
//! and mel filterbank projection. No external FFT library required — the DFT is
//! computed via matrix multiplication with precomputed basis vectors.
//!
//! # Example
//!
//! ```rust,ignore
//! use any_tts::mel::{MelConfig, MelSpectrogram};
//! use candle_core::Device;
//!
//! let mel = MelSpectrogram::new(MelConfig::kokoro(), &Device::Cpu)?;
//! let audio = candle_core::Tensor::zeros(24000, candle_core::DType::F32, &Device::Cpu)?;
//! let spectrogram = mel.compute(&audio)?;
//! // spectrogram shape: [1, 80, num_frames]
//! ```

use candle_core::{DType, Device, Tensor};

use crate::error::{TtsError, TtsResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for mel spectrogram extraction.
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// FFT size (number of frequency bins before taking positive half).
    pub n_fft: usize,
    /// Hop length between STFT frames, in samples.
    pub hop_length: usize,
    /// Analysis window length, in samples. May be ≤ `n_fft`.
    pub win_length: usize,
    /// Number of mel frequency bands.
    pub n_mels: usize,
    /// Expected sample rate of input audio (Hz).
    pub sample_rate: u32,
    /// Log-mel normalization mean (subtracted before dividing by `log_std`).
    pub log_mean: f64,
    /// Log-mel normalization std (divides after subtracting `log_mean`).
    pub log_std: f64,
}

impl MelConfig {
    /// Config matching Kokoro's style encoder preprocessing.
    ///
    /// ```text
    /// n_fft=2048, hop=300, win=1200, 80 mels, 24 kHz
    /// norm: (log(1e-5 + mel) - (-4)) / 4
    /// ```
    pub fn kokoro() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 300,
            win_length: 1200,
            n_mels: 80,
            sample_rate: 24000,
            log_mean: -4.0,
            log_std: 4.0,
        }
    }

    /// Number of positive frequency bins: `n_fft / 2 + 1`.
    pub fn n_freq(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

// ---------------------------------------------------------------------------
// MelSpectrogram
// ---------------------------------------------------------------------------

/// Mel spectrogram extractor.
///
/// Pre-computes DFT basis vectors, Hann window, and mel filterbank on
/// construction so that repeated calls to [`compute`](Self::compute) are fast.
pub struct MelSpectrogram {
    config: MelConfig,
    /// DFT cosine basis `[n_freq, n_fft]`.
    dft_cos: Tensor,
    /// DFT sine basis `[n_freq, n_fft]`.
    dft_sin: Tensor,
    /// Hann window, zero-padded to `n_fft` length.
    window: Tensor,
    /// Mel filterbank `[n_mels, n_freq]`.
    mel_basis: Tensor,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram extractor.
    pub fn new(config: MelConfig, device: &Device) -> TtsResult<Self> {
        let n_fft = config.n_fft;
        let n_freq = config.n_freq();

        // ── DFT basis matrices ────────────────────────────────────────
        let mut cos_data = vec![0f32; n_freq * n_fft];
        let mut sin_data = vec![0f32; n_freq * n_fft];
        for k in 0..n_freq {
            for n in 0..n_fft {
                let angle = 2.0 * std::f32::consts::PI * (k as f32) * (n as f32) / (n_fft as f32);
                cos_data[k * n_fft + n] = angle.cos();
                sin_data[k * n_fft + n] = angle.sin();
            }
        }
        let dft_cos = Tensor::new(cos_data.as_slice(), device)?.reshape((n_freq, n_fft))?;
        let dft_sin = Tensor::new(sin_data.as_slice(), device)?.reshape((n_freq, n_fft))?;

        // ── Hann window (centre-padded to n_fft) ─────────────────────
        let mut window_data = vec![0f32; n_fft];
        let pad_left = (n_fft - config.win_length) / 2;
        for i in 0..config.win_length {
            let w = 0.5
                * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / config.win_length as f32).cos());
            window_data[pad_left + i] = w;
        }
        let window = Tensor::new(window_data.as_slice(), device)?;

        // ── Mel filterbank ────────────────────────────────────────────
        let mel_basis =
            Self::build_mel_filterbank(config.n_mels, n_freq, config.sample_rate, device)?;

        Ok(Self {
            config,
            dft_cos,
            dft_sin,
            window,
            mel_basis,
        })
    }

    /// Compute a log-mel spectrogram from raw PCM audio.
    ///
    /// * **Input:** 1-D `[num_samples]` f32 tensor at [`MelConfig::sample_rate`].
    /// * **Output:** `[1, n_mels, num_frames]` normalised log-mel spectrogram.
    pub fn compute(&self, audio: &Tensor) -> TtsResult<Tensor> {
        let audio = audio.to_dtype(DType::F32)?;
        let n_samples = audio.dim(0)?;
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;

        // ── Reflect-pad the signal ────────────────────────────────────
        let pad_len = n_fft / 2;
        let zeros_l = Tensor::zeros(pad_len, DType::F32, audio.device())?;
        let zeros_r_len = (n_samples + 2 * pad_len).saturating_sub(n_samples + pad_len);
        let zeros_r = Tensor::zeros(pad_len.max(zeros_r_len), DType::F32, audio.device())?;
        let padded = Tensor::cat(&[&zeros_l, &audio, &zeros_r], 0)?;
        let padded_len = padded.dim(0)?;

        // ── Frame the signal ──────────────────────────────────────────
        let num_frames = padded_len.saturating_sub(n_fft) / hop + 1;
        if num_frames == 0 {
            return Err(TtsError::ModelError(
                "Audio too short for mel spectrogram extraction".into(),
            ));
        }

        let mut frames = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let start = i * hop;
            let frame = padded.narrow(0, start, n_fft)?;
            let windowed = (&frame * &self.window)?;
            frames.push(windowed);
        }
        let frames = Tensor::stack(&frames, 0)?; // [num_frames, n_fft]

        // ── DFT via matrix multiply ───────────────────────────────────
        let x_real = frames.matmul(&self.dft_cos.t()?)?; // [num_frames, n_freq]
        let x_imag = frames.matmul(&self.dft_sin.t()?)?;

        // Power spectrum
        let power = (x_real.sqr()? + x_imag.sqr()?)?; // [num_frames, n_freq]

        // ── Mel filterbank projection ─────────────────────────────────
        // mel_basis [n_mels, n_freq] × power^T [n_freq, num_frames] → [n_mels, frames]
        let mel = self.mel_basis.matmul(&power.t()?)?;

        // ── Log compression + normalisation ───────────────────────────
        let log_mel = (mel + 1e-5)?.log()?;
        let normalised = log_mel.affine(
            1.0 / self.config.log_std,
            -self.config.log_mean / self.config.log_std,
        )?;

        // [1, n_mels, num_frames]
        normalised.unsqueeze(0).map_err(TtsError::from)
    }

    /// Get the config used by this extractor.
    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    // ── Private helpers ───────────────────────────────────────────────

    /// Build a triangular mel filterbank `[n_mels, n_freq]`.
    fn build_mel_filterbank(
        n_mels: usize,
        n_freq: usize,
        sample_rate: u32,
        device: &Device,
    ) -> TtsResult<Tensor> {
        let sr = sample_rate as f32;
        let fmax = sr / 2.0;

        let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
        let mel_to_hz = |m: f32| -> f32 { 700.0 * (10.0f32.powf(m / 2595.0) - 1.0) };

        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(fmax);

        // n_mels + 2 equally-spaced points in mel space
        let n_points = n_mels + 2;
        let mel_points: Vec<f32> = (0..n_points)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32)
            .collect();
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // Convert Hz → FFT bin (fractional)
        let bin_points: Vec<f32> = hz_points
            .iter()
            .map(|&hz| hz * (n_freq as f32 - 1.0) * 2.0 / sr)
            .collect();

        // Triangular filters
        let mut filters = vec![0f32; n_mels * n_freq];
        for m in 0..n_mels {
            let f_left = bin_points[m];
            let f_center = bin_points[m + 1];
            let f_right = bin_points[m + 2];

            for k in 0..n_freq {
                let kf = k as f32;
                if kf >= f_left && kf <= f_center && f_center > f_left {
                    filters[m * n_freq + k] = (kf - f_left) / (f_center - f_left);
                } else if kf > f_center && kf <= f_right && f_right > f_center {
                    filters[m * n_freq + k] = (f_right - kf) / (f_right - f_center);
                }
            }
        }

        Tensor::new(filters.as_slice(), device)?
            .reshape((n_mels, n_freq))
            .map_err(TtsError::from)
    }
}

/// Resample audio from `src_rate` to `dst_rate` using linear interpolation.
///
/// For voice cloning, the reference audio must match the model's expected
/// sample rate. This function handles the conversion.
pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_idx = i as f64 / ratio;
        let idx_floor = src_idx.floor() as usize;
        let frac = (src_idx - idx_floor as f64) as f32;

        let s0 = samples[idx_floor.min(samples.len() - 1)];
        let s1 = samples[(idx_floor + 1).min(samples.len() - 1)];
        output.push(s0 + frac * (s1 - s0));
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_config_kokoro() {
        let cfg = MelConfig::kokoro();
        assert_eq!(cfg.n_fft, 2048);
        assert_eq!(cfg.n_freq(), 1025);
        assert_eq!(cfg.n_mels, 80);
        assert_eq!(cfg.sample_rate, 24000);
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let device = Device::Cpu;
        let cfg = MelConfig::kokoro();
        let mel = MelSpectrogram::new(cfg, &device).unwrap();

        // 1 second of audio at 24kHz
        let audio = Tensor::zeros(24000, DType::F32, &device).unwrap();
        let spec = mel.compute(&audio).unwrap();

        assert_eq!(spec.dims()[0], 1); // batch
        assert_eq!(spec.dims()[1], 80); // n_mels
                                        // num_frames ≈ (24000 + 2048) / 300 = ~86
        assert!(spec.dims()[2] > 50);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let device = Device::Cpu;
        let fb = MelSpectrogram::build_mel_filterbank(80, 1025, 24000, &device).unwrap();
        assert_eq!(fb.dims(), &[80, 1025]);
    }

    #[test]
    fn test_mel_filterbank_values() {
        let device = Device::Cpu;
        let fb = MelSpectrogram::build_mel_filterbank(80, 1025, 24000, &device).unwrap();
        let data: Vec<Vec<f32>> = fb.to_vec2().unwrap();

        // Each row should have at least some non-zero values (triangular filter)
        for row in &data {
            let sum: f32 = row.iter().sum();
            assert!(sum > 0.0, "Mel filter band has zero energy");
        }
    }

    #[test]
    fn test_resample_identity() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let out = resample_linear(&samples, 16000, 16000);
        assert_eq!(out, samples);
    }

    #[test]
    fn test_resample_upsample() {
        let samples = vec![0.0, 1.0];
        let out = resample_linear(&samples, 1, 4);
        assert_eq!(out.len(), 8);
        // Should interpolate between 0.0 and 1.0
        assert!((out[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_resample_empty() {
        let out = resample_linear(&[], 16000, 24000);
        assert!(out.is_empty());
    }
}
