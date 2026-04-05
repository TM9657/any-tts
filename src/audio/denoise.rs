use std::cmp::Ordering;
use std::f32::consts::PI;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use super::AudioSamples;

/// Parameters for the speech-focused denoiser.
///
/// The algorithm combines a speech-band filter with a short-time spectral
/// gate. It is intended to attenuate steady background noise and background
/// music, not to perform full source separation.
#[derive(Debug, Clone, Copy)]
pub struct DenoiseOptions {
    /// FFT frame size used by the spectral gate.
    pub frame_size: usize,
    /// Frame hop size in samples.
    pub hop_size: usize,
    /// Lower cutoff for the speech-band high-pass filter.
    pub speech_low_hz: f32,
    /// Upper cutoff for the speech-band low-pass filter.
    pub speech_high_hz: f32,
    /// Fraction of the quietest frames used to estimate the noise profile.
    pub noise_estimation_percentile: f32,
    /// Multiplier applied to the estimated noise spectrum.
    pub noise_reduction: f32,
    /// Residual spectral floor kept to reduce musical-noise artifacts.
    pub residual_floor: f32,
    /// Blend between denoised output and the speech-band filtered signal.
    pub wet_mix: f32,
}

impl Default for DenoiseOptions {
    fn default() -> Self {
        Self {
            frame_size: 1024,
            hop_size: 256,
            speech_low_hz: 110.0,
            speech_high_hz: 5_800.0,
            noise_estimation_percentile: 0.2,
            noise_reduction: 1.35,
            residual_floor: 0.08,
            wet_mix: 0.9,
        }
    }
}

pub(super) fn denoise_audio_samples(audio: &AudioSamples, options: DenoiseOptions) -> AudioSamples {
    if audio.is_empty() || audio.sample_rate == 0 {
        return audio.clone();
    }

    let config = SanitizedDenoiseOptions::new(audio.sample_rate, options);
    let filtered = apply_speech_bandpass(&audio.samples, &config);
    let window = hann_window(config.frame_size);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(config.frame_size);
    let ifft = planner.plan_fft_inverse(config.frame_size);
    let noise_profile = estimate_noise_profile(&filtered, &config, &window, fft.as_ref());
    let cleaned = render_denoised_samples(&filtered, &config, &window, fft.as_ref(), ifft.as_ref(), &noise_profile);

    AudioSamples::new(cleaned, audio.sample_rate)
}

fn estimate_noise_profile(
    samples: &[f32],
    config: &SanitizedDenoiseOptions,
    window: &[f32],
    fft: &dyn Fft<f32>,
) -> Vec<f32> {
    let selected_offsets = select_quiet_frame_offsets(samples, config);
    let mut profile = vec![0.0; config.frame_size];
    let mut buffer = vec![Complex32::default(); config.frame_size];

    for start in selected_offsets {
        load_windowed_frame(samples, start, window, &mut buffer);
        fft.process(&mut buffer);
        for (value, spectrum) in profile.iter_mut().zip(&buffer) {
            *value += spectrum.norm();
        }
    }

    normalize_and_smooth(&mut profile, quiet_frame_count(samples, config) as f32);
    profile
}

fn select_quiet_frame_offsets(samples: &[f32], config: &SanitizedDenoiseOptions) -> Vec<usize> {
    let mut ranked: Vec<(usize, f32)> = frame_offsets(samples.len(), config.frame_size, config.hop_size)
        .into_iter()
        .map(|start| (start, frame_rms(samples, start, config.frame_size)))
        .collect();
    ranked.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));

    ranked
        .into_iter()
        .take(quiet_frame_count(samples, config))
        .map(|(start, _)| start)
        .collect()
}

fn quiet_frame_count(samples: &[f32], config: &SanitizedDenoiseOptions) -> usize {
    let total_frames = frame_offsets(samples.len(), config.frame_size, config.hop_size)
        .len()
        .max(1);
    ((total_frames as f32 * config.noise_estimation_percentile)
        .ceil() as usize)
        .clamp(1, total_frames)
}

fn render_denoised_samples(
    samples: &[f32],
    config: &SanitizedDenoiseOptions,
    window: &[f32],
    fft: &dyn Fft<f32>,
    ifft: &dyn Fft<f32>,
    noise_profile: &[f32],
) -> Vec<f32> {
    let offsets = frame_offsets(samples.len(), config.frame_size, config.hop_size);
    let mut overlap_add = vec![0.0; samples.len() + config.frame_size];
    let mut normalization = vec![0.0; samples.len() + config.frame_size];
    let mut buffer = vec![Complex32::default(); config.frame_size];
    let mut mask = vec![0.0; config.frame_size];

    for start in offsets {
        load_windowed_frame(samples, start, window, &mut buffer);
        fft.process(&mut buffer);
        build_spectral_mask(&buffer, noise_profile, config, &mut mask);
        apply_mask(&mut buffer, &mask);
        ifft.process(&mut buffer);
        overlap_add_frame(&buffer, start, window, &mut overlap_add, &mut normalization);
    }

    finalize_samples(samples, &overlap_add, &normalization, config)
}

fn build_spectral_mask(
    spectrum: &[Complex32],
    noise_profile: &[f32],
    config: &SanitizedDenoiseOptions,
    mask: &mut [f32],
) {
    for index in 0..mask.len() {
        let magnitude = spectrum[index].norm();
        let noise = noise_profile[index];
        let spectral_subtraction =
            (magnitude - noise * config.noise_reduction).max(0.0) / (magnitude + 1e-6);
        let ratio = magnitude / (noise + 1e-6);
        let gate = ((ratio - 0.9) / 3.0).clamp(0.0, 1.0);
        mask[index] = config.residual_floor
            + (1.0 - config.residual_floor) * spectral_subtraction * (0.35 + 0.65 * gate);
    }
    smooth_in_place(mask);
}

fn apply_mask(spectrum: &mut [Complex32], mask: &[f32]) {
    for (bin, value) in spectrum.iter_mut().zip(mask) {
        *bin *= *value;
    }
}

fn overlap_add_frame(
    frame: &[Complex32],
    start: usize,
    window: &[f32],
    overlap_add: &mut [f32],
    normalization: &mut [f32],
) {
    let frame_size = window.len() as f32;
    for index in 0..window.len() {
        let sample = frame[index].re / frame_size;
        let windowed = sample * window[index];
        overlap_add[start + index] += windowed;
        normalization[start + index] += window[index] * window[index];
    }
}

fn finalize_samples(
    filtered: &[f32],
    overlap_add: &[f32],
    normalization: &[f32],
    config: &SanitizedDenoiseOptions,
) -> Vec<f32> {
    (0..filtered.len())
        .map(|index| {
            let restored = if normalization[index] > 1e-6 {
                overlap_add[index] / normalization[index]
            } else {
                0.0
            };
            let blended = restored * config.wet_mix + filtered[index] * (1.0 - config.wet_mix);
            blended.clamp(-1.0, 1.0)
        })
        .collect()
}

fn normalize_and_smooth(values: &mut [f32], divisor: f32) {
    for value in values.iter_mut() {
        *value /= divisor.max(1.0);
    }
    smooth_in_place(values);
}

fn apply_speech_bandpass(samples: &[f32], config: &SanitizedDenoiseOptions) -> Vec<f32> {
    let mut output = samples.to_vec();
    if let Some(mut high_pass) = Biquad::high_pass(config.sample_rate, config.low_hz, 0.707) {
        for sample in &mut output {
            *sample = high_pass.process(*sample);
        }
    }
    if let Some(mut low_pass) = Biquad::low_pass(config.sample_rate, config.high_hz, 0.707) {
        for sample in &mut output {
            *sample = low_pass.process(*sample);
        }
    }
    output
}

fn hann_window(frame_size: usize) -> Vec<f32> {
    if frame_size <= 1 {
        return vec![1.0; frame_size.max(1)];
    }

    (0..frame_size)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / frame_size as f32).cos())
        .collect()
}

fn frame_offsets(sample_count: usize, frame_size: usize, hop_size: usize) -> Vec<usize> {
    if sample_count <= frame_size {
        return vec![0];
    }

    let mut offsets = Vec::new();
    let mut start = 0usize;
    while start < sample_count {
        offsets.push(start);
        if start + frame_size >= sample_count {
            break;
        }
        start = start.saturating_add(hop_size);
    }
    offsets
}

fn frame_rms(samples: &[f32], start: usize, frame_size: usize) -> f32 {
    let mut sum = 0.0;
    for index in 0..frame_size {
        let sample = samples.get(start + index).copied().unwrap_or(0.0);
        sum += sample * sample;
    }
    (sum / frame_size as f32).sqrt()
}

fn load_windowed_frame(samples: &[f32], start: usize, window: &[f32], buffer: &mut [Complex32]) {
    for (index, value) in buffer.iter_mut().enumerate() {
        let sample = samples.get(start + index).copied().unwrap_or(0.0);
        *value = Complex32::new(sample * window[index], 0.0);
    }
}

fn smooth_in_place(values: &mut [f32]) {
    if values.len() < 3 {
        return;
    }

    let original = values.to_vec();
    for index in 0..values.len() {
        let left = index.saturating_sub(1);
        let right = (index + 1).min(values.len() - 1);
        let width = (right - left + 1) as f32;
        values[index] = original[left..=right].iter().copied().sum::<f32>() / width;
    }
}

#[derive(Debug, Clone, Copy)]
struct SanitizedDenoiseOptions {
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
    low_hz: f32,
    high_hz: f32,
    noise_estimation_percentile: f32,
    noise_reduction: f32,
    residual_floor: f32,
    wet_mix: f32,
}

impl SanitizedDenoiseOptions {
    fn new(sample_rate: u32, options: DenoiseOptions) -> Self {
        let frame_size = options.frame_size.max(128);
        let hop_size = options.hop_size.max(1).min(frame_size);
        let nyquist = sample_rate as f32 * 0.5;

        Self {
            sample_rate,
            frame_size,
            hop_size,
            low_hz: options.speech_low_hz.max(0.0).min(nyquist * 0.9),
            high_hz: options
                .speech_high_hz
                .max(options.speech_low_hz + 1.0)
                .min(nyquist * 0.98),
            noise_estimation_percentile: options.noise_estimation_percentile.clamp(0.05, 0.8),
            noise_reduction: options.noise_reduction.max(0.0),
            residual_floor: options.residual_floor.clamp(0.0, 1.0),
            wet_mix: options.wet_mix.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn high_pass(sample_rate: u32, cutoff_hz: f32, q: f32) -> Option<Self> {
        Self::from_coefficients(sample_rate, cutoff_hz, q, FilterKind::HighPass)
    }

    fn low_pass(sample_rate: u32, cutoff_hz: f32, q: f32) -> Option<Self> {
        Self::from_coefficients(sample_rate, cutoff_hz, q, FilterKind::LowPass)
    }

    fn from_coefficients(sample_rate: u32, cutoff_hz: f32, q: f32, kind: FilterKind) -> Option<Self> {
        if sample_rate == 0 || cutoff_hz <= 0.0 || cutoff_hz >= sample_rate as f32 * 0.5 {
            return None;
        }

        let omega = 2.0 * PI * cutoff_hz / sample_rate as f32;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * q.max(1e-3));
        let (b0, b1, b2) = match kind {
            FilterKind::LowPass => ((1.0 - cos_omega) * 0.5, 1.0 - cos_omega, (1.0 - cos_omega) * 0.5),
            FilterKind::HighPass => ((1.0 + cos_omega) * 0.5, -(1.0 + cos_omega), (1.0 + cos_omega) * 0.5),
        };
        let a0 = 1.0 + alpha;

        Some(Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: (-2.0 * cos_omega) / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        })
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = input * self.b0 + self.z1;
        self.z1 = input * self.b1 + self.z2 - self.a1 * output;
        self.z2 = input * self.b2 - self.a2 * output;
        output
    }
}

#[derive(Debug, Clone, Copy)]
enum FilterKind {
    LowPass,
    HighPass,
}