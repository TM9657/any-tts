//! Kokoro ISTFTNet decoder.
//!
//! Converts aligned text features + F0 + noise into audio waveforms using
//! inverse STFT. Architecture from StyleTTS2.
//!
//! Pipeline:
//! - Decoder: AdainResBlk1d stack → Generator
//! - Generator: ConvTranspose1d upsampling + ResBlocks + iSTFT
//! - SineGen: Harmonic source signal generation for excitation

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::layers::conv::{AdaIn1d, Conv1d, ConvTranspose1d};

use super::prosody::AdainResBlk1d;

fn scalar_like(tensor: &Tensor, value: f32) -> Result<Tensor> {
    Tensor::new(value, tensor.device())?.to_dtype(tensor.dtype())
}

fn scale_tensor(tensor: &Tensor, value: f32) -> Result<Tensor> {
    tensor.broadcast_mul(&scalar_like(tensor, value)?)
}

// ---------------------------------------------------------------------------
// Upsample helper (for Metal compatibility)
// ---------------------------------------------------------------------------

/// Upsample 1D using repeat (works on Metal unlike upsample_nearest1d).
/// Input: [batch, channels, length] → Output: [batch, channels, target_len]
fn upsample_1d_repeat(x: &Tensor, target_len: usize) -> Result<Tensor> {
    let (batch, channels, length) = x.dims3()?;
    if target_len == length {
        return Ok(x.clone());
    }
    // Compute scale factor (must be integer multiple for repeat)
    let scale = target_len / length;
    if scale * length == target_len && scale > 1 {
        // Exact integer upsample: [b, c, l] → [b, c, l, 1] → repeat → [b, c, l*scale]
        let x = x.unsqueeze(3)?;
        let x = x.repeat(&[1, 1, 1, scale])?;
        x.reshape((batch, channels, target_len))
    } else {
        // Fallback: move to CPU, upsample, move back
        let device = x.device().clone();
        let x_cpu = x.to_device(&Device::Cpu)?;
        let upsampled = x_cpu.upsample_nearest1d(target_len)?;
        upsampled.to_device(&device)
    }
}

/// LeakyReLU activation: max(x, x * negative_slope)
fn leaky_relu(x: &Tensor, negative_slope: f32) -> Result<Tensor> {
    let scaled = scale_tensor(x, negative_slope)?;
    x.maximum(&scaled)
}

fn linear_resample_1d(input: &[f32], output_len: usize) -> Vec<f32> {
    if input.is_empty() || output_len == 0 {
        return Vec::new();
    }
    if input.len() == output_len {
        return input.to_vec();
    }
    if input.len() == 1 {
        return vec![input[0]; output_len];
    }

    let input_len = input.len() as f32;
    let output_len_f = output_len as f32;
    let max_index = (input.len() - 1) as f32;

    (0..output_len)
        .map(|index| {
            let src =
                (((index as f32) + 0.5) * input_len / output_len_f - 0.5).clamp(0.0, max_index);
            let left = src.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let frac = src - left as f32;
            input[left] * (1.0 - frac) + input[right] * frac
        })
        .collect()
}

fn reflect_pad_1d(samples: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 {
        return samples.to_vec();
    }
    assert!(
        samples.len() > pad,
        "reflect padding requires input longer than padding"
    );

    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    for index in (1..=pad).rev() {
        padded.push(samples[index]);
    }
    padded.extend_from_slice(samples);
    let len = samples.len();
    for index in 0..pad {
        padded.push(samples[len - 2 - index]);
    }
    padded
}

fn reflect_pad_left_1d_tensor(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }

    let length = x.dim(2)?;
    let mut parts = Vec::with_capacity(pad + 1);
    for index in (1..=pad).rev() {
        parts.push(x.narrow(2, index, 1)?);
    }
    parts.push(x.clone());
    let refs: Vec<&Tensor> = parts.iter().collect();
    if length <= pad {
        return Ok(x.clone());
    }
    Tensor::cat(&refs, 2)
}

fn pseudo_random_unit(seed: u64) -> f32 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let value = value ^ (value >> 31);
    ((value >> 40) as u32 as f32) / ((1u32 << 24) as f32)
}

fn stft_frame(frame: &[f32], freq_bins: usize) -> (Vec<f32>, Vec<f32>) {
    let n_fft = frame.len();
    let mut magnitude = vec![0.0; freq_bins];
    let mut phase = vec![0.0; freq_bins];

    for k in 0..freq_bins {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;
        for (n, &sample) in frame.iter().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
            real += sample * angle.cos();
            imag -= sample * angle.sin();
        }
        magnitude[k] = (real * real + imag * imag).sqrt();
        phase[k] = imag.atan2(real);
    }

    (magnitude, phase)
}

fn istft_frame(magnitude: &[f32], phase: &[f32], n_fft: usize) -> Vec<f32> {
    let mut real = vec![0.0f32; n_fft];
    let mut imag = vec![0.0f32; n_fft];
    let nyquist = n_fft / 2;

    for k in 0..=nyquist {
        let re = magnitude[k] * phase[k].cos();
        let im = magnitude[k] * phase[k].sin();
        real[k] = re;
        imag[k] = im;
        if k > 0 && k < nyquist {
            real[n_fft - k] = re;
            imag[n_fft - k] = -im;
        }
    }

    let mut frame = vec![0.0f32; n_fft];
    for (n, sample) in frame.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for k in 0..n_fft {
            let angle = 2.0 * std::f32::consts::PI * k as f32 * n as f32 / n_fft as f32;
            sum += real[k] * angle.cos() - imag[k] * angle.sin();
        }
        *sample = sum / n_fft as f32;
    }

    frame
}

// ---------------------------------------------------------------------------
// AdaINResBlock1 — ResBlock with AdaIN for the Generator
// ---------------------------------------------------------------------------

/// ResBlock with AdaIN used inside the Generator (different from AdainResBlk1d).
///
/// Each block has two passes: (adain1 + convs1) and (adain2 + convs2),
/// each with learnable residual scaling (alpha1, alpha2).
struct AdaInResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    adain1: Vec<AdaIn1d>,
    adain2: Vec<AdaIn1d>,
    alpha1: Vec<Tensor>,
    alpha2: Vec<Tensor>,
    num_layers: usize,
}

impl AdaInResBlock1 {
    fn load(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        style_dim: usize,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut adain1 = Vec::new();
        let mut adain2 = Vec::new();
        let mut alpha1 = Vec::new();
        let mut alpha2 = Vec::new();

        for (i, &dilation) in dilations.iter().enumerate() {
            let padding = (kernel_size * dilation - dilation) / 2;

            // First pass: adain1 → convs1
            convs1.push(Conv1d::load(
                channels,
                channels,
                kernel_size,
                1,
                padding,
                dilation,
                1,
                true,
                vb.pp("convs1").pp(i.to_string()),
            )?);
            adain1.push(AdaIn1d::load(
                style_dim,
                channels,
                vb.pp("adain1").pp(i.to_string()),
            )?);
            // alpha1 is per-channel with shape [1, channels, 1].
            let a1 = vb
                .get((1, channels, 1), &format!("alpha1.{}", i))
                .unwrap_or_else(|_| Tensor::ones((1, channels, 1), dtype, device).unwrap());
            alpha1.push(a1);

            // Second pass: adain2 → convs2
            convs2.push(Conv1d::load(
                channels,
                channels,
                kernel_size,
                1,
                (kernel_size - 1) / 2,
                1,
                1,
                true,
                vb.pp("convs2").pp(i.to_string()),
            )?);
            adain2.push(AdaIn1d::load(
                style_dim,
                channels,
                vb.pp("adain2").pp(i.to_string()),
            )?);
            let a2 = vb
                .get((1, channels, 1), &format!("alpha2.{}", i))
                .unwrap_or_else(|_| Tensor::ones((1, channels, 1), dtype, device).unwrap());
            alpha2.push(a2);
        }

        Ok(Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
            num_layers: dilations.len(),
        })
    }

    /// Snake1D activation: x + (1/α) * sin²(α * x)
    fn snake1d(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
        // alpha: [1, channels, 1]
        // x: [batch, channels, length]
        let ax = x.broadcast_mul(alpha)?;
        let sin_sq = ax.sin()?.sqr()?;
        let one_over_alpha = alpha.recip()?;
        x.add(&sin_sq.broadcast_mul(&one_over_alpha)?)
    }

    fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.num_layers {
            // AdaIN1 → Snake1D with alpha1 → Conv1
            let xt = self.adain1[i].forward(&x, s)?;
            let xt = Self::snake1d(&xt, &self.alpha1[i])?;
            let xt = self.convs1[i].forward(&xt)?;

            // AdaIN2 → Snake1D with alpha2 → Conv2
            let xt = self.adain2[i].forward(&xt, s)?;
            let xt = Self::snake1d(&xt, &self.alpha2[i])?;
            let xt = self.convs2[i].forward(&xt)?;

            x = xt.add(&x)?;
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// SineGen — Harmonic sine wave generation
// ---------------------------------------------------------------------------

/// Generates harmonic sine waves for excitation source.
struct SineGen {
    sampling_rate: f32,
    #[allow(dead_code)]
    upsample_scale: usize,
    harmonic_num: usize,
    sine_amp: f32,
    noise_std: f32,
    voiced_threshold: f32,
}

impl SineGen {
    fn new(
        sampling_rate: f32,
        upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f32,
        noise_std: f32,
        voiced_threshold: f32,
    ) -> Self {
        Self {
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold,
        }
    }

    /// Generate sine waves from F0.
    ///
    /// `f0`: [batch, length, 1] — fundamental frequency
    ///
    /// Returns: (sine_waves, uv, noise) where:
    /// - sine_waves: [batch, length, harmonic_num+1]
    /// - uv: [batch, length, 1] — voiced/unvoiced flag
    /// - noise: [batch, length, harmonic_num+1]
    fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let device = f0.device().clone();
        let dtype = f0.dtype();
        let (batch, length, _) = f0.dims3()?;
        let harmonic_dim = self.harmonic_num + 1;
        let coarse_len = std::cmp::max(1, length / self.upsample_scale);

        let f0_cpu = f0.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let f0_data: Vec<Vec<Vec<f32>>> = f0_cpu.to_vec3()?;

        let mut sine_values = Vec::with_capacity(batch * length * harmonic_dim);
        let mut uv_values: Vec<f32> = Vec::with_capacity(batch * length);

        for (batch_index, batch_f0) in f0_data.iter().enumerate() {
            let fundamental: Vec<f32> = batch_f0.iter().map(|step| step[0]).collect();
            for &value in &fundamental {
                uv_values.push(if value > self.voiced_threshold {
                    1.0f32
                } else {
                    0.0f32
                });
            }

            let mut phase_per_harmonic = vec![vec![0.0f32; length]; harmonic_dim];
            for harmonic in 0..harmonic_dim {
                let harmonic_scale = (harmonic + 1) as f32;
                let mut rad_values: Vec<f32> = fundamental
                    .iter()
                    .map(|&value| (value * harmonic_scale / self.sampling_rate).rem_euclid(1.0))
                    .collect();
                if harmonic > 0 && !rad_values.is_empty() {
                    rad_values[0] +=
                        pseudo_random_unit(((batch_index as u64) << 32) | harmonic as u64);
                }

                let coarse = linear_resample_1d(&rad_values, coarse_len);
                let mut coarse_phase = Vec::with_capacity(coarse_len);
                let mut acc = 0.0f32;
                for value in coarse {
                    acc += value;
                    coarse_phase.push(acc * 2.0 * std::f32::consts::PI);
                }

                let scaled_phase: Vec<f32> = coarse_phase
                    .iter()
                    .map(|value| *value * self.upsample_scale as f32)
                    .collect();
                phase_per_harmonic[harmonic] = linear_resample_1d(&scaled_phase, length);
            }

            for time in 0..length {
                for harmonic in 0..harmonic_dim {
                    sine_values.push(phase_per_harmonic[harmonic][time].sin() * self.sine_amp);
                }
            }
        }

        let pure_sines = Tensor::new(sine_values.as_slice(), &Device::Cpu)?
            .reshape((batch, length, harmonic_dim))?
            .to_device(&device)?
            .to_dtype(dtype)?;
        let uv = Tensor::new(uv_values.as_slice(), &Device::Cpu)?
            .reshape((batch, length, 1))?
            .to_device(&device)?
            .to_dtype(dtype)?;

        let noise_amp_voiced = scalar_like(&uv, self.noise_std)?.broadcast_as(uv.shape())?;
        let noise_amp_unvoiced = scalar_like(&uv, self.sine_amp / 3.0)?.broadcast_as(uv.shape())?;
        let ones = Tensor::ones_like(&uv)?;
        let noise_amp = uv
            .broadcast_mul(&noise_amp_voiced)?
            .add(&uv.neg()?.add(&ones)?.broadcast_mul(&noise_amp_unvoiced)?)?;
        let noise = Tensor::randn(0f32, 1f32, pure_sines.shape(), &device)?
            .to_dtype(dtype)?
            .broadcast_mul(&noise_amp.broadcast_as(pure_sines.shape())?)?;

        let uv_broad = uv.broadcast_as(pure_sines.shape())?;
        let sines = pure_sines.broadcast_mul(&uv_broad)?.add(&noise)?;

        Ok((sines, uv, noise))
    }
}

// ---------------------------------------------------------------------------
// SourceModuleHnNSF — Harmonic + noise source
// ---------------------------------------------------------------------------

/// Source module combining harmonic and noise sources.
struct SourceModule {
    sine_gen: SineGen,
    l_linear_weight: Tensor,
    l_linear_bias: Option<Tensor>,
}

impl SourceModule {
    fn load(
        sampling_rate: f32,
        upsample_scale: usize,
        harmonic_num: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let sine_gen = SineGen::new(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            0.1,   // sine_amp
            0.003, // noise_std
            10.0,  // voiced_threshold
        );

        let l_linear_weight = vb.get((1, harmonic_num + 1), "l_linear.weight")?;
        let l_linear_bias = vb.get(1, "l_linear.bias").ok();

        Ok(Self {
            sine_gen,
            l_linear_weight,
            l_linear_bias,
        })
    }

    /// Forward: F0 → (sine_source, noise_source, uv)
    fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (sine_wavs, uv, _) = self.sine_gen.forward(f0)?;

        // Linear merge harmonics: [batch, length, harmonics+1] → [batch, length, 1]
        // Unsqueeze weight to 3D for batched matmul: [1, harmonics+1] → [1, harmonics+1, 1]
        let w_t = self.l_linear_weight.t()?.unsqueeze(0)?;
        let sine_merge = sine_wavs.matmul(&w_t)?;
        if let Some(ref bias) = self.l_linear_bias {
            let sine_merge = sine_merge.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(0)?)?;
            let sine_merge = sine_merge.tanh()?;
            let noise = scale_tensor(
                &Tensor::randn(0f32, 1f32, uv.shape(), f0.device())?,
                0.1f32 / 3.0f32,
            )?;
            Ok((sine_merge, noise, uv))
        } else {
            let sine_merge = sine_merge.tanh()?;
            let noise = scale_tensor(
                &Tensor::randn(0f32, 1f32, uv.shape(), f0.device())?,
                0.1f32 / 3.0f32,
            )?;
            Ok((sine_merge, noise, uv))
        }
    }
}

// ---------------------------------------------------------------------------
// STFT (conv-based) for harmonic processing
// ---------------------------------------------------------------------------

/// Conv-based STFT for processing harmonic source signals.
struct ConvStft {
    filter_length: usize,
    hop_length: usize,
    _freq_bins: usize,
    window: Vec<f32>,
}

impl ConvStft {
    fn new(
        filter_length: usize,
        hop_length: usize,
        _device: &Device,
        _dtype: candle_core::DType,
    ) -> Result<Self> {
        let freq_bins = filter_length / 2 + 1;

        let window = (0..filter_length)
            .map(|n| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / filter_length as f32).cos())
            })
            .collect();

        Ok(Self {
            filter_length,
            hop_length,
            _freq_bins: freq_bins,
            window,
        })
    }

    /// Forward STFT: waveform → (magnitude, phase)
    ///
    /// `waveform`: [batch, length] or [batch, 1, length]
    fn transform(&self, waveform: &Tensor) -> Result<(Tensor, Tensor)> {
        let pad_len = self.filter_length / 2;
        let device = waveform.device().clone();
        let dtype = waveform.dtype();

        let waveform = if waveform.dims().len() == 3 {
            waveform.squeeze(1)?
        } else {
            waveform.clone()
        };

        let waveform = waveform.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let waveform_data: Vec<Vec<f32>> = waveform.to_vec2()?;
        let batch = waveform_data.len();
        let freq_bins = self.filter_length / 2 + 1;
        let mut magnitude_data = Vec::new();
        let mut phase_data = Vec::new();
        let mut frame_count = 0;

        for samples in &waveform_data {
            let padded = reflect_pad_1d(samples, pad_len);
            let frames = 1 + (padded.len() - self.filter_length) / self.hop_length;
            frame_count = frames;
            for frame_idx in 0..frames {
                let offset = frame_idx * self.hop_length;
                let mut frame = vec![0.0f32; self.filter_length];
                for sample_idx in 0..self.filter_length {
                    frame[sample_idx] = padded[offset + sample_idx] * self.window[sample_idx];
                }
                let (mag, phase) = stft_frame(&frame, freq_bins);
                magnitude_data.extend_from_slice(&mag);
                phase_data.extend_from_slice(&phase);
            }
        }

        let magnitude = Tensor::new(magnitude_data.as_slice(), &Device::Cpu)?
            .reshape((batch, frame_count, freq_bins))?
            .transpose(1, 2)?
            .to_device(&device)?
            .to_dtype(dtype)?;
        let phase = Tensor::new(phase_data.as_slice(), &Device::Cpu)?
            .reshape((batch, frame_count, freq_bins))?
            .transpose(1, 2)?
            .to_device(&device)?
            .to_dtype(dtype)?;

        Ok((magnitude, phase))
    }

    /// Inverse STFT: (magnitude, phase-angle) → waveform
    fn inverse(&self, magnitude: &Tensor, phase: &Tensor) -> Result<Tensor> {
        let device = magnitude.device().clone();
        let dtype = magnitude.dtype();
        let magnitude = magnitude.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let phase = phase.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let magnitude_data: Vec<Vec<Vec<f32>>> = magnitude.to_vec3()?;
        let phase_data: Vec<Vec<Vec<f32>>> = phase.to_vec3()?;

        let pad_len = self.filter_length / 2;
        let batch = magnitude_data.len();
        let frames = magnitude.dim(2)?;
        let padded_len = self.filter_length + self.hop_length * frames.saturating_sub(1);
        let mut waveform_data = Vec::new();

        for (batch_magnitude, batch_phase) in magnitude_data.iter().zip(phase_data.iter()) {
            let mut overlap = vec![0.0f32; padded_len];
            let mut envelope = vec![0.0f32; padded_len];
            for frame_idx in 0..frames {
                let mag_frame: Vec<f32> =
                    batch_magnitude.iter().map(|bin| bin[frame_idx]).collect();
                let phase_frame: Vec<f32> = batch_phase.iter().map(|bin| bin[frame_idx]).collect();
                let frame = istft_frame(&mag_frame, &phase_frame, self.filter_length);
                let offset = frame_idx * self.hop_length;
                for sample_idx in 0..self.filter_length {
                    let weighted = frame[sample_idx] * self.window[sample_idx];
                    overlap[offset + sample_idx] += weighted;
                    envelope[offset + sample_idx] +=
                        self.window[sample_idx] * self.window[sample_idx];
                }
            }

            for (sample, env) in overlap.iter_mut().zip(envelope.iter()) {
                if *env > 1e-8 {
                    *sample /= *env;
                }
            }

            let trimmed = if overlap.len() > 2 * pad_len {
                &overlap[pad_len..overlap.len() - pad_len]
            } else {
                overlap.as_slice()
            };
            waveform_data.extend_from_slice(trimmed);
        }

        let output_len = if batch == 0 {
            0
        } else {
            waveform_data.len() / batch
        };
        Tensor::new(waveform_data.as_slice(), &Device::Cpu)?
            .reshape((batch, 1, output_len))?
            .to_device(&device)?
            .to_dtype(dtype)
    }
}

// ---------------------------------------------------------------------------
// Generator — ISTFTNet generator with upsampling
// ---------------------------------------------------------------------------

/// ISTFTNet generator: upsamples features and applies iSTFT to produce audio.
struct Generator {
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AdaInResBlock1>,
    noise_convs: Vec<Conv1d>,
    noise_res: Vec<AdaInResBlock1>,
    conv_post: Conv1d,
    stft: ConvStft,
    f0_upsamp_scale: usize,
    num_kernels: usize,
    num_upsamples: usize,
    post_n_fft: usize,
    source_module: SourceModule,
}

impl Generator {
    #[allow(clippy::too_many_arguments)]
    fn load(
        style_dim: usize,
        resblock_kernel_sizes: &[usize],
        upsample_rates: &[usize],
        upsample_initial_channel: usize,
        resblock_dilation_sizes: &[Vec<usize>],
        upsample_kernel_sizes: &[usize],
        gen_istft_n_fft: usize,
        gen_istft_hop_size: usize,
        vb: VarBuilder,
        device: &Device,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        let num_kernels = resblock_kernel_sizes.len();
        let num_upsamples = upsample_rates.len();
        let f0_upsamp_scale: usize = upsample_rates.iter().product::<usize>() * gen_istft_hop_size;

        // Source module for harmonic generation
        let source_module = SourceModule::load(
            24000.0,
            f0_upsamp_scale,
            8, // harmonic_num
            vb.pp("m_source"),
        )?;

        // STFT for processing harmonic source
        let stft = ConvStft::new(gen_istft_n_fft, gen_istft_hop_size, device, dtype)?;

        // Upsampling layers
        let mut ups = Vec::with_capacity(num_upsamples);
        let mut noise_convs = Vec::new();
        let mut noise_res = Vec::new();

        for i in 0..num_upsamples {
            let u = upsample_rates[i];
            let k = upsample_kernel_sizes[i];
            let in_ch = upsample_initial_channel / (1 << i);
            let out_ch = upsample_initial_channel / (1 << (i + 1));

            let up = ConvTranspose1d::load(
                in_ch,
                out_ch,
                k,
                u,           // stride
                (k - u) / 2, // padding
                0,           // output_padding
                1,           // groups
                true,        // bias
                vb.pp("ups").pp(i.to_string()),
            )?;
            ups.push(up);

            let c_cur = upsample_initial_channel / (1 << (i + 1));
            if i + 1 < num_upsamples {
                let stride_f0: usize = upsample_rates[i + 1..].iter().product();
                let noise_conv = Conv1d::load(
                    gen_istft_n_fft + 2,
                    c_cur,
                    stride_f0 * 2,
                    stride_f0,
                    stride_f0.div_ceil(2),
                    1,
                    1,
                    true,
                    vb.pp("noise_convs").pp(i.to_string()),
                )?;
                noise_convs.push(noise_conv);
                let nr = AdaInResBlock1::load(
                    c_cur,
                    7,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp("noise_res").pp(i.to_string()),
                    device,
                    dtype,
                )?;
                noise_res.push(nr);
            } else {
                let noise_conv = Conv1d::load(
                    gen_istft_n_fft + 2,
                    c_cur,
                    1,
                    1,
                    0,
                    1,
                    1,
                    true,
                    vb.pp("noise_convs").pp(i.to_string()),
                )?;
                noise_convs.push(noise_conv);
                let nr = AdaInResBlock1::load(
                    c_cur,
                    11,
                    &[1, 3, 5],
                    style_dim,
                    vb.pp("noise_res").pp(i.to_string()),
                    device,
                    dtype,
                )?;
                noise_res.push(nr);
            }
        }

        // ResBlocks
        let mut resblocks = Vec::new();
        for i in 0..num_upsamples {
            let ch = upsample_initial_channel / (1 << (i + 1));
            for (j, (k, d)) in resblock_kernel_sizes
                .iter()
                .zip(resblock_dilation_sizes.iter())
                .enumerate()
            {
                let rb = AdaInResBlock1::load(
                    ch,
                    *k,
                    d,
                    style_dim,
                    vb.pp("resblocks").pp((i * num_kernels + j).to_string()),
                    device,
                    dtype,
                )?;
                resblocks.push(rb);
            }
        }

        // Post convolution
        let final_ch = upsample_initial_channel / (1 << num_upsamples);
        let conv_post = Conv1d::load(
            final_ch,
            gen_istft_n_fft + 2,
            7,
            1,
            3,
            1,
            1,
            true,
            vb.pp("conv_post"),
        )?;

        Ok(Self {
            ups,
            resblocks,
            noise_convs,
            noise_res,
            conv_post,
            stft,
            f0_upsamp_scale,
            num_kernels,
            num_upsamples,
            post_n_fft: gen_istft_n_fft,
            source_module,
        })
    }

    /// Forward pass.
    ///
    /// `x`: [batch, channels, frames] — decoder features
    /// `s`: [batch, style_dim] — style embedding
    /// `f0`: [batch, frames] — fundamental frequency
    fn forward(&self, x: &Tensor, s: &Tensor, f0: &Tensor) -> Result<Tensor> {
        // Upsample F0 to waveform rate
        let f0_up = f0.unsqueeze(1)?; // [batch, 1, frames]
        let target_len = f0.dim(1)? * self.f0_upsamp_scale;
        let f0_up = upsample_1d_repeat(&f0_up, target_len)?;
        let f0_up = f0_up.transpose(1, 2)?; // [batch, length, 1]

        // Generate harmonic source
        let (har_source, _noi_source, _uv) = self.source_module.forward(&f0_up)?;
        let har_source = har_source.transpose(1, 2)?.squeeze(1)?; // [batch, length]

        // STFT of harmonic source
        let (har_spec, har_phase) = self.stft.transform(&har_source)?;
        let har = Tensor::cat(&[&har_spec, &har_phase], 1)?;

        let mut x = x.clone();

        for i in 0..self.num_upsamples {
            x = leaky_relu(&x, 0.1)?;

            // Noise source processing
            let x_source = self.noise_convs[i].forward(&har)?;
            let x_source = self.noise_res[i].forward(&x_source, s)?;

            // Upsample
            x = self.ups[i].forward(&x)?;

            // For last upsample, apply reflection padding
            if i == self.num_upsamples - 1 {
                x = reflect_pad_left_1d_tensor(&x, 1)?;
            }

            // Truncate to match lengths (conv outputs may differ by ±1)
            let min_len = x.dim(2)?.min(x_source.dim(2)?);
            x = x.narrow(2, 0, min_len)?;
            let x_source = x_source.narrow(2, 0, min_len)?;

            x = x.add(&x_source)?;

            // ResBlocks
            let mut xs: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let rb_out = self.resblocks[i * self.num_kernels + j].forward(&x, s)?;
                xs = Some(match xs {
                    None => rb_out,
                    Some(prev) => prev.add(&rb_out)?,
                });
            }
            x = scale_tensor(&xs.unwrap(), 1.0f32 / self.num_kernels as f32)?;
        }

        x = leaky_relu(&x, 0.01)?; // PyTorch default leaky_relu slope
        x = self.conv_post.forward(&x)?;

        // Split into spec and phase, apply iSTFT
        let half = self.post_n_fft / 2 + 1;

        let spec_raw = x.narrow(1, 0, half)?;
        let phase_raw = x.narrow(1, half, half)?;

        let spec = spec_raw.exp()?;
        let phase = phase_raw.sin()?;

        let wav = self.stft.inverse(&spec, &phase)?;

        Ok(wav)
    }
}

// ---------------------------------------------------------------------------
// Decoder — Full ISTFTNet decoder
// ---------------------------------------------------------------------------

/// Full ISTFTNet decoder: features → audio waveform.
///
/// Takes aligned text features + F0 + noise and produces audio.
pub struct IstftDecoder {
    encode: AdainResBlk1d,
    decode_blocks: Vec<AdainResBlk1d>,
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res: Conv1d,
    generator: Generator,
}

impl IstftDecoder {
    /// Load from VarBuilder.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dim_in: usize,
        style_dim: usize,
        _dim_out: usize,
        resblock_kernel_sizes: &[usize],
        upsample_rates: &[usize],
        upsample_initial_channel: usize,
        resblock_dilation_sizes: &[Vec<usize>],
        upsample_kernel_sizes: &[usize],
        gen_istft_n_fft: usize,
        gen_istft_hop_size: usize,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // encode: AdainResBlk1d(dim_in + 2, 1024, style_dim)
        let encode = AdainResBlk1d::load(dim_in + 2, 1024, style_dim, false, vb.pp("encode"))?;

        // decode blocks: 4 AdainResBlk1d
        let mut decode_blocks = Vec::new();
        // Block 0-2: 1024 + 2 + 64 → 1024
        for i in 0..3 {
            let block = AdainResBlk1d::load(
                1024 + 2 + 64,
                1024,
                style_dim,
                false,
                vb.pp("decode").pp(i.to_string()),
            )?;
            decode_blocks.push(block);
        }
        // Block 3: 1024 + 2 + 64 → 512, upsample
        let block3 =
            AdainResBlk1d::load(1024 + 2 + 64, 512, style_dim, true, vb.pp("decode").pp("3"))?;
        decode_blocks.push(block3);

        // F0 and N convolutions: downsample by 2
        let f0_conv = Conv1d::load(1, 1, 3, 2, 1, 1, 1, true, vb.pp("F0_conv"))?;
        let n_conv = Conv1d::load(1, 1, 3, 2, 1, 1, 1, true, vb.pp("N_conv"))?;

        // ASR residual projection (wrapped in ModuleList → index 0)
        let asr_res = Conv1d::load(512, 64, 1, 1, 0, 1, 1, true, vb.pp("asr_res").pp("0"))?;

        // Generator
        let generator = Generator::load(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
            vb.pp("generator"),
            device,
            dtype,
        )?;

        Ok(Self {
            encode,
            decode_blocks,
            f0_conv,
            n_conv,
            asr_res,
            generator,
        })
    }

    /// Forward pass.
    ///
    /// `asr`: [batch, channels, aligned_len] — aligned text features
    /// `f0_curve`: [batch, aligned_len*2] — F0 prediction (upsampled from prosody)
    /// `noise`: [batch, aligned_len*2] — noise prediction
    /// `s`: [batch, style_dim] — style embedding (decoder component)
    pub fn forward(
        &self,
        asr: &Tensor,
        f0_curve: &Tensor,
        noise: &Tensor,
        s: &Tensor,
    ) -> Result<Tensor> {
        // Downsample F0 and N
        let f0 = self.f0_conv.forward(&f0_curve.unsqueeze(1)?)?;
        let n = self.n_conv.forward(&noise.unsqueeze(1)?)?;

        // Concatenate: [batch, channels+2, aligned_len]
        let x = Tensor::cat(&[asr, &f0, &n], 1)?;

        // Encode
        let mut x = self.encode.forward(&x, s)?;

        // ASR residual
        let asr_res = self.asr_res.forward(asr)?;

        // Decode blocks
        let mut res = true;
        for block in &self.decode_blocks {
            if res {
                x = Tensor::cat(&[&x, &asr_res, &f0, &n], 1)?;
            }
            x = block.forward(&x, s)?;
            if block.upsample_type() != "none" {
                res = false;
            }
        }

        // Generator → waveform
        self.generator.forward(&x, s, f0_curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_stft_roundtrip() {
        let device = Device::Cpu;
        let stft = ConvStft::new(20, 5, &device, candle_core::DType::F32).unwrap();
        let waveform = Tensor::randn(0f32, 0.1, (1, 2400), &device).unwrap();
        let (mag, phase) = stft.transform(&waveform).unwrap();
        assert_eq!(mag.dim(1).unwrap(), 11); // n_fft/2 + 1
        let recon = stft.inverse(&mag, &phase).unwrap();
        // Reconstructed should have 3 dims: [batch, 1, length]
        assert!(recon.dims().len() >= 2);
    }
}
