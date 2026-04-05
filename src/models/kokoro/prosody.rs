//! Kokoro prosody predictor.
//!
//! Predicts duration, F0 (fundamental frequency), and noise parameters
//! from encoded text features and style embeddings.
//!
//! Architecture from StyleTTS2:
//! - DurationEncoder: LSTM + AdaLayerNorm
//! - ProsodyPredictor: DurationEncoder + duration/F0/noise prediction heads

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::layers::conv::{AdaIn1d, AdaLayerNorm, Conv1d, ConvTranspose1d, LinearNorm};
use crate::layers::lstm::Lstm;

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

// ---------------------------------------------------------------------------
// AdainResBlk1d — Residual block with AdaIN (used in F0/N prediction)
// ---------------------------------------------------------------------------

/// Residual block with Adaptive Instance Normalization.
pub struct AdainResBlk1d {
    conv1: Conv1d,
    conv2: Conv1d,
    norm1: AdaIn1d,
    norm2: AdaIn1d,
    /// Channel-matching 1×1 shortcut (when dim_in != dim_out).
    conv1x1: Option<Conv1d>,
    /// Depthwise transposed convolution used on the residual path when upsampling.
    pool: Option<ConvTranspose1d>,
    upsample: bool,
}

impl AdainResBlk1d {
    /// Load from VarBuilder.
    pub fn load(
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        upsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = Conv1d::load(dim_in, dim_out, 3, 1, 1, 1, 1, true, vb.pp("conv1"))?;
        let conv2 = Conv1d::load(dim_out, dim_out, 3, 1, 1, 1, 1, true, vb.pp("conv2"))?;
        let norm1 = AdaIn1d::load(style_dim, dim_in, vb.pp("norm1"))?;
        let norm2 = AdaIn1d::load(style_dim, dim_out, vb.pp("norm2"))?;

        let conv1x1 = if dim_in != dim_out {
            Some(Conv1d::load(
                dim_in,
                dim_out,
                1,
                1,
                0,
                1,
                1,
                false,
                vb.pp("conv1x1"),
            )?)
        } else {
            None
        };

        // Upstream uses a depthwise ConvTranspose1d on the residual path.
        let pool = if upsample {
            Some(ConvTranspose1d::load(
                dim_in,
                dim_in,
                3,
                2,
                1,
                1,
                dim_in,
                true,
                vb.pp("pool"),
            )?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            conv2,
            norm1,
            norm2,
            conv1x1,
            pool,
            upsample,
        })
    }

    fn shortcut(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        if self.upsample {
            let (_b, _c, length) = out.dims3()?;
            out = upsample_1d_repeat(&out, length * 2)?;
        }
        if let Some(ref sc) = self.conv1x1 {
            out = sc.forward(&out)?;
        }
        Ok(out)
    }

    fn residual(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut out = self.norm1.forward(x, s)?;
        out = leaky_relu(&out, 0.2)?;
        if let Some(ref pool) = self.pool {
            out = pool.forward(&out)?;
        }
        out = self.conv1.forward(&out)?;
        out = self.norm2.forward(&out, s)?;
        out = leaky_relu(&out, 0.2)?;
        self.conv2.forward(&out)
    }

    /// Forward pass.
    /// `x`: [batch, channels, length]
    /// `s`: [batch, style_dim]
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let residual = self.shortcut(x)?;
        let out = self.residual(x, s)?;
        let combined = out.add(&residual)?;
        scale_tensor(&combined, std::f32::consts::FRAC_1_SQRT_2)
    }

    /// Whether this block upsamples.
    pub fn upsample_type(&self) -> &str {
        if self.upsample {
            "nearest"
        } else {
            "none"
        }
    }
}

// ---------------------------------------------------------------------------
// DurationEncoder — LSTM + AdaLayerNorm for duration encoding
// ---------------------------------------------------------------------------

/// Duration encoder: LSTM layers interleaved with AdaLayerNorm.
pub struct DurationEncoder {
    lstms: Vec<Lstm>,
    ada_norms: Vec<AdaLayerNorm>,
    sty_dim: usize,
}

impl DurationEncoder {
    /// Load from VarBuilder.
    pub fn load(
        sty_dim: usize,
        d_model: usize,
        nlayers: usize,
        vb: VarBuilder,
        _device: &Device,
    ) -> Result<Self> {
        let mut lstms = Vec::with_capacity(nlayers);
        let mut ada_norms = Vec::with_capacity(nlayers);

        for i in 0..nlayers {
            let lstm = Lstm::load(
                1,                 // num_layers
                d_model + sty_dim, // input_size
                d_model / 2,       // hidden_size
                true,              // bidirectional → output = d_model
                vb.pp("lstms").pp((i * 2).to_string()),
            )?;
            lstms.push(lstm);

            let norm =
                AdaLayerNorm::load(sty_dim, d_model, vb.pp("lstms").pp((i * 2 + 1).to_string()))?;
            ada_norms.push(norm);
        }

        Ok(Self {
            lstms,
            ada_norms,
            sty_dim,
        })
    }

    /// Forward pass.
    ///
    /// `x`: [batch, channels, seq_len] — text features (transposed)
    /// `style`: [batch, style_dim] — style embedding
    /// `text_lengths`: [batch] — sequence lengths
    /// `mask`: [batch, seq_len] — True for padded positions
    ///
    /// Returns: [batch, seq_len, d_model + sty_dim]
    pub fn forward(
        &self,
        x: &Tensor,
        style: &Tensor,
        _text_lengths: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch, _channels, seq_len) = x.dims3()?;

        // x: [batch, channels, seq_len] → [seq_len, batch, channels]
        let mut x = x.permute((2, 0, 1))?;

        // style: [batch, sty_dim] → [seq_len, batch, sty_dim]
        let s = style
            .unsqueeze(0)?
            .broadcast_as((seq_len, batch, self.sty_dim))?;

        // Concatenate x and style along last dim
        x = Tensor::cat(&[&x, &s], 2)?;

        // Apply mask: [batch, seq_len] → [seq_len, batch, 1]
        let mask_f = mask.to_dtype(x.dtype())?.transpose(0, 1)?.unsqueeze(2)?;
        let inv_mask = mask_f.neg()?.add(&Tensor::ones_like(&mask_f)?)?;
        x = x.broadcast_mul(&inv_mask)?;

        // Transpose to [batch, seq_len, d_model + sty_dim]
        x = x.transpose(0, 1)?;

        // Run through LSTM + AdaLayerNorm pairs
        for (lstm, norm) in self.lstms.iter().zip(self.ada_norms.iter()) {
            // x: [batch, seq_len, d_model + sty_dim] → LSTM → [batch, seq_len, d_model]
            x = lstm.forward(&x)?;

            // Apply AdaLayerNorm: [batch, seq_len, d_model]
            x = norm.forward(&x, style)?;

            // Re-concatenate style
            let s_batch = style
                .unsqueeze(1)?
                .broadcast_as((batch, seq_len, self.sty_dim))?;
            x = Tensor::cat(&[&x, &s_batch], 2)?;

            // Apply mask
            let batch_mask = mask.unsqueeze(2)?.to_dtype(x.dtype())?;
            let inv = batch_mask.neg()?.add(&Tensor::ones_like(&batch_mask)?)?;
            x = x.broadcast_mul(&inv)?;
        }

        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// ProsodyPredictor — Duration + F0 + Noise prediction
// ---------------------------------------------------------------------------

/// Full prosody predictor: predicts duration, F0, and noise.
pub struct ProsodyPredictor {
    /// Duration encoder (LSTM + AdaLayerNorm).
    pub text_encoder: DurationEncoder,
    /// LSTM for duration refinement.
    pub lstm: Lstm,
    /// Duration projection (to max_dur classes).
    pub duration_proj: LinearNorm,
    /// Shared LSTM for F0 and noise.
    pub shared: Lstm,
    /// F0 prediction blocks.
    pub f0_blocks: Vec<AdainResBlk1d>,
    /// F0 projection to single channel.
    pub f0_proj: Conv1d,
    /// Noise prediction blocks.
    pub n_blocks: Vec<AdainResBlk1d>,
    /// Noise projection to single channel.
    pub n_proj: Conv1d,
}

impl ProsodyPredictor {
    /// Load from VarBuilder.
    pub fn load(
        style_dim: usize,
        d_hid: usize,
        nlayers: usize,
        max_dur: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let text_encoder =
            DurationEncoder::load(style_dim, d_hid, nlayers, vb.pp("text_encoder"), device)?;

        // LSTM for duration
        let lstm = Lstm::load(1, d_hid + style_dim, d_hid / 2, true, vb.pp("lstm"))?;

        let duration_proj =
            LinearNorm::load(d_hid, max_dur, vb.pp("duration_proj").pp("linear_layer"))?;

        // Shared LSTM for F0/N
        let shared = Lstm::load(1, d_hid + style_dim, d_hid / 2, true, vb.pp("shared"))?;

        // F0 prediction: 2 AdainResBlk1d blocks + Conv1d projection
        let f0_0 = AdainResBlk1d::load(d_hid, d_hid, style_dim, false, vb.pp("F0").pp("0"))?;
        let f0_1 = AdainResBlk1d::load(
            d_hid,
            d_hid / 2,
            style_dim,
            true, // upsample
            vb.pp("F0").pp("1"),
        )?;
        let f0_2 =
            AdainResBlk1d::load(d_hid / 2, d_hid / 2, style_dim, false, vb.pp("F0").pp("2"))?;
        let f0_proj = Conv1d::load(d_hid / 2, 1, 1, 1, 0, 1, 1, true, vb.pp("F0_proj"))?;

        // Noise prediction: same structure as F0
        let n_0 = AdainResBlk1d::load(d_hid, d_hid, style_dim, false, vb.pp("N").pp("0"))?;
        let n_1 = AdainResBlk1d::load(
            d_hid,
            d_hid / 2,
            style_dim,
            true, // upsample
            vb.pp("N").pp("1"),
        )?;
        let n_2 = AdainResBlk1d::load(d_hid / 2, d_hid / 2, style_dim, false, vb.pp("N").pp("2"))?;
        let n_proj = Conv1d::load(d_hid / 2, 1, 1, 1, 0, 1, 1, true, vb.pp("N_proj"))?;

        Ok(Self {
            text_encoder,
            lstm,
            duration_proj,
            shared,
            f0_blocks: vec![f0_0, f0_1, f0_2],
            f0_proj,
            n_blocks: vec![n_0, n_1, n_2],
            n_proj,
        })
    }

    /// Predict durations from encoded text features.
    ///
    /// `d`: [batch, seq_len, d_hid + style_dim] — output from DurationEncoder
    /// `s`: [batch, style_dim] — style embedding (unused here; already fused into `d`)
    ///
    /// Returns: [batch, seq_len] — sigmoid duration sums per token
    pub fn predict_duration(&self, d: &Tensor, _s: &Tensor) -> Result<Tensor> {
        // Upstream DurationEncoder already returns d_hid + style_dim features.
        let x = self.lstm.forward(d)?;

        // Duration projection: [batch, seq_len, d_hid] → [batch, seq_len, max_dur]
        let dur = self.duration_proj.forward(&x)?;

        // Sigmoid → sum over last axis → [batch, seq_len]
        let dur = candle_nn::ops::sigmoid(&dur)?;
        dur.sum(2) // Sum over max_dur dimension
    }

    /// Predict F0 and noise from aligned features.
    ///
    /// `x`: [batch, d_hid + style_dim, aligned_len]
    /// `s`: [batch, style_dim]
    ///
    /// Returns: (f0, noise) — both [batch, aligned_len*2] (upsampled)
    pub fn f0_n_predict(&self, x: &Tensor, s: &Tensor) -> Result<(Tensor, Tensor)> {
        // Upstream alignment features already include the concatenated style channels.
        let x_t = x.transpose(1, 2)?;
        let shared_out = self.shared.forward(&x_t)?;

        // F0 path
        let mut f0 = shared_out.transpose(1, 2)?; // [batch, d_hid, seq_len]
        for block in &self.f0_blocks {
            f0 = block.forward(&f0, s)?;
        }
        let f0 = self.f0_proj.forward(&f0)?.squeeze(1)?; // [batch, aligned_len*2]

        // Noise path
        let mut n = shared_out.transpose(1, 2)?;
        for block in &self.n_blocks {
            n = block.forward(&n, s)?;
        }
        let n = self.n_proj.forward(&n)?.squeeze(1)?; // [batch, aligned_len*2]

        Ok((f0, n))
    }
}
