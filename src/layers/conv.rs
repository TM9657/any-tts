//! Convolution building blocks for model implementations.
//!
//! Provides Conv1d, ConvTranspose1d, and related utilities used by
//! Kokoro's ISTFTNet decoder and text encoder.

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// 1D convolution layer.
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl Conv1d {
    /// Load a Conv1d layer from a VarBuilder.
    ///
    /// Weight shape: [out_channels, in_channels/groups, kernel_size]
    ///
    /// Supports both plain `weight` tensors and weight-normalized
    /// `weight_g` + `weight_v` pairs (from PyTorch `weight_norm`).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight_shape = (out_channels, in_channels / groups, kernel_size);
        let weight = match vb.get(weight_shape, "weight") {
            Ok(w) => w,
            Err(_) => {
                // Fall back to weight normalization: weight = g * (v / ||v||)
                let weight_v = vb.get(weight_shape, "weight_v")?;
                // PyTorch stores weight_g as [out, 1, 1]; try 3D first, then 1D
                let weight_g = vb.get((out_channels, 1, 1), "weight_g")
                    .or_else(|_| vb.get(out_channels, "weight_g"))?;
                apply_weight_norm(&weight_v, &weight_g)?
            }
        };
        let bias = if use_bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        })
    }

    /// Forward pass: input shape [batch, in_channels, length]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = if self.groups == 1 {
            x.conv1d(&self.weight, self.padding, self.stride, self.dilation, 1)?
        } else {
            // candle supports groups parameter
            x.conv1d(
                &self.weight,
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
            )?
        };
        match &self.bias {
            Some(bias) => {
                let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
                out.broadcast_add(&bias)
            }
            None => Ok(out),
        }
    }

    /// Get a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

/// 1D transposed convolution layer.
pub struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
}

impl ConvTranspose1d {
    /// Load a ConvTranspose1d from a VarBuilder.
    ///
    /// Weight shape: [in_channels, out_channels, kernel_size]
    ///
    /// Supports both plain `weight` tensors and weight-normalized
    /// `weight_g` + `weight_v` pairs.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight_shape = (in_channels, out_channels / groups, kernel_size);
        let weight = match vb.get(weight_shape, "weight") {
            Ok(w) => w,
            Err(_) => {
                let weight_v = vb.get(weight_shape, "weight_v")?;
                // PyTorch stores weight_g as [in, 1, 1]; try 3D first, then 1D
                let weight_g = vb.get((in_channels, 1, 1), "weight_g")
                    .or_else(|_| vb.get(in_channels, "weight_g"))?;
                apply_weight_norm(&weight_v, &weight_g)?
            }
        };
        let bias = if use_bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
        })
    }

    /// Forward pass: input shape [batch, in_channels, length]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.conv_transpose1d(
            &self.weight,
            self.padding,
            self.output_padding,
            self.stride,
            1, // dilation
            self.groups,
        )?;
        match &self.bias {
            Some(bias) => {
                let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
                out.broadcast_add(&bias)
            }
            None => Ok(out),
        }
    }
}

/// Instance Normalization 1D.
///
/// Normalizes each channel independently across the spatial dimension.
/// Formula: (x - mean) / sqrt(var + eps) * weight + bias
pub struct InstanceNorm1d {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f64,
}

impl InstanceNorm1d {
    /// Load from VarBuilder with affine parameters.
    pub fn load(num_features: usize, eps: f64, affine: bool, vb: VarBuilder) -> Result<Self> {
        let (weight, bias) = if affine {
            (
                Some(vb.get(num_features, "weight")?),
                Some(vb.get(num_features, "bias")?),
            )
        } else {
            (None, None)
        };
        Ok(Self { weight, bias, eps })
    }

    /// Forward pass: input shape [batch, channels, length]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;

        // Compute mean and variance per channel per batch element
        let mean = x_f32.mean_keepdim(2)?;
        let x_centered = x_f32.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(2)?;
        let x_normed = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;

        let mut result = x_normed.to_dtype(dtype)?;

        if let Some(ref weight) = self.weight {
            let w = weight.unsqueeze(0)?.unsqueeze(2)?;
            result = result.broadcast_mul(&w)?;
        }
        if let Some(ref bias) = self.bias {
            let b = bias.unsqueeze(0)?.unsqueeze(2)?;
            result = result.broadcast_add(&b)?;
        }

        Ok(result)
    }
}

/// Adaptive Instance Normalization (AdaIN).
///
/// Applies InstanceNorm then modulates with style-dependent gamma/beta.
/// Used in Kokoro's ISTFTNet decoder.
pub struct AdaIn1d {
    norm: InstanceNorm1d,
    fc: candle_nn::Linear,
}

impl AdaIn1d {
    /// Load from VarBuilder.
    ///
    /// The InstanceNorm inside AdaIN does NOT have learnable affine
    /// parameters — the style-dependent gamma/beta from `fc` provide
    /// the affine modulation.
    pub fn load(style_dim: usize, num_features: usize, vb: VarBuilder) -> Result<Self> {
        let norm = InstanceNorm1d::load(num_features, 1e-5, true, vb.pp("norm"))
            .or_else(|_| InstanceNorm1d::load(num_features, 1e-5, false, vb.pp("norm")))?;
        let fc = candle_nn::linear(style_dim, num_features * 2, vb.pp("fc"))?;
        Ok(Self { norm, fc })
    }

    /// Forward pass.
    ///
    /// `x`: [batch, channels, length]
    /// `s`: [batch, style_dim]
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(s)?;
        let h = h.unsqueeze(2)?;
        let half = h.dim(1)? / 2;
        let gamma = h.narrow(1, 0, half)?;
        let beta = h.narrow(1, half, half)?;

        let normed = self.norm.forward(x)?;
        let one = Tensor::ones_like(&gamma)?;
        normed
            .broadcast_mul(&one.add(&gamma)?)?
            .broadcast_add(&beta)
    }
}

/// Adaptive Layer Normalization.
///
/// Used in Kokoro's DurationEncoder.
pub struct AdaLayerNorm {
    fc: candle_nn::Linear,
    eps: f64,
    num_features: usize,
}

impl AdaLayerNorm {
    /// Load from VarBuilder.
    pub fn load(style_dim: usize, num_features: usize, vb: VarBuilder) -> Result<Self> {
        let fc = candle_nn::linear(style_dim, num_features * 2, vb.pp("fc"))?;
        Ok(Self {
            fc,
            eps: 1e-5,
            num_features,
        })
    }

    /// Forward pass.
    ///
    /// `x`: [batch, length, features]
    /// `s`: [batch, style_dim]
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(s)?;
        let half = self.num_features;
        let gamma = h.narrow(1, 0, half)?.unsqueeze(1)?;
        let beta = h.narrow(1, half, half)?.unsqueeze(1)?;

        // Layer normalization
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x_f32.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(dtype)?;

        let one = Tensor::ones_like(&gamma)?;
        x_normed
            .broadcast_mul(&one.add(&gamma)?)?
            .broadcast_add(&beta)
    }
}

/// Weight normalization wrapper.
///
/// Applies weight normalization to a weight tensor: w = g * (v / ||v||)
/// During inference, the normalized weight is precomputed.
pub fn apply_weight_norm(weight: &Tensor, weight_g: &Tensor) -> Result<Tensor> {
    // weight_g: [out_channels, 1, 1] or [out_channels]
    // weight_v: [out_channels, in_channels, kernel_size]
    let norm = weight
        .sqr()?
        .sum_keepdim(candle_core::D::Minus1)?
        .sum_keepdim(candle_core::D::Minus2)?
        .sqrt()?;
    let weight_g = if weight_g.dims().len() == 1 {
        weight_g.unsqueeze(1)?.unsqueeze(2)?
    } else {
        weight_g.clone()
    };
    weight.broadcast_div(&norm)?.broadcast_mul(&weight_g)
}

/// Channel-wise Layer Normalization.
///
/// Like LayerNorm but loads `gamma`/`beta` instead of `weight`/`bias`.
/// Used by Kokoro's ChannelLayerNorm in the text encoder CNN.
pub struct ChannelNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f64,
}

impl ChannelNorm {
    /// Load from VarBuilder (expects `gamma` and `beta` tensors).
    pub fn load(num_features: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get(num_features, "gamma")?;
        let beta = vb.get(num_features, "beta")?;
        Ok(Self {
            gamma,
            beta,
            eps: 1e-5,
        })
    }

    /// Forward pass: input shape [batch, features] or [batch, seq_len, features].
    /// Normalizes over the last dimension.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;
        let mean = x_f32.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x_f32.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(dtype)?;
        x_normed.broadcast_mul(&self.gamma)?.broadcast_add(&self.beta)
    }
}

/// Linear layer with weight normalization.
pub struct LinearNorm {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl LinearNorm {
    /// Load from VarBuilder.
    pub fn load(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        let bias = vb.get(out_features, "bias").ok();
        Ok(Self { weight, bias })
    }

    /// Forward pass: x shape [..., in_features] → [..., out_features]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w_t = self.weight.t()?;
        let out = x.broadcast_matmul(&w_t)?;
        match &self.bias {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_instance_norm_shape() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (2, 16, 100), &device).unwrap();

        let norm = InstanceNorm1d {
            weight: Some(Tensor::ones(16, candle_core::DType::F32, &device).unwrap()),
            bias: Some(Tensor::zeros(16, candle_core::DType::F32, &device).unwrap()),
            eps: 1e-5,
        };

        let out = norm.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 16, 100]);
    }

    #[test]
    fn test_linear_norm() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0f32, 0.1, (32, 16), &device).unwrap();
        let bias = Tensor::zeros(32, candle_core::DType::F32, &device).unwrap();
        let linear = LinearNorm {
            weight,
            bias: Some(bias),
        };
        let x = Tensor::randn(0f32, 1.0, (2, 10, 16), &device).unwrap();
        let out = linear.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 10, 32]);
    }
}
