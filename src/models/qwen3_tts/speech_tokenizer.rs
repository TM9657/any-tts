//! Speech Tokenizer Decoder — converts discrete codec tokens to audio.
//!
//! Loaded from the separate `Qwen/Qwen3-TTS-Tokenizer-12Hz` model.
//!
//! Architecture (from actual weights):
//! ```text
//! codec_tokens (16 groups × seq_len)
//!     ↓
//! RVQ Dequantize: group-0 → rvq_first, groups 1-15 → rvq_rest
//!   ⊕ sum all residuals → (batch, 512, seq_len)
//!     ↓
//! Pre-Conv1d  (512 → 1024, k=3)
//!     ↓
//! Pre-Transformer (1024 → 512 → 8×TransformerLayer → 512 → 1024)
//!     ↓
//! Upsample Blocks × 2  (ConvNeXt-style, stride=2 each → 4× upsample)
//!     ↓
//! Decoder Sequential (SNAC-style):
//!   Conv1d(1024, 1536, k=7)
//!   4× { SnakeBeta → ConvTranspose1d → 3× ResBlock(SnakeBeta+Conv) }
//!     channels: 1536→768→384→192→96
//!     kernels:   16,  10,   8,   6
//!   SnakeBeta → Conv1d(96, 1, k=7) → output
//!     ↓
//! tanh clamp → 24 kHz mono PCM
//! ```

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module, VarBuilder,
};

use crate::tensor_utils::{
    apply_rotary_emb, cpu_flash_attention, precompute_rope_freqs, snake_beta,
};

/// Causal (left-only) padding for 1-D convolutions.
///
/// Pads `left` zeros on the left side of the last dimension.
/// Input shape: (batch, channels, seq_len) → (batch, channels, left + seq_len).
///
/// This matches the reference `F.pad(hidden_state, (self.padding, 0))`.
fn causal_pad(x: &Tensor, left: usize) -> Result<Tensor> {
    if left == 0 {
        return Ok(x.clone());
    }
    x.pad_with_zeros(2, left, 0)
}

/// Configuration for the speech tokenizer decoder.
#[derive(Debug, Clone)]
pub struct SpeechTokenizerConfig {
    /// Number of codebook groups (16 = 1 rvq_first + 15 rvq_rest).
    pub num_groups: usize,
    /// Number of entries per codebook.
    pub codebook_size: usize,
    /// Embedding dimension inside the VQ codebook.
    pub vq_embed_dim: usize,
    /// Quantizer feature dimension (input/output projection target).
    pub quantizer_dim: usize,
    /// Number of pre-transformer layers.
    pub pre_transformer_layers: usize,
    /// Pre-transformer hidden size.
    pub pre_transformer_hidden: usize,
    /// Number of attention heads in pre-transformer.
    pub pre_transformer_heads: usize,
    /// Number of KV heads in pre-transformer.
    pub pre_transformer_kv_heads: usize,
    /// Pre-transformer head dimension.
    pub pre_transformer_head_dim: usize,
    /// MLP intermediate size in pre-transformer.
    pub pre_transformer_intermediate: usize,
    /// Number of upsample stages.
    pub num_upsample_stages: usize,
    /// Decoder initial channel (after pre-conv/upsample).
    pub decoder_channels: usize,
    /// Transposed conv kernel sizes for each decoder stage.
    pub decoder_kernels: Vec<usize>,
    /// Channel multipliers for decoder.
    pub channel_sequence: Vec<usize>,
    /// Output sample rate.
    pub sample_rate: u32,
}

impl Default for SpeechTokenizerConfig {
    fn default() -> Self {
        Self {
            num_groups: 16,
            codebook_size: 2048,
            vq_embed_dim: 256,
            quantizer_dim: 512,
            pre_transformer_layers: 8,
            pre_transformer_hidden: 512,
            pre_transformer_heads: 16,
            pre_transformer_kv_heads: 16,
            pre_transformer_head_dim: 64,
            pre_transformer_intermediate: 1024,
            num_upsample_stages: 2,
            decoder_channels: 1024,
            decoder_kernels: vec![16, 10, 8, 6],
            channel_sequence: vec![1536, 768, 384, 192, 96],
            sample_rate: 24000,
        }
    }
}

// ─── RVQ Codebook ──────────────────────────────────────────────────────────

/// A single VQ codebook using EMA-style embedding_sum weights.
struct VqCodebook {
    /// Codebook embeddings (from `_codebook.embedding_sum`).
    /// Shape: (codebook_size, vq_embed_dim).
    embeddings: Tensor,
    /// Cluster usage counts for normalization.
    cluster_usage: Tensor,
}

impl VqCodebook {
    fn load(codebook_size: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let cb_vb = vb.pp("_codebook");
        let embeddings = cb_vb.get((codebook_size, embed_dim), "embedding_sum")?;
        let cluster_usage = cb_vb.get(codebook_size, "cluster_usage")?;
        // Cast to F32 if needed (weights may be BF16)
        let embeddings = embeddings.to_dtype(DType::F32)?;
        let cluster_usage = cluster_usage.to_dtype(DType::F32)?;
        Ok(Self {
            embeddings,
            cluster_usage,
        })
    }

    /// Look up embedding by indices, normalizing by cluster usage.
    fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        // Normalize: embed = embedding_sum / cluster_usage.unsqueeze(-1)
        let usage = self.cluster_usage.unsqueeze(1)?; // (codebook_size, 1)
        let usage = usage.clamp(1e-7, f64::MAX)?;
        let normalized = self.embeddings.broadcast_div(&usage)?;
        // Index: flatten → select → reshape
        let flat = indices.flatten_all()?;
        let looked_up = normalized.index_select(&flat, 0)?;
        let shape = indices.dims();
        let embed_dim = normalized.dim(1)?;
        let mut new_shape = shape.to_vec();
        new_shape.push(embed_dim);
        looked_up.reshape(new_shape.as_slice())
    }
}

/// Residual VQ layer with multiple codebook layers and input/output projection.
struct RvqLayer {
    output_proj: Conv1d,
    codebooks: Vec<VqCodebook>,
}

impl RvqLayer {
    fn load(
        num_layers: usize,
        codebook_size: usize,
        embed_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // output_proj: Conv1d(vq_embed_dim=256, quantizer_dim=512, 1) — no bias
        let output_proj =
            candle_nn::conv1d_no_bias(256, 512, 1, Conv1dConfig::default(), vb.pp("output_proj"))?;
        let vq_vb = vb.pp("vq");
        let mut codebooks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            codebooks.push(VqCodebook::load(
                codebook_size,
                embed_dim,
                vq_vb.pp(format!("layers.{}", i)),
            )?);
        }
        Ok(Self {
            output_proj,
            codebooks,
        })
    }

    /// Dequantize: look up all layers and sum → project to quantizer_dim.
    /// tokens shape: (batch, num_layers_in_this_rvq, seq_len)
    fn dequantize(&self, tokens: &Tensor) -> Result<Tensor> {
        let num_layers = self.codebooks.len();
        let (batch, _n_layers, seq_len) = tokens.dims3()?;

        // Sum all codebook lookups in VQ space (256-d)
        let mut summed = Tensor::zeros((batch, seq_len, 256), DType::F32, tokens.device())?;
        for i in 0..num_layers {
            let layer_tokens = tokens.i((.., i, ..))?; // (batch, seq_len)
            let emb = self.codebooks[i].forward(&layer_tokens)?; // (batch, seq_len, 256)
            summed = (&summed + &emb)?;
        }

        // Transpose to (batch, 256, seq_len) for Conv1d
        let h = summed.transpose(1, 2)?;
        // Cast to the output_proj weight dtype (may be BF16 on GPU) before Conv1d.
        let conv_dtype = self.output_proj.weight().dtype();
        let h = h.to_dtype(conv_dtype)?;
        // Project back to quantizer_dim (512)
        self.output_proj.forward(&h)
    }
}

// ─── Pre-Transformer ───────────────────────────────────────────────────────

/// A single transformer layer with layer scale (RmsNorm, no bias).
struct PreTransformerLayer {
    input_layernorm: crate::tensor_utils::RmsNorm,
    post_attention_layernorm: crate::tensor_utils::RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    self_attn_layer_scale: Tensor,
    mlp_layer_scale: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl PreTransformerLayer {
    fn load(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_layernorm =
            crate::tensor_utils::RmsNorm::load(hidden, 1e-5, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            crate::tensor_utils::RmsNorm::load(hidden, 1e-5, vb.pp("post_attention_layernorm"))?;

        let attn_vb = vb.pp("self_attn");
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let q_proj = candle_nn::linear_no_bias(hidden, q_dim, attn_vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(hidden, kv_dim, attn_vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(hidden, kv_dim, attn_vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(q_dim, hidden, attn_vb.pp("o_proj"))?;

        let mlp_vb = vb.pp("mlp");
        let gate_proj = candle_nn::linear_no_bias(hidden, intermediate, mlp_vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden, intermediate, mlp_vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(intermediate, hidden, mlp_vb.pp("down_proj"))?;

        let self_attn_layer_scale = vb.pp("self_attn_layer_scale").get(hidden, "scale")?;
        let mlp_layer_scale = vb.pp("mlp_layer_scale").get(hidden, "scale")?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            self_attn_layer_scale,
            mlp_layer_scale,
            num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        causal_mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Self-attention with pre-norm
        let residual = x;
        let h = self.input_layernorm.forward(x)?;

        let q = self.q_proj.forward(&h)?;
        let k = self.k_proj.forward(&h)?;
        let v = self.v_proj.forward(&h)?;

        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE to Q and K
        let cos_slice = rope_cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin_slice = rope_sin.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let (q, k) = apply_rotary_emb(&q, &k, &cos_slice, &sin_slice)?;

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let out = if let Some(cpu_attn_out) =
            cpu_flash_attention(&q, &k, &v, Some(causal_mask), scale)?
        {
            cpu_attn_out
        } else {
            let attn = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
            // Apply causal mask: (1, 1, seq, seq) → -inf for future positions
            let attn = attn.broadcast_add(causal_mask)?;
            let attn = candle_nn::ops::softmax_last_dim(&attn)?;
            attn.matmul(&v)?
        };
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        let out = self.o_proj.forward(&out)?;

        // Apply layer scale and residual
        let scale = self.self_attn_layer_scale.unsqueeze(0)?.unsqueeze(0)?;
        let x = (residual + out.broadcast_mul(&scale)?)?;

        // MLP with pre-norm
        let residual = &x;
        let h = self.post_attention_layernorm.forward(&x)?;
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(&h)?)?;
        let up = self.up_proj.forward(&h)?;
        let h = self.down_proj.forward(&(gate * up)?)?;

        let scale = self.mlp_layer_scale.unsqueeze(0)?.unsqueeze(0)?;
        residual + h.broadcast_mul(&scale)?
    }
}

/// Pre-transformer: projects 1024→512, runs transformer, projects 512→1024.
struct PreTransformer {
    input_proj: Linear,
    output_proj: Linear,
    layers: Vec<PreTransformerLayer>,
    norm: crate::tensor_utils::RmsNorm,
    /// Precomputed RoPE cosine: (max_seq, head_dim).
    rope_cos: Tensor,
    /// Precomputed RoPE sine: (max_seq, head_dim).
    rope_sin: Tensor,
}

impl PreTransformer {
    fn load(config: &SpeechTokenizerConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let input_proj = candle_nn::linear(
            config.decoder_channels,
            config.pre_transformer_hidden,
            vb.pp("input_proj"),
        )?;
        let output_proj = candle_nn::linear(
            config.pre_transformer_hidden,
            config.decoder_channels,
            vb.pp("output_proj"),
        )?;

        let mut layers = Vec::with_capacity(config.pre_transformer_layers);
        for i in 0..config.pre_transformer_layers {
            layers.push(PreTransformerLayer::load(
                config.pre_transformer_hidden,
                config.pre_transformer_heads,
                config.pre_transformer_kv_heads,
                config.pre_transformer_head_dim,
                config.pre_transformer_intermediate,
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        let norm =
            crate::tensor_utils::RmsNorm::load(config.pre_transformer_hidden, 1e-5, vb.pp("norm"))?;

        // Precompute RoPE for up to 4096 positions (speech tokenizer codec sequences).
        // Use the model's dtype (BF16 on Metal) to avoid dtype mismatch during RoPE application.
        let max_seq = 4096;
        let rope_dtype = vb.dtype();
        let (rope_cos, rope_sin) = precompute_rope_freqs(
            config.pre_transformer_head_dim,
            max_seq,
            10000.0, // standard RoPE theta
            device,
            rope_dtype,
        )?;

        Ok(Self {
            input_proj,
            output_proj,
            layers,
            norm,
            rope_cos,
            rope_sin,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, channels, seq_len)
        // Transpose to (batch, seq_len, channels) for transformer
        let h = x.transpose(1, 2)?;
        let h = self.input_proj.forward(&h)?;
        let seq_len = h.dim(1)?;

        // Build causal mask: (1, 1, seq_len, seq_len) with 0 for valid, -inf for future
        // Also applies sliding window of 72 (from config).
        let sliding_window: usize = 72;
        let mask_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if j <= i && i - j < sliding_window {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let causal_mask = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), h.device())?
            .to_dtype(h.dtype())?;

        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(&h, &self.rope_cos, &self.rope_sin, &causal_mask)?;
        }
        let h = self.norm.forward(&h)?;
        let h = self.output_proj.forward(&h)?;
        // Transpose back to (batch, channels, seq_len) — NO residual around transformer
        h.transpose(1, 2)
    }
}

// ─── ConvNeXt Block ────────────────────────────────────────────────────────

/// ConvNeXt v2-style block used in upsample stages.
struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm_weight: Tensor,
    norm_bias: Tensor,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(channels: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        // CausalConv1d depthwise: kernel=7, groups=channels → left_pad=6, no built-in padding
        let dwconv = candle_nn::conv1d(
            channels,
            channels,
            7,
            Conv1dConfig {
                padding: 0,
                groups: channels,
                ..Default::default()
            },
            vb.pp("dwconv.conv"),
        )?;
        let norm_weight = vb.pp("norm").get(channels, "weight")?;
        let norm_bias = vb.pp("norm").get(channels, "bias")?;
        let pwconv1 = candle_nn::linear(channels, intermediate, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(intermediate, channels, vb.pp("pwconv2"))?;
        let gamma = vb.get(channels, "gamma")?;

        Ok(Self {
            dwconv,
            norm_weight,
            norm_bias,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        // Causal left-pad for depthwise conv (kernel=7, left_pad=6)
        let h = self.dwconv.forward(&causal_pad(x, 6)?)?;
        // Layer norm over channels: transpose → normalize → transpose
        let h = h.transpose(1, 2)?; // (batch, seq, ch)
                                    // Manual LayerNorm
        let mean = h.mean_keepdim(candle_core::D::Minus1)?;
        let h_centered = h.broadcast_sub(&mean)?;
        let var = h_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let h_norm = h_centered.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        let norm_w = self.norm_weight.unsqueeze(0)?.unsqueeze(0)?;
        let norm_b = self.norm_bias.unsqueeze(0)?.unsqueeze(0)?;
        let h = h_norm.broadcast_mul(&norm_w)?.broadcast_add(&norm_b)?;
        let h = self.pwconv1.forward(&h)?;
        let h = candle_nn::Activation::Gelu.forward(&h)?;
        let h = self.pwconv2.forward(&h)?;
        let gamma = self.gamma.unsqueeze(0)?.unsqueeze(0)?;
        let h = h.broadcast_mul(&gamma)?;
        let h = h.transpose(1, 2)?; // back to (batch, ch, seq)
        residual + &h
    }
}

// ─── Upsample Stage ────────────────────────────────────────────────────────

/// Upsample stage: ConvTranspose1d (stride=2) + ConvNeXt block.
struct UpsampleStage {
    conv: ConvTranspose1d,
    convnext: ConvNeXtBlock,
}

impl UpsampleStage {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        // ConvTranspose1d(channels, channels, kernel=2, stride=2)
        let conv = candle_nn::conv_transpose1d(
            channels,
            channels,
            2,
            ConvTranspose1dConfig {
                stride: 2,
                ..Default::default()
            },
            vb.pp("0.conv"),
        )?;
        let convnext = ConvNeXtBlock::load(channels, channels * 4, vb.pp("1"))?;
        Ok(Self { conv, convnext })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv.forward(x)?;
        self.convnext.forward(&h)
    }
}

// ─── SNAC-style Decoder Blocks ─────────────────────────────────────────────

/// A residual block with dual SnakeBeta activations.
struct SnacResBlock {
    act1_alpha: Tensor,
    act1_beta: Tensor,
    conv1: Conv1d,
    act2_alpha: Tensor,
    act2_beta: Tensor,
    conv2: Conv1d,
    /// Left-pad amount for conv1 (causal): (kernel-1) * dilation.
    conv1_left_pad: usize,
}

impl SnacResBlock {
    fn load(channels: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let act1_alpha = vb.pp("act1").get(channels, "alpha")?;
        let act1_beta = vb.pp("act1").get(channels, "beta")?;
        // CausalConv1d: kernel=7, dilation=d, stride=1 → left_pad = (7-1)*d = 6*d
        let conv1_left_pad = 6 * dilation;
        let conv1 = candle_nn::conv1d(
            channels,
            channels,
            7,
            Conv1dConfig {
                padding: 0,
                dilation,
                ..Default::default()
            },
            vb.pp("conv1.conv"),
        )?;
        let act2_alpha = vb.pp("act2").get(channels, "alpha")?;
        let act2_beta = vb.pp("act2").get(channels, "beta")?;
        // conv2: kernel=1, no padding needed
        let conv2 = candle_nn::conv1d(
            channels,
            channels,
            1,
            Conv1dConfig::default(),
            vb.pp("conv2.conv"),
        )?;
        Ok(Self {
            act1_alpha,
            act1_beta,
            conv1,
            act2_alpha,
            act2_beta,
            conv2,
            conv1_left_pad,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let alpha1 = self.act1_alpha.unsqueeze(0)?.unsqueeze(2)?;
        let beta1 = self.act1_beta.unsqueeze(0)?.unsqueeze(2)?;
        let h = snake_beta(x, &alpha1, &beta1)?;
        // Causal left-pad for dilated conv1
        let h = self.conv1.forward(&causal_pad(&h, self.conv1_left_pad)?)?;
        let alpha2 = self.act2_alpha.unsqueeze(0)?.unsqueeze(2)?;
        let beta2 = self.act2_beta.unsqueeze(0)?.unsqueeze(2)?;
        let h = snake_beta(&h, &alpha2, &beta2)?;
        let h = self.conv2.forward(&h)?;
        x + &h
    }
}

/// A decoder stage: SnakeBeta → ConvTranspose1d → 3× ResBlock.
struct DecoderStage {
    snake_alpha_param: Tensor,
    snake_beta_param: Tensor,
    upsample_conv: ConvTranspose1d,
    res_blocks: Vec<SnacResBlock>,
}

impl DecoderStage {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        num_res_blocks: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let block_vb = vb.pp("block");
        // block.0 = SnakeBeta
        let snake_alpha_param = block_vb.pp("0").get(in_channels, "alpha")?;
        let snake_beta_param = block_vb.pp("0").get(in_channels, "beta")?;

        // block.1 = ConvTranspose1d (CausalTransConvNet)
        let stride = kernel_size / 2;
        let upsample_conv = candle_nn::conv_transpose1d(
            in_channels,
            out_channels,
            kernel_size,
            ConvTranspose1dConfig {
                stride,
                ..Default::default()
            },
            block_vb.pp("1.conv"),
        )?;

        // block.2, 3, 4 = ResBlocks with dilations 1, 3, 9
        let dilations = [1, 3, 9];
        if num_res_blocks > dilations.len() {
            return Err(candle_core::Error::Msg(format!(
                "Unsupported decoder stage with {num_res_blocks} residual blocks"
            )));
        }
        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for (i, dilation) in dilations.iter().copied().enumerate().take(num_res_blocks) {
            res_blocks.push(SnacResBlock::load(
                out_channels,
                dilation,
                block_vb.pp(format!("{}", i + 2)),
            )?);
        }

        Ok(Self {
            snake_alpha_param,
            snake_beta_param,
            upsample_conv,
            res_blocks,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let alpha = self.snake_alpha_param.unsqueeze(0)?.unsqueeze(2)?;
        let beta = self.snake_beta_param.unsqueeze(0)?.unsqueeze(2)?;
        let h = snake_beta(x, &alpha, &beta)?;
        let mut h = self.upsample_conv.forward(&h)?;
        // Trim right side for causal ConvTranspose1d
        let kernel_size = self.upsample_conv.weight().dim(2)?;
        let stride = kernel_size / 2;
        let right_pad = kernel_size - stride;
        if right_pad > 0 {
            let out_len = h.dim(2)?;
            h = h.narrow(2, 0, out_len - right_pad)?;
        }
        for block in &self.res_blocks {
            h = block.forward(&h)?;
        }
        Ok(h)
    }
}

// ─── Main Decoder ──────────────────────────────────────────────────────────

/// The Speech Tokenizer Decoder.
///
/// Converts discrete codec tokens (from Talker + Code Predictor) into a
/// continuous 24kHz audio waveform.
pub struct SpeechTokenizerDecoder {
    /// RVQ for group 0.
    rvq_first: RvqLayer,
    /// RVQ for groups 1..15.
    rvq_rest: RvqLayer,
    /// Pre-convolution: quantizer_dim → decoder_channels.
    pre_conv: Conv1d,
    /// Pre-transformer for refinement.
    pre_transformer: PreTransformer,
    /// Upsample stages (ConvNeXt-style).
    upsample_stages: Vec<UpsampleStage>,
    /// Initial decoder conv (expands channels).
    decoder_init_conv: Conv1d,
    /// Decoder stages (SnakeBeta + ConvTranspose + ResBlocks).
    decoder_stages: Vec<DecoderStage>,
    /// Final SnakeBeta alpha parameter.
    final_snake_alpha: Tensor,
    /// Final SnakeBeta beta parameter.
    final_snake_beta: Tensor,
    /// Final convolution → 1 channel.
    final_conv: Conv1d,
    /// Config reference.
    config: SpeechTokenizerConfig,
}

impl SpeechTokenizerDecoder {
    /// Load the speech tokenizer decoder from a VarBuilder.
    ///
    /// Expected weight prefix: `decoder.` from `Qwen/Qwen3-TTS-Tokenizer-12Hz`.
    pub fn load(config: &SpeechTokenizerConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let dec_vb = vb.pp("decoder");

        // ── Quantizer ──
        let q_vb = dec_vb.pp("quantizer");
        let rvq_first = RvqLayer::load(
            1,
            config.codebook_size,
            config.vq_embed_dim,
            q_vb.pp("rvq_first"),
        )?;
        let rvq_rest = RvqLayer::load(
            config.num_groups - 1,
            config.codebook_size,
            config.vq_embed_dim,
            q_vb.pp("rvq_rest"),
        )?;

        // ── Pre-conv: CausalConv1d(512, 1024, k=3) — left_pad=2, no built-in padding ──
        let pre_conv = candle_nn::conv1d(
            config.quantizer_dim,
            config.decoder_channels,
            3,
            Conv1dConfig {
                padding: 0,
                ..Default::default()
            },
            dec_vb.pp("pre_conv.conv"),
        )?;

        // ── Pre-transformer ──
        let pre_transformer = PreTransformer::load(config, dec_vb.pp("pre_transformer"), device)?;

        // ── Upsample stages ──
        let mut upsample_stages = Vec::with_capacity(config.num_upsample_stages);
        for i in 0..config.num_upsample_stages {
            upsample_stages.push(UpsampleStage::load(
                config.decoder_channels,
                dec_vb.pp(format!("upsample.{}", i)),
            )?);
        }

        // ── Decoder sequential ──
        let ddec_vb = dec_vb.pp("decoder");

        // decoder.0: CausalConv1d(1024, 1536, k=7) — left_pad=6, no built-in padding
        let decoder_init_conv = candle_nn::conv1d(
            config.decoder_channels,
            config.channel_sequence[0],
            7,
            Conv1dConfig {
                padding: 0,
                ..Default::default()
            },
            ddec_vb.pp("0.conv"),
        )?;

        // decoder stages 1-4
        let mut decoder_stages = Vec::with_capacity(config.decoder_kernels.len());
        for (i, &kernel) in config.decoder_kernels.iter().enumerate() {
            let in_ch = config.channel_sequence[i];
            let out_ch = config.channel_sequence[i + 1];
            decoder_stages.push(DecoderStage::load(
                in_ch,
                out_ch,
                kernel,
                3,
                ddec_vb.pp(format!("{}", i + 1)),
            )?);
        }

        // decoder.5: SnakeBeta (final activation before output conv)
        let last_ch = *config.channel_sequence.last().unwrap();
        let final_snake_alpha = ddec_vb.pp("5").get(last_ch, "alpha")?;
        let final_snake_beta = ddec_vb.pp("5").get(last_ch, "beta")?;

        // decoder.6: CausalConv1d(96, 1, k=7) — left_pad=6, no built-in padding
        let final_conv = candle_nn::conv1d(
            last_ch,
            1,
            7,
            Conv1dConfig {
                padding: 0,
                ..Default::default()
            },
            ddec_vb.pp("6.conv"),
        )?;

        Ok(Self {
            rvq_first,
            rvq_rest,
            pre_conv,
            pre_transformer,
            upsample_stages,
            decoder_init_conv,
            decoder_stages,
            final_snake_alpha,
            final_snake_beta,
            final_conv,
            config: config.clone(),
        })
    }

    /// Decode codec tokens into a 24kHz mono waveform.
    ///
    /// * `tokens` — (batch, num_groups, seq_len) integer tensor of codec indices.
    ///
    /// Returns a 1-D f32 tensor of audio samples in `[-1, 1]`.
    pub fn decode(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch, num_groups, _seq_len) = tokens.dims3()?;
        assert_eq!(
            num_groups, self.config.num_groups,
            "expected {} groups, got {}",
            self.config.num_groups, num_groups
        );

        // ── 1. RVQ Dequantize ─────────────────────────────────────────
        // Group 0 → rvq_first, Groups 1..15 → rvq_rest
        let first_tokens = tokens.i((.., 0..1, ..))?; // (batch, 1, seq_len)
        let rest_tokens = tokens.i((.., 1.., ..))?; // (batch, 15, seq_len)

        let first_features = self.rvq_first.dequantize(&first_tokens)?;
        let rest_features = self.rvq_rest.dequantize(&rest_tokens)?;
        let h = (&first_features + &rest_features)?; // (batch, 512, seq_len)

        // ── 2. Pre-conv (causal: left-pad 2) ──────────────────────────
        let h = self.pre_conv.forward(&causal_pad(&h, 2)?)?; // (batch, 1024, seq_len)

        // ── 3. Pre-transformer ────────────────────────────────────────
        let h = self.pre_transformer.forward(&h)?;

        // ── 4. Upsample stages ────────────────────────────────────────
        let mut h = h;
        for stage in &self.upsample_stages {
            h = stage.forward(&h)?; // 2× each → 4× total
        }

        // ── 5. Decoder sequential (init conv: causal left-pad 6) ──────
        h = self.decoder_init_conv.forward(&causal_pad(&h, 6)?)?;

        for stage in &self.decoder_stages {
            h = stage.forward(&h)?;
        }

        // ── 6. Final activation + conv (causal left-pad 6) ─────────────
        let alpha = self.final_snake_alpha.unsqueeze(0)?.unsqueeze(2)?;
        let beta = self.final_snake_beta.unsqueeze(0)?.unsqueeze(2)?;
        let h = snake_beta(&h, &alpha, &beta)?;
        let h = self.final_conv.forward(&causal_pad(&h, 6)?)?; // (batch, 1, samples)

        // ── 7. clamp to [-1, 1] (matches reference: .clamp(min=-1, max=1)) ─
        let h = h.clamp(-1f32, 1f32)?;

        // Squeeze channel dim
        let h = h.squeeze(1)?;

        if batch == 1 {
            h.squeeze(0)
        } else {
            Ok(h)
        }
    }

    /// Total temporal upsampling factor.
    pub fn upsample_factor(&self) -> usize {
        // 2 upsample stages (2× each = 4×) × decoder stage strides
        let upsample: usize = (0..self.config.num_upsample_stages).map(|_| 2).product();
        let decoder: usize = self.config.decoder_kernels.iter().map(|k| k / 2).product();
        upsample * decoder
    }
}

impl std::fmt::Debug for SpeechTokenizerDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechTokenizerDecoder")
            .field("num_groups", &self.config.num_groups)
            .field("codebook_size", &self.config.codebook_size)
            .field("vq_embed_dim", &self.config.vq_embed_dim)
            .field("upsample_factor", &self.upsample_factor())
            .finish()
    }
}
