//! Grouped-Query Attention (GQA) layer.
//!
//! GQA is the attention variant used by Qwen 2 / Qwen 2.5 / Qwen 3
//! family models. It reduces KV-cache memory by sharing key/value heads
//! across multiple query heads.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::tensor_utils::{apply_rotary_emb, cpu_flash_attention, RmsNorm};

/// Configuration for a GQA layer, extracted from model config.
#[derive(Debug, Clone)]
pub struct GqaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub attention_bias: bool,
    pub cpu_flash_attention: bool,
}

impl GqaConfig {
    pub fn new(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
    ) -> Self {
        let head_dim = hidden_size / num_attention_heads;
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            attention_bias: false,
            cpu_flash_attention: true,
        }
    }

    /// Create a GqaConfig with an explicit head_dim (for models where
    /// head_dim != hidden_size / num_attention_heads, e.g. code predictor).
    pub fn with_head_dim(
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        rms_norm_eps: f64,
    ) -> Self {
        Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            max_position_embeddings,
            rope_theta,
            rms_norm_eps,
            attention_bias: false,
            cpu_flash_attention: true,
        }
    }

    pub fn with_attention_bias(mut self, attention_bias: bool) -> Self {
        self.attention_bias = attention_bias;
        self
    }

    pub fn with_cpu_flash_attention(mut self, cpu_flash_attention: bool) -> Self {
        self.cpu_flash_attention = cpu_flash_attention;
        self
    }
}

/// Grouped-Query Attention with rotary position embeddings.
pub struct GroupedQueryAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cpu_flash_attention: bool,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl GroupedQueryAttention {
    /// Load attention weights from a VarBuilder.
    ///
    /// Expected weight names under `vb`:
    /// - `q_proj.weight` — (num_heads * head_dim, hidden_size)
    /// - `k_proj.weight` — (num_kv_heads * head_dim, hidden_size)
    /// - `v_proj.weight` — (num_kv_heads * head_dim, hidden_size)
    /// - `o_proj.weight` — (hidden_size, num_heads * head_dim)
    pub fn load(config: &GqaConfig, vb: VarBuilder) -> Result<Self> {
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;

        let q_proj = if config.attention_bias {
            candle_nn::linear(config.hidden_size, q_dim, vb.pp("q_proj"))?
        } else {
            candle_nn::linear_no_bias(config.hidden_size, q_dim, vb.pp("q_proj"))?
        };
        let k_proj = if config.attention_bias {
            candle_nn::linear(config.hidden_size, kv_dim, vb.pp("k_proj"))?
        } else {
            candle_nn::linear_no_bias(config.hidden_size, kv_dim, vb.pp("k_proj"))?
        };
        let v_proj = if config.attention_bias {
            candle_nn::linear(config.hidden_size, kv_dim, vb.pp("v_proj"))?
        } else {
            candle_nn::linear_no_bias(config.hidden_size, kv_dim, vb.pp("v_proj"))?
        };
        let o_proj = candle_nn::linear_no_bias(q_dim, config.hidden_size, vb.pp("o_proj"))?;

        // Qwen3 models have per-head QK-norm; Qwen2 models do not.
        let q_norm = RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("q_norm")).ok();
        let k_norm = RmsNorm::load(config.head_dim, config.rms_norm_eps, vb.pp("k_norm")).ok();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            cpu_flash_attention: config.cpu_flash_attention,
            kv_cache: None,
        })
    }

    /// Forward pass with rotary embeddings and optional KV-cache.
    ///
    /// * `x`     — (batch, seq_len, hidden_size)
    /// * `cos`   — (max_seq, head_dim) precomputed RoPE cos
    /// * `sin`   — (max_seq, head_dim) precomputed RoPE sin
    /// * `start_pos` — position offset for KV-cache (0 for full sequence)
    /// * `mask`  — optional causal attention mask
    pub fn forward(
        &mut self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        start_pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, heads, seq_len, head_dim)
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK-norm if present (Qwen3-style)
        let q = if let Some(ref qn) = self.q_norm {
            qn.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref kn) = self.k_norm {
            kn.forward(&k)?
        } else {
            k
        };

        // Apply RoPE
        let cos_slice = cos
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin_slice = sin
            .narrow(0, start_pos, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let (q, k) = apply_rotary_emb(&q, &k, &cos_slice, &sin_slice)?;

        // Update KV-cache
        let (k, v) = if let Some((prev_k, prev_v)) = &self.kv_cache {
            let k = Tensor::cat(&[prev_k, &k], 2)?;
            let v = Tensor::cat(&[prev_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let attn_output = if self.cpu_flash_attention {
            if let Some(cpu_attn_output) =
            cpu_flash_attention(&q, &k, &v, mask, scale)?
            {
                cpu_attn_output
            } else if seq_len == 1
                && mask.is_none()
                && matches!(x.device(), candle_core::Device::Metal(_))
            {
                candle_nn::ops::sdpa(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    None,
                    false,
                    scale,
                    1.0,
                )?
            } else {
                // Repeat KV heads to match Q heads (GQA expansion)
                // Use repeat_interleave semantics: [h0,h0,h1,h1,...] not repeat: [h0..h7,h0..h7]
                // so that Q head i pairs with KV head i/repeat_factor.
                let repeat_factor = self.num_heads / self.num_kv_heads;
                let kv_len = k.dim(2)?;
                let k = if repeat_factor > 1 {
                    k.unsqueeze(2)?
                        .repeat(&[1, 1, repeat_factor, 1, 1])?
                        .reshape((batch, self.num_heads, kv_len, self.head_dim))?
                } else {
                    k
                };
                let v = if repeat_factor > 1 {
                    v.unsqueeze(2)?
                        .repeat(&[1, 1, repeat_factor, 1, 1])?
                        .reshape((batch, self.num_heads, kv_len, self.head_dim))?
                } else {
                    v
                };

                let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
                let attn_weights = if let Some(mask) = mask {
                    attn_weights.broadcast_add(mask)?
                } else {
                    attn_weights
                };

                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&v)?
            }
        } else if seq_len == 1
            && mask.is_none()
            && matches!(x.device(), candle_core::Device::Metal(_))
        {
            candle_nn::ops::sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                None,
                false,
                scale,
                1.0,
            )?
        } else {
            // Repeat KV heads to match Q heads (GQA expansion)
            // Use repeat_interleave semantics: [h0,h0,h1,h1,...] not repeat: [h0..h7,h0..h7]
            // so that Q head i pairs with KV head i/repeat_factor.
            let repeat_factor = self.num_heads / self.num_kv_heads;
            let kv_len = k.dim(2)?;
            let k = if repeat_factor > 1 {
                k.unsqueeze(2)?
                    .repeat(&[1, 1, repeat_factor, 1, 1])?
                    .reshape((batch, self.num_heads, kv_len, self.head_dim))?
            } else {
                k
            };
            let v = if repeat_factor > 1 {
                v.unsqueeze(2)?
                    .repeat(&[1, 1, repeat_factor, 1, 1])?
                    .reshape((batch, self.num_heads, kv_len, self.head_dim))?
            } else {
                v
            };

            let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale as f64, 0.0)?;
            let attn_weights = if let Some(mask) = mask {
                attn_weights.broadcast_add(mask)?
            } else {
                attn_weights
            };

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };

        // Reshape back: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, hidden)
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }

    /// Clear the KV-cache (call between independent sequences).
    pub fn clear_cache(&mut self) {
        self.kv_cache = None;
    }

    /// Snapshot the current KV-cache so another caller can restore it later.
    pub fn cache_state(&self) -> Option<(Tensor, Tensor)> {
        self.kv_cache.clone()
    }

    /// Restore a previously captured KV-cache.
    pub fn set_cache_state(&mut self, kv_cache: Option<(Tensor, Tensor)>) {
        self.kv_cache = kv_cache;
    }
}

impl std::fmt::Debug for GroupedQueryAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedQueryAttention")
            .field("num_heads", &self.num_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("has_cache", &self.kv_cache.is_some())
            .finish()
    }
}
