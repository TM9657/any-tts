//! ALBERT (A Lite BERT) implementation for Kokoro's PL-BERT text encoder.
//!
//! ALBERT shares transformer parameters across all layers, significantly
//! reducing model size. Kokoro uses a modified ALBERT ("CustomAlbert")
//! for phoneme-level text encoding.
//!
//! Weight names follow PyTorch's `transformers.AlbertModel`:
//! - `embeddings.word_embeddings.weight`
//! - `embeddings.position_embeddings.weight`
//! - `embeddings.token_type_embeddings.weight`
//! - `embeddings.LayerNorm.weight/bias`
//! - `encoder.embedding_hidden_mapping_in.weight/bias`
//! - `encoder.albert_layer_groups.0.albert_layers.0.attention.*`
//! - `encoder.albert_layer_groups.0.albert_layers.0.ffn.*`

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::config::PlbertConfig;

/// ALBERT model for PL-BERT text encoding.
pub struct Albert {
    word_embeddings: candle_nn::Embedding,
    position_embeddings: candle_nn::Embedding,
    token_type_embeddings: candle_nn::Embedding,
    embed_layer_norm: candle_nn::LayerNorm,
    embedding_projection: Option<candle_nn::Linear>,
    shared_layer: AlbertLayer,
    num_hidden_layers: usize,
    _num_hidden_groups: usize,
    hidden_size: usize,
}

/// A single ALBERT transformer layer (shared across all layers).
struct AlbertLayer {
    attention: AlbertAttention,
    ffn: candle_nn::Linear,
    ffn_output: candle_nn::Linear,
    attention_norm: candle_nn::LayerNorm,
    output_norm: candle_nn::LayerNorm,
}

struct AlbertAttention {
    query: candle_nn::Linear,
    key: candle_nn::Linear,
    value: candle_nn::Linear,
    dense: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Albert {
    /// Load from a VarBuilder.
    pub fn load(config: &PlbertConfig, vb: VarBuilder, _device: &Device) -> Result<Self> {
        let emb_vb = vb.pp("embeddings");

        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.embedding_size,
            emb_vb.pp("word_embeddings"),
        )?;
        let position_embeddings = candle_nn::embedding(
            config.max_position_embeddings,
            config.embedding_size,
            emb_vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            config.type_vocab_size,
            config.embedding_size,
            emb_vb.pp("token_type_embeddings"),
        )?;
        let embed_layer_norm = candle_nn::layer_norm(
            config.embedding_size,
            candle_nn::LayerNormConfig::default(),
            emb_vb.pp("LayerNorm"),
        )?;

        // Projection from embedding_size to hidden_size (if different)
        let enc_vb = vb.pp("encoder");
        let embedding_projection = if config.embedding_size != config.hidden_size {
            Some(candle_nn::linear(
                config.embedding_size,
                config.hidden_size,
                enc_vb.pp("embedding_hidden_mapping_in"),
            )?)
        } else {
            None
        };

        // Shared transformer layer
        let layer_vb = enc_vb
            .pp("albert_layer_groups")
            .pp("0")
            .pp("albert_layers")
            .pp("0");

        let shared_layer = AlbertLayer::load(config, layer_vb)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embed_layer_norm,
            embedding_projection,
            shared_layer,
            num_hidden_layers: config.num_hidden_layers,
            _num_hidden_groups: config.num_hidden_groups,
            hidden_size: config.hidden_size,
        })
    }

    /// Forward pass.
    ///
    /// `input_ids`: [batch, seq_len] — phoneme token IDs
    /// `attention_mask`: [batch, seq_len] — 1 for valid, 0 for padded
    ///
    /// Returns: [batch, seq_len, hidden_size]
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;
        let device = input_ids.device();

        // Build position IDs
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        let position_ids = Tensor::new(position_ids.as_slice(), device)?
            .unsqueeze(0)?
            .broadcast_as(input_ids.shape())?;

        // Token type IDs (all zeros for single-segment)
        let token_type_ids = Tensor::zeros(input_ids.shape(), candle_core::DType::U32, device)?;

        // Embeddings
        let word_emb = self.word_embeddings.forward(input_ids)?;
        let pos_emb = self.position_embeddings.forward(&position_ids)?;
        let type_emb = self.token_type_embeddings.forward(&token_type_ids)?;

        let embeddings = word_emb.add(&pos_emb)?.add(&type_emb)?;
        let embeddings = self.embed_layer_norm.forward(&embeddings)?;

        // Project to hidden size if needed
        let mut hidden = match &self.embedding_projection {
            Some(proj) => proj.forward(&embeddings)?,
            None => embeddings,
        };

        // Build causal attention mask: [batch, 1, 1, seq_len]
        let attn_mask = attention_mask
            .to_dtype(hidden.dtype())?
            .unsqueeze(1)?
            .unsqueeze(2)?;
        // Convert 0/1 mask to additive mask: 0 → 0.0, 1 → -10000.0 inverted
        // Actually: 1 means attend, 0 means don't → we need (1-mask)*-10000
        let inv_mask = attn_mask.neg()?.add(&Tensor::ones_like(&attn_mask)?)?;
        let attn_bias = (inv_mask * -10000.0)?;

        // Apply shared layer N times
        for _ in 0..self.num_hidden_layers {
            hidden = self.shared_layer.forward(&hidden, &attn_bias)?;
        }

        Ok(hidden)
    }

    /// Hidden size of the output.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

impl AlbertLayer {
    fn load(config: &PlbertConfig, vb: VarBuilder) -> Result<Self> {
        let attn_vb = vb.pp("attention");
        let attention = AlbertAttention::load(config, attn_vb)?;

        let ffn = candle_nn::linear(config.hidden_size, config.intermediate_size, vb.pp("ffn"))?;
        let ffn_output = candle_nn::linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("ffn_output"),
        )?;

        let attention_norm = candle_nn::layer_norm(
            config.hidden_size,
            candle_nn::LayerNormConfig::default(),
            vb.pp("attention").pp("LayerNorm"),
        )?;
        let output_norm = candle_nn::layer_norm(
            config.hidden_size,
            candle_nn::LayerNormConfig::default(),
            vb.pp("full_layer_layer_norm"),
        )?;

        Ok(Self {
            attention,
            ffn,
            ffn_output,
            attention_norm,
            output_norm,
        })
    }

    fn forward(&self, hidden: &Tensor, attn_bias: &Tensor) -> Result<Tensor> {
        // Self-attention + residual + layer norm
        let attn_out = self.attention.forward(hidden, attn_bias)?;
        let hidden = self.attention_norm.forward(&hidden.add(&attn_out)?)?;

        // FFN + residual + layer norm
        let ffn_out = self.ffn.forward(&hidden)?;
        let ffn_out = ffn_out.gelu_erf()?;
        let ffn_out = self.ffn_output.forward(&ffn_out)?;
        let hidden = self.output_norm.forward(&hidden.add(&ffn_out)?)?;

        Ok(hidden)
    }
}

impl AlbertAttention {
    fn load(config: &PlbertConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;

        let query = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("query"))?;
        let key = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("key"))?;
        let value = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("value"))?;
        let dense = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;

        Ok(Self {
            query,
            key,
            value,
            dense,
            num_heads: config.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden: &Tensor, attn_bias: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden.dims3()?;

        let q = self.query.forward(hidden)?;
        let k = self.key.forward(hidden)?;
        let v = self.value.forward(hidden)?;

        // Reshape to [batch, heads, seq_len, head_dim]
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

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn_weights = q.matmul(&k_t)?.affine(1.0 / scale, 0.0)?;
        let attn_weights = attn_weights.broadcast_add(attn_bias)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let attn_out = attn_weights.matmul(&v)?;
        let attn_out =
            attn_out
                .transpose(1, 2)?
                .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.dense.forward(&attn_out)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_albert_config_defaults() {
        use super::super::config::PlbertConfig;
        let config: PlbertConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.embedding_size, 128);
    }
}
