//! Kokoro text encoder.
//!
//! Conv1d stack + bidirectional LSTM encoder. Operates on phoneme embeddings
//! and produces hidden representations for duration alignment.
//!
//! Architecture from StyleTTS2:
//! - Embedding(n_token, channels)
//! - Conv1d stack (depth layers, kernel_size, channels)
//! - Bidirectional LSTM
//! - Linear projection

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::layers::conv::{ChannelNorm, Conv1d};
use crate::layers::lstm::Lstm;

fn leaky_relu(x: &Tensor, negative_slope: f64) -> Result<Tensor> {
    let scaled = (x * negative_slope)?;
    x.maximum(&scaled)
}

/// Text encoder: phoneme embeddings → Conv1d stack → BiLSTM → projection.
pub struct TextEncoder {
    embedding: candle_nn::Embedding,
    cnn: Vec<CnnBlock>,
    lstm: Lstm,
    channels: usize,
}

struct CnnBlock {
    conv: Conv1d,
    norm: ChannelNorm,
}

impl TextEncoder {
    /// Load from VarBuilder.
    ///
    /// `channels`: hidden dimension (512)
    /// `kernel_size`: convolution kernel size (5)
    /// `depth`: number of Conv1d layers (3)
    /// `n_symbols`: phoneme vocabulary size (178)
    pub fn load(
        channels: usize,
        kernel_size: usize,
        depth: usize,
        n_symbols: usize,
        vb: VarBuilder,
        _device: &Device,
    ) -> Result<Self> {
        let embedding = candle_nn::embedding(n_symbols, channels, vb.pp("embedding"))?;

        let mut cnn = Vec::with_capacity(depth);
        for i in 0..depth {
            let padding = kernel_size / 2;
            let block_vb = vb.pp("cnn").pp(i.to_string());
            // PyTorch Sequential: [0] = weight_norm(Conv1d), [1] = ChannelLayerNorm
            let conv = Conv1d::load(
                channels,
                channels,
                kernel_size,
                1, // stride
                padding,
                1,    // dilation
                1,    // groups
                true, // bias
                block_vb.pp("0"),
            )?;
            let norm = ChannelNorm::load(channels, block_vb.pp("1"))?;
            cnn.push(CnnBlock { conv, norm });
        }

        // Bidirectional LSTM: input=channels, hidden=channels/2, output=channels
        let lstm = Lstm::load(
            1,            // num_layers
            channels,     // input_size
            channels / 2, // hidden_size (bidirectional → output = channels)
            true,         // bidirectional
            vb.pp("lstm"),
        )?;

        Ok(Self {
            embedding,
            cnn,
            lstm,
            channels,
        })
    }

    /// Forward pass.
    ///
    /// `input_ids`: [batch, seq_len] — phoneme token IDs
    /// `input_lengths`: [batch] — actual lengths (unused in inference with single seq)
    /// `text_mask`: [batch, seq_len] — True for padded positions
    ///
    /// Returns: [batch, channels, seq_len] (transposed for downstream use)
    pub fn forward(
        &self,
        input_ids: &Tensor,
        _input_lengths: &Tensor,
        text_mask: &Tensor,
    ) -> Result<Tensor> {
        // Embed: [batch, seq_len] → [batch, seq_len, channels]
        let mut x = self.embedding.forward(input_ids)?;

        // Transpose to [batch, channels, seq_len] for Conv1d
        x = x.transpose(1, 2)?;

        // Apply mask: expand mask to [batch, 1, seq_len] and zero out padding
        let mask = text_mask.unsqueeze(1)?.to_dtype(x.dtype())?;

        // Zero out padding positions
        let inv_mask = mask.neg()?.add(&Tensor::ones_like(&mask)?)?;
        x = x.broadcast_mul(&inv_mask)?;

        // CNN blocks
        for block in &self.cnn {
            x = block.conv.forward(&x)?;
            // Apply channel norm: transpose [batch, channels, seq_len] → [batch, seq_len, channels]
            let x_t = x.transpose(1, 2)?;
            let x_normed = block.norm.forward(&x_t)?;
            x = x_normed.transpose(1, 2)?;
            x = leaky_relu(&x, 0.2)?;
            // Re-apply mask
            x = x.broadcast_mul(&inv_mask)?;
        }

        // LSTM expects [batch, seq_len, channels]
        let x_t = x.transpose(1, 2)?;
        let lstm_out = self.lstm.forward(&x_t)?;

        // Back to [batch, channels, seq_len]
        let result = lstm_out.transpose(1, 2)?;

        // Re-apply mask
        result.broadcast_mul(&inv_mask)
    }

    /// Output channel dimension.
    pub fn channels(&self) -> usize {
        self.channels
    }
}
