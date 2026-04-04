//! LSTM (Long Short-Term Memory) implementation for candle.
//!
//! Provides inference-only LSTM layers compatible with PyTorch weight
//! format. Used by Kokoro's text encoder and prosody predictor.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// A single LSTM layer (unidirectional or bidirectional).
///
/// Compatible with PyTorch's `nn.LSTM` weight naming:
/// - `weight_ih_l0` shape [4*hidden, input]
/// - `weight_hh_l0` shape [4*hidden, hidden]
/// - `bias_ih_l0` shape [4*hidden]
/// - `bias_hh_l0` shape [4*hidden]
/// - For reverse: `weight_ih_l0_reverse`, etc.
pub struct Lstm {
    layers: Vec<LstmLayer>,
    bidirectional: bool,
    hidden_size: usize,
}

struct LstmLayer {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    // Reverse direction weights (only for bidirectional)
    weight_ih_rev: Option<Tensor>,
    weight_hh_rev: Option<Tensor>,
    bias_ih_rev: Option<Tensor>,
    bias_hh_rev: Option<Tensor>,
    hidden_size: usize,
}

impl Lstm {
    /// Load an LSTM from a VarBuilder.
    ///
    /// `num_layers`: number of stacked LSTM layers
    /// `input_size`: input feature dimension
    /// `hidden_size`: hidden state dimension
    /// `bidirectional`: whether to use bidirectional LSTM
    pub fn load(
        num_layers: usize,
        input_size: usize,
        hidden_size: usize,
        bidirectional: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let dir_factor = if bidirectional { 2 } else { 1 };

        for layer_idx in 0..num_layers {
            let in_size = if layer_idx == 0 {
                input_size
            } else {
                hidden_size * dir_factor
            };

            let weight_ih = vb.get(
                (4 * hidden_size, in_size),
                &format!("weight_ih_l{}", layer_idx),
            )?;
            let weight_hh = vb.get(
                (4 * hidden_size, hidden_size),
                &format!("weight_hh_l{}", layer_idx),
            )?;
            let bias_ih = vb.get(4 * hidden_size, &format!("bias_ih_l{}", layer_idx))?;
            let bias_hh = vb.get(4 * hidden_size, &format!("bias_hh_l{}", layer_idx))?;

            let (weight_ih_rev, weight_hh_rev, bias_ih_rev, bias_hh_rev) = if bidirectional {
                let w_ih = vb.get(
                    (4 * hidden_size, in_size),
                    &format!("weight_ih_l{}_reverse", layer_idx),
                )?;
                let w_hh = vb.get(
                    (4 * hidden_size, hidden_size),
                    &format!("weight_hh_l{}_reverse", layer_idx),
                )?;
                let b_ih = vb.get(4 * hidden_size, &format!("bias_ih_l{}_reverse", layer_idx))?;
                let b_hh = vb.get(4 * hidden_size, &format!("bias_hh_l{}_reverse", layer_idx))?;
                (Some(w_ih), Some(w_hh), Some(b_ih), Some(b_hh))
            } else {
                (None, None, None, None)
            };

            layers.push(LstmLayer {
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                weight_ih_rev,
                weight_hh_rev,
                bias_ih_rev,
                bias_hh_rev,
                hidden_size,
            });
        }

        Ok(Self {
            layers,
            bidirectional,
            hidden_size,
        })
    }

    /// Forward pass through the LSTM.
    ///
    /// `input`: tensor of shape `[batch, seq_len, input_size]`
    ///
    /// Returns output of shape `[batch, seq_len, hidden_size * num_directions]`
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut current = input.clone();

        for layer in &self.layers {
            current = layer.forward(&current, self.bidirectional)?;
        }

        Ok(current)
    }

    /// Output size (hidden_size * num_directions).
    pub fn output_size(&self) -> usize {
        self.hidden_size * if self.bidirectional { 2 } else { 1 }
    }
}

impl LstmLayer {
    fn forward(&self, input: &Tensor, bidirectional: bool) -> Result<Tensor> {
        let fwd = self.forward_direction(
            input,
            &self.weight_ih,
            &self.weight_hh,
            &self.bias_ih,
            &self.bias_hh,
            false,
        )?;

        if bidirectional {
            let rev = self.forward_direction(
                input,
                self.weight_ih_rev.as_ref().unwrap(),
                self.weight_hh_rev.as_ref().unwrap(),
                self.bias_ih_rev.as_ref().unwrap(),
                self.bias_hh_rev.as_ref().unwrap(),
                true,
            )?;
            Tensor::cat(&[&fwd, &rev], 2)
        } else {
            Ok(fwd)
        }
    }

    fn forward_direction(
        &self,
        input: &Tensor,
        weight_ih: &Tensor,
        weight_hh: &Tensor,
        bias_ih: &Tensor,
        bias_hh: &Tensor,
        reverse: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = input.dims3()?;
        let device = input.device();
        let dtype = input.dtype();

        let mut h = Tensor::zeros((batch, self.hidden_size), dtype, device)?;
        let mut c = Tensor::zeros((batch, self.hidden_size), dtype, device)?;
        let mut outputs = Vec::with_capacity(seq_len);

        let range: Vec<usize> = if reverse {
            (0..seq_len).rev().collect()
        } else {
            (0..seq_len).collect()
        };

        for &t in &range {
            let x_t = input.narrow(1, t, 1)?.squeeze(1)?.contiguous()?;

            // gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
            let w_ih_t = weight_ih.t()?.contiguous()?;
            let w_hh_t = weight_hh.t()?.contiguous()?;
            let gates = x_t
                .matmul(&w_ih_t)?
                .broadcast_add(bias_ih)?
                .add(&h.matmul(&w_hh_t)?.broadcast_add(bias_hh)?)?;

            // Split into 4 gates: input, forget, cell, output
            let gate_size = self.hidden_size;
            let i_gate = candle_nn::ops::sigmoid(&gates.narrow(1, 0, gate_size)?)?;
            let f_gate = candle_nn::ops::sigmoid(&gates.narrow(1, gate_size, gate_size)?)?;
            let g_gate = gates.narrow(1, 2 * gate_size, gate_size)?.tanh()?;
            let o_gate = candle_nn::ops::sigmoid(&gates.narrow(1, 3 * gate_size, gate_size)?)?;

            c = f_gate.mul(&c)?.add(&i_gate.mul(&g_gate)?)?;
            h = o_gate.mul(&c.tanh()?)?;

            outputs.push(h.unsqueeze(1)?);
        }

        // If reversed, reverse the outputs back to original order
        if reverse {
            outputs.reverse();
        }

        Tensor::cat(&outputs, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_lstm_output_shape() {
        let device = Device::Cpu;
        let dtype = candle_core::DType::F32;

        let hidden = 32;
        let input_size = 16;

        // Create mock weights
        let weight_ih = Tensor::randn(0f32, 0.1, (4 * hidden, input_size), &device).unwrap();
        let weight_hh = Tensor::randn(0f32, 0.1, (4 * hidden, hidden), &device).unwrap();
        let bias_ih = Tensor::zeros(4 * hidden, dtype, &device).unwrap();
        let bias_hh = Tensor::zeros(4 * hidden, dtype, &device).unwrap();

        let layer = LstmLayer {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            weight_ih_rev: None,
            weight_hh_rev: None,
            bias_ih_rev: None,
            bias_hh_rev: None,
            hidden_size: hidden,
        };

        let lstm = Lstm {
            layers: vec![layer],
            bidirectional: false,
            hidden_size: hidden,
        };

        let input = Tensor::randn(0f32, 1.0, (2, 10, input_size), &device).unwrap();
        let output = lstm.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 10, hidden]);
    }
}
