//! SiLU-gated MLP (feed-forward network).
//!
//! The standard Qwen / Llama MLP: gate_proj and up_proj are applied in
//! parallel, gated with SiLU, then fused through down_proj.
//!
//! ```text
//! output = down_proj( silu(gate_proj(x)) * up_proj(x) )
//! ```

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::tensor_utils::silu;

/// SiLU-gated feed-forward network.
pub struct SiluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SiluMlp {
    /// Load MLP weights from a VarBuilder.
    ///
    /// Expected weight names:
    /// - `gate_proj.weight` — (intermediate_size, hidden_size)
    /// - `up_proj.weight`   — (intermediate_size, hidden_size)
    /// - `down_proj.weight` — (hidden_size, intermediate_size)
    pub fn load(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj =
            candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj =
            candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass: `down_proj( silu(gate_proj(x)) * up_proj(x) )`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

impl std::fmt::Debug for SiluMlp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SiluMlp").finish()
    }
}
