//! Native OmniVoice backend.

use crate::config::{
    preferred_runtime_choice as preferred_model_runtime_choice,
    preferred_runtime_choices as preferred_model_runtime_choices, DType, RuntimeChoice,
};
use crate::device::DeviceSelection;
use crate::models::ModelType;

mod audio_tokenizer;
mod config;
pub mod model;

pub use model::OmniVoiceModel;

/// Recommended runtime choice for native OmniVoice on the current binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OmniVoiceRuntimeChoice {
    pub device: DeviceSelection,
    pub dtype: DType,
}

impl From<RuntimeChoice> for OmniVoiceRuntimeChoice {
    fn from(choice: RuntimeChoice) -> Self {
        Self {
            device: choice.device,
            dtype: choice.dtype,
        }
    }
}

impl OmniVoiceRuntimeChoice {
    /// Combined backend label used by examples and benchmarks.
    pub fn label(&self) -> String {
        format!("{} ({})", self.device.label(), self.dtype.label())
    }

    /// Human-readable floating-point dtype label.
    pub fn dtype_label(&self) -> &'static str {
        self.dtype.label()
    }
}

/// Preferred OmniVoice runtime choices ordered by expected performance.
pub fn preferred_runtime_choices() -> Vec<OmniVoiceRuntimeChoice> {
    preferred_model_runtime_choices(ModelType::OmniVoice)
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Best OmniVoice runtime choice for the current binary with CPU fallback.
pub fn preferred_runtime_choice() -> OmniVoiceRuntimeChoice {
    preferred_model_runtime_choice(ModelType::OmniVoice).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_runtime_prefers_f32() {
        let choice = OmniVoiceRuntimeChoice {
            device: DeviceSelection::Cpu,
            dtype: DType::F32,
        };
        assert_eq!(choice.dtype, DType::F32);
        assert_eq!(choice.label(), "cpu (f32)");
    }

    #[test]
    fn test_metal_runtime_prefers_f32() {
        let choice = OmniVoiceRuntimeChoice {
            device: DeviceSelection::Metal(0),
            dtype: DType::F32,
        };
        assert_eq!(choice.dtype, DType::F32);
        assert_eq!(choice.label(), "metal:0 (f32)");
    }
}
