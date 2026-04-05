//! Native OmniVoice backend.

use crate::config::DType;
use crate::device::DeviceSelection;

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

impl OmniVoiceRuntimeChoice {
    /// Combined backend label used by examples and benchmarks.
    pub fn label(&self) -> String {
        format!("{} ({})", self.device.label(), dtype_label(self.dtype))
    }

    /// Human-readable floating-point dtype label.
    pub fn dtype_label(&self) -> &'static str {
        dtype_label(self.dtype)
    }
}

/// Preferred OmniVoice runtime choices ordered by expected performance.
pub fn preferred_runtime_choices() -> Vec<OmniVoiceRuntimeChoice> {
    DeviceSelection::available_runtime_candidates()
        .into_iter()
        .map(|device| OmniVoiceRuntimeChoice {
            dtype: preferred_dtype_for(device),
            device,
        })
        .collect()
}

/// Best OmniVoice runtime choice for the current binary with CPU fallback.
pub fn preferred_runtime_choice() -> OmniVoiceRuntimeChoice {
    preferred_runtime_choices()
        .into_iter()
        .next()
        .unwrap_or(OmniVoiceRuntimeChoice {
            device: DeviceSelection::Cpu,
            dtype: DType::F32,
        })
}

fn preferred_dtype_for(device: DeviceSelection) -> DType {
    match device {
        DeviceSelection::Cpu => DType::F32,
        DeviceSelection::Cuda(_) => DType::BF16,
        // Metal F16 diverges from the CPU path at the first OmniVoice logits step.
        DeviceSelection::Metal(_) => DType::F32,
        DeviceSelection::Auto => DType::BF16,
    }
}

fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_runtime_prefers_f32() {
        let choice = OmniVoiceRuntimeChoice {
            device: DeviceSelection::Cpu,
            dtype: preferred_dtype_for(DeviceSelection::Cpu),
        };
        assert_eq!(choice.dtype, DType::F32);
        assert_eq!(choice.label(), "cpu (f32)");
    }

    #[test]
    fn test_metal_runtime_prefers_f32() {
        let choice = OmniVoiceRuntimeChoice {
            device: DeviceSelection::Metal(0),
            dtype: preferred_dtype_for(DeviceSelection::Metal(0)),
        };
        assert_eq!(choice.dtype, DType::F32);
        assert_eq!(choice.label(), "metal:0 (f32)");
    }
}
