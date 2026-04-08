//! Device selection utilities.
//!
//! Automatically selects the best available compute device based on compiled
//! feature flags and hardware availability.

use candle_core::Device;
use tracing::info;

/// Strategy for selecting a compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceSelection {
    /// Automatically select the best available device (CUDA → Metal → CPU).
    #[default]
    Auto,
    /// Force CPU execution.
    Cpu,
    /// Force CUDA execution on the specified GPU ordinal.
    Cuda(usize),
    /// Force Metal execution on the specified GPU ordinal.
    Metal(usize),
}

impl DeviceSelection {
    /// Resolve this selection into a concrete candle [`Device`].
    ///
    /// For [`DeviceSelection::Auto`], the priority is:
    /// 1. CUDA (if `cuda` feature enabled and device available)
    /// 2. Metal (if `metal` feature enabled and device available)
    /// 3. CPU (always available)
    pub fn resolve(&self) -> candle_core::Result<Device> {
        match self {
            Self::Cpu => {
                info!("Using CPU device");
                Ok(Device::Cpu)
            }
            Self::Cuda(ordinal) => {
                info!("Requesting CUDA device {}", ordinal);
                new_cuda(*ordinal)
            }
            Self::Metal(ordinal) => {
                info!("Requesting Metal device {}", ordinal);
                new_metal(*ordinal)
            }
            Self::Auto => auto_select(),
        }
    }

    /// Human-readable backend label.
    pub fn label(&self) -> String {
        match self {
            Self::Auto => "auto".to_string(),
            Self::Cpu => "cpu".to_string(),
            Self::Cuda(ordinal) => format!("cuda:{ordinal}"),
            Self::Metal(ordinal) => format!("metal:{ordinal}"),
        }
    }

    /// Preferred runtime candidates for the current binary, ordered fastest-first.
    pub fn preferred_runtime_candidates() -> Vec<Self> {
        vec![
            #[cfg(feature = "cuda")]
            Self::Cuda(0),
            #[cfg(feature = "metal")]
            Self::Metal(0),
            Self::Cpu,
        ]
    }

    /// Runtime candidates that successfully resolve on the current machine.
    pub fn available_runtime_candidates() -> Vec<Self> {
        let mut available = Vec::new();
        for candidate in Self::preferred_runtime_candidates() {
            if candidate.resolve().is_ok() {
                available.push(candidate);
            }
        }

        if available.is_empty() {
            available.push(Self::Cpu);
        }

        available
    }
}

/// Attempt to create a CUDA device.
fn new_cuda(ordinal: usize) -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    {
        let device = Device::new_cuda(ordinal)?;
        info!("CUDA device {} initialized", ordinal);
        Ok(device)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        Err(candle_core::Error::Msg(
            "CUDA feature not enabled at compile time".to_string(),
        ))
    }
}

/// Attempt to create a Metal device.
fn new_metal(ordinal: usize) -> candle_core::Result<Device> {
    #[cfg(feature = "metal")]
    {
        let device = Device::new_metal(ordinal)?;
        info!("Metal device {} initialized", ordinal);
        Ok(device)
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = ordinal;
        Err(candle_core::Error::Msg(
            "Metal feature not enabled at compile time".to_string(),
        ))
    }
}

/// Auto-select the best device based on compiled features and availability.
fn auto_select() -> candle_core::Result<Device> {
    // Try CUDA first
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("Auto-selected CUDA device 0");
                return Ok(device);
            }
            Err(e) => {
                info!("CUDA not available: {}, falling back", e);
            }
        }
    }

    // Try Metal
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("Auto-selected Metal device 0");
                return Ok(device);
            }
            Err(e) => {
                info!("Metal not available: {}, falling back", e);
            }
        }
    }

    // Fallback to CPU
    info!("Auto-selected CPU device");
    Ok(Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_always_works() {
        let device = DeviceSelection::Cpu.resolve().unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_auto_resolves_to_some_device() {
        let device = DeviceSelection::Auto.resolve().unwrap();
        // Should always succeed — worst case returns CPU
        match device {
            Device::Cpu => {}
            Device::Cuda(_) => {}
            Device::Metal(_) => {}
        }
    }

    #[test]
    fn test_default_is_auto() {
        assert_eq!(DeviceSelection::default(), DeviceSelection::Auto);
    }

    #[test]
    fn test_preferred_candidates_end_with_cpu() {
        let candidates = DeviceSelection::preferred_runtime_candidates();
        assert_eq!(candidates.last(), Some(&DeviceSelection::Cpu));
    }

    #[test]
    fn test_device_labels_are_stable() {
        assert_eq!(DeviceSelection::Cpu.label(), "cpu");
        assert_eq!(DeviceSelection::Cuda(0).label(), "cuda:0");
        assert_eq!(DeviceSelection::Metal(0).label(), "metal:0");
    }
}
