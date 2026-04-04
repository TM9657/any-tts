//! Tests for device selection logic.

use any_tts::DeviceSelection;

#[test]
fn test_cpu_always_available() {
    let device = DeviceSelection::Cpu.resolve().unwrap();
    assert!(matches!(device, candle_core::Device::Cpu));
}

#[test]
fn test_auto_never_fails() {
    // Auto should always succeed — worst case returns CPU.
    let device = DeviceSelection::Auto.resolve().unwrap();
    match device {
        candle_core::Device::Cpu => {}
        candle_core::Device::Cuda(_) => {}
        candle_core::Device::Metal(_) => {}
    }
}

#[test]
fn test_default_is_auto() {
    assert_eq!(DeviceSelection::default(), DeviceSelection::Auto);
}

#[test]
fn test_cuda_without_feature_or_hardware() {
    // On CI/dev machines without CUDA, this should fail gracefully.
    // If CUDA feature is not enabled, it returns an error.
    // If CUDA feature is enabled but no GPU, it also returns an error.
    let result = DeviceSelection::Cuda(0).resolve();
    // We don't assert error here because the test might run on a CUDA machine.
    // Just verify it doesn't panic.
    let _ = result;
}

#[test]
fn test_metal_without_feature_or_hardware() {
    let result = DeviceSelection::Metal(0).resolve();
    // Same as CUDA — might succeed on macOS with Metal feature.
    let _ = result;
}
