//! Generate a sample WAV file with the native VibeVoice backend.
//!
//! Run with:
//!   cargo run --example generate_vibevoice --release --no-default-features --features vibevoice,download
//!
//! Add `metal` on Apple builds or `cuda` on NVIDIA builds to enable faster
//! backends.

use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use any_tts::{load_model, DeviceSelection, ModelType, SynthesisRequest, TtsConfig};

#[derive(Debug, Deserialize)]
struct VibeVoiceExampleConfig {
    text: String,
    cfg_scale: f64,
    max_tokens: usize,
    temperature: f64,
    output: String,
}

fn load_example_config() -> VibeVoiceExampleConfig {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/vibevoice_example.json");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {err}", path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("Failed to parse {}: {err}", path.display()))
}

fn main() {
    let example = load_example_config();
    let max_tokens = env::var("VIBEVOICE_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(example.max_tokens);
    let device = match env::var("VIBEVOICE_DEVICE").ok().as_deref() {
        Some("cpu") => DeviceSelection::Cpu,
        Some("metal") => DeviceSelection::Metal(0),
        Some("cuda") => DeviceSelection::Cuda(0),
        _ => DeviceSelection::Auto,
    };
    let model = load_model(TtsConfig::new(ModelType::VibeVoice).with_device(device))
        .expect("Failed to initialize VibeVoice");

    let request = SynthesisRequest::new(&example.text)
        .with_cfg_scale(example.cfg_scale)
        .with_max_tokens(max_tokens)
        .with_temperature(example.temperature);

    let audio = model
        .synthesize(&request)
        .expect("VibeVoice synthesis failed");

    let output_path = PathBuf::from(&example.output);
    if let Some(parent) = output_path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("Failed to create {}: {err}", parent.display()));
    }
    audio
        .save_wav(Path::new(&example.output))
        .expect("Failed to write WAV");

    println!(
        "Saved VibeVoice sample to {} ({} samples @ {} Hz)",
        example.output,
        audio.len(),
        audio.sample_rate
    );
}