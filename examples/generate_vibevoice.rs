//! Generate a sample WAV file with the native VibeVoice backend.
//!
//! Run with:
//!   cargo run --example generate_vibevoice --release --no-default-features --features vibevoice,download
//!
//! Add `metal` on Apple builds or `cuda` on NVIDIA builds to enable faster
//! backends.

use any_tts::{
    load_model, AudioSamples, DenoiseOptions, DeviceSelection, ModelType, SynthesisRequest,
    TtsConfig,
};
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct VibeVoiceExampleConfig {
    text: String,
    cfg_scale: f64,
    max_tokens: usize,
    temperature: f64,
    output: String,
}

#[derive(Debug, Clone, Copy)]
struct DenoiseVariant {
    label: &'static str,
    options: DenoiseOptions,
}

fn load_example_config() -> VibeVoiceExampleConfig {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/vibevoice_example.json");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {err}", path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("Failed to parse {}: {err}", path.display()))
}

fn denoise_variants() -> [DenoiseVariant; 2] {
    [
        DenoiseVariant {
            label: "denoised_default",
            options: DenoiseOptions::default(),
        },
        DenoiseVariant {
            label: "denoised_aggressive",
            options: DenoiseOptions {
                noise_reduction: 1.65,
                residual_floor: 0.05,
                wet_mix: 1.0,
                ..DenoiseOptions::default()
            },
        },
    ]
}

fn denoise_output_dir() -> PathBuf {
    env::var("VIBEVOICE_DENOISE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("output/denoise"))
}

fn output_stem(output_path: &Path) -> String {
    output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("vibevoice_demo")
        .to_string()
}

fn write_denoise_variants(audio: &AudioSamples, output_path: &Path) -> std::io::Result<Vec<PathBuf>> {
    let denoise_dir = denoise_output_dir();
    fs::create_dir_all(&denoise_dir)?;

    let stem = output_stem(output_path);
    let variants = denoise_variants();
    let mut saved_paths = Vec::with_capacity(1 + variants.len());

    let base_path = denoise_dir.join(format!("{stem}_base.wav"));
    audio.save_wav(&base_path)?;
    saved_paths.push(base_path);

    for variant in variants {
        let cleaned = audio.denoise_speech(variant.options);
        let variant_path = denoise_dir.join(format!("{stem}_{}.wav", variant.label));
        cleaned.save_wav(&variant_path)?;
        saved_paths.push(variant_path);
    }

    Ok(saved_paths)
}

fn main() {
    let example = load_example_config();
    let max_tokens = env::var("VIBEVOICE_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(example.max_tokens);
    let output_path = env::var("VIBEVOICE_OUTPUT").unwrap_or_else(|_| example.output.clone());
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

    let output_path = PathBuf::from(&output_path);
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("Failed to create {}: {err}", parent.display()));
    }
    audio
        .save_wav(Path::new(&output_path))
        .expect("Failed to write WAV");
    let denoise_paths = write_denoise_variants(&audio, &output_path)
        .expect("Failed to write denoised WAV variants");

    println!(
        "Saved VibeVoice sample to {} ({} samples @ {} Hz)",
        output_path.display(),
        audio.len(),
        audio.sample_rate
    );
    println!("Saved VibeVoice denoise variants:");
    for path in denoise_paths {
        println!("  {}", path.display());
    }
}
