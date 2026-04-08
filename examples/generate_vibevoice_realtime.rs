//! Generate a WAV file with the native VibeVoice Realtime-0.5B backend.
//!
//! Default Metal example on macOS:
//!   cargo run --example generate_vibevoice_realtime --release --no-default-features --features metal,vibevoice,download
//!
//! CPU example:
//!   VIBEVOICE_REALTIME_DEVICE=cpu \
//!   cargo run --example generate_vibevoice_realtime --release --no-default-features --features vibevoice,download
//!
//! This example reuses `examples/vibevoice_example.json` for text and generation settings.
//! Optional overrides are still available through `VIBEVOICE_REALTIME_*` env vars.

use any_tts::{load_model, DeviceSelection, ModelType, SynthesisRequest, TtsConfig};
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

#[derive(Debug)]
struct CliArgs {
    device: DeviceSelection,
    voice: Option<String>,
    output_path: PathBuf,
    model_path: Option<String>,
    voices_dir: PathBuf,
}

fn load_example_config() -> VibeVoiceExampleConfig {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/vibevoice_example.json");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {err}", path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("Failed to parse {}: {err}", path.display()))
}

fn load_options(example: &VibeVoiceExampleConfig) -> CliArgs {
    let device = env::var("VIBEVOICE_REALTIME_DEVICE")
        .ok()
        .map(|value| parse_device(&value))
        .unwrap_or(DeviceSelection::Auto);
    let output_path = env::var("VIBEVOICE_REALTIME_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_realtime_output_path(&example.output));
    let voices_dir = env::var("VIBEVOICE_REALTIME_VOICES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_voices_dir());

    CliArgs {
        device,
        voice: env::var("VIBEVOICE_REALTIME_VOICE").ok(),
        output_path,
        model_path: env::var("VIBEVOICE_REALTIME_MODEL_PATH").ok(),
        voices_dir,
    }
}

fn default_voices_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/VibeVoice-Realtime-0.5B/voices")
}

fn default_realtime_output_path(output: &str) -> PathBuf {
    let output_path = PathBuf::from(output);
    let parent = output_path
        .parent()
        .unwrap_or_else(|| Path::new("output/vibevoice"));
    let stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("vibevoice_demo");
    let extension = output_path
        .extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .unwrap_or("wav");
    parent.join(format!("{stem}_realtime.{extension}"))
}

fn parse_device(value: &str) -> DeviceSelection {
    if value.eq_ignore_ascii_case("auto") {
        return DeviceSelection::Auto;
    }
    if value.eq_ignore_ascii_case("cpu") {
        return DeviceSelection::Cpu;
    }
    if let Some(ordinal) = value.strip_prefix("metal:") {
        return DeviceSelection::Metal(
            ordinal
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("Invalid metal device ordinal: {value}")),
        );
    }
    if value.eq_ignore_ascii_case("metal") {
        return DeviceSelection::Metal(0);
    }
    if let Some(ordinal) = value.strip_prefix("cuda:") {
        return DeviceSelection::Cuda(
            ordinal
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("Invalid cuda device ordinal: {value}")),
        );
    }
    if value.eq_ignore_ascii_case("cuda") {
        return DeviceSelection::Cuda(0);
    }

    panic!("Unsupported device '{value}'. Expected auto, cpu, metal[:N], or cuda[:N]");
}

fn device_label(device: DeviceSelection) -> &'static str {
    match device {
        DeviceSelection::Auto => "auto",
        DeviceSelection::Cpu => "cpu",
        DeviceSelection::Metal(_) => "metal",
        DeviceSelection::Cuda(_) => "cuda",
    }
}

fn main() {
    let example = load_example_config();
    let args = load_options(&example);
    let mut config = TtsConfig::new(ModelType::VibeVoiceRealtime).with_device(args.device);
    if let Some(model_path) = args.model_path.as_deref() {
        config = config.with_model_path(model_path);
    }
    config = config.with_voices_dir(args.voices_dir.to_string_lossy().into_owned());

    let model = load_model(config).expect("Failed to initialize VibeVoice Realtime");
    let info = model.model_info();
    println!("Model       : {}", info.name);
    println!("Variant     : {}", info.variant);
    println!("Sample rate : {} Hz", info.sample_rate);
    println!("Device      : {}", device_label(args.device));
    println!("Voices dir  : {}", args.voices_dir.display());
    println!("Output      : {}", args.output_path.display());
    println!();

    let mut request = SynthesisRequest::new(&example.text)
        .with_cfg_scale(example.cfg_scale)
        .with_max_tokens(example.max_tokens)
        .with_temperature(example.temperature);
    if let Some(voice) = args.voice {
        request = request.with_voice(voice);
    }

    let audio = model
        .synthesize(&request)
        .expect("VibeVoice Realtime synthesis failed");

    let output_path = args.output_path;
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("Failed to create {}: {err}", parent.display()));
    }
    audio
        .save_wav(&output_path)
        .expect("Failed to write VibeVoice Realtime WAV");

    println!(
        "Saved VibeVoice Realtime sample to {} ({} samples @ {} Hz)",
        output_path.display(),
        audio.len(),
        audio.sample_rate
    );
}
