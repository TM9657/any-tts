//! Generate sample WAV files from Qwen3-TTS-1.7B.
//!
//! Run with:
//!   cargo run --example generate_qwen3_tts --release
//!
//! Force CPU output into a dedicated folder:
//!   QWEN3_TTS_DEVICE=cpu QWEN3_TTS_OUTPUT_DIR=output/qwen3_tts/cpu cargo run --example generate_qwen3_tts --release
//!
//! Force Metal output into a dedicated folder:
//!   QWEN3_TTS_DEVICE=metal QWEN3_TTS_OUTPUT_DIR=output/qwen3_tts/metal cargo run --example generate_qwen3_tts --release --features metal
//!
//! ⚠ Requires ~4.5 GB of model weights. They will be downloaded from
//!   HuggingFace on first run if the `download` feature is enabled.
//!
//! Output goes to `output/qwen3_tts/` in the project root.

use any_tts::DeviceSelection;
use any_tts::models::qwen3_tts::Qwen3TtsModel;
use any_tts::traits::TtsModel;
use any_tts::{ModelType, SynthesisRequest, TtsConfig};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct CliArgs {
    device: DeviceSelection,
    output_dir: PathBuf,
    model_path: Option<String>,
}

type SampleSpec = (&'static str, &'static str, &'static str);

const SAMPLE_SPECS: &[SampleSpec] = &[
    (
        "english",
        "english_hello",
        "Hello! This is a test of the Qwen3 text to speech model, running entirely in Rust.",
    ),
    (
        "german",
        "german_hallo",
        "Hallo! Dies ist ein Test der Qwen3 Sprachsynthese, vollständig in Rust implementiert.",
    ),
    (
        "german",
        "german_long",
        "Die Entwicklung von Sprachsynthese-Systemen hat in den letzten Jahren enorme \
         Fortschritte gemacht. Neuronale Netzwerke ermöglichen eine natürlich klingende \
         Ausgabe, die kaum von menschlicher Sprache zu unterscheiden ist.",
    ),
    (
        "chinese",
        "chinese_nihao",
        "你好！这是Qwen3文本转语音模型的测试，完全用Rust实现。",
    ),
    (
        "japanese",
        "japanese_konnichiwa",
        "こんにちは！これはQwen3テキスト読み上げモデルのテストです。",
    ),
    (
        "korean",
        "korean_annyeong",
        "안녕하세요! 이것은 Qwen3 텍스트 음성 변환 모델의 테스트입니다.",
    ),
];

fn load_options() -> CliArgs {
    let device = env::var("QWEN3_TTS_DEVICE")
        .ok()
        .map(|value| parse_device(&value))
        .unwrap_or(DeviceSelection::Auto);
    let output_dir = env::var("QWEN3_TTS_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("output/qwen3_tts"));
    let model_path = env::var("QWEN3_TTS_MODEL_PATH").ok();

    CliArgs {
        device,
        output_dir,
        model_path,
    }
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

fn load_model(args: &CliArgs) -> Qwen3TtsModel {
    let mut config = TtsConfig::new(ModelType::Qwen3Tts).with_device(args.device);
    if let Some(model_path) = args.model_path.as_deref() {
        config = config.with_model_path(model_path);
    }

    Qwen3TtsModel::load(config).expect("Failed to load model")
}

fn print_model_summary(model: &Qwen3TtsModel, output_dir: &Path) {
    let info = model.model_info();
    println!("  Model       : {}", info.name);
    println!("  Device      : {:?}", model.device());
    println!("  Sample rate : {} Hz", model.sample_rate());
    println!("  Voices      : {:?}", model.supported_voices());
    println!("  Languages   : {:?}", model.supported_languages());
    println!("  Output dir  : {}", output_dir.display());
    println!();
}

fn render_samples(model: &Qwen3TtsModel, output_dir: &Path) {
    let voices = model.supported_voices();
    let voice = voices
        .iter()
        .find(|candidate| candidate.as_str() == "dylan")
        .or_else(|| voices.first())
        .map(String::as_str);

    for (lang, name, text) in SAMPLE_SPECS {
        let stem = format!("qwen3tts_{name}");
        println!("▸ [{lang}] {stem}");
        println!("  \"{text}\"");

        let mut request = SynthesisRequest::new(*text).with_language(*lang);
        if let Some(selected_voice) = voice {
            request = request.with_voice(selected_voice);
        }

        match model.synthesize(&request) {
            Ok(audio) => {
                println!(
                    "  {:.2}s  ({} samples @ {} Hz)",
                    audio.duration_secs(),
                    audio.len(),
                    audio.sample_rate
                );

                let wav_path = output_dir.join(format!("{stem}.wav"));
                audio.save_wav(&wav_path).expect("Failed to write WAV");
                println!("  ✓ {}", wav_path.display());
            }
            Err(e) => eprintln!("  ✗ {e}"),
        }
        println!();
    }
}

fn main() {
    let args = load_options();
    fs::create_dir_all(&args.output_dir)
        .unwrap_or_else(|err| panic!("Failed to create {}: {err}", args.output_dir.display()));

    println!("═══════════════════════════════════════════════════════");
    println!("  Qwen3-TTS-1.7B  —  Sample Audio Generation");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("Loading Qwen3-TTS-1.7B (this may take a while) …");

    let model = load_model(&args);
    print_model_summary(&model, &args.output_dir);
    render_samples(&model, &args.output_dir);

    println!("Done! Check {}", args.output_dir.display());
}
