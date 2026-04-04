//! Generate a sample WAV file with the native Rust Voxtral backend.
//!
//! Prerequisites:
//!   Voxtral model assets available locally or downloadable via Hugging Face
//!
//! Run with:
//!   cargo run --example generate_voxtral --release
//!   cargo run --example generate_voxtral --release -- --text "Hello" --output output/voxtral/hello.wav --device cpu

use std::env;

use any_tts::{load_model, DeviceSelection, ModelType, SynthesisRequest, TtsConfig};

struct CliArgs {
    text: String,
    voice: String,
    language: Option<String>,
    max_tokens: usize,
    output: String,
    device: DeviceSelection,
}

fn main() {
    let args = parse_args(env::args().skip(1).collect());

    let config = TtsConfig::new(ModelType::Voxtral).with_device(args.device);
    let model = load_model(config).expect("Failed to initialize native Voxtral backend");

    let mut request = SynthesisRequest::new(args.text)
        .with_voice(args.voice)
        .with_max_tokens(args.max_tokens);
    if let Some(language) = args.language {
        request = request.with_language(language);
    }

    let audio = model
        .synthesize(&request)
        .expect("Voxtral synthesis failed");
    audio
        .save_wav(&args.output)
        .expect("Failed to write WAV");

    println!(
        "Saved Voxtral sample to {} ({} samples @ {} Hz)",
        args.output,
        audio.len(),
        audio.sample_rate
    );
}

fn parse_args(raw_args: Vec<String>) -> CliArgs {
    let mut text = None;
    let mut voice = Some("neutral_male".to_string());
    let mut language = None;
    let mut max_tokens = Some(128usize);
    let mut output = Some("output/voxtral/voxtral_demo.wav".to_string());
    let mut device = Some(DeviceSelection::Auto);
    let mut positional = Vec::new();

    let mut index = 0usize;
    while index < raw_args.len() {
        let arg = &raw_args[index];
        if !arg.starts_with("--") {
            positional.push(arg.clone());
            index += 1;
            continue;
        }

        let value = raw_args.get(index + 1).cloned().unwrap_or_else(|| {
            panic!("Missing value for argument {arg}");
        });

        match arg.as_str() {
            "--text" => text = Some(value),
            "--voice" => voice = Some(value),
            "--language" => language = Some(value),
            "--max-tokens" => {
                max_tokens = Some(
                    value
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("Invalid --max-tokens value: {value}")),
                )
            }
            "--output" => output = Some(value),
            "--device" => device = Some(parse_device(&value)),
            "--help" | "-h" => print_help_and_exit(),
            _ => panic!("Unknown argument: {arg}"),
        }

        index += 2;
    }

    if text.is_none() {
        text = positional.first().cloned();
    }
    if positional.len() > 1 {
        voice = Some(positional[1].clone());
    }
    if positional.len() > 2 {
        max_tokens = Some(
            positional[2]
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("Invalid positional max_tokens: {}", positional[2])),
        );
    }

    CliArgs {
        text: text.unwrap_or_else(|| "Hello from Voxtral, running natively in Rust.".to_string()),
        voice: voice.unwrap_or_else(|| "neutral_male".to_string()),
        language,
        max_tokens: max_tokens.unwrap_or(128),
        output: output.unwrap_or_else(|| "output/voxtral/voxtral_demo.wav".to_string()),
        device: device.unwrap_or(DeviceSelection::Auto),
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

fn print_help_and_exit() -> ! {
    println!(
        "generate_voxtral [text voice max_tokens] [--text TEXT] [--voice VOICE] [--language LANG] [--max-tokens N] [--output PATH] [--device auto|cpu|metal[:N]|cuda[:N]]"
    );
    std::process::exit(0);
}
