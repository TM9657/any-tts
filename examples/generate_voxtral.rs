//! Generate a sample WAV file with the native Rust Voxtral backend.
//!
//! Prerequisites:
//!   Voxtral model assets available locally or downloadable via Hugging Face
//!
//! Run with:
//!   cargo run --example generate_voxtral --release

use std::env;

use tts_rs::{load_model, ModelType, SynthesisRequest, TtsConfig};

fn main() {
    let config = TtsConfig::new(ModelType::Voxtral);
    let model = load_model(config).expect("Failed to initialize native Voxtral backend");

    let mut args = env::args().skip(1);
    let text = args
        .next()
        .unwrap_or_else(|| "Hello from Voxtral, running natively in Rust.".to_string());
    let voice = args.next().unwrap_or_else(|| "neutral_male".to_string());
    let max_tokens = args
        .next()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128);

    let request = SynthesisRequest::new(text)
        .with_voice(voice)
        .with_max_tokens(max_tokens);

    let audio = model
        .synthesize(&request)
        .expect("Voxtral synthesis failed");
    audio
        .save_wav("output/voxtral/voxtral_demo.wav")
        .expect("Failed to write WAV");

    println!(
        "Saved Voxtral sample to output/voxtral/voxtral_demo.wav ({} samples @ {} Hz)",
        audio.len(),
        audio.sample_rate
    );
}
