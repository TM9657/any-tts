//! Generate sample WAV files from Kokoro-82M in multiple languages.
//!
//! Run with:
//!   cargo run --example generate_kokoro --release
//!
//! With MP3 output:
//!   cargo run --example generate_kokoro --release --features mp3
//!
//! Output goes to `output/kokoro/` in the project root.

use any_tts::models::kokoro::KokoroModel;
use any_tts::traits::TtsModel;
use any_tts::{ModelType, SynthesisRequest, TtsConfig};

fn main() {
    let out_dir = std::path::Path::new("output/kokoro");
    let english_only = std::env::var_os("KOKORO_ONLY_ENGLISH").is_some();

    println!("═══════════════════════════════════════════════════════");
    println!("  Kokoro-82M  —  Sample Audio Generation");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("Loading Kokoro-82M …");

    let config = TtsConfig::new(ModelType::Kokoro).with_model_path("./models/Kokoro-82M");
    let model = KokoroModel::load(config).expect("Failed to load model");

    println!("  Sample rate : {} Hz", model.sample_rate());
    println!("  Voices      : {:?}", model.supported_voices());
    println!("  Languages   : {:?}", model.supported_languages());
    println!();

    let samples: &[(&str, &str, &str)] = &[
        // English
        (
            "en",
            "english_hello",
            "Hello! This is a test of Kokoro text to speech, running entirely in Rust.",
        ),
        // German
        (
            "de",
            "german_hallo",
            "Hallo! Dies ist ein Test der Kokoro Sprachsynthese, vollständig in Rust implementiert.",
        ),
        (
            "de",
            "german_long",
            "Die Entwicklung von Sprachsynthese-Systemen hat in den letzten Jahren enorme \
             Fortschritte gemacht. Neuronale Netzwerke ermöglichen eine natürlich klingende \
             Ausgabe, die kaum von menschlicher Sprache zu unterscheiden ist.",
        ),
        // French
        (
            "fr",
            "french_bonjour",
            "Bonjour! Ceci est un test de la synthèse vocale Kokoro, entièrement en Rust.",
        ),
        // Japanese
        (
            "ja",
            "japanese_konnichiwa",
            "こんにちは！これはKokoroテキスト読み上げのテストです。",
        ),
        // Spanish
        (
            "es",
            "spanish_hola",
            "¡Hola! Esta es una prueba de síntesis de voz Kokoro, implementada en Rust.",
        ),
        // Italian
        (
            "it",
            "italian_ciao",
            "Ciao! Questo è un test della sintesi vocale Kokoro, interamente in Rust.",
        ),
    ];

    for (lang, name, text) in samples {
        if english_only && *lang != "en" {
            continue;
        }

        let stem = format!("kokoro_{name}");
        println!("▸ [{lang}] {stem}");
        println!("  \"{text}\"");

        let request = SynthesisRequest::new(*text).with_language(*lang);

        match model.synthesize(&request) {
            Ok(audio) => {
                println!(
                    "  {:.2}s  ({} samples @ {} Hz)",
                    audio.duration_secs(),
                    audio.len(),
                    audio.sample_rate
                );

                // WAV
                let wav_path = out_dir.join(format!("{stem}.wav"));
                audio.save_wav(&wav_path).expect("Failed to write WAV");
                println!("  ✓ {}", wav_path.display());

                // MP3 (if compiled with --features mp3)
                #[cfg(feature = "mp3")]
                {
                    let mp3_path = out_dir.join(format!("{stem}.mp3"));
                    audio.save_mp3(&mp3_path).expect("Failed to write MP3");
                    println!("  ✓ {}", mp3_path.display());
                }
            }
            Err(e) => eprintln!("  ✗ {e}"),
        }
        println!();
    }

    println!("Done! Check output/kokoro/");
}
