//! Generate sample WAV files from Qwen3-TTS-1.7B.
//!
//! Run with:
//!   cargo run --example generate_qwen3_tts --release
//!
//! With MP3 output:
//!   cargo run --example generate_qwen3_tts --release --features mp3
//!
//! ⚠ Requires ~4.5 GB of model weights. They will be downloaded from
//!   HuggingFace on first run if the `download` feature is enabled.
//!
//! Output goes to `output/qwen3_tts/` in the project root.

use tts_rs::models::qwen3_tts::Qwen3TtsModel;
use tts_rs::traits::TtsModel;
use tts_rs::{ModelType, SynthesisRequest, TtsConfig};

fn main() {
    let out_dir = std::path::Path::new("output/qwen3_tts");

    println!("═══════════════════════════════════════════════════════");
    println!("  Qwen3-TTS-1.7B  —  Sample Audio Generation");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("Loading Qwen3-TTS-1.7B (this may take a while) …");

    let config = TtsConfig::new(ModelType::Qwen3Tts).with_model_path("./models/Qwen3-TTS");
    let model = Qwen3TtsModel::load(config).expect("Failed to load model");

    let info = model.model_info();
    println!("  Model       : {}", info.name);
    println!("  Sample rate : {} Hz", model.sample_rate());
    println!("  Voices      : {:?}", model.supported_voices());
    println!("  Languages   : {:?}", model.supported_languages());
    println!();

    let samples: &[(&str, &str, &str)] = &[
        // English
        (
            "english",
            "english_hello",
            "Hello! This is a test of the Qwen3 text to speech model, running entirely in Rust.",
        ),
        // German
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
        // Chinese
        (
            "chinese",
            "chinese_nihao",
            "你好！这是Qwen3文本转语音模型的测试，完全用Rust实现。",
        ),
        // Japanese
        (
            "japanese",
            "japanese_konnichiwa",
            "こんにちは！これはQwen3テキスト読み上げモデルのテストです。",
        ),
        // Korean
        (
            "korean",
            "korean_annyeong",
            "안녕하세요! 이것은 Qwen3 텍스트 음성 변환 모델의 테스트입니다.",
        ),
    ];

    // Use the first available voice, if any
    let voices = model.supported_voices();
    let voice = voices.first().map(|v| v.as_str());
    let voice = Some("dylan");

    for (lang, name, text) in samples {
        let stem = format!("qwen3tts_{name}");
        println!("▸ [{lang}] {stem}");
        println!("  \"{text}\"");

        let mut request = SynthesisRequest::new(*text).with_language(*lang);
        if let Some(v) = voice {
            request = request.with_voice(v);
        }

        match model.synthesize(&request) {
            Ok(audio) => {
                println!(
                    "  {:.2}s  ({} samples @ {} Hz)",
                    audio.duration_secs(),
                    audio.len(),
                    audio.sample_rate
                );

                let wav_path = out_dir.join(format!("{stem}.wav"));
                audio.save_wav(&wav_path).expect("Failed to write WAV");
                println!("  ✓ {}", wav_path.display());

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

    println!("Done! Check output/qwen3_tts/");
}
