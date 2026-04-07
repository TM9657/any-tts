//! Generate sample WAV files from Kokoro-82M with the in-tree pure-Rust phonemizer.
//!
//! Run with:
//!   cargo run --example generate_kokoro --release
//!
//! Output goes to `output/kokoro/` in the project root.

use any_tts::models::kokoro::KokoroModel;
use any_tts::traits::TtsModel;
use any_tts::{ModelType, SynthesisRequest, TtsConfig};

const SAMPLE_REQUESTS: [(&str, &str, &str); 19] = [
    (
        "en",
        "english_hello",
        "Hello! This is a test of Kokoro text to speech, running entirely in Rust.",
    ),
    (
        "en",
        "english_numbers",
        "I have 3 apples, 2 oranges, and 1 surprisingly patient compiler.",
    ),
    (
        "en",
        "english_acronyms",
        "The AI API serves GPU TTS on the CPU.",
    ),
    (
        "en-gb",
        "british_schedule",
        "The schedule is full, but the aluminium prototype still shipped on Thursday.",
    ),
    (
        "en-gb",
        "british_initials",
        "A CLI and an HTTP API help the GPU team.",
    ),
    (
        "en",
        "english_phrase_stress",
        "Speech and text are entirely different in Kokoro.",
    ),
    (
        "en-gb",
        "british_article_vowel",
        "The aluminium API opens on Thursday for Kokoro.",
    ),
    (
        "de",
        "german_hallo",
        "Hallo! Dies ist ein Test der Kokoro Sprachsynthese, vollständig in Rust implementiert.",
    ),
    (
        "de",
        "german_compound",
        "Die Sprachsynthese bleibt überraschend gut.",
    ),
    (
        "de",
        "german_kokoro_rust",
        "Kokoro und Rust bleiben überraschend gut.",
    ),
    (
        "fr",
        "french_bonjour",
        "Bonjour! Ceci est un test de la synthèse vocale Kokoro, entièrement en Rust.",
    ),
    (
        "es",
        "spanish_hola",
        "¡Hola! Esta es una prueba de síntesis de voz Kokoro, implementada en Rust.",
    ),
    (
        "it",
        "italian_ciao",
        "Ciao! Questo è un test della sintesi vocale Kokoro, interamente in Rust.",
    ),
    (
        "pt",
        "portuguese_ola",
        "Olá! Este é um teste da síntese de fala Kokoro, implementada inteiramente em Rust.",
    ),
    (
        "ja",
        "japanese_konnichiwa",
        "こんにちは！これはKokoroテキスト読み上げのテストです。",
    ),
    (
        "ja",
        "japanese_mixed_rust",
        "RustでKokoroの音声合成を試します。",
    ),
    ("zh", "chinese_nihao", "你好！这是Kokoro文本转语音的测试。"),
    (
        "ko",
        "korean_annyeong",
        "안녕하세요! 이것은 Rust로 작성한 Kokoro 음성 합성 테스트입니다.",
    ),
    (
        "hi",
        "hindi_namaste",
        "नमस्ते! यह Rust में लिखे गए Kokoro पाठ-से-भाषण का परीक्षण है।",
    ),
];

fn print_header() {
    println!("═══════════════════════════════════════════════════════");
    println!("  Kokoro-82M  —  Sample Audio Generation");
    println!("═══════════════════════════════════════════════════════");
    println!();
    println!("Loading Kokoro-82M …");
}

fn print_model_info(model: &KokoroModel) {
    println!("  Sample rate : {} Hz", model.sample_rate());
    println!("  Voices      : {:?}", model.supported_voices());
    println!("  Languages   : {:?}", model.supported_languages());
    println!("  Note        : pure-Rust text phonemization now runs in-tree for Kokoro's current language set");
    println!();
}

fn synthesize_sample(
    model: &KokoroModel,
    out_dir: &std::path::Path,
    lang: &str,
    name: &str,
    text: &str,
) {
    let stem = format!("kokoro_{name}");
    println!("▸ [{lang}] {stem}");
    println!("  \"{text}\"");

    let request = SynthesisRequest::new(text).with_language(lang);

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
        }
        Err(e) => eprintln!("  ✗ {e}"),
    }
    println!();
}

fn main() {
    let out_dir = std::path::Path::new("output/kokoro");
    print_header();

    let config = TtsConfig::new(ModelType::Kokoro).with_model_path("./models/Kokoro-82M");
    let model = KokoroModel::load(config).expect("Failed to load model");

    print_model_info(&model);

    for (lang, name, text) in SAMPLE_REQUESTS {
        synthesize_sample(&model, out_dir, lang, name, text);
    }

    println!("Done! Check output/kokoro/");
}
