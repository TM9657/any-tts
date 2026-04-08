//! # any-tts
//!
//! A Rust text-to-speech library powered primarily by the
//! [candle](https://github.com/huggingface/candle) ML framework.
//! Provides a unified trait-based API with pluggable model backends, including
//! native Candle implementations and adapters for official upstream runtimes.
//!
//! ## Supported Models
//!
//! - **Kokoro-82M** — 82M parameter StyleTTS2 model with ISTFTNet decoder for fast, high-quality speech
//! - **OmniVoice** — native Candle implementation of the OmniVoice zero-shot TTS model
//! - **Qwen3-TTS-12Hz-1.7B-CustomVoice** — 1.7B parameter multi-codebook LM for 10 languages
//! - **Qwen3-TTS-12Hz-1.7B-VoiceDesign** — 1.7B model with natural language voice descriptions
//! - **VibeVoice-1.5B** — native Candle implementation of Microsoft's multi-speaker speech diffusion model
//! - **VibeVoice-Realtime-0.5B** — native Candle implementation of Microsoft's cached-prompt realtime TTS model
//! - **Voxtral-4B-TTS-2603** — native Candle implementation of Mistral's 4B TTS model
//!
//! ## Feature Flags
//!
//! - `cuda` — Enable CUDA GPU acceleration
//! - `metal` — Enable Metal GPU acceleration (macOS/iOS)
//! - `accelerate` — Enable Apple Accelerate framework
//! - `kokoro` — Build Kokoro model support (default)
//! - `omnivoice` — Build native OmniVoice support (default)
//! - `qwen3-tts` — Build Qwen3-TTS model support (default)
//! - `vibevoice` — Build native VibeVoice support (default)
//! - `voxtral` — Build native Voxtral support (default)
//! - `download` — Enable automatic model downloading from HuggingFace Hub (default)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use any_tts::{TtsModel, TtsConfig, SynthesisRequest, ModelType};
//!
//! // Load a model
//! let config = TtsConfig::new(ModelType::Qwen3Tts)
//!     .with_model_path("/path/to/model");
//! let model = any_tts::load_model(config).unwrap();
//!
//! // Synthesize speech
//! let request = SynthesisRequest::new("Hello, world!")
//!     .with_language("en");
//! let audio = model.synthesize(&request).unwrap();
//!
//! // audio.samples contains f32 PCM data at model.sample_rate() Hz
//! let wav_bytes = audio.get_wav();
//! let _ = wav_bytes;
//! ```

pub mod audio;
pub mod config;
pub mod device;
pub mod error;
pub mod layers;
pub mod mel;
pub mod models;
pub mod tensor_utils;
pub mod tokenizer;
pub mod traits;

#[cfg(feature = "download")]
pub mod download;

// Re-export primary API types
pub use audio::{AudioSamples, DenoiseOptions};
pub use config::{
    preferred_runtime_choice, preferred_runtime_choices, DType, ModelAsset, ModelAssetBundle,
    ModelAssetDir, ModelFiles, RuntimeChoice, TtsConfig,
};
pub use device::DeviceSelection;
pub use error::TtsError;
pub use mel::{MelConfig, MelSpectrogram};
pub use models::{ModelAssetRequirement, ModelType};
pub use traits::{
    ModelInfo, ReferenceAudio, SynthesisRequest, TtsModel, VoiceCloning, VoiceEmbedding,
};

/// Load a TTS model based on the provided configuration.
///
/// This is the main entry point for creating a model instance. It dispatches
/// to the appropriate model backend based on `config.model_type`.
pub fn load_model(config: TtsConfig) -> Result<Box<dyn TtsModel>, TtsError> {
    match config.model_type {
        #[cfg(feature = "kokoro")]
        ModelType::Kokoro => {
            let model = models::kokoro::KokoroModel::load(config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "omnivoice")]
        ModelType::OmniVoice => {
            let model = models::omnivoice::OmniVoiceModel::load(config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "qwen3-tts")]
        ModelType::Qwen3Tts => {
            let model = models::qwen3_tts::Qwen3TtsModel::load(config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "vibevoice")]
        ModelType::VibeVoice => {
            let model = models::vibevoice::VibeVoiceModel::load(config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "vibevoice")]
        ModelType::VibeVoiceRealtime => {
            let model = models::vibevoice_realtime::VibeVoiceRealtimeModel::load(config)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "voxtral")]
        ModelType::Voxtral => {
            let model = models::voxtral::VoxtralModel::load(config)?;
            Ok(Box::new(model))
        }
        #[allow(unreachable_patterns)]
        _ => Err(TtsError::UnsupportedModel(format!(
            "Model type {:?} is not enabled. Enable the corresponding feature flag.",
            config.model_type
        ))),
    }
}
