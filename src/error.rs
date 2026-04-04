//! Error types for any-tts.

use thiserror::Error;

/// Primary error type for all TTS operations.
#[derive(Error, Debug)]
pub enum TtsError {
    /// The requested model type is not compiled in or not supported.
    #[error("Unsupported model: {0}")]
    UnsupportedModel(String),

    /// Failed to load model weights from disk.
    #[error("Failed to load model weights: {0}")]
    WeightLoadError(String),

    /// Failed to parse model configuration.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Error during tensor computation (candle).
    #[error("Compute error: {0}")]
    ComputeError(#[from] candle_core::Error),

    /// Error during text tokenization.
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    /// The requested voice/speaker is not available.
    #[error("Unknown voice: {0}")]
    UnknownVoice(String),

    /// The requested language is not supported.
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// I/O error (file access, download, etc.).
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Model path was not provided and download is not available.
    #[error("Model path not specified and download feature is not enabled")]
    ModelPathMissing,

    /// A required model file was not provided and could not be resolved.
    #[error("Required file missing: {0}")]
    FileMissing(String),

    /// Generic error for model-specific issues.
    #[error("Model error: {0}")]
    ModelError(String),

    /// Error from an external runtime, process, or HTTP service.
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Error during audio encoding (WAV, MP3, etc.).
    #[error("Audio encoding error: {0}")]
    AudioError(String),
}

/// Convenience type alias for Results with [`TtsError`].
pub type TtsResult<T> = Result<T, TtsError>;
