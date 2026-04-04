//! Model backends for TTS synthesis.

#[cfg(feature = "kokoro")]
pub mod kokoro;

#[cfg(feature = "omnivoice")]
pub mod omnivoice;

#[cfg(feature = "qwen3-tts")]
pub mod qwen3_tts;

#[cfg(feature = "vibevoice")]
pub mod vibevoice;

#[cfg(feature = "voxtral")]
pub mod voxtral;

/// Supported model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// Kokoro-82M: 82M parameter StyleTTS2 model with ISTFTNet decoder.
    Kokoro,
    /// OmniVoice: native Candle implementation for omnilingual zero-shot TTS.
    OmniVoice,
    /// Qwen3-TTS-12Hz-1.7B-CustomVoice: 1.7B multi-codebook LM.
    Qwen3Tts,
    /// VibeVoice-1.5B: native Candle implementation with diffusion speech tokens.
    VibeVoice,
    /// Voxtral-4B-TTS-2603: native Candle implementation.
    Voxtral,
}
