use serde::Deserialize;

use crate::error::TtsError;
use crate::models::vibevoice::config::{
    VibeVoiceDecoderConfig, VibeVoiceDiffusionHeadConfig, VibeVoiceTokenizerConfig,
};

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceRealtimeConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_acoustic_vae_dim")]
    pub acoustic_vae_dim: usize,
    #[serde(default)]
    pub decoder_config: VibeVoiceDecoderConfig,
    #[serde(default)]
    pub acoustic_tokenizer_config: VibeVoiceTokenizerConfig,
    #[serde(default)]
    pub diffusion_head_config: VibeVoiceDiffusionHeadConfig,
    #[serde(default = "default_tts_backbone_num_hidden_layers")]
    pub tts_backbone_num_hidden_layers: usize,
}

impl VibeVoiceRealtimeConfig {
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, TtsError> {
        serde_json::from_slice(bytes.as_ref()).map_err(|error| {
            TtsError::ConfigError(format!(
                "Failed to parse VibeVoice Realtime config: {error}"
            ))
        })
    }
}

fn default_model_type() -> String {
    "vibevoice_streaming".to_string()
}

fn default_acoustic_vae_dim() -> usize {
    64
}

fn default_tts_backbone_num_hidden_layers() -> usize {
    20
}
