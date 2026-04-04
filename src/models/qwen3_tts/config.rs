//! Qwen3-TTS model configuration.
//!
//! Parsed from `config.json` in the HuggingFace repo.

use serde::Deserialize;
use std::collections::HashMap;

/// Top-level Qwen3-TTS config.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsConfig {
    /// Model type identifier.
    #[serde(default = "default_model_type")]
    pub model_type: String,

    /// Talker (main LM) configuration.
    #[serde(default)]
    pub talker_config: TalkerConfig,

    /// Speaker encoder configuration (used only for Base/clone models).
    #[serde(default)]
    pub speaker_encoder_config: Option<SpeakerEncoderConfig>,

    /// Speech tokenizer type ("qwen3_tts_tokenizer_12hz" or "qwen3_tts_tokenizer_25hz").
    #[serde(default = "default_tokenizer_type")]
    pub tokenizer_type: String,

    /// Model size variant.
    #[serde(default)]
    pub tts_model_size: Option<String>,

    /// Model type: "custom_voice", "voice_design", or "base".
    #[serde(default = "default_tts_model_type")]
    pub tts_model_type: String,

    /// Special token IDs.
    #[serde(default = "default_im_start_token_id")]
    pub im_start_token_id: u32,
    #[serde(default = "default_im_end_token_id")]
    pub im_end_token_id: u32,
    #[serde(default = "default_tts_pad_token_id")]
    pub tts_pad_token_id: u32,
    #[serde(default = "default_tts_bos_token_id")]
    pub tts_bos_token_id: u32,
    #[serde(default = "default_tts_eos_token_id")]
    pub tts_eos_token_id: u32,
}

/// Configuration for the Talker (main language model).
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    /// Hidden size of the talker transformer.
    #[serde(default = "default_talker_hidden_size")]
    pub hidden_size: usize,

    /// Number of hidden layers.
    #[serde(default = "default_talker_num_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    #[serde(default = "default_talker_num_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA).
    #[serde(default = "default_talker_num_kv_heads")]
    pub num_key_value_heads: usize,

    /// Per-head dimension (may differ from hidden_size / num_heads).
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    /// Intermediate size in the MLP.
    #[serde(default = "default_talker_intermediate_size")]
    pub intermediate_size: usize,

    /// Codec vocabulary size.
    #[serde(default = "default_talker_vocab_size")]
    pub vocab_size: usize,

    /// Text vocabulary size.
    #[serde(default = "default_text_vocab_size")]
    pub text_vocab_size: usize,

    /// Text embedding hidden size.
    #[serde(default = "default_text_hidden_size")]
    pub text_hidden_size: usize,

    /// Number of codebook groups.
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,

    /// RMS norm epsilon.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// RoPE theta base.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Maximum position embeddings.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Hidden activation function.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// Codec EOS token ID.
    #[serde(default = "default_codec_eos_token_id")]
    pub codec_eos_token_id: u32,

    /// Codec BOS token ID (VoiceDesign: 2149).
    #[serde(default)]
    pub codec_bos_id: Option<u32>,

    /// Codec PAD token ID (VoiceDesign: 2148).
    #[serde(default)]
    pub codec_pad_id: Option<u32>,

    /// Codec think token ID (VoiceDesign: 2154).
    #[serde(default)]
    pub codec_think_id: Option<u32>,

    /// Codec no-think token ID (VoiceDesign: 2155).
    #[serde(default)]
    pub codec_nothink_id: Option<u32>,

    /// Codec think BOS token ID (VoiceDesign: 2156).
    #[serde(default)]
    pub codec_think_bos_id: Option<u32>,

    /// Codec think EOS token ID (VoiceDesign: 2157).
    #[serde(default)]
    pub codec_think_eos_id: Option<u32>,

    /// Position IDs per second for MRoPE (VoiceDesign: 13).
    #[serde(default)]
    pub position_id_per_seconds: Option<u32>,

    /// MRoPE section sizes (VoiceDesign: [24, 20, 20]).
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,

    /// Whether MRoPE uses interleaved layout.
    #[serde(default)]
    pub interleaved: Option<bool>,

    /// Speaker ID mapping.
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,

    /// Language ID mapping.
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,

    /// Code predictor configuration.
    #[serde(default)]
    pub code_predictor_config: Option<CodePredictorConfig>,
}

/// Configuration for the Code Predictor sub-model.
#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    /// Hidden size.
    #[serde(default = "default_code_predictor_hidden")]
    pub hidden_size: usize,

    /// Number of hidden layers.
    #[serde(default = "default_code_predictor_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads.
    #[serde(default = "default_code_predictor_heads")]
    pub num_attention_heads: usize,

    /// Number of KV heads.
    #[serde(default = "default_code_predictor_kv_heads")]
    pub num_key_value_heads: usize,

    /// Per-head dimension.
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    /// Intermediate size for MLP.
    #[serde(default = "default_code_predictor_intermediate")]
    pub intermediate_size: usize,

    /// Vocab size (codec vocabulary for the code predictor).
    #[serde(default = "default_code_predictor_vocab")]
    pub vocab_size: usize,

    /// Number of code groups.
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
}

/// Speaker encoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerEncoderConfig {
    /// Input sample rate for speaker encoder.
    #[serde(default = "default_speaker_sample_rate")]
    pub sample_rate: u32,
}

// Default value functions
fn default_model_type() -> String {
    "qwen3_tts".to_string()
}
fn default_tokenizer_type() -> String {
    "qwen3_tts_tokenizer_12hz".to_string()
}
fn default_tts_model_type() -> String {
    "custom_voice".to_string()
}
fn default_im_start_token_id() -> u32 {
    151644
}
fn default_im_end_token_id() -> u32 {
    151645
}
fn default_tts_pad_token_id() -> u32 {
    151671
}
fn default_tts_bos_token_id() -> u32 {
    151672
}
fn default_tts_eos_token_id() -> u32 {
    151673
}
fn default_talker_hidden_size() -> usize {
    2048
}
fn default_talker_num_layers() -> usize {
    28
}
fn default_talker_num_heads() -> usize {
    16
}
fn default_talker_num_kv_heads() -> usize {
    8
}
fn default_head_dim() -> usize {
    128
}
fn default_talker_intermediate_size() -> usize {
    6144
}
fn default_talker_vocab_size() -> usize {
    3072
}
fn default_text_vocab_size() -> usize {
    151936
}
fn default_text_hidden_size() -> usize {
    2048
}
fn default_num_code_groups() -> usize {
    16
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1000000.0
}
fn default_max_position_embeddings() -> usize {
    32768
}
fn default_hidden_act() -> String {
    "silu".to_string()
}
fn default_codec_eos_token_id() -> u32 {
    0
}
fn default_code_predictor_hidden() -> usize {
    1024
}
fn default_code_predictor_layers() -> usize {
    5
}
fn default_code_predictor_heads() -> usize {
    16
}
fn default_code_predictor_kv_heads() -> usize {
    8
}
fn default_code_predictor_intermediate() -> usize {
    3072
}
fn default_code_predictor_vocab() -> usize {
    2048
}
fn default_speaker_sample_rate() -> u32 {
    16000
}

impl Default for TalkerConfig {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

impl Qwen3TtsConfig {
    /// Load config from a `config.json` file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::TtsError> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Whether this is a VoiceDesign model (uses text descriptions, no named speakers).
    pub fn is_voice_design(&self) -> bool {
        self.tts_model_type == "voice_design"
    }

    /// Get the list of supported speaker names.
    /// VoiceDesign models return empty (they use text descriptions instead).
    pub fn speakers(&self) -> Vec<String> {
        self.talker_config.spk_id.keys().cloned().collect()
    }

    /// Get the list of supported languages (excluding dialect variants).
    pub fn languages(&self) -> Vec<String> {
        self.talker_config
            .codec_language_id
            .keys()
            .filter(|k| !k.contains("dialect"))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let json = "{}";
        let config: Qwen3TtsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model_type, "qwen3_tts");
        assert_eq!(config.tts_model_type, "custom_voice");
        assert_eq!(config.tts_bos_token_id, 151672);
    }

    #[test]
    fn test_talker_config_defaults() {
        let json = r#"{"talker_config": {}}"#;
        let config: Qwen3TtsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.talker_config.hidden_size, 2048);
        assert_eq!(config.talker_config.num_code_groups, 16);
    }

    #[test]
    fn test_speaker_id_parsing() {
        let json = r#"{
            "talker_config": {
                "spk_id": {"Vivian": 0, "Ryan": 1, "Serena": 2}
            }
        }"#;
        let config: Qwen3TtsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.talker_config.spk_id.len(), 3);
        assert_eq!(config.talker_config.spk_id["Vivian"], 0);
    }

    #[test]
    fn test_language_filtering() {
        let json = r#"{
            "talker_config": {
                "codec_language_id": {
                    "Chinese": 0,
                    "English": 1,
                    "Chinese_dialect_sichuan": 2
                }
            }
        }"#;
        let config: Qwen3TtsConfig = serde_json::from_str(json).unwrap();
        let langs = config.languages();
        assert!(langs.contains(&"Chinese".to_string()));
        assert!(langs.contains(&"English".to_string()));
        assert!(!langs.iter().any(|l| l.contains("dialect")));
    }
}
