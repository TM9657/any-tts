//! Kokoro model configuration.
//!
//! Parsed from `config.json` in the HuggingFace repo.
//! Contains PLBert, ISTFTNet, and phoneme vocabulary configurations.

use serde::Deserialize;
use std::collections::HashMap;

/// Top-level Kokoro config.
#[derive(Debug, Clone, Deserialize)]
pub struct KokoroConfig {
    /// Hidden dimension for the model (default: 512).
    #[serde(default = "default_hidden_dim")]
    pub hidden_dim: usize,

    /// Style dimension (default: 128). Total voice embedding is 2 × style_dim.
    #[serde(default = "default_style_dim")]
    pub style_dim: usize,

    /// Number of mel spectrogram channels (default: 80).
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,

    /// Number of tokens in the phoneme vocabulary (default: 178).
    #[serde(default = "default_n_token")]
    pub n_token: usize,

    /// Number of transformer/LSTM layers (default: 3).
    #[serde(default = "default_n_layer")]
    pub n_layer: usize,

    /// Input dimension for decoder (default: 64).
    #[serde(default = "default_dim_in")]
    pub dim_in: usize,

    /// Dropout rate (unused during inference).
    #[serde(default = "default_dropout")]
    pub dropout: f64,

    /// Maximum convolution dimension (default: 512).
    #[serde(default = "default_max_conv_dim")]
    pub max_conv_dim: usize,

    /// Maximum duration in frames (default: 50).
    #[serde(default = "default_max_dur")]
    pub max_dur: usize,

    /// Whether the model supports multiple speakers.
    #[serde(default = "default_multispeaker")]
    pub multispeaker: bool,

    /// Text encoder convolution kernel size (default: 5).
    #[serde(default = "default_text_encoder_kernel_size")]
    pub text_encoder_kernel_size: usize,

    /// PL-BERT configuration.
    #[serde(default)]
    pub plbert: PlbertConfig,

    /// ISTFTNet decoder configuration.
    #[serde(default)]
    pub istftnet: IstftNetConfig,

    /// Phoneme vocabulary: maps IPA phoneme strings to token IDs.
    #[serde(default)]
    pub vocab: HashMap<String, u32>,
}

/// PL-BERT (ALBERT-based) text encoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct PlbertConfig {
    /// Vocabulary size (default: 178).
    #[serde(default = "default_plbert_vocab_size")]
    pub vocab_size: usize,

    /// Hidden size (default: 768).
    #[serde(default = "default_plbert_hidden_size")]
    pub hidden_size: usize,

    /// Number of attention heads (default: 12).
    #[serde(default = "default_plbert_num_heads")]
    pub num_attention_heads: usize,

    /// Number of hidden layers (default: 12).
    #[serde(default = "default_plbert_num_layers")]
    pub num_hidden_layers: usize,

    /// Intermediate (FFN) size (default: 2048).
    #[serde(default = "default_plbert_intermediate_size")]
    pub intermediate_size: usize,

    /// Maximum position embeddings (default: 512).
    #[serde(default = "default_plbert_max_position")]
    pub max_position_embeddings: usize,

    /// Embedding size (default: 768).
    #[serde(default = "default_plbert_embedding_size")]
    pub embedding_size: usize,

    /// Number of hidden groups for ALBERT sharing (default: 1).
    #[serde(default = "default_plbert_num_hidden_groups")]
    pub num_hidden_groups: usize,

    /// Hidden act (default: "gelu").
    #[serde(default = "default_plbert_hidden_act")]
    pub hidden_act: String,

    /// Hidden dropout probability.
    #[serde(default = "default_plbert_dropout")]
    pub hidden_dropout_prob: f64,

    /// Attention dropout probability.
    #[serde(default = "default_plbert_dropout")]
    pub attention_probs_dropout_prob: f64,

    /// Type vocabulary size (default: 2).
    #[serde(default = "default_plbert_type_vocab_size")]
    pub type_vocab_size: usize,
}

/// ISTFTNet decoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct IstftNetConfig {
    /// Upsampling rates (default: [10, 6]).
    #[serde(default = "default_upsample_rates")]
    pub upsample_rates: Vec<usize>,

    /// Upsampling kernel sizes (default: [20, 12]).
    #[serde(default = "default_upsample_kernel_sizes")]
    pub upsample_kernel_sizes: Vec<usize>,

    /// Initial channel count for upsampling (default: 512).
    #[serde(default = "default_upsample_initial_channel")]
    pub upsample_initial_channel: usize,

    /// ResBlock kernel sizes (default: [3, 7, 11]).
    #[serde(default = "default_resblock_kernel_sizes")]
    pub resblock_kernel_sizes: Vec<usize>,

    /// ResBlock dilation sizes (default: [[1,3,5],[1,3,5],[1,3,5]]).
    #[serde(default = "default_resblock_dilation_sizes")]
    pub resblock_dilation_sizes: Vec<Vec<usize>>,

    /// Generator iSTFT N-FFT (default: 20).
    #[serde(default = "default_gen_istft_n_fft")]
    pub gen_istft_n_fft: usize,

    /// Generator iSTFT hop size (default: 5).
    #[serde(default = "default_gen_istft_hop_size")]
    pub gen_istft_hop_size: usize,
}

// Default value functions
fn default_hidden_dim() -> usize {
    512
}
fn default_style_dim() -> usize {
    128
}
fn default_n_mels() -> usize {
    80
}
fn default_n_token() -> usize {
    178
}
fn default_n_layer() -> usize {
    3
}
fn default_dim_in() -> usize {
    64
}
fn default_dropout() -> f64 {
    0.2
}
fn default_max_conv_dim() -> usize {
    512
}
fn default_max_dur() -> usize {
    50
}
fn default_multispeaker() -> bool {
    true
}
fn default_text_encoder_kernel_size() -> usize {
    5
}

fn default_plbert_vocab_size() -> usize {
    178
}
fn default_plbert_hidden_size() -> usize {
    768
}
fn default_plbert_num_heads() -> usize {
    12
}
fn default_plbert_num_layers() -> usize {
    12
}
fn default_plbert_intermediate_size() -> usize {
    2048
}
fn default_plbert_max_position() -> usize {
    512
}
fn default_plbert_embedding_size() -> usize {
    128
}
fn default_plbert_num_hidden_groups() -> usize {
    1
}
fn default_plbert_hidden_act() -> String {
    "gelu".to_string()
}
fn default_plbert_dropout() -> f64 {
    0.1
}
fn default_plbert_type_vocab_size() -> usize {
    2
}

fn default_upsample_rates() -> Vec<usize> {
    vec![10, 6]
}
fn default_upsample_kernel_sizes() -> Vec<usize> {
    vec![20, 12]
}
fn default_upsample_initial_channel() -> usize {
    512
}
fn default_resblock_kernel_sizes() -> Vec<usize> {
    vec![3, 7, 11]
}
fn default_resblock_dilation_sizes() -> Vec<Vec<usize>> {
    vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
}
fn default_gen_istft_n_fft() -> usize {
    20
}
fn default_gen_istft_hop_size() -> usize {
    5
}

impl Default for PlbertConfig {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

impl Default for IstftNetConfig {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

impl KokoroConfig {
    /// Load config from a `config.json` file.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::TtsError> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// Load config from in-memory `config.json` bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, crate::error::TtsError> {
        let config: Self = serde_json::from_slice(bytes.as_ref())?;
        Ok(config)
    }

    /// Total upsampling factor of the ISTFTNet decoder.
    pub fn upsample_factor(&self) -> usize {
        let conv_factor: usize = self.istftnet.upsample_rates.iter().product();
        conv_factor * self.istftnet.gen_istft_hop_size
    }

    /// Get the full style dimension (2 × style_dim for decoder + predictor).
    pub fn full_style_dim(&self) -> usize {
        self.style_dim * 2
    }

    /// Output sample rate.
    pub fn sample_rate(&self) -> u32 {
        24000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let json = "{}";
        let config: KokoroConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.style_dim, 128);
        assert_eq!(config.n_token, 178);
        assert_eq!(config.n_layer, 3);
    }

    #[test]
    fn test_plbert_defaults() {
        let json = r#"{"plbert": {}}"#;
        let config: KokoroConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.plbert.hidden_size, 768);
        assert_eq!(config.plbert.num_attention_heads, 12);
        assert_eq!(config.plbert.num_hidden_layers, 12);
    }

    #[test]
    fn test_istftnet_defaults() {
        let json = r#"{"istftnet": {}}"#;
        let config: KokoroConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.istftnet.upsample_rates, vec![10, 6]);
        assert_eq!(config.istftnet.gen_istft_hop_size, 5);
    }

    #[test]
    fn test_upsample_factor() {
        let config: KokoroConfig = serde_json::from_str("{}").unwrap();
        // 10 * 6 * 5 = 300
        assert_eq!(config.upsample_factor(), 300);
    }

    #[test]
    fn test_vocab_parsing() {
        let json = r#"{
            "vocab": {
                ";": 1,
                "a": 2,
                "b": 3
            }
        }"#;
        let config: KokoroConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.vocab.len(), 3);
        assert_eq!(config.vocab["a"], 2);
    }
}
