use std::path::Path;

use serde::Deserialize;

use crate::error::TtsError;

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: String,
    #[serde(default = "default_acoustic_vae_dim")]
    pub acoustic_vae_dim: usize,
    #[serde(default = "default_semantic_vae_dim")]
    pub semantic_vae_dim: usize,
    #[serde(default)]
    pub decoder_config: VibeVoiceDecoderConfig,
    #[serde(default)]
    pub acoustic_tokenizer_config: VibeVoiceTokenizerConfig,
    #[serde(default)]
    pub semantic_tokenizer_config: VibeVoiceTokenizerConfig,
    #[serde(default)]
    pub diffusion_head_config: VibeVoiceDiffusionHeadConfig,
}

impl VibeVoiceConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TtsError> {
        Self::from_bytes(std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, TtsError> {
        serde_json::from_slice(bytes.as_ref())
            .map_err(|e| TtsError::ConfigError(format!("Failed to parse VibeVoice config: {}", e)))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceDecoderConfig {
    #[serde(default = "default_decoder_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
}

impl Default for VibeVoiceDecoderConfig {
    fn default() -> Self {
        Self {
            model_type: default_decoder_model_type(),
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            intermediate_size: default_intermediate_size(),
            vocab_size: default_vocab_size(),
            max_position_embeddings: default_max_position_embeddings(),
            rope_theta: default_rope_theta(),
            rms_norm_eps: default_rms_norm_eps(),
            hidden_act: default_hidden_act(),
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: default_tie_word_embeddings(),
            attention_bias: default_attention_bias(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceTokenizerConfig {
    #[serde(default = "default_tokenizer_model_type")]
    pub model_type: String,
    #[serde(default = "default_channels")]
    pub channels: usize,
    #[serde(default = "default_corpus_normalize")]
    pub corpus_normalize: f64,
    #[serde(default = "default_causal")]
    pub causal: bool,
    #[serde(default = "default_acoustic_vae_dim")]
    pub vae_dim: usize,
    #[serde(default = "default_fix_std")]
    pub fix_std: f64,
    #[serde(default = "default_std_dist_type")]
    pub std_dist_type: String,
    #[serde(default = "default_mixer_layer")]
    pub mixer_layer: String,
    #[serde(default = "default_conv_norm")]
    pub conv_norm: String,
    #[serde(default = "default_pad_mode")]
    pub pad_mode: String,
    #[serde(default = "default_disable_last_norm")]
    pub disable_last_norm: bool,
    #[serde(default = "default_layernorm")]
    pub layernorm: String,
    #[serde(default = "default_layernorm_eps")]
    pub layernorm_eps: f64,
    #[serde(default = "default_layernorm_elementwise_affine")]
    pub layernorm_elementwise_affine: bool,
    #[serde(default = "default_conv_bias")]
    pub conv_bias: bool,
    #[serde(default = "default_layer_scale_init_value")]
    pub layer_scale_init_value: f64,
    #[serde(default = "default_weight_init_value")]
    pub weight_init_value: f64,
    #[serde(default = "default_encoder_n_filters")]
    pub encoder_n_filters: usize,
    #[serde(default = "default_encoder_ratios")]
    pub encoder_ratios: Vec<usize>,
    #[serde(default = "default_encoder_depths")]
    pub encoder_depths: String,
    #[serde(default = "default_decoder_n_filters")]
    pub decoder_n_filters: usize,
    #[serde(default)]
    pub decoder_ratios: Option<Vec<usize>>,
    #[serde(default)]
    pub decoder_depths: Option<String>,
}

impl Default for VibeVoiceTokenizerConfig {
    fn default() -> Self {
        Self {
            model_type: default_tokenizer_model_type(),
            channels: default_channels(),
            corpus_normalize: default_corpus_normalize(),
            causal: default_causal(),
            vae_dim: default_acoustic_vae_dim(),
            fix_std: default_fix_std(),
            std_dist_type: default_std_dist_type(),
            mixer_layer: default_mixer_layer(),
            conv_norm: default_conv_norm(),
            pad_mode: default_pad_mode(),
            disable_last_norm: default_disable_last_norm(),
            layernorm: default_layernorm(),
            layernorm_eps: default_layernorm_eps(),
            layernorm_elementwise_affine: default_layernorm_elementwise_affine(),
            conv_bias: default_conv_bias(),
            layer_scale_init_value: default_layer_scale_init_value(),
            weight_init_value: default_weight_init_value(),
            encoder_n_filters: default_encoder_n_filters(),
            encoder_ratios: default_encoder_ratios(),
            encoder_depths: default_encoder_depths(),
            decoder_n_filters: default_decoder_n_filters(),
            decoder_ratios: None,
            decoder_depths: None,
        }
    }
}

impl VibeVoiceTokenizerConfig {
    pub fn encoder_depths(&self) -> Vec<usize> {
        parse_depths(&self.encoder_depths)
    }

    pub fn decoder_depths(&self) -> Vec<usize> {
        self.decoder_depths
            .as_deref()
            .map(parse_depths)
            .unwrap_or_else(|| {
                let mut depths = self.encoder_depths();
                depths.reverse();
                depths
            })
    }

    pub fn decoder_ratios(&self) -> Vec<usize> {
        self.decoder_ratios
            .clone()
            .unwrap_or_else(|| self.encoder_ratios.clone())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceDiffusionHeadConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_head_ffn_ratio")]
    pub head_ffn_ratio: f64,
    #[serde(default = "default_diffusion_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_acoustic_vae_dim")]
    pub latent_size: usize,
    #[serde(default)]
    pub speech_vae_dim: Option<usize>,
    #[serde(default = "default_prediction_type")]
    pub prediction_type: String,
    #[serde(default = "default_diffusion_type")]
    pub diffusion_type: String,
    #[serde(default = "default_ddpm_num_steps")]
    pub ddpm_num_steps: usize,
    #[serde(default = "default_ddpm_num_inference_steps")]
    pub ddpm_num_inference_steps: usize,
    #[serde(default = "default_ddpm_beta_schedule")]
    pub ddpm_beta_schedule: String,
    #[serde(default = "default_ddpm_batch_mul")]
    pub ddpm_batch_mul: usize,
}

impl Default for VibeVoiceDiffusionHeadConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            head_layers: default_head_layers(),
            head_ffn_ratio: default_head_ffn_ratio(),
            rms_norm_eps: default_diffusion_rms_norm_eps(),
            latent_size: default_acoustic_vae_dim(),
            speech_vae_dim: Some(default_acoustic_vae_dim()),
            prediction_type: default_prediction_type(),
            diffusion_type: default_diffusion_type(),
            ddpm_num_steps: default_ddpm_num_steps(),
            ddpm_num_inference_steps: default_ddpm_num_inference_steps(),
            ddpm_beta_schedule: default_ddpm_beta_schedule(),
            ddpm_batch_mul: default_ddpm_batch_mul(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoicePreprocessorConfig {
    #[serde(default = "default_speech_tok_compress_ratio")]
    pub speech_tok_compress_ratio: usize,
    #[serde(default = "default_db_normalize")]
    pub db_normalize: bool,
    #[serde(default = "default_language_model_pretrained_name")]
    pub language_model_pretrained_name: String,
    #[serde(default)]
    pub audio_processor: VibeVoiceAudioProcessorConfig,
}

impl Default for VibeVoicePreprocessorConfig {
    fn default() -> Self {
        Self {
            speech_tok_compress_ratio: default_speech_tok_compress_ratio(),
            db_normalize: default_db_normalize(),
            language_model_pretrained_name: default_language_model_pretrained_name(),
            audio_processor: VibeVoiceAudioProcessorConfig::default(),
        }
    }
}

impl VibeVoicePreprocessorConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TtsError> {
        Self::from_bytes(std::fs::read(path)?)
    }

    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, TtsError> {
        serde_json::from_slice(bytes.as_ref()).map_err(|e| {
            TtsError::ConfigError(format!(
                "Failed to parse VibeVoice preprocessor config: {}",
                e
            ))
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceAudioProcessorConfig {
    #[serde(default = "default_sample_rate")]
    pub sampling_rate: u32,
    #[serde(default = "default_normalize_audio")]
    pub normalize_audio: bool,
    #[serde(default = "default_target_db_fs")]
    pub target_d_b_fs: f32,
    #[serde(default = "default_audio_eps")]
    pub eps: f32,
}

impl Default for VibeVoiceAudioProcessorConfig {
    fn default() -> Self {
        Self {
            sampling_rate: default_sample_rate(),
            normalize_audio: default_normalize_audio(),
            target_d_b_fs: default_target_db_fs(),
            eps: default_audio_eps(),
        }
    }
}

pub fn parse_depths(spec: &str) -> Vec<usize> {
    spec.split('-')
        .filter_map(|value| value.parse::<usize>().ok())
        .collect()
}

fn default_model_type() -> String {
    "vibevoice".to_string()
}

fn default_decoder_model_type() -> String {
    "qwen2".to_string()
}

fn default_tokenizer_model_type() -> String {
    "vibevoice_acoustic_tokenizer".to_string()
}

fn default_torch_dtype() -> String {
    "bfloat16".to_string()
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_attention_bias() -> bool {
    true
}

fn default_hidden_size() -> usize {
    1536
}

fn default_num_hidden_layers() -> usize {
    28
}

fn default_num_attention_heads() -> usize {
    12
}

fn default_num_key_value_heads() -> usize {
    2
}

fn default_intermediate_size() -> usize {
    8960
}

fn default_vocab_size() -> usize {
    151_936
}

fn default_max_position_embeddings() -> usize {
    65_536
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

fn default_channels() -> usize {
    1
}

fn default_corpus_normalize() -> f64 {
    0.0
}

fn default_causal() -> bool {
    true
}

fn default_acoustic_vae_dim() -> usize {
    64
}

fn default_semantic_vae_dim() -> usize {
    128
}

fn default_fix_std() -> f64 {
    0.5
}

fn default_std_dist_type() -> String {
    "gaussian".to_string()
}

fn default_mixer_layer() -> String {
    "depthwise_conv".to_string()
}

fn default_conv_norm() -> String {
    "none".to_string()
}

fn default_pad_mode() -> String {
    "constant".to_string()
}

fn default_disable_last_norm() -> bool {
    true
}

fn default_layernorm() -> String {
    "RMSNorm".to_string()
}

fn default_layernorm_eps() -> f64 {
    1e-5
}

fn default_layernorm_elementwise_affine() -> bool {
    true
}

fn default_conv_bias() -> bool {
    true
}

fn default_layer_scale_init_value() -> f64 {
    1e-6
}

fn default_weight_init_value() -> f64 {
    1e-2
}

fn default_encoder_n_filters() -> usize {
    32
}

fn default_decoder_n_filters() -> usize {
    32
}

fn default_encoder_ratios() -> Vec<usize> {
    vec![8, 5, 5, 4, 2, 2]
}

fn default_encoder_depths() -> String {
    "3-3-3-3-3-3-8".to_string()
}

fn default_head_layers() -> usize {
    4
}

fn default_head_ffn_ratio() -> f64 {
    3.0
}

fn default_diffusion_rms_norm_eps() -> f64 {
    1e-5
}

fn default_prediction_type() -> String {
    "v_prediction".to_string()
}

fn default_diffusion_type() -> String {
    "ddpm".to_string()
}

fn default_ddpm_num_steps() -> usize {
    1000
}

fn default_ddpm_num_inference_steps() -> usize {
    20
}

fn default_ddpm_beta_schedule() -> String {
    "cosine".to_string()
}

fn default_ddpm_batch_mul() -> usize {
    4
}

fn default_speech_tok_compress_ratio() -> usize {
    3200
}

fn default_db_normalize() -> bool {
    true
}

fn default_language_model_pretrained_name() -> String {
    "Qwen/Qwen2.5-1.5B".to_string()
}

fn default_sample_rate() -> u32 {
    24_000
}

fn default_normalize_audio() -> bool {
    true
}

fn default_target_db_fs() -> f32 {
    -25.0
}

fn default_audio_eps() -> f32 {
    1e-6
}
