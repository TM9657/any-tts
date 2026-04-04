use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::TtsError;

const AUDIO_SPECIAL_TOKEN_COUNT: usize = 2;

#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub multimodal: MultimodalConfig,
}

impl VoxtralConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TtsError> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MultimodalConfig {
    pub bos_token_id: u32,
    pub audio_model_args: MultimodalAudioModelArgs,
    pub audio_tokenizer_args: AudioTokenizerArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncodingArgs {
    pub sampling_rate: u32,
    pub frame_rate: f64,
    pub num_codebooks: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AcousticTransformerArgs {
    pub input_dim: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    #[serde(default)]
    pub use_biases: bool,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_sigma")]
    pub sigma: f64,
    #[serde(default)]
    pub sigma_max: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MultimodalAudioModelArgs {
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub n_acoustic_codebook: usize,
    pub audio_encoding_args: AudioEncodingArgs,
    pub audio_token_id: u32,
    pub begin_audio_token_id: u32,
    #[serde(default)]
    pub input_embedding_concat_type: Option<String>,
    pub acoustic_transformer_args: AcousticTransformerArgs,
    #[serde(default)]
    pub p_uncond: Option<f64>,
    #[serde(default)]
    pub text_feature_bugged: Option<bool>,
    #[serde(default)]
    pub condition_dropped_token_id: Option<u32>,
}

impl MultimodalAudioModelArgs {
    pub fn codebook_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::with_capacity(self.n_acoustic_codebook + 1);
        sizes.push(self.semantic_codebook_size);
        sizes.extend(std::iter::repeat_n(
            self.acoustic_codebook_size,
            self.n_acoustic_codebook,
        ));
        sizes
    }

    pub fn get_codebook_sizes(
        &self,
        pad_to_multiple: Option<usize>,
        include_special_tokens: bool,
    ) -> Vec<usize> {
        self.codebook_sizes()
            .into_iter()
            .map(|mut size| {
                if include_special_tokens {
                    size += AUDIO_SPECIAL_TOKEN_COUNT;
                }
                if let Some(multiple) = pad_to_multiple {
                    size = round_up_to_multiple(size, multiple);
                }
                size
            })
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AudioTokenizerArgs {
    pub channels: usize,
    pub sampling_rate: u32,
    pub pretransform_patch_size: usize,
    pub patch_proj_kernel_size: usize,
    pub semantic_codebook_size: usize,
    pub semantic_dim: usize,
    pub acoustic_codebook_size: usize,
    pub acoustic_dim: usize,
    pub conv_weight_norm: bool,
    pub causal: bool,
    pub attn_sliding_window_size: usize,
    pub half_attn_window_upon_downsampling: bool,
    pub dim: usize,
    pub hidden_dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub qk_norm_eps: f64,
    pub qk_norm: bool,
    pub use_biases: bool,
    pub norm_eps: f64,
    pub layer_scale: bool,
    pub layer_scale_init: Option<f64>,
    pub encoder_transformer_lengths_str: String,
    pub encoder_convs_kernels_str: String,
    pub encoder_convs_strides_str: String,
    pub decoder_transformer_lengths_str: String,
    pub decoder_convs_kernels_str: String,
    pub decoder_convs_strides_str: String,
    pub voice: BTreeMap<String, u32>,
}

impl Default for AudioTokenizerArgs {
    fn default() -> Self {
        Self {
            channels: 1,
            sampling_rate: 24_000,
            pretransform_patch_size: 240,
            patch_proj_kernel_size: 7,
            semantic_codebook_size: 8192,
            semantic_dim: 256,
            acoustic_codebook_size: 21,
            acoustic_dim: 36,
            conv_weight_norm: true,
            causal: true,
            attn_sliding_window_size: 16,
            half_attn_window_upon_downsampling: true,
            dim: 1024,
            hidden_dim: 4096,
            head_dim: 128,
            n_heads: 8,
            n_kv_heads: 8,
            qk_norm_eps: 1e-6,
            qk_norm: true,
            use_biases: false,
            norm_eps: 1e-2,
            layer_scale: true,
            layer_scale_init: None,
            encoder_transformer_lengths_str: "2,2,2,2".to_string(),
            encoder_convs_kernels_str: "4,4,4,3".to_string(),
            encoder_convs_strides_str: "2,2,2,1".to_string(),
            decoder_transformer_lengths_str: "2,2,2,2".to_string(),
            decoder_convs_kernels_str: "3,4,4,4".to_string(),
            decoder_convs_strides_str: "1,2,2,2".to_string(),
            voice: BTreeMap::new(),
        }
    }
}

impl AudioTokenizerArgs {
    pub fn encoder_transformer_lengths(&self) -> Result<Vec<usize>, TtsError> {
        parse_csv_usize(&self.encoder_transformer_lengths_str)
    }

    pub fn encoder_convs_kernels(&self) -> Result<Vec<usize>, TtsError> {
        parse_csv_usize(&self.encoder_convs_kernels_str)
    }

    pub fn encoder_convs_strides(&self) -> Result<Vec<usize>, TtsError> {
        parse_csv_usize(&self.encoder_convs_strides_str)
    }

    pub fn decoder_transformer_lengths(&self) -> Result<Vec<usize>, TtsError> {
        parse_csv_usize(&self.decoder_transformer_lengths_str)
    }

    pub fn decoder_convs_kernels(&self) -> Result<Vec<usize>, TtsError> {
        parse_csv_usize(&self.decoder_convs_kernels_str)
    }

    pub fn decoder_convs_strides(&self) -> Result<Vec<usize>, TtsError> {
        parse_csv_usize(&self.decoder_convs_strides_str)
    }

    pub fn frame_rate(&self) -> Result<f64, TtsError> {
        let scale_factor: usize = self.encoder_convs_strides()?.into_iter().product();
        Ok(self.sampling_rate as f64 / (self.pretransform_patch_size * scale_factor) as f64)
    }

    pub fn voice_names(&self) -> Vec<String> {
        let mut entries: Vec<_> = self.voice.iter().collect();
        entries.sort_by_key(|(_, index)| **index);
        entries.into_iter().map(|(name, _)| name.clone()).collect()
    }
}

fn default_norm_eps() -> f64 {
    1e-5
}

fn default_sigma() -> f64 {
    1e-5
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    multiple * value.div_ceil(multiple)
}

fn parse_csv_usize(value: &str) -> Result<Vec<usize>, TtsError> {
    value
        .split(',')
        .map(|part| {
            part.parse::<usize>().map_err(|err| {
                TtsError::ConfigError(format!("Invalid Voxtral config list '{value}': {err}"))
            })
        })
        .collect()
}