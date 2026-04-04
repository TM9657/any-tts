use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use tracing::info;

use crate::audio::AudioSamples;
use crate::config::{ModelFiles, TtsConfig};
use crate::error::TtsError;
use crate::layers::attention::GqaConfig;
use crate::layers::transformer::TransformerBlock;
use crate::tensor_utils::{precompute_rope_freqs, RmsNorm};
use crate::tokenizer::TextTokenizer;
use crate::traits::{ModelInfo, SynthesisRequest, TtsModel};

use super::config::{
    VibeVoiceConfig,
    VibeVoiceDecoderConfig,
    VibeVoicePreprocessorConfig,
    VibeVoiceTokenizerConfig,
};
use super::diffusion::{DpmSolverMultistepScheduler, VibeVoiceDiffusionHead};
use super::processor::{PreparedVibeVoiceInput, VibeVoiceProcessor, VibeVoiceTokenizerSpec};
use super::speech_tokenizer::{
    VibeVoiceAcousticTokenizer,
    VibeVoiceSemanticTokenizer,
    VibeVoiceTokenizerEncoderOutput,
};

const DEFAULT_CFG_SCALE: f32 = 3.0;
const DEFAULT_GENERATION_SEED: u64 = 299_792_458;

struct SpeechConnector {
    fc1: Linear,
    norm: RmsNorm,
    fc2: Linear,
}

impl SpeechConnector {
    fn load(input_dim: usize, output_dim: usize, vb: VarBuilder) -> CandleResult<Self> {
        let fc1 = candle_nn::linear(input_dim, output_dim, vb.pp("fc1"))?;
        let norm = RmsNorm::load(output_dim, 1e-6, vb.pp("norm"))?;
        let fc2 = candle_nn::linear(output_dim, output_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, norm, fc2 })
    }

    fn forward(&self, features: &Tensor) -> CandleResult<Tensor> {
        let original_dims = features.dims().to_vec();
        let input_dim = *original_dims.last().unwrap_or(&0);
        let leading = if original_dims.len() <= 1 {
            1
        } else {
            original_dims[..original_dims.len() - 1]
                .iter()
                .product::<usize>()
        };
        let features_2d = if original_dims.len() == 2 {
            features.clone()
        } else {
            features.reshape((leading, input_dim))?
        };

        let hidden = self.fc1.forward(&features_2d)?;
        let hidden = self.norm.forward(&hidden)?;
        let hidden = self.fc2.forward(&hidden)?;

        if original_dims.len() == 2 {
            return Ok(hidden);
        }

        let mut output_dims = original_dims;
        if let Some(last) = output_dims.last_mut() {
            *last = hidden.dim(candle_core::D::Minus1)?;
        }
        hidden.reshape(output_dims)
    }
}

struct VibeVoiceLanguageModel {
    embed_tokens: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    rope_cos: Tensor,
    rope_sin: Tensor,
    dtype: DType,
}

impl VibeVoiceLanguageModel {
    fn load(
        config: &VibeVoiceDecoderConfig,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, TtsError> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let gqa_config = GqaConfig::with_head_dim(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rms_norm_eps,
        )
        .with_attention_bias(config.attention_bias);

        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(TransformerBlock::load(
                &gqa_config,
                config.intermediate_size,
                vb.pp(format!("layers.{}", index)),
            )?);
        }

        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let (rope_cos, rope_sin) = precompute_rope_freqs(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope_cos,
            rope_sin,
            dtype,
        })
    }

    fn embed(&self, token_ids: &Tensor) -> CandleResult<Tensor> {
        self.embed_tokens.forward(token_ids)
    }

    fn forward(&mut self, input_embeds: &Tensor) -> CandleResult<Tensor> {
        let (_batch, seq_len, _) = input_embeds.dims3()?;

        for layer in &mut self.layers {
            layer.clear_cache();
        }

        let mask = if seq_len > 1 {
            let mut mask_data = vec![f32::NEG_INFINITY; seq_len * seq_len];
            for row in 0..seq_len {
                for col in 0..=row {
                    mask_data[row * seq_len + col] = 0.0;
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), input_embeds.device())?
                .to_dtype(self.dtype)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            Some(mask)
        } else {
            None
        };

        let mut hidden = input_embeds.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(
                &hidden,
                &self.rope_cos,
                &self.rope_sin,
                0,
                mask.as_ref(),
            )?;
        }
        self.norm.forward(&hidden)
    }

    fn next_logits(&self, last_hidden: &Tensor) -> CandleResult<Tensor> {
        let weight = self.embed_tokens.embeddings().transpose(0, 1)?;
        last_hidden.matmul(&weight)
    }
}

pub struct VibeVoiceModel {
    config: VibeVoiceConfig,
    preprocessor_config: VibeVoicePreprocessorConfig,
    device: Device,
    dtype: DType,
    files: ModelFiles,
    processor: VibeVoiceProcessor,
    language_model: Mutex<VibeVoiceLanguageModel>,
    acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    semantic_tokenizer: VibeVoiceSemanticTokenizer,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    prediction_head: VibeVoiceDiffusionHead,
    noise_scheduler: Mutex<DpmSolverMultistepScheduler>,
    speech_scaling_factor: f32,
    speech_bias_factor: f32,
}

impl TtsModel for VibeVoiceModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError> {
        let device = config.device.resolve()?;
        let mut dtype = config.dtype.to_candle();

        if matches!(device, Device::Cpu) && dtype == DType::BF16 {
            info!("BF16 is not supported on CPU; falling back to F32 for VibeVoice");
            dtype = DType::F32;
        } else if matches!(device, Device::Metal(_)) {
            info!("VibeVoice uses F32 on Metal to match the Python reference path and avoid unstable first-token behavior");
            dtype = DType::F32;
        }

        let files = config.resolve_files()?;
        let model_config = VibeVoiceConfig::from_file(
            files.config.as_ref().expect("validated by resolve_files"),
        )?;
        let preprocessor_config = if let Some(path) = &files.preprocessor_config {
            VibeVoicePreprocessorConfig::from_file(path)?
        } else {
            VibeVoicePreprocessorConfig::default()
        };

        let tokenizer = TextTokenizer::from_file(
            files
                .tokenizer
                .as_ref()
                .expect("validated by resolve_files"),
        )?;
        let tokenizer_spec = VibeVoiceTokenizerSpec::from_tokenizer(&tokenizer)?;
        let processor = VibeVoiceProcessor::new(tokenizer, tokenizer_spec, preprocessor_config.clone());

        let vb = ModelFiles::load_safetensors_vb(&files.weights, dtype, &device)?;
        let model_vb = vb.pp("model");
        let language_model = VibeVoiceLanguageModel::load(
            &model_config.decoder_config,
            model_vb.pp("language_model"),
            &device,
            dtype,
        )?;
        let acoustic_tokenizer = VibeVoiceAcousticTokenizer::load(
            &model_config.acoustic_tokenizer_config,
            model_vb.pp("acoustic_tokenizer"),
        )?;
        let semantic_tokenizer = VibeVoiceSemanticTokenizer::load(
            &model_config.semantic_tokenizer_config,
            model_vb.pp("semantic_tokenizer"),
        )?;
        let acoustic_connector = SpeechConnector::load(
            model_config.acoustic_vae_dim,
            model_config.decoder_config.hidden_size,
            model_vb.pp("acoustic_connector"),
        )?;
        let semantic_connector = SpeechConnector::load(
            model_config.semantic_vae_dim,
            model_config.decoder_config.hidden_size,
            model_vb.pp("semantic_connector"),
        )?;
        let prediction_head = VibeVoiceDiffusionHead::load(
            &model_config.diffusion_head_config,
            model_vb.pp("prediction_head"),
        )?;

        let scaling_tensor = model_vb
            .get(1, "speech_scaling_factor")
            .or_else(|_| model_vb.get((), "speech_scaling_factor"))?;
        let bias_tensor = model_vb
            .get(1, "speech_bias_factor")
            .or_else(|_| model_vb.get((), "speech_bias_factor"))?;
        let speech_scaling_factor = scalar_from_tensor(&scaling_tensor)?;
        let speech_bias_factor = scalar_from_tensor(&bias_tensor)?;

        info!(
            "Loaded VibeVoice on {:?}: layers={}, hidden_size={}, acoustic_dim={}, semantic_dim={}",
            device,
            model_config.decoder_config.num_hidden_layers,
            model_config.decoder_config.hidden_size,
            model_config.acoustic_vae_dim,
            model_config.semantic_vae_dim,
        );

        Ok(Self {
            config: model_config.clone(),
            preprocessor_config,
            device,
            dtype,
            files,
            processor,
            language_model: Mutex::new(language_model),
            acoustic_tokenizer,
            semantic_tokenizer,
            acoustic_connector,
            semantic_connector,
            prediction_head,
            noise_scheduler: Mutex::new(DpmSolverMultistepScheduler::new(
                &model_config.diffusion_head_config,
            )),
            speech_scaling_factor,
            speech_bias_factor,
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        if request.voice.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice does not use named voice presets; provide reference audio instead."
                    .to_string(),
            ));
        }
        if request.voice_embedding.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice does not support pre-extracted voice embeddings yet.".to_string(),
            ));
        }

        let prepared = self.processor.prepare_request(request, &self.device)?;
        let mut rng = SimpleRng::new(DEFAULT_GENERATION_SEED);
        let mut token_ids = prepared.input_ids.clone();
        let mut embedding_overrides = self.prepare_prompt_overrides(&prepared, &mut rng)?;
        let spec = self.processor.tokenizer_spec().clone();
        let valid_token_ids = valid_generated_tokens(&spec);
        let mut current_segment = Vec::new();
        let mut finished_segments = Vec::new();
        let mut negative_tokens = vec![spec.speech_start_id];
        let mut negative_embedding_overrides = HashMap::new();
        let mut generated_trace = Vec::new();
        let max_new_tokens = request.max_tokens.unwrap_or_else(|| {
            self.config
                .decoder_config
                .max_position_embeddings
                .saturating_sub(token_ids.len())
                .min(2_048)
        });
        let temperature = request.temperature.unwrap_or(0.0) as f32;
        let cfg_scale = request.cfg_scale.unwrap_or(DEFAULT_CFG_SCALE as f64) as f32;

        for _step in 0..max_new_tokens {
            if token_ids.len() >= self.config.decoder_config.max_position_embeddings {
                break;
            }

            let (positive_hidden, logits) = self.forward_for_next_token(&token_ids, &embedding_overrides)?;
            let next_token = sample_token(&logits, &valid_token_ids, temperature, &mut rng)?;
            token_ids.push(next_token);
            generated_trace.push(next_token);

            if next_token == spec.speech_start_id {
                if !current_segment.is_empty() {
                    finished_segments.push(std::mem::take(&mut current_segment));
                }
                negative_tokens.clear();
                negative_tokens.push(spec.speech_start_id);
                continue;
            }

            if next_token == spec.eos_id {
                break;
            }

            if next_token == spec.speech_end_id {
                if !current_segment.is_empty() {
                    finished_segments.push(std::mem::take(&mut current_segment));
                }
                continue;
            }

            if next_token != spec.speech_diffusion_id {
                continue;
            }

            let negative_hidden = self.forward_negative_condition(
                &negative_tokens,
                &negative_embedding_overrides,
            )?;

            let speech_latent = self.sample_speech_token(&positive_hidden, &negative_hidden, cfg_scale, &mut rng)?;
            current_segment.push(speech_latent.clone());
            let diffusion_embed = self.build_diffusion_override(&current_segment)?;
            embedding_overrides.insert(token_ids.len() - 1, diffusion_embed);
            negative_tokens.push(spec.speech_diffusion_id);
            negative_embedding_overrides.insert(
                negative_tokens.len() - 1,
                embedding_overrides
                    .get(&(token_ids.len() - 1))
                    .expect("diffusion embedding must exist")
                    .clone(),
            );
        }

        if !current_segment.is_empty() {
            finished_segments.push(current_segment);
        }

        if std::env::var_os("VIBEVOICE_DEBUG_TRACE").is_some() {
            eprintln!("VibeVoice generated tokens: {:?}", generated_trace);
        }

        let samples = self.decode_segments(&finished_segments)?;
        if samples.is_empty() {
            return Err(TtsError::ModelError(
                format!(
                    "VibeVoice did not produce any speech tokens for this prompt. Generated tokens: {:?}",
                    generated_trace,
                ),
            ));
        }

        Ok(AudioSamples::new(
            samples,
            self.preprocessor_config.audio_processor.sampling_rate,
        ))
    }

    fn sample_rate(&self) -> u32 {
        self.preprocessor_config.audio_processor.sampling_rate
    }

    fn supported_languages(&self) -> Vec<String> {
        vec!["auto".to_string(), "multilingual".to_string()]
    }

    fn supported_voices(&self) -> Vec<String> {
        Vec::new()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "VibeVoice-1.5B".to_string(),
            variant: self.config.model_type.clone(),
            parameters: 1_500_000_000,
            sample_rate: self.sample_rate(),
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl VibeVoiceModel {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn files(&self) -> &ModelFiles {
        &self.files
    }

    fn prepare_prompt_overrides(
        &self,
        prepared: &PreparedVibeVoiceInput,
        rng: &mut SimpleRng,
    ) -> Result<HashMap<usize, Tensor>, TtsError> {
        let mut overrides = HashMap::new();
        let Some(speech_inputs) = &prepared.speech_inputs else {
            return Ok(overrides);
        };

        let speech_tensors = speech_inputs.speech_tensors.unsqueeze(1)?;
        let encoder_output = self.acoustic_tokenizer.encode(&speech_tensors)?;
        let acoustic_latents = sample_encoder_output(
            &encoder_output,
            &self.config.acoustic_tokenizer_config,
            &self.device,
            rng,
        )?;
        let acoustic_features = scale_acoustic_features(
            &acoustic_latents,
            self.speech_bias_factor,
            self.speech_scaling_factor,
            &self.device,
        )?;
        let acoustic_connected = self.acoustic_connector.forward(&acoustic_features)?;
        let prompt_positions = prepared
            .speech_input_mask
            .iter()
            .enumerate()
            .filter_map(|(index, is_speech)| is_speech.then_some(index))
            .collect::<Vec<_>>();

        let available = acoustic_connected.dim(1)?;
        for (row, position) in prompt_positions.into_iter().enumerate().take(available) {
            let embed = acoustic_connected
                .narrow(1, row, 1)?
                .squeeze(0)?
                .squeeze(0)?;
            overrides.insert(position, embed);
        }

        Ok(overrides)
    }

    fn forward_for_next_token(
        &self,
        token_ids: &[u32],
        embedding_overrides: &HashMap<usize, Tensor>,
    ) -> Result<(Tensor, Tensor), TtsError> {
        let inputs = self.build_input_embeddings(token_ids, embedding_overrides)?;
        let mut language_model = self
            .language_model
            .lock()
            .map_err(|_| TtsError::RuntimeError("VibeVoice language model mutex poisoned".to_string()))?;
        let hidden = language_model.forward(&inputs)?;
        let last_hidden = hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)?;
        let logits = language_model.next_logits(&last_hidden)?;
        Ok((last_hidden, logits))
    }

    fn forward_negative_condition(
        &self,
        negative_tokens: &[u32],
        negative_embedding_overrides: &HashMap<usize, Tensor>,
    ) -> Result<Tensor, TtsError> {
        let (hidden, _logits) = self.forward_for_next_token(negative_tokens, negative_embedding_overrides)?;
        Ok(hidden)
    }

    fn build_input_embeddings(
        &self,
        token_ids: &[u32],
        embedding_overrides: &HashMap<usize, Tensor>,
    ) -> Result<Tensor, TtsError> {
        let token_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let language_model = self
            .language_model
            .lock()
            .map_err(|_| TtsError::RuntimeError("VibeVoice language model mutex poisoned".to_string()))?;
        let base = language_model.embed(&token_tensor)?;
        drop(language_model);

        if embedding_overrides.is_empty() {
            return Ok(base);
        }

        let mut pieces = Vec::with_capacity(token_ids.len());
        for position in 0..token_ids.len() {
            if let Some(override_embed) = embedding_overrides.get(&position) {
                pieces.push(override_embed.unsqueeze(0)?.unsqueeze(0)?);
            } else {
                pieces.push(base.narrow(1, position, 1)?);
            }
        }
        let piece_refs = pieces.iter().collect::<Vec<_>>();
        Tensor::cat(&piece_refs, 1).map_err(Into::into)
    }

    fn sample_speech_token(
        &self,
        positive_condition: &Tensor,
        negative_condition: &Tensor,
        cfg_scale: f32,
        rng: &mut SimpleRng,
    ) -> Result<Tensor, TtsError> {
        let mut scheduler = self
            .noise_scheduler
            .lock()
            .map_err(|_| TtsError::RuntimeError("VibeVoice scheduler mutex poisoned".to_string()))?;
        scheduler.set_timesteps(self.config.diffusion_head_config.ddpm_num_inference_steps);

        let condition = Tensor::cat(&[positive_condition, negative_condition], 0)?;
        let latent_size = self.config.diffusion_head_config.latent_size;
        let mut speech = random_normal_tensor((2, latent_size), self.dtype, &self.device, rng)?;

        for timestep in scheduler.timesteps().to_vec() {
            let half = speech.narrow(0, 0, 1)?;
            let combined = Tensor::cat(&[&half, &half], 0)?;
            let timestep_tensor = Tensor::from_vec(
                vec![timestep as f32, timestep as f32],
                (2,),
                &self.device,
            )?
            .to_dtype(self.dtype)?;
            let eps = self.prediction_head.forward(&combined, &timestep_tensor, &condition)?;
            let cond_eps = eps.narrow(0, 0, 1)?;
            let uncond_eps = eps.narrow(0, 1, 1)?;
            let guided = uncond_eps.broadcast_add(
                &cond_eps
                    .broadcast_sub(&uncond_eps)?
                    .broadcast_mul(&Tensor::new(cfg_scale, &self.device)?)?,
            )?;
            let expanded = Tensor::cat(&[&guided, &guided], 0)?;
            speech = scheduler.step(&expanded, &speech)?;
        }

        speech.narrow(0, 0, 1)?.squeeze(0).map_err(Into::into)
    }

    fn build_diffusion_override(&self, segment_latents: &[Tensor]) -> Result<Tensor, TtsError> {
        let segment = stack_latents(segment_latents)?;
        if std::env::var("VIBEVOICE_FEEDBACK_MODE").ok().as_deref() == Some("token") {
            return self.diffusion_token_embedding();
        }

        let acoustic_last = segment.narrow(1, segment.dim(1)? - 1, 1)?;
        let acoustic_embed = self.acoustic_connector.forward(&acoustic_last)?;

        if std::env::var("VIBEVOICE_FEEDBACK_MODE").ok().as_deref() == Some("acoustic") {
            return acoustic_embed.squeeze(0)?.squeeze(0).map_err(Into::into);
        }

        let decoded_audio = self
            .acoustic_tokenizer
            .decode(&self.unscale_generated_latents(&segment)?)?;
        let semantic = self.semantic_tokenizer.encode(&decoded_audio)?.mean;
        let semantic_last = semantic
            .narrow(1, semantic.dim(1)? - 1, 1)?
            .squeeze(1)?;
        let semantic_embed = self.semantic_connector.forward(&semantic_last.unsqueeze(1)?)?;
        acoustic_embed
            .broadcast_add(&semantic_embed)?
            .squeeze(0)?
            .squeeze(0)
            .map_err(Into::into)
    }

    fn decode_segments(&self, segments: &[Vec<Tensor>]) -> Result<Vec<f32>, TtsError> {
        let mut all_samples = Vec::new();
        for segment in segments {
            if segment.is_empty() {
                continue;
            }
            let latents = stack_latents(segment)?;
            let audio = self
                .acoustic_tokenizer
                .decode(&self.unscale_generated_latents(&latents)?)?;
            let mut samples = audio
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            all_samples.append(&mut samples);
        }
        Ok(all_samples)
    }

    fn unscale_generated_latents(&self, latents: &Tensor) -> Result<Tensor, TtsError> {
        latents
            .broadcast_div(&Tensor::new(self.speech_scaling_factor, &self.device)?)?
            .broadcast_sub(&Tensor::new(self.speech_bias_factor, &self.device)?)
            .map_err(Into::into)
    }

    fn diffusion_token_embedding(&self) -> Result<Tensor, TtsError> {
        let token_id = self.processor.tokenizer_spec().speech_diffusion_id;
        let token_tensor = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
        let language_model = self
            .language_model
            .lock()
            .map_err(|_| TtsError::RuntimeError("VibeVoice language model mutex poisoned".to_string()))?;
        language_model
            .embed(&token_tensor)?
            .squeeze(0)?
            .squeeze(0)
            .map_err(Into::into)
    }
}

fn sample_encoder_output(
    output: &VibeVoiceTokenizerEncoderOutput,
    config: &VibeVoiceTokenizerConfig,
    device: &Device,
    rng: &mut SimpleRng,
) -> Result<Tensor, TtsError> {
    if config.std_dist_type != "gaussian" {
        return Ok(output.mean.clone());
    }

    let std = output.std.unwrap_or(config.fix_std) as f32;
    let noise = random_normal_tensor(output.mean.shape().clone(), DType::F32, device, rng)?
        .to_dtype(output.mean.dtype())?;
    output
        .mean
        .broadcast_add(&noise.broadcast_mul(&Tensor::new(std, device)?)?)
        .map_err(Into::into)
}

fn scale_acoustic_features(
    acoustic_latents: &Tensor,
    bias_factor: f32,
    scaling_factor: f32,
    device: &Device,
) -> Result<Tensor, TtsError> {
    acoustic_latents
        .broadcast_add(&Tensor::new(bias_factor, device)?)?
        .broadcast_mul(&Tensor::new(scaling_factor, device)?)
        .map_err(Into::into)
}

fn stack_latents(latents: &[Tensor]) -> Result<Tensor, TtsError> {
    let pieces = latents
        .iter()
        .map(|latent| latent.unsqueeze(0))
        .collect::<CandleResult<Vec<_>>>()?;
    let piece_refs = pieces.iter().collect::<Vec<_>>();
    Tensor::cat(&piece_refs, 0)?
        .unsqueeze(0)
        .map_err(Into::into)
}

fn valid_generated_tokens(spec: &VibeVoiceTokenizerSpec) -> Vec<u32> {
    let mut tokens = vec![
        spec.speech_start_id,
        spec.speech_end_id,
        spec.speech_diffusion_id,
        spec.eos_id,
    ];
    if let Some(bos_id) = spec.bos_id {
        tokens.push(bos_id);
    }
    tokens
}

fn sample_token(
    logits: &Tensor,
    valid_tokens: &[u32],
    temperature: f32,
    rng: &mut SimpleRng,
) -> Result<u32, TtsError> {
    let logits = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if temperature <= 0.0 {
        return valid_tokens
            .iter()
            .copied()
            .max_by(|left, right| {
                logits[*left as usize]
                    .partial_cmp(&logits[*right as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| TtsError::ModelError("No valid VibeVoice tokens available".to_string()));
    }

    let mut max_logit = f32::NEG_INFINITY;
    for token in valid_tokens {
        max_logit = max_logit.max(logits[*token as usize] / temperature);
    }

    let mut cumulative = 0.0f32;
    let mut weights = Vec::with_capacity(valid_tokens.len());
    for token in valid_tokens {
        let weight = ((logits[*token as usize] / temperature) - max_logit).exp();
        cumulative += weight;
        weights.push(weight);
    }

    let threshold = rng.next_f32() * cumulative.max(f32::EPSILON);
    let mut running = 0.0f32;
    for (index, token) in valid_tokens.iter().enumerate() {
        running += weights[index];
        if running >= threshold {
            return Ok(*token);
        }
    }

    valid_tokens
        .last()
        .copied()
        .ok_or_else(|| TtsError::ModelError("No valid VibeVoice tokens available".to_string()))
}

fn scalar_from_tensor(tensor: &Tensor) -> Result<f32, TtsError> {
    let values = tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| TtsError::ModelError("Expected scalar tensor".to_string()))
}

fn random_normal_tensor<S: Into<candle_core::Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
    rng: &mut SimpleRng,
) -> Result<Tensor, TtsError> {
    let shape = shape.into();
    let elem_count = shape.elem_count();
    let mut values = Vec::with_capacity(elem_count);
    while values.len() < elem_count {
        let u1 = rng.next_f32().clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
        let u2 = rng.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        values.push(radius * theta.cos());
        if values.len() < elem_count {
            values.push(radius * theta.sin());
        }
    }
    Tensor::from_vec(values, shape, device)
        .and_then(|tensor| tensor.to_dtype(dtype))
        .map_err(Into::into)
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}