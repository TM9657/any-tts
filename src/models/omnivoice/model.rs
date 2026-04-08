use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module};
use tracing::info;

use crate::audio::AudioSamples;
use crate::config::{ModelFiles, TtsConfig};
use crate::error::TtsError;
use crate::layers::attention::GqaConfig;
use crate::layers::transformer::TransformerBlock;
use crate::tensor_utils::{precompute_rope_freqs, RmsNorm};
use crate::tokenizer::TextTokenizer;
use crate::traits::{ModelInfo, SynthesisRequest, TtsModel};

use super::audio_tokenizer::OmniVoiceAudioTokenizerDecoder;
use super::config::{OmniVoiceAudioTokenizerConfig, OmniVoiceConfig};

const DEFAULT_NUM_STEPS: usize = 32;
const DEFAULT_GUIDANCE_SCALE: f64 = 2.0;
const DEFAULT_T_SHIFT: f64 = 0.1;
const DEFAULT_LAYER_PENALTY: f32 = 5.0;
const DEFAULT_POSITION_TEMPERATURE: f32 = 5.0;
const DEFAULT_CLASS_TEMPERATURE: f32 = 0.0;

struct OmniVoiceGenerationConfig {
    num_step: usize,
    guidance_scale: f64,
    t_shift: f64,
    layer_penalty_factor: f32,
    position_temperature: f32,
    class_temperature: f32,
    denoise: bool,
}

impl OmniVoiceGenerationConfig {
    fn from_request(request: &SynthesisRequest) -> Self {
        Self {
            num_step: DEFAULT_NUM_STEPS,
            guidance_scale: request.cfg_scale.unwrap_or(DEFAULT_GUIDANCE_SCALE),
            t_shift: DEFAULT_T_SHIFT,
            layer_penalty_factor: DEFAULT_LAYER_PENALTY,
            position_temperature: env_override_f32(
                "OMNIVOICE_POSITION_TEMPERATURE",
                DEFAULT_POSITION_TEMPERATURE,
            ),
            class_temperature: request
                .temperature
                .unwrap_or(DEFAULT_CLASS_TEMPERATURE as f64) as f32,
            denoise: true,
        }
    }
}

struct InferenceInputs {
    input_ids: Vec<u32>,
    audio_mask: Vec<u8>,
    seq_len: usize,
    target_len: usize,
}

struct OmniVoiceBackbone {
    config: OmniVoiceConfig,
    text_embeddings: Embedding,
    audio_embeddings: Embedding,
    audio_head: Linear,
    norm: RmsNorm,
    layers: Vec<TransformerBlock>,
    rope_cos: Tensor,
    rope_sin: Tensor,
    codebook_layer_offsets: Tensor,
}

impl OmniVoiceBackbone {
    fn load(
        config: &OmniVoiceConfig,
        vb: candle_nn::VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self, TtsError> {
        let llm = &config.llm_config;
        let gqa_config = GqaConfig::with_head_dim(
            llm.hidden_size,
            llm.num_attention_heads,
            llm.num_key_value_heads,
            llm.head_dim(),
            llm.max_position_embeddings,
            llm.rope_theta(),
            llm.rms_norm_eps,
        );

        let text_embeddings =
            candle_nn::embedding(llm.vocab_size, llm.hidden_size, vb.pp("llm.embed_tokens"))?;
        let audio_embeddings = candle_nn::embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            llm.hidden_size,
            vb.pp("audio_embeddings"),
        )?;
        let audio_head = candle_nn::linear_no_bias(
            llm.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            vb.pp("audio_heads"),
        )?;
        let norm = RmsNorm::load(llm.hidden_size, llm.rms_norm_eps, vb.pp("llm.norm"))?;

        let mut layers = Vec::with_capacity(llm.num_hidden_layers);
        for layer_index in 0..llm.num_hidden_layers {
            layers.push(TransformerBlock::load(
                &gqa_config,
                llm.intermediate_size,
                vb.pp(format!("llm.layers.{}", layer_index)),
            )?);
        }

        let (rope_cos, rope_sin) = precompute_rope_freqs(
            gqa_config.head_dim,
            llm.max_position_embeddings,
            llm.rope_theta(),
            device,
            dtype,
        )?;
        let codebook_layer_offsets = vb
            .get(config.num_audio_codebook, "codebook_layer_offsets")?
            .to_dtype(DType::U32)?
            .reshape((1, config.num_audio_codebook, 1))?;

        Ok(Self {
            config: config.clone(),
            text_embeddings,
            audio_embeddings,
            audio_head,
            norm,
            layers,
            rope_cos,
            rope_sin,
            codebook_layer_offsets,
        })
    }

    fn forward(
        &mut self,
        input_ids: &Tensor,
        audio_mask_u32: &Tensor,
        audio_mask_f32: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, TtsError> {
        for layer in &mut self.layers {
            layer.clear_cache();
        }

        let text_ids = input_ids.i((.., 0, ..))?;
        let text_embeds = self.text_embeddings.forward(&text_ids)?;

        let shifted_audio_ids = input_ids
            .broadcast_mul(&audio_mask_u32.unsqueeze(1)?)?
            .broadcast_add(&self.codebook_layer_offsets)?;
        let audio_embeds = self.audio_embeddings.forward(&shifted_audio_ids)?.sum(1)?;

        let audio_mask = audio_mask_f32.to_dtype(text_embeds.dtype())?.unsqueeze(2)?;
        let text_mask = audio_mask.neg()?.add(&Tensor::ones_like(&audio_mask)?)?;
        let mut hidden = text_embeds
            .broadcast_mul(&text_mask)?
            .add(&audio_embeds.broadcast_mul(&audio_mask)?)?;

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rope_cos, &self.rope_sin, 0, attention_mask)?;
        }
        hidden = self.norm.forward(&hidden)?;

        let (batch, seq_len, _) = hidden.dims3()?;
        let logits_flat = self.audio_head.forward(&hidden)?;
        logits_flat
            .reshape((
                batch,
                seq_len,
                self.config.num_audio_codebook,
                self.config.audio_vocab_size,
            ))?
            .transpose(1, 2)
            .map_err(Into::into)
    }
}

pub struct OmniVoiceModel {
    config: OmniVoiceConfig,
    files: ModelFiles,
    frame_rate: usize,
    compute_device: Device,
    compute_dtype: DType,
    audio_device: Device,
    tokenizer: TextTokenizer,
    backbone: Mutex<OmniVoiceBackbone>,
    causal_mask_cache: Mutex<HashMap<usize, Tensor>>,
    audio_tokenizer: OmniVoiceAudioTokenizerDecoder,
    mask_token_penalty: Tensor,
}

impl TtsModel for OmniVoiceModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError> {
        let compute_device = config.device.resolve()?;
        let dtype_override = env_override_dtype("OMNIVOICE_DTYPE");
        let mut compute_dtype = dtype_override.unwrap_or_else(|| config.dtype.to_candle());
        if matches!(compute_device, Device::Cpu) && compute_dtype == DType::BF16 {
            info!("BF16 is not supported on CPU; falling back to F32 for OmniVoice");
            compute_dtype = DType::F32;
        } else if matches!(compute_device, Device::Metal(_)) {
            if dtype_override.is_none() && compute_dtype != DType::F32 {
                info!(
                    "OmniVoice Metal F16/BF16 generation is numerically unstable; preferring F32"
                );
                compute_dtype = DType::F32;
            } else if compute_dtype == DType::BF16 {
                info!("BF16 is not supported on Metal for OmniVoice; preferring F32");
                compute_dtype = DType::F32;
            }
        }

        let files = config.resolve_files()?;
        let config_bytes = files
            .config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let model_config = OmniVoiceConfig::from_bytes(config_bytes.as_ref())?;
        let tokenizer = TextTokenizer::from_asset(
            files
                .tokenizer
                .as_ref()
                .expect("validated by resolve_files"),
        )?;
        let audio_tokenizer_config_bytes = files
            .speech_tokenizer_config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let audio_tokenizer_config =
            OmniVoiceAudioTokenizerConfig::from_bytes(audio_tokenizer_config_bytes.as_ref())?;

        let main_vb =
            ModelFiles::load_safetensors_vb(&files.weights, compute_dtype, &compute_device)?;
        let backbone =
            OmniVoiceBackbone::load(&model_config, main_vb, &compute_device, compute_dtype)?;

        let audio_device = if matches!(compute_device, Device::Metal(_)) {
            Device::Cpu
        } else {
            compute_device.clone()
        };
        let audio_dtype = if matches!(audio_device, Device::Cpu) {
            DType::F32
        } else {
            compute_dtype
        };
        let audio_vb = ModelFiles::load_safetensors_vb(
            &files.speech_tokenizer_weights,
            audio_dtype,
            &audio_device,
        )?;
        let audio_tokenizer = OmniVoiceAudioTokenizerDecoder::load(
            &audio_tokenizer_config,
            model_config.num_audio_codebook,
            audio_vb,
            &audio_device,
        )?;

        let mut mask_penalty = vec![0.0f32; model_config.audio_vocab_size];
        if (model_config.audio_mask_id as usize) < mask_penalty.len() {
            mask_penalty[model_config.audio_mask_id as usize] = f32::NEG_INFINITY;
        }
        let mask_token_penalty = Tensor::new(mask_penalty.as_slice(), &compute_device)?
            .reshape((1, 1, 1, model_config.audio_vocab_size))?;

        info!(
            "Loading OmniVoice on {:?} (audio decoder on {:?})",
            compute_device, audio_device
        );

        Ok(Self {
            config: model_config,
            files,
            frame_rate: audio_tokenizer_config.frame_rate(),
            compute_device,
            compute_dtype,
            audio_device,
            tokenizer,
            backbone: Mutex::new(backbone),
            causal_mask_cache: Mutex::new(HashMap::new()),
            audio_tokenizer,
            mask_token_penalty,
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        if request.voice.is_some() {
            return Err(TtsError::ModelError(
                "OmniVoice does not use named voices. Use `instruct` for voice design instead."
                    .to_string(),
            ));
        }
        if request.reference_audio.is_some() {
            return Err(TtsError::ModelError(
                "Native OmniVoice voice cloning is not implemented yet. Reference-audio synthesis no longer falls back to Python.".to_string(),
            ));
        }
        if request.voice_embedding.is_some() {
            return Err(TtsError::ModelError(
                "Native OmniVoice does not support reusable voice embeddings yet.".to_string(),
            ));
        }

        let gen_config = OmniVoiceGenerationConfig::from_request(request);
        let normalized_language = self.normalize_language(request.language.as_deref());
        let normalized_instruct = normalize_instruct(request.instruct.as_deref(), &request.text);
        let target_len = request
            .max_tokens
            .unwrap_or_else(|| estimate_target_tokens(&request.text, self.frame_rate));
        let inputs = self.prepare_inference_inputs(
            &request.text,
            target_len,
            normalized_language.as_deref(),
            normalized_instruct.as_deref(),
            gen_config.denoise,
        )?;

        let cond_len = inputs.seq_len;
        let target_len = inputs.target_len;
        let mask_id = self.config.audio_mask_id;
        let total_masked = self.config.num_audio_codebook * target_len;
        let schedule = build_schedule(total_masked, gen_config.num_step, gen_config.t_shift);
        let use_causal_mask = std::env::var_os("OMNIVOICE_CAUSAL_MASK").is_some();
        let cond_attention_mask = if use_causal_mask {
            Some(self.causal_attention_mask(cond_len)?)
        } else {
            None
        };
        let uncond_attention_mask = if use_causal_mask {
            Some(self.causal_attention_mask(target_len)?)
        } else {
            None
        };

        let mut cond_input_ids = inputs.input_ids.clone();
        let mut uncond_input_ids = vec![mask_id; self.config.num_audio_codebook * target_len];

        let cond_audio_mask_u32_values: Vec<u32> =
            inputs.audio_mask.iter().copied().map(u32::from).collect();
        let cond_audio_mask_f32_values: Vec<f32> =
            inputs.audio_mask.iter().copied().map(f32::from).collect();
        let cond_audio_mask_u32 =
            Tensor::new(cond_audio_mask_u32_values.as_slice(), &self.compute_device)?
                .reshape((1, cond_len))?;
        let cond_audio_mask_f32 =
            Tensor::new(cond_audio_mask_f32_values.as_slice(), &self.compute_device)?
                .reshape((1, cond_len))?;
        let uncond_audio_mask_u32 =
            Tensor::ones((1, target_len), DType::U32, &self.compute_device)?;
        let uncond_audio_mask_f32 =
            Tensor::ones((1, target_len), DType::F32, &self.compute_device)?;
        let mut tokens = vec![mask_id; self.config.num_audio_codebook * target_len];
        let mut rng = SimpleRng::new(omnivoice_seed());
        let layer_penalties: Vec<f32> = (0..self.config.num_audio_codebook)
            .map(|layer| layer as f32 * gen_config.layer_penalty_factor)
            .collect();
        let mut step_trace = Vec::new();

        for (step_index, step_k) in schedule.iter().copied().enumerate() {
            if step_k == 0 {
                continue;
            }

            let cond_input_ids_tensor = Tensor::new(
                cond_input_ids.as_slice(),
                &self.compute_device,
            )?
            .reshape((1, self.config.num_audio_codebook, cond_len))?;
            let cond_logits = self
                .backbone
                .lock()
                .map_err(|_| {
                    TtsError::RuntimeError("OmniVoice model mutex was poisoned".to_string())
                })?
                .forward(
                    &cond_input_ids_tensor,
                    &cond_audio_mask_u32,
                    &cond_audio_mask_f32,
                    cond_attention_mask.as_ref(),
                )?
                .to_dtype(DType::F32)?
                .i((0..1, .., cond_len - target_len..cond_len, ..))?;
            let uncond_input_ids_tensor = Tensor::new(
                uncond_input_ids.as_slice(),
                &self.compute_device,
            )?
            .reshape((1, self.config.num_audio_codebook, target_len))?;
            let u_logits = self
                .backbone
                .lock()
                .map_err(|_| {
                    TtsError::RuntimeError("OmniVoice model mutex was poisoned".to_string())
                })?
                .forward(
                    &uncond_input_ids_tensor,
                    &uncond_audio_mask_u32,
                    &uncond_audio_mask_f32,
                    uncond_attention_mask.as_ref(),
                )?
                .to_dtype(DType::F32)?;
            let (pred_tokens, scores) =
                self.predict_tokens_with_scoring(&cond_logits, &u_logits, &gen_config, &mut rng)?;
            maybe_dump_debug_first_step(&cond_logits, &u_logits, &pred_tokens, &scores)?;

            let pred_tokens_flat = pred_tokens.flatten_all()?.to_vec1::<u32>()?;
            let mut scores_flat = scores.flatten_all()?.to_vec1::<f32>()?;
            for (layer_scores, penalty) in scores_flat
                .chunks_exact_mut(target_len)
                .zip(layer_penalties.iter().copied())
            {
                for score in layer_scores {
                    *score -= penalty;
                }
            }

            let chosen_positions = select_positions(
                &scores_flat,
                &tokens,
                mask_id,
                step_k,
                gen_config.position_temperature,
                &mut rng,
            );
            maybe_record_step_trace(
                &mut step_trace,
                step_index,
                step_k,
                &chosen_positions,
                &pred_tokens_flat,
                target_len,
            );
            for flat_index in chosen_positions {
                tokens[flat_index] = pred_tokens_flat[flat_index];
                let layer = flat_index / target_len;
                let pos = flat_index % target_len;
                cond_input_ids[index_2d(layer, cond_len - target_len + pos, cond_len)] =
                    pred_tokens_flat[flat_index];
                uncond_input_ids[index_2d(layer, pos, target_len)] = pred_tokens_flat[flat_index];
            }
        }

        maybe_dump_step_trace(&step_trace)?;
        maybe_dump_debug_tokens(&tokens, self.config.num_audio_codebook, target_len)?;

        let token_tensor = Tensor::new(tokens.as_slice(), &self.audio_device)?.reshape((
            1,
            self.config.num_audio_codebook,
            target_len,
        ))?;
        let waveform = self.audio_tokenizer.decode(&token_tensor)?;
        let waveform = waveform.squeeze(1)?.squeeze(0)?.to_device(&Device::Cpu)?;
        let mut samples = waveform.to_vec1::<f32>()?;
        post_process_audio(&mut samples);

        Ok(AudioSamples::new(samples, self.sample_rate()))
    }

    fn sample_rate(&self) -> u32 {
        self.audio_tokenizer.sample_rate()
    }

    fn supported_languages(&self) -> Vec<String> {
        vec![
            "auto".to_string(),
            "en".to_string(),
            "zh".to_string(),
            "ja".to_string(),
            "ko".to_string(),
            "de".to_string(),
            "fr".to_string(),
            "es".to_string(),
            "pt".to_string(),
            "ru".to_string(),
            "it".to_string(),
        ]
    }

    fn supported_voices(&self) -> Vec<String> {
        Vec::new()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "OmniVoice".to_string(),
            variant: format!(
                "native ({})",
                self.files
                    .config
                    .as_ref()
                    .map(|path| path.display_name())
                    .unwrap_or_else(|| "unresolved".to_string())
            ),
            parameters: 600_000_000,
            sample_rate: self.sample_rate(),
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl OmniVoiceModel {
    fn causal_attention_mask(&self, seq_len: usize) -> Result<Tensor, TtsError> {
        if let Some(mask) = self
            .causal_mask_cache
            .lock()
            .map_err(|_| {
                TtsError::RuntimeError("OmniVoice mask cache mutex was poisoned".to_string())
            })?
            .get(&seq_len)
        {
            return Ok(mask.clone());
        }

        let mask = build_causal_attention_mask(seq_len, &self.compute_device, self.compute_dtype)?;
        self.causal_mask_cache
            .lock()
            .map_err(|_| {
                TtsError::RuntimeError("OmniVoice mask cache mutex was poisoned".to_string())
            })?
            .insert(seq_len, mask.clone());
        Ok(mask)
    }

    fn normalize_language(&self, language: Option<&str>) -> Option<String> {
        let language = language?.trim();
        if language.is_empty() || matches!(language.to_ascii_lowercase().as_str(), "auto" | "none")
        {
            return None;
        }

        let lower = language.to_ascii_lowercase();
        let mapped = match lower.as_str() {
            "english" => "en",
            "chinese" => "zh",
            "japanese" => "ja",
            "korean" => "ko",
            "german" => "de",
            "french" => "fr",
            "spanish" => "es",
            "portuguese" => "pt",
            "russian" => "ru",
            "italian" => "it",
            _ => lower.as_str(),
        };
        Some(mapped.to_string())
    }

    fn prepare_inference_inputs(
        &self,
        text: &str,
        target_len: usize,
        language: Option<&str>,
        instruct: Option<&str>,
        denoise: bool,
    ) -> Result<InferenceInputs, TtsError> {
        let mut style_text = String::new();
        if denoise {
            style_text.push_str("<|denoise|>");
        }
        style_text.push_str("<|lang_start|>");
        style_text.push_str(language.unwrap_or("None"));
        style_text.push_str("<|lang_end|>");
        style_text.push_str("<|instruct_start|>");
        style_text.push_str(instruct.unwrap_or("None"));
        style_text.push_str("<|instruct_end|>");

        let text_prompt = format!("<|text_start|>{}<|text_end|>", combine_text(text));
        let style_ids = self.tokenizer.encode(&style_text)?;
        let text_ids = self.tokenizer.encode(&text_prompt)?;
        let style_len = style_ids.len();
        let text_len = text_ids.len();
        let seq_len = style_len + text_len + target_len;

        let mut input_ids =
            vec![self.config.audio_mask_id; self.config.num_audio_codebook * seq_len];
        for layer in 0..self.config.num_audio_codebook {
            for (index, token_id) in style_ids.iter().copied().enumerate() {
                input_ids[index_2d(layer, index, seq_len)] = token_id;
            }
            for (index, token_id) in text_ids.iter().copied().enumerate() {
                input_ids[index_2d(layer, style_len + index, seq_len)] = token_id;
            }
        }

        let mut audio_mask = vec![0u8; seq_len];
        for audio_flag in audio_mask.iter_mut().skip(seq_len - target_len) {
            *audio_flag = 1;
        }

        Ok(InferenceInputs {
            input_ids,
            audio_mask,
            seq_len,
            target_len,
        })
    }

    fn predict_tokens_with_scoring(
        &self,
        c_logits: &Tensor,
        u_logits: &Tensor,
        gen_config: &OmniVoiceGenerationConfig,
        rng: &mut SimpleRng,
    ) -> Result<(Tensor, Tensor), TtsError> {
        let mut log_probs = if gen_config.guidance_scale != 0.0 {
            let c_log_probs = log_softmax_last_dim(c_logits)?;
            let u_log_probs = log_softmax_last_dim(u_logits)?;
            let guidance = c_log_probs.broadcast_sub(&u_log_probs)?;
            let guided = c_log_probs.broadcast_add(&(guidance * gen_config.guidance_scale)?)?;
            log_softmax_last_dim(&guided)?
        } else {
            log_softmax_last_dim(c_logits)?
        };

        log_probs = log_probs.broadcast_add(&self.mask_token_penalty)?;

        if gen_config.class_temperature > 0.0 {
            let flat = log_probs.flatten_all()?.to_vec1::<f32>()?;
            let (_batch, codebooks, seq_len, vocab_size) = log_probs.dims4()?;
            let top_k = ((vocab_size as f32 * 0.1).ceil() as usize).max(1);
            let mut pred_tokens = vec![0u32; codebooks * seq_len];
            let mut scores = vec![f32::NEG_INFINITY; codebooks * seq_len];

            for layer in 0..codebooks {
                for pos in 0..seq_len {
                    let start = ((layer * seq_len) + pos) * vocab_size;
                    let mut candidates: Vec<(f32, usize)> = flat[start..start + vocab_size]
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(token, score)| (score, token))
                        .collect();
                    candidates.sort_by(|left, right| right.0.total_cmp(&left.0));
                    candidates.truncate(top_k);
                    let mut best = (f32::NEG_INFINITY, 0usize);
                    for (score, token) in candidates.iter().copied() {
                        let noisy = score / gen_config.class_temperature + rng.next_gumbel();
                        if noisy > best.0 {
                            best = (noisy, token);
                        }
                    }
                    pred_tokens[index_2d(layer, pos, seq_len)] = best.1 as u32;
                    scores[index_2d(layer, pos, seq_len)] = candidates[0].0;
                }
            }

            let pred_tokens = Tensor::new(pred_tokens.as_slice(), &self.compute_device)?
                .reshape((1, codebooks, seq_len))?;
            let scores = Tensor::new(scores.as_slice(), &self.compute_device)?
                .reshape((1, codebooks, seq_len))?;
            Ok((pred_tokens, scores))
        } else {
            let pred_tokens = log_probs.argmax(candle_core::D::Minus1)?;
            let scores = log_probs.max(candle_core::D::Minus1)?;
            Ok((pred_tokens, scores))
        }
    }
}

fn build_schedule(total_masked: usize, num_steps: usize, t_shift: f64) -> Vec<usize> {
    let timesteps = get_time_steps(num_steps + 1, t_shift);
    let mut remaining = total_masked;
    let mut schedule = Vec::with_capacity(num_steps);
    for step in 0..num_steps {
        let count = if step == num_steps - 1 {
            remaining
        } else {
            let delta = (timesteps[step + 1] - timesteps[step]).max(0.0);
            let count = ((total_masked as f64) * delta).ceil() as usize;
            count.min(remaining)
        };
        schedule.push(count);
        remaining = remaining.saturating_sub(count);
    }
    schedule
}

fn get_time_steps(num_step: usize, t_shift: f64) -> Vec<f64> {
    let mut timesteps = Vec::with_capacity(num_step + 1);
    for index in 0..=num_step {
        let t = index as f64 / num_step.max(1) as f64;
        timesteps.push(t_shift * t / (1.0 + (t_shift - 1.0) * t));
    }
    timesteps
}

fn build_causal_attention_mask(
    seq_len: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor, TtsError> {
    let mut values = vec![0.0f32; seq_len * seq_len];
    for query_index in 0..seq_len {
        for key_index in query_index + 1..seq_len {
            values[index_2d(query_index, key_index, seq_len)] = -1e4;
        }
    }
    Ok(Tensor::new(values.as_slice(), device)?
        .reshape((1, 1, seq_len, seq_len))?
        .to_dtype(dtype)?)
}

fn select_positions(
    scores: &[f32],
    current_tokens: &[u32],
    mask_id: u32,
    count: usize,
    temperature: f32,
    rng: &mut SimpleRng,
) -> Vec<usize> {
    let mut candidates = Vec::new();
    for (index, score) in scores.iter().copied().enumerate() {
        if !score.is_finite() {
            continue;
        }
        let value = if temperature > 0.0 {
            score / temperature + rng.next_gumbel()
        } else {
            score
        };
        if current_tokens[index] != mask_id {
            continue;
        }
        candidates.push((value, index));
    }
    candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    candidates
        .into_iter()
        .take(count)
        .map(|(_, index)| index)
        .collect()
}

fn omnivoice_seed() -> u64 {
    std::env::var("OMNIVOICE_SEED")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        })
}

fn env_override_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_override_dtype(name: &str) -> Option<DType> {
    let value = std::env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Some(DType::F32),
        "f16" => Some(DType::F16),
        "bf16" => Some(DType::BF16),
        _ => None,
    }
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
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

    fn next_gumbel(&mut self) -> f32 {
        let unit = self.next_f32().clamp(0.0, 1.0 - 1e-7);
        let inner = -(unit + 1e-10).ln() + 1e-10;
        -inner.ln()
    }
}

fn log_softmax_last_dim(logits: &Tensor) -> Result<Tensor, TtsError> {
    Ok(candle_nn::ops::log_softmax(
        &logits.contiguous()?,
        candle_core::D::Minus1,
    )?)
}

fn maybe_dump_debug_tokens(
    tokens: &[u32],
    num_audio_codebook: usize,
    target_len: usize,
) -> Result<(), TtsError> {
    let Some(path) = std::env::var_os("OMNIVOICE_DEBUG_TOKEN_DUMP") else {
        return Ok(());
    };

    let layers: Vec<Vec<u32>> = tokens
        .chunks(target_len)
        .take(num_audio_codebook)
        .map(|layer| layer.to_vec())
        .collect();

    let payload = serde_json::json!({
        "num_audio_codebook": num_audio_codebook,
        "target_len": target_len,
        "layers": layers,
    });

    std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
    Ok(())
}

fn maybe_dump_debug_first_step(
    cond_logits: &Tensor,
    uncond_logits: &Tensor,
    pred_tokens: &Tensor,
    scores: &Tensor,
) -> Result<(), TtsError> {
    let Some(path) = std::env::var_os("OMNIVOICE_DEBUG_FIRST_STEP_DUMP") else {
        return Ok(());
    };

    let payload = build_first_step_debug_payload(cond_logits, uncond_logits, pred_tokens, scores)?;
    std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
    std::env::remove_var("OMNIVOICE_DEBUG_FIRST_STEP_DUMP");
    Ok(())
}

fn maybe_record_step_trace(
    step_trace: &mut Vec<serde_json::Value>,
    step_index: usize,
    step_k: usize,
    chosen_positions: &[usize],
    pred_tokens_flat: &[u32],
    target_len: usize,
) {
    if std::env::var_os("OMNIVOICE_DEBUG_STEP_TRACE").is_none() {
        return;
    }

    let chosen: Vec<serde_json::Value> = chosen_positions
        .iter()
        .copied()
        .map(|flat_index| {
            serde_json::json!({
                "flat_index": flat_index,
                "layer": flat_index / target_len,
                "position": flat_index % target_len,
                "token": pred_tokens_flat[flat_index],
            })
        })
        .collect();
    step_trace.push(serde_json::json!({
        "step_index": step_index,
        "count": step_k,
        "chosen": chosen,
    }));
}

fn maybe_dump_step_trace(step_trace: &[serde_json::Value]) -> Result<(), TtsError> {
    let Some(path) = std::env::var_os("OMNIVOICE_DEBUG_STEP_TRACE") else {
        return Ok(());
    };

    let payload = serde_json::json!({
        "steps": step_trace,
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)?;
    Ok(())
}

fn build_first_step_debug_payload(
    cond_logits: &Tensor,
    uncond_logits: &Tensor,
    pred_tokens: &Tensor,
    scores: &Tensor,
) -> Result<serde_json::Value, TtsError> {
    let (_batch, codebooks, seq_len, vocab_size) = cond_logits.dims4()?;
    let cond_flat = cond_logits.flatten_all()?.to_vec1::<f32>()?;
    let uncond_flat = uncond_logits.flatten_all()?.to_vec1::<f32>()?;
    let pred_tokens_flat = pred_tokens.flatten_all()?.to_vec1::<u32>()?;
    let scores_flat = scores.flatten_all()?.to_vec1::<f32>()?;
    let position_count = seq_len.min(4);

    let mut layers = Vec::with_capacity(codebooks);
    for layer in 0..codebooks {
        let mut positions = Vec::with_capacity(position_count);
        for pos in 0..position_count {
            let start = ((layer * seq_len) + pos) * vocab_size;
            let end = start + vocab_size;
            positions.push(serde_json::json!({
                "position": pos,
                "cond_top": top_k_summary(&cond_flat[start..end], 10),
                "uncond_top": top_k_summary(&uncond_flat[start..end], 10),
                "pred_token": pred_tokens_flat[index_2d(layer, pos, seq_len)],
                "score": scores_flat[index_2d(layer, pos, seq_len)],
            }));
        }
        layers.push(serde_json::json!({
            "layer": layer,
            "positions": positions,
        }));
    }

    Ok(serde_json::json!({
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "layers": layers,
    }))
}

fn top_k_summary(values: &[f32], count: usize) -> Vec<serde_json::Value> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1));
    indexed
        .into_iter()
        .take(count)
        .map(|(token, logit)| {
            serde_json::json!({
                "token": token,
                "logit": logit,
            })
        })
        .collect()
}

fn estimate_target_tokens(text: &str, frame_rate: usize) -> usize {
    let ref_text = "Nice to meet you.";
    let ref_tokens = frame_rate.max(1) as f32;
    let ref_weight = text_weight(ref_text).max(1.0);
    let target_weight = text_weight(text).max(1.0);
    let mut estimate = target_weight / (ref_weight / ref_tokens);
    if estimate < 50.0 {
        estimate = 50.0 * (estimate / 50.0).powf(1.0 / 3.0);
    }
    estimate.max(1.0).round() as usize
}

fn text_weight(text: &str) -> f32 {
    text.chars().map(char_weight).sum()
}

fn char_weight(ch: char) -> f32 {
    if ch.is_ascii_whitespace() {
        return 0.2;
    }
    if ch.is_ascii_punctuation() || (ch.is_ascii_graphic() && !ch.is_alphanumeric()) {
        return 0.5;
    }
    if ch.is_ascii_digit() || ch.is_numeric() {
        return 3.5;
    }
    let code = ch as u32;
    match code {
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF => 0.0,
        0x3040..=0x30FF => 2.2,
        0x3400..=0x9FFF | 0xF900..=0xFAFF => 3.0,
        0xAC00..=0xD7AF | 0x1100..=0x11FF | 0x3130..=0x318F => 2.5,
        0x0590..=0x08FF => 1.5,
        0x0900..=0x0DFF | 0x1900..=0x1CFF | 0xA800..=0xABFF => 1.8,
        0x0E00..=0x0EFF => 1.5,
        _ => 1.0,
    }
}

fn combine_text(text: &str) -> String {
    let chars: Vec<char> = text.trim().chars().collect();
    let mut normalized = String::with_capacity(chars.len());
    for (index, ch) in chars.iter().copied().enumerate() {
        if ch == '\n' || ch == '\r' {
            if !normalized.ends_with('.') {
                normalized.push('.');
            }
            continue;
        }
        if ch.is_whitespace() {
            let prev_is_cjk = index > 0 && is_cjk(chars[index - 1]);
            let next_is_cjk = chars.get(index + 1).copied().is_some_and(is_cjk);
            if prev_is_cjk || next_is_cjk {
                continue;
            }
            if !normalized.ends_with(' ') {
                normalized.push(' ');
            }
            continue;
        }
        normalized.push(ch);
    }
    normalized.trim().to_string()
}

fn normalize_instruct(instruct: Option<&str>, text: &str) -> Option<String> {
    let instruct = instruct?.trim();
    if instruct.is_empty() {
        return None;
    }
    let prefer_zh = text.chars().any(is_cjk) || instruct.chars().any(is_cjk);
    let items: Vec<String> = instruct
        .split([',', '，'])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| {
            if item.chars().any(is_cjk) {
                item.to_string()
            } else {
                item.to_ascii_lowercase()
            }
        })
        .collect();
    if items.is_empty() {
        None
    } else if prefer_zh {
        Some(items.join("，"))
    } else {
        Some(items.join(", "))
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF)
}

fn post_process_audio(samples: &mut Vec<f32>) {
    let peak = samples.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if peak > 1e-6 {
        let scale = 0.5 / peak;
        for sample in samples.iter_mut() {
            *sample = (*sample * scale).clamp(-1.0, 1.0);
        }
    }

    let fade = samples.len().min(256);
    for index in 0..fade {
        let weight = index as f32 / fade.max(1) as f32;
        samples[index] *= weight;
        let tail = samples.len() - 1 - index;
        samples[tail] *= weight;
    }

    let pad = 256usize;
    let mut padded = Vec::with_capacity(samples.len() + pad * 2);
    padded.extend(std::iter::repeat_n(0.0, pad));
    padded.extend(samples.iter().copied());
    padded.extend(std::iter::repeat_n(0.0, pad));
    *samples = padded;
}

fn index_2d(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}
