use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module};
use tracing::info;

use crate::audio::AudioSamples;
use crate::config::{ModelAssetDir, ModelFiles, TtsConfig};
use crate::error::TtsError;
use crate::models::vibevoice::config::VibeVoicePreprocessorConfig;
use crate::models::vibevoice::diffusion::{DpmSolverMultistepScheduler, VibeVoiceDiffusionHead};
use crate::models::vibevoice::generation::{
    generation_seed, random_normal_tensor, stack_latents, LayerKvCache, SimpleRng,
};
use crate::models::vibevoice::processor::VibeVoiceTokenizerSpec;
use crate::models::vibevoice::runtime::SpeechConnector;
use crate::models::vibevoice::speech_tokenizer::VibeVoiceAcousticTokenizer;
use crate::tokenizer::TextTokenizer;
use crate::traits::{ModelInfo, SynthesisRequest, TtsModel};

use super::config::VibeVoiceRealtimeConfig;
use super::preset::VoicePreset;
use super::processor::VibeVoiceRealtimeProcessor;
use super::runtime::{BinaryClassifier, RealtimeDecoderState, RealtimeLanguageModel};

const DEFAULT_CFG_SCALE: f32 = 1.5;
const TTS_TEXT_WINDOW_SIZE: usize = 5;
const TTS_SPEECH_WINDOW_SIZE: usize = 6;

pub struct VibeVoiceRealtimeModel {
    config: VibeVoiceRealtimeConfig,
    preprocessor_config: VibeVoicePreprocessorConfig,
    device: Device,
    dtype: DType,
    files: ModelFiles,
    processor: VibeVoiceRealtimeProcessor,
    language_model: Mutex<RealtimeLanguageModel>,
    tts_language_model: Mutex<RealtimeLanguageModel>,
    tts_input_types: Embedding,
    acoustic_tokenizer: VibeVoiceAcousticTokenizer,
    acoustic_connector: SpeechConnector,
    prediction_head: VibeVoiceDiffusionHead,
    noise_scheduler: Mutex<DpmSolverMultistepScheduler>,
    tts_eos_classifier: BinaryClassifier,
    speech_scaling_factor: f32,
    speech_bias_factor: f32,
    voice_names: Vec<String>,
    voice_cache: Mutex<HashMap<String, VoicePreset>>,
}

impl TtsModel for VibeVoiceRealtimeModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError> {
        let device = config.device.resolve()?;
        let dtype = resolve_runtime_dtype(&device, config.dtype.to_candle());
        let files = config.resolve_files()?;
        let config_bytes = files
            .config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let model_config = VibeVoiceRealtimeConfig::from_bytes(config_bytes.as_ref())?;
        let preprocessor_config = if let Some(asset) = &files.preprocessor_config {
            VibeVoicePreprocessorConfig::from_bytes(asset.read_bytes()?.as_ref())?
        } else {
            VibeVoicePreprocessorConfig::default()
        };
        let tokenizer = TextTokenizer::from_asset(
            files
                .tokenizer
                .as_ref()
                .expect("validated by resolve_files"),
        )?;
        let tokenizer_spec = VibeVoiceTokenizerSpec::from_tokenizer(&tokenizer)?;
        let processor =
            VibeVoiceRealtimeProcessor::new(tokenizer, tokenizer_spec, preprocessor_config.clone());
        let vb = ModelFiles::load_safetensors_vb(&files.weights, dtype, &device)?;
        let model_vb = vb.pp("model");

        let lm_layer_count = model_config
            .decoder_config
            .num_hidden_layers
            .saturating_sub(model_config.tts_backbone_num_hidden_layers);
        let language_model = RealtimeLanguageModel::load(
            &model_config,
            model_vb.pp("language_model"),
            &device,
            dtype,
            lm_layer_count,
            false,
        )?;
        let tts_language_model = RealtimeLanguageModel::load(
            &model_config,
            model_vb.pp("tts_language_model"),
            &device,
            dtype,
            model_config.tts_backbone_num_hidden_layers,
            true,
        )?;
        let tts_input_types = candle_nn::embedding(
            2,
            model_config.decoder_config.hidden_size,
            model_vb.pp("tts_input_types"),
        )?;
        let acoustic_tokenizer = VibeVoiceAcousticTokenizer::load_decoder_only(
            &model_config.acoustic_tokenizer_config,
            model_vb.pp("acoustic_tokenizer"),
        )?;
        let acoustic_connector = SpeechConnector::load(
            model_config.acoustic_vae_dim,
            model_config.decoder_config.hidden_size,
            model_vb.pp("acoustic_connector"),
        )?;
        let prediction_head = VibeVoiceDiffusionHead::load(
            &model_config.diffusion_head_config,
            model_vb.pp("prediction_head"),
        )?;
        let tts_eos_classifier = BinaryClassifier::load(
            model_config.decoder_config.hidden_size,
            vb.pp("tts_eos_classifier"),
        )?;
        let voice_names = discover_voice_names(files.voices_dir.as_ref())?;

        info!(
            "Loaded VibeVoice Realtime on {:?}: base_layers={}, tts_layers={}, hidden_size={}, voices={}",
            device,
            lm_layer_count,
            model_config.tts_backbone_num_hidden_layers,
            model_config.decoder_config.hidden_size,
            voice_names.len(),
        );

        Ok(Self {
            config: model_config.clone(),
            preprocessor_config,
            device,
            dtype,
            files,
            processor,
            language_model: Mutex::new(language_model),
            tts_language_model: Mutex::new(tts_language_model),
            tts_input_types,
            acoustic_tokenizer,
            acoustic_connector,
            prediction_head,
            noise_scheduler: Mutex::new(DpmSolverMultistepScheduler::new(
                &model_config.diffusion_head_config,
            )),
            tts_eos_classifier,
            speech_scaling_factor: load_scalar(&model_vb, "speech_scaling_factor")?,
            speech_bias_factor: load_scalar(&model_vb, "speech_bias_factor")?,
            voice_names,
            voice_cache: Mutex::new(HashMap::new()),
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        self.validate_request(request)?;
        let voice_name = resolve_voice_name(request.voice.as_deref(), &self.voice_names)?;
        let preset = self.load_voice_preset(&voice_name)?;
        let text_ids = self.processor.prepare_text(request)?;
        let prompt_spec = self.processor.tokenizer_spec().clone();
        let mut rng = SimpleRng::new(generation_seed());
        let max_steps = request.max_tokens.unwrap_or_else(|| {
            self.config
                .decoder_config
                .max_position_embeddings
                .saturating_sub(preset.tts_lm.prompt_len)
        });
        let cfg_scale = request.cfg_scale.unwrap_or(DEFAULT_CFG_SCALE as f64) as f32;
        let latents = self.generate_latents(
            &preset,
            &text_ids,
            &prompt_spec,
            cfg_scale,
            max_steps,
            &mut rng,
        )?;
        if latents.is_empty() {
            return Err(TtsError::ModelError(
                "VibeVoice Realtime did not produce any speech tokens".to_string(),
            ));
        }

        let segment = stack_latents(&latents)?;
        let audio = self
            .acoustic_tokenizer
            .decode(&self.unscale_latents(&segment)?)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        Ok(AudioSamples::new(
            audio,
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
        self.voice_names.clone()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "VibeVoice-Realtime-0.5B".to_string(),
            variant: self.config.model_type.clone(),
            parameters: 500_000_000,
            sample_rate: self.sample_rate(),
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl VibeVoiceRealtimeModel {
    pub fn files(&self) -> &ModelFiles {
        &self.files
    }

    fn validate_request(&self, request: &SynthesisRequest) -> Result<(), TtsError> {
        if request.reference_audio.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice Realtime uses cached prompt presets, not reference audio".to_string(),
            ));
        }
        if request.voice_embedding.is_some() {
            return Err(TtsError::ModelError(
                "VibeVoice Realtime does not support pre-extracted voice embeddings".to_string(),
            ));
        }
        if self.voice_names.is_empty() {
            return Err(TtsError::FileMissing(
                "voices/ — VibeVoice Realtime cached prompt presets (*.pt)".to_string(),
            ));
        }
        Ok(())
    }

    fn load_voice_preset(&self, voice_name: &str) -> Result<VoicePreset, TtsError> {
        if let Some(cached) = self
            .voice_cache
            .lock()
            .map_err(|_| {
                TtsError::RuntimeError("VibeVoice Realtime voice cache poisoned".to_string())
            })?
            .get(voice_name)
            .cloned()
        {
            return Ok(cached);
        }

        let file_name = format!("{voice_name}.pt");
        let asset = self
            .files
            .voices_dir
            .as_ref()
            .ok_or_else(|| {
                TtsError::FileMissing(
                    "voices/ — VibeVoice Realtime cached prompt presets (*.pt)".to_string(),
                )
            })?
            .load_file(&file_name)?;
        let preset = VoicePreset::load(&asset, &self.device, self.dtype)?;

        self.voice_cache
            .lock()
            .map_err(|_| {
                TtsError::RuntimeError("VibeVoice Realtime voice cache poisoned".to_string())
            })?
            .insert(voice_name.to_string(), preset.clone());
        Ok(preset)
    }

    fn generate_latents(
        &self,
        preset: &VoicePreset,
        text_ids: &[u32],
        _tokenizer_spec: &VibeVoiceTokenizerSpec,
        cfg_scale: f32,
        max_steps: usize,
        rng: &mut SimpleRng,
    ) -> Result<Vec<Tensor>, TtsError> {
        let mut text_index = 0usize;
        let mut step = preset.tts_lm.prompt_len;
        let mut lm_state = clone_branch_state(&preset.lm.state);
        let mut tts_state = clone_branch_state(&preset.tts_lm.state);
        let mut neg_tts_state = clone_branch_state(&preset.neg_tts_lm.state);
        let mut latents = Vec::new();
        let max_positions = self.config.decoder_config.max_position_embeddings;

        while step < max_steps && step < max_positions {
            let remaining_text = text_ids.len().saturating_sub(text_index);
            let text_window = remaining_text.min(TTS_TEXT_WINDOW_SIZE);
            if text_window > 0 {
                for token_id in &text_ids[text_index..text_index + text_window] {
                    lm_state = self.advance_lm_state(&lm_state, *token_id)?;
                    tts_state = self.advance_tts_state(
                        &tts_state,
                        *token_id,
                        &lm_state.last_hidden().clone(),
                        true,
                    )?;
                    step += 1;
                    if step >= max_steps || step >= max_positions {
                        break;
                    }
                }
                text_index += text_window;
            }

            let mut finished = false;
            for _ in 0..TTS_SPEECH_WINDOW_SIZE {
                let speech_latent = self.sample_speech_token(
                    tts_state.last_hidden(),
                    neg_tts_state.last_hidden(),
                    cfg_scale,
                    rng,
                )?;
                latents.push(speech_latent.clone());
                let acoustic_embed = self.acoustic_feedback_embedding(&speech_latent)?;
                tts_state = self.advance_tts_state(&tts_state, 1, &acoustic_embed, false)?;
                neg_tts_state =
                    self.advance_tts_state(&neg_tts_state, 1, &acoustic_embed, false)?;
                step += 1;

                let eos_logit = self
                    .tts_eos_classifier
                    .forward(tts_state.last_hidden())?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                if eos_logit > 0.0 {
                    finished = true;
                    break;
                }
                if step >= max_steps || step >= max_positions {
                    break;
                }
            }

            if finished || text_index >= text_ids.len() && step >= max_steps {
                break;
            }
            if text_index >= text_ids.len() && finished {
                break;
            }
            if text_index >= text_ids.len() && step >= max_positions {
                break;
            }
            if text_index >= text_ids.len() && max_steps <= step {
                break;
            }
        }

        Ok(latents)
    }

    fn advance_lm_state(
        &self,
        state: &RealtimeDecoderState,
        token_id: u32,
    ) -> Result<RealtimeDecoderState, TtsError> {
        let token_tensor = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
        let mut language_model = self.language_model.lock().map_err(|_| {
            TtsError::RuntimeError("VibeVoice Realtime language model mutex poisoned".to_string())
        })?;
        language_model.load_cache_state(state.layer_caches())?;
        let embedding = language_model.embed(&token_tensor)?;
        language_model.decode_step(&embedding, state.next_position())
    }

    fn advance_tts_state(
        &self,
        state: &RealtimeDecoderState,
        token_id: u32,
        hidden_override: &Tensor,
        is_text: bool,
    ) -> Result<RealtimeDecoderState, TtsError> {
        let token_tensor = Tensor::new(&[token_id], &self.device)?.unsqueeze(0)?;
        let mut tts_language_model = self.tts_language_model.lock().map_err(|_| {
            TtsError::RuntimeError(
                "VibeVoice Realtime TTS language model mutex poisoned".to_string(),
            )
        })?;
        tts_language_model.load_cache_state(state.layer_caches())?;
        let base_embedding = tts_language_model.embed(&token_tensor)?;
        let type_token = if is_text { 1u32 } else { 0u32 };
        let type_embedding = self
            .tts_input_types
            .forward(&Tensor::new(&[type_token], &self.device)?.unsqueeze(0)?)?;
        let hidden = normalize_override_embedding(hidden_override)?;
        let input_embedding = hidden.broadcast_add(&type_embedding)?;
        let _ = base_embedding;
        tts_language_model.decode_step(&input_embedding, state.next_position())
    }

    fn sample_speech_token(
        &self,
        positive_condition: &Tensor,
        negative_condition: &Tensor,
        cfg_scale: f32,
        rng: &mut SimpleRng,
    ) -> Result<Tensor, TtsError> {
        let mut scheduler = self.noise_scheduler.lock().map_err(|_| {
            TtsError::RuntimeError("VibeVoice Realtime scheduler mutex poisoned".to_string())
        })?;
        scheduler.set_timesteps(self.config.diffusion_head_config.ddpm_num_inference_steps);

        let condition = Tensor::cat(&[positive_condition, negative_condition], 0)?;
        let half = random_normal_tensor(
            (1, self.config.acoustic_vae_dim),
            self.dtype,
            &self.device,
            rng,
        )?;
        let mut speech = Tensor::cat(&[&half, &half], 0)?;
        let cfg_scale_tensor = Tensor::new(cfg_scale, &self.device)?;

        for timestep in scheduler.timesteps().to_vec() {
            let timestep_tensor =
                Tensor::from_vec(vec![timestep as f32, timestep as f32], (2,), &self.device)?
                    .to_dtype(self.dtype)?;
            let eps = self
                .prediction_head
                .forward(&speech, &timestep_tensor, &condition)?;
            let cond_eps = eps.narrow(0, 0, 1)?;
            let uncond_eps = eps.narrow(0, 1, 1)?;
            let guided = uncond_eps.broadcast_add(
                &cond_eps
                    .broadcast_sub(&uncond_eps)?
                    .broadcast_mul(&cfg_scale_tensor)?,
            )?;
            let expanded = Tensor::cat(&[&guided, &guided], 0)?;
            speech = scheduler.step(&expanded, &speech)?;
        }

        speech.narrow(0, 0, 1)?.squeeze(0).map_err(Into::into)
    }

    fn acoustic_feedback_embedding(&self, speech_latent: &Tensor) -> Result<Tensor, TtsError> {
        self.acoustic_connector
            .forward(&speech_latent.unsqueeze(0)?.unsqueeze(1)?)?
            .squeeze(0)?
            .squeeze(0)
            .map_err(Into::into)
    }

    fn unscale_latents(&self, latents: &Tensor) -> Result<Tensor, TtsError> {
        latents
            .broadcast_div(&Tensor::new(self.speech_scaling_factor, &self.device)?)?
            .broadcast_sub(&Tensor::new(self.speech_bias_factor, &self.device)?)
            .map_err(Into::into)
    }
}

fn resolve_runtime_dtype(device: &Device, requested: DType) -> DType {
    if matches!(device, Device::Cpu) && requested == DType::BF16 {
        return DType::F32;
    }
    if matches!(device, Device::Metal(_)) {
        return DType::F32;
    }
    requested
}

fn discover_voice_names(voices_dir: Option<&ModelAssetDir>) -> Result<Vec<String>, TtsError> {
    let Some(voices_dir) = voices_dir else {
        return Ok(Vec::new());
    };
    let mut names = voices_dir
        .file_names()?
        .into_iter()
        .filter_map(|name| name.strip_suffix(".pt").map(str::to_string))
        .collect::<Vec<_>>();
    names.sort();
    Ok(names)
}

fn resolve_voice_name(requested: Option<&str>, available: &[String]) -> Result<String, TtsError> {
    if available.is_empty() {
        return Err(TtsError::FileMissing(
            "voices/ — VibeVoice Realtime cached prompt presets (*.pt)".to_string(),
        ));
    }

    if let Some(requested) = requested {
        if let Some(exact) = available.iter().find(|voice| voice == &requested) {
            return Ok(exact.clone());
        }

        let lowered = requested.to_ascii_lowercase();
        if let Some(case_insensitive) = available
            .iter()
            .find(|voice| voice.to_ascii_lowercase() == lowered)
        {
            return Ok(case_insensitive.clone());
        }

        if let Some(partial) = available
            .iter()
            .find(|voice| voice.to_ascii_lowercase().contains(&lowered))
        {
            return Ok(partial.clone());
        }

        return Err(TtsError::UnknownVoice(requested.to_string()));
    }

    if let Some(default_voice) = available
        .iter()
        .find(|voice| voice.as_str() == "en-Carter_man")
    {
        return Ok(default_voice.clone());
    }
    Ok(available[0].clone())
}

fn normalize_override_embedding(hidden: &Tensor) -> Result<Tensor, TtsError> {
    match hidden.rank() {
        1 => hidden.unsqueeze(0)?.unsqueeze(0).map_err(Into::into),
        2 => hidden.unsqueeze(1).map_err(Into::into),
        3 => Ok(hidden.clone()),
        _ => Err(TtsError::ModelError(
            "Unexpected VibeVoice Realtime hidden override rank".to_string(),
        )),
    }
}

fn clone_branch_state(state: &RealtimeDecoderState) -> RealtimeDecoderState {
    RealtimeDecoderState::new(
        state.next_position(),
        state.last_hidden().clone(),
        clone_layer_caches(state.layer_caches()),
    )
}

fn clone_layer_caches(layer_caches: &[LayerKvCache]) -> Vec<LayerKvCache> {
    layer_caches
        .iter()
        .map(|cache| {
            cache
                .as_ref()
                .map(|(key, value)| (key.clone(), value.clone()))
        })
        .collect()
}

fn load_scalar(model_vb: &candle_nn::VarBuilder, name: &str) -> Result<f32, TtsError> {
    let tensor = model_vb.get(1, name).or_else(|_| model_vb.get((), name))?;
    let values = tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    values.first().copied().ok_or_else(|| {
        TtsError::ModelError("Expected scalar tensor in VibeVoice Realtime weights".to_string())
    })
}
