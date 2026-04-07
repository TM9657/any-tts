//! Kokoro-82M full model implementation.
//!
//! Ties together Albert, TextEncoder, ProsodyPredictor, and IstftDecoder
//! into a unified `TtsModel` that converts plain text to 24 kHz audio.
//!
//! ## Inference pipeline
//!
//! 1. Plain text → espeak-ng phonemization → Kokoro IPA post-processing
//! 2. IPA phoneme chars → char-level token IDs via vocab
//! 3. ALBERT (PL-BERT) encodes phonemes → `bert_encoder` projects to hidden dim
//! 4. ProsodyPredictor predicts duration, F0, and noise from encoded features + style
//! 5. Duration alignment creates expanded representation
//! 6. TextEncoder produces aligned features
//! 7. ISTFTNet Decoder generates 24 kHz audio waveform
//!
//! ## Voice embeddings
//!
//! Voice `.pt` files contain `[512, 256]` tensors — 512 context positions ×
//! (128 decoder style + 128 predictor style). The appropriate row is selected
//! based on input length.

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use tracing::info;

use crate::audio::AudioSamples;
use crate::config::{ModelAsset, ModelAssetDir, ModelFiles, TtsConfig};
use crate::error::{TtsError, TtsResult};
use crate::traits::{
    ModelInfo, ReferenceAudio, SynthesisRequest, TtsModel, VoiceCloning, VoiceEmbedding,
};

use super::albert::Albert;
use super::config::KokoroConfig;
use super::decoder::IstftDecoder;
use super::phonemizer;
use super::prosody::ProsodyPredictor;
use super::style_encoder::StyleEncoder;
use super::text_encoder::TextEncoder;

/// Style dimension (per component: decoder or predictor).
const STYLE_DIM: usize = 128;
/// Full voice embedding dimension (decoder + predictor).
const _FULL_STYLE_DIM: usize = STYLE_DIM * 2;
/// Maximum context positions in voice embedding tensor.
const _MAX_VOICE_POSITIONS: usize = 512;
/// Maximum offset index for voice selection.
const MAX_VOICE_OFFSET: usize = 509;

/// Kokoro-82M TTS model.
pub struct KokoroModel {
    config: KokoroConfig,
    bert: Albert,
    bert_encoder: candle_nn::Linear,
    predictor: ProsodyPredictor,
    text_encoder: TextEncoder,
    decoder: IstftDecoder,
    device: Device,
    dtype: DType,
    voices_dir: Option<ModelAssetDir>,
    /// Style encoder pair for voice cloning (if weights were found).
    style_encoder: Option<StyleEncoder>,
    /// Cached voice embeddings: voice_name → [MAX_VOICE_POSITIONS, FULL_STYLE_DIM]
    voice_cache: std::sync::RwLock<HashMap<String, Tensor>>,
    /// Reverse vocab for debugging: id → phoneme
    #[allow(dead_code)]
    id_to_phoneme: HashMap<u32, String>,
}

impl std::fmt::Debug for KokoroModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KokoroModel")
            .field("hidden_dim", &self.config.hidden_dim)
            .field("style_dim", &self.config.style_dim)
            .field("n_token", &self.config.n_token)
            .field("device", &self.device)
            .field("has_voices_dir", &self.voices_dir.is_some())
            .field("has_style_encoder", &self.style_encoder.is_some())
            .finish()
    }
}

impl TtsModel for KokoroModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError>
    where
        Self: Sized,
    {
        let files = config.resolve_files()?;
        let device = config.device.resolve()?;
        let mut dtype = config.dtype.to_candle();

        if dtype == DType::BF16 {
            if device.is_cpu() {
                info!("BF16 not supported on CPU, falling back to F32");
                dtype = DType::F32;
            } else if matches!(device, Device::Metal(_)) {
                info!("BF16 is not supported on Metal for Kokoro; falling back to F32");
                dtype = DType::F32;
            }
        }

        // Parse model config
        let config_bytes = files
            .config
            .as_ref()
            .ok_or_else(|| TtsError::FileMissing("config.json".into()))?
            .read_bytes()?;
        let kokoro_config = KokoroConfig::from_bytes(config_bytes.as_ref())?;

        info!(
            "Loading Kokoro-82M: hidden_dim={}, style_dim={}, n_token={}, vocab_size={}",
            kokoro_config.hidden_dim,
            kokoro_config.style_dim,
            kokoro_config.n_token,
            kokoro_config.vocab.len(),
        );

        // Load weights — try safetensors first, then .pth
        let vb = Self::load_weights(&files.weights, dtype, &device)?;

        // Build submodules
        let bert = Albert::load(&kokoro_config.plbert, vb.pp("bert"), &device)?;
        let bert_encoder = candle_nn::linear(
            kokoro_config.plbert.hidden_size,
            kokoro_config.hidden_dim,
            vb.pp("bert_encoder"),
        )?;

        let predictor = ProsodyPredictor::load(
            kokoro_config.style_dim,
            kokoro_config.hidden_dim,
            kokoro_config.n_layer,
            kokoro_config.max_dur,
            vb.pp("predictor"),
            &device,
        )?;

        let text_encoder = TextEncoder::load(
            kokoro_config.hidden_dim,
            kokoro_config.text_encoder_kernel_size,
            kokoro_config.n_layer,
            kokoro_config.n_token,
            vb.pp("text_encoder"),
            &device,
        )?;

        let istft = &kokoro_config.istftnet;
        let decoder = IstftDecoder::load(
            kokoro_config.hidden_dim,
            kokoro_config.style_dim,
            kokoro_config.n_mels,
            &istft.resblock_kernel_sizes,
            &istft.upsample_rates,
            istft.upsample_initial_channel,
            &istft.resblock_dilation_sizes,
            &istft.upsample_kernel_sizes,
            istft.gen_istft_n_fft,
            istft.gen_istft_hop_size,
            vb.pp("decoder"),
            &device,
            dtype,
        )?;

        // Try to load style encoders for voice cloning
        let style_encoder = StyleEncoder::try_load(
            kokoro_config.dim_in,
            kokoro_config.style_dim,
            kokoro_config.max_conv_dim,
            kokoro_config.sample_rate(),
            &vb,
            &device,
        )?;

        // Build reverse vocab
        let id_to_phoneme: HashMap<u32, String> = kokoro_config
            .vocab
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        info!("Kokoro-82M loaded successfully");

        Ok(Self {
            config: kokoro_config,
            bert,
            bert_encoder,
            predictor,
            text_encoder,
            decoder,
            device,
            dtype,
            voices_dir: files.voices_dir,
            style_encoder,
            voice_cache: std::sync::RwLock::new(HashMap::new()),
            id_to_phoneme,
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        // Determine language: explicit > inferred from voice name > default "en"
        let voice_name = request.voice.as_deref().unwrap_or("af_heart");
        let language = request
            .language
            .as_deref()
            .unwrap_or_else(|| phonemizer::language_from_voice(voice_name));

        // Convert plain text → IPA phonemes → token IDs
        let phonemes = phonemizer::phonemize(&request.text, language, &self.config.vocab)?;
        let token_ids = self.text_to_ids(&phonemes)?;
        if token_ids.is_empty() {
            return Err(TtsError::TokenizerError(
                "No valid phoneme tokens found in input".into(),
            ));
        }

        // Pad with boundary tokens: [0, ...ids..., 0]
        let mut padded_ids = Vec::with_capacity(token_ids.len() + 2);
        padded_ids.push(0u32); // Start boundary
        padded_ids.extend_from_slice(&token_ids);
        padded_ids.push(0u32); // End boundary
        let seq_len = padded_ids.len();

        if std::env::var_os("KOKORO_DEBUG_INPUTS").is_some() {
            eprintln!("[kokoro] language={language} voice={voice_name}");
            eprintln!("[kokoro] phonemes={phonemes}");
            eprintln!("[kokoro] input_ids={padded_ids:?}");
        }

        // Resolve voice embedding — priority: voice_embedding > reference_audio > named voice
        let ref_s = if let Some(ref embedding) = request.voice_embedding {
            self.voice_embedding_to_tensor(embedding, seq_len)?
        } else if let Some(ref reference) = request.reference_audio {
            self.voice_from_reference(reference)?
        } else {
            self.load_voice_embedding(voice_name, seq_len)?
        };

        // Speed factor
        let speed = request.temperature.unwrap_or(1.0).clamp(0.1, 5.0);

        // Run inference
        let audio_tensor = self.forward_with_tokens(&padded_ids, &ref_s, speed)?;

        // Extract audio samples
        let audio_data: Vec<f32> = audio_tensor
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1()?;

        Ok(AudioSamples::new(audio_data, self.sample_rate()))
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate()
    }

    fn supported_languages(&self) -> Vec<String> {
        vec![
            "en".into(),
            "en-gb".into(),
            "ja".into(),
            "zh".into(),
            "ko".into(),
            "fr".into(),
            "de".into(),
            "it".into(),
            "pt".into(),
            "es".into(),
            "hi".into(),
        ]
    }

    fn supported_voices(&self) -> Vec<String> {
        self.discover_voices().unwrap_or_default()
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "Kokoro".into(),
            variant: "82M".into(),
            parameters: 82_000_000,
            sample_rate: self.sample_rate(),
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl KokoroModel {
    /// Load model weights from file(s).
    ///
    /// Supports both SafeTensors (`.safetensors`) and PyTorch (`.pth`) formats.
    fn load_weights(
        weight_assets: &[ModelAsset],
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>, TtsError> {
        if weight_assets.is_empty() {
            return Err(TtsError::FileMissing("model weights".into()));
        }

        let first = &weight_assets[0];
        let ext = first.extension().unwrap_or("");

        match ext {
            "safetensors" => ModelFiles::load_safetensors_vb(weight_assets, dtype, device),
            "pth" => {
                // Load PyTorch .pth via candle's pickle support.
                //
                // Kokoro .pth files have a NESTED state dict structure:
                // top-level keys (bert, bert_encoder, decoder, predictor,
                // text_encoder) each map to an OrderedDict of tensors.
                // Additionally, each sub-dict has a `.module.` prefix from
                // DataParallel wrapping.
                //
                // We parse the archive from raw bytes so the same path works
                // for filesystem-backed assets and object-store byte blobs.
                info!("Loading .pth weights from {}", first.display_name());

                let mut tensors: HashMap<String, Tensor> = HashMap::new();
                let archive_bytes = first.read_bytes()?;
                let top_keys = [
                    "bert",
                    "bert_encoder",
                    "decoder",
                    "predictor",
                    "text_encoder",
                ];

                for top_key in &top_keys {
                    let sub_tensors =
                        Self::load_tensors_from_pth_bytes(archive_bytes.as_ref(), Some(top_key))?;

                    for (name, tensor) in sub_tensors {
                        let full_name = format!("{}.{}", top_key, name);
                        let clean_name = full_name.replace(".module.", ".");
                        tensors.insert(clean_name, tensor);
                    }
                }

                info!(
                    "Loaded {} tensors from .pth (DataParallel unwrapped)",
                    tensors.len()
                );
                Ok(VarBuilder::from_tensors(tensors, dtype, device))
            }
            other => Err(TtsError::WeightLoadError(format!(
                "Unsupported weight format: .{other}. Expected .safetensors or .pth"
            ))),
        }
    }

    /// Convert IPA phoneme text to token IDs using the vocab map.
    ///
    /// Each character in the input string is looked up in the phoneme vocab.
    /// Unknown characters are silently skipped.
    fn text_to_ids(&self, text: &str) -> TtsResult<Vec<u32>> {
        let mut ids = Vec::with_capacity(text.len());

        for ch in text.chars() {
            let key = ch.to_string();
            if let Some(&id) = self.config.vocab.get(&key) {
                ids.push(id);
            }
            // Silently skip unknown characters
        }

        Ok(ids)
    }

    /// Load a voice embedding tensor from a `.pt` file.
    ///
    /// Voice files contain `[512, 256]` tensors. We select the row based
    /// on input sequence length.
    fn load_voice_embedding(&self, voice_name: &str, num_tokens: usize) -> TtsResult<Tensor> {
        // Check cache first
        {
            let cache = self
                .voice_cache
                .read()
                .map_err(|e| TtsError::ModelError(format!("Voice cache lock error: {}", e)))?;
            if let Some(voice_data) = cache.get(voice_name) {
                return Self::select_voice_row(voice_data, num_tokens);
            }
        }

        // Load from disk
        let voices_dir = self.voices_dir.as_ref().ok_or_else(|| {
            TtsError::UnknownVoice(format!(
                "No voices directory configured. Cannot load voice '{}'",
                voice_name
            ))
        })?;

        let voice_asset = voices_dir
            .load_file(&format!("{}.pt", voice_name))
            .map_err(|_| {
                TtsError::UnknownVoice(format!("Voice file not found: {}.pt", voice_name))
            })?;

        info!("Loading voice embedding: {}", voice_asset.display_name());

        // Load .pt tensor — Kokoro voice files contain a bare tensor (not a dict)
        let voice_data = Self::load_bare_pt_tensor(&voice_asset)?;
        let voice_data = voice_data.to_device(&self.device)?.to_dtype(self.dtype)?;

        // Cache it
        {
            let mut cache = self
                .voice_cache
                .write()
                .map_err(|e| TtsError::ModelError(format!("Voice cache lock error: {}", e)))?;
            cache.insert(voice_name.to_string(), voice_data.clone());
        }

        Self::select_voice_row(&voice_data, num_tokens)
    }

    /// Load a bare tensor from a PyTorch .pt file.
    ///
    /// Kokoro voice .pt files store a raw tensor (not wrapped in a dict),
    /// which candle's `PthTensors` cannot handle. We use candle's public
    /// `Stack` + `Object` API and the `zip` crate to parse the pickle
    /// and read the tensor data directly.
    fn load_bare_pt_tensor(asset: &ModelAsset) -> TtsResult<Tensor> {
        use std::io::{BufReader, Cursor, Read};

        let bytes = asset.read_bytes()?;

        // First try dict-wrapped tensors.
        let dict_tensors = Self::load_tensors_from_pth_bytes(bytes.as_ref(), None)?;
        for key in &["", "0", "data", "weight"] {
            if let Some(tensor) = dict_tensors.get(*key) {
                return Ok(tensor.clone());
            }
        }

        // Fall back to bare-tensor parsing for non-dict .pt files
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes.as_ref()))
            .map_err(|e| TtsError::WeightLoadError(format!("Invalid .pt ZIP: {}", e)))?;

        // Find the .pkl file and the data directory prefix
        let pkl_name = {
            let names: Vec<String> = zip.file_names().map(|s| s.to_string()).collect();
            names
                .into_iter()
                .find(|n| n.ends_with("data.pkl"))
                .ok_or_else(|| TtsError::WeightLoadError("No data.pkl in .pt file".into()))?
        };
        let _dir_prefix = pkl_name.strip_suffix("data.pkl").unwrap_or("").to_string();

        // Parse the pickle to extract tensor metadata
        let pkl_reader = zip
            .by_name(&pkl_name)
            .map_err(|e| TtsError::WeightLoadError(format!("Cannot read data.pkl: {}", e)))?;
        let mut buf_reader = BufReader::new(pkl_reader);
        let mut stack = candle_core::pickle::Stack::empty();
        stack
            .read_loop(&mut buf_reader)
            .map_err(|e| TtsError::WeightLoadError(format!("Pickle parse error: {}", e)))?;
        let obj = stack
            .finalize()
            .map_err(|e| TtsError::WeightLoadError(format!("Pickle finalize error: {}", e)))?;

        // Convert the top-level object to TensorInfo
        let dir_name = std::path::PathBuf::from(pkl_name.strip_suffix(".pkl").unwrap_or(&pkl_name));
        let dummy_name = candle_core::pickle::Object::Unicode(String::new());
        let tensor_info = obj
            .into_tensor_info(dummy_name, &dir_name)
            .map_err(|e| TtsError::WeightLoadError(format!("Not a tensor object: {}", e)))?
            .ok_or_else(|| {
                TtsError::WeightLoadError("Top-level .pt object is not a tensor".into())
            })?;

        // Read the raw tensor data from the ZIP
        // Re-open the ZIP since we consumed the reader
        let mut zip2 = zip::ZipArchive::new(Cursor::new(bytes.as_ref()))
            .map_err(|e| TtsError::WeightLoadError(format!("Invalid .pt ZIP: {}", e)))?;

        let data_path = &tensor_info.path;
        let mut data_reader = zip2.by_name(data_path).map_err(|e| {
            TtsError::WeightLoadError(format!("Cannot find data file '{}': {}", data_path, e))
        })?;

        // Skip to the start offset if layout is offset
        let start_offset = tensor_info.layout.start_offset();
        if start_offset > 0 {
            std::io::copy(
                &mut data_reader.by_ref().take(start_offset as u64),
                &mut std::io::sink(),
            )
            .map_err(|e| TtsError::WeightLoadError(format!("Seek error: {}", e)))?;
        }

        // Read the tensor data
        let elem_count = tensor_info.layout.shape().elem_count();
        let dtype = tensor_info.dtype;
        let shape = tensor_info.layout.shape().clone();

        let tensor = match dtype {
            candle_core::DType::F32 => {
                let mut data = vec![0u8; elem_count * 4];
                data_reader
                    .read_exact(&mut data)
                    .map_err(|e| TtsError::WeightLoadError(format!("Read error: {}", e)))?;
                Tensor::from_raw_buffer(&data, dtype, shape.dims(), &candle_core::Device::Cpu)
                    .map_err(|e| {
                        TtsError::WeightLoadError(format!("Tensor creation error: {}", e))
                    })?
            }
            candle_core::DType::F16 | candle_core::DType::BF16 => {
                let mut data = vec![0u8; elem_count * 2];
                data_reader
                    .read_exact(&mut data)
                    .map_err(|e| TtsError::WeightLoadError(format!("Read error: {}", e)))?;
                Tensor::from_raw_buffer(&data, dtype, shape.dims(), &candle_core::Device::Cpu)
                    .map_err(|e| {
                        TtsError::WeightLoadError(format!("Tensor creation error: {}", e))
                    })?
            }
            candle_core::DType::F64 => {
                let mut data = vec![0u8; elem_count * 8];
                data_reader
                    .read_exact(&mut data)
                    .map_err(|e| TtsError::WeightLoadError(format!("Read error: {}", e)))?;
                Tensor::from_raw_buffer(&data, dtype, shape.dims(), &candle_core::Device::Cpu)
                    .map_err(|e| {
                        TtsError::WeightLoadError(format!("Tensor creation error: {}", e))
                    })?
            }
            other => {
                return Err(TtsError::WeightLoadError(format!(
                    "Unsupported dtype for voice tensor: {:?}",
                    other
                )));
            }
        };

        info!(
            "Loaded bare voice tensor: shape={:?}, dtype={:?}",
            shape, dtype
        );
        Ok(tensor)
    }

    /// Select the voice embedding row for the given sequence length.
    ///
    /// `voice_data`: [N, 1, 256] or [N, 256] tensor
    /// `num_tokens`: number of input tokens (including boundary padding)
    ///
    /// Returns: [1, 256] style vector
    fn select_voice_row(voice_data: &Tensor, num_tokens: usize) -> TtsResult<Tensor> {
        // Squeeze any singleton middle dimensions: [N, 1, 256] → [N, 256]
        let voice_data = voice_data.squeeze(1).unwrap_or_else(|_| voice_data.clone());
        let n = voice_data.dim(0)?;
        let offset = num_tokens
            .saturating_sub(3)
            .min(n.saturating_sub(1))
            .min(MAX_VOICE_OFFSET);

        if std::env::var_os("KOKORO_DEBUG_INPUTS").is_some() {
            eprintln!("[kokoro] voice_row_index={offset} total_rows={n} num_tokens={num_tokens}");
        }

        let row = voice_data.i(offset)?.unsqueeze(0)?; // [1, 256]
        Ok(row)
    }

    /// Full forward pass with pre-tokenized input.
    ///
    /// `input_ids`: padded phoneme token IDs [0, ...ids..., 0]
    /// `ref_s`: [1, 256] voice embedding
    /// `speed`: speaking speed factor (1.0 = normal)
    ///
    /// Returns: [1, num_samples] audio waveform
    fn forward_with_tokens(
        &self,
        input_ids: &[u32],
        ref_s: &Tensor,
        speed: f64,
    ) -> TtsResult<Tensor> {
        let seq_len = input_ids.len();

        // Create input tensor: [1, seq_len]
        let ids_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;

        // Create input lengths tensor
        let input_lengths = Tensor::new(&[seq_len as u32], &self.device)?;

        // Create text mask: True for positions beyond the actual length
        // For single-sequence inference, all positions are valid → mask is all False
        let text_mask = Tensor::zeros((1, seq_len), DType::U8, &self.device)?;

        // Create attention mask for ALBERT: 1 for valid, 0 for padded
        let attention_mask =
            Tensor::ones((1, seq_len), DType::F32, &self.device)?.to_dtype(self.dtype)?;

        // Step 1: ALBERT (PL-BERT) encoding
        let bert_out = self.bert.forward(&ids_tensor, &attention_mask)?;
        // [1, seq_len, plbert_hidden_size]

        // Step 2: Project to hidden dim and transpose
        let d_en = self.bert_encoder.forward(&bert_out)?;
        // [1, seq_len, hidden_dim]
        let d_en = d_en.transpose(1, 2)?;
        // [1, hidden_dim, seq_len]

        // Step 3: Split style vector
        // ref_s[:, 128:] → predictor style
        // ref_s[:, :128] → decoder style
        let s_pred = ref_s.narrow(1, STYLE_DIM, STYLE_DIM)?;
        let s_dec = ref_s.narrow(1, 0, STYLE_DIM)?;

        // Step 4: Prosody prediction — duration
        let d = self
            .predictor
            .text_encoder
            .forward(&d_en, &s_pred, &input_lengths, &text_mask)?;
        // d: [1, seq_len, hidden_dim]

        let duration = self.predictor.predict_duration(&d, &s_pred)?;
        // [1, seq_len]

        // Apply speed
        let duration = duration.broadcast_mul(
            &Tensor::new((1.0 / speed) as f32, duration.device())?.to_dtype(duration.dtype())?,
        )?;

        // Round and clamp durations
        let pred_dur = Self::round_and_clamp_durations(&duration)?;
        // Vec<usize>

        if std::env::var_os("KOKORO_DEBUG_INPUTS").is_some() {
            eprintln!("[kokoro] pred_dur={pred_dur:?}");
            eprintln!("[kokoro] aligned_len={}", pred_dur.iter().sum::<usize>());
        }

        // Step 5: Duration alignment
        let aligned_len: usize = pred_dur.iter().sum();
        if aligned_len == 0 {
            return Err(TtsError::ModelError(
                "All predicted durations are zero".into(),
            ));
        }

        let pred_aln_trg = Self::build_alignment_matrix(
            &pred_dur,
            seq_len,
            aligned_len,
            &self.device,
            self.dtype,
        )?;
        // [1, seq_len, aligned_len]

        // Step 6: Expand encoded features
        // d: [1, seq_len, hidden_dim] → transpose to [1, hidden_dim, seq_len]
        let d_t = d.transpose(1, 2)?.contiguous()?;
        let pred_aln_trg = pred_aln_trg.contiguous()?;
        let en = d_t.matmul(&pred_aln_trg)?;
        // [1, hidden_dim, aligned_len]

        // Step 7: Predict F0 and noise
        let (f0_pred, n_pred) = self.predictor.f0_n_predict(&en, &s_pred)?;
        // f0_pred: [1, aligned_len * 2]
        // n_pred: [1, aligned_len * 2]

        // Step 8: Text encoder
        let t_en = self
            .text_encoder
            .forward(&ids_tensor, &input_lengths, &text_mask)?;
        // [1, hidden_dim, seq_len]

        // Step 9: Align text encoder output
        let t_en = t_en.contiguous()?;
        let asr = t_en.matmul(&pred_aln_trg)?;
        // [1, hidden_dim, aligned_len]

        // Step 10: Decode to audio
        let audio = self.decoder.forward(&asr, &f0_pred, &n_pred, &s_dec)?;
        // [1, num_samples]

        Ok(audio)
    }

    /// Round durations to integers and clamp to minimum of 1.
    fn round_and_clamp_durations(duration: &Tensor) -> TtsResult<Vec<usize>> {
        const DURATION_ROUND_EPSILON: f32 = 2.5e-4;

        let dur_f32: Vec<f32> = duration
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1()
            .map_err(TtsError::ComputeError)?;

        Ok(dur_f32
            .iter()
            .map(|&d| (d + DURATION_ROUND_EPSILON).round().max(1.0) as usize)
            .collect())
    }

    /// Build duration alignment matrix.
    ///
    /// Creates a [1, seq_len, aligned_len] tensor where each row i has 1s
    /// in the columns corresponding to phoneme i's duration span.
    fn build_alignment_matrix(
        pred_dur: &[usize],
        seq_len: usize,
        aligned_len: usize,
        device: &Device,
        dtype: DType,
    ) -> TtsResult<Tensor> {
        let mut aln_data = vec![0f32; seq_len * aligned_len];
        let mut col = 0;
        for (i, &dur) in pred_dur.iter().enumerate() {
            for j in 0..dur {
                if col + j < aligned_len {
                    aln_data[i * aligned_len + col + j] = 1.0;
                }
            }
            col += dur;
        }

        let aln = Tensor::new(aln_data.as_slice(), device)?
            .reshape((1, seq_len, aligned_len))?
            .to_dtype(dtype)?;

        Ok(aln)
    }

    /// Discover available voice names from the voices directory.
    fn discover_voices(&self) -> TtsResult<Vec<String>> {
        let voices_dir = match &self.voices_dir {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };

        let mut names: Vec<String> = voices_dir
            .file_names()?
            .into_iter()
            .filter_map(|name| {
                let path = std::path::Path::new(&name);
                if path.extension().and_then(|ext| ext.to_str()) == Some("pt") {
                    path.file_stem()
                        .and_then(|stem| stem.to_str())
                        .map(String::from)
                } else {
                    None
                }
            })
            .collect();

        names.sort();
        Ok(names)
    }

    /// Convert a [`VoiceEmbedding`] to a style tensor for synthesis.
    ///
    /// The embedding must be of type `"kokoro"` and have shape `[1, 256]`.
    fn voice_embedding_to_tensor(
        &self,
        embedding: &VoiceEmbedding,
        _num_tokens: usize,
    ) -> TtsResult<Tensor> {
        if embedding.model_type() != "kokoro" {
            return Err(TtsError::ModelError(format!(
                "Voice embedding type '{}' is not compatible with Kokoro (expected 'kokoro')",
                embedding.model_type()
            )));
        }
        embedding
            .to_tensor(&self.device)?
            .to_dtype(self.dtype)
            .map_err(TtsError::from)
    }

    /// Extract a style vector from reference audio using the style encoders.
    fn voice_from_reference(&self, audio: &ReferenceAudio) -> TtsResult<Tensor> {
        let se = self.style_encoder.as_ref().ok_or_else(|| {
            TtsError::ModelError(
                "Voice cloning not available: style encoder weights were not found \
                 in the model checkpoint. Use a pre-computed voice pack (.pt file) \
                 or a VoiceEmbedding instead."
                    .into(),
            )
        })?;

        if audio.is_empty() {
            return Err(TtsError::ModelError("Reference audio is empty".into()));
        }

        se.encode(audio, self.dtype)
    }

    fn load_tensors_from_pth_bytes(
        bytes: &[u8],
        key: Option<&str>,
    ) -> Result<HashMap<String, Tensor>, TtsError> {
        let mut tensors = HashMap::new();
        for tensor_info in Self::read_pth_tensor_info_from_bytes(bytes, key)? {
            let tensor = Self::read_tensor_from_pth_bytes(bytes, &tensor_info)?;
            tensors.insert(tensor_info.name.clone(), tensor);
        }
        Ok(tensors)
    }

    fn read_pth_tensor_info_from_bytes(
        bytes: &[u8],
        key: Option<&str>,
    ) -> Result<Vec<candle_core::pickle::TensorInfo>, TtsError> {
        use candle_core::pickle::{Object, Stack};

        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| {
            TtsError::WeightLoadError(format!("Failed to open .pth archive: {}", e))
        })?;
        let zip_file_names = zip
            .file_names()
            .map(|name| name.to_string())
            .collect::<Vec<_>>();

        let mut tensor_infos = Vec::new();
        for file_name in zip_file_names
            .iter()
            .filter(|name| name.ends_with("data.pkl"))
        {
            let dir_name = std::path::PathBuf::from(
                file_name
                    .strip_suffix(".pkl")
                    .ok_or_else(|| TtsError::WeightLoadError("Missing .pkl suffix".into()))?,
            );
            let reader = zip.by_name(file_name).map_err(|e| {
                TtsError::WeightLoadError(format!("Failed to read {}: {}", file_name, e))
            })?;
            let mut reader = std::io::BufReader::new(reader);
            let mut stack = Stack::empty();
            stack.read_loop(&mut reader).map_err(|e| {
                TtsError::WeightLoadError(format!("Pickle parse error in {}: {}", file_name, e))
            })?;
            let obj = stack.finalize().map_err(|e| {
                TtsError::WeightLoadError(format!("Pickle finalize error in {}: {}", file_name, e))
            })?;

            let obj = match obj {
                Object::Build { callable, args } => match *callable {
                    Object::Reduce { callable, args: _ } => match *callable {
                        Object::Class {
                            module_name,
                            class_name,
                        } if module_name == "__torch__" && class_name == "Module" => *args,
                        _ => continue,
                    },
                    _ => continue,
                },
                obj => obj,
            };

            let obj = if let Some(key) = key {
                if let Object::Dict(key_values) = obj {
                    key_values
                        .into_iter()
                        .find(|(k, _)| *k == Object::Unicode(key.to_string()))
                        .map(|(_, value)| value)
                        .ok_or_else(|| {
                            TtsError::WeightLoadError(format!("Missing .pth key '{}'", key))
                        })?
                } else {
                    obj
                }
            } else {
                obj
            };

            if let Object::Dict(key_values) = obj {
                for (name, value) in key_values {
                    match value.into_tensor_info(name, &dir_name) {
                        Ok(Some(tensor_info)) => tensor_infos.push(tensor_info),
                        Ok(None) => {}
                        Err(_) => {}
                    }
                }
            }
        }

        Ok(tensor_infos)
    }

    fn read_tensor_from_pth_bytes(
        bytes: &[u8],
        tensor_info: &candle_core::pickle::TensorInfo,
    ) -> Result<Tensor, TtsError> {
        use std::io::Read;

        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| {
            TtsError::WeightLoadError(format!("Failed to open .pth archive: {}", e))
        })?;
        let mut reader = zip.by_name(&tensor_info.path).map_err(|e| {
            TtsError::WeightLoadError(format!(
                "Failed to open tensor payload '{}': {}",
                tensor_info.path, e
            ))
        })?;
        let is_fortran_contiguous = tensor_info.layout.is_fortran_contiguous();
        let rank = tensor_info.layout.shape().rank();

        if !tensor_info.layout.is_contiguous() && !is_fortran_contiguous {
            return Err(TtsError::WeightLoadError(format!(
                "Unsupported non-contiguous tensor layout for '{}'",
                tensor_info.name
            )));
        }

        let start_offset = tensor_info.layout.start_offset();
        if start_offset > 0 {
            std::io::copy(
                &mut reader.by_ref().take(start_offset as u64),
                &mut std::io::sink(),
            )
            .map_err(TtsError::from)?;
        }

        let elem_count = tensor_info.layout.shape().elem_count();
        let byte_count = elem_count * tensor_info.dtype.size_in_bytes();
        let mut raw = vec![0u8; byte_count];
        reader.read_exact(&mut raw)?;
        let tensor = Tensor::from_raw_buffer(
            &raw,
            tensor_info.dtype,
            tensor_info.layout.shape().dims(),
            &Device::Cpu,
        )?;

        if rank > 1 && is_fortran_contiguous {
            let shape_reversed: Vec<_> = tensor_info.layout.dims().iter().rev().copied().collect();
            let tensor = tensor.reshape(shape_reversed)?;
            let dim_indices_reversed: Vec<_> = (0..rank).rev().collect();
            Ok(tensor.permute(dim_indices_reversed)?)
        } else {
            Ok(tensor)
        }
    }
}

// ---------------------------------------------------------------------------
// VoiceCloning implementation
// ---------------------------------------------------------------------------

impl VoiceCloning for KokoroModel {
    fn supports_voice_cloning(&self) -> bool {
        self.style_encoder.is_some()
    }

    fn extract_voice(&self, audio: &ReferenceAudio) -> Result<VoiceEmbedding, TtsError> {
        let style_tensor = self.voice_from_reference(audio)?;
        // style_tensor: [1, 256]
        let data: Vec<f32> = style_tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let shape: Vec<usize> = style_tensor.dims().to_vec();

        Ok(VoiceEmbedding::new(data, shape, "kokoro"))
    }

    fn synthesize_with_voice(
        &self,
        request: &SynthesisRequest,
        voice: &VoiceEmbedding,
    ) -> Result<AudioSamples, TtsError> {
        // Build a request clone with the embedding attached
        let mut req = request.clone();
        req.voice_embedding = Some(voice.clone());
        // Clear reference_audio so we use the embedding directly
        req.reference_audio = None;
        self.synthesize(&req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_alignment_matrix() {
        let pred_dur = vec![2, 3, 1];
        let seq_len = 3;
        let aligned_len = 6;
        let device = Device::Cpu;

        let aln = KokoroModel::build_alignment_matrix(
            &pred_dur,
            seq_len,
            aligned_len,
            &device,
            DType::F32,
        )
        .unwrap();

        assert_eq!(aln.dims(), &[1, 3, 6]);

        let data: Vec<Vec<f32>> = aln.squeeze(0).unwrap().to_vec2().unwrap();
        // Row 0: dur=2 → cols 0,1
        assert_eq!(data[0], vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        // Row 1: dur=3 → cols 2,3,4
        assert_eq!(data[1], vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
        // Row 2: dur=1 → col 5
        assert_eq!(data[2], vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_round_and_clamp_durations() {
        let device = Device::Cpu;
        let dur = Tensor::new(&[[0.3f32, 1.7, 0.0, 2.5]], &device).unwrap();
        let result = KokoroModel::round_and_clamp_durations(&dur).unwrap();
        // 0.3 → round(0.3)=0 → clamp(min=1) = 1
        // 1.7 → round(1.7)=2
        // 0.0 → round(0.0)=0 → clamp(min=1) = 1
        // 2.5 → round(2.5)=3 (rounds to even in Rust, actually 2.5.round() = 3)
        assert_eq!(result, vec![1, 2, 1, 3]);
    }
}
