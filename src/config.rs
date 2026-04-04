//! Configuration types for TTS models.
//!
//! Provides a builder-pattern API for specifying model files individually
//! (for custom download managers) or by directory, with HuggingFace Hub
//! download as a fallback.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_nn::VarBuilder;
use tracing::info;

use crate::device::DeviceSelection;
use crate::error::TtsError;
use crate::models::ModelType;

// ---------------------------------------------------------------------------
// ModelFiles — per-file path storage
// ---------------------------------------------------------------------------

/// Resolved file paths for model loading.
///
/// Each model type requires a specific set of files. You can provide them
/// individually using the builder methods on [`TtsConfig`], set
/// [`TtsConfig::model_path`] to a directory that contains all of them, or
/// rely on automatic HuggingFace Hub download (if the `download` feature
/// is enabled).
///
/// ## File resolution order (per file)
///
/// 1. **Explicit path** — set via `with_*_file()` / `with_*_dir()` on
///    [`TtsConfig`]. Use this when your project has its own download
///    manager (e.g. flow-like hash-based local caching).
/// 2. **Auto-discovery** — if `model_path` is set, the library looks for
///    well-known filenames inside that directory.
/// 3. **HuggingFace Hub download** — if the `download` feature is enabled
///    and the file is still missing, it is fetched from the Hub. This is
///    the convenient fallback for quick prototyping.
#[derive(Debug, Clone, Default)]
pub struct ModelFiles {
    // ── Shared across all models ──────────────────────────────────────
    /// Path to **`config.json`** — model architecture configuration.
    ///
    /// **Expected format:** JSON object describing the neural-network
    /// hyperparameters (hidden size, number of layers, vocab size, …).
    /// This is the standard HuggingFace `config.json` format.
    /// Each backend stores its architecture metadata here, such as
    /// transformer dimensions, tokenizer sizes, sample rates, or
    /// auxiliary decoder configuration.
    pub config: Option<PathBuf>,

    /// Path to **`tokenizer.json`** — BPE text tokenizer definition.
    ///
    /// **Expected format:** [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers)
    /// self-contained JSON file. Contains the full vocabulary, merge rules,
    /// special tokens, and pre/post-processing steps. No separate
    /// `vocab.json` or `merges.txt` required when this file is present.
    ///
    /// Used by both models to convert input text into token IDs before
    /// feeding them to the transformer backbone.
    pub tokenizer: Option<PathBuf>,

    /// Paths to **model weight files** (`.safetensors`).
    ///
    /// **Expected format:** One or more [SafeTensors](https://huggingface.co/docs/safetensors)
    /// files containing the neural-network parameters.
    ///
    /// * **Single file** — `model.safetensors` (for models < ~5 GB).
    /// * **Sharded** — `model-00001-of-00004.safetensors`, … When
    ///   sharded, the library also expects `model.safetensors.index.json`
    ///   in the same directory (auto-discovered or downloaded).
    /// * **Other formats** — some backends use `consolidated.safetensors`
    ///   or `.pth` files instead of the standard filename.
    pub weights: Vec<PathBuf>,

    // ── Voice asset directories ───────────────────────────────────────
    /// Path to a **voice asset directory** for backends that ship preset voices.
    ///
    /// Supported layouts include:
    ///
    /// ```text
    /// voices/                ← Kokoro preset voices (`*.pt`)
    /// voice_embedding/       ← Voxtral preset voices (`*.pt`)
    /// ```
    ///
    /// The exact file format depends on the backend.
    pub voices_dir: Option<PathBuf>,

    // ── Qwen3-TTS / OmniVoice-specific ────────────────────────────────
    /// Paths to the **speech/audio tokenizer decoder** weight files.
    ///
    /// **Expected format:** SafeTensors files for the auxiliary decoder used
    /// by models that emit discrete audio codec tokens.
    ///
    /// Contains:
    /// * Residual VQ codebooks (16 groups × 2048 codes × dim)
    /// * Pre-conv + pre-transformer layers
    /// * Upsampling layers (transposed convolutions + SnakeBeta)
    /// * Final decoder convolution
    ///
    /// * **Qwen3-TTS** uses the separate
    ///   `Qwen/Qwen3-TTS-Tokenizer-12Hz` repository.
    /// * **OmniVoice** uses the `audio_tokenizer/` subdirectory inside the
    ///   main model snapshot.
    pub speech_tokenizer_weights: Vec<PathBuf>,

    /// Path to **`config.json`** of the speech/audio tokenizer.
    ///
    /// **Expected format:** JSON config for the speech tokenizer decoder
    /// model, including codebook dimensions, upsampling ratios, and
    /// activation parameters.
    ///
    /// If not provided, will be auto-discovered from a nested
    /// `audio_tokenizer/` directory or downloaded from HuggingFace.
    pub speech_tokenizer_config: Option<PathBuf>,

    /// Path to **`generation_config.json`** (optional).
    ///
    /// **Expected format:** Standard HuggingFace generation configuration
    /// with fields like `max_new_tokens`, `top_p`, `temperature`,
    /// `do_sample`, `repetition_penalty`, etc.
    ///
    /// If not provided, sensible per-model defaults are used.
    pub generation_config: Option<PathBuf>,

    /// Path to **`preprocessor_config.json`** (optional).
    ///
    /// Used by backends such as VibeVoice that publish prompt-building and
    /// audio-normalization defaults separately from `config.json`.
    pub preprocessor_config: Option<PathBuf>,
}

impl ModelFiles {
    /// Scan a directory for well-known model files and fill any that are
    /// still `None` / empty.
    pub fn fill_from_directory(&mut self, dir: &Path) {
        // config.json
        if self.config.is_none() {
            let p = dir.join("config.json");
            if p.exists() {
                info!("Auto-discovered config: {}", p.display());
                self.config = Some(p);
            } else {
                let p = dir.join("params.json");
                if p.exists() {
                    info!("Auto-discovered config: {}", p.display());
                    self.config = Some(p);
                }
            }
        }

        // tokenizer.json
        if self.tokenizer.is_none() {
            let p = dir.join("tokenizer.json");
            if p.exists() {
                info!("Auto-discovered tokenizer: {}", p.display());
                self.tokenizer = Some(p);
            } else {
                let p = dir.join("tekken.json");
                if p.exists() {
                    info!("Auto-discovered tokenizer: {}", p.display());
                    self.tokenizer = Some(p);
                }
            }
        }

        // Model weights
        if self.weights.is_empty() {
            let single = dir.join("model.safetensors");
            if single.exists() {
                info!("Auto-discovered single weight file");
                self.weights.push(single);
            } else {
                let single = dir.join("consolidated.safetensors");
                if single.exists() {
                    info!("Auto-discovered single weight file");
                    self.weights.push(single);
                } else {
                    self.discover_sharded_weights(dir);
                }
            }
            // Fall back to .pth files (Kokoro uses PyTorch .pth format)
            if self.weights.is_empty() {
                self.discover_pth_weights(dir);
            }
        }

        // Voice asset directory (Kokoro / Voxtral)
        if self.voices_dir.is_none() {
            let p = dir.join("voices");
            if p.is_dir() {
                info!("Auto-discovered voices dir: {}", p.display());
                self.voices_dir = Some(p);
            } else {
                let p = dir.join("voice_embedding");
                if p.is_dir() {
                    info!("Auto-discovered voices dir: {}", p.display());
                    self.voices_dir = Some(p);
                }
            }
        }

        // generation_config.json
        if self.generation_config.is_none() {
            let p = dir.join("generation_config.json");
            if p.exists() {
                info!("Auto-discovered generation config: {}", p.display());
                self.generation_config = Some(p);
            }
        }

        if self.preprocessor_config.is_none() {
            let p = dir.join("preprocessor_config.json");
            if p.exists() {
                info!("Auto-discovered preprocessor config: {}", p.display());
                self.preprocessor_config = Some(p);
            }
        }

        // Nested audio_tokenizer/ assets (OmniVoice)
        let audio_tokenizer_dir = dir.join("audio_tokenizer");
        if audio_tokenizer_dir.is_dir() {
            if self.speech_tokenizer_config.is_none() {
                let p = audio_tokenizer_dir.join("config.json");
                if p.exists() {
                    info!("Auto-discovered audio tokenizer config: {}", p.display());
                    self.speech_tokenizer_config = Some(p);
                }
            }

            if self.speech_tokenizer_weights.is_empty() {
                let single = audio_tokenizer_dir.join("model.safetensors");
                if single.exists() {
                    info!("Auto-discovered audio tokenizer weight file");
                    self.speech_tokenizer_weights.push(single);
                } else {
                    let mut shards = Self::discover_sharded_weights_in_dir(&audio_tokenizer_dir);
                    if !shards.is_empty() {
                        info!(
                            "Auto-discovered {} audio tokenizer weight shards",
                            shards.len()
                        );
                        self.speech_tokenizer_weights.append(&mut shards);
                    }
                }
            }
        }
    }

    /// Look for `.pth` PyTorch weight files (e.g. Kokoro's kokoro-v1_0.pth).
    fn discover_pth_weights(&mut self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };

        let mut pth_files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext == "pth")
            })
            .collect();

        if !pth_files.is_empty() {
            pth_files.sort();
            info!("Auto-discovered {} .pth weight file(s)", pth_files.len());
            self.weights = pth_files;
        }
    }

    /// Look for `model-NNNNN-of-NNNNN.safetensors` shard files.
    fn discover_sharded_weights(&mut self, dir: &Path) {
        let shards = Self::discover_sharded_weights_in_dir(dir);

        if !shards.is_empty() {
            info!("Auto-discovered {} weight shards", shards.len());
            self.weights = shards;
        }
    }

    fn discover_sharded_weights_in_dir(dir: &Path) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };

        let mut shards: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
            })
            .collect();
        shards.sort();
        shards
    }

    /// Build a [`VarBuilder`] by reading safetensors files fully into memory.
    ///
    /// This is the **safe** alternative to `VarBuilder::from_mmaped_safetensors`
    /// which requires `unsafe` due to memory-mapping. The trade-off is a brief
    /// peak in memory while the raw bytes and parsed tensors coexist, but for
    /// model loading this is negligible compared to the final tensor footprint.
    pub fn load_safetensors_vb(
        paths: &[PathBuf],
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> Result<VarBuilder<'static>, TtsError> {
        if paths.is_empty() {
            return Err(TtsError::FileMissing("safetensors weight files".into()));
        }

        // Single-file fast path
        if paths.len() == 1 {
            let data = std::fs::read(&paths[0]).map_err(|e| {
                TtsError::WeightLoadError(format!("Failed to read {}: {}", paths[0].display(), e))
            })?;
            return VarBuilder::from_buffered_safetensors(data, dtype, device)
                .map_err(|e| TtsError::WeightLoadError(e.to_string()));
        }

        // Multi-file: read each shard, collect all tensors into one HashMap
        let mut all_tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
        for path in paths {
            let data = std::fs::read(path).map_err(|e| {
                TtsError::WeightLoadError(format!("Failed to read {}: {}", path.display(), e))
            })?;
            let tensors = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
                TtsError::WeightLoadError(format!("Failed to parse {}: {}", path.display(), e))
            })?;
            for (name, view) in tensors.tensors() {
                // Get the native dtype from the safetensors file
                let native_dtype = match view.dtype() {
                    safetensors::Dtype::F16 => candle_core::DType::F16,
                    safetensors::Dtype::BF16 => candle_core::DType::BF16,
                    safetensors::Dtype::F32 => candle_core::DType::F32,
                    safetensors::Dtype::F64 => candle_core::DType::F64,
                    safetensors::Dtype::I64 => candle_core::DType::I64,
                    safetensors::Dtype::I32 => candle_core::DType::I64, // candle has no I32
                    safetensors::Dtype::U32 => candle_core::DType::U32,
                    safetensors::Dtype::U8 => candle_core::DType::U8,
                    _ => candle_core::DType::F32, // Fallback
                };

                // Load in native dtype first
                let tensor = candle_core::Tensor::from_raw_buffer(
                    view.data(),
                    native_dtype,
                    view.shape(),
                    device,
                )
                .map_err(|e| {
                    TtsError::WeightLoadError(format!("Failed to load tensor '{}': {}", name, e))
                })?;

                // Convert to target dtype if different
                let tensor = if native_dtype != dtype {
                    tensor.to_dtype(dtype).map_err(|e| {
                        TtsError::WeightLoadError(format!(
                            "Failed to convert tensor '{}' to {:?}: {}",
                            name, dtype, e
                        ))
                    })?
                } else {
                    tensor
                };

                all_tensors.insert(name, tensor);
            }
        }

        Ok(VarBuilder::from_tensors(all_tensors, dtype, device))
    }

    /// Download missing files from HuggingFace Hub.
    ///
    /// `model_type` determines which files are required.
    #[cfg(feature = "download")]
    pub fn fill_from_hf(&mut self, model_id: &str, model_type: ModelType) -> Result<(), TtsError> {
        use crate::download::download_file;

        // config
        if self.config.is_none() {
            let config_name = if model_type == ModelType::Voxtral {
                "params.json"
            } else {
                "config.json"
            };
            info!("Downloading {} from {}", config_name, model_id);
            self.config = Some(download_file(model_id, config_name)?);
        }

        // tokenizer.json (Kokoro uses phoneme vocab from config.json instead)
        if model_type != ModelType::Kokoro && self.tokenizer.is_none() {
            let tokenizer_name = if model_type == ModelType::Voxtral {
                "tekken.json"
            } else {
                "tokenizer.json"
            };
            info!("Downloading {} from {}", tokenizer_name, model_id);
            match download_file(model_id, tokenizer_name) {
                Ok(p) => self.tokenizer = Some(p),
                Err(_) => {
                    if model_type == ModelType::Voxtral {
                        return Err(TtsError::FileMissing(
                            "tekken.json — Voxtral Tekken tokenizer".to_string(),
                        ));
                    }
                    let fallback_repo = match model_type {
                        ModelType::Qwen3Tts => "Qwen/Qwen2.5-0.5B",
                        ModelType::VibeVoice => "Qwen/Qwen2.5-1.5B",
                        _ => "Qwen/Qwen2.5-0.5B",
                    };
                    info!(
                        "tokenizer.json not in {}; falling back to {}",
                        model_id, fallback_repo
                    );
                    self.tokenizer = Some(download_file(fallback_repo, "tokenizer.json")?);
                }
            }
        }

        // generation_config.json (optional — ignore download errors)
        if self.generation_config.is_none() {
            if let Ok(p) = download_file(model_id, "generation_config.json") {
                self.generation_config = Some(p);
            }
        }

        if self.preprocessor_config.is_none() {
            if let Ok(p) = download_file(model_id, "preprocessor_config.json") {
                self.preprocessor_config = Some(p);
            }
        }

        // Model weights
        if self.weights.is_empty() {
            self.download_weights_from_hf(model_id)?;
        }

        // Model-specific extras
        match model_type {
            ModelType::Kokoro => {
                self.download_kokoro_extras(model_id)?;
            }
            ModelType::OmniVoice => {
                self.download_omnivoice_extras(model_id)?;
            }
            ModelType::Voxtral => {
                self.download_voxtral_extras(model_id)?;
            }
            ModelType::Qwen3Tts => {
                self.download_qwen3tts_extras()?;
            }
            ModelType::VibeVoice => {
                self.download_vibevoice_extras(model_id)?;
            }
        }

        Ok(())
    }

    /// Download weight files (single or sharded) from HuggingFace.
    #[cfg(feature = "download")]
    fn download_weights_from_hf(&mut self, model_id: &str) -> Result<(), TtsError> {
        use crate::download::download_file;

        // Try single safetensors file first
        if let Ok(p) = download_file(model_id, "model.safetensors") {
            self.weights.push(p);
            return Ok(());
        }

        // Voxtral ships a single consolidated.safetensors file.
        if let Ok(p) = download_file(model_id, "consolidated.safetensors") {
            self.weights.push(p);
            return Ok(());
        }

        // Try .pth file (Kokoro uses PyTorch format)
        for pth_name in &["kokoro-v1_0.pth", "kokoro-v1_1-zh.pth", "model.pth"] {
            if let Ok(p) = download_file(model_id, pth_name) {
                self.weights.push(p);
                return Ok(());
            }
        }

        // Fall back to sharded — download the index
        let index_path = download_file(model_id, "model.safetensors.index.json")?;
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;

        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
            let mut shard_names: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shard_names.sort();
            shard_names.dedup();

            for shard_name in &shard_names {
                info!("Downloading shard: {}", shard_name);
                let p = download_file(model_id, shard_name)?;
                self.weights.push(p);
            }
        }

        Ok(())
    }

    /// Download Kokoro-specific files (voices directory).
    #[cfg(feature = "download")]
    fn download_kokoro_extras(&mut self, model_id: &str) -> Result<(), TtsError> {
        use crate::download::download_file;

        if self.voices_dir.is_none() {
            // Download a well-known voice to discover the voices directory
            if let Ok(voice_path) = download_file(model_id, "voices/af_heart.pt") {
                if let Some(parent) = voice_path.parent() {
                    self.voices_dir = Some(parent.to_path_buf());
                }
            }
        }

        Ok(())
    }

    /// Download Qwen3-TTS-specific files (speech tokenizer from separate repo).
    #[cfg(feature = "download")]
    fn download_qwen3tts_extras(&mut self) -> Result<(), TtsError> {
        use crate::download::download_file;

        let tokenizer_repo = "Qwen/Qwen3-TTS-Tokenizer-12Hz";

        if self.speech_tokenizer_config.is_none() {
            info!(
                "Downloading speech tokenizer config from {}",
                tokenizer_repo
            );
            if let Ok(p) = download_file(tokenizer_repo, "config.json") {
                self.speech_tokenizer_config = Some(p);
            }
        }

        if self.speech_tokenizer_weights.is_empty() {
            info!(
                "Downloading speech tokenizer weights from {}",
                tokenizer_repo
            );
            if let Ok(p) = download_file(tokenizer_repo, "model.safetensors") {
                self.speech_tokenizer_weights.push(p);
            } else if let Ok(index_path) =
                download_file(tokenizer_repo, "model.safetensors.index.json")
            {
                if let Ok(content) = std::fs::read_to_string(&index_path) {
                    if let Ok(index) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(weight_map) =
                            index.get("weight_map").and_then(|v| v.as_object())
                        {
                            let mut shard_names: Vec<String> = weight_map
                                .values()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect();
                            shard_names.sort();
                            shard_names.dedup();

                            for shard_name in &shard_names {
                                if let Ok(p) = download_file(tokenizer_repo, shard_name) {
                                    self.speech_tokenizer_weights.push(p);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(feature = "download")]
    fn download_vibevoice_extras(&mut self, model_id: &str) -> Result<(), TtsError> {
        use crate::download::download_file;

        if self.preprocessor_config.is_none() {
            if let Ok(p) = download_file(model_id, "preprocessor_config.json") {
                self.preprocessor_config = Some(p);
            }
        }

        Ok(())
    }

    /// Download OmniVoice-specific files (audio tokenizer subdirectory).
    #[cfg(feature = "download")]
    fn download_omnivoice_extras(&mut self, model_id: &str) -> Result<(), TtsError> {
        use crate::download::download_file;

        if self.speech_tokenizer_config.is_none() {
            if let Ok(p) = download_file(model_id, "audio_tokenizer/config.json") {
                self.speech_tokenizer_config = Some(p);
            }
        }

        if self.speech_tokenizer_weights.is_empty() {
            if let Ok(p) = download_file(model_id, "audio_tokenizer/model.safetensors") {
                self.speech_tokenizer_weights.push(p);
            } else if let Ok(index_path) =
                download_file(model_id, "audio_tokenizer/model.safetensors.index.json")
            {
                if let Ok(content) = std::fs::read_to_string(&index_path) {
                    if let Ok(index) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(weight_map) =
                            index.get("weight_map").and_then(|v| v.as_object())
                        {
                            let mut shard_names: Vec<String> = weight_map
                                .values()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect();
                            shard_names.sort();
                            shard_names.dedup();

                            for shard_name in &shard_names {
                                let shard_path = format!("audio_tokenizer/{}", shard_name);
                                if let Ok(p) = download_file(model_id, &shard_path) {
                                    self.speech_tokenizer_weights.push(p);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Download Voxtral-specific files (preset voice embeddings).
    #[cfg(feature = "download")]
    fn download_voxtral_extras(&mut self, model_id: &str) -> Result<(), TtsError> {
        use crate::download::download_file;

        if self.voices_dir.is_some() {
            return Ok(());
        }

        let config_path = self.config.as_ref().ok_or_else(|| {
            TtsError::FileMissing("params.json — Voxtral model configuration".to_string())
        })?;
        let content = std::fs::read_to_string(config_path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;
        let voices = config
            .get("multimodal")
            .and_then(|v| v.get("audio_tokenizer_args"))
            .and_then(|v| v.get("voice"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                TtsError::ConfigError(
                    "params.json is missing multimodal.audio_tokenizer_args.voice".to_string(),
                )
            })?;

        let mut discovered_dir: Option<PathBuf> = None;
        for voice_name in voices.keys() {
            let path = download_file(model_id, &format!("voice_embedding/{voice_name}.pt"))?;
            if discovered_dir.is_none() {
                discovered_dir = path.parent().map(Path::to_path_buf);
            }
        }

        self.voices_dir = discovered_dir;
        Ok(())
    }

    /// Check whether all required files for the given model type are present.
    pub fn validate(&self, model_type: ModelType) -> Result<(), TtsError> {
        if model_type == ModelType::Voxtral {
            if self.config.is_none() {
                return Err(TtsError::FileMissing(
                    "params.json — Voxtral model configuration".to_string(),
                ));
            }
            if self.tokenizer.is_none() {
                return Err(TtsError::FileMissing(
                    "tekken.json — Voxtral Tekken tokenizer".to_string(),
                ));
            }
            if self.weights.is_empty() {
                return Err(TtsError::FileMissing(
                    "consolidated.safetensors — Voxtral model weights".to_string(),
                ));
            }
            if self.voices_dir.is_none() {
                return Err(TtsError::FileMissing(
                    "voice_embedding/ — Voxtral preset voice embeddings".to_string(),
                ));
            }
            return Ok(());
        }

        if self.config.is_none() {
            return Err(TtsError::FileMissing(
                "config.json — model architecture configuration".to_string(),
            ));
        }
        // Kokoro uses phoneme vocab from config.json, no tokenizer.json needed
        if model_type != ModelType::Kokoro && self.tokenizer.is_none() {
            return Err(TtsError::FileMissing(
                "tokenizer.json — BPE text tokenizer".to_string(),
            ));
        }
        if self.weights.is_empty() {
            return Err(TtsError::FileMissing(
                "model weight files (.safetensors or .pth)".to_string(),
            ));
        }

        match model_type {
            ModelType::OmniVoice => {
                if self.speech_tokenizer_config.is_none() {
                    return Err(TtsError::FileMissing(
                        "audio tokenizer config (audio_tokenizer/config.json) \
                         — configures OmniVoice's codec decoder"
                            .to_string(),
                    ));
                }
                if self.speech_tokenizer_weights.is_empty() {
                    return Err(TtsError::FileMissing(
                        "audio tokenizer weights (audio_tokenizer/model.safetensors) \
                         — converts OmniVoice codec tokens to audio waveform"
                            .to_string(),
                    ));
                }
            }
            ModelType::Qwen3Tts => {
                if self.speech_tokenizer_weights.is_empty() {
                    return Err(TtsError::FileMissing(
                        "speech tokenizer weights (Qwen3-TTS-Tokenizer-12Hz model.safetensors) \
                         — converts codec tokens to audio waveform"
                            .to_string(),
                    ));
                }
            }
            ModelType::Kokoro => {
                // voices_dir is optional
            }
            ModelType::VibeVoice => {}
            ModelType::Voxtral => unreachable!(),
        }

        Ok(())
    }

    /// Return the list of files that are required but not yet set.
    pub fn missing_files(&self, model_type: ModelType) -> Vec<&'static str> {
        if model_type == ModelType::Voxtral {
            let mut missing = Vec::new();
            if self.config.is_none() {
                missing.push("params.json");
            }
            if self.tokenizer.is_none() {
                missing.push("tekken.json");
            }
            if self.weights.is_empty() {
                missing.push("consolidated.safetensors");
            }
            if self.voices_dir.is_none() {
                missing.push("voice_embedding");
            }
            return missing;
        }

        let mut missing = Vec::new();

        if self.config.is_none() {
            missing.push("config.json");
        }
        if model_type != ModelType::Kokoro && self.tokenizer.is_none() {
            missing.push("tokenizer.json");
        }
        if self.weights.is_empty() {
            missing.push("model weight files");
        }
        if model_type == ModelType::OmniVoice && self.speech_tokenizer_config.is_none() {
            missing.push("audio tokenizer config");
        }
        if model_type == ModelType::OmniVoice && self.speech_tokenizer_weights.is_empty() {
            missing.push("audio tokenizer weights");
        }
        if model_type == ModelType::Qwen3Tts && self.speech_tokenizer_weights.is_empty() {
            missing.push("speech tokenizer weights");
        }

        missing
    }
}

// ---------------------------------------------------------------------------
// DType
// ---------------------------------------------------------------------------

/// Floating-point data type for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    /// 32-bit float — maximum compatibility, highest memory.
    F32,
    /// 16-bit float — good balance.
    F16,
    /// Brain float 16 — preferred for transformer models.
    #[default]
    BF16,
}

impl DType {
    /// Convert to candle's DType.
    pub fn to_candle(self) -> candle_core::DType {
        match self {
            Self::F32 => candle_core::DType::F32,
            Self::F16 => candle_core::DType::F16,
            Self::BF16 => candle_core::DType::BF16,
        }
    }
}

// ---------------------------------------------------------------------------
// TtsConfig — main configuration + builder
// ---------------------------------------------------------------------------

/// Top-level configuration for loading a TTS model.
///
/// # Providing model files
///
/// There are three ways to tell tts-rs where to find the model files,
/// listed from highest to lowest priority:
///
/// 1. **Individual file paths** (for custom download managers):
///    ```rust
///    # use tts_rs::{TtsConfig, ModelType};
///    let config = TtsConfig::new(ModelType::Qwen3Tts)
///        .with_config_file("/cache/sha256-abc/config.json")
///        .with_tokenizer_file("/cache/sha256-def/tokenizer.json")
///        .with_weight_file("/cache/sha256-012/model.safetensors");
///    ```
///
/// 2. **Directory path** (all files in one folder):
///    ```rust
///    # use tts_rs::{TtsConfig, ModelType};
///    let config = TtsConfig::new(ModelType::Qwen3Tts)
///        .with_model_path("/models/qwen3-tts");
///    ```
///
/// 3. **HuggingFace Hub download** (automatic fallback):
///    ```rust
///    # use tts_rs::{TtsConfig, ModelType};
///    let config = TtsConfig::new(ModelType::Qwen3Tts); // downloads automatically
///    ```
///
/// These can be mixed: set some files explicitly and let the rest be
/// auto-discovered or downloaded.
#[derive(Debug, Clone)]
pub struct TtsConfig {
    /// Which model backend to use.
    pub model_type: ModelType,

    /// Path to a local directory containing model weights and config.
    /// Files inside are auto-discovered by well-known filenames.
    ///
    /// Some backends may also accept a local model directory instead of a
    /// HuggingFace model ID.
    pub model_path: Option<String>,

    /// HuggingFace model ID (e.g. `"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"`).
    /// Used as download fallback when files are not found locally.
    pub hf_model_id: Option<String>,

    /// Override for an external runtime command when a backend needs one.
    pub runtime_command: Option<String>,

    /// Override for an external runtime endpoint when a backend needs one.
    pub runtime_endpoint: Option<String>,

    /// Bearer token used for external HTTP runtimes.
    pub bearer_token: Option<String>,

    /// Device selection strategy.
    pub device: DeviceSelection,

    /// Data type for model weights. Defaults to BFloat16 where supported.
    pub dtype: DType,

    /// Individually specified model files.
    pub files: ModelFiles,
}

impl TtsConfig {
    /// Create a new configuration for the specified model type.
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            model_path: None,
            hf_model_id: None,
            runtime_command: None,
            runtime_endpoint: None,
            bearer_token: None,
            device: DeviceSelection::Auto,
            dtype: DType::default(),
            files: ModelFiles::default(),
        }
    }

    // ── Directory / HF shortcuts ──────────────────────────────────────

    /// Set the local directory containing all model files.
    ///
    /// The directory will be scanned for well-known filenames
    /// (`config.json`, `tokenizer.json`, `model.safetensors`, …).
    pub fn with_model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the HuggingFace model ID for automatic download.
    ///
    /// Only used as a fallback when files cannot be found locally.
    pub fn with_hf_model_id(mut self, id: impl Into<String>) -> Self {
        self.hf_model_id = Some(id.into());
        self
    }

    /// Override the executable used by runtime-adapter backends.
    pub fn with_runtime_command(mut self, command: impl Into<String>) -> Self {
        self.runtime_command = Some(command.into());
        self
    }

    /// Override the HTTP endpoint used by runtime-adapter backends.
    pub fn with_runtime_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.runtime_endpoint = Some(endpoint.into());
        self
    }

    /// Set the bearer token used by HTTP runtime adapters.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    // ── Device / dtype ────────────────────────────────────────────────

    /// Set the device selection strategy.
    pub fn with_device(mut self, device: DeviceSelection) -> Self {
        self.device = device;
        self
    }

    /// Set the data type for model weights.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    // ── Individual file builders ──────────────────────────────────────

    /// Set the path to **`config.json`**.
    ///
    /// This JSON file describes the model architecture: hidden size,
    /// number of layers, vocabulary size, attention head counts, etc.
    /// It follows the standard HuggingFace config format.
    pub fn with_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.config = Some(path.into());
        self
    }

    /// Set the path to **`tokenizer.json`**.
    ///
    /// A self-contained HuggingFace Tokenizers JSON file with the full
    /// BPE vocabulary, merge rules, and special-token definitions.
    /// Both model backends use this to convert input text to token IDs.
    pub fn with_tokenizer_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.tokenizer = Some(path.into());
        self
    }

    /// Append a single **model weight file** (`.safetensors`).
    ///
    /// Call this repeatedly when you need to provide several shards
    /// explicitly, or once for single-file models:
    ///
    /// ```rust
    /// # use tts_rs::{TtsConfig, ModelType};
    /// let config = TtsConfig::new(ModelType::Qwen3Tts)
    ///     .with_weight_file("/cache/model.safetensors");
    /// ```
    pub fn with_weight_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.weights.push(path.into());
        self
    }

    /// Set **all model weight files** at once, replacing any previously added.
    pub fn with_weight_files(mut self, paths: Vec<PathBuf>) -> Self {
        self.files.weights = paths;
        self
    }

    /// Set a **voice asset directory** for backends that use preset voices.
    ///
    /// This is used by backends such as Kokoro (`voices/*.pt`) and
    /// Voxtral (`voice_embedding/*.pt`).
    pub fn with_voices_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.voices_dir = Some(path.into());
        self
    }

    /// Append a single **speech tokenizer weight file** (Qwen3-TTS only).
    ///
    /// These weights belong to the separate speech tokenizer decoder
    /// model (`Qwen/Qwen3-TTS-Tokenizer-12Hz`) that converts discrete
    /// codec tokens into a continuous 24 kHz audio waveform.
    pub fn with_speech_tokenizer_weight_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.speech_tokenizer_weights.push(path.into());
        self
    }

    /// Set **all speech tokenizer weight files** at once (Qwen3-TTS only).
    pub fn with_speech_tokenizer_weight_files(mut self, paths: Vec<PathBuf>) -> Self {
        self.files.speech_tokenizer_weights = paths;
        self
    }

    /// Set the **speech tokenizer config** file (Qwen3-TTS only).
    ///
    /// JSON config for the speech tokenizer decoder model, including
    /// codebook dimensions, upsampling ratios, and activation parameters.
    pub fn with_speech_tokenizer_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.speech_tokenizer_config = Some(path.into());
        self
    }

    /// Set the **generation config** file (optional).
    ///
    /// Standard HuggingFace `generation_config.json` with parameters
    /// like `max_new_tokens`, `top_p`, `temperature`, `do_sample`, etc.
    pub fn with_generation_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.generation_config = Some(path.into());
        self
    }

    /// Set the **preprocessor config** file (optional).
    ///
    /// This stores published preprocessing defaults such as audio
    /// normalization parameters and speech token compression ratios.
    pub fn with_preprocessor_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.preprocessor_config = Some(path.into());
        self
    }

    // ── Resolution ────────────────────────────────────────────────────

    /// Resolve all model files using the three-tier strategy:
    ///
    /// 1. Explicit paths already set on `self.files`
    /// 2. Auto-discovery from `self.model_path` directory
    /// 3. HuggingFace Hub download (if `download` feature enabled)
    ///
    /// Returns a fully populated [`ModelFiles`] or an error listing
    /// which files are missing.
    pub fn resolve_files(&self) -> Result<ModelFiles, TtsError> {
        let mut files = self.files.clone();

        // Tier 2: fill from model_path directory
        if let Some(ref dir) = self.model_path {
            files.fill_from_directory(Path::new(dir));
        }

        // Tier 3: HuggingFace Hub download fallback
        #[cfg(feature = "download")]
        {
            if !files.missing_files(self.model_type).is_empty() {
                let hf_id = self.effective_hf_model_id();
                info!("Downloading missing files from HuggingFace: {}", hf_id);
                files.fill_from_hf(hf_id, self.model_type)?;
            }
        }

        // Validate completeness
        files.validate(self.model_type)?;

        Ok(files)
    }

    /// Get the default HuggingFace model ID for this model type.
    pub fn default_hf_model_id(&self) -> &str {
        match self.model_type {
            ModelType::Kokoro => "hexgrad/Kokoro-82M",
            ModelType::OmniVoice => "k2-fsa/OmniVoice",
            ModelType::Qwen3Tts => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            ModelType::VibeVoice => "microsoft/VibeVoice-1.5B",
            ModelType::Voxtral => "mistralai/Voxtral-4B-TTS-2603",
        }
    }

    /// Resolve the effective HuggingFace model ID.
    pub fn effective_hf_model_id(&self) -> &str {
        self.hf_model_id
            .as_deref()
            .unwrap_or_else(|| self.default_hf_model_id())
    }

    /// Resolve the model reference to forward to an external runtime.
    pub fn effective_model_ref(&self) -> &str {
        self.model_path
            .as_deref()
            .unwrap_or_else(|| self.effective_hf_model_id())
    }

    /// Get the default runtime command for the configured model, if any.
    pub fn default_runtime_command(&self) -> Option<&str> {
        match self.model_type {
            ModelType::Voxtral => Some("python3"),
            ModelType::Kokoro | ModelType::OmniVoice | ModelType::Qwen3Tts | ModelType::VibeVoice => None,
        }
    }

    /// Resolve the effective runtime command for adapter backends.
    pub fn effective_runtime_command(&self) -> Option<&str> {
        self.runtime_command
            .as_deref()
            .or_else(|| self.default_runtime_command())
    }

    /// Get the default runtime endpoint for the configured model, if any.
    pub fn default_runtime_endpoint(&self) -> Option<&str> {
        match self.model_type {
            ModelType::Kokoro | ModelType::OmniVoice | ModelType::Qwen3Tts | ModelType::VibeVoice | ModelType::Voxtral => None,
        }
    }

    /// Resolve the effective runtime endpoint for adapter backends.
    pub fn effective_runtime_endpoint(&self) -> Option<&str> {
        self.runtime_endpoint
            .as_deref()
            .or_else(|| self.default_runtime_endpoint())
    }

    /// Resolve the bearer token used by external HTTP runtimes.
    pub fn effective_bearer_token(&self) -> &str {
        self.bearer_token.as_deref().unwrap_or("EMPTY")
    }
}
