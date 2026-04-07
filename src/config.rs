//! Configuration types for TTS models.
//!
//! Provides a builder-pattern API for specifying model files individually
//! (for custom download managers) or by directory, with HuggingFace Hub
//! download as a fallback.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_nn::VarBuilder;
use tracing::info;

use crate::device::DeviceSelection;
use crate::error::TtsError;
use crate::models::ModelType;

fn normalize_asset_path(path: impl AsRef<str>) -> String {
    path.as_ref()
        .replace('\\', "/")
        .trim_start_matches("./")
        .trim_start_matches('/')
        .to_string()
}

/// A single model asset that can come from disk or in-memory bytes.
#[derive(Debug, Clone)]
pub enum ModelAsset {
    Path(PathBuf),
    Bytes { name: String, data: Arc<[u8]> },
}

impl ModelAsset {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    pub fn from_bytes(name: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        Self::Bytes {
            name: normalize_asset_path(name.into()),
            data: Arc::from(bytes.into()),
        }
    }

    pub fn as_path(&self) -> Option<&Path> {
        match self {
            Self::Path(path) => Some(path),
            Self::Bytes { .. } => None,
        }
    }

    pub fn file_name(&self) -> Option<&str> {
        match self {
            Self::Path(path) => path.file_name().and_then(|name| name.to_str()),
            Self::Bytes { name, .. } => Path::new(name).file_name().and_then(|value| value.to_str()),
        }
    }

    pub fn extension(&self) -> Option<&str> {
        match self {
            Self::Path(path) => path.extension().and_then(|ext| ext.to_str()),
            Self::Bytes { name, .. } => Path::new(name).extension().and_then(|ext| ext.to_str()),
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            Self::Path(path) => path.display().to_string(),
            Self::Bytes { name, .. } => name.clone(),
        }
    }

    pub fn read_bytes(&self) -> Result<Arc<[u8]>, TtsError> {
        match self {
            Self::Path(path) => std::fs::read(path)
                .map(Arc::from)
                .map_err(TtsError::from),
            Self::Bytes { data, .. } => Ok(data.clone()),
        }
    }
}

/// A logical model asset directory, backed either by the filesystem or by in-memory bytes.
#[derive(Debug, Clone)]
pub enum ModelAssetDir {
    Path(PathBuf),
    Bytes(BTreeMap<String, Arc<[u8]>>),
}

impl ModelAssetDir {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    pub fn from_bytes(entries: BTreeMap<String, Arc<[u8]>>) -> Self {
        Self::Bytes(entries)
    }

    pub fn load_file(&self, name: &str) -> Result<ModelAsset, TtsError> {
        match self {
            Self::Path(path) => {
                let full_path = path.join(name);
                if !full_path.exists() {
                    return Err(TtsError::FileMissing(format!(
                        "{} in {}",
                        name,
                        path.display()
                    )));
                }
                Ok(ModelAsset::from_path(full_path))
            }
            Self::Bytes(entries) => entries
                .get(name)
                .cloned()
                .map(|data| ModelAsset::Bytes {
                    name: name.to_string(),
                    data,
                })
                .ok_or_else(|| TtsError::FileMissing(name.to_string())),
        }
    }

    pub fn file_names(&self) -> Result<Vec<String>, TtsError> {
        match self {
            Self::Path(path) => {
                let mut names = Vec::new();
                for entry in std::fs::read_dir(path)? {
                    let entry = entry?;
                    let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                        continue;
                    };
                    names.push(name);
                }
                names.sort();
                Ok(names)
            }
            Self::Bytes(entries) => Ok(entries.keys().cloned().collect()),
        }
    }
}

/// A named collection of relative-path assets for byte-first loading.
#[derive(Debug, Clone, Default)]
pub struct ModelAssetBundle {
    entries: BTreeMap<String, Arc<[u8]>>,
}

impl ModelAssetBundle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_bytes(
        &mut self,
        relative_path: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> &mut Self {
        let relative_path = normalize_asset_path(relative_path.into());
        self.entries.insert(relative_path, Arc::from(bytes.into()));
        self
    }

    pub fn with_bytes(
        mut self,
        relative_path: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        self.insert_bytes(relative_path, bytes);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn get(&self, relative_path: &str) -> Option<ModelAsset> {
        let relative_path = normalize_asset_path(relative_path);
        self.entries.get(&relative_path).cloned().map(|data| ModelAsset::Bytes {
            name: relative_path,
            data,
        })
    }

    fn collect_directory(&self, prefix: &str) -> Option<ModelAssetDir> {
        let prefix = normalize_asset_path(prefix);
        let prefix = if prefix.ends_with('/') {
            prefix
        } else {
            format!("{prefix}/")
        };

        let mut entries = BTreeMap::new();
        for (path, data) in &self.entries {
            let Some(rest) = path.strip_prefix(&prefix) else {
                continue;
            };
            if rest.is_empty() || rest.contains('/') {
                continue;
            }
            entries.insert(rest.to_string(), data.clone());
        }

        if entries.is_empty() {
            None
        } else {
            Some(ModelAssetDir::from_bytes(entries))
        }
    }

    fn discover_sharded_weights(&self, prefix: &str) -> Vec<ModelAsset> {
        let prefix = normalize_asset_path(prefix);
        let prefix = if prefix.is_empty() {
            String::new()
        } else if prefix.ends_with('/') {
            prefix
        } else {
            format!("{prefix}/")
        };

        let mut shards = self
            .entries
            .iter()
            .filter_map(|(path, data)| {
                let rest = if prefix.is_empty() {
                    path.as_str()
                } else {
                    path.strip_prefix(&prefix)?
                };
                if rest.contains('/')
                    || !rest.starts_with("model-")
                    || !rest.ends_with(".safetensors")
                {
                    return None;
                }
                Some(ModelAsset::Bytes {
                    name: path.clone(),
                    data: data.clone(),
                })
            })
            .collect::<Vec<_>>();
        shards.sort_by_key(ModelAsset::display_name);
        shards
    }

    fn discover_pth_weights(&self, prefix: &str) -> Vec<ModelAsset> {
        let prefix = normalize_asset_path(prefix);
        let prefix = if prefix.is_empty() {
            String::new()
        } else if prefix.ends_with('/') {
            prefix
        } else {
            format!("{prefix}/")
        };

        let mut weights = self
            .entries
            .iter()
            .filter_map(|(path, data)| {
                let rest = if prefix.is_empty() {
                    path.as_str()
                } else {
                    path.strip_prefix(&prefix)?
                };
                if rest.contains('/') || !rest.ends_with(".pth") {
                    return None;
                }
                Some(ModelAsset::Bytes {
                    name: path.clone(),
                    data: data.clone(),
                })
            })
            .collect::<Vec<_>>();
        weights.sort_by_key(ModelAsset::display_name);
        weights
    }
}

// ---------------------------------------------------------------------------
// ModelFiles — resolved model asset storage
// ---------------------------------------------------------------------------

/// Resolved model assets for loading.
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
    pub config: Option<ModelAsset>,

    /// Path to **`tokenizer.json`** — BPE text tokenizer definition.
    ///
    /// **Expected format:** [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers)
    /// self-contained JSON file. Contains the full vocabulary, merge rules,
    /// special tokens, and pre/post-processing steps. No separate
    /// `vocab.json` or `merges.txt` required when this file is present.
    ///
    /// Used by both models to convert input text into token IDs before
    /// feeding them to the transformer backbone.
    pub tokenizer: Option<ModelAsset>,

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
    pub weights: Vec<ModelAsset>,

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
    pub voices_dir: Option<ModelAssetDir>,

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
    pub speech_tokenizer_weights: Vec<ModelAsset>,

    /// Path to **`config.json`** of the speech/audio tokenizer.
    ///
    /// **Expected format:** JSON config for the speech tokenizer decoder
    /// model, including codebook dimensions, upsampling ratios, and
    /// activation parameters.
    ///
    /// If not provided, will be auto-discovered from a nested
    /// `audio_tokenizer/` directory or downloaded from HuggingFace.
    pub speech_tokenizer_config: Option<ModelAsset>,

    /// Path to **`generation_config.json`** (optional).
    ///
    /// **Expected format:** Standard HuggingFace generation configuration
    /// with fields like `max_new_tokens`, `top_p`, `temperature`,
    /// `do_sample`, `repetition_penalty`, etc.
    ///
    /// If not provided, sensible per-model defaults are used.
    pub generation_config: Option<ModelAsset>,

    /// Path to **`preprocessor_config.json`** (optional).
    ///
    /// Used by backends such as VibeVoice that publish prompt-building and
    /// audio-normalization defaults separately from `config.json`.
    pub preprocessor_config: Option<ModelAsset>,
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
                self.config = Some(ModelAsset::from_path(p));
            } else {
                let p = dir.join("params.json");
                if p.exists() {
                    info!("Auto-discovered config: {}", p.display());
                    self.config = Some(ModelAsset::from_path(p));
                }
            }
        }

        // tokenizer.json
        if self.tokenizer.is_none() {
            let p = dir.join("tokenizer.json");
            if p.exists() {
                info!("Auto-discovered tokenizer: {}", p.display());
                self.tokenizer = Some(ModelAsset::from_path(p));
            } else {
                let p = dir.join("tekken.json");
                if p.exists() {
                    info!("Auto-discovered tokenizer: {}", p.display());
                    self.tokenizer = Some(ModelAsset::from_path(p));
                }
            }
        }

        // Model weights
        if self.weights.is_empty() {
            let single = dir.join("model.safetensors");
            if single.exists() {
                info!("Auto-discovered single weight file");
                self.weights.push(ModelAsset::from_path(single));
            } else {
                let single = dir.join("consolidated.safetensors");
                if single.exists() {
                    info!("Auto-discovered single weight file");
                    self.weights.push(ModelAsset::from_path(single));
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
                self.voices_dir = Some(ModelAssetDir::from_path(p));
            } else {
                let p = dir.join("voice_embedding");
                if p.is_dir() {
                    info!("Auto-discovered voices dir: {}", p.display());
                    self.voices_dir = Some(ModelAssetDir::from_path(p));
                }
            }
        }

        // generation_config.json
        if self.generation_config.is_none() {
            let p = dir.join("generation_config.json");
            if p.exists() {
                info!("Auto-discovered generation config: {}", p.display());
                self.generation_config = Some(ModelAsset::from_path(p));
            }
        }

        if self.preprocessor_config.is_none() {
            let p = dir.join("preprocessor_config.json");
            if p.exists() {
                info!("Auto-discovered preprocessor config: {}", p.display());
                self.preprocessor_config = Some(ModelAsset::from_path(p));
            }
        }

        for nested_dir_name in ["audio_tokenizer", "speech_tokenizer"] {
            let nested_dir = dir.join(nested_dir_name);
            if !nested_dir.is_dir() {
                continue;
            }

            if self.speech_tokenizer_config.is_none() {
                let p = nested_dir.join("config.json");
                if p.exists() {
                    info!(
                        "Auto-discovered {} config: {}",
                        nested_dir_name,
                        p.display()
                    );
                    self.speech_tokenizer_config = Some(ModelAsset::from_path(p));
                }
            }

            if self.speech_tokenizer_weights.is_empty() {
                let single = nested_dir.join("model.safetensors");
                if single.exists() {
                    info!("Auto-discovered {} weight file", nested_dir_name);
                    self.speech_tokenizer_weights
                        .push(ModelAsset::from_path(single));
                } else {
                    let mut shards = Self::discover_sharded_weights_in_dir(&nested_dir);
                    if !shards.is_empty() {
                        info!(
                            "Auto-discovered {} {} weight shards",
                            shards.len(),
                            nested_dir_name
                        );
                        self.speech_tokenizer_weights.append(&mut shards);
                    }
                }
            }
        }
    }

    /// Scan an in-memory asset bundle for well-known model files.
    pub fn fill_from_asset_bundle(&mut self, bundle: &ModelAssetBundle) {
        if self.config.is_none() {
            self.config = bundle.get("config.json").or_else(|| bundle.get("params.json"));
        }

        if self.tokenizer.is_none() {
            self.tokenizer = bundle
                .get("tokenizer.json")
                .or_else(|| bundle.get("tekken.json"));
        }

        if self.weights.is_empty() {
            if let Some(asset) = bundle.get("model.safetensors") {
                self.weights.push(asset);
            } else if let Some(asset) = bundle.get("consolidated.safetensors") {
                self.weights.push(asset);
            } else {
                self.weights = bundle.discover_sharded_weights("");
            }
            if self.weights.is_empty() {
                self.weights = bundle.discover_pth_weights("");
            }
        }

        if self.voices_dir.is_none() {
            self.voices_dir = bundle
                .collect_directory("voices")
                .or_else(|| bundle.collect_directory("voice_embedding"));
        }

        if self.generation_config.is_none() {
            self.generation_config = bundle.get("generation_config.json");
        }

        if self.preprocessor_config.is_none() {
            self.preprocessor_config = bundle.get("preprocessor_config.json");
        }

        for nested_dir_name in ["audio_tokenizer", "speech_tokenizer"] {
            if self.speech_tokenizer_config.is_none() {
                self.speech_tokenizer_config = bundle.get(format!("{nested_dir_name}/config.json").as_str());
            }

            if self.speech_tokenizer_weights.is_empty() {
                if let Some(asset) = bundle.get(format!("{nested_dir_name}/model.safetensors").as_str()) {
                    self.speech_tokenizer_weights.push(asset);
                } else {
                    self.speech_tokenizer_weights = bundle.discover_sharded_weights(nested_dir_name);
                }
            }
        }
    }

    /// Look for `.pth` PyTorch weight files (e.g. Kokoro's kokoro-v1_0.pth).
    fn discover_pth_weights(&mut self, dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };

        let mut pth_files: Vec<ModelAsset> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext == "pth")
            })
            .map(ModelAsset::from_path)
            .collect();

        if !pth_files.is_empty() {
            pth_files.sort_by_key(ModelAsset::display_name);
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

    fn discover_sharded_weights_in_dir(dir: &Path) -> Vec<ModelAsset> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };

        let mut shards: Vec<ModelAsset> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
            })
            .map(ModelAsset::from_path)
            .collect();
        shards.sort_by_key(ModelAsset::display_name);
        shards
    }

    /// Build a [`VarBuilder`] by reading safetensors files fully into memory.
    ///
    /// This is the **safe** alternative to `VarBuilder::from_mmaped_safetensors`
    /// which requires `unsafe` due to memory-mapping. The trade-off is a brief
    /// peak in memory while the raw bytes and parsed tensors coexist, but for
    /// model loading this is negligible compared to the final tensor footprint.
    pub fn load_safetensors_vb(
        assets: &[ModelAsset],
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> Result<VarBuilder<'static>, TtsError> {
        if assets.is_empty() {
            return Err(TtsError::FileMissing("safetensors weight files".into()));
        }

        if assets.len() == 1 {
            if let Some(path) = assets[0].as_path() {
                let data = std::fs::read(path).map_err(|e| {
                    TtsError::WeightLoadError(format!("Failed to read {}: {}", path.display(), e))
                })?;
                return VarBuilder::from_buffered_safetensors(data, dtype, device)
                    .map_err(|e| TtsError::WeightLoadError(e.to_string()));
            }
        }

        // Multi-file: read each shard, collect all tensors into one HashMap
        let mut all_tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
        for asset in assets {
            let data = asset.read_bytes().map_err(|e| {
                TtsError::WeightLoadError(format!(
                    "Failed to read {}: {}",
                    asset.display_name(),
                    e
                ))
            })?;
            let tensors = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
                TtsError::WeightLoadError(format!(
                    "Failed to parse {}: {}",
                    asset.display_name(),
                    e
                ))
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
    pub fn fill_from_hf(
        &mut self,
        model_id: &str,
        model_type: ModelType,
        bearer_token: Option<&str>,
    ) -> Result<(), TtsError> {
        use crate::download::download_file_with_token;

        let download = |repo: &str, file: &str| download_file_with_token(repo, file, bearer_token);

        // config
        if self.config.is_none() {
            let config_name = if model_type == ModelType::Voxtral {
                "params.json"
            } else {
                "config.json"
            };
            info!("Downloading {} from {}", config_name, model_id);
            self.config = Some(ModelAsset::from_path(download(model_id, config_name)?));
        }

        // tokenizer.json (Kokoro uses phoneme vocab from config.json instead)
        if model_type != ModelType::Kokoro && self.tokenizer.is_none() {
            let tokenizer_name = if model_type == ModelType::Voxtral {
                "tekken.json"
            } else {
                "tokenizer.json"
            };
            info!("Downloading {} from {}", tokenizer_name, model_id);
            match download(model_id, tokenizer_name) {
                Ok(p) => self.tokenizer = Some(ModelAsset::from_path(p)),
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
                    self.tokenizer = Some(ModelAsset::from_path(download(
                        fallback_repo,
                        "tokenizer.json",
                    )?));
                }
            }
        }

        // generation_config.json (optional — ignore download errors)
        if self.generation_config.is_none() {
            if let Ok(p) = download(model_id, "generation_config.json") {
                self.generation_config = Some(ModelAsset::from_path(p));
            }
        }

        if self.preprocessor_config.is_none() {
            if let Ok(p) = download(model_id, "preprocessor_config.json") {
                self.preprocessor_config = Some(ModelAsset::from_path(p));
            }
        }

        // Model weights
        if self.weights.is_empty() {
            self.download_weights_from_hf(model_id, bearer_token)?;
        }

        // Model-specific extras
        match model_type {
            ModelType::Kokoro => {
                self.download_kokoro_extras(model_id, bearer_token)?;
            }
            ModelType::OmniVoice => {
                self.download_omnivoice_extras(model_id, bearer_token)?;
            }
            ModelType::Voxtral => {
                self.download_voxtral_extras(model_id, bearer_token)?;
            }
            ModelType::Qwen3Tts => {
                self.download_qwen3tts_extras(bearer_token)?;
            }
            ModelType::VibeVoice => {
                self.download_vibevoice_extras(model_id, bearer_token)?;
            }
        }

        Ok(())
    }

    /// Download weight files (single or sharded) from HuggingFace.
    #[cfg(feature = "download")]
    fn download_weights_from_hf(
        &mut self,
        model_id: &str,
        bearer_token: Option<&str>,
    ) -> Result<(), TtsError> {
        use crate::download::download_file_with_token;

        let download = |repo: &str, file: &str| download_file_with_token(repo, file, bearer_token);

        // Try single safetensors file first
        if let Ok(p) = download(model_id, "model.safetensors") {
            self.weights.push(ModelAsset::from_path(p));
            return Ok(());
        }

        // Voxtral ships a single consolidated.safetensors file.
        if let Ok(p) = download(model_id, "consolidated.safetensors") {
            self.weights.push(ModelAsset::from_path(p));
            return Ok(());
        }

        // Try .pth file (Kokoro uses PyTorch format)
        for pth_name in &["kokoro-v1_0.pth", "kokoro-v1_1-zh.pth", "model.pth"] {
            if let Ok(p) = download(model_id, pth_name) {
                self.weights.push(ModelAsset::from_path(p));
                return Ok(());
            }
        }

        // Fall back to sharded — download the index
        let index_path = download(model_id, "model.safetensors.index.json")?;
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
                let p = download(model_id, shard_name)?;
                self.weights.push(ModelAsset::from_path(p));
            }
        }

        Ok(())
    }

    /// Download Kokoro-specific files (voices directory).
    #[cfg(feature = "download")]
    fn download_kokoro_extras(
        &mut self,
        model_id: &str,
        bearer_token: Option<&str>,
    ) -> Result<(), TtsError> {
        use crate::download::download_file_with_token;

        let download = |repo: &str, file: &str| download_file_with_token(repo, file, bearer_token);

        if self.voices_dir.is_none() {
            // Download a well-known voice to discover the voices directory
            if let Ok(voice_path) = download(model_id, "voices/af_heart.pt") {
                if let Some(parent) = voice_path.parent() {
                    self.voices_dir = Some(ModelAssetDir::from_path(parent.to_path_buf()));
                }
            }
        }

        Ok(())
    }

    /// Download Qwen3-TTS-specific files (speech tokenizer from separate repo).
    #[cfg(feature = "download")]
    fn download_qwen3tts_extras(&mut self, bearer_token: Option<&str>) -> Result<(), TtsError> {
        use crate::download::download_file_with_token;

        let tokenizer_repo = "Qwen/Qwen3-TTS-Tokenizer-12Hz";
        let download = |repo: &str, file: &str| download_file_with_token(repo, file, bearer_token);

        if self.speech_tokenizer_config.is_none() {
            info!(
                "Downloading speech tokenizer config from {}",
                tokenizer_repo
            );
            if let Ok(p) = download(tokenizer_repo, "config.json") {
                self.speech_tokenizer_config = Some(ModelAsset::from_path(p));
            }
        }

        if self.speech_tokenizer_weights.is_empty() {
            info!(
                "Downloading speech tokenizer weights from {}",
                tokenizer_repo
            );
            if let Ok(p) = download(tokenizer_repo, "model.safetensors") {
                self.speech_tokenizer_weights.push(ModelAsset::from_path(p));
            } else if let Ok(index_path) = download(tokenizer_repo, "model.safetensors.index.json") {
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
                                if let Ok(p) = download(tokenizer_repo, shard_name) {
                                    self.speech_tokenizer_weights.push(ModelAsset::from_path(p));
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
    fn download_vibevoice_extras(
        &mut self,
        model_id: &str,
        bearer_token: Option<&str>,
    ) -> Result<(), TtsError> {
        use crate::download::download_file_with_token;

        let download = |repo: &str, file: &str| download_file_with_token(repo, file, bearer_token);

        if self.preprocessor_config.is_none() {
            if let Ok(p) = download(model_id, "preprocessor_config.json") {
                self.preprocessor_config = Some(ModelAsset::from_path(p));
            }
        }

        Ok(())
    }

    /// Download OmniVoice-specific files (audio tokenizer subdirectory).
    #[cfg(feature = "download")]
    fn download_omnivoice_extras(
        &mut self,
        model_id: &str,
        bearer_token: Option<&str>,
    ) -> Result<(), TtsError> {
        use crate::download::download_file_with_token;

        let download = |repo: &str, file: &str| download_file_with_token(repo, file, bearer_token);

        if self.speech_tokenizer_config.is_none() {
            if let Ok(p) = download(model_id, "audio_tokenizer/config.json") {
                self.speech_tokenizer_config = Some(ModelAsset::from_path(p));
            }
        }

        if self.speech_tokenizer_weights.is_empty() {
            if let Ok(p) = download(model_id, "audio_tokenizer/model.safetensors") {
                self.speech_tokenizer_weights.push(ModelAsset::from_path(p));
            } else if let Ok(index_path) = download(model_id, "audio_tokenizer/model.safetensors.index.json") {
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
                                if let Ok(p) = download(model_id, &shard_path) {
                                    self.speech_tokenizer_weights.push(ModelAsset::from_path(p));
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
    fn download_voxtral_extras(
        &mut self,
        model_id: &str,
        bearer_token: Option<&str>,
    ) -> Result<(), TtsError> {
        use crate::download::download_file_with_token;

        let download = |repo: &str, file: &str| download_file_with_token(repo, file, bearer_token);

        if self.voices_dir.is_some() {
            return Ok(());
        }

        let config_path = self.config.as_ref().ok_or_else(|| {
            TtsError::FileMissing("params.json — Voxtral model configuration".to_string())
        })?;
        let content = config_path.read_bytes()?;
        let config: serde_json::Value = serde_json::from_slice(&content)?;
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

        let mut discovered_dir: Option<ModelAssetDir> = None;
        for voice_name in voices.keys() {
            let path = download(model_id, &format!("voice_embedding/{voice_name}.pt"))?;
            if discovered_dir.is_none() {
                discovered_dir = path.parent().map(|parent| ModelAssetDir::from_path(parent.to_path_buf()));
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

    /// Human-readable dtype label.
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
        }
    }
}

/// Preferred runtime choice for a model on the current machine.
///
/// This reflects the compiled backend features, runtime hardware
/// availability, and the crate's model-specific dtype safety rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeChoice {
    /// Concrete device selection for this runtime.
    pub device: DeviceSelection,
    /// Recommended dtype for the chosen device and model.
    pub dtype: DType,
}

impl RuntimeChoice {
    /// Combined device and dtype label.
    pub fn label(&self) -> String {
        format!("{} ({})", self.device.label(), self.dtype.label())
    }
}

/// Preferred runtime choices for a model, ordered fastest-first.
pub fn preferred_runtime_choices(model_type: ModelType) -> Vec<RuntimeChoice> {
    DeviceSelection::available_runtime_candidates()
        .into_iter()
        .map(|device| RuntimeChoice {
            device,
            dtype: preferred_dtype_for(model_type, device),
        })
        .collect()
}

/// Best runtime choice for a model on the current machine.
pub fn preferred_runtime_choice(model_type: ModelType) -> RuntimeChoice {
    preferred_runtime_choices(model_type)
        .into_iter()
        .next()
        .unwrap_or(RuntimeChoice {
            device: DeviceSelection::Cpu,
            dtype: DType::F32,
        })
}

fn preferred_dtype_for(model_type: ModelType, device: DeviceSelection) -> DType {
    match model_type {
        ModelType::OmniVoice => match device {
            DeviceSelection::Cpu => DType::F32,
            DeviceSelection::Cuda(_) => DType::BF16,
            DeviceSelection::Metal(_) => DType::F32,
            DeviceSelection::Auto => DType::BF16,
        },
        ModelType::Kokoro => match device {
            DeviceSelection::Cpu => DType::F32,
            DeviceSelection::Cuda(_) => DType::BF16,
            DeviceSelection::Metal(_) => DType::F32,
            DeviceSelection::Auto => DType::BF16,
        },
        ModelType::Qwen3Tts => match device {
            DeviceSelection::Cpu => DType::F32,
            DeviceSelection::Cuda(_) => DType::BF16,
            DeviceSelection::Metal(_) => DType::BF16,
            DeviceSelection::Auto => DType::BF16,
        },
        ModelType::VibeVoice => match device {
            DeviceSelection::Cpu => DType::F32,
            DeviceSelection::Cuda(_) => DType::BF16,
            DeviceSelection::Metal(_) => DType::F32,
            DeviceSelection::Auto => DType::BF16,
        },
        ModelType::Voxtral => match device {
            DeviceSelection::Cpu => DType::F32,
            DeviceSelection::Cuda(_) => DType::BF16,
            DeviceSelection::Metal(_) => DType::F32,
            DeviceSelection::Auto => DType::BF16,
        },
    }
}

// ---------------------------------------------------------------------------
// TtsConfig — main configuration + builder
// ---------------------------------------------------------------------------

/// Top-level configuration for loading a TTS model.
///
/// # Providing model files
///
/// There are four ways to tell any-tts where to find the model files,
/// listed from highest to lowest priority:
///
/// 1. **Individual file paths** (for custom download managers):
///    ```rust
///    # use any_tts::{TtsConfig, ModelType};
///    let config = TtsConfig::new(ModelType::Qwen3Tts)
///        .with_config_file("/cache/sha256-abc/config.json")
///        .with_tokenizer_file("/cache/sha256-def/tokenizer.json")
///        .with_weight_file("/cache/sha256-012/model.safetensors");
///    ```
///
/// 2. **Named in-memory assets** (for object stores and byte-first runtimes):
///    ```rust
///    # use any_tts::{ModelAssetBundle, ModelType, TtsConfig};
///    let bundle = ModelAssetBundle::new()
///        .with_bytes("config.json", vec![])
///        .with_bytes("tokenizer.json", vec![])
///        .with_bytes("model.safetensors", vec![]);
///    let config = TtsConfig::new(ModelType::Qwen3Tts)
///        .with_asset_bundle(bundle);
///    ```
///
/// 3. **Directory path** (all files in one folder):
///    ```rust
///    # use any_tts::{TtsConfig, ModelType};
///    let config = TtsConfig::new(ModelType::Qwen3Tts)
///        .with_model_path("/models/qwen3-tts");
///    ```
///
/// 4. **HuggingFace Hub download** (automatic fallback):
///    ```rust
///    # use any_tts::{TtsConfig, ModelType};
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

    /// Named byte assets that can be auto-discovered like a model directory.
    pub asset_bundle: ModelAssetBundle,
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
            asset_bundle: ModelAssetBundle::default(),
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

    /// Add a complete in-memory asset bundle using model-relative paths.
    pub fn with_asset_bundle(mut self, bundle: ModelAssetBundle) -> Self {
        self.asset_bundle = bundle;
        self
    }

    /// Add a single in-memory asset using a model-relative path such as
    /// `config.json` or `audio_tokenizer/model.safetensors`.
    pub fn with_asset_bytes(
        mut self,
        relative_path: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        self.asset_bundle.insert_bytes(relative_path, bytes);
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

    /// Apply the fastest safe runtime choice for this model on the current machine.
    ///
    /// This resolves the current machine's preferred backend and dtype now,
    /// then stores the concrete selection in the config.
    pub fn with_preferred_runtime(mut self) -> Self {
        let runtime = preferred_runtime_choice(self.model_type);
        self.device = runtime.device;
        self.dtype = runtime.dtype;
        self
    }

    // ── Individual file builders ──────────────────────────────────────

    /// Set the path to **`config.json`**.
    ///
    /// This JSON file describes the model architecture: hidden size,
    /// number of layers, vocabulary size, attention head counts, etc.
    /// It follows the standard HuggingFace config format.
    pub fn with_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.config = Some(ModelAsset::from_path(path.into()));
        self
    }

    /// Set `config.json` from in-memory bytes.
    pub fn with_config_bytes(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.files.config = Some(ModelAsset::from_bytes("config.json", bytes));
        self
    }

    /// Set the path to **`tokenizer.json`**.
    ///
    /// A self-contained HuggingFace Tokenizers JSON file with the full
    /// BPE vocabulary, merge rules, and special-token definitions.
    /// Both model backends use this to convert input text to token IDs.
    pub fn with_tokenizer_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.tokenizer = Some(ModelAsset::from_path(path.into()));
        self
    }

    /// Set `tokenizer.json` from in-memory bytes.
    pub fn with_tokenizer_bytes(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.files.tokenizer = Some(ModelAsset::from_bytes("tokenizer.json", bytes));
        self
    }

    /// Append a single **model weight file** (`.safetensors`).
    ///
    /// Call this repeatedly when you need to provide several shards
    /// explicitly, or once for single-file models:
    ///
    /// ```rust
    /// # use any_tts::{TtsConfig, ModelType};
    /// let config = TtsConfig::new(ModelType::Qwen3Tts)
    ///     .with_weight_file("/cache/model.safetensors");
    /// ```
    pub fn with_weight_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.weights.push(ModelAsset::from_path(path.into()));
        self
    }

    /// Append a single in-memory weight file.
    pub fn with_weight_bytes(
        mut self,
        file_name: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        self.files
            .weights
            .push(ModelAsset::from_bytes(file_name.into(), bytes));
        self
    }

    /// Set **all model weight files** at once, replacing any previously added.
    pub fn with_weight_files(mut self, paths: Vec<PathBuf>) -> Self {
        self.files.weights = paths.into_iter().map(ModelAsset::from_path).collect();
        self
    }

    /// Set a **voice asset directory** for backends that use preset voices.
    ///
    /// This is used by backends such as Kokoro (`voices/*.pt`) and
    /// Voxtral (`voice_embedding/*.pt`).
    pub fn with_voices_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.voices_dir = Some(ModelAssetDir::from_path(path.into()));
        self
    }

    /// Add a single in-memory preset voice asset.
    pub fn with_voice_bytes(
        mut self,
        voice_name: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        let voice_file = format!("{}.pt", voice_name.into());
        match self.files.voices_dir.take() {
            Some(ModelAssetDir::Bytes(mut entries)) => {
                entries.insert(voice_file, Arc::from(bytes.into()));
                self.files.voices_dir = Some(ModelAssetDir::from_bytes(entries));
            }
            Some(ModelAssetDir::Path(path)) => {
                self.files.voices_dir = Some(ModelAssetDir::Path(path));
                self.asset_bundle.insert_bytes(format!("voices/{voice_file}"), bytes);
            }
            None => {
                let mut entries = BTreeMap::new();
                entries.insert(voice_file, Arc::from(bytes.into()));
                self.files.voices_dir = Some(ModelAssetDir::from_bytes(entries));
            }
        }
        self
    }

    /// Append a single **speech tokenizer weight file** (Qwen3-TTS only).
    ///
    /// These weights belong to the separate speech tokenizer decoder
    /// model (`Qwen/Qwen3-TTS-Tokenizer-12Hz`) that converts discrete
    /// codec tokens into a continuous 24 kHz audio waveform.
    pub fn with_speech_tokenizer_weight_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files
            .speech_tokenizer_weights
            .push(ModelAsset::from_path(path.into()));
        self
    }

    /// Append a single in-memory speech-tokenizer weight file.
    pub fn with_speech_tokenizer_weight_bytes(
        mut self,
        file_name: impl Into<String>,
        bytes: impl Into<Vec<u8>>,
    ) -> Self {
        self.files.speech_tokenizer_weights.push(ModelAsset::from_bytes(
            file_name.into(),
            bytes,
        ));
        self
    }

    /// Set **all speech tokenizer weight files** at once (Qwen3-TTS only).
    pub fn with_speech_tokenizer_weight_files(mut self, paths: Vec<PathBuf>) -> Self {
        self.files.speech_tokenizer_weights = paths
            .into_iter()
            .map(ModelAsset::from_path)
            .collect();
        self
    }

    /// Set the **speech tokenizer config** file (Qwen3-TTS only).
    ///
    /// JSON config for the speech tokenizer decoder model, including
    /// codebook dimensions, upsampling ratios, and activation parameters.
    pub fn with_speech_tokenizer_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.speech_tokenizer_config = Some(ModelAsset::from_path(path.into()));
        self
    }

    /// Set the speech-tokenizer config from in-memory bytes.
    pub fn with_speech_tokenizer_config_bytes(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.files.speech_tokenizer_config =
            Some(ModelAsset::from_bytes("speech_tokenizer/config.json", bytes));
        self
    }

    /// Set the **generation config** file (optional).
    ///
    /// Standard HuggingFace `generation_config.json` with parameters
    /// like `max_new_tokens`, `top_p`, `temperature`, `do_sample`, etc.
    pub fn with_generation_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.generation_config = Some(ModelAsset::from_path(path.into()));
        self
    }

    /// Set `generation_config.json` from in-memory bytes.
    pub fn with_generation_config_bytes(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.files.generation_config = Some(ModelAsset::from_bytes("generation_config.json", bytes));
        self
    }

    /// Set the **preprocessor config** file (optional).
    ///
    /// This stores published preprocessing defaults such as audio
    /// normalization parameters and speech token compression ratios.
    pub fn with_preprocessor_config_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.files.preprocessor_config = Some(ModelAsset::from_path(path.into()));
        self
    }

    /// Set `preprocessor_config.json` from in-memory bytes.
    pub fn with_preprocessor_config_bytes(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.files.preprocessor_config = Some(ModelAsset::from_bytes(
            "preprocessor_config.json",
            bytes,
        ));
        self
    }

    // ── Resolution ────────────────────────────────────────────────────

    /// Resolve all model files using the four-tier strategy:
    ///
    /// 1. Explicit paths already set on `self.files`
    /// 2. Auto-discovery from `self.asset_bundle`
    /// 3. Auto-discovery from `self.model_path` directory
    /// 4. HuggingFace Hub download (if `download` feature enabled)
    ///
    /// Returns a fully populated [`ModelFiles`] or an error listing
    /// which files are missing.
    pub fn resolve_files(&self) -> Result<ModelFiles, TtsError> {
        let mut files = self.files.clone();

        if !self.asset_bundle.is_empty() {
            files.fill_from_asset_bundle(&self.asset_bundle);
        }

        // Tier 3: fill from model_path directory
        if let Some(ref dir) = self.model_path {
            files.fill_from_directory(Path::new(dir));
        }

        // Tier 4: HuggingFace Hub download fallback
        #[cfg(feature = "download")]
        {
            if !files.missing_files(self.model_type).is_empty() {
                let hf_id = self.effective_hf_model_id();
                info!("Downloading missing files from HuggingFace: {}", hf_id);
                files.fill_from_hf(hf_id, self.model_type, self.bearer_token.as_deref())?;
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
            ModelType::Kokoro
            | ModelType::OmniVoice
            | ModelType::Qwen3Tts
            | ModelType::VibeVoice => None,
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
            ModelType::Kokoro
            | ModelType::OmniVoice
            | ModelType::Qwen3Tts
            | ModelType::VibeVoice
            | ModelType::Voxtral => None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_labels_are_stable() {
        assert_eq!(DType::F32.label(), "f32");
        assert_eq!(DType::F16.label(), "f16");
        assert_eq!(DType::BF16.label(), "bf16");
    }

    #[test]
    fn test_kokoro_metal_prefers_f32() {
        assert_eq!(
            preferred_dtype_for(ModelType::Kokoro, DeviceSelection::Metal(0)),
            DType::F32
        );
    }

    #[test]
    fn test_qwen3_metal_prefers_bf16() {
        assert_eq!(
            preferred_dtype_for(ModelType::Qwen3Tts, DeviceSelection::Metal(0)),
            DType::BF16
        );
    }

    #[test]
    fn test_omnivoice_metal_prefers_f32() {
        let choice = RuntimeChoice {
            device: DeviceSelection::Metal(0),
            dtype: preferred_dtype_for(ModelType::OmniVoice, DeviceSelection::Metal(0)),
        };
        assert_eq!(choice.label(), "metal:0 (f32)");
    }

    #[test]
    fn test_with_preferred_runtime_applies_choice() {
        let expected = preferred_runtime_choice(ModelType::VibeVoice);
        let config = TtsConfig::new(ModelType::VibeVoice).with_preferred_runtime();
        assert_eq!(config.device, expected.device);
        assert_eq!(config.dtype, expected.dtype);
    }

    #[test]
    fn test_resolve_files_from_in_memory_omnivoice_assets() {
        let bundle = ModelAssetBundle::new()
            .with_bytes("config.json", vec![1])
            .with_bytes("tokenizer.json", vec![2])
            .with_bytes("model.safetensors", vec![3])
            .with_bytes("audio_tokenizer/config.json", vec![4])
            .with_bytes("audio_tokenizer/model.safetensors", vec![5]);

        let files = TtsConfig::new(ModelType::OmniVoice)
            .with_asset_bundle(bundle)
            .resolve_files()
            .unwrap();

        assert!(matches!(files.config, Some(ModelAsset::Bytes { .. })));
        assert!(matches!(files.tokenizer, Some(ModelAsset::Bytes { .. })));
        assert_eq!(files.weights.len(), 1);
        assert_eq!(files.speech_tokenizer_weights.len(), 1);
    }

    #[test]
    fn test_with_voice_bytes_creates_in_memory_voice_dir() {
        let config = TtsConfig::new(ModelType::Kokoro).with_voice_bytes("af_heart", vec![1, 2]);
        let voices_dir = config.files.voices_dir.as_ref().unwrap();
        assert_eq!(voices_dir.file_names().unwrap(), vec!["af_heart.pt"]);
    }

    #[test]
    fn test_model_asset_manifest_is_available() {
        let requirements = ModelType::Voxtral.asset_requirements();
        assert!(!requirements.is_empty());
        assert!(requirements.iter().any(|entry| entry.pattern == "voice_embedding/*.pt"));
    }
}
