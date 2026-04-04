//! Text tokenizer wrapper.
//!
//! Wraps the HuggingFace `tokenizers` crate for BPE tokenization used by
//! both model backends.

use crate::error::TtsError;
use std::path::Path;

/// A BPE text tokenizer loaded from vocab/merges files or a tokenizer.json.
pub struct TextTokenizer {
    inner: tokenizers::Tokenizer,
}

impl TextTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TtsError> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref()).map_err(|e| {
            TtsError::TokenizerError(format!("Failed to load tokenizer: {}", e))
        })?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a pretrained model directory.
    /// Looks for `tokenizer.json` first, then falls back to `vocab.json` + `merges.txt`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, TtsError> {
        let dir = dir.as_ref();

        // Try tokenizer.json first
        let tokenizer_json = dir.join("tokenizer.json");
        if tokenizer_json.exists() {
            return Self::from_file(tokenizer_json);
        }

        // Fall back to vocab.json + merges.txt
        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");

        if vocab_path.exists() && merges_path.exists() {
            let inner = tokenizers::Tokenizer::from_file(&vocab_path).map_err(|e| {
                TtsError::TokenizerError(format!(
                    "Failed to load tokenizer from vocab.json: {}",
                    e
                ))
            })?;
            Ok(Self { inner })
        } else {
            Err(TtsError::TokenizerError(format!(
                "No tokenizer files found in {}",
                dir.display()
            )))
        }
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TtsError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TtsError::TokenizerError(format!("Encoding failed: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TtsError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TtsError::TokenizerError(format!("Decoding failed: {}", e)))
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the token ID for a specific token string, if it exists.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}
