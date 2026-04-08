use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use base64::Engine;
use rustc_hash::FxHasher;
use serde::Deserialize;
use tiktoken_rs::CoreBPE;

use crate::error::TtsError;

use super::config::VoxtralConfig;

const BOS_TOKEN: &str = "<s>";
const EOS_TOKEN: &str = "</s>";
const AUDIO_TOKEN: &str = "[AUDIO]";
const BEGIN_AUDIO_TOKEN: &str = "[BEGIN_AUDIO]";
const NEXT_AUDIO_TEXT_TOKEN: &str = "[NEXT_AUDIO_TEXT]";
const REPEAT_AUDIO_TEXT_TOKEN: &str = "[REPEAT_AUDIO_TEXT]";

type TekkenMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Debug, Deserialize)]
struct TekkenFile {
    config: TekkenConfig,
    vocab: Vec<TekkenVocabEntry>,
    special_tokens: Vec<TekkenSpecialToken>,
}

#[derive(Debug, Deserialize)]
struct TekkenConfig {
    pattern: String,
    default_vocab_size: usize,
    default_num_special_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct TekkenVocabEntry {
    rank: u32,
    token_bytes: String,
}

#[derive(Debug, Deserialize)]
struct TekkenSpecialToken {
    rank: u32,
    token_str: String,
}

pub struct VoxtralTokenizer {
    bpe: CoreBPE,
    bos_token_id: u32,
    audio_token_id: u32,
    begin_audio_token_id: u32,
    next_audio_text_token_id: u32,
    repeat_audio_text_token_id: u32,
}

impl VoxtralTokenizer {
    pub fn from_bytes(bytes: impl AsRef<[u8]>, config: &VoxtralConfig) -> Result<Self, TtsError> {
        let tekken: TekkenFile = serde_json::from_slice(bytes.as_ref())?;

        let num_special_tokens = tekken.config.default_num_special_tokens as u32;
        let inner_vocab_size = tekken
            .config
            .default_vocab_size
            .checked_sub(tekken.config.default_num_special_tokens)
            .ok_or_else(|| {
                TtsError::TokenizerError(
                    "Tekken default vocab size is smaller than the number of special tokens"
                        .to_string(),
                )
            })?;

        let mut encoder = TekkenMap::default();
        for entry in tekken
            .vocab
            .iter()
            .filter(|entry| entry.rank < inner_vocab_size as u32)
        {
            let token_bytes = base64::engine::general_purpose::STANDARD
                .decode(&entry.token_bytes)
                .map_err(|err| {
                    TtsError::TokenizerError(format!(
                        "Failed to decode Tekken token bytes '{}': {}",
                        entry.token_bytes, err
                    ))
                })?;
            encoder.insert(token_bytes, entry.rank + num_special_tokens);
        }
        if encoder.len() != inner_vocab_size {
            return Err(TtsError::TokenizerError(format!(
                "Tekken vocabulary truncation produced {} tokens, expected {}",
                encoder.len(),
                inner_vocab_size
            )));
        }

        let special_tokens = tekken
            .special_tokens
            .iter()
            .map(|entry| (entry.token_str.clone(), entry.rank))
            .collect::<TekkenMap<_, _>>();

        let bpe = CoreBPE::new(encoder, TekkenMap::default(), &tekken.config.pattern)
            .map_err(|err| TtsError::TokenizerError(err.to_string()))?;

        let bos_token_id = *special_tokens
            .get(BOS_TOKEN)
            .ok_or_else(|| missing_token(BOS_TOKEN))?;
        special_tokens
            .get(EOS_TOKEN)
            .ok_or_else(|| missing_token(EOS_TOKEN))?;
        let audio_token_id = *special_tokens
            .get(AUDIO_TOKEN)
            .ok_or_else(|| missing_token(AUDIO_TOKEN))?;
        let begin_audio_token_id = *special_tokens
            .get(BEGIN_AUDIO_TOKEN)
            .ok_or_else(|| missing_token(BEGIN_AUDIO_TOKEN))?;
        let next_audio_text_token_id = *special_tokens
            .get(NEXT_AUDIO_TEXT_TOKEN)
            .ok_or_else(|| missing_token(NEXT_AUDIO_TEXT_TOKEN))?;
        let repeat_audio_text_token_id = *special_tokens
            .get(REPEAT_AUDIO_TEXT_TOKEN)
            .ok_or_else(|| missing_token(REPEAT_AUDIO_TEXT_TOKEN))?;

        if bos_token_id != config.multimodal.bos_token_id {
            return Err(TtsError::TokenizerError(format!(
                "Tekken bos token id {} does not match params.json value {}",
                bos_token_id, config.multimodal.bos_token_id
            )));
        }
        if audio_token_id != config.multimodal.audio_model_args.audio_token_id {
            return Err(TtsError::TokenizerError(format!(
                "Tekken audio token id {} does not match params.json value {}",
                audio_token_id, config.multimodal.audio_model_args.audio_token_id
            )));
        }
        if begin_audio_token_id != config.multimodal.audio_model_args.begin_audio_token_id {
            return Err(TtsError::TokenizerError(format!(
                "Tekken begin-audio token id {} does not match params.json value {}",
                begin_audio_token_id, config.multimodal.audio_model_args.begin_audio_token_id
            )));
        }

        Ok(Self {
            bpe,
            bos_token_id,
            audio_token_id,
            begin_audio_token_id,
            next_audio_text_token_id,
            repeat_audio_text_token_id,
        })
    }

    pub fn encode_text(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_ordinary(text)
    }

    pub fn build_speech_prompt(&self, text: &str, voice_audio_tokens: usize) -> Vec<u32> {
        let text_tokens = self.encode_text(text);
        let mut prompt = Vec::with_capacity(text_tokens.len() + voice_audio_tokens + 5);
        prompt.push(self.bos_token_id);
        prompt.push(self.begin_audio_token_id);
        prompt.extend(std::iter::repeat_n(self.audio_token_id, voice_audio_tokens));
        prompt.push(self.next_audio_text_token_id);
        prompt.extend(text_tokens);
        prompt.push(self.repeat_audio_text_token_id);
        prompt.push(self.begin_audio_token_id);
        prompt
    }
}

fn missing_token(token: &str) -> TtsError {
    TtsError::TokenizerError(format!("Tekken tokenizer is missing special token {token}"))
}
