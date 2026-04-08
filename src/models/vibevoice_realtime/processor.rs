use crate::error::TtsError;
use crate::tokenizer::TextTokenizer;
use crate::traits::SynthesisRequest;

use crate::models::vibevoice::config::VibeVoicePreprocessorConfig;
use crate::models::vibevoice::processor::VibeVoiceTokenizerSpec;

pub struct VibeVoiceRealtimeProcessor {
    tokenizer: TextTokenizer,
    tokenizer_spec: VibeVoiceTokenizerSpec,
}

impl VibeVoiceRealtimeProcessor {
    pub fn new(
        tokenizer: TextTokenizer,
        tokenizer_spec: VibeVoiceTokenizerSpec,
        _config: VibeVoicePreprocessorConfig,
    ) -> Self {
        Self {
            tokenizer,
            tokenizer_spec,
        }
    }

    pub fn tokenizer_spec(&self) -> &VibeVoiceTokenizerSpec {
        &self.tokenizer_spec
    }

    pub fn prepare_text(&self, request: &SynthesisRequest) -> Result<Vec<u32>, TtsError> {
        self.tokenizer.encode(&format!("{}\n", request.text.trim()))
    }
}
