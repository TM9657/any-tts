//! Text-to-phoneme conversion for Kokoro-82M.
//!
//! Converts plain text to IPA phoneme strings using espeak-ng (via `espeak-rs`).
//! Applies Kokoro-specific post-processing to map espeak's IPA output to the
//! phoneme set that Kokoro was trained on.
//!
//! ## Thread safety
//!
//! espeak-ng uses global C state and is **not** thread-safe. All calls are
//! serialised through a [`std::sync::Mutex`].

use std::collections::HashMap;
use std::sync::Mutex;

use espeak_rs::text_to_phonemes;

use crate::error::{TtsError, TtsResult};

/// Global mutex to serialize espeak-ng calls (espeak uses global state).
static ESPEAK_MUTEX: Mutex<()> = Mutex::new(());

/// Map Kokoro language codes (ISO 639-1) to espeak-ng voice identifiers.
fn lang_to_espeak(lang: &str) -> &'static str {
    match lang {
        "en" | "en-us" | "a" => "en-us",
        "en-gb" | "b" => "en-gb",
        "ja" | "j" => "ja",
        "zh" | "z" => "cmn",
        "ko" | "k" => "ko",
        "fr" | "f" => "fr",
        "de" | "d" => "de",
        "it" | "i" => "it",
        "pt" | "p" => "pt",
        "es" | "e" => "es",
        "hi" | "h" => "hi",
        _ => "en-us",
    }
}

/// Infer the language from a Kokoro voice name's first letter.
///
/// Kokoro voice naming convention:
///   first letter = language (a=American, b=British, e=Spanish, f=French,
///   h=Hindi, i=Italian, j=Japanese, p=Portuguese, z=Chinese)
pub fn language_from_voice(voice: &str) -> &'static str {
    match voice.chars().next() {
        Some('a') => "en",
        Some('b') => "en-gb",
        Some('e') => "es",
        Some('f') => "fr",
        Some('h') => "hi",
        Some('i') => "it",
        Some('j') => "ja",
        Some('k') => "ko",
        Some('p') => "pt",
        Some('z') => "zh",
        _ => "en",
    }
}

/// Apply Kokoro-specific character replacements to espeak IPA output.
///
/// espeak-ng produces standard IPA that differs from Kokoro's training phoneme
/// set. These replacements mirror the Python `misaki` library's mapping.
fn apply_kokoro_replacements(phonemes: &str) -> String {
    let mut s = phonemes
        .replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ")
        .replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ")
        .replace('ʲ', "j")
        .replace('r', "ɹ")
        .replace('x', "k")
        .replace("ɬ", "l");

    // Remove tie bars (espeak uses combining tie U+0361 for affricates)
    s = s.replace('\u{0361}', "");

    s
}

/// Filter a phoneme string to only characters present in the Kokoro vocab.
fn filter_to_vocab(phonemes: &str, vocab: &HashMap<String, u32>) -> String {
    phonemes
        .chars()
        .filter(|c| {
            let key = c.to_string();
            vocab.contains_key(&key)
        })
        .collect()
}

/// Convert plain text to Kokoro-compatible IPA phoneme string.
///
/// Pipeline:
/// 1. Call espeak-ng to get IPA phonemes for the given language
/// 2. Apply Kokoro-specific character replacements
/// 3. Filter to only characters in the Kokoro vocab
pub fn phonemize(text: &str, language: &str, vocab: &HashMap<String, u32>) -> TtsResult<String> {
    let espeak_lang = lang_to_espeak(language);

    let raw_phonemes = {
        let _guard = ESPEAK_MUTEX
            .lock()
            .map_err(|e| TtsError::ModelError(format!("espeak mutex poisoned: {e}")))?;
        text_to_phonemes(text, espeak_lang, None, true, false).map_err(|e| {
            TtsError::ModelError(format!(
                "espeak-ng phonemization failed for lang '{language}' (espeak voice '{espeak_lang}'): {e}"
            ))
        })?
    };

    let joined = raw_phonemes.join("");
    let replaced = apply_kokoro_replacements(&joined);
    let filtered = filter_to_vocab(&replaced, vocab);

    if filtered.is_empty() {
        return Err(TtsError::TokenizerError(format!(
            "Phonemization produced no valid tokens for text: \"{text}\" (lang: {language})"
        )));
    }

    Ok(filtered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_vocab() -> HashMap<String, u32> {
        // Build a minimal vocab containing common IPA chars
        let chars = "$;:,.!?¡¿—…\"«»\u{201c}\u{201d} \
            ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnoprstuvwxyz\
            ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰ\
            ŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ\
            ǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘ᵻ";
        let mut vocab = HashMap::new();
        for (i, c) in chars.chars().enumerate() {
            vocab.insert(c.to_string(), i as u32);
        }
        vocab
    }

    #[test]
    fn test_language_from_voice() {
        assert_eq!(language_from_voice("af_heart"), "en");
        assert_eq!(language_from_voice("dm_speaker"), "en"); // unknown prefix
        assert_eq!(language_from_voice("jf_alpha"), "ja");
        assert_eq!(language_from_voice("ef_dora"), "es");
        assert_eq!(language_from_voice("ff_siwis"), "fr");
    }

    #[test]
    fn test_kokoro_replacements() {
        let input = "ʲrxɬ";
        let output = apply_kokoro_replacements(input);
        assert_eq!(output, "jɹkl");
    }

    #[test]
    fn test_phonemize_english() {
        let vocab = dummy_vocab();
        let result = phonemize("Hello world", "en", &vocab);
        assert!(result.is_ok(), "phonemize failed: {:?}", result.err());
        let ph = result.unwrap();
        assert!(!ph.is_empty(), "phonemes should not be empty");
        // Should contain IPA characters, not ASCII "Hello"
        assert!(
            ph.contains('ə') || ph.contains('ɛ') || ph.contains('ˈ') || ph.contains('l'),
            "Expected IPA phonemes, got: {ph}"
        );
    }

    #[test]
    fn test_phonemize_german() {
        let vocab = dummy_vocab();
        let result = phonemize("Guten Tag", "de", &vocab);
        assert!(result.is_ok(), "phonemize failed: {:?}", result.err());
        let ph = result.unwrap();
        assert!(!ph.is_empty());
    }

    #[test]
    fn test_filter_to_vocab() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 1);
        vocab.insert("b".to_string(), 2);
        assert_eq!(filter_to_vocab("abc", &vocab), "ab");
    }
}
