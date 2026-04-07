//! Built-in Hugging Face model download utilities.
//!
//! The crate only needs a narrow subset of Hub functionality: resolve a known
//! repository file by name, cache it locally, and return the cache path.
//! These helpers provide that behavior without depending on `hf-hub`.

#[cfg(feature = "download")]
use crate::error::TtsError;

#[cfg(feature = "download")]
const HF_HOME_ENV: &str = "HF_HOME";
#[cfg(feature = "download")]
const HF_CACHE_ENV: &str = "HUGGINGFACE_HUB_CACHE";
#[cfg(feature = "download")]
const HF_TOKEN_ENV: &str = "HF_TOKEN";
#[cfg(feature = "download")]
const HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";
#[cfg(feature = "download")]
const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
#[cfg(feature = "download")]
const DEFAULT_REVISION: &str = "main";

/// Download a **single** file from a HuggingFace Hub model repository.
///
/// Returns the absolute path to the locally cached file.
///
/// # Errors
///
/// Returns [`TtsError::WeightLoadError`] if the file cannot be fetched
/// (network error, file not found in repo, etc.).
#[cfg(feature = "download")]
pub fn download_file(model_id: &str, filename: &str) -> Result<std::path::PathBuf, TtsError> {
    download_file_with_token(model_id, filename, None)
}

/// Download a single file from a Hugging Face repo with an optional bearer token.
#[cfg(feature = "download")]
pub fn download_file_with_token(
    model_id: &str,
    filename: &str,
    bearer_token: Option<&str>,
) -> Result<std::path::PathBuf, TtsError> {
    use std::fs::File;
    use std::io;

    use tracing::info;

    let cache_root = cache_root();
    let path = cached_file_path(&cache_root, model_id, filename);
    if path.exists() {
        return Ok(path);
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let url = file_url(model_id, filename);
    let token = resolve_bearer_token(bearer_token, &cache_root);
    info!("Downloading {} from {}", filename, model_id);

    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .user_agent(format!("any-tts/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| TtsError::RuntimeError(format!("Failed to initialize HTTP client: {e}")))?;

    let mut request = client.get(url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    let mut response = request.send().map_err(|e| {
        TtsError::WeightLoadError(format!(
            "Failed to download {} from {}: {}",
            filename, model_id, e
        ))
    })?;

    let status = response.status();
    if !status.is_success() {
        return Err(TtsError::WeightLoadError(format!(
            "Failed to download {} from {}: HTTP {}",
            filename, model_id, status
        )));
    }

    let partial_path = partial_file_path(&path);
    let write_result = (|| -> Result<(), TtsError> {
        let mut file = File::create(&partial_path)?;
        io::copy(&mut response, &mut file)?;
        file.sync_all()?;
        std::fs::rename(&partial_path, &path)?;
        Ok(())
    })();

    if write_result.is_err() {
        let _ = std::fs::remove_file(&partial_path);
    }
    write_result?;

    Ok(path)
}

/// Batch-download multiple files from a HuggingFace Hub model repository.
///
/// Returns the cache directory containing the downloaded files.
#[cfg(feature = "download")]
pub fn download_model(model_id: &str, filenames: &[&str]) -> Result<std::path::PathBuf, TtsError> {
    download_model_with_token(model_id, filenames, None)
}

/// Batch-download multiple files from a Hugging Face Hub model repository.
#[cfg(feature = "download")]
pub fn download_model_with_token(
    model_id: &str,
    filenames: &[&str],
    bearer_token: Option<&str>,
) -> Result<std::path::PathBuf, TtsError> {
    use tracing::info;

    info!("Downloading model {} from HuggingFace Hub", model_id);

    let mut first_path: Option<std::path::PathBuf> = None;

    for filename in filenames {
        let path = download_file_with_token(model_id, filename, bearer_token)?;
        if first_path.is_none() {
            first_path = Some(path);
        }
    }

    let first =
        first_path.ok_or_else(|| TtsError::WeightLoadError("No files to download".to_string()))?;

    let model_dir = first.parent().unwrap_or(&first).to_path_buf();
    info!("Model files cached at: {}", model_dir.display());
    Ok(model_dir)
}

#[cfg(feature = "download")]
fn cache_root() -> std::path::PathBuf {
    cache_root_from_env(
        std::env::var_os(HF_CACHE_ENV).map(std::path::PathBuf::from),
        std::env::var_os(HF_HOME_ENV).map(std::path::PathBuf::from),
        std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from),
        std::env::var_os("HOME").map(std::path::PathBuf::from),
        std::env::var_os("LOCALAPPDATA").map(std::path::PathBuf::from),
        std::env::var_os("USERPROFILE").map(std::path::PathBuf::from),
    )
}

#[cfg(feature = "download")]
fn cache_root_from_env(
    hub_cache: Option<std::path::PathBuf>,
    hf_home: Option<std::path::PathBuf>,
    xdg_cache_home: Option<std::path::PathBuf>,
    home: Option<std::path::PathBuf>,
    local_app_data: Option<std::path::PathBuf>,
    user_profile: Option<std::path::PathBuf>,
) -> std::path::PathBuf {
    if let Some(path) = hub_cache {
        return path;
    }
    if let Some(path) = hf_home {
        return path.join("hub");
    }
    if let Some(path) = xdg_cache_home {
        return path.join("huggingface").join("hub");
    }
    if let Some(path) = home {
        return path.join(".cache").join("huggingface").join("hub");
    }
    if let Some(path) = local_app_data {
        return path.join("huggingface").join("hub");
    }
    if let Some(path) = user_profile {
        return path.join(".cache").join("huggingface").join("hub");
    }

    std::env::temp_dir().join("huggingface").join("hub")
}

#[cfg(feature = "download")]
fn cached_file_path(
    cache_root: &std::path::Path,
    model_id: &str,
    filename: &str,
) -> std::path::PathBuf {
    cache_root
        .join(format!("models--{}", model_id.replace('/', "--")))
        .join("snapshots")
        .join(DEFAULT_REVISION)
        .join(filename)
}

#[cfg(feature = "download")]
fn partial_file_path(path: &std::path::Path) -> std::path::PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    path.parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(format!(".{file_name}.part"))
}

#[cfg(feature = "download")]
fn file_url(model_id: &str, filename: &str) -> String {
    let endpoint = std::env::var(HF_ENDPOINT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string());
    format!(
        "{}/{}/resolve/{}/{}",
        endpoint.trim_end_matches('/'),
        model_id,
        DEFAULT_REVISION,
        filename
    )
}

#[cfg(feature = "download")]
fn resolve_bearer_token(
    explicit_token: Option<&str>,
    cache_root: &std::path::Path,
) -> Option<String> {
    normalize_token(explicit_token)
        .or_else(|| normalize_token(std::env::var(HF_TOKEN_ENV).ok().as_deref()))
        .or_else(|| read_token_file(&token_path(cache_root)))
}

#[cfg(feature = "download")]
fn token_path(cache_root: &std::path::Path) -> std::path::PathBuf {
    cache_root.parent().unwrap_or(cache_root).join("token")
}

#[cfg(feature = "download")]
fn read_token_file(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|token| normalize_token(Some(token.as_str())))
}

#[cfg(feature = "download")]
fn normalize_token(token: Option<&str>) -> Option<String> {
    let token = token?.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

#[cfg(all(test, feature = "download"))]
mod tests {
    use super::*;

    #[test]
    fn test_cache_root_prefers_explicit_hub_cache() {
        let path = cache_root_from_env(
            Some("/tmp/hf-cache".into()),
            Some("/tmp/hf-home".into()),
            Some("/tmp/xdg-cache".into()),
            Some("/tmp/home".into()),
            None,
            None,
        );
        assert_eq!(path, std::path::PathBuf::from("/tmp/hf-cache"));
    }

    #[test]
    fn test_cache_root_falls_back_to_hf_home() {
        let path = cache_root_from_env(
            None,
            Some("/tmp/hf-home".into()),
            Some("/tmp/xdg-cache".into()),
            Some("/tmp/home".into()),
            None,
            None,
        );
        assert_eq!(path, std::path::PathBuf::from("/tmp/hf-home/hub"));
    }

    #[test]
    fn test_cached_file_path_preserves_subdirectories() {
        let path = cached_file_path(
            std::path::Path::new("/cache"),
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "audio_tokenizer/model.safetensors",
        );
        assert_eq!(
            path,
            std::path::PathBuf::from(
                "/cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/main/audio_tokenizer/model.safetensors"
            )
        );
    }

    #[test]
    fn test_token_path_uses_hf_home_parent() {
        let path = token_path(std::path::Path::new("/tmp/huggingface/hub"));
        assert_eq!(path, std::path::PathBuf::from("/tmp/huggingface/token"));
    }
}
