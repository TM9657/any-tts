//! HuggingFace Hub model download utilities.
//!
//! Two levels of API:
//!
//! * [`download_file`] — download a **single** file by name and return its
//!   local cache path. This is the primitive used by
//!   [`ModelFiles::fill_from_hf`](crate::config::ModelFiles::fill_from_hf).
//! * [`download_model`] — batch-download a list of files and return the
//!   cache directory.

#[cfg(feature = "download")]
use crate::error::TtsError;

/// Download a **single** file from a HuggingFace Hub model repository.
///
/// Returns the absolute path to the locally cached file.
///
/// # Errors
///
/// Returns [`TtsError::WeightLoadError`] if the file cannot be fetched
/// (network error, file not found in repo, etc.).
#[cfg(feature = "download")]
pub fn download_file(
    model_id: &str,
    filename: &str,
) -> Result<std::path::PathBuf, TtsError> {
    use hf_hub::api::sync::Api;
    use tracing::info;

    let api = Api::new().map_err(|e| {
        TtsError::IoError(std::io::Error::other(
            format!("Failed to initialize HF Hub API: {}", e),
        ))
    })?;

    let repo = api.model(model_id.to_string());

    info!("Downloading {} from {}", filename, model_id);
    let path = repo.get(filename).map_err(|e| {
        TtsError::WeightLoadError(format!(
            "Failed to download {} from {}: {}",
            filename, model_id, e
        ))
    })?;

    Ok(path)
}

/// Batch-download multiple files from a HuggingFace Hub model repository.
///
/// Returns the cache directory containing the downloaded files.
#[cfg(feature = "download")]
pub fn download_model(
    model_id: &str,
    filenames: &[&str],
) -> Result<std::path::PathBuf, TtsError> {
    use tracing::info;

    info!("Downloading model {} from HuggingFace Hub", model_id);

    let mut first_path: Option<std::path::PathBuf> = None;

    for filename in filenames {
        let path = download_file(model_id, filename)?;
        if first_path.is_none() {
            first_path = Some(path);
        }
    }

    let first = first_path.ok_or_else(|| {
        TtsError::WeightLoadError("No files to download".to_string())
    })?;

    let model_dir = first.parent().unwrap_or(&first).to_path_buf();
    info!("Model files cached at: {}", model_dir.display());
    Ok(model_dir)
}
