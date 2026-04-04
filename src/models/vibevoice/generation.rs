use std::collections::HashMap;
use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Shape, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::TtsError;

const DIFFUSION_NOISE_FORMAT: &str = "vibevoice-diffusion-noise-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionNoiseFixture {
    #[serde(default = "default_diffusion_noise_format")]
    pub format: String,
    pub latent_size: usize,
    #[serde(default)]
    pub noises: Vec<Vec<f32>>,
}

impl DiffusionNoiseFixture {
    pub fn from_file(path: &Path) -> Result<Self, TtsError> {
        let content = fs::read_to_string(path)?;
        let fixture: Self = serde_json::from_str(&content)?;
        fixture.validate()?;
        Ok(fixture)
    }

    pub fn cursor(&self) -> DiffusionNoiseCursor<'_> {
        DiffusionNoiseCursor {
            fixture: self,
            index: 0,
        }
    }

    fn validate(&self) -> Result<(), TtsError> {
        if self.format != DIFFUSION_NOISE_FORMAT {
            return Err(TtsError::ModelError(format!(
                "Unsupported VibeVoice diffusion noise fixture format: {}",
                self.format
            )));
        }

        if self.latent_size == 0 {
            return Err(TtsError::ModelError(
                "VibeVoice diffusion noise fixture must declare a positive latent size".to_string(),
            ));
        }

        for (index, noise) in self.noises.iter().enumerate() {
            if noise.len() != self.latent_size {
                return Err(TtsError::ModelError(format!(
                    "VibeVoice diffusion noise row {} has width {}, expected {}",
                    index,
                    noise.len(),
                    self.latent_size,
                )));
            }
        }

        Ok(())
    }
}

pub struct DiffusionNoiseCursor<'a> {
    fixture: &'a DiffusionNoiseFixture,
    index: usize,
}

impl<'a> DiffusionNoiseCursor<'a> {
    pub fn next_tensor(&mut self, device: &Device, dtype: DType) -> Result<Tensor, TtsError> {
        let noise = self
            .fixture
            .noises
            .get(self.index)
            .ok_or_else(|| {
                TtsError::ModelError(format!(
                    "VibeVoice diffusion noise fixture ended after {} token(s)",
                    self.index,
                ))
            })?
            .clone();
        self.index += 1;

        Tensor::new(noise.as_slice(), device)?
            .unsqueeze(0)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

pub struct TokenSequenceState {
    token_ids: Vec<u32>,
    embedding_rows: Vec<Tensor>,
}

impl TokenSequenceState {
    pub fn empty() -> Self {
        Self {
            token_ids: Vec::new(),
            embedding_rows: Vec::new(),
        }
    }

    pub fn from_base_embeddings(
        token_ids: &[u32],
        base_embeddings: &Tensor,
        embedding_overrides: &HashMap<usize, Tensor>,
    ) -> Result<Self, TtsError> {
        let (batch, seq_len, _hidden) = base_embeddings.dims3()?;
        if batch != 1 || seq_len != token_ids.len() {
            return Err(TtsError::ModelError(
                "Unexpected VibeVoice embedding shape while building generation state".to_string(),
            ));
        }

        let mut embedding_rows = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let row = if let Some(override_embedding) = embedding_overrides.get(&position) {
                normalize_embedding_row(override_embedding.clone())?
            } else {
                base_embeddings.narrow(1, position, 1)?.squeeze(0)?
            };
            embedding_rows.push(row);
        }

        Ok(Self {
            token_ids: token_ids.to_vec(),
            embedding_rows,
        })
    }

    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn push_token(&mut self, token_id: u32, embedding: Tensor) -> Result<(), TtsError> {
        self.token_ids.push(token_id);
        self.embedding_rows
            .push(normalize_embedding_row(embedding)?);
        Ok(())
    }

    pub fn replace_last_embedding(&mut self, embedding: Tensor) -> Result<(), TtsError> {
        let last = self.embedding_rows.last_mut().ok_or_else(|| {
            TtsError::ModelError(
                "Cannot replace a VibeVoice embedding in an empty sequence".to_string(),
            )
        })?;
        *last = normalize_embedding_row(embedding)?;
        Ok(())
    }

    pub fn input_embeddings(&self) -> Result<Tensor, TtsError> {
        if self.embedding_rows.is_empty() {
            return Err(TtsError::ModelError(
                "Cannot build VibeVoice input embeddings for an empty sequence".to_string(),
            ));
        }

        let rows = self.embedding_rows.iter().collect::<Vec<_>>();
        Tensor::cat(&rows, 0)?.unsqueeze(0).map_err(Into::into)
    }
}

fn normalize_embedding_row(embedding: Tensor) -> Result<Tensor, TtsError> {
    match embedding.rank() {
        1 => embedding.unsqueeze(0).map_err(Into::into),
        2 => Ok(embedding),
        _ => Err(TtsError::ModelError(
            "Unexpected VibeVoice embedding rank while updating generation state".to_string(),
        )),
    }
}

fn default_diffusion_noise_format() -> String {
    DIFFUSION_NOISE_FORMAT.to_string()
}

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state as u32
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

pub fn random_normal_tensor<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
    rng: &mut SimpleRng,
) -> Result<Tensor, TtsError> {
    let shape = shape.into();
    let elem_count = shape.elem_count();
    let mut values = Vec::with_capacity(elem_count);
    while values.len() < elem_count {
        let u1 = rng.next_f32().clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
        let u2 = rng.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        values.push(radius * theta.cos());
        if values.len() < elem_count {
            values.push(radius * theta.sin());
        }
    }

    Tensor::from_vec(values, shape, device)
        .and_then(|tensor| tensor.to_dtype(dtype))
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};

    use super::{default_diffusion_noise_format, DiffusionNoiseFixture, TokenSequenceState};

    #[test]
    fn token_sequence_state_applies_overrides() {
        let device = Device::Cpu;
        let base = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 3, 2), &device).unwrap();
        let mut overrides = HashMap::new();
        overrides.insert(
            1usize,
            Tensor::from_vec(vec![30f32, 40.], 2, &device).unwrap(),
        );

        let state =
            TokenSequenceState::from_base_embeddings(&[10, 11, 12], &base, &overrides).unwrap();
        let rows = state
            .input_embeddings()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        assert_eq!(
            rows,
            vec![vec![vec![1.0, 2.0], vec![30.0, 40.0], vec![5.0, 6.0]]]
        );
    }

    #[test]
    fn diffusion_noise_fixture_validates_row_widths() {
        let fixture = DiffusionNoiseFixture {
            format: default_diffusion_noise_format(),
            latent_size: 4,
            noises: vec![vec![0.0; 4], vec![1.0; 4]],
        };
        let device = Device::Cpu;
        let mut cursor = fixture.cursor();
        let tensor = cursor.next_tensor(&device, DType::F32).unwrap();

        assert_eq!(tensor.dims(), &[1, 4]);
    }
}
