use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::layers::mlp::SiluMlp;
use crate::tensor_utils::RmsNorm;

use super::config::VibeVoiceDiffusionHeadConfig;

pub struct VibeVoiceDiffusionHead {
    noisy_images_proj: Linear,
    cond_proj: Linear,
    t_embedder: TimestepEmbedder,
    layers: Vec<DiffusionHeadLayer>,
    final_layer: DiffusionFinalLayer,
}

impl VibeVoiceDiffusionHead {
    pub fn load(config: &VibeVoiceDiffusionHeadConfig, vb: VarBuilder) -> Result<Self> {
        let noisy_images_proj = candle_nn::linear_no_bias(
            config.latent_size,
            config.hidden_size,
            vb.pp("noisy_images_proj"),
        )?;
        let cond_proj =
            candle_nn::linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("cond_proj"))?;
        let t_embedder = TimestepEmbedder::load(config.hidden_size, vb.pp("t_embedder"))?;

        let mut layers = Vec::with_capacity(config.head_layers);
        for index in 0..config.head_layers {
            layers.push(DiffusionHeadLayer::load(
                config.hidden_size,
                (config.hidden_size as f64 * config.head_ffn_ratio) as usize,
                config.rms_norm_eps,
                vb.pp(format!("layers.{}", index)),
            )?);
        }

        let final_layer = DiffusionFinalLayer::load(
            config.hidden_size,
            config.latent_size,
            config.rms_norm_eps,
            vb.pp("final_layer"),
        )?;

        Ok(Self {
            noisy_images_proj,
            cond_proj,
            t_embedder,
            layers,
            final_layer,
        })
    }

    pub fn forward(
        &self,
        noisy_images: &Tensor,
        timesteps: &Tensor,
        condition: &Tensor,
    ) -> Result<Tensor> {
        let mut x = self.noisy_images_proj.forward(noisy_images)?;
        let t = self.t_embedder.forward(timesteps)?;
        let condition = self.cond_proj.forward(condition)?;
        let c = condition.broadcast_add(&t)?;

        for layer in &self.layers {
            x = layer.forward(&x, &c)?;
        }

        self.final_layer.forward(&x, &c)
    }
}

struct TimestepEmbedder {
    linear1: Linear,
    linear2: Linear,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    fn load(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear_no_bias(256, hidden_size, vb.pp("mlp.0"))?;
        let linear2 = candle_nn::linear_no_bias(hidden_size, hidden_size, vb.pp("mlp.2"))?;
        Ok(Self {
            linear1,
            linear2,
            frequency_embedding_size: 256,
        })
    }

    fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let embedding = timestep_embedding(timesteps, self.frequency_embedding_size)?;
        let embedding = self.linear1.forward(&embedding)?;
        let embedding = candle_nn::Activation::Silu.forward(&embedding)?;
        self.linear2.forward(&embedding)
    }
}

struct DiffusionHeadLayer {
    ffn: SiluMlp,
    norm: RmsNorm,
    ada_ln_modulation: Linear,
}

impl DiffusionHeadLayer {
    fn load(
        hidden_size: usize,
        intermediate_size: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ffn = SiluMlp::load(hidden_size, intermediate_size, vb.pp("ffn"))?;
        let norm = RmsNorm::load(hidden_size, eps, vb.pp("norm"))?;
        let ada_ln_modulation =
            candle_nn::linear_no_bias(hidden_size, hidden_size * 3, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            ffn,
            norm,
            ada_ln_modulation,
        })
    }

    fn forward(&self, x: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let modulation = self
            .ada_ln_modulation
            .forward(&candle_nn::Activation::Silu.forward(condition)?)?;
        let hidden = x.dim(candle_core::D::Minus1)?;
        let shift = modulation.narrow(candle_core::D::Minus1, 0, hidden)?;
        let scale = modulation.narrow(candle_core::D::Minus1, hidden, hidden)?;
        let gate = modulation.narrow(candle_core::D::Minus1, hidden * 2, hidden)?;
        let modulated = modulate(&self.norm.forward(x)?, &shift, &scale)?;
        let ffn_out = self.ffn.forward(&modulated)?;
        x.broadcast_add(&ffn_out.broadcast_mul(&gate)?)
    }
}

struct DiffusionFinalLayer {
    linear: Linear,
    ada_ln_modulation: Linear,
    eps: f64,
}

impl DiffusionFinalLayer {
    fn load(hidden_size: usize, output_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear_no_bias(hidden_size, output_size, vb.pp("linear"))?;
        let ada_ln_modulation =
            candle_nn::linear_no_bias(hidden_size, hidden_size * 2, vb.pp("adaLN_modulation.1"))?;
        Ok(Self {
            linear,
            ada_ln_modulation,
            eps,
        })
    }

    fn forward(&self, x: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let modulation = self
            .ada_ln_modulation
            .forward(&candle_nn::Activation::Silu.forward(condition)?)?;
        let hidden = x.dim(candle_core::D::Minus1)?;
        let shift = modulation.narrow(candle_core::D::Minus1, 0, hidden)?;
        let scale = modulation.narrow(candle_core::D::Minus1, hidden, hidden)?;
        let normalized = rms_norm_without_weight(x, self.eps)?;
        let modulated = modulate(&normalized, &shift, &scale)?;
        self.linear.forward(&modulated)
    }
}

pub struct DpmSolverMultistepScheduler {
    train_sigmas: Vec<f64>,
    sigmas: Vec<f64>,
    timesteps: Vec<usize>,
    model_outputs: Vec<Option<Tensor>>,
    lower_order_nums: usize,
    step_index: usize,
    solver_order: usize,
    prediction_type: String,
}

impl DpmSolverMultistepScheduler {
    pub fn new(config: &VibeVoiceDiffusionHeadConfig) -> Self {
        let betas = match config.ddpm_beta_schedule.as_str() {
            "cosine" => betas_for_alpha_bar(config.ddpm_num_steps),
            _ => betas_for_alpha_bar(config.ddpm_num_steps),
        };
        let mut alphas_cumprod = Vec::with_capacity(betas.len());
        let mut cumulative = 1.0f64;
        for beta in betas {
            cumulative *= 1.0 - beta;
            alphas_cumprod.push(cumulative);
        }
        let sigmas = alphas_cumprod
            .iter()
            .map(|alpha_cumprod| ((1.0 - alpha_cumprod) / alpha_cumprod).sqrt())
            .collect::<Vec<_>>();

        Self {
            train_sigmas: sigmas.clone(),
            sigmas,
            timesteps: Vec::new(),
            model_outputs: vec![None, None],
            lower_order_nums: 0,
            step_index: 0,
            solver_order: 2,
            prediction_type: config.prediction_type.clone(),
        }
    }

    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        let max_timestep = self.train_sigmas.len() - 1;
        self.timesteps = (0..num_inference_steps)
            .map(|index| {
                let position = index as f64 * max_timestep as f64 / num_inference_steps as f64;
                max_timestep.saturating_sub(position.round() as usize)
            })
            .collect();

        let schedule = self
            .timesteps
            .iter()
            .map(|timestep| interpolate_sigma(&self.train_sigmas, *timestep as f64))
            .collect::<Vec<_>>();
        self.sigmas = schedule;
        self.sigmas.push(0.0);
        self.step_index = 0;
        self.lower_order_nums = 0;
        self.model_outputs = vec![None, None];
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let converted = self.convert_model_output(model_output, sample)?;
        self.model_outputs.remove(0);
        self.model_outputs.push(Some(converted.clone()));

        let lower_order_final = self.step_index == self.timesteps.len().saturating_sub(1);
        let prev_sample =
            if self.solver_order == 1 || self.lower_order_nums < 1 || lower_order_final {
                self.first_order_update(&converted, sample)?
            } else {
                self.second_order_update(sample)?
            };

        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        Ok(prev_sample)
    }

    fn convert_model_output(&self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let (alpha_t, sigma_t) = sigma_to_alpha_sigma_t(sigma);
        match self.prediction_type.as_str() {
            "v_prediction" => sample
                .broadcast_mul(&Tensor::new(alpha_t as f32, sample.device())?)?
                .broadcast_sub(
                    &model_output
                        .broadcast_mul(&Tensor::new(sigma_t as f32, model_output.device())?)?,
                ),
            "sample" => Ok(model_output.clone()),
            _ => sample
                .broadcast_sub(
                    &model_output
                        .broadcast_mul(&Tensor::new(sigma_t as f32, model_output.device())?)?,
                )?
                .broadcast_div(&Tensor::new(alpha_t as f32, sample.device())?),
        }
    }

    fn first_order_update(&self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma_t = self.sigmas[self.step_index + 1];
        let sigma_s = self.sigmas[self.step_index];
        let (alpha_t, sigma_t_hat) = sigma_to_alpha_sigma_t(sigma_t);
        let (alpha_s, sigma_s_hat) = sigma_to_alpha_sigma_t(sigma_s);
        let lambda_t = alpha_t.ln() - sigma_t_hat.ln();
        let lambda_s = alpha_s.ln() - sigma_s_hat.ln();
        let h = lambda_t - lambda_s;

        let sample_scale = (sigma_t_hat / sigma_s_hat) as f32;
        let model_scale = (alpha_t * ((-h).exp() - 1.0)) as f32;
        sample
            .broadcast_mul(&Tensor::new(sample_scale, sample.device())?)?
            .broadcast_sub(
                &model_output.broadcast_mul(&Tensor::new(model_scale, model_output.device())?)?,
            )
    }

    fn second_order_update(&self, sample: &Tensor) -> Result<Tensor> {
        let m0 = self.model_outputs[1]
            .as_ref()
            .expect("current model output must be available");
        let m1 = self.model_outputs[0]
            .as_ref()
            .expect("previous model output must be available");

        let sigma_t = self.sigmas[self.step_index + 1];
        let sigma_s0 = self.sigmas[self.step_index];
        let sigma_s1 = self.sigmas[self.step_index - 1];
        let (alpha_t, sigma_t_hat) = sigma_to_alpha_sigma_t(sigma_t);
        let (alpha_s0, sigma_s0_hat) = sigma_to_alpha_sigma_t(sigma_s0);
        let (alpha_s1, sigma_s1_hat) = sigma_to_alpha_sigma_t(sigma_s1);
        let lambda_t = alpha_t.ln() - sigma_t_hat.ln();
        let lambda_s0 = alpha_s0.ln() - sigma_s0_hat.ln();
        let lambda_s1 = alpha_s1.ln() - sigma_s1_hat.ln();

        let h = lambda_t - lambda_s0;
        let h0 = lambda_s0 - lambda_s1;
        let r0 = h0 / h;
        let d1 =
            (m0.broadcast_sub(m1)?).broadcast_mul(&Tensor::new((1.0 / r0) as f32, m0.device())?)?;

        let sample_term = sample.broadcast_mul(&Tensor::new(
            (sigma_t_hat / sigma_s0_hat) as f32,
            sample.device(),
        )?)?;
        let coeff = alpha_t * ((-h).exp() - 1.0);
        let d0_term = m0.broadcast_mul(&Tensor::new(coeff as f32, m0.device())?)?;
        let d1_term = d1.broadcast_mul(&Tensor::new((0.5 * coeff) as f32, d1.device())?)?;
        sample_term.broadcast_sub(&d0_term.broadcast_add(&d1_term)?)
    }
}

fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one = Tensor::ones_like(scale)?;
    x.broadcast_mul(&one.broadcast_add(scale)?)?
        .broadcast_add(shift)
}

fn rms_norm_without_weight(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    x_f32
        .broadcast_div(&(variance + eps)?.sqrt()?)?
        .to_dtype(dtype)
}

fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    let device = timesteps.device().clone();
    let timesteps = timesteps
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let half = dim / 2;
    let mut data = Vec::with_capacity(timesteps.len() * dim);
    let freqs = (0..half)
        .map(|index| (-10_000f32.ln() * index as f32 / half as f32).exp())
        .collect::<Vec<_>>();

    let batch = timesteps.len();
    for timestep in &timesteps {
        for freq in &freqs {
            data.push((timestep * freq).cos());
        }
        for freq in &freqs {
            data.push((timestep * freq).sin());
        }
        if dim % 2 == 1 {
            data.push(0.0);
        }
    }

    Tensor::from_vec(data, (batch, dim), &device)
}

fn betas_for_alpha_bar(num_diffusion_timesteps: usize) -> Vec<f64> {
    let mut betas = Vec::with_capacity(num_diffusion_timesteps);
    for index in 0..num_diffusion_timesteps {
        let t1 = index as f64 / num_diffusion_timesteps as f64;
        let t2 = (index + 1) as f64 / num_diffusion_timesteps as f64;
        let alpha_bar_1 = ((t1 + 0.008) / 1.008 * std::f64::consts::PI / 2.0)
            .cos()
            .powi(2);
        let alpha_bar_2 = ((t2 + 0.008) / 1.008 * std::f64::consts::PI / 2.0)
            .cos()
            .powi(2);
        betas.push((1.0 - alpha_bar_2 / alpha_bar_1).min(0.999));
    }
    betas
}

fn interpolate_sigma(sigmas: &[f64], timestep: f64) -> f64 {
    let low = timestep.floor() as usize;
    let high = timestep.ceil() as usize;
    if low == high {
        return sigmas[low];
    }
    let weight = timestep - low as f64;
    sigmas[low] * (1.0 - weight) + sigmas[high] * weight
}

fn sigma_to_alpha_sigma_t(sigma: f64) -> (f64, f64) {
    let alpha_t = 1.0 / (sigma * sigma + 1.0).sqrt();
    let sigma_t = sigma * alpha_t;
    (alpha_t, sigma_t)
}
