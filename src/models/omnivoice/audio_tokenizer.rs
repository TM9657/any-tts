use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::layers::conv::{Conv1d, ConvTranspose1d};

use super::config::OmniVoiceAudioTokenizerConfig;

fn snake1d(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let sin_sq = alpha.broadcast_mul(x)?.sin()?.sqr()?;
    let inv_alpha = (alpha + 1e-9)?.recip()?;
    x.add(&sin_sq.broadcast_mul(&inv_alpha)?)
}

fn trim_residual_input(input: &Tensor, target: &Tensor) -> Result<Tensor> {
    let input_len = input.dim(2)?;
    let target_len = target.dim(2)?;
    if input_len <= target_len {
        return Ok(input.clone());
    }
    let padding = (input_len - target_len) / 2;
    input.narrow(2, padding, target_len)
}

struct OmniVoiceVectorQuantizer {
    codebook: Tensor,
    project_out: Linear,
}

impl OmniVoiceVectorQuantizer {
    fn load(config: &OmniVoiceAudioTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let codebook = vb
            .pp("codebook")
            .get((config.codebook_size, config.codebook_dim), "embed")?;
        let project_out = candle_nn::linear(
            config.codebook_dim,
            config.acoustic_model_config.decoder_hidden_size,
            vb.pp("project_out"),
        )?;
        Ok(Self {
            codebook,
            project_out,
        })
    }

    fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = indices.dims2()?;
        let flat = indices.flatten_all()?;
        let quantized = self.codebook.index_select(&flat, 0)?;
        let quantized = quantized.reshape((batch, seq_len, self.codebook.dim(1)?))?;
        let projected = self.project_out.forward(&quantized)?;
        projected.transpose(1, 2)
    }
}

struct OmniVoiceResidualVectorQuantizer {
    quantizers: Vec<OmniVoiceVectorQuantizer>,
}

impl OmniVoiceResidualVectorQuantizer {
    fn load(
        config: &OmniVoiceAudioTokenizerConfig,
        num_quantizers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut quantizers = Vec::with_capacity(num_quantizers);
        for index in 0..num_quantizers {
            quantizers.push(OmniVoiceVectorQuantizer::load(
                config,
                vb.pp(format!("quantizers.{}", index)),
            )?);
        }
        Ok(Self { quantizers })
    }

    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut summed: Option<Tensor> = None;
        for (index, quantizer) in self.quantizers.iter().enumerate() {
            let indices = codes.i((.., index, ..))?;
            let decoded = quantizer.decode(&indices)?;
            summed = Some(match summed {
                Some(acc) => acc.add(&decoded)?,
                None => decoded,
            });
        }
        summed.ok_or_else(|| {
            candle_core::Error::Msg("OmniVoice audio decoder has no quantizers".to_string())
        })
    }
}

struct DacResidualUnit {
    snake1_alpha: Tensor,
    conv1: Conv1d,
    snake2_alpha: Tensor,
    conv2: Conv1d,
}

impl DacResidualUnit {
    fn load(channels: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let pad = 3 * dilation;
        let snake1_alpha = vb.pp("snake1").get((1, channels, 1), "alpha")?;
        let conv1 = Conv1d::load(
            channels,
            channels,
            7,
            1,
            pad,
            dilation,
            1,
            true,
            vb.pp("conv1"),
        )?;
        let snake2_alpha = vb.pp("snake2").get((1, channels, 1), "alpha")?;
        let conv2 = Conv1d::load(channels, channels, 1, 1, 0, 1, 1, true, vb.pp("conv2"))?;
        Ok(Self {
            snake1_alpha,
            conv1,
            snake2_alpha,
            conv2,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&snake1d(hidden, &self.snake1_alpha)?)?;
        let h = self.conv2.forward(&snake1d(&h, &self.snake2_alpha)?)?;
        trim_residual_input(hidden, &h)?.add(&h)
    }
}

struct DacDecoderBlock {
    snake1_alpha: Tensor,
    conv_t1: ConvTranspose1d,
    res_unit1: DacResidualUnit,
    res_unit2: DacResidualUnit,
    res_unit3: DacResidualUnit,
}

impl DacDecoderBlock {
    fn load(
        config: &OmniVoiceAudioTokenizerConfig,
        stride: usize,
        stride_index: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_dim = config.acoustic_model_config.decoder_hidden_size / (1usize << stride_index);
        let output_dim =
            config.acoustic_model_config.decoder_hidden_size / (1usize << (stride_index + 1));
        let snake1_alpha = vb.pp("snake1").get((1, input_dim, 1), "alpha")?;
        let conv_t1 = ConvTranspose1d::load(
            input_dim,
            output_dim,
            2 * stride,
            stride,
            (stride + 1) / 2,
            stride % 2,
            1,
            true,
            vb.pp("conv_t1"),
        )?;
        let res_unit1 = DacResidualUnit::load(output_dim, 1, vb.pp("res_unit1"))?;
        let res_unit2 = DacResidualUnit::load(output_dim, 3, vb.pp("res_unit2"))?;
        let res_unit3 = DacResidualUnit::load(output_dim, 9, vb.pp("res_unit3"))?;
        Ok(Self {
            snake1_alpha,
            conv_t1,
            res_unit1,
            res_unit2,
            res_unit3,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let h = snake1d(hidden, &self.snake1_alpha)?;
        let h = self.conv_t1.forward(&h)?;
        let h = self.res_unit1.forward(&h)?;
        let h = self.res_unit2.forward(&h)?;
        self.res_unit3.forward(&h)
    }
}

struct DacDecoder {
    conv1: Conv1d,
    blocks: Vec<DacDecoderBlock>,
    snake1_alpha: Tensor,
    conv2: Conv1d,
}

impl DacDecoder {
    fn load(config: &OmniVoiceAudioTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let conv1 = Conv1d::load(
            config.acoustic_model_config.hidden_size,
            config.acoustic_model_config.decoder_hidden_size,
            7,
            1,
            3,
            1,
            1,
            true,
            vb.pp("conv1"),
        )?;
        let mut blocks = Vec::with_capacity(config.acoustic_model_config.upsampling_ratios.len());
        for (index, stride) in config
            .acoustic_model_config
            .upsampling_ratios
            .iter()
            .copied()
            .enumerate()
        {
            blocks.push(DacDecoderBlock::load(
                config,
                stride,
                index,
                vb.pp(format!("block.{}", index)),
            )?);
        }
        let output_dim = config.acoustic_model_config.decoder_hidden_size
            / (1usize << config.acoustic_model_config.upsampling_ratios.len());
        let snake1_alpha = vb.pp("snake1").get((1, output_dim, 1), "alpha")?;
        let conv2 = Conv1d::load(output_dim, 1, 7, 1, 3, 1, 1, true, vb.pp("conv2"))?;
        Ok(Self {
            conv1,
            blocks,
            snake1_alpha,
            conv2,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let mut h = self.conv1.forward(hidden)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let h = snake1d(&h, &self.snake1_alpha)?;
        self.conv2.forward(&h)?.clamp(-1f32, 1f32)
    }
}

pub struct OmniVoiceAudioTokenizerDecoder {
    quantizer: OmniVoiceResidualVectorQuantizer,
    fc2: Linear,
    acoustic_decoder: DacDecoder,
    sample_rate: u32,
}

impl OmniVoiceAudioTokenizerDecoder {
    pub fn load(
        config: &OmniVoiceAudioTokenizerConfig,
        num_quantizers: usize,
        vb: VarBuilder,
        _device: &Device,
    ) -> Result<Self> {
        let quantizer =
            OmniVoiceResidualVectorQuantizer::load(config, num_quantizers, vb.pp("quantizer"))?;
        let fc2 = candle_nn::linear(
            config.acoustic_model_config.decoder_hidden_size,
            config.acoustic_model_config.hidden_size,
            vb.pp("fc2"),
        )?;
        let acoustic_decoder = DacDecoder::load(config, vb.pp("acoustic_decoder"))?;
        Ok(Self {
            quantizer,
            fc2,
            acoustic_decoder,
            sample_rate: config.sample_rate,
        })
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let quantized = self.quantizer.decode(codes)?;
        let quantized = self
            .fc2
            .forward(&quantized.transpose(1, 2)?)?
            .transpose(1, 2)?;
        self.acoustic_decoder.forward(&quantized)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
