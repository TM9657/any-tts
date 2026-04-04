<p align="center">
  <img src="https://flow-like.com/favicon.svg" alt="Flow-Like icon in use here" width="84" height="84">
</p>

# any-tts

<p align="center">
  Rust-native text-to-speech for modern open models.
</p>

<p align="center">
  <a href="https://github.com/TM9657/any-tts"><img src="https://img.shields.io/badge/repo-TM9657%2Fany--tts-111111" alt="Repository"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/crate_license-MIT%20OR%20Apache--2.0-2d6cdf" alt="Crate license"></a>
    <img src="https://img.shields.io/badge/models-5_public%20backends-0a7f5a" alt="Public models">
  <img src="https://img.shields.io/badge/runtime-Candle%20CPU%20%7C%20CUDA%20%7C%20Metal%20%7C%20Accelerate-8a4fff" alt="Backends">
</p>

The Flow-Like icon above is intentionally in use here at the top of this README.

any-tts is a Rust text-to-speech library built around Candle with one trait-based API for multiple open-weight model families. You can point it at local files, hand it explicit paths from your own cache, or let it resolve missing assets from Hugging Face and keep the synthesis call site unchanged.

If you want one Rust TTS surface for small local models, multilingual research checkpoints, and agent-oriented voice stacks without rewriting your application around each model family, this is the repo.

## Why this repo exists

- One API for Kokoro, OmniVoice, Qwen3-TTS, VibeVoice, and Voxtral.
- Native Rust backends across the public model surface.
- Local path loading, per-file wiring, or Hugging Face fallback.
- CPU first, GPU when available: CUDA, Metal, and Accelerate build targets.
- Request-level control for `language`, `voice`, `instruct`, `max_tokens`, `temperature`, and `cfg_scale`.
- WAV output everywhere and optional MP3 export through the `mp3` feature.

## Public model support

| Model | Status in any-tts | Default upstream | Best at | Main tradeoff | Model license |
| --- | --- | --- | --- | --- | --- |
| Kokoro-82M | Public, native, lightweight | `hexgrad/Kokoro-82M` | Fast local TTS with small weights | Requires espeak-based phonemization; default open release is mostly preset voices | Apache-2.0 |
| OmniVoice | Public, native | `k2-fsa/OmniVoice` | Huge language coverage and instruct-driven voice design | The current Rust backend does not yet expose upstream zero-shot cloning | Apache-2.0 |
| Qwen3-TTS-12Hz-1.7B | Public, native | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Strong multilingual control, named speakers, and instruct handling | Heavy weights and extra speech-tokenizer assets | Apache-2.0 |
| VibeVoice-1.5B | Public, native | `microsoft/VibeVoice-1.5B` | Long-form multi-speaker speech diffusion with native Rust inference | Still early and currently optimized for single-request parity work rather than streaming performance | MIT |
| Voxtral-4B-TTS-2603 | Public, native | `mistralai/Voxtral-4B-TTS-2603` | Production-style voice agents, preset voices, low-latency oriented stack | Largest backend here and not commercially permissive | CC BY-NC 4.0 |

Important: the Rust crate is dual licensed under `MIT OR Apache-2.0`. The model weights are not. Always check the model-specific license before shipping. Voxtral is the one that changes the deployment story the most because its published checkpoint is `CC BY-NC 4.0`.

## Referenced but not yet public API

| Model | Where it appears in this repo | Current status | Upstream license | Notes |
| --- | --- | --- | --- | --- |
| KugelAudio-0-Open | `src/models/kugelaudio/` and `examples/generate_kugelaudio.rs` | In-tree experiment, not exported from the public `ModelType` enum | MIT | Focused on 24 European languages and pre-encoded voices, but the current example targets a model variant that is not part of the crate's exported API surface yet. |

That split matters. The README below treats Kokoro, OmniVoice, Qwen3-TTS, VibeVoice, and Voxtral as supported top-level backends, and it treats KugelAudio as work in progress.

## Installation

For CPU-only builds, a recent stable Rust toolchain is enough. For GPU builds, compile the feature set that matches your machine. For Kokoro, install an espeak-ng compatible phonemizer on the host because the backend uses `espeak-rs` to turn text into IPA before synthesis.

Add the crate from git:

```toml
[dependencies]
any-tts = { git = "https://github.com/TM9657/any-tts" }
```

Or opt into a smaller feature set:

```toml
[dependencies]
any-tts = { git = "https://github.com/TM9657/any-tts", default-features = false, features = ["kokoro", "download", "metal"] }
```

### Feature flags

By default the crate enables `qwen3-tts`, `kokoro`, `omnivoice`, `vibevoice`, `voxtral`, and `download`.

| Feature | What it does |
| --- | --- |
| `kokoro` | Enables the Kokoro backend. |
| `omnivoice` | Enables the native OmniVoice backend. |
| `qwen3-tts` | Enables the Qwen3-TTS backend. |
| `vibevoice` | Enables the native VibeVoice backend. |
| `voxtral` | Enables the native Voxtral backend. |
| `download` | Allows missing model files to be pulled from Hugging Face Hub. |
| `cuda` | Builds Candle with CUDA support. |
| `metal` | Builds Candle with Metal support for Apple GPUs. |
| `accelerate` | Enables Apple Accelerate support for CPU-heavy Apple builds. |
| `mp3` | Enables MP3 export through `mp3lame-encoder`. |

### Backend selection

- `DeviceSelection::Auto` tries CUDA first, then Metal, then CPU.
- `DeviceSelection::Cpu`, `DeviceSelection::Cuda(0)`, and `DeviceSelection::Metal(0)` let you force the runtime target.
- `DType` can be set to `F32`, `F16`, or `BF16`.
- On CPU, models that cannot safely run BF16 fall back to `F32`.
- The native OmniVoice helper prefers `cuda:0 (bf16)`, then `metal:0 (f16)`, then `cpu (f32)`.

## Quick start

```rust,no_run
use any_tts::{load_model, ModelType, SynthesisRequest, TtsConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_model(
        TtsConfig::new(ModelType::Qwen3Tts)
            .with_model_path("./models/Qwen3-TTS")
    )?;

    let audio = model.synthesize(
        &SynthesisRequest::new("Hello from Rust TTS.")
            .with_language("English")
            .with_voice("Ryan")
            .with_instruct("Calm, clear, slightly upbeat."),
    )?;

    audio.save_wav("hello.wav")?;
    Ok(())
}
```

## File resolution flow

any-tts resolves model assets in three tiers, in this order:

1. Explicit files you set on `TtsConfig` with methods like `with_config_file()` or `with_weight_file()`.
2. Auto-discovery from `with_model_path()` using the expected filenames for that backend.
3. Hugging Face fallback through the `download` feature.

That means you can mix strategies. A service with its own artifact cache can hand over a few exact files and let the crate discover or download the rest.

## Request controls

`SynthesisRequest` keeps the per-call control surface stable across models.

| Field | Purpose | Notes |
| --- | --- | --- |
| `text` | Input text to synthesize | Required for every backend. |
| `language` | Language tag or model-specific language name | Supports ISO tags in several backends and `auto` where available. |
| `voice` | Named speaker or preset voice | Works for Kokoro, Qwen3 CustomVoice, and Voxtral. OmniVoice rejects named voices. |
| `instruct` | Natural-language style control | Most useful on OmniVoice and Qwen3. |
| `max_tokens` | Upper bound on generated codec/audio tokens | Helpful for latency testing and smoke tests. |
| `temperature` | Sampling temperature | Supported where the backend uses it. |
| `cfg_scale` | Classifier-free guidance scale | Used by OmniVoice and other backends that expose CFG-like control. |
| `reference_audio` | Reference clip for voice cloning | Only partially supported today; unsupported backends return an explicit error. |
| `voice_embedding` | Precomputed embedding payload | Currently reusable with backends that accept embeddings directly. |

## Examples in this repo

These are the example entry points that match the current public crate surface:

```bash
cargo run --example generate_kokoro --release
cargo run --example generate_qwen3_tts --release
cargo run --example generate_vibevoice --release --no-default-features --features vibevoice,download,metal
cargo run --example generate_voxtral --release
cargo run --example generate_omnivoice --release --no-default-features --features omnivoice,download,metal
cargo run --example benchmark_omnivoice --release --no-default-features --features omnivoice,download,metal -- --warmup 1 --iterations 3
```

Outputs are written under `output/` by the example binaries.

Note on KugelAudio: there is an in-tree `generate_kugelaudio` example, but it currently targets a non-exported `ModelType` variant and should be treated as experimental repo work rather than public API.

## Model guide

### Kokoro-82M

**What it is**

Kokoro is the compact option in this repo: an 82M-parameter StyleTTS2 plus ISTFTNet stack with Apache-licensed weights. In practice, it is the backend you reach for when you want a fast local model, simple deployment, and a much smaller download than the larger multilingual checkpoints.

**What works in any-tts today**

- Native Rust inference.
- Default output at 24 kHz.
- Named preset voices discovered from the `voices/` directory.
- Language tags exposed by the current backend: `en`, `ja`, `zh`, `ko`, `fr`, `de`, `it`, `pt`, `es`, `hi`.
- Optional voice-cloning support only when a checkpoint includes style-encoder weights.

**Pros**

- Small enough to be the practical local-first choice.
- Apache-2.0 model license makes deployment straightforward.
- Good fit for desktop apps, tools, and low-latency local generation.
- Simple model layout relative to the larger codebook-based stacks.

**Cons**

- Depends on espeak-based phonemization, so system setup matters.
- The common open release is mostly about preset voice packs, not raw zero-shot cloning.
- Less expressive control than the bigger instruct-heavy model families.

**License**

- Upstream model weights: Apache-2.0.
- Crate code using the model: `MIT OR Apache-2.0`.

### OmniVoice

**What it is**

OmniVoice is the ambition play in this repo. Upstream, it is a diffusion language model TTS stack aimed at omnilingual zero-shot speech generation with voice design and massive language coverage.

**What works in any-tts today**

- Native Candle backend.
- `language`, `instruct`, `cfg_scale`, and `max_tokens` request controls.
- Automatic runtime preference selection for CPU, CUDA, or Metal.
- Repo-exposed language set: `auto`, `en`, `zh`, `ja`, `ko`, `de`, `fr`, `es`, `pt`, `ru`, `it`.

**What does not work yet in the Rust backend**

- Named voices.
- Reference-audio voice cloning.
- Reusable voice embeddings.

The code returns explicit errors for those cases instead of silently falling back to Python.

**Pros**

- Strong upstream story for language coverage.
- Good fit for instruction-driven voice design.
- Benchmark helper in this repo already makes backend comparisons easy.
- Apache-2.0 model license.

**Cons**

- The current Rust implementation exposes less than the upstream model card promises.
- If your main requirement is zero-shot cloning from reference audio, this backend is not there yet in this crate.
- Heavier than Kokoro and less turnkey than the small local-first path.

**License**

- Upstream model weights: Apache-2.0.
- Crate code using the model: `MIT OR Apache-2.0`.

### Qwen3-TTS

**What it is**

Qwen3-TTS is the control-heavy multilingual option. It uses a discrete multi-codebook language model plus a speech-tokenizer decoder and is designed for named speakers, instruction-following, and multiple TTS operating modes.

**What works in any-tts today**

- Native Rust backend.
- Default path points to `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`.
- Named speaker generation for CustomVoice checkpoints.
- VoiceDesign checkpoints also work when selected through `with_hf_model_id()` or local files.
- 24 kHz output with the extra speech-tokenizer weights resolved alongside the main model.
- Repo-level language support tracks the current checkpoint config and includes `auto`.

**Example: switch from CustomVoice to VoiceDesign**

```rust,no_run
use any_tts::{load_model, ModelType, SynthesisRequest, TtsConfig};

let model = load_model(
    TtsConfig::new(ModelType::Qwen3Tts)
        .with_hf_model_id("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
)?;

let audio = model.synthesize(
    &SynthesisRequest::new("This voice should sound sharp, precise, and quietly confident.")
        .with_language("English")
        .with_instruct("Female presenter voice, low warmth, clear diction, subtle authority."),
)?;
```

**Pros**

- Best overall control surface in the current public crate API.
- Strong multilingual coverage: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian upstream.
- Named speakers are easy to use from a single request builder.
- VoiceDesign gives you a second mode without changing the main crate API.

**Cons**

- Large download and memory footprint compared with Kokoro.
- Requires the separate speech-tokenizer decoder assets in addition to the main weights.
- Upstream Base-model voice cloning exists, but reference-audio cloning is not implemented in this crate yet.

**License**

- Upstream model weights: Apache-2.0.
- Crate code using the model: `MIT OR Apache-2.0`.

### Voxtral-4B-TTS-2603

**What it is**

Voxtral is the biggest public backend in the repo and the most obviously voice-agent-oriented. It pairs a language model with acoustic generation and preset voice embeddings, and the published checkpoint is tuned for multilingual, low-latency TTS deployment scenarios.

**What works in any-tts today**

- Native Rust backend.
- Preset voice selection from the checkpoint's `voice_embedding/` assets.
- Optional direct `voice_embedding` reuse when you already have a compatible embedding.
- Repo-exposed languages: `en`, `fr`, `es`, `de`, `it`, `pt`, `nl`, `ar`, `hi`.
- Default sample rate is resolved from the model config and outputs at 24 kHz with the published checkpoint.

**Pros**

- Best fit here for production-style voice-agent workloads.
- Upstream model card emphasizes streaming and low time-to-first-audio.
- Comes with preset voices and a clear multilingual story.

**Cons**

- The open checkpoint does not ship reference-audio encoder weights, so raw voice cloning is unavailable.
- This is the heaviest public backend in the crate.
- The published model license is `CC BY-NC 4.0`, so commercial deployment needs extra care or a different model choice.

**License**

- Upstream model weights: CC BY-NC 4.0.
- Crate code using the model: `MIT OR Apache-2.0`.

### KugelAudio-0-Open

**What it is**

KugelAudio is the European-language-focused experimental backend that already lives in the repo tree but is not yet wired into the public `load_model()` surface. The upstream project positions it as an open-source AR plus diffusion TTS stack trained for 24 European languages with pre-encoded voices.

**What the repo tells us today**

- There is a full in-tree Rust model implementation.
- The example targets `KugelAudio-0-Open` and expects a large model footprint.
- The Rust code explicitly states that raw reference-audio cloning is not implemented yet.
- The public API does not currently export the model, so treat it as active development rather than a supported stable backend.

**Pros**

- Strong European language positioning.
- MIT-licensed upstream software.
- Clear room for a future public backend if the exported API catches up with the in-tree implementation.

**Cons**

- Not part of the exported crate surface yet.
- Large model size and memory requirements.
- Documentation and examples in this repo should currently be read as experimental for this model.

**License**

- Upstream software repository: MIT.
- Public model and deployment terms should still be verified case by case before production use.

## Usage patterns that hold across models

### Local directory loading

```rust,no_run
use any_tts::{load_model, ModelType, TtsConfig};

let model = load_model(
    TtsConfig::new(ModelType::Kokoro)
        .with_model_path("./models/Kokoro-82M")
)?;
```

### Explicit file-path loading

```rust,no_run
use any_tts::{load_model, ModelType, TtsConfig};

let model = load_model(
    TtsConfig::new(ModelType::Qwen3Tts)
        .with_config_file("/cache/config.json")
        .with_tokenizer_file("/cache/tokenizer.json")
        .with_weight_file("/cache/model-00001-of-00002.safetensors")
        .with_weight_file("/cache/model-00002-of-00002.safetensors")
        .with_speech_tokenizer_weight_file("/cache/speech-tokenizer.safetensors")
)?;
```

### Device and dtype selection

```rust,no_run
use any_tts::config::DType;
use any_tts::{load_model, DeviceSelection, ModelType, TtsConfig};

let model = load_model(
    TtsConfig::new(ModelType::OmniVoice)
        .with_device(DeviceSelection::Metal(0))
        .with_dtype(DType::F16)
)?;
```

## Repo health files

This repo now includes the standard GitHub community files you would expect for an active project:

- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Support Guide](SUPPORT.md)
- Issue templates under `.github/ISSUE_TEMPLATE/`
- A pull request template at `.github/PULL_REQUEST_TEMPLATE.md`

## Contributing

If you are adding a backend, model variant, or new loading flow, keep the public story honest: unsupported features should fail explicitly, examples should match exported API, and docs should separate experimental repo work from supported top-level surfaces.

The short version is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Model weights, runtime backends, and artifact loading all change the risk profile of TTS systems. Please read [SECURITY.md](SECURITY.md) before disclosing a vulnerability publicly.

## License

The crate metadata declares `MIT OR Apache-2.0` for this repository's Rust code. That does not supersede the terms attached to any model weights you download and run through it.

## Status

This repo already has a strong shape: five public native backends, one obvious experimental sixth backend, trait-based loading, and example coverage for the core synthesis paths. The right way to think about it is not "a single-model wrapper" but "a Rust TTS platform layer that is learning how to speak multiple open ecosystems without hiding their differences."