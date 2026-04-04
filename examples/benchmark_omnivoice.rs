//! Benchmark native OmniVoice across the compiled runtime backends.
//!
//! Example:
//!   cargo run --example benchmark_omnivoice --release --no-default-features --features omnivoice,download,metal -- --warmup 1 --iterations 3

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;
use any_tts::models::omnivoice::{preferred_runtime_choice, preferred_runtime_choices, OmniVoiceRuntimeChoice};
use any_tts::{load_model, ModelType, SynthesisRequest, TtsConfig};

#[derive(Debug, Deserialize)]
struct OmniVoiceExampleConfig {
    text: String,
    language: String,
    instruct: String,
    cfg_scale: f64,
    output: String,
}

#[derive(Debug)]
struct BenchmarkArgs {
    config: PathBuf,
    warmup: usize,
    iterations: usize,
    max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    backend: String,
    dtype: String,
    load_ms: Option<f64>,
    warmup_ms: Vec<f64>,
    iteration_ms: Vec<f64>,
    mean_ms: Option<f64>,
    median_ms: Option<f64>,
    sample_rate: Option<u32>,
    samples: Option<usize>,
    duration_s: Option<f32>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    config: String,
    text: String,
    language: String,
    instruct: String,
    output: String,
    warmup: usize,
    iterations: usize,
    requested_max_tokens: Option<usize>,
    preferred_runtime: String,
    candidates: Vec<String>,
    results: Vec<BenchmarkResult>,
    best_runtime: Option<String>,
    best_median_ms: Option<f64>,
}

fn main() {
    let args = parse_args();
    let example = load_example_config(&args.config);
    let choices = preferred_runtime_choices();
    let request = build_request(&example, args.max_tokens);

    let mut results = Vec::with_capacity(choices.len());
    for choice in choices.iter().copied() {
        results.push(run_benchmark(choice, &request, &args));
    }

    let best_result = results
        .iter()
        .filter_map(|result| result.median_ms.map(|median| (median, result.backend.clone(), result.dtype.clone())))
        .min_by(|left, right| left.0.total_cmp(&right.0));

    let best_runtime = best_result
        .as_ref()
        .map(|(_, backend, dtype)| format!("{} ({})", backend, dtype));
    let best_median_ms = best_result.as_ref().map(|(median, _, _)| *median);

    let report = BenchmarkReport {
        config: args.config.display().to_string(),
        text: example.text,
        language: example.language,
        instruct: example.instruct,
        output: example.output,
        warmup: args.warmup,
        iterations: args.iterations,
        requested_max_tokens: args.max_tokens,
        preferred_runtime: preferred_runtime_choice().label(),
        candidates: choices.iter().map(OmniVoiceRuntimeChoice::label).collect(),
        results,
        best_runtime,
        best_median_ms,
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("Failed to serialize benchmark report")
    );
}

fn parse_args() -> BenchmarkArgs {
    let mut args = BenchmarkArgs {
        config: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/omnivoice_example.json"),
        warmup: 1,
        iterations: 3,
        max_tokens: None,
    };

    let mut cli = env::args().skip(1);
    while let Some(flag) = cli.next() {
        match flag.as_str() {
            "--config" => {
                let value = cli.next().expect("--config requires a path");
                args.config = PathBuf::from(value);
            }
            "--warmup" => {
                let value = cli.next().expect("--warmup requires an integer");
                args.warmup = value.parse().expect("--warmup must be an integer");
            }
            "--iterations" => {
                let value = cli.next().expect("--iterations requires an integer");
                args.iterations = value.parse().expect("--iterations must be an integer");
            }
            "--max-tokens" => {
                let value = cli.next().expect("--max-tokens requires an integer");
                args.max_tokens = Some(value.parse().expect("--max-tokens must be an integer"));
            }
            other => panic!("Unknown flag: {other}"),
        }
    }

    assert!(args.iterations > 0, "--iterations must be at least 1");
    args
}

fn load_example_config(path: &Path) -> OmniVoiceExampleConfig {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {err}", path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("Failed to parse {}: {err}", path.display()))
}

fn build_request(example: &OmniVoiceExampleConfig, max_tokens: Option<usize>) -> SynthesisRequest {
    let request = SynthesisRequest::new(&example.text)
        .with_language(&example.language)
        .with_instruct(&example.instruct)
        .with_cfg_scale(example.cfg_scale);

    if let Some(max_tokens) = max_tokens {
        request.with_max_tokens(max_tokens)
    } else {
        request
    }
}

fn run_benchmark(
    choice: OmniVoiceRuntimeChoice,
    request: &SynthesisRequest,
    args: &BenchmarkArgs,
) -> BenchmarkResult {
    let backend = choice.device.label();
    let dtype = choice.dtype_label().to_string();

    let load_started = Instant::now();
    let model = match load_model(
        TtsConfig::new(ModelType::OmniVoice)
            .with_device(choice.device)
            .with_dtype(choice.dtype),
    ) {
        Ok(model) => model,
        Err(err) => {
            return BenchmarkResult {
                backend,
                dtype,
                load_ms: None,
                warmup_ms: Vec::new(),
                iteration_ms: Vec::new(),
                mean_ms: None,
                median_ms: None,
                sample_rate: None,
                samples: None,
                duration_s: None,
                error: Some(format!("load failed: {err}")),
            }
        }
    };
    let load_ms = duration_ms(load_started.elapsed());

    let mut warmup_ms = Vec::with_capacity(args.warmup);
    for _ in 0..args.warmup {
        let started = Instant::now();
        let audio = match model.synthesize(request) {
            Ok(audio) => audio,
            Err(err) => {
                return BenchmarkResult {
                    backend,
                    dtype,
                    load_ms: Some(load_ms),
                    warmup_ms,
                    iteration_ms: Vec::new(),
                    mean_ms: None,
                    median_ms: None,
                    sample_rate: None,
                    samples: None,
                    duration_s: None,
                    error: Some(format!("warmup failed: {err}")),
                }
            }
        };
        black_box(audio.len());
        warmup_ms.push(duration_ms(started.elapsed()));
    }

    let mut iteration_ms = Vec::with_capacity(args.iterations);
    let mut last_audio = None;
    for _ in 0..args.iterations {
        let started = Instant::now();
        let audio = match model.synthesize(request) {
            Ok(audio) => audio,
            Err(err) => {
                return BenchmarkResult {
                    backend,
                    dtype,
                    load_ms: Some(load_ms),
                    warmup_ms,
                    iteration_ms,
                    mean_ms: None,
                    median_ms: None,
                    sample_rate: None,
                    samples: None,
                    duration_s: None,
                    error: Some(format!("iteration failed: {err}")),
                }
            }
        };
        black_box(audio.len());
        iteration_ms.push(duration_ms(started.elapsed()));
        last_audio = Some(audio);
    }

    let audio = last_audio.expect("at least one iteration is required");
    BenchmarkResult {
        backend,
        dtype,
        load_ms: Some(load_ms),
        warmup_ms,
        mean_ms: Some(mean(&iteration_ms)),
        median_ms: Some(median(&iteration_ms)),
        sample_rate: Some(audio.sample_rate),
        samples: Some(audio.len()),
        duration_s: Some(audio.duration_secs()),
        iteration_ms,
        error: None,
    }
}

fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn mean(values: &[f64]) -> f64 {
    values.iter().copied().sum::<f64>() / values.len() as f64
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}