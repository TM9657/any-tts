//! Generate a side-by-side cross-model comparison suite.
//!
//! This example renders the same longer English and German passages for every
//! enabled backend, one model at a time and one runtime at a time, so the run
//! stays within tighter memory budgets.
//!
//! Run with:
//!   cargo run --example generate_comparison_suite --release --features metal -- --runtime all
//!
//! Optional filters:
//!   --runtime cpu|metal|all
//!   --models kokoro,omnivoice,qwen3_tts,vibevoice,voxtral
//!   --config examples/model_comparison_texts.json
//!   --output-root output/model_comparison
//!
//! Output layout:
//!   output/model_comparison/cpu/<model>/<sample>.wav
//!   output/model_comparison/cpu/report.json
//!   output/model_comparison/metal/<model>/<sample>.wav
//!   output/model_comparison/metal/report.json

use any_tts::{load_model, DeviceSelection, ModelType, SynthesisRequest, TtsConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_CONFIG_PATH: &str = "examples/model_comparison_texts.json";
const DEFAULT_OMNIVOICE_CFG_SCALE: f64 = 2.0;
const DEFAULT_VIBEVOICE_CFG_SCALE: f64 = 1.3;
const DEFAULT_VIBEVOICE_TEMPERATURE: f64 = 0.0;

#[derive(Debug, Deserialize)]
struct ComparisonConfig {
    output_root: String,
    samples: Vec<ComparisonSample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComparisonSample {
    id: String,
    language_code: String,
    language_name: String,
    text: String,
}

#[derive(Debug)]
struct CliArgs {
    config: PathBuf,
    output_root: Option<PathBuf>,
    runtime: RuntimeFilter,
    models: Option<HashSet<String>>,
}

#[derive(Debug, Clone, Copy)]
enum RuntimeFilter {
    All,
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy)]
struct RuntimePlan {
    key: &'static str,
    device: DeviceSelection,
}

#[derive(Debug, Clone, Copy)]
struct ModelPlan {
    key: &'static str,
    label: &'static str,
    model_type: ModelType,
    model_path: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct SuiteIndex {
    config_path: String,
    output_root: String,
    sample_ids: Vec<String>,
    model_keys: Vec<String>,
    runtimes: Vec<SuiteRuntimeIndex>,
}

#[derive(Debug, Serialize)]
struct SuiteRuntimeIndex {
    runtime: String,
    report_path: String,
    runtime_available: bool,
    runtime_error: Option<String>,
    model_count: usize,
}

#[derive(Debug, Serialize)]
struct RuntimeReport {
    runtime: String,
    device: String,
    output_dir: String,
    config_path: String,
    generated_at_unix_s: u64,
    samples: Vec<ComparisonSample>,
    runtime_available: bool,
    runtime_error: Option<String>,
    models: Vec<ModelReport>,
}

#[derive(Debug, Serialize)]
struct ModelReport {
    key: String,
    label: String,
    device: String,
    hf_model_id: String,
    model_path: Option<String>,
    load_ms: Option<f64>,
    supported_languages: Vec<String>,
    supported_voices: Vec<String>,
    selected_voice: Option<String>,
    sample_rate: Option<u32>,
    sample_results: Vec<SampleReport>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct SampleReport {
    sample_id: String,
    language_code: String,
    output_wav: Option<String>,
    synth_ms: Option<f64>,
    audio_duration_s: Option<f32>,
    realtime_factor: Option<f64>,
    samples: Option<usize>,
    sample_rate: Option<u32>,
    request_language: Option<String>,
    request_voice: Option<String>,
    request_instruct: Option<String>,
    request_cfg_scale: Option<f64>,
    request_temperature: Option<f64>,
    request_max_tokens: Option<usize>,
    error: Option<String>,
}

fn main() {
    let args = parse_args();
    let config = load_config(&args.config);
    assert!(
        !config.samples.is_empty(),
        "{} must contain at least one sample",
        args.config.display()
    );

    let output_root = args
        .output_root
        .clone()
        .unwrap_or_else(|| PathBuf::from(&config.output_root));
    fs::create_dir_all(&output_root)
        .unwrap_or_else(|err| panic!("Failed to create {}: {err}", output_root.display()));

    let model_plans = filter_model_plans(available_model_plans(), args.models.as_ref());
    assert!(
        !model_plans.is_empty(),
        "No enabled models matched the requested filter"
    );

    let runtimes = runtime_plans(args.runtime);
    let mut runtime_reports = Vec::with_capacity(runtimes.len());

    for runtime in runtimes {
        println!("=== Runtime: {} ===", runtime.key);
        let report = run_runtime(
            runtime,
            &model_plans,
            &config.samples,
            &args.config,
            &output_root,
        );
        runtime_reports.push(report);
    }

    let index = SuiteIndex {
        config_path: args.config.display().to_string(),
        output_root: output_root.display().to_string(),
        sample_ids: config
            .samples
            .iter()
            .map(|sample| sample.id.clone())
            .collect(),
        model_keys: model_plans
            .iter()
            .map(|plan| plan.key.to_string())
            .collect(),
        runtimes: runtime_reports
            .iter()
            .map(|report| SuiteRuntimeIndex {
                runtime: report.runtime.clone(),
                report_path: PathBuf::from(&report.output_dir)
                    .join("report.json")
                    .display()
                    .to_string(),
                runtime_available: report.runtime_available,
                runtime_error: report.runtime_error.clone(),
                model_count: report.models.len(),
            })
            .collect(),
    };
    write_json(&output_root.join("index.json"), &index);

    println!(
        "Wrote comparison outputs and reports to {}",
        output_root.display()
    );
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs {
        config: PathBuf::from(DEFAULT_CONFIG_PATH),
        output_root: None,
        runtime: RuntimeFilter::All,
        models: None,
    };

    let mut cli = env::args().skip(1);
    while let Some(flag) = cli.next() {
        match flag.as_str() {
            "--config" => {
                let value = cli.next().expect("--config requires a path");
                args.config = PathBuf::from(value);
            }
            "--output-root" => {
                let value = cli.next().expect("--output-root requires a path");
                args.output_root = Some(PathBuf::from(value));
            }
            "--runtime" => {
                let value = cli.next().expect("--runtime requires cpu, metal, or all");
                args.runtime = parse_runtime_filter(&value);
            }
            "--models" => {
                let value = cli
                    .next()
                    .expect("--models requires a comma-separated list");
                let parsed = value
                    .split(',')
                    .map(canonical_model_key)
                    .filter(|item| !item.is_empty())
                    .collect::<HashSet<_>>();
                args.models = Some(parsed);
            }
            "--help" | "-h" => print_help_and_exit(),
            other => panic!("Unknown flag: {other}"),
        }
    }

    args
}

fn parse_runtime_filter(value: &str) -> RuntimeFilter {
    match value.to_ascii_lowercase().as_str() {
        "all" => RuntimeFilter::All,
        "cpu" => RuntimeFilter::Cpu,
        "metal" => RuntimeFilter::Metal,
        _ => panic!("Unsupported runtime '{value}'. Expected cpu, metal, or all."),
    }
}

fn print_help_and_exit() -> ! {
    let enabled = available_model_plans()
        .into_iter()
        .map(|plan| plan.key)
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "generate_comparison_suite [--config PATH] [--output-root DIR] [--runtime cpu|metal|all] [--models comma,separated,list]\n\nEnabled models in this build: {enabled}"
    );
    std::process::exit(0);
}

fn canonical_model_key(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

fn load_config(path: &Path) -> ComparisonConfig {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("Failed to read {}: {err}", path.display()));
    serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("Failed to parse {}: {err}", path.display()))
}

fn runtime_plans(filter: RuntimeFilter) -> Vec<RuntimePlan> {
    match filter {
        RuntimeFilter::All => vec![
            RuntimePlan {
                key: "cpu",
                device: DeviceSelection::Cpu,
            },
            RuntimePlan {
                key: "metal",
                device: DeviceSelection::Metal(0),
            },
        ],
        RuntimeFilter::Cpu => vec![RuntimePlan {
            key: "cpu",
            device: DeviceSelection::Cpu,
        }],
        RuntimeFilter::Metal => vec![RuntimePlan {
            key: "metal",
            device: DeviceSelection::Metal(0),
        }],
    }
}

fn available_model_plans() -> Vec<ModelPlan> {
    vec![
        #[cfg(feature = "kokoro")]
        ModelPlan {
            key: "kokoro",
            label: "Kokoro-82M",
            model_type: ModelType::Kokoro,
            model_path: Some("./models/Kokoro-82M"),
        },
        #[cfg(feature = "omnivoice")]
        ModelPlan {
            key: "omnivoice",
            label: "OmniVoice",
            model_type: ModelType::OmniVoice,
            model_path: None,
        },
        #[cfg(feature = "qwen3-tts")]
        ModelPlan {
            key: "qwen3_tts",
            label: "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            model_type: ModelType::Qwen3Tts,
            model_path: None,
        },
        #[cfg(feature = "vibevoice")]
        ModelPlan {
            key: "vibevoice",
            label: "VibeVoice-1.5B",
            model_type: ModelType::VibeVoice,
            model_path: None,
        },
        #[cfg(feature = "voxtral")]
        ModelPlan {
            key: "voxtral",
            label: "Voxtral-4B-TTS-2603",
            model_type: ModelType::Voxtral,
            model_path: None,
        },
    ]
}

fn filter_model_plans(
    mut plans: Vec<ModelPlan>,
    requested: Option<&HashSet<String>>,
) -> Vec<ModelPlan> {
    let Some(requested) = requested else {
        return plans;
    };

    let available = plans
        .iter()
        .map(|plan| plan.key.to_string())
        .collect::<HashSet<_>>();
    let mut unknown = requested
        .iter()
        .filter(|key| !available.contains(*key))
        .cloned()
        .collect::<Vec<_>>();
    unknown.sort();
    assert!(
        unknown.is_empty(),
        "Unknown or unavailable models: {}",
        unknown.join(", ")
    );

    plans.retain(|plan| requested.contains(plan.key));
    plans
}

fn run_runtime(
    runtime: RuntimePlan,
    model_plans: &[ModelPlan],
    samples: &[ComparisonSample],
    config_path: &Path,
    output_root: &Path,
) -> RuntimeReport {
    let runtime_dir = output_root.join(runtime.key);
    fs::create_dir_all(&runtime_dir)
        .unwrap_or_else(|err| panic!("Failed to create {}: {err}", runtime_dir.display()));
    let report_path = runtime_dir.join("report.json");

    let mut report = RuntimeReport {
        runtime: runtime.key.to_string(),
        device: runtime.device.label(),
        output_dir: runtime_dir.display().to_string(),
        config_path: config_path.display().to_string(),
        generated_at_unix_s: unix_now(),
        samples: samples.to_vec(),
        runtime_available: false,
        runtime_error: None,
        models: Vec::new(),
    };

    match runtime.device.resolve() {
        Ok(_) => {
            report.runtime_available = true;
            write_json(&report_path, &report);
        }
        Err(err) => {
            report.runtime_error = Some(err.to_string());
            write_json(&report_path, &report);
            return report;
        }
    }

    for plan in model_plans {
        println!("Loading {} on {}", plan.label, runtime.key);
        let model_report = run_model(plan, runtime, samples, &runtime_dir);
        report.models.push(model_report);
        write_json(&report_path, &report);
    }

    report
}

fn run_model(
    plan: &ModelPlan,
    runtime: RuntimePlan,
    samples: &[ComparisonSample],
    runtime_dir: &Path,
) -> ModelReport {
    let model_dir = runtime_dir.join(plan.key);
    fs::create_dir_all(&model_dir)
        .unwrap_or_else(|err| panic!("Failed to create {}: {err}", model_dir.display()));

    let mut config = TtsConfig::new(plan.model_type).with_device(runtime.device);
    if let Some(model_path) = plan.model_path {
        config = config.with_model_path(model_path);
    }

    let hf_model_id = config.effective_hf_model_id().to_string();
    let model_path = config.model_path.clone();
    let load_started = Instant::now();
    let model = match load_model(config) {
        Ok(model) => model,
        Err(err) => {
            return ModelReport {
                key: plan.key.to_string(),
                label: plan.label.to_string(),
                device: runtime.device.label(),
                hf_model_id,
                model_path,
                load_ms: None,
                supported_languages: Vec::new(),
                supported_voices: Vec::new(),
                selected_voice: None,
                sample_rate: None,
                sample_results: Vec::new(),
                error: Some(format!("load failed: {err}")),
            };
        }
    };
    let load_ms = duration_ms(load_started.elapsed());

    let info = model.model_info();
    let supported_languages = model.supported_languages();
    let supported_voices = model.supported_voices();
    let selected_voice = select_voice(plan.model_type, &supported_voices);

    let mut sample_results = Vec::with_capacity(samples.len());
    for sample in samples {
        let output_path = model_dir.join(format!("{}.wav", sample.id));
        let request = build_request(plan.model_type, sample, selected_voice.as_deref());

        println!("  Rendering {} / {}", plan.key, sample.id);
        let synth_started = Instant::now();
        match model.synthesize(&request) {
            Ok(audio) => {
                let synth_ms = duration_ms(synth_started.elapsed());
                match audio.save_wav(&output_path) {
                    Ok(()) => {
                        let duration_s = audio.duration_secs();
                        sample_results.push(SampleReport {
                            sample_id: sample.id.clone(),
                            language_code: sample.language_code.clone(),
                            output_wav: Some(output_path.display().to_string()),
                            synth_ms: Some(synth_ms),
                            audio_duration_s: Some(duration_s),
                            realtime_factor: realtime_factor(synth_ms, duration_s),
                            samples: Some(audio.len()),
                            sample_rate: Some(audio.sample_rate),
                            request_language: request.language.clone(),
                            request_voice: request.voice.clone(),
                            request_instruct: request.instruct.clone(),
                            request_cfg_scale: request.cfg_scale,
                            request_temperature: request.temperature,
                            request_max_tokens: request.max_tokens,
                            error: None,
                        });
                    }
                    Err(err) => {
                        sample_results.push(SampleReport {
                            sample_id: sample.id.clone(),
                            language_code: sample.language_code.clone(),
                            output_wav: None,
                            synth_ms: Some(synth_ms),
                            audio_duration_s: Some(audio.duration_secs()),
                            realtime_factor: realtime_factor(synth_ms, audio.duration_secs()),
                            samples: Some(audio.len()),
                            sample_rate: Some(audio.sample_rate),
                            request_language: request.language.clone(),
                            request_voice: request.voice.clone(),
                            request_instruct: request.instruct.clone(),
                            request_cfg_scale: request.cfg_scale,
                            request_temperature: request.temperature,
                            request_max_tokens: request.max_tokens,
                            error: Some(format!("save failed: {err}")),
                        });
                    }
                }
            }
            Err(err) => {
                sample_results.push(SampleReport {
                    sample_id: sample.id.clone(),
                    language_code: sample.language_code.clone(),
                    output_wav: None,
                    synth_ms: None,
                    audio_duration_s: None,
                    realtime_factor: None,
                    samples: None,
                    sample_rate: None,
                    request_language: request.language.clone(),
                    request_voice: request.voice.clone(),
                    request_instruct: request.instruct.clone(),
                    request_cfg_scale: request.cfg_scale,
                    request_temperature: request.temperature,
                    request_max_tokens: request.max_tokens,
                    error: Some(format!("synthesis failed: {err}")),
                });
            }
        }
    }

    drop(model);

    ModelReport {
        key: plan.key.to_string(),
        label: info.name,
        device: runtime.device.label(),
        hf_model_id,
        model_path,
        load_ms: Some(load_ms),
        supported_languages,
        supported_voices,
        selected_voice,
        sample_rate: Some(info.sample_rate),
        sample_results,
        error: None,
    }
}

fn select_voice(model_type: ModelType, supported_voices: &[String]) -> Option<String> {
    let preferred = match model_type {
        ModelType::Kokoro => &["af_heart"][..],
        ModelType::Qwen3Tts => &["dylan"][..],
        ModelType::Voxtral => &["neutral_male"][..],
        ModelType::VibeVoiceRealtime => &["en-Emma_woman"][..],
        ModelType::OmniVoice | ModelType::VibeVoice => &[][..],
    };

    for candidate in preferred {
        if let Some(voice) = supported_voices
            .iter()
            .find(|voice| voice.as_str() == *candidate)
        {
            return Some(voice.clone());
        }
    }

    supported_voices.first().cloned()
}

fn build_request(
    model_type: ModelType,
    sample: &ComparisonSample,
    selected_voice: Option<&str>,
) -> SynthesisRequest {
    match model_type {
        ModelType::Kokoro => {
            let mut request =
                SynthesisRequest::new(&sample.text).with_language(&sample.language_code);
            if let Some(voice) = selected_voice {
                request = request.with_voice(voice);
            }
            request
        }
        ModelType::OmniVoice => SynthesisRequest::new(&sample.text)
            .with_language(&sample.language_code)
            .with_instruct(omnivoice_instruct(&sample.language_code))
            .with_cfg_scale(DEFAULT_OMNIVOICE_CFG_SCALE),
        ModelType::Qwen3Tts => {
            let mut request =
                SynthesisRequest::new(&sample.text).with_language(&sample.language_code);
            if let Some(voice) = selected_voice {
                request = request.with_voice(voice);
            }
            request
        }
        ModelType::VibeVoice => SynthesisRequest::new(&sample.text)
            .with_cfg_scale(DEFAULT_VIBEVOICE_CFG_SCALE)
            .with_temperature(DEFAULT_VIBEVOICE_TEMPERATURE),
        ModelType::VibeVoiceRealtime => {
            let mut request = SynthesisRequest::new(&sample.text)
                .with_cfg_scale(DEFAULT_VIBEVOICE_CFG_SCALE)
                .with_temperature(DEFAULT_VIBEVOICE_TEMPERATURE);
            if let Some(voice) = selected_voice {
                request = request.with_voice(voice);
            }
            request
        }
        ModelType::Voxtral => {
            let mut request =
                SynthesisRequest::new(&sample.text).with_language(&sample.language_code);
            if let Some(voice) = selected_voice {
                request = request.with_voice(voice);
            }
            request
        }
    }
}

fn omnivoice_instruct(language_code: &str) -> &'static str {
    if language_code.eq_ignore_ascii_case("de") {
        "female, calm delivery, neutral studio voice, standard German accent"
    } else {
        "female, calm delivery, neutral studio voice, clear pacing"
    }
}

fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn realtime_factor(synth_ms: f64, audio_duration_s: f32) -> Option<f64> {
    if audio_duration_s <= 0.0 {
        None
    } else {
        Some((synth_ms / 1000.0) / f64::from(audio_duration_s))
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System clock is before UNIX_EPOCH")
        .as_secs()
}

fn write_json<T: Serialize>(path: &Path, value: &T) {
    let payload = serde_json::to_string_pretty(value)
        .unwrap_or_else(|err| panic!("Failed to serialize {}: {err}", path.display()));
    fs::write(path, payload)
        .unwrap_or_else(|err| panic!("Failed to write {}: {err}", path.display()));
}
