//! Apply multiple speech-denoise strengths to an existing WAV or MP3 file.
//!
//! Example:
//!   DENOISE_INPUT=output/vibevoice/vibevoice_semantic_default_512.wav \
//!   cargo run --example denoise_audio --no-default-features

use any_tts::{AudioSamples, DenoiseOptions};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
struct DenoiseVariant {
    label: &'static str,
    description: &'static str,
    options: DenoiseOptions,
}

fn denoise_variants() -> [DenoiseVariant; 5] {
    [
        DenoiseVariant {
            label: "denoised_light",
            description: "light cleanup",
            options: DenoiseOptions {
                noise_reduction: 0.95,
                residual_floor: 0.14,
                wet_mix: 0.72,
                ..DenoiseOptions::default()
            },
        },
        DenoiseVariant {
            label: "denoised_default",
            description: "balanced cleanup",
            options: DenoiseOptions::default(),
        },
        DenoiseVariant {
            label: "denoised_strong",
            description: "strong cleanup",
            options: DenoiseOptions {
                noise_reduction: 1.6,
                residual_floor: 0.055,
                wet_mix: 0.95,
                ..DenoiseOptions::default()
            },
        },
        DenoiseVariant {
            label: "denoised_aggressive",
            description: "aggressive cleanup",
            options: DenoiseOptions {
                noise_reduction: 1.65,
                residual_floor: 0.05,
                wet_mix: 1.0,
                ..DenoiseOptions::default()
            },
        },
        DenoiseVariant {
            label: "denoised_max",
            description: "maximum cleanup",
            options: DenoiseOptions {
                speech_low_hz: 140.0,
                speech_high_hz: 5_200.0,
                noise_estimation_percentile: 0.28,
                noise_reduction: 2.15,
                residual_floor: 0.025,
                wet_mix: 1.0,
                ..DenoiseOptions::default()
            },
        },
    ]
}

fn output_stem(input_path: &Path) -> String {
    input_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .unwrap_or("denoise_demo")
        .to_string()
}

fn output_dir(input_path: &Path) -> PathBuf {
    env::var("DENOISE_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("output/denoise_compare").join(output_stem(input_path)))
}

fn main() {
    let input_path = PathBuf::from(
        env::var("DENOISE_INPUT")
            .expect("Set DENOISE_INPUT to an existing WAV or MP3 file to denoise"),
    );
    let audio = AudioSamples::from_audio_file(&input_path)
        .unwrap_or_else(|err| panic!("Failed to decode {}: {err}", input_path.display()));
    let output_dir = output_dir(&input_path);
    fs::create_dir_all(&output_dir)
        .unwrap_or_else(|err| panic!("Failed to create {}: {err}", output_dir.display()));

    let stem = output_stem(&input_path);
    let base_path = output_dir.join(format!("{stem}_base.wav"));
    audio
        .save_wav(&base_path)
        .unwrap_or_else(|err| panic!("Failed to write {}: {err}", base_path.display()));

    println!("Input       : {}", input_path.display());
    println!("Output dir  : {}", output_dir.display());
    println!("Sample rate : {} Hz", audio.sample_rate);
    println!("Duration    : {:.2}s", audio.duration_secs());
    println!();
    println!("Saved base  : {}", base_path.display());

    for variant in denoise_variants() {
        let cleaned = audio.denoise_speech(variant.options);
        let output_path = output_dir.join(format!("{stem}_{}.wav", variant.label));
        cleaned
            .save_wav(&output_path)
            .unwrap_or_else(|err| panic!("Failed to write {}: {err}", output_path.display()));
        println!("Saved {}: {}", variant.description, output_path.display());
    }
}
