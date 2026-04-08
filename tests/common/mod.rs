//! Shared test utilities.

use any_tts::audio::AudioSamples;

/// Validate basic audio properties.
pub fn assert_valid_audio(audio: &AudioSamples) {
    assert!(!audio.is_empty(), "Audio should not be empty");
    assert!(audio.sample_rate > 0, "Sample rate must be positive");
    assert_eq!(audio.channels, 1, "Audio should be mono");
    assert!(audio.duration_secs() > 0.0, "Duration should be positive");

    // All samples should be in valid range
    for &sample in &audio.samples {
        assert!(
            sample.is_finite(),
            "Audio sample must be finite, got {}",
            sample
        );
        assert!(
            (-1.5..=1.5).contains(&sample),
            "Audio sample out of range: {}",
            sample
        );
    }
}
