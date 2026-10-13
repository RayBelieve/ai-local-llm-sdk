use std::path::Path;

use model_sdk::{task::TaskControl, ModelSdk, TranscriptionOptions};

#[test]
#[ignore = "requires locally installed engine, model, FFmpeg, and speech audio"]
fn transcribes_with_installed_resources() {
    let root = std::env::var("MODEL_SDK_SMOKE_ROOT").unwrap();
    let audio = std::env::var("MODEL_SDK_SMOKE_AUDIO").unwrap();
    let engine = std::env::var("MODEL_SDK_SMOKE_ENGINE").unwrap();
    let model = std::env::var("MODEL_SDK_SMOKE_MODEL").unwrap();
    let sdk = ModelSdk::new(root).unwrap();
    let options = TranscriptionOptions {
        align_timestamps: std::env::var("MODEL_SDK_SMOKE_ALIGN")
            .map_or(true, |value| value != "off"),
        language: std::env::var("MODEL_SDK_SMOKE_LANGUAGE").ok(),
    };
    let task = TaskControl::new(|stage, percent, message| {
        println!("{stage} {percent}% {message}");
    });
    let segments = sdk
        .transcribe_with_options(
            &engine,
            &model,
            Path::new(&audio),
            Path::new("ffmpeg"),
            true,
            &options,
            &task,
        )
        .unwrap();
    assert!(segments
        .iter()
        .all(|segment| segment.end > segment.start && !segment.text.trim().is_empty()));
    assert!(!segments.is_empty());
    if !options.align_timestamps {
        assert_eq!(segments.len(), 1);
    }
}
