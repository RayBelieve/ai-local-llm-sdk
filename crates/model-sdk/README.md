# Model SDK

`model-sdk` is a Rust library for local inference resources. It does not depend on Tauri and does not ship engine executables or model weights. Applications choose the storage directory and the engine/model pair for each request.

## Current capabilities

- `speech-to-text`: CrispASR with Whisper GGML or Qwen3-ASR 1.7B GGUF models (Q4/Q8, standard/Japanese anime), and OpenVINO CPU with Qwen3-ASR 0.6B INT8.
- Windows x86-64: automatic installation for the listed engines and models.
- Other platforms: local registration when a compatible runtime is available. MLX is not implemented yet.
- `install_ffmpeg(&task)` downloads and verifies a platform binary on Windows x86-64, macOS x86-64/ARM64, or Linux x86-64/ARM64. The returned path can be passed to `transcribe_with_options`; the SDK never bundles FFmpeg.

The bundled OpenVINO Python worker and its small preprocessing data are adapter code. The Qwen weights, Silero VAD ONNX weights, and Qwen3 ForcedAligner GGUF weights are downloaded or imported from local files. VAD and ForcedAligner weights are fetched on first use when their respective inference paths need them.

## API

```rust
use std::path::Path;
use model_sdk::{ModelSdk, TranscriptionOptions, TRANSCRIPTION, task::TaskControl};

let sdk = ModelSdk::new("app-resources")?;
let task = TaskControl::new(|stage, percent, message| {
    println!("{stage} {percent}% {message}");
});

let engines = sdk.engines()?;
let models = sdk.models()?;
let ready = sdk.check("vulkan", "base", TRANSCRIPTION)?;
if ready.ready {
    let segments = sdk.transcribe_with_options(
        "vulkan", "base", Path::new("audio.mp3"), Path::new("ffmpeg"), false,
        &TranscriptionOptions { align_timestamps: false, language: Some("zh".into()) },
        &task,
    )?;
    println!("{} segments", segments.len());
}
```

Use `download_engine` and `download_model` for catalog resources, or `register_local_engine` and `register_local_model` for existing files. `register_local_vad_model` and `register_local_aligner_model` import optional weights for offline use. Local model registration takes a `ModelDescriptor`; a new model using a supported format can be registered without changing the SDK. `transcribe` retains the default aligned/automatic-language behavior; `transcribe_with_options` accepts `align_timestamps` and an optional language code (`zh`, `en`, `ja`, `ko`, `yue`, `fr`, `de`, `es`, `pt`, `ru`, `ar`, `th`, `vi`, `id`, `ms`). Disabling alignment returns one segment containing the complete transcript. CrispASR selects the Qwen3 backend for `.gguf` files and loads the Q5 ForcedAligner only when alignment is enabled. A new engine or model format needs an adapter implementation. `TaskControl::cancel()` stops an active child process and download loops.

The SDK writes `sdk-resources.json` under the chosen storage directory. It imports existing `enginePaths` and `modelPaths` from the subtitle workbench's `resources.json` without moving resource files. The workbench retains its engine/model selection and FFmpeg preference in `resources.json`.

## Checks

Run `cargo test --manifest-path crates/model-sdk/Cargo.toml` from the repository root. The ignored `tests/smoke.rs` test runs an installed engine against speech audio when `MODEL_SDK_SMOKE_ROOT`, `MODEL_SDK_SMOKE_AUDIO`, `MODEL_SDK_SMOKE_ENGINE`, and `MODEL_SDK_SMOKE_MODEL` are set. The ignored VAD and FFmpeg tests download and verify their pinned resources.
