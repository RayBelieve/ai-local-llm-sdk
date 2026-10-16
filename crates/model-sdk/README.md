# Model SDK

`model-sdk` is a Rust library for local inference resources. It does not depend on Tauri and does not ship engine executables or model weights. Applications choose the storage directory and the engine/model pair for each request.

## Current capabilities

- `speech-to-text`: CrispASR with Whisper GGML or Qwen3-ASR 1.7B GGUF models (Q4/Q8, standard/Japanese anime), and OpenVINO CPU with Qwen3-ASR 0.6B INT8.
- `text-generation`: llama.cpp server with a downloaded or locally registered GGUF text model on Windows x86-64. One `ModelSdk` instance reuses its loopback server for the same engine/model pair and stops it when dropped or cancelled.
- `object-detection`: YOLOv8 and YOLO11 ONNX models using ONNX Runtime CPU.
- `image-tagging`: nine WD-14 v2/v3 ONNX models using the same ONNX Runtime CPU engine. `tag_image` returns every tag's original name, category, and score plus the four ratings; applications choose any filtering, translation, or prompt formatting.
- Windows x86-64: automatic installation for the listed engines and models.
- Other platforms: local registration when a compatible runtime is available. MLX is not implemented yet.
- `install_ffmpeg(&task)` downloads and verifies a platform binary on Windows x86-64, macOS x86-64/ARM64, or Linux x86-64/ARM64. The returned path can be passed to `transcribe_with_options`; the SDK never bundles FFmpeg.

The bundled OpenVINO Python worker and its small preprocessing data are adapter code. The Qwen weights, Silero VAD ONNX weights, and Qwen3 ForcedAligner GGUF weights are downloaded or imported from local files. VAD and ForcedAligner weights are fetched on first use when their respective inference paths need them.

## API

```rust
use std::path::Path;
use model_sdk::{ModelSdk, TaskKind, TranscriptionOptions, task::TaskControl};

let sdk = ModelSdk::new("app-resources")?;
let task = TaskControl::new(|stage, percent, message| {
    println!("{stage} {percent}% {message}");
});

let engines = sdk.engines()?;
let models = sdk.models()?;
let ready = sdk.check("crispasr-vulkan", "base", TaskKind::Transcription)?;
if ready.ready {
    let segments = sdk.transcribe_with_options(
        "crispasr-vulkan", "base", Path::new("audio.mp3"), Path::new("ffmpeg"), false,
        &TranscriptionOptions { align_timestamps: false, language: Some("zh".into()) },
        &task,
    )?;
    println!("{} segments", segments.len());
}
```

Use `download_engine` and `download_model` for catalog resources, or `register_local_engine` and `register_local_model` for existing files matching a built-in ID. `register_local_vad_model` and `register_local_aligner_model` import optional weights for offline use. `validate_pair_for(engine_id, model_id, TaskKind)` checks static compatibility; `check` also checks installation, hardware and runtime readiness. `transcribe_with_options` accepts `align_timestamps` and an optional language code (`zh`, `en`, `ja`, `ko`, `yue`, `fr`, `de`, `es`, `pt`, `ru`, `ar`, `th`, `vi`, `id`, `ms`). Disabling alignment returns one segment containing the complete transcript. CrispASR selects the Qwen3 backend from the model profile. `TaskControl::cancel()` stops active child processes and aborts in-flight downloads. See [API.md](API.md) for the complete interface and compatibility model.

For llama.cpp, inspect `engines()` for `llama-cpu`, `llama-vulkan`, `llama-cuda12`, and `llama-cuda13` suitability. `download_engine(engine_id, &task)` installs that exact `b9500` variant into its own directory, including its DLLs. `download_model(HY_MT_MODEL_ID, mirror, &task)` installs the pinned HY-MT1.5-1.8B Q4_K_M GGUF with size and SHA-256 verification. Existing local `llama-server.exe` and matching built-in model weights may also be registered. The engine ZIPs have pinned sizes and are validated by extraction and an executable version check, but do not yet have pinned SHA-256 values. The SDK chooses a 2048-16384 token context from available RAM or, when CUDA is selected, free VRAM and model size. CUDA and Vulkan GPU loads fall back to CPU on failure. Generation accepts 1-1024 output tokens and checks the message token budget before generation. Calls on one SDK instance reuse the server until the engine/model changes, `stop_text_generation()` is called, or the instance is dropped. Each call is stateless: pass prior messages again on subsequent turns.

```rust
use model_sdk::{ModelSdk, task::TaskControl, resources::HY_MT_MODEL_ID};

let sdk = ModelSdk::new("app-resources")?;
let task = TaskControl::default();
sdk.download_engine("llama-cpu", &task)?;
sdk.download_model(HY_MT_MODEL_ID, false, &task)?;
let answer = sdk.generate_text("llama-cpu", HY_MT_MODEL_ID, "Translate into English: 你好", 128, &task)?;
```

For multi-turn generation, pass the full conversation in order. `GenerationOptions` supports `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, and `seed`; omitted fields use llama.cpp defaults. The streaming callback runs on the calling thread; the method also returns the complete text on success. A callback may have received partial text before an error. Cancellation returns an error and stops the active server.

```rust
use model_sdk::{ChatMessage, ChatRole, GenerationOptions};

let messages = vec![
    ChatMessage { role: ChatRole::System, content: "Be concise.".into() },
    ChatMessage { role: ChatRole::User, content: "Hello".into() },
];
let options = GenerationOptions {
    max_tokens: 128,
    temperature: Some(0.7),
    top_p: Some(0.9),
    ..GenerationOptions::default()
};
let answer = sdk.generate_chat_stream(
    "llama-cpu", HY_MT_MODEL_ID, &messages, &options, &TaskControl::default(),
    |delta| print!("{delta}"),
)?;
```

The SDK writes a versioned `sdk-resources.json` under the chosen storage directory. It does not read the workbench's `resources.json`. Old SDK records without a supported version fail with an explicit error; they are not silently migrated or overwritten.

## Checks

Run `cargo test --manifest-path crates/model-sdk/Cargo.toml` from the repository root. The ignored `tests/smoke.rs` test runs an installed engine against speech audio when `MODEL_SDK_SMOKE_ROOT`, `MODEL_SDK_SMOKE_AUDIO`, `MODEL_SDK_SMOKE_ENGINE`, and `MODEL_SDK_SMOKE_MODEL` are set. The ignored VAD and FFmpeg tests download and verify their pinned resources.
The ignored `generates_with_local_llama_server` smoke test uses `MODEL_SDK_LLAMA_ENGINE` and `MODEL_SDK_LLAMA_MODEL` to load a real local text model.
The ignored `tags_image_with_wd14` smoke test uses `MODEL_SDK_WD14_ENGINE` (Python executable), `MODEL_SDK_WD14_MODEL` (directory containing `model.onnx` and `selected_tags.csv`), `MODEL_SDK_WD14_IMAGE`, and optionally `MODEL_SDK_WD14_MODEL_ID`.
