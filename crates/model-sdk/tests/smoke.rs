use std::path::Path;

use model_sdk::{
    task::TaskControl, ChatMessage, ChatRole, GenerationOptions, ModelDescriptor, ModelSdk,
    TranscriptionOptions, LLAMA_ENGINE_ID, TEXT_GENERATION,
};

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

#[cfg(windows)]
#[test]
#[ignore = "requires a local llama-server.exe and text GGUF model"]
fn generates_with_local_llama_server() {
    let engine = std::env::var("MODEL_SDK_LLAMA_ENGINE").unwrap();
    let model = std::env::var("MODEL_SDK_LLAMA_MODEL").unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("model-sdk-llama-smoke-{nonce}"));
    let sdk = ModelSdk::new(&root).unwrap();
    sdk.register_local_engine(LLAMA_ENGINE_ID, Path::new(&engine))
        .unwrap();
    sdk.register_local_model(
        ModelDescriptor {
            id: "local-text".into(),
            label: "Local text model".into(),
            format: "llama-gguf".into(),
            capabilities: vec![TEXT_GENERATION.into()],
        },
        Path::new(&model),
    )
    .unwrap();
    let output = sdk
        .generate_text(
            LLAMA_ENGINE_ID,
            "local-text",
            "Reply with a single short word: hello",
            16,
            &TaskControl::default(),
        )
        .unwrap();
    assert!(!output.trim().is_empty());
    let output = sdk
        .generate_text(
            LLAMA_ENGINE_ID,
            "local-text",
            "Reply with a single short word: world",
            16,
            &TaskControl::default(),
        )
        .unwrap();
    assert!(!output.trim().is_empty());
    let jobs = root.join("jobs");
    assert_eq!(std::fs::read_dir(&jobs).unwrap().count(), 1);

    let messages = [
        ChatMessage {
            role: ChatRole::System,
            content: "Reply briefly.".into(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: "Say hello.".into(),
        },
        ChatMessage {
            role: ChatRole::Assistant,
            content: output,
        },
        ChatMessage {
            role: ChatRole::User,
            content: "Now say goodbye.".into(),
        },
    ];
    let options = GenerationOptions {
        max_tokens: 32,
        temperature: Some(0.2),
        top_p: Some(0.9),
        ..GenerationOptions::default()
    };
    let mut deltas = String::new();
    let streamed = sdk
        .generate_chat_stream(
            LLAMA_ENGINE_ID,
            "local-text",
            &messages,
            &options,
            &TaskControl::default(),
            |delta| deltas.push_str(delta),
        )
        .unwrap();
    assert_eq!(streamed, deltas);
    assert!(!streamed.trim().is_empty());
    assert!(!sdk
        .generate_chat(
            LLAMA_ENGINE_ID,
            "local-text",
            &messages,
            &options,
            &TaskControl::default(),
        )
        .unwrap()
        .trim()
        .is_empty());

    let too_long = "word ".repeat(20000);
    assert!(sdk
        .generate_text(
            LLAMA_ENGINE_ID,
            "local-text",
            &too_long,
            128,
            &TaskControl::default(),
        )
        .unwrap_err()
        .contains("上下文"));
    let (started, ready) = std::sync::mpsc::channel();
    let control = std::sync::Arc::new(TaskControl::new(move |stage, percent, _| {
        if stage == "generate" && percent == 30 {
            let _ = started.send(());
        }
    }));
    std::thread::scope(|scope| {
        let running = control.clone();
        let sdk = &sdk;
        let handle = scope.spawn(move || {
            sdk.generate_text(
                LLAMA_ENGINE_ID,
                "local-text",
                "Write a long answer about local inference.",
                256,
                &running,
            )
        });
        ready
            .recv_timeout(std::time::Duration::from_secs(15))
            .unwrap();
        control.cancel().unwrap();
        assert!(handle.join().unwrap().is_err());
    });
    let output = sdk
        .generate_text(
            LLAMA_ENGINE_ID,
            "local-text",
            "Reply with a single short word: done",
            16,
            &TaskControl::default(),
        )
        .unwrap();
    assert!(!output.trim().is_empty());
    assert_eq!(std::fs::read_dir(&jobs).unwrap().count(), 3);
    sdk.stop_text_generation().unwrap();
    drop(sdk);
    std::fs::remove_dir_all(root).unwrap();
}
