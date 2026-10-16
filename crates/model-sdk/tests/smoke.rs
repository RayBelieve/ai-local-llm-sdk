use std::path::Path;

use model_sdk::{
    task::TaskControl, ChatMessage, ChatRole, DetectionOptions, GenerationOptions, ModelSdk,
    TaskKind, TranscriptionOptions,
};
const LLAMA_ENGINE_ID: &str = "llama-cpu";

#[test]
#[ignore = "requires Kokoro model directory and ONNX Runtime Python with misaki[zh]"]
fn synthesizes_chinese_with_kokoro() {
    let model = std::env::var("MODEL_SDK_KOKORO_MODEL").unwrap();
    let engine = std::env::var("MODEL_SDK_KOKORO_ENGINE").unwrap();
    let root = std::env::temp_dir().join(format!("model-sdk-kokoro-smoke-{}", std::process::id()));
    let sdk = ModelSdk::new(&root).unwrap();
    let task = TaskControl::default();
    sdk.register_local_engine("onnxruntime-cpu", Path::new(&engine))
        .unwrap();
    sdk.register_local_model(model_sdk::resources::KOKORO_MODEL_ID, Path::new(&model))
        .unwrap();
    assert!(
        sdk.check(
            "onnxruntime-cpu",
            model_sdk::resources::KOKORO_MODEL_ID,
            TaskKind::SpeechSynthesis
        )
        .unwrap()
        .ready
    );
    let output = root.join("speech.wav");
    sdk.synthesize_speech(
        "onnxruntime-cpu",
        model_sdk::resources::KOKORO_MODEL_ID,
        "你好，欢迎使用语音合成。",
        1.0,
        &output,
        &task,
    )
    .unwrap();
    let bytes = std::fs::read(output).unwrap();
    assert!(bytes.starts_with(b"RIFF") && bytes.len() > 44);
    drop(sdk);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires a local WD-14 model directory, ONNX Runtime Python, and test image"]
fn tags_image_with_wd14() {
    let image = std::env::var("MODEL_SDK_WD14_IMAGE").unwrap();
    let model = std::env::var("MODEL_SDK_WD14_MODEL").unwrap();
    let engine = std::env::var("MODEL_SDK_WD14_ENGINE").unwrap();
    let model_id =
        std::env::var("MODEL_SDK_WD14_MODEL_ID").unwrap_or_else(|_| "wd-swinv2-tagger-v3".into());
    let root = std::env::temp_dir().join(format!("model-sdk-wd14-smoke-{}", std::process::id()));
    let sdk = ModelSdk::new(&root).unwrap();
    sdk.register_local_engine("onnxruntime-cpu", Path::new(&engine))
        .unwrap();
    sdk.register_local_model(&model_id, Path::new(&model))
        .unwrap();
    let result = sdk
        .tag_image(
            "onnxruntime-cpu",
            &model_id,
            Path::new(&image),
            &TaskControl::default(),
        )
        .unwrap();
    assert_eq!(result.ratings.len(), 4);
    assert!(!result.tags.is_empty());
    assert!(result
        .ratings
        .iter()
        .chain(&result.tags)
        .all(|entry| entry.score.is_finite()));
    drop(sdk);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires a test image and either network access or local ONNX Runtime and YOLO weights"]
fn detects_objects_with_yolo() {
    let image = std::env::var("MODEL_SDK_YOLO_IMAGE").unwrap();
    let model_id =
        std::env::var("MODEL_SDK_YOLO_MODEL_ID").unwrap_or_else(|_| "yolov8n-det-fp32".into());
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("model-sdk-yolo-smoke-{nonce}"));
    let sdk = ModelSdk::new(&root).unwrap();
    let task = TaskControl::default();
    sdk.validate_pair_for("onnxruntime-cpu", &model_id, TaskKind::ObjectDetection)
        .unwrap();
    if let Ok(engine) = std::env::var("MODEL_SDK_YOLO_ENGINE") {
        sdk.register_local_engine("onnxruntime-cpu", Path::new(&engine))
            .unwrap();
    } else {
        sdk.download_engine("onnxruntime-cpu", &task).unwrap();
    }
    if let Ok(model) = std::env::var("MODEL_SDK_YOLO_MODEL") {
        sdk.register_local_model(&model_id, Path::new(&model))
            .unwrap();
    } else {
        sdk.download_model(&model_id, false, &task).unwrap();
    }
    assert!(
        sdk.check("onnxruntime-cpu", &model_id, TaskKind::ObjectDetection)
            .unwrap()
            .ready
    );
    let result = sdk
        .detect_objects(
            "onnxruntime-cpu",
            &model_id,
            Path::new(&image),
            &DetectionOptions::default(),
            &task,
        )
        .unwrap();
    assert!(result.image_width > 0 && result.image_height > 0);
    assert!(result.detections.iter().all(|item| {
        item.class_id < 80
            && item.confidence >= 0.25
            && item.x1 >= 0.0
            && item.y1 >= 0.0
            && item.x2 <= result.image_width as f32
            && item.y2 <= result.image_height as f32
            && item.x1 < item.x2
            && item.y1 < item.y2
    }));
    if let Ok(label) = std::env::var("MODEL_SDK_YOLO_EXPECT_LABEL") {
        assert!(result.detections.iter().any(|item| item.label == label));
    }
    drop(sdk);
    std::fs::remove_dir_all(root).unwrap();
}

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
    sdk.register_local_model(model_sdk::resources::HY_MT_MODEL_ID, Path::new(&model))
        .unwrap();
    let output = sdk
        .generate_text(
            LLAMA_ENGINE_ID,
            model_sdk::resources::HY_MT_MODEL_ID,
            "Reply with a single short word: hello",
            16,
            &TaskControl::default(),
        )
        .unwrap();
    assert!(!output.trim().is_empty());
    let output = sdk
        .generate_text(
            LLAMA_ENGINE_ID,
            model_sdk::resources::HY_MT_MODEL_ID,
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
            model_sdk::resources::HY_MT_MODEL_ID,
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
            model_sdk::resources::HY_MT_MODEL_ID,
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
            model_sdk::resources::HY_MT_MODEL_ID,
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
                model_sdk::resources::HY_MT_MODEL_ID,
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
            model_sdk::resources::HY_MT_MODEL_ID,
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
