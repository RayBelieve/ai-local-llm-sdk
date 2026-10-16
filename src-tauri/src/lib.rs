use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use model_sdk::{
    task::TaskControl, ChatMessage, ChatRole, DetectionOptions, DetectionResult, EngineInfo,
    GenerationOptions, InstalledModel, ModelSdk, Readiness, TagResult, TaskKind,
    TranscriptionOptions,
};
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager};

#[derive(Default)]
struct AppState {
    sdk: Mutex<Option<Arc<ModelSdk>>>,
    active: Mutex<Option<Arc<TaskControl>>>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Progress {
    stage: String,
    percent: u8,
    message: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Catalog {
    storage_dir: String,
    ffmpeg_path: String,
    engines: Vec<EngineInfo>,
    models: Vec<InstalledModel>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TextDelta {
    run_id: u64,
    delta: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
enum GenerateMode {
    Text,
    Chat,
    Stream,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Deserialize)]
struct MessageInput {
    role: MessageRole,
    content: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateInput {
    engine_id: String,
    model_id: String,
    mode: GenerateMode,
    prompt: String,
    messages: Vec<MessageInput>,
    max_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    run_id: u64,
}

fn data_dir() -> Result<PathBuf, String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let executable_dir = executable.parent().ok_or("无法确定程序所在目录。")?;
    #[cfg(target_os = "macos")]
    let executable_dir = executable
        .ancestors()
        .find(|path| path.extension().is_some_and(|extension| extension == "app"))
        .and_then(Path::parent)
        .unwrap_or(executable_dir);
    let path = executable_dir.join("resources");
    fs::create_dir_all(&path).map_err(|error| {
        format!(
            "无法创建资源目录 {}: {error}。请将程序放在可写目录。",
            path.display()
        )
    })?;
    Ok(path)
}

fn sdk(app: &tauri::AppHandle) -> Result<Arc<ModelSdk>, String> {
    let state = app.state::<AppState>();
    let mut current = state.sdk.lock().map_err(|error| error.to_string())?;
    if current.is_none() {
        *current = Some(Arc::new(ModelSdk::new(data_dir()?)?));
    }
    Ok(current.as_ref().expect("SDK initialized").clone())
}

fn task(app: &tauri::AppHandle) -> Arc<TaskControl> {
    let app = app.clone();
    Arc::new(TaskControl::new(move |stage, percent, message| {
        let _ = app.emit(
            "task-progress",
            Progress {
                stage: stage.into(),
                percent,
                message: message.into(),
            },
        );
    }))
}

async fn run_task<R: Send + 'static>(
    app: tauri::AppHandle,
    work: impl FnOnce(Arc<ModelSdk>, Arc<TaskControl>) -> Result<R, String> + Send + 'static,
) -> Result<R, String> {
    let sdk = sdk(&app)?;
    let control = task(&app);
    let state = app.state::<AppState>();
    {
        let mut active = state.active.lock().map_err(|error| error.to_string())?;
        if active.is_some() {
            return Err("已有任务正在运行。".into());
        }
        *active = Some(control.clone());
    }
    let result = tauri::async_runtime::spawn_blocking(move || work(sdk, control))
        .await
        .map_err(|error| error.to_string());
    *state.active.lock().map_err(|error| error.to_string())? = None;
    result?
}

#[tauri::command]
async fn get_catalog(app: tauri::AppHandle) -> Result<Catalog, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let sdk = sdk(&app)?;
        let managed = sdk.root().join("tools/ffmpeg").join(if cfg!(windows) {
            "ffmpeg.exe"
        } else {
            "ffmpeg"
        });
        Ok(Catalog {
            storage_dir: sdk.root().to_string_lossy().into_owned(),
            ffmpeg_path: if managed.is_file() {
                managed.to_string_lossy().into_owned()
            } else {
                "ffmpeg".into()
            },
            engines: sdk.engines()?,
            models: sdk.models()?,
        })
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn validate_pair(
    app: tauri::AppHandle,
    engine_id: String,
    model_id: String,
    task_kind: TaskKind,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        sdk(&app)?.validate_pair_for(&engine_id, &model_id, task_kind)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn check_pair(
    app: tauri::AppHandle,
    engine_id: String,
    model_id: String,
    task_kind: TaskKind,
) -> Result<Readiness, String> {
    tauri::async_runtime::spawn_blocking(move || sdk(&app)?.check(&engine_id, &model_id, task_kind))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn download_engine(app: tauri::AppHandle, engine_id: String) -> Result<String, String> {
    run_task(app, move |sdk, task| {
        sdk.download_engine(&engine_id, &task)
            .map(|path| path.to_string_lossy().into_owned())
    })
    .await
}

#[tauri::command]
async fn download_model(
    app: tauri::AppHandle,
    model_id: String,
    mirror: bool,
) -> Result<String, String> {
    run_task(app, move |sdk, task| {
        sdk.download_model(&model_id, mirror, &task)
            .map(|path| path.to_string_lossy().into_owned())
    })
    .await
}

#[tauri::command]
async fn register_local_engine(
    app: tauri::AppHandle,
    engine_id: String,
    path: String,
) -> Result<(), String> {
    run_task(app, move |sdk, _| {
        sdk.register_local_engine(&engine_id, Path::new(&path))
    })
    .await
}

#[tauri::command]
async fn register_local_model(
    app: tauri::AppHandle,
    model_id: String,
    path: String,
) -> Result<(), String> {
    run_task(app, move |sdk, _| {
        sdk.register_local_model(&model_id, Path::new(&path))
    })
    .await
}

#[tauri::command]
async fn install_ffmpeg(app: tauri::AppHandle) -> Result<String, String> {
    run_task(app, |sdk, task| {
        sdk.install_ffmpeg(&task)
            .map(|path| path.to_string_lossy().into_owned())
    })
    .await
}

#[tauri::command]
async fn download_aligner(app: tauri::AppHandle) -> Result<String, String> {
    run_task(app, |sdk, task| {
        sdk.ensure_aligner_model(&task)
            .map(|path| path.to_string_lossy().into_owned())
    })
    .await
}

#[tauri::command]
async fn register_local_aligner(app: tauri::AppHandle, path: String) -> Result<String, String> {
    run_task(app, move |sdk, _| {
        sdk.register_local_aligner_model(Path::new(&path))
            .map(|path| path.to_string_lossy().into_owned())
    })
    .await
}

#[tauri::command]
fn allow_preview(app: tauri::AppHandle, path: String) -> Result<(), String> {
    let file = PathBuf::from(path);
    if !file.is_file() {
        return Err("选中的文件不存在。".into());
    }
    let extension = file
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ![
        "mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "aac", "flac", "ogg", "jpg", "jpeg",
        "png", "webp", "bmp",
    ]
    .contains(&extension.as_str())
    {
        return Err("不支持预览此文件格式。".into());
    }
    app.asset_protocol_scope()
        .allow_file(&file)
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn test_transcribe(
    app: tauri::AppHandle,
    engine_id: String,
    model_id: String,
    media_path: String,
    ffmpeg_path: String,
    vad_enabled: bool,
    with_options: bool,
    align_timestamps: bool,
    language: Option<String>,
) -> Result<Vec<model_sdk::subtitle::Segment>, String> {
    run_task(app, move |sdk, task| {
        if with_options {
            sdk.transcribe_with_options(
                &engine_id,
                &model_id,
                Path::new(&media_path),
                Path::new(&ffmpeg_path),
                vad_enabled,
                &TranscriptionOptions {
                    align_timestamps,
                    language,
                },
                &task,
            )
        } else {
            sdk.transcribe(
                &engine_id,
                &model_id,
                Path::new(&media_path),
                Path::new(&ffmpeg_path),
                vad_enabled,
                &task,
            )
        }
    })
    .await
}

#[tauri::command]
async fn test_generate(app: tauri::AppHandle, input: GenerateInput) -> Result<String, String> {
    let events = app.clone();
    run_task(app, move |sdk, task| {
        let options = GenerationOptions {
            max_tokens: input.max_tokens,
            temperature: input.temperature,
            top_p: input.top_p,
            ..GenerationOptions::default()
        };
        let messages = input
            .messages
            .into_iter()
            .map(|message| ChatMessage {
                role: match message.role {
                    MessageRole::System => ChatRole::System,
                    MessageRole::User => ChatRole::User,
                    MessageRole::Assistant => ChatRole::Assistant,
                },
                content: message.content,
            })
            .collect::<Vec<_>>();
        match input.mode {
            GenerateMode::Text => sdk.generate_text(
                &input.engine_id,
                &input.model_id,
                &input.prompt,
                input.max_tokens,
                &task,
            ),
            GenerateMode::Chat => sdk.generate_chat(
                &input.engine_id,
                &input.model_id,
                &messages,
                &options,
                &task,
            ),
            GenerateMode::Stream => sdk.generate_chat_stream(
                &input.engine_id,
                &input.model_id,
                &messages,
                &options,
                &task,
                |delta| {
                    let _ = events.emit(
                        "text-delta",
                        TextDelta {
                            run_id: input.run_id,
                            delta: delta.into(),
                        },
                    );
                },
            ),
        }
    })
    .await
}

#[tauri::command]
async fn test_detect(
    app: tauri::AppHandle,
    engine_id: String,
    model_id: String,
    image_path: String,
    confidence_threshold: f32,
    iou_threshold: f32,
    max_detections: usize,
) -> Result<DetectionResult, String> {
    run_task(app, move |sdk, task| {
        sdk.detect_objects(
            &engine_id,
            &model_id,
            Path::new(&image_path),
            &DetectionOptions {
                confidence_threshold,
                iou_threshold,
                max_detections,
            },
            &task,
        )
    })
    .await
}

#[tauri::command]
async fn test_tag(
    app: tauri::AppHandle,
    engine_id: String,
    model_id: String,
    image_path: String,
) -> Result<TagResult, String> {
    run_task(app, move |sdk, task| {
        sdk.tag_image(&engine_id, &model_id, Path::new(&image_path), &task)
    })
    .await
}

#[tauri::command]
async fn test_synthesize(
    app: tauri::AppHandle,
    engine_id: String,
    model_id: String,
    text: String,
    speed: f32,
) -> Result<String, String> {
    run_task(app.clone(), move |sdk, task| {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let output = sdk
            .root()
            .join("outputs")
            .join(format!("speech-{nonce}.wav"));
        fs::create_dir_all(output.parent().expect("output parent"))
            .map_err(|error| error.to_string())?;
        sdk.synthesize_speech(&engine_id, &model_id, &text, speed, &output, &task)?;
        app.asset_protocol_scope()
            .allow_file(&output)
            .map_err(|error| error.to_string())?;
        Ok(output.to_string_lossy().into_owned())
    })
    .await
}

#[tauri::command]
fn cancel_task(app: tauri::AppHandle) -> Result<(), String> {
    let control = app
        .state::<AppState>()
        .active
        .lock()
        .map_err(|error| error.to_string())?
        .clone();
    if let Some(control) = control {
        control.cancel()?;
    }
    Ok(())
}

#[tauri::command]
fn stop_generation(app: tauri::AppHandle) -> Result<(), String> {
    sdk(&app)?.stop_text_generation()
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_catalog,
            validate_pair,
            check_pair,
            download_engine,
            download_model,
            register_local_engine,
            register_local_model,
            install_ffmpeg,
            download_aligner,
            register_local_aligner,
            allow_preview,
            test_transcribe,
            test_generate,
            test_detect,
            test_tag,
            test_synthesize,
            cancel_task,
            stop_generation,
        ])
        .run(tauri::generate_context!())
        .expect("failed to start desktop application");
}
