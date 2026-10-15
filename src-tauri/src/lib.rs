use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use model_sdk::{
    diagnostics,
    process::hidden_command,
    resources,
    subtitle::{write_srt, Segment},
    task::TaskControl,
    ArtifactFormat, ModelProfile, ModelSdk, TaskKind, TranscriptionOptions,
};
use serde::Serialize;
use tauri::{Emitter, Manager};

mod translate;
const LLAMA_ENGINE_ID: &str = "llama-cpu";

#[derive(Default)]
struct AppState {
    active: Mutex<Option<Arc<TaskControl>>>,
    resources: Mutex<()>,
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
struct Environment {
    platform: String,
    hardware: String,
    recommended_backend: String,
    engine_path: Option<String>,
    installed_backend: Option<String>,
    model_path: Option<String>,
    ffmpeg_available: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EngineOption {
    id: String,
    family: String,
    variant: String,
    task: TaskKind,
    label: String,
    version: String,
    download_mb: u64,
    installed_path: Option<String>,
    suitable: bool,
    reason: Option<String>,
    recommended: bool,
    compatible_model_ids: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelOption {
    id: String,
    label: String,
    profile: ModelProfile,
    artifact_format: ArtifactFormat,
    task: TaskKind,
    size_bytes: Option<u64>,
    downloadable: bool,
    installed_path: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ResourceStatus {
    system: diagnostics::SystemCheck,
    storage_dir: String,
    aligner_path: Option<String>,
    engines: Vec<EngineOption>,
    models: Vec<ModelOption>,
    selected_engine_id: String,
    selected_model_id: String,
    selected_translation_model_id: Option<String>,
    selected_translation_engine_id: String,
    translation_ready: bool,
    translation_reason: Option<String>,
    vad_enabled: bool,
    recommended_engine_id: String,
    ready: bool,
}

fn data_dir(_app: &tauri::AppHandle) -> Result<PathBuf, String> {
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

fn sdk(app: &tauri::AppHandle) -> Result<ModelSdk, String> {
    ModelSdk::new(data_dir(app)?)
}

fn resource_status(app: &tauri::AppHandle) -> Result<ResourceStatus, String> {
    let dir = data_dir(app)?;
    let selection = resources::load_selection(&dir)?;
    let system = diagnostics::system_check(&selection);
    let sdk = ModelSdk::new(&dir)?;
    let mut engines = sdk
        .engines()?
        .into_iter()
        .map(|info| EngineOption {
            id: info.id,
            family: info.family,
            variant: info.variant,
            task: info.task,
            label: info.label,
            version: info.version,
            download_mb: info.download_mb,
            installed_path: info.path.map(|path| path.to_string_lossy().into_owned()),
            suitable: info.suitable,
            reason: info.reason,
            recommended: false,
            compatible_model_ids: Vec::new(),
        })
        .collect::<Vec<_>>();
    let models = sdk
        .models()?
        .into_iter()
        .map(|info| ModelOption {
            id: info.descriptor.id,
            label: info.descriptor.label,
            profile: info.descriptor.profile,
            artifact_format: info.descriptor.artifact_format,
            task: info.descriptor.task,
            size_bytes: info.size_bytes,
            downloadable: info.downloadable,
            installed_path: info.path.map(|path| path.to_string_lossy().into_owned()),
        })
        .collect::<Vec<_>>();
    for engine in &mut engines {
        engine.compatible_model_ids = models
            .iter()
            .filter(|model| {
                sdk.validate_pair_for(&engine.id, &model.id, engine.task)
                    .is_ok()
            })
            .map(|model| model.id.clone())
            .collect();
    }
    let selected_translation_model_id = selection
        .translation_model_id
        .clone()
        .filter(|id| {
            models
                .iter()
                .any(|model| model.id == *id && model.task == TaskKind::TextGeneration)
        })
        .or_else(|| {
            models
                .iter()
                .find(|model| {
                    model.task == TaskKind::TextGeneration && model.installed_path.is_some()
                })
                .map(|model| model.id.clone())
        })
        .or_else(|| Some(resources::HY_MT_MODEL_ID.into()));
    let selected_translation_engine_id = selection
        .translation_engine_id
        .clone()
        .unwrap_or_else(|| LLAMA_ENGINE_ID.into());
    let (translation_ready, translation_reason) = if let Some(id) = &selected_translation_model_id {
        let readiness = sdk.check(
            &selected_translation_engine_id,
            id,
            TaskKind::TextGeneration,
        )?;
        (readiness.ready, readiness.reason)
    } else {
        (false, Some("请下载或登记内置翻译模型。".into()))
    };
    let readiness = sdk.check(
        &selection.engine_id,
        &selection.model_id,
        TaskKind::Transcription,
    )?;
    if !readiness.ready
        && models
            .iter()
            .any(|model| model.id == selection.model_id && model.installed_path.is_some())
    {
        if let Some(engine) = engines
            .iter_mut()
            .find(|engine| engine.id == selection.engine_id && engine.installed_path.is_some())
        {
            engine.suitable = false;
            engine.reason = readiness.reason;
        }
    }
    let recommended_engine_id = diagnostics::recommended_engine_id(|id| {
        engines
            .iter()
            .any(|engine| engine.id == id && engine.suitable)
    })
    .unwrap_or_default()
    .to_string();
    for engine in &mut engines {
        engine.recommended = engine.id == recommended_engine_id;
    }
    if let Some(engine) = engines
        .iter_mut()
        .rev()
        .find(|engine| engine.family == "llama" && engine.suitable)
    {
        engine.recommended = true;
    }
    Ok(ResourceStatus {
        ready: system.ffmpeg_available && readiness.ready,
        system,
        storage_dir: dir.to_string_lossy().into_owned(),
        aligner_path: sdk
            .aligner_path()
            .map(|path| path.to_string_lossy().into_owned()),
        engines,
        models,
        selected_engine_id: selection.engine_id,
        selected_model_id: selection.model_id,
        selected_translation_model_id,
        selected_translation_engine_id,
        translation_ready,
        translation_reason,
        vad_enabled: selection.vad_enabled,
        recommended_engine_id,
    })
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

#[tauri::command]
fn get_environment(app: tauri::AppHandle) -> Result<Environment, String> {
    let status = resource_status(&app)?;
    let engine = status
        .engines
        .iter()
        .find(|item| item.id == status.selected_engine_id);
    let model = status
        .models
        .iter()
        .find(|item| item.id == status.selected_model_id);
    Ok(Environment {
        platform: status.system.platform,
        hardware: status
            .system
            .gpus
            .iter()
            .map(|gpu| gpu.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        recommended_backend: status.recommended_engine_id,
        engine_path: engine.and_then(|item| item.installed_path.clone()),
        installed_backend: Some(status.selected_engine_id),
        model_path: model.and_then(|item| item.installed_path.clone()),
        ffmpeg_available: status.system.ffmpeg_available,
    })
}

#[tauri::command]
fn allow_media_preview(app: tauri::AppHandle, path: String) -> Result<(), String> {
    let media = PathBuf::from(path);
    if !media.is_file() {
        return Err("选中的素材不存在。".into());
    }
    let extension = media
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ![
        "mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "aac", "flac", "ogg",
    ]
    .contains(&extension.as_str())
    {
        return Err("不支持的音视频格式。".into());
    }
    app.asset_protocol_scope()
        .allow_file(&media)
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_resource_status(app: tauri::AppHandle) -> Result<ResourceStatus, String> {
    tauri::async_runtime::spawn_blocking(move || resource_status(&app))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn select_resources(
    app: tauri::AppHandle,
    engine_id: String,
    model_id: String,
) -> Result<ResourceStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dir = data_dir(&app)?;
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        sdk(&app)?.validate_pair_for(&engine_id, &model_id, TaskKind::Transcription)?;
        let mut selection = resources::load_selection(&dir)?;
        selection.engine_id = engine_id;
        selection.model_id = model_id;
        resources::save_selection(&dir, &selection)?;
        resource_status(&app)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn set_vad_enabled(app: tauri::AppHandle, enabled: bool) -> Result<ResourceStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dir = data_dir(&app)?;
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        let mut selection = resources::load_selection(&dir)?;
        selection.vad_enabled = enabled;
        resources::save_selection(&dir, &selection)?;
        resource_status(&app)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn migrate_legacy_resources(_app: tauri::AppHandle) -> Result<ResourceStatus, String> {
    Err("当前 SDK 不自动迁移旧资源 ID 和记录；请按内置 ID 重新登记本地资源。".into())
}

#[tauri::command]
fn register_local_engine(
    app: tauri::AppHandle,
    engine_id: String,
    path: String,
) -> Result<(), String> {
    let state = app.state::<AppState>();
    let _guard = state.resources.lock().map_err(|error| error.to_string())?;
    sdk(&app)?.register_local_engine(&engine_id, Path::new(&path))?;
    if engine_id.starts_with("llama-") {
        let dir = data_dir(&app)?;
        let mut selection = resources::load_selection(&dir)?;
        selection.translation_engine_id = Some(engine_id);
        resources::save_selection(&dir, &selection)?;
    }
    Ok(())
}

#[tauri::command]
async fn register_local_model(
    app: tauri::AppHandle,
    model_id: String,
    path: String,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        let sdk = sdk(&app)?;
        sdk.register_local_model(&model_id, Path::new(&path))
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn register_translation_model(
    _app: tauri::AppHandle,
    _path: String,
) -> Result<ResourceStatus, String> {
    Err("当前 SDK 仅支持内置模型 ID；请登记目录中的翻译模型权重。".into())
}

#[tauri::command]
async fn select_translation_model(
    app: tauri::AppHandle,
    model_id: String,
) -> Result<ResourceStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        let selection = resources::load_selection(&data_dir(&app)?)?;
        let engine_id = selection
            .translation_engine_id
            .as_deref()
            .unwrap_or(LLAMA_ENGINE_ID);
        sdk(&app)?.validate_pair_for(engine_id, &model_id, TaskKind::TextGeneration)?;
        let dir = data_dir(&app)?;
        let mut selection = resources::load_selection(&dir)?;
        selection.translation_model_id = Some(model_id);
        resources::save_selection(&dir, &selection)?;
        resource_status(&app)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn select_translation_engine(
    app: tauri::AppHandle,
    engine_id: String,
) -> Result<ResourceStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        let dir = data_dir(&app)?;
        let mut selection = resources::load_selection(&dir)?;
        let model_id = selection
            .translation_model_id
            .as_deref()
            .unwrap_or(resources::HY_MT_MODEL_ID);
        sdk(&app)?.validate_pair_for(&engine_id, model_id, TaskKind::TextGeneration)?;
        selection.translation_engine_id = Some(engine_id);
        resources::save_selection(&dir, &selection)?;
        resource_status(&app)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn register_local_aligner(app: tauri::AppHandle, path: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        sdk(&app)?
            .register_local_aligner_model(Path::new(&path))
            .map(|path| path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn download_aligner(app: tauri::AppHandle) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        sdk(&app)?
            .ensure_aligner_model(&task(&app))
            .map(|path| path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
fn set_ffmpeg_path(app: tauri::AppHandle, path: Option<String>) -> Result<(), String> {
    if let Some(ref value) = path {
        let executable = PathBuf::from(value);
        if !executable.is_absolute()
            || !executable.is_file()
            || !hidden_command(&executable)
                .arg("-version")
                .output()
                .is_ok_and(|output| output.status.success())
        {
            return Err("FFmpeg 路径无效或无法运行。".into());
        }
    }
    let dir = data_dir(&app)?;
    let state = app.state::<AppState>();
    let _guard = state.resources.lock().map_err(|error| error.to_string())?;
    let mut selection = resources::load_selection(&dir)?;
    selection.ffmpeg_path = path;
    resources::save_selection(&dir, &selection)
}

#[tauri::command]
async fn install_ffmpeg(app: tauri::AppHandle) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        let path = sdk(&app)?.install_ffmpeg(&task(&app))?;
        let dir = data_dir(&app)?;
        let mut selection = resources::load_selection(&dir)?;
        selection.ffmpeg_path = Some(path.to_string_lossy().into_owned());
        resources::save_selection(&dir, &selection)?;
        Ok(path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn download_engine_variant(
    app: tauri::AppHandle,
    engine_id: String,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        sdk(&app)?
            .download_engine(&engine_id, &task(&app))
            .map(|path| path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn download_model_variant(
    app: tauri::AppHandle,
    model_id: String,
    mirror: bool,
) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        sdk(&app)?
            .download_model(&model_id, mirror, &task(&app))
            .map(|path| path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn download_engine(app: tauri::AppHandle, backend: String) -> Result<String, String> {
    let path = download_engine_variant(app.clone(), backend.clone()).await?;
    let dir = data_dir(&app)?;
    let state = app.state::<AppState>();
    let _guard = state.resources.lock().map_err(|error| error.to_string())?;
    let mut selection = resources::load_selection(&dir)?;
    selection.engine_id = backend;
    sdk(&app)?.validate_pair_for(
        &selection.engine_id,
        &selection.model_id,
        TaskKind::Transcription,
    )?;
    resources::save_selection(&dir, &selection)?;
    Ok(path)
}

#[tauri::command]
async fn download_model(app: tauri::AppHandle, mirror: bool) -> Result<String, String> {
    let path = download_model_variant(app.clone(), "base".into(), mirror).await?;
    let dir = data_dir(&app)?;
    let state = app.state::<AppState>();
    let _guard = state.resources.lock().map_err(|error| error.to_string())?;
    let mut selection = resources::load_selection(&dir)?;
    selection.model_id = "base".into();
    sdk(&app)?.validate_pair_for(
        &selection.engine_id,
        &selection.model_id,
        TaskKind::Transcription,
    )?;
    resources::save_selection(&dir, &selection)?;
    Ok(path)
}

async fn run_transcription(
    app: tauri::AppHandle,
    media_path: String,
    selection: resources::Selection,
    options: TranscriptionOptions,
) -> Result<Vec<Segment>, String> {
    let control = task(&app);
    let state = app.state::<AppState>();
    {
        let mut active = state.active.lock().map_err(|error| error.to_string())?;
        if active.is_some() {
            return Err("已有任务正在运行。".into());
        }
        *active = Some(control.clone());
    }
    let worker_app = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let dir = data_dir(&worker_app)?;
        let ffmpeg = selection
            .ffmpeg_path
            .clone()
            .unwrap_or_else(|| "ffmpeg".into());
        let sdk = ModelSdk::new(dir)?;
        sdk.transcribe_with_options(
            &selection.engine_id,
            &selection.model_id,
            Path::new(&media_path),
            Path::new(&ffmpeg),
            selection.vad_enabled,
            &options,
            &control,
        )
    })
    .await
    .map_err(|error| error.to_string());
    *state.active.lock().map_err(|error| error.to_string())? = None;
    result?
}

#[tauri::command]
async fn transcribe(
    _app: tauri::AppHandle,
    _media_path: String,
    _engine_path: String,
    _model_path: String,
    _gpu_backend: String,
    _provider_id: Option<String>,
    _vad_enabled: Option<bool>,
    _align_timestamps: Option<bool>,
    _language: Option<String>,
) -> Result<Vec<Segment>, String> {
    Err("当前 SDK 不支持直接路径转写；请先登记内置模型 ID，再选择引擎和模型。".into())
}

#[tauri::command]
async fn transcribe_selected(
    app: tauri::AppHandle,
    media_path: String,
    align_timestamps: Option<bool>,
    language: Option<String>,
) -> Result<Vec<Segment>, String> {
    let selection = resources::load_selection(&data_dir(&app)?)?;
    let status = get_resource_status(app.clone()).await?;
    if status.selected_engine_id != selection.engine_id
        || status.selected_model_id != selection.model_id
    {
        return Err("资源选择已变化，请重试。".into());
    }
    if !status.ready {
        return Err("所选推理引擎、模型或 FFmpeg 尚未就绪，请先完成资源自检。".into());
    }
    run_transcription(
        app,
        media_path,
        selection,
        TranscriptionOptions {
            align_timestamps: align_timestamps.unwrap_or(true),
            language,
        },
    )
    .await
}

#[tauri::command]
async fn translate_subtitles(
    app: tauri::AppHandle,
    segments: Vec<Segment>,
    model_id: String,
    target_language: String,
) -> Result<Vec<Segment>, String> {
    let control = task(&app);
    let state = app.state::<AppState>();
    {
        let mut active = state.active.lock().map_err(|error| error.to_string())?;
        if active.is_some() {
            return Err("已有任务正在运行。".into());
        }
        *active = Some(control.clone());
    }
    let worker_app = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let sdk = sdk(&worker_app)?;
        let selection = resources::load_selection(&data_dir(&worker_app)?)?;
        let engine_id = selection
            .translation_engine_id
            .as_deref()
            .unwrap_or(LLAMA_ENGINE_ID);
        translate::translate_segments(
            &sdk,
            engine_id,
            &model_id,
            &target_language,
            &segments,
            &control,
        )
    })
    .await
    .map_err(|error| error.to_string());
    *state.active.lock().map_err(|error| error.to_string())? = None;
    result?
}

#[tauri::command]
fn cancel_transcription(app: tauri::AppHandle) -> Result<(), String> {
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
fn export_srt(path: String, segments: Vec<Segment>) -> Result<(), String> {
    let content = write_srt(&segments)?;
    fs::write(path, content).map_err(|error| error.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_environment,
            get_resource_status,
            migrate_legacy_resources,
            select_resources,
            set_vad_enabled,
            register_local_engine,
            register_local_model,
            register_translation_model,
            select_translation_model,
            select_translation_engine,
            register_local_aligner,
            download_aligner,
            set_ffmpeg_path,
            install_ffmpeg,
            allow_media_preview,
            download_model,
            download_model_variant,
            download_engine,
            download_engine_variant,
            transcribe,
            transcribe_selected,
            translate_subtitles,
            cancel_transcription,
            export_srt
        ])
        .run(tauri::generate_context!())
        .expect("failed to start desktop application");
}
