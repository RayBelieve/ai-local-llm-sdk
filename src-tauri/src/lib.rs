use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use model_sdk::{
    process::hidden_command,
    resources,
    subtitle::{write_srt, Segment},
    task::TaskControl,
    ModelSdk, TranscriptionOptions, TRANSCRIPTION,
};
use serde::Serialize;
use tauri::{Emitter, Manager};

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

fn resource_status(app: &tauri::AppHandle) -> Result<resources::ResourceStatus, String> {
    let dir = data_dir(app)?;
    let mut status = resources::status(&dir)?;
    let sdk = ModelSdk::new(&dir)?;
    status.aligner_path = sdk
        .aligner_path()
        .map(|path| path.to_string_lossy().into_owned());
    let sdk_engines = sdk.engines()?;
    for engine in &mut status.engines {
        if let Some(info) = sdk_engines.iter().find(|item| item.id == engine.id) {
            engine.installed_path = info
                .path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned());
            engine.suitable = info.suitable;
            engine.reason = info.reason.clone();
        }
    }
    for model in &mut status.models {
        model.installed_path = sdk
            .model_path(&model.id)?
            .map(|path| path.to_string_lossy().into_owned());
    }
    let readiness = sdk.check(
        &status.selected_engine_id,
        &status.selected_model_id,
        TRANSCRIPTION,
    )?;
    if !readiness.ready
        && status
            .models
            .iter()
            .any(|model| model.id == status.selected_model_id && model.installed_path.is_some())
    {
        if let Some(engine) = status.engines.iter_mut().find(|engine| {
            engine.id == status.selected_engine_id && engine.installed_path.is_some()
        }) {
            engine.suitable = false;
            engine.reason = readiness.reason;
        }
    }
    status.recommended_engine_id = resources::recommended_engine_id(|id| {
        status
            .engines
            .iter()
            .any(|engine| engine.id == id && engine.suitable)
    })
    .unwrap_or_default()
    .to_string();
    for engine in &mut status.engines {
        engine.recommended = engine.id == status.recommended_engine_id;
    }
    status.ready = status.system.ffmpeg_available
        && status
            .engines
            .iter()
            .any(|engine| engine.id == status.selected_engine_id && engine.suitable)
        && readiness.ready;
    let legacy = app
        .path()
        .app_local_data_dir()
        .map_err(|error| error.to_string())?;
    status.legacy_resources_available = resources::legacy_resources_available(&legacy);
    Ok(status)
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
async fn get_resource_status(app: tauri::AppHandle) -> Result<resources::ResourceStatus, String> {
    tauri::async_runtime::spawn_blocking(move || resource_status(&app))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
async fn select_resources(
    app: tauri::AppHandle,
    engine_id: String,
    model_id: String,
) -> Result<resources::ResourceStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        resources::validate_pair(&engine_id, &model_id)?;
        let dir = data_dir(&app)?;
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
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
async fn set_vad_enabled(
    app: tauri::AppHandle,
    enabled: bool,
) -> Result<resources::ResourceStatus, String> {
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
async fn migrate_legacy_resources(
    app: tauri::AppHandle,
) -> Result<resources::ResourceStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let dir = data_dir(&app)?;
        let legacy = app
            .path()
            .app_local_data_dir()
            .map_err(|error| error.to_string())?;
        let state = app.state::<AppState>();
        let _guard = state.resources.lock().map_err(|error| error.to_string())?;
        resources::migrate_legacy_resources(&legacy, &dir)?;
        sdk(&app)?.import_legacy_paths()?;
        resource_status(&app)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
fn register_local_engine(
    app: tauri::AppHandle,
    engine_id: String,
    path: String,
) -> Result<(), String> {
    let state = app.state::<AppState>();
    let _guard = state.resources.lock().map_err(|error| error.to_string())?;
    sdk(&app)?.register_local_engine(&engine_id, Path::new(&path))
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
        let descriptor = sdk
            .models()?
            .into_iter()
            .find(|item| item.descriptor.id == model_id)
            .ok_or_else(|| format!("未知模型: {model_id}"))?
            .descriptor;
        sdk.register_local_model(descriptor, Path::new(&path))
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
    resources::save_selection(&dir, &selection)?;
    Ok(path)
}

async fn run_transcription(
    app: tauri::AppHandle,
    media_path: String,
    explicit: Option<(String, String, String, Option<String>, Option<bool>)>,
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
        if let Some((engine, model, backend, provider, vad)) = explicit {
            sdk.transcribe_paths_with_options(
                Path::new(&media_path),
                Path::new(&engine),
                Path::new(&model),
                Path::new(&ffmpeg),
                &backend,
                provider.as_deref() == Some("openvino"),
                vad.unwrap_or(true),
                &options,
                &control,
            )
        } else {
            sdk.transcribe_with_options(
                &selection.engine_id,
                &selection.model_id,
                Path::new(&media_path),
                Path::new(&ffmpeg),
                selection.vad_enabled,
                &options,
                &control,
            )
        }
    })
    .await
    .map_err(|error| error.to_string());
    *state.active.lock().map_err(|error| error.to_string())? = None;
    result?
}

#[tauri::command]
async fn transcribe(
    app: tauri::AppHandle,
    media_path: String,
    engine_path: String,
    model_path: String,
    gpu_backend: String,
    provider_id: Option<String>,
    vad_enabled: Option<bool>,
    align_timestamps: Option<bool>,
    language: Option<String>,
) -> Result<Vec<Segment>, String> {
    let selection = resources::load_selection(&data_dir(&app)?)?;
    run_transcription(
        app,
        media_path,
        Some((
            engine_path,
            model_path,
            gpu_backend,
            provider_id,
            vad_enabled,
        )),
        selection,
        TranscriptionOptions {
            align_timestamps: align_timestamps.unwrap_or(true),
            language,
        },
    )
    .await
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
        None,
        selection,
        TranscriptionOptions {
            align_timestamps: align_timestamps.unwrap_or(true),
            language,
        },
    )
    .await
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
            cancel_transcription,
            export_srt
        ])
        .run(tauri::generate_context!())
        .expect("failed to start desktop application");
}
