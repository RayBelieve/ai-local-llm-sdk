use std::{
    collections::HashMap,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::process::hidden_command;

pub const ENGINE_VERSION: &str = "v0.8.33";
pub const OPENVINO_ENGINE_ID: &str = "openvino-cpu";
pub const OPENVINO_MODEL_ID: &str = "qwen3-asr-0.6b-int8";
pub const OPENVINO_MODEL_FILES: &[&str] = &[
    "audio_encoder_model.bin",
    "audio_encoder_model.xml",
    "decoder_model.bin",
    "decoder_model.xml",
    "thinker_embeddings_model.bin",
    "thinker_embeddings_model.xml",
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
];
pub const OPENVINO_MODEL_HASHES: &[(&str, &str)] = &[
    (
        "audio_encoder_model.bin",
        "d892464d9b6986719dd6e5c3962b880a2708d874c2c9bdead8958581be2dacb9",
    ),
    (
        "decoder_model.bin",
        "cc4363c401f5faf41e2bfcb4aea80c72144b8ea66d13ca5ca62cf49421a25778",
    ),
    (
        "thinker_embeddings_model.bin",
        "a7818fcbd77240fb8705bc47c2a15da98498056cdd419742b7685719b5dc2a44",
    ),
];
pub const ENGINE_BASE: &str = "https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.33";
const WHISPER_REPO: &str = "ggerganov/whisper.cpp";

pub struct EngineSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub asset: &'static str,
    pub gpu_backend: &'static str,
    pub size_mb: u64,
    pub requires_avx2: bool,
}

pub const ENGINES: &[EngineSpec] = &[
    EngineSpec {
        id: "vulkan",
        label: "Vulkan",
        asset: "crispasr-windows-x86_64-vulkan.zip",
        gpu_backend: "vulkan",
        size_mb: 35,
        requires_avx2: false,
    },
    EngineSpec {
        id: "cuda12",
        label: "CUDA 12",
        asset: "crispasr-windows-x86_64-cuda.zip",
        gpu_backend: "cuda",
        size_mb: 692,
        requires_avx2: false,
    },
    EngineSpec {
        id: "cuda13",
        label: "CUDA 13",
        asset: "crispasr-windows-x86_64-cuda13.zip",
        gpu_backend: "cuda",
        size_mb: 486,
        requires_avx2: false,
    },
    EngineSpec {
        id: "cpu",
        label: "CPU",
        asset: "crispasr-windows-x86_64-cpu.zip",
        gpu_backend: "cpu",
        size_mb: 8,
        requires_avx2: true,
    },
    EngineSpec {
        id: "cpu-legacy",
        label: "CPU 兼容版",
        asset: "crispasr-windows-x86_64-cpu-legacy.zip",
        gpu_backend: "cpu",
        size_mb: 8,
        requires_avx2: false,
    },
];

pub struct ModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub file: &'static str,
    pub repo: &'static str,
    pub format: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

// Sizes and LFS SHA-256 values come from each Hugging Face model repository.
pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "tiny",
        label: "Whisper Tiny",
        file: "ggml-tiny.bin",
        repo: WHISPER_REPO,
        format: "whisper-ggml",
        size: 77_691_713,
        sha256: "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
    },
    ModelSpec {
        id: "base",
        label: "Whisper Base",
        file: "ggml-base.bin",
        repo: WHISPER_REPO,
        format: "whisper-ggml",
        size: 147_951_465,
        sha256: "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
    },
    ModelSpec {
        id: "small",
        label: "Whisper Small",
        file: "ggml-small.bin",
        repo: WHISPER_REPO,
        format: "whisper-ggml",
        size: 487_601_967,
        sha256: "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
    },
    ModelSpec {
        id: "medium",
        label: "Whisper Medium",
        file: "ggml-medium.bin",
        repo: WHISPER_REPO,
        format: "whisper-ggml",
        size: 1_533_763_059,
        sha256: "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
    },
    ModelSpec {
        id: "turbo",
        label: "Whisper Large V3 Turbo",
        file: "ggml-large-v3-turbo.bin",
        repo: WHISPER_REPO,
        format: "whisper-ggml",
        size: 1_624_555_275,
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    },
    ModelSpec {
        id: "large-v2",
        label: "Whisper Large V2",
        file: "ggml-large-v2.bin",
        repo: WHISPER_REPO,
        format: "whisper-ggml",
        size: 3_094_623_691,
        sha256: "9a423fe4d40c82774b6af34115b8b935f34152246eb19e80e376071d3f999487",
    },
    ModelSpec {
        id: "qwen3-1.7b-q4",
        label: "Qwen3-ASR 1.7B Q4",
        file: "qwen3-asr-1.7b-q4_k.gguf",
        repo: "cstr/qwen3-asr-1.7b-GGUF",
        format: "qwen3-gguf",
        size: 1_490_915_200,
        sha256: "ec197cef7ccc589fdcae1becc3f4a3de119d0a41e790b898b519b1a048dad8d4",
    },
    ModelSpec {
        id: "qwen3-1.7b-q8",
        label: "Qwen3-ASR 1.7B Q8",
        file: "qwen3-asr-1.7b-q8_0.gguf",
        repo: "cstr/qwen3-asr-1.7b-GGUF",
        format: "qwen3-gguf",
        size: 2_506_723_200,
        sha256: "9851ab996591a2d0cb0efb216002764b509c86bd40c95e613d7b65b8e69c8a6e",
    },
    ModelSpec {
        id: "qwen3-1.7b-ja-q4",
        label: "Qwen3-ASR 1.7B 日语动漫 Q4",
        file: "qwen3-asr-1.7b-ja-anime-q4_k.gguf",
        repo: "cstr/qwen3-asr-1.7b-ja-anime-GGUF",
        format: "qwen3-gguf",
        size: 1_490_917_376,
        sha256: "da00cec556885a60d47234b4def0dbdd08394df4d4df13866d7cd9726c23ccd9",
    },
    ModelSpec {
        id: "qwen3-1.7b-ja-q8",
        label: "Qwen3-ASR 1.7B 日语动漫 Q8",
        file: "qwen3-asr-1.7b-ja-anime-q8_0.gguf",
        repo: "cstr/qwen3-asr-1.7b-ja-anime-GGUF",
        format: "qwen3-gguf",
        size: 2_506_725_376,
        sha256: "f173ceddb15b59177575a101491862bd362cd85f2e7a6bac811fbc53411596a1",
    },
];

pub fn engine_spec(id: &str) -> Result<&'static EngineSpec, String> {
    ENGINES
        .iter()
        .find(|spec| spec.id == id)
        .ok_or_else(|| format!("未知推理引擎版本: {id}"))
}

pub fn model_spec(id: &str) -> Result<&'static ModelSpec, String> {
    MODELS
        .iter()
        .find(|spec| spec.id == id)
        .ok_or_else(|| format!("未知转写模型: {id}"))
}

pub fn model_url(spec: &ModelSpec, mirror: bool) -> String {
    let host = if mirror {
        "https://hf-mirror.com"
    } else {
        "https://huggingface.co"
    };
    format!("{host}/{}/resolve/main/{}", spec.repo, spec.file)
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Selection {
    pub engine_id: String,
    pub model_id: String,
    #[serde(default = "default_vad_enabled")]
    pub vad_enabled: bool,
    #[serde(default)]
    pub engine_paths: HashMap<String, String>,
    #[serde(default)]
    pub model_paths: HashMap<String, String>,
    #[serde(default)]
    pub ffmpeg_path: Option<String>,
}

fn default_vad_enabled() -> bool {
    true
}

pub fn validate_pair(engine_id: &str, model_id: &str) -> Result<(), String> {
    if engine_id == OPENVINO_ENGINE_ID && model_id == OPENVINO_MODEL_ID {
        return Ok(());
    }
    engine_spec(engine_id)?;
    model_spec(model_id)?;
    Ok(())
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            engine_id: "vulkan".into(),
            model_id: "base".into(),
            vad_enabled: true,
            engine_paths: HashMap::new(),
            model_paths: HashMap::new(),
            ffmpeg_path: None,
        }
    }
}

pub fn load_selection(dir: &Path) -> Result<Selection, String> {
    let file = dir.join("resources.json");
    let source = if file.exists() {
        file
    } else {
        dir.join("resources.json.bak")
    };
    if !source.exists() {
        let mut selection = Selection::default();
        if let Ok(backend) = fs::read_to_string(dir.join("engines/crispasr/backend.txt")) {
            if engine_spec(backend.trim()).is_ok() {
                selection.engine_id = backend.trim().into();
            }
        }
        return Ok(selection);
    }
    let selection: Selection =
        serde_json::from_slice(&fs::read(source).map_err(|error| error.to_string())?)
            .map_err(|error| format!("资源选择配置无效: {error}"))?;
    validate_pair(&selection.engine_id, &selection.model_id)?;
    Ok(selection)
}

pub fn save_selection(dir: &Path, selection: &Selection) -> Result<(), String> {
    validate_pair(&selection.engine_id, &selection.model_id)?;
    fs::create_dir_all(dir).map_err(|error| error.to_string())?;
    let path = dir.join("resources.json");
    let temporary = dir.join("resources.json.tmp");
    let backup = dir.join("resources.json.bak");
    let content = serde_json::to_vec_pretty(selection).map_err(|error| error.to_string())?;
    fs::write(&temporary, content).map_err(|error| error.to_string())?;
    if path.exists() {
        if backup.exists() {
            fs::remove_file(&backup).map_err(|error| error.to_string())?;
        }
        fs::rename(&path, &backup).map_err(|error| error.to_string())?;
    }
    if let Err(error) = fs::rename(&temporary, &path) {
        if backup.exists() {
            let _ = fs::rename(&backup, &path);
        }
        return Err(error.to_string());
    }
    if backup.exists() {
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

pub fn legacy_resources_available(dir: &Path) -> bool {
    [dir.join("models"), dir.join("engines/crispasr")]
        .iter()
        .any(|path| fs::read_dir(path).is_ok_and(|mut entries| entries.next().is_some()))
}

fn file_hash(path: &Path) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hash.finalize().to_vec())
}

fn same_file(source: &Path, target: &Path) -> Result<bool, String> {
    let source_size = fs::metadata(source)
        .map_err(|error| error.to_string())?
        .len();
    let target_size = fs::metadata(target)
        .map_err(|error| error.to_string())?
        .len();
    Ok(source_size == target_size && file_hash(source)? == file_hash(target)?)
}

fn move_managed_file(source: &Path, target: &Path) -> Result<(), String> {
    if target.exists() {
        if !same_file(source, target)? {
            return Err(format!(
                "资源迁移遇到同名但内容不同的文件: {}",
                target.display()
            ));
        }
    } else {
        fs::create_dir_all(target.parent().ok_or("资源目标路径无效。")?)
            .map_err(|error| error.to_string())?;
        if fs::rename(source, target).is_ok() {
            return Ok(());
        }
        let mut temporary_name = target.as_os_str().to_os_string();
        temporary_name.push(".migrating");
        let temporary = PathBuf::from(temporary_name);
        fs::copy(source, &temporary).map_err(|error| error.to_string())?;
        if !same_file(source, &temporary)? {
            let _ = fs::remove_file(&temporary);
            return Err(format!("资源复制校验失败: {}", source.display()));
        }
        fs::rename(&temporary, target).map_err(|error| error.to_string())?;
    }
    fs::remove_file(source).map_err(|error| error.to_string())
}

fn move_managed_tree(source: &Path, target: &Path) -> Result<(), String> {
    if !source.exists() {
        return Ok(());
    }
    fs::create_dir_all(target).map_err(|error| error.to_string())?;
    for entry in fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        let destination = target.join(entry.file_name());
        if kind.is_dir() {
            move_managed_tree(&entry.path(), &destination)?;
        } else if kind.is_file() {
            move_managed_file(&entry.path(), &destination)?;
        } else {
            return Err(format!("无法迁移特殊文件: {}", entry.path().display()));
        }
    }
    fs::remove_dir(source).map_err(|error| error.to_string())
}

pub fn migrate_legacy_resources(legacy: &Path, portable: &Path) -> Result<(), String> {
    if legacy == portable || !legacy_resources_available(legacy) {
        return Ok(());
    }
    if !portable.join("resources.json").exists() {
        let mut selection = load_selection(legacy)?;
        for path in selection
            .engine_paths
            .values_mut()
            .chain(selection.model_paths.values_mut())
        {
            if let Ok(relative) = Path::new(path).strip_prefix(legacy) {
                *path = portable.join(relative).to_string_lossy().into_owned();
            }
        }
        save_selection(portable, &selection)?;
    }
    move_managed_tree(&legacy.join("models"), &portable.join("models"))?;
    move_managed_tree(
        &legacy.join("engines/crispasr"),
        &portable.join("engines/crispasr"),
    )?;
    let legacy_engine_dir = legacy.join("engines");
    if legacy_engine_dir.is_dir()
        && fs::read_dir(&legacy_engine_dir).is_ok_and(|mut entries| entries.next().is_none())
    {
        let _ = fs::remove_dir(legacy_engine_dir);
    }
    Ok(())
}

pub fn engine_path(dir: &Path, selection: &Selection, id: &str) -> Option<PathBuf> {
    if let Some(path) = selection.engine_paths.get(id) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let installed = dir.join("engines/crispasr").join(id).join("crispasr.exe");
    if installed.is_file() {
        return Some(installed);
    }
    let legacy = dir.join("engines/crispasr/crispasr.exe");
    let legacy_id = fs::read_to_string(dir.join("engines/crispasr/backend.txt")).ok()?;
    (legacy_id.trim() == id && legacy.is_file()).then_some(legacy)
}

pub fn model_path(dir: &Path, selection: &Selection, spec: &ModelSpec) -> Option<PathBuf> {
    if let Some(path) = selection.model_paths.get(spec.id) {
        let path = PathBuf::from(path);
        if path
            .metadata()
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == spec.size)
        {
            return Some(path);
        }
    }
    let path = dir.join("models").join(spec.file);
    path.metadata()
        .ok()
        .filter(|metadata| metadata.is_file() && metadata.len() == spec.size)
        .map(|_| path)
}

pub fn verify_model(path: &Path, spec: &ModelSpec) -> Result<(), String> {
    let size = path.metadata().map_err(|error| error.to_string())?.len();
    if size != spec.size {
        return Err(format!(
            "模型大小不符: {size} 字节，应为 {} 字节",
            spec.size
        ));
    }
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    if format!("{:x}", hash.finalize()) != spec.sha256 {
        return Err("模型 SHA-256 校验失败。".into());
    }
    Ok(())
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    pub driver: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemCheck {
    pub platform: String,
    pub architecture: String,
    pub cpu: String,
    pub avx2: bool,
    pub gpus: Vec<GpuInfo>,
    pub nvidia_driver: Option<String>,
    pub ffmpeg_path: Option<String>,
    pub ffmpeg_available: bool,
    pub diagnostics: Vec<String>,
}

fn gpu_vendor(name: &str) -> &'static str {
    let name = name.to_ascii_lowercase();
    if name.contains("nvidia") {
        "nvidia"
    } else if name.contains("amd") || name.contains("radeon") {
        "amd"
    } else if name.contains("intel") {
        "intel"
    } else {
        "other"
    }
}

fn driver_major(version: &str) -> Option<u32> {
    version.trim().split('.').next()?.parse().ok()
}

fn recommended_engine(check: &SystemCheck) -> &'static str {
    if check.platform != "windows" || check.architecture != "x86_64" {
        return "cpu";
    }
    if check.gpus.iter().any(|gpu| gpu.vendor == "nvidia") {
        return match check.nvidia_driver.as_deref().and_then(driver_major) {
            Some(version) if version >= 580 => "cuda13",
            Some(_) => "cuda12",
            None => "vulkan",
        };
    }
    if check.gpus.is_empty() {
        if check.avx2 {
            "cpu"
        } else {
            "cpu-legacy"
        }
    } else {
        "vulkan"
    }
}

fn ffmpeg_works(path: &str) -> bool {
    hidden_command(path)
        .arg("-version")
        .output()
        .map(|result| result.status.success())
        .unwrap_or(false)
}

pub fn system_check(selection: &Selection) -> SystemCheck {
    let platform = std::env::consts::OS.to_string();
    let architecture = std::env::consts::ARCH.to_string();
    let avx2 = if cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86") {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        {
            false
        }
    } else {
        false
    };
    let mut diagnostics = Vec::new();
    let mut gpus = Vec::new();
    if platform == "windows" {
        let result = hidden_command("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"])
            .output();
        match result {
            Ok(output) if output.status.success() => {
                let value = serde_json::from_slice::<serde_json::Value>(&output.stdout);
                match value {
                    Ok(value) => {
                        let entries: Vec<&serde_json::Value> = match &value {
                            serde_json::Value::Array(items) => items.iter().collect(),
                            serde_json::Value::Object(_) => vec![&value],
                            _ => Vec::new(),
                        };
                        for entry in entries {
                            let name = entry["Name"].as_str().unwrap_or("").trim();
                            let lower = name.to_ascii_lowercase();
                            if name.is_empty()
                                || ["microsoft basic", "virtual", "remote", "todesk"]
                                    .iter()
                                    .any(|part| lower.contains(part))
                            {
                                continue;
                            }
                            gpus.push(GpuInfo {
                                name: name.into(),
                                vendor: gpu_vendor(name).into(),
                                driver: entry["DriverVersion"].as_str().unwrap_or("").into(),
                            });
                        }
                    }
                    Err(_) => diagnostics.push("显卡自检结果无法解析。".into()),
                }
            }
            _ => diagnostics.push("无法读取显卡信息。".into()),
        }
    } else {
        diagnostics.push("当前版本尚未提供此平台的自动引擎安装。".into());
    }
    let nvidia_driver = if gpus.iter().any(|gpu| gpu.vendor == "nvidia") {
        hidden_command("nvidia-smi")
            .args(["--query-gpu=driver_version", "--format=csv,noheader"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|output| output.lines().next().map(str::trim).map(str::to_owned))
            .filter(|value| !value.is_empty())
    } else {
        None
    };
    let ffmpeg_path = selection.ffmpeg_path.clone().or_else(|| {
        let locator = if platform == "windows" {
            "where.exe"
        } else {
            "which"
        };
        hidden_command(locator)
            .arg("ffmpeg")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|output| output.lines().next().map(str::trim).map(str::to_owned))
    });
    let ffmpeg_available = ffmpeg_path.as_deref().is_some_and(ffmpeg_works);
    if !ffmpeg_available {
        diagnostics.push("未找到可运行的 FFmpeg，请安装或登记本机路径。".into());
    }
    SystemCheck {
        platform,
        architecture,
        cpu: std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "未知处理器".into()),
        avx2,
        gpus,
        nvidia_driver,
        ffmpeg_path,
        ffmpeg_available,
        diagnostics,
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineOption {
    pub id: String,
    pub provider: String,
    pub label: String,
    pub version: String,
    pub download_mb: u64,
    pub installed_path: Option<String>,
    pub suitable: bool,
    pub reason: Option<String>,
    pub recommended: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelOption {
    pub id: String,
    pub provider: String,
    pub label: String,
    pub format: String,
    pub size_bytes: u64,
    pub installed_path: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceStatus {
    pub system: SystemCheck,
    pub storage_dir: String,
    pub aligner_path: Option<String>,
    pub legacy_resources_available: bool,
    pub engines: Vec<EngineOption>,
    pub models: Vec<ModelOption>,
    pub selected_engine_id: String,
    pub selected_model_id: String,
    pub vad_enabled: bool,
    pub recommended_engine_id: String,
    pub ready: bool,
}

pub fn openvino_python(dir: &Path, selection: &Selection) -> Option<PathBuf> {
    let registered = selection
        .engine_paths
        .get(OPENVINO_ENGINE_ID)
        .map(PathBuf::from);
    registered.filter(|path| path.is_file()).or_else(|| {
        let path = if cfg!(windows) {
            dir.join("engines/openvino/venv/Scripts/python.exe")
        } else {
            dir.join("engines/openvino/venv/bin/python3")
        };
        (path.is_file() && dir.join("engines/openvino/ready.txt").is_file()).then_some(path)
    })
}

pub fn openvino_model(dir: &Path, selection: &Selection) -> Option<PathBuf> {
    let registered = selection
        .model_paths
        .get(OPENVINO_MODEL_ID)
        .map(PathBuf::from);
    registered
        .into_iter()
        .chain(std::iter::once(dir.join("models/openvino/qwen3_asr_int8")))
        .find(|path| verify_openvino_model(path).is_ok())
}

pub fn verify_openvino_model(path: &Path) -> Result<(), String> {
    if !path.is_dir() {
        return Err("请选择 OpenVINO 模型目录。".into());
    }
    for name in OPENVINO_MODEL_FILES {
        let file = path.join(name);
        let metadata = file.metadata().map_err(|_| format!("模型缺少 {name}"))?;
        if !metadata.is_file() || metadata.len() < 100 {
            return Err(format!("模型文件无效: {name}"));
        }
    }
    for (name, _) in OPENVINO_MODEL_HASHES {
        if path
            .join(name)
            .metadata()
            .is_ok_and(|value| value.len() < 10_000_000)
        {
            return Err(format!("模型权重不完整: {name}"));
        }
    }
    Ok(())
}

pub fn suitability(spec: &EngineSpec, check: &SystemCheck) -> (bool, Option<String>) {
    if check.platform != "windows" || check.architecture != "x86_64" {
        return (false, Some("自动安装目前仅支持 Windows x86-64。".into()));
    }
    if spec.requires_avx2 && !check.avx2 {
        return (false, Some("处理器缺少 AVX2，请选 CPU 兼容版。".into()));
    }
    if spec.id.starts_with("cuda") {
        if !check.gpus.iter().any(|gpu| gpu.vendor == "nvidia") {
            return (false, Some("需要 NVIDIA 显卡。".into()));
        }
        let driver = check.nvidia_driver.as_deref().and_then(driver_major);
        if spec.id == "cuda13" && driver.is_some_and(|version| version < 580) {
            return (
                false,
                Some("CUDA 13 需要 580 或更高版本的 NVIDIA 驱动。".into()),
            );
        }
        if driver.is_none() {
            return (
                true,
                Some("无法读取驱动版本，请先确认 CUDA 兼容性。".into()),
            );
        }
    }
    (true, None)
}

pub fn status(dir: &Path) -> Result<ResourceStatus, String> {
    let selection = load_selection(dir)?;
    let system = system_check(&selection);
    let recommended_engine_id = recommended_engine(&system).to_string();
    let mut engines = ENGINES
        .iter()
        .map(|spec| {
            let (suitable, reason) = suitability(spec, &system);
            EngineOption {
                id: spec.id.into(),
                provider: "crispasr".into(),
                label: spec.label.into(),
                version: ENGINE_VERSION.into(),
                download_mb: spec.size_mb,
                installed_path: engine_path(dir, &selection, spec.id)
                    .map(|path| path.to_string_lossy().into_owned()),
                suitable,
                reason,
                recommended: spec.id == recommended_engine_id,
            }
        })
        .collect::<Vec<_>>();
    let openvino_installed = openvino_python(dir, &selection);
    let openvino_auto_install = system.platform == "windows" && system.architecture == "x86_64";
    engines.push(EngineOption {
        id: OPENVINO_ENGINE_ID.into(),
        provider: "openvino".into(),
        label: "OpenVINO CPU".into(),
        version: "2026.4.1".into(),
        download_mb: 0,
        installed_path: openvino_installed
            .clone()
            .map(|path| path.to_string_lossy().into_owned()),
        suitable: openvino_auto_install || openvino_installed.is_some(),
        reason: (!openvino_auto_install).then_some("当前自动安装仅支持 Windows x86-64。".into()),
        recommended: false,
    });
    let mut models = MODELS
        .iter()
        .map(|spec| ModelOption {
            id: spec.id.into(),
            provider: "crispasr".into(),
            label: spec.label.into(),
            format: spec.format.into(),
            size_bytes: spec.size,
            installed_path: model_path(dir, &selection, spec)
                .map(|path| path.to_string_lossy().into_owned()),
        })
        .collect::<Vec<_>>();
    models.push(ModelOption {
        id: OPENVINO_MODEL_ID.into(),
        provider: "openvino".into(),
        label: "Qwen3-ASR 0.6B INT8".into(),
        format: "qwen3-openvino".into(),
        size_bytes: 1_281_675_864,
        installed_path: openvino_model(dir, &selection)
            .map(|path| path.to_string_lossy().into_owned()),
    });
    let ready = system.ffmpeg_available
        && engines.iter().any(|engine| {
            engine.id == selection.engine_id && engine.installed_path.is_some() && engine.suitable
        })
        && models.iter().any(|model| {
            model.id == selection.model_id
                && model.installed_path.is_some()
                && model.provider
                    == engines
                        .iter()
                        .find(|engine| engine.id == selection.engine_id)
                        .map(|engine| engine.provider.as_str())
                        .unwrap_or("")
        });
    Ok(ResourceStatus {
        system,
        storage_dir: dir.to_string_lossy().into_owned(),
        aligner_path: None,
        legacy_resources_available: false,
        engines,
        models,
        selected_engine_id: selection.engine_id,
        selected_model_id: selection.model_id,
        vad_enabled: selection.vad_enabled,
        recommended_engine_id,
        ready,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(gpus: Vec<GpuInfo>, driver: Option<&str>, avx2: bool) -> SystemCheck {
        SystemCheck {
            platform: "windows".into(),
            architecture: "x86_64".into(),
            cpu: "CPU".into(),
            avx2,
            gpus,
            nvidia_driver: driver.map(str::to_owned),
            ffmpeg_path: None,
            ffmpeg_available: false,
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn recommends_cuda_only_with_known_compatible_driver() {
        let nvidia = vec![GpuInfo {
            name: "NVIDIA GPU".into(),
            vendor: "nvidia".into(),
            driver: "".into(),
        }];
        assert_eq!(
            recommended_engine(&check(nvidia.clone(), Some("581.15"), true)),
            "cuda13"
        );
        assert_eq!(
            recommended_engine(&check(nvidia.clone(), Some("576.88"), true)),
            "cuda12"
        );
        assert_eq!(recommended_engine(&check(nvidia, None, true)), "vulkan");
    }

    #[test]
    fn old_cpu_uses_legacy_variant() {
        assert_eq!(
            recommended_engine(&check(Vec::new(), None, false)),
            "cpu-legacy"
        );
        assert!(!suitability(engine_spec("cpu").unwrap(), &check(Vec::new(), None, false)).0);
    }

    #[test]
    fn unknown_resource_ids_are_rejected() {
        assert!(engine_spec("openvino").is_err());
        assert!(model_spec("qwen3").is_err());
        assert!(validate_pair(OPENVINO_ENGINE_ID, OPENVINO_MODEL_ID).is_ok());
        assert!(validate_pair(OPENVINO_ENGINE_ID, "base").is_err());
        assert!(validate_pair("vulkan", OPENVINO_MODEL_ID).is_err());
    }

    #[test]
    fn qwen3_gguf_catalog_uses_matching_repositories() {
        for id in [
            "qwen3-1.7b-q4",
            "qwen3-1.7b-q8",
            "qwen3-1.7b-ja-q4",
            "qwen3-1.7b-ja-q8",
        ] {
            let spec = model_spec(id).unwrap();
            assert_eq!(spec.format, "qwen3-gguf");
            assert!(spec.file.ends_with(".gguf"));
            assert_eq!(spec.sha256.len(), 64);
            assert!(model_url(spec, false).contains(spec.repo));
            assert!(model_url(spec, true).starts_with("https://hf-mirror.com/"));
            assert!(validate_pair("vulkan", id).is_ok());
            assert!(validate_pair(OPENVINO_ENGINE_ID, id).is_err());
        }
    }

    #[test]
    fn old_selection_defaults_to_vad_enabled() {
        let selection: Selection =
            serde_json::from_str(r#"{"engineId":"vulkan","modelId":"base"}"#).unwrap();
        assert!(selection.vad_enabled);
    }

    #[test]
    fn legacy_vulkan_install_remains_selectable() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("subtitle-resource-test-{nonce}"));
        let engine_dir = dir.join("engines/crispasr");
        fs::create_dir_all(&engine_dir).unwrap();
        fs::write(engine_dir.join("backend.txt"), "vulkan").unwrap();
        fs::write(engine_dir.join("crispasr.exe"), b"test").unwrap();
        let selection = load_selection(&dir).unwrap();
        assert_eq!(selection.engine_id, "vulkan");
        assert_eq!(
            engine_path(&dir, &selection, "vulkan"),
            Some(engine_dir.join("crispasr.exe"))
        );
        assert_eq!(engine_path(&dir, &selection, "cpu"), None);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn migrates_legacy_resources_and_selection() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subtitle-migration-test-{nonce}"));
        let legacy = root.join("legacy");
        let portable = root.join("portable");
        fs::create_dir_all(legacy.join("engines/crispasr")).unwrap();
        fs::create_dir_all(legacy.join("models")).unwrap();
        fs::write(legacy.join("engines/crispasr/backend.txt"), "vulkan").unwrap();
        fs::write(legacy.join("engines/crispasr/crispasr.exe"), b"engine").unwrap();
        fs::write(legacy.join("models/ggml-base.bin"), b"model").unwrap();
        save_selection(&legacy, &Selection::default()).unwrap();

        migrate_legacy_resources(&legacy, &portable).unwrap();

        assert!(!legacy_resources_available(&legacy));
        assert_eq!(
            fs::read(portable.join("models/ggml-base.bin")).unwrap(),
            b"model"
        );
        let selection = load_selection(&portable).unwrap();
        assert_eq!(selection.engine_id, "vulkan");
        assert_eq!(selection.model_id, "base");
        assert!(engine_path(&portable, &selection, "vulkan").is_some());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn migration_keeps_source_on_conflict() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("subtitle-migration-conflict-{nonce}"));
        let legacy = root.join("legacy");
        let portable = root.join("portable");
        fs::create_dir_all(legacy.join("models")).unwrap();
        fs::create_dir_all(portable.join("models")).unwrap();
        fs::write(legacy.join("models/ggml-base.bin"), b"old").unwrap();
        fs::write(portable.join("models/ggml-base.bin"), b"new").unwrap();

        assert!(migrate_legacy_resources(&legacy, &portable).is_err());
        assert_eq!(
            fs::read(legacy.join("models/ggml-base.bin")).unwrap(),
            b"old"
        );
        assert_eq!(
            fs::read(portable.join("models/ggml-base.bin")).unwrap(),
            b"new"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
