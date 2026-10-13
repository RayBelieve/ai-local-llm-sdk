use std::{
    collections::HashMap,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ENGINE_VERSION: &str = "v0.8.33";
pub const LLAMA_ENGINE_VERSION: &str = "b9500";
pub const HY_MT_MODEL_ID: &str = "hy-mt-1.5-1.8b-q4";
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
pub const LLAMA_ENGINE_BASE: &str = "https://github.com/ggml-org/llama.cpp/releases/download/b9500";
const WHISPER_REPO: &str = "ggerganov/whisper.cpp";

pub struct LlamaAsset {
    pub file: &'static str,
    pub size: u64,
}

pub struct LlamaVariant {
    pub id: &'static str,
    pub label: &'static str,
    pub assets: &'static [LlamaAsset],
}

// shortcut: upstream release ZIPs have pinned sizes but no recorded SHA-256; pin digests before adding new versions.
pub const LLAMA_VARIANTS: &[LlamaVariant] = &[
    LlamaVariant {
        id: "cpu",
        label: "CPU",
        assets: &[LlamaAsset {
            file: "llama-b9500-bin-win-cpu-x64.zip",
            size: 16_906_094,
        }],
    },
    LlamaVariant {
        id: "vulkan",
        label: "Vulkan",
        assets: &[LlamaAsset {
            file: "llama-b9500-bin-win-vulkan-x64.zip",
            size: 33_878_631,
        }],
    },
    LlamaVariant {
        id: "cuda12",
        label: "CUDA 12.4",
        assets: &[
            LlamaAsset {
                file: "llama-b9500-bin-win-cuda-12.4-x64.zip",
                size: 261_143_138,
            },
            LlamaAsset {
                file: "cudart-llama-bin-win-cuda-12.4-x64.zip",
                size: 391_443_627,
            },
        ],
    },
    LlamaVariant {
        id: "cuda13",
        label: "CUDA 13.3",
        assets: &[
            LlamaAsset {
                file: "llama-b9500-bin-win-cuda-13.3-x64.zip",
                size: 159_076_541,
            },
            LlamaAsset {
                file: "cudart-llama-bin-win-cuda-13.3-x64.zip",
                size: 390_970_417,
            },
        ],
    },
];

pub fn llama_variant(id: &str) -> Result<&'static LlamaVariant, String> {
    LLAMA_VARIANTS
        .iter()
        .find(|variant| variant.id == id)
        .ok_or_else(|| format!("未知 llama.cpp 运行时版本: {id}"))
}

pub struct EngineSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub asset: &'static str,
    pub gpu_backend: &'static str,
    pub size_mb: u64,
    pub requires_avx2: bool,
    pub min_nvidia_driver: Option<u32>,
    pub min_compute_capability: Option<u32>,
}

pub const ENGINES: &[EngineSpec] = &[
    EngineSpec {
        id: "vulkan",
        label: "Vulkan",
        asset: "crispasr-windows-x86_64-vulkan.zip",
        gpu_backend: "vulkan",
        size_mb: 35,
        requires_avx2: false,
        min_nvidia_driver: None,
        min_compute_capability: None,
    },
    EngineSpec {
        id: "cuda12",
        label: "CUDA 12",
        asset: "crispasr-windows-x86_64-cuda.zip",
        gpu_backend: "cuda",
        size_mb: 692,
        requires_avx2: false,
        min_nvidia_driver: Some(528),
        min_compute_capability: Some(50),
    },
    EngineSpec {
        id: "cuda13",
        label: "CUDA 13",
        asset: "crispasr-windows-x86_64-cuda13.zip",
        gpu_backend: "cuda",
        size_mb: 486,
        requires_avx2: false,
        min_nvidia_driver: Some(580),
        min_compute_capability: Some(75),
    },
    EngineSpec {
        id: "cpu",
        label: "CPU",
        asset: "crispasr-windows-x86_64-cpu.zip",
        gpu_backend: "cpu",
        size_mb: 8,
        requires_avx2: true,
        min_nvidia_driver: None,
        min_compute_capability: None,
    },
    EngineSpec {
        id: "cpu-legacy",
        label: "CPU 兼容版",
        asset: "crispasr-windows-x86_64-cpu-legacy.zip",
        gpu_backend: "cpu",
        size_mb: 8,
        requires_avx2: false,
        min_nvidia_driver: None,
        min_compute_capability: None,
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
    ModelSpec {
        id: HY_MT_MODEL_ID,
        label: "HY-MT1.5 1.8B Q4_K_M",
        file: "HY-MT1.5-1.8B-Q4_K_M.gguf",
        repo: "tencent/HY-MT1.5-1.8B-GGUF",
        format: "llama-gguf",
        size: 1_133_080_512,
        sha256: "4383ac0c3c8e476de98ff979c2a3f069f8c4fb385e7860cf2d28da896cc477c7",
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
        .ok_or_else(|| format!("未知模型: {id}"))
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
    #[serde(default)]
    pub translation_model_id: Option<String>,
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

impl Default for Selection {
    fn default() -> Self {
        Self {
            engine_id: "vulkan".into(),
            model_id: "base".into(),
            translation_model_id: None,
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
    serde_json::from_slice(&fs::read(source).map_err(|error| error.to_string())?)
        .map_err(|error| format!("资源选择配置无效: {error}"))
}

pub fn save_selection(dir: &Path, selection: &Selection) -> Result<(), String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_resource_ids_are_rejected() {
        assert!(engine_spec("openvino").is_err());
        assert!(model_spec("qwen3").is_err());
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
        }
    }

    #[test]
    fn old_selection_defaults_to_vad_enabled() {
        let selection: Selection =
            serde_json::from_str(r#"{"engineId":"vulkan","modelId":"base"}"#).unwrap();
        assert!(selection.vad_enabled);
        assert_eq!(selection.translation_model_id, None);
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
}
