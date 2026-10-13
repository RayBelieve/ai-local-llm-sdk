use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use serde::{Deserialize, Serialize};

use crate::{
    diagnostics,
    process::hidden_command,
    resources::{self, Selection},
};

pub const TRANSCRIPTION: &str = "speech-to-text";
pub const TEXT_GENERATION: &str = "text-generation";
pub const LLAMA_ENGINE_ID: &str = "llama-server";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDescriptor {
    pub id: String,
    pub label: String,
    pub format: String,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    pub descriptor: ModelDescriptor,
    pub path: Option<PathBuf>,
    pub provider: String,
    pub size_bytes: Option<u64>,
    pub downloadable: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineInfo {
    pub id: String,
    pub provider: String,
    pub label: String,
    pub version: String,
    pub download_mb: u64,
    pub path: Option<PathBuf>,
    pub suitable: bool,
    pub reason: Option<String>,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LlamaDownloadOption {
    pub id: String,
    pub label: String,
    pub download_mb: u64,
    pub suitable: bool,
    pub recommended: bool,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Readiness {
    pub ready: bool,
    pub reason: Option<String>,
}

#[derive(Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct Registry {
    #[serde(default)]
    engine_paths: HashMap<String, String>,
    #[serde(default)]
    model_paths: HashMap<String, String>,
    #[serde(default)]
    custom_models: HashMap<String, ModelDescriptor>,
}

pub struct ModelSdk {
    root: PathBuf,
    pub(crate) llama_server: Mutex<Option<crate::generate::LlamaServer>>,
}

impl ModelSdk {
    pub fn llama_download_options(&self) -> Vec<LlamaDownloadOption> {
        Self::llama_download_options_for(&diagnostics::system_check(&Selection::default()))
    }

    pub fn llama_download_options_for(
        system: &diagnostics::SystemCheck,
    ) -> Vec<LlamaDownloadOption> {
        let windows = system.platform == "windows" && system.architecture == "x86_64";
        let gpu = system
            .gpus
            .iter()
            .any(|item| ["nvidia", "amd", "intel"].contains(&item.vendor.as_str()));
        let nvidia = system.gpus.iter().any(|item| item.vendor == "nvidia");
        let driver = system
            .nvidia_driver
            .as_deref()
            .and_then(diagnostics::driver_major);
        let compute = system
            .nvidia_compute_capability
            .as_deref()
            .and_then(diagnostics::compute_capability);
        let mut options = resources::LLAMA_VARIANTS
            .iter()
            .map(|variant| {
                let reason = if !windows {
                    Some("llama.cpp 自动安装仅支持 Windows x86-64。".into())
                } else if variant.id == "vulkan" && !gpu {
                    Some("未检测到可用显卡。".into())
                } else if variant.id.starts_with("cuda") && !nvidia {
                    Some("需要 NVIDIA 显卡。".into())
                } else if variant.id == "cuda13"
                    && (driver.unwrap_or(0) < 580 || compute.unwrap_or(0) < 75)
                {
                    Some("需要 580 以上驱动和计算能力 7.5 以上的 NVIDIA 显卡。".into())
                } else if variant.id == "cuda12"
                    && (driver.unwrap_or(0) < 551 || compute.unwrap_or(0) < 50)
                {
                    Some("需要 551 以上驱动和计算能力 5.0 以上的 NVIDIA 显卡。".into())
                } else {
                    None
                };
                LlamaDownloadOption {
                    id: variant.id.into(),
                    label: variant.label.into(),
                    download_mb: (variant.assets.iter().map(|asset| asset.size).sum::<u64>()
                        + 999_999)
                        / 1_000_000,
                    suitable: reason.is_none(),
                    recommended: false,
                    reason,
                }
            })
            .collect::<Vec<_>>();
        if let Some(option) = options.iter_mut().rev().find(|option| option.suitable) {
            option.recommended = true;
        }
        options
    }

    fn provider_for_format(format: &str) -> Option<&'static str> {
        match format {
            "whisper-ggml" | "qwen3-gguf" => Some("crispasr"),
            "qwen3-openvino" => Some("openvino"),
            "llama-gguf" => Some("llama"),
            _ => None,
        }
    }

    fn provider_for_engine(id: &str) -> Result<&'static str, String> {
        if id == LLAMA_ENGINE_ID {
            Ok("llama")
        } else if id == resources::OPENVINO_ENGINE_ID {
            Ok("openvino")
        } else {
            resources::engine_spec(id)?;
            Ok("crispasr")
        }
    }

    fn supports_descriptor(
        engine_id: &str,
        descriptor: &ModelDescriptor,
        capability: &str,
    ) -> Result<bool, String> {
        let provider = Self::provider_for_engine(engine_id)?;
        let supported = if provider == "llama" {
            TEXT_GENERATION
        } else {
            TRANSCRIPTION
        };
        Ok(
            Self::provider_for_format(&descriptor.format) == Some(provider)
                && capability == supported
                && descriptor
                    .capabilities
                    .iter()
                    .any(|item| item == capability),
        )
    }

    pub fn new(root: impl Into<PathBuf>) -> Result<Self, String> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|error| error.to_string())?;
        Ok(Self {
            root: fs::canonicalize(root).map_err(|error| error.to_string())?,
            llama_server: Mutex::new(None),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn registry(&self) -> Result<Registry, String> {
        let path = self.root.join("sdk-resources.json");
        let backup = self.root.join("sdk-resources.json.bak");
        if path.is_file() {
            return serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
                .map_err(|error| format!("SDK 资源记录无效: {error}"));
        }
        if backup.is_file() {
            return serde_json::from_slice(&fs::read(backup).map_err(|error| error.to_string())?)
                .map_err(|error| format!("SDK 资源备份无效: {error}"));
        }
        // Import paths without moving or rewriting existing application resources.
        let legacy = resources::load_selection(&self.root)?;
        Ok(Registry {
            engine_paths: legacy.engine_paths,
            model_paths: legacy.model_paths,
            custom_models: HashMap::new(),
        })
    }

    fn save_registry(&self, registry: &Registry) -> Result<(), String> {
        // shortcut: callers sharing one directory must serialize mutations; add a file lock for concurrent processes.
        let path = self.root.join("sdk-resources.json");
        let temporary = self.root.join("sdk-resources.json.tmp");
        fs::write(
            &temporary,
            serde_json::to_vec_pretty(registry).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        // Windows cannot rename over an existing file. Keep a recoverable copy until replacement succeeds.
        let backup = self.root.join("sdk-resources.json.bak");
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
        let _ = fs::remove_file(backup);
        Ok(())
    }

    pub fn import_legacy_paths(&self) -> Result<(), String> {
        let legacy = resources::load_selection(&self.root)?;
        let mut registry = self.registry()?;
        for (id, path) in legacy.engine_paths {
            registry.engine_paths.entry(id).or_insert(path);
        }
        for (id, path) in legacy.model_paths {
            registry.model_paths.entry(id).or_insert(path);
        }
        self.save_registry(&registry)
    }

    pub(crate) fn clear_engine_override(&self, id: &str) -> Result<(), String> {
        let mut registry = self.registry()?;
        if registry.engine_paths.remove(id).is_some() {
            self.save_registry(&registry)?;
        }
        Ok(())
    }

    pub(crate) fn clear_model_override(&self, id: &str) -> Result<(), String> {
        let mut registry = self.registry()?;
        if registry.model_paths.remove(id).is_some() {
            self.save_registry(&registry)?;
        }
        Ok(())
    }

    pub fn engines(&self) -> Result<Vec<EngineInfo>, String> {
        let registry = self.registry()?;
        let selection = resources::Selection::default();
        let system = diagnostics::system_check(&selection);
        let mut engines = Vec::new();
        for spec in resources::ENGINES {
            let path = self.engine_path_with(&registry, spec.id)?;
            let (mut suitable, mut reason) = diagnostics::suitability(spec, &system);
            if suitable {
                if let Some(path) = &path {
                    if let Err(error) = diagnostics::probe_crispasr(spec, path, &system, None) {
                        suitable = false;
                        reason = Some(error);
                    }
                }
            }
            engines.push(EngineInfo {
                id: spec.id.into(),
                provider: "crispasr".into(),
                label: spec.label.into(),
                version: resources::ENGINE_VERSION.into(),
                download_mb: spec.size_mb,
                path,
                suitable,
                reason,
                capabilities: vec![TRANSCRIPTION.into()],
            });
        }
        let path = self.engine_path_with(&registry, resources::OPENVINO_ENGINE_ID)?;
        let suitable =
            (system.platform == "windows" && system.architecture == "x86_64") || path.is_some();
        engines.push(EngineInfo {
            id: resources::OPENVINO_ENGINE_ID.into(),
            provider: "openvino".into(),
            label: "OpenVINO CPU".into(),
            version: "2026.4.1".into(),
            download_mb: 0,
            path,
            suitable,
            reason: (!suitable).then_some("当前自动安装仅支持 Windows x86-64。".into()),
            capabilities: vec![TRANSCRIPTION.into()],
        });
        let suitable = cfg!(all(windows, target_arch = "x86_64"));
        let recommended_llama = Self::llama_download_options_for(&system)
            .into_iter()
            .find(|option| option.recommended);
        let llama_path = self.engine_path_with(&registry, LLAMA_ENGINE_ID)?;
        let managed_llama = llama_path
            .as_ref()
            .is_some_and(|path| path == &self.root.join("engines/llama/llama-server.exe"));
        engines.push(EngineInfo {
            id: LLAMA_ENGINE_ID.into(),
            provider: "llama".into(),
            label: "llama.cpp".into(),
            version: if managed_llama {
                resources::LLAMA_ENGINE_VERSION
            } else {
                "local"
            }
            .into(),
            download_mb: recommended_llama.map_or(0, |option| option.download_mb),
            path: llama_path,
            suitable,
            reason: (!suitable).then_some("llama.cpp 接入目前仅支持 Windows x86-64。".into()),
            capabilities: vec![TEXT_GENERATION.into()],
        });
        Ok(engines)
    }

    pub fn models(&self) -> Result<Vec<InstalledModel>, String> {
        let registry = self.registry()?;
        let mut models = resources::MODELS
            .iter()
            .map(|spec| {
                let capability = if spec.format == "llama-gguf" {
                    TEXT_GENERATION
                } else {
                    TRANSCRIPTION
                };
                let descriptor = ModelDescriptor {
                    id: spec.id.into(),
                    label: spec.label.into(),
                    format: spec.format.into(),
                    capabilities: vec![capability.into()],
                };
                InstalledModel {
                    path: self.model_path_with(&registry, &descriptor),
                    descriptor,
                    provider: Self::provider_for_format(spec.format)
                        .expect("catalog model format")
                        .into(),
                    size_bytes: Some(spec.size),
                    downloadable: true,
                }
            })
            .collect::<Vec<_>>();
        let descriptor = ModelDescriptor {
            id: resources::OPENVINO_MODEL_ID.into(),
            label: "Qwen3-ASR 0.6B INT8".into(),
            format: "qwen3-openvino".into(),
            capabilities: vec![TRANSCRIPTION.into()],
        };
        models.push(InstalledModel {
            path: self.model_path_with(&registry, &descriptor),
            descriptor,
            provider: "openvino".into(),
            size_bytes: Some(1_281_675_864),
            downloadable: true,
        });
        let mut custom_models = registry.custom_models.values().collect::<Vec<_>>();
        custom_models.sort_unstable_by(|left, right| left.id.cmp(&right.id));
        for descriptor in custom_models {
            models.push(InstalledModel {
                path: self.model_path_with(&registry, descriptor),
                provider: Self::provider_for_format(&descriptor.format)
                    .ok_or_else(|| format!("尚未支持的模型格式: {}", descriptor.format))?
                    .into(),
                size_bytes: None,
                downloadable: false,
                descriptor: descriptor.clone(),
            });
        }
        Ok(models)
    }

    pub fn engine_path(&self, id: &str) -> Result<Option<PathBuf>, String> {
        self.engine_path_with(&self.registry()?, id)
    }

    fn engine_path_with(&self, registry: &Registry, id: &str) -> Result<Option<PathBuf>, String> {
        if id != resources::OPENVINO_ENGINE_ID && id != LLAMA_ENGINE_ID {
            resources::engine_spec(id)?;
        }
        let selection = Selection {
            engine_paths: registry.engine_paths.clone(),
            model_paths: registry.model_paths.clone(),
            ..Selection::default()
        };
        Ok(if id == LLAMA_ENGINE_ID {
            registry
                .engine_paths
                .get(id)
                .map(PathBuf::from)
                .filter(|path| path.is_file())
                .or_else(|| {
                    let path = self.root.join("engines/llama/llama-server.exe");
                    path.is_file().then_some(path)
                })
        } else if id == resources::OPENVINO_ENGINE_ID {
            resources::openvino_python(&self.root, &selection)
        } else {
            resources::engine_path(&self.root, &selection, id)
        })
    }

    pub fn model_path(&self, id: &str) -> Result<Option<PathBuf>, String> {
        let registry = self.registry()?;
        let descriptor = self.descriptor_with(&registry, id)?;
        Ok(self.model_path_with(&registry, &descriptor))
    }

    fn model_path_with(
        &self,
        registry: &Registry,
        descriptor: &ModelDescriptor,
    ) -> Option<PathBuf> {
        let selection = Selection {
            engine_paths: registry.engine_paths.clone(),
            model_paths: registry.model_paths.clone(),
            ..Selection::default()
        };
        if descriptor.id == resources::OPENVINO_MODEL_ID {
            resources::openvino_model(&self.root, &selection)
        } else if let Ok(spec) = resources::model_spec(&descriptor.id) {
            resources::model_path(&self.root, &selection, spec)
        } else {
            registry
                .model_paths
                .get(&descriptor.id)
                .map(PathBuf::from)
                .filter(|path| self.validate_model(descriptor, path).is_ok())
        }
    }

    fn descriptor_with(&self, registry: &Registry, id: &str) -> Result<ModelDescriptor, String> {
        if id == resources::OPENVINO_MODEL_ID {
            return Ok(ModelDescriptor {
                id: id.into(),
                label: "Qwen3-ASR 0.6B INT8".into(),
                format: "qwen3-openvino".into(),
                capabilities: vec![TRANSCRIPTION.into()],
            });
        }
        if let Ok(spec) = resources::model_spec(id) {
            let capability = if spec.format == "llama-gguf" {
                TEXT_GENERATION
            } else {
                TRANSCRIPTION
            };
            return Ok(ModelDescriptor {
                id: id.into(),
                label: spec.label.into(),
                format: spec.format.into(),
                capabilities: vec![capability.into()],
            });
        }
        registry
            .custom_models
            .get(id)
            .cloned()
            .ok_or_else(|| format!("未知模型: {id}"))
    }

    fn validate_model(&self, descriptor: &ModelDescriptor, path: &Path) -> Result<(), String> {
        if descriptor.id == resources::OPENVINO_MODEL_ID {
            resources::verify_openvino_model(path)?;
            for (name, hash) in resources::OPENVINO_MODEL_HASHES {
                crate::install::verify_sha256(&path.join(name), hash)?;
            }
            return Ok(());
        }
        if let Ok(spec) = resources::model_spec(&descriptor.id) {
            return resources::verify_model(path, spec);
        }
        match descriptor.format.as_str() {
            "whisper-ggml" => {
                use std::io::Read;
                let mut header = [0; 4];
                fs::File::open(path)
                    .and_then(|mut file| file.read_exact(&mut header))
                    .map_err(|error| format!("无法读取 GGML 模型: {error}"))?;
                if &header != b"lmgg" {
                    return Err("模型不是 Whisper GGML 格式。".into());
                }
            }
            "qwen3-gguf" | "llama-gguf" => {
                use std::io::Read;
                let mut header = [0; 4];
                if !path
                    .extension()
                    .is_some_and(|extension| extension == "gguf")
                {
                    return Err("GGUF 模型需要 .gguf 文件。".into());
                }
                fs::File::open(path)
                    .and_then(|mut file| file.read_exact(&mut header))
                    .map_err(|error| format!("无法读取 GGUF 模型: {error}"))?;
                if &header != b"GGUF" {
                    return Err("模型不是 GGUF 格式。".into());
                }
            }
            "qwen3-openvino" => resources::verify_openvino_model(path)?,
            _ => return Err(format!("尚未支持的模型格式: {}", descriptor.format)),
        }
        Ok(())
    }

    pub fn register_local_engine(&self, id: &str, path: &Path) -> Result<(), String> {
        if id != resources::OPENVINO_ENGINE_ID && id != LLAMA_ENGINE_ID {
            resources::engine_spec(id)?;
        }
        let expected = if id == LLAMA_ENGINE_ID {
            "llama-server.exe"
        } else if id == resources::OPENVINO_ENGINE_ID {
            if cfg!(windows) {
                "python.exe"
            } else {
                "python3"
            }
        } else {
            "crispasr.exe"
        };
        let valid_name = path.file_name().is_some_and(|name| {
            name.to_string_lossy().eq_ignore_ascii_case(expected)
                || (id == resources::OPENVINO_ENGINE_ID && !cfg!(windows) && name == "python")
        });
        if !path.is_absolute() || !path.is_file() || !valid_name {
            return Err(format!("请选择有效的 {expected} 完整路径。"));
        }
        if id == LLAMA_ENGINE_ID {
            if !cfg!(all(windows, target_arch = "x86_64")) {
                return Err("llama.cpp 接入目前仅支持 Windows x86-64。".into());
            }
            if !hidden_command(path)
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success())
            {
                return Err("此 llama-server.exe 无法运行。".into());
            }
        }
        if id == resources::OPENVINO_ENGINE_ID
            && !hidden_command(path)
                .args(["-c", "import openvino, onnxruntime, numpy"])
                .output()
                .is_ok_and(|output| output.status.success())
        {
            return Err("此 Python 环境缺少 OpenVINO、ONNX Runtime 或 NumPy。".into());
        }
        if id != resources::OPENVINO_ENGINE_ID && id != LLAMA_ENGINE_ID {
            let spec = resources::engine_spec(id)?;
            let system = diagnostics::system_check(&Selection::default());
            let (suitable, reason) = diagnostics::suitability(spec, &system);
            if !suitable {
                return Err(reason.unwrap_or_else(|| "此引擎不适合当前设备。".into()));
            }
            let model = self
                .models()?
                .into_iter()
                .filter(|item| {
                    item.descriptor.format == "whisper-ggml"
                        || item.descriptor.format == "qwen3-gguf"
                })
                .find_map(|item| item.path);
            diagnostics::probe_crispasr(spec, path, &system, model.as_deref())?;
        }
        let mut registry = self.registry()?;
        registry
            .engine_paths
            .insert(id.into(), path.to_string_lossy().into_owned());
        self.save_registry(&registry)
    }

    pub fn register_local_model(
        &self,
        descriptor: ModelDescriptor,
        path: &Path,
    ) -> Result<(), String> {
        if !path.is_absolute()
            || descriptor.id.is_empty()
            || !descriptor
                .id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
        {
            return Err("模型 ID 或路径无效。".into());
        }
        let supported = match descriptor.format.as_str() {
            "whisper-ggml" | "qwen3-gguf" | "qwen3-openvino" => TRANSCRIPTION,
            "llama-gguf" => TEXT_GENERATION,
            _ => return Err(format!("尚未支持的模型格式: {}", descriptor.format)),
        };
        if descriptor.capabilities.is_empty()
            || !descriptor.capabilities.iter().all(|item| item == supported)
        {
            return Err("当前引擎尚不支持所声明的模型能力。".into());
        }
        let mut registry = self.registry()?;
        if descriptor.id == resources::OPENVINO_MODEL_ID
            || resources::model_spec(&descriptor.id).is_ok()
        {
            let known = self.descriptor_with(&registry, &descriptor.id)?;
            if known.format != descriptor.format {
                return Err("模型格式与清单不符。".into());
            }
        } else {
            registry
                .custom_models
                .insert(descriptor.id.clone(), descriptor.clone());
        }
        self.validate_model(&descriptor, path)?;
        registry
            .model_paths
            .insert(descriptor.id, path.to_string_lossy().into_owned());
        self.save_registry(&registry)
    }

    pub fn check(
        &self,
        engine_id: &str,
        model_id: &str,
        capability: &str,
    ) -> Result<Readiness, String> {
        let registry = self.registry()?;
        let descriptor = self.descriptor_with(&registry, model_id)?;
        if !Self::supports_descriptor(engine_id, &descriptor, capability)? {
            return Ok(Readiness {
                ready: false,
                reason: Some("引擎与模型格式或任务能力不兼容。".into()),
            });
        }
        let engine = self
            .engines()?
            .into_iter()
            .find(|item| item.id == engine_id)
            .ok_or_else(|| format!("未知引擎: {engine_id}"))?;
        if !engine.suitable {
            return Ok(Readiness {
                ready: false,
                reason: engine.reason,
            });
        }
        if engine.path.is_none() {
            return Ok(Readiness {
                ready: false,
                reason: Some("推理引擎尚未安装或登记。".into()),
            });
        }
        let Some(model) = self.model_path_with(&registry, &descriptor) else {
            return Ok(Readiness {
                ready: false,
                reason: Some("模型尚未安装或登记。".into()),
            });
        };
        if Self::provider_for_engine(engine_id)? == "crispasr" {
            let spec = resources::engine_spec(engine_id)?;
            let system = diagnostics::system_check(&Selection::default());
            if let Err(error) = diagnostics::probe_crispasr(
                spec,
                engine.path.as_deref().unwrap(),
                &system,
                Some(&model),
            ) {
                return Ok(Readiness {
                    ready: false,
                    reason: Some(error),
                });
            }
        }
        Ok(Readiness {
            ready: true,
            reason: None,
        })
    }

    pub fn validate_pair(&self, engine_id: &str, model_id: &str) -> Result<(), String> {
        let capability = if Self::provider_for_engine(engine_id)? == "llama" {
            TEXT_GENERATION
        } else {
            TRANSCRIPTION
        };
        self.validate_pair_for(engine_id, model_id, capability)
    }

    pub fn validate_pair_for(
        &self,
        engine_id: &str,
        model_id: &str,
        capability: &str,
    ) -> Result<(), String> {
        let descriptor = self.descriptor_with(&self.registry()?, model_id)?;
        if Self::supports_descriptor(engine_id, &descriptor, capability)? {
            Ok(())
        } else {
            Err("引擎与模型格式或任务能力不兼容。".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_model_format_and_incompatible_pair() {
        let root = std::env::temp_dir().join(format!("model-sdk-test-{}", std::process::id()));
        let sdk = ModelSdk::new(&root).unwrap();
        let descriptor = ModelDescriptor {
            id: "custom".into(),
            label: "Custom".into(),
            format: "unknown".into(),
            capabilities: vec![TRANSCRIPTION.into()],
        };
        assert!(sdk
            .register_local_model(descriptor, &root.join("missing"))
            .is_err());
        assert!(
            !sdk.check("vulkan", resources::OPENVINO_MODEL_ID, TRANSCRIPTION)
                .unwrap()
                .ready
        );
        sdk.validate_pair("vulkan", "base").unwrap();
        sdk.validate_pair(resources::OPENVINO_ENGINE_ID, resources::OPENVINO_MODEL_ID)
            .unwrap();
        assert!(sdk
            .validate_pair("vulkan", resources::OPENVINO_MODEL_ID)
            .is_err());
        assert!(sdk
            .validate_pair(resources::OPENVINO_ENGINE_ID, "base")
            .is_err());
        assert!(sdk.validate_pair("unknown", "base").is_err());
        assert!(sdk.validate_pair("vulkan", "unknown").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn imports_existing_paths_and_persists_a_local_model() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("model-sdk-registry-{nonce}"));
        let sdk = ModelSdk::new(&root).unwrap();
        let engine = root.join("crispasr.exe");
        fs::write(&engine, b"existing engine").unwrap();
        let mut selection = Selection::default();
        selection
            .engine_paths
            .insert("vulkan".into(), engine.to_string_lossy().into_owned());
        resources::save_selection(&root, &selection).unwrap();
        let model = root.join("custom.bin");
        fs::write(&model, b"lmggsample").unwrap();
        sdk.register_local_model(
            ModelDescriptor {
                id: "custom".into(),
                label: "Custom".into(),
                format: "whisper-ggml".into(),
                capabilities: vec![TRANSCRIPTION.into()],
            },
            &model,
        )
        .unwrap();
        let reopened = ModelSdk::new(&root).unwrap();
        assert_eq!(reopened.engine_path("vulkan").unwrap(), Some(engine));
        assert_eq!(reopened.model_path("custom").unwrap(), Some(model));
        reopened.validate_pair("vulkan", "custom").unwrap();
        assert!(reopened
            .validate_pair(resources::OPENVINO_ENGINE_ID, "custom")
            .is_err());
        let models = reopened.models().unwrap();
        let custom = models
            .iter()
            .find(|item| item.descriptor.id == "custom")
            .unwrap();
        assert_eq!(custom.provider, "crispasr");
        assert_eq!(custom.size_bytes, None);
        assert!(!custom.downloadable);
        selection.model_id = "custom".into();
        resources::save_selection(&root, &selection).unwrap();
        assert_eq!(resources::load_selection(&root).unwrap().model_id, "custom");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn registers_qwen3_gguf_for_crispasr() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("model-sdk-qwen3-{nonce}"));
        let sdk = ModelSdk::new(&root).unwrap();
        let engine = root.join("crispasr.exe");
        fs::write(&engine, b"test engine").unwrap();
        assert!(sdk.register_local_engine("vulkan", &engine).is_err());
        let model = root.join("qwen3-asr-1.7b-custom.gguf");
        fs::write(&model, b"GGUFtest").unwrap();
        sdk.register_local_model(
            ModelDescriptor {
                id: "qwen3-custom".into(),
                label: "Custom Qwen3".into(),
                format: "qwen3-gguf".into(),
                capabilities: vec![TRANSCRIPTION.into()],
            },
            &model,
        )
        .unwrap();
        assert!(
            !sdk.check("vulkan", "qwen3-custom", TRANSCRIPTION)
                .unwrap()
                .ready
        );
        assert!(
            !sdk.check(resources::OPENVINO_ENGINE_ID, "qwen3-custom", TRANSCRIPTION)
                .unwrap()
                .ready
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn llama_model_only_supports_text_generation() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("model-sdk-llama-{nonce}"));
        let sdk = ModelSdk::new(&root).unwrap();
        let model = root.join("text.gguf");
        fs::write(&model, b"GGUFtest").unwrap();
        let descriptor = ModelDescriptor {
            id: "text".into(),
            label: "Text model".into(),
            format: "llama-gguf".into(),
            capabilities: vec![TEXT_GENERATION.into()],
        };
        sdk.register_local_model(descriptor.clone(), &model)
            .unwrap();
        sdk.validate_pair(LLAMA_ENGINE_ID, "text").unwrap();
        sdk.validate_pair_for(LLAMA_ENGINE_ID, "text", TEXT_GENERATION)
            .unwrap();
        assert!(sdk
            .validate_pair_for(LLAMA_ENGINE_ID, "text", TRANSCRIPTION)
            .is_err());
        assert!(sdk.validate_pair("cpu", "text").is_err());
        assert!(
            !sdk.check(LLAMA_ENGINE_ID, "text", TRANSCRIPTION)
                .unwrap()
                .ready
        );
        let registered = sdk.models().unwrap();
        let item = registered
            .iter()
            .find(|item| item.descriptor.id == "text")
            .unwrap();
        assert_eq!(item.provider, "llama");
        assert_eq!(item.path.as_deref(), Some(model.as_path()));
        assert!(sdk
            .register_local_model(
                ModelDescriptor {
                    capabilities: vec![TRANSCRIPTION.into()],
                    ..descriptor
                },
                &model,
            )
            .is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn llama_download_options_follow_hardware() {
        let mut system = diagnostics::SystemCheck {
            platform: "windows".into(),
            architecture: "x86_64".into(),
            windows_build: Some(19045),
            cpu: String::new(),
            avx2: true,
            gpus: vec![],
            nvidia_driver: None,
            nvidia_compute_capability: None,
            available_memory_bytes: None,
            nvidia_free_vram_bytes: None,
            ffmpeg_path: None,
            ffmpeg_available: false,
            diagnostics: vec![],
        };
        let recommended = |system: &diagnostics::SystemCheck| {
            ModelSdk::llama_download_options_for(system)
                .into_iter()
                .find(|option| option.recommended)
                .map(|option| option.id)
        };
        assert_eq!(recommended(&system).as_deref(), Some("cpu"));
        system.gpus.push(diagnostics::GpuInfo {
            name: "AMD Radeon".into(),
            vendor: "amd".into(),
            driver: String::new(),
        });
        assert_eq!(recommended(&system).as_deref(), Some("vulkan"));
        system.gpus[0].vendor = "nvidia".into();
        system.nvidia_driver = Some("570.0".into());
        system.nvidia_compute_capability = Some("8.9".into());
        assert_eq!(recommended(&system).as_deref(), Some("cuda12"));
        system.nvidia_driver = Some("580.0".into());
        assert_eq!(recommended(&system).as_deref(), Some("cuda13"));
        system.nvidia_driver = Some("500.0".into());
        assert_eq!(recommended(&system).as_deref(), Some("vulkan"));
        system.platform = "linux".into();
        assert_eq!(recommended(&system), None);
    }

    #[test]
    fn hy_mt_is_a_downloadable_text_model() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("model-sdk-hy-mt-{nonce}"));
        let sdk = ModelSdk::new(&root).unwrap();
        let model = sdk
            .models()
            .unwrap()
            .into_iter()
            .find(|model| model.descriptor.id == resources::HY_MT_MODEL_ID)
            .unwrap();
        assert_eq!(model.provider, "llama");
        assert_eq!(model.descriptor.capabilities, vec![TEXT_GENERATION]);
        assert!(model.downloadable);
        sdk.validate_pair_for(LLAMA_ENGINE_ID, resources::HY_MT_MODEL_ID, TEXT_GENERATION)
            .unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn managed_llama_runtime_is_found_after_local_override_is_cleared() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("model-sdk-llama-path-{nonce}"));
        let sdk = ModelSdk::new(&root).unwrap();
        let managed = sdk.root().join("engines/llama/llama-server.exe");
        fs::create_dir_all(managed.parent().unwrap()).unwrap();
        fs::write(&managed, b"managed").unwrap();
        let external = root.join("llama-server.exe");
        fs::write(&external, b"external").unwrap();
        assert_eq!(
            sdk.engine_path(LLAMA_ENGINE_ID).unwrap(),
            Some(managed.clone())
        );
        let mut registry = sdk.registry().unwrap();
        registry
            .engine_paths
            .insert(LLAMA_ENGINE_ID.into(), external.to_string_lossy().into());
        sdk.save_registry(&registry).unwrap();
        assert_eq!(sdk.engine_path(LLAMA_ENGINE_ID).unwrap(), Some(external));
        sdk.clear_engine_override(LLAMA_ENGINE_ID).unwrap();
        assert_eq!(sdk.engine_path(LLAMA_ENGINE_ID).unwrap(), Some(managed));
        fs::remove_dir_all(root).unwrap();
    }
}
