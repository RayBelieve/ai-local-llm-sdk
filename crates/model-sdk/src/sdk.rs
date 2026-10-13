use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    process::hidden_command,
    resources::{self, Selection},
};

pub const TRANSCRIPTION: &str = "speech-to-text";

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
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineInfo {
    pub id: String,
    pub provider: String,
    pub label: String,
    pub path: Option<PathBuf>,
    pub suitable: bool,
    pub reason: Option<String>,
    pub capabilities: Vec<String>,
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
}

impl ModelSdk {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, String> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|error| error.to_string())?;
        Ok(Self {
            root: fs::canonicalize(root).map_err(|error| error.to_string())?,
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
        let system = resources::system_check(&selection);
        let mut engines = Vec::new();
        for spec in resources::ENGINES {
            let (suitable, reason) = resources::suitability(spec, &system);
            engines.push(EngineInfo {
                id: spec.id.into(),
                provider: "crispasr".into(),
                label: spec.label.into(),
                path: self.engine_path_with(&registry, spec.id)?,
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
            path,
            suitable,
            reason: (!suitable).then_some("当前自动安装仅支持 Windows x86-64。".into()),
            capabilities: vec![TRANSCRIPTION.into()],
        });
        Ok(engines)
    }

    pub fn models(&self) -> Result<Vec<InstalledModel>, String> {
        let registry = self.registry()?;
        let mut models = resources::MODELS
            .iter()
            .map(|spec| {
                let descriptor = ModelDescriptor {
                    id: spec.id.into(),
                    label: spec.label.into(),
                    format: spec.format.into(),
                    capabilities: vec![TRANSCRIPTION.into()],
                };
                InstalledModel {
                    path: self.model_path_with(&registry, &descriptor),
                    descriptor,
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
        });
        for descriptor in registry.custom_models.values() {
            models.push(InstalledModel {
                path: self.model_path_with(&registry, descriptor),
                descriptor: descriptor.clone(),
            });
        }
        Ok(models)
    }

    pub fn engine_path(&self, id: &str) -> Result<Option<PathBuf>, String> {
        self.engine_path_with(&self.registry()?, id)
    }

    fn engine_path_with(&self, registry: &Registry, id: &str) -> Result<Option<PathBuf>, String> {
        if id != resources::OPENVINO_ENGINE_ID {
            resources::engine_spec(id)?;
        }
        let selection = Selection {
            engine_paths: registry.engine_paths.clone(),
            model_paths: registry.model_paths.clone(),
            ..Selection::default()
        };
        Ok(if id == resources::OPENVINO_ENGINE_ID {
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
            return Ok(ModelDescriptor {
                id: id.into(),
                label: spec.label.into(),
                format: spec.format.into(),
                capabilities: vec![TRANSCRIPTION.into()],
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
            "qwen3-gguf" => {
                use std::io::Read;
                let mut header = [0; 4];
                if !path
                    .extension()
                    .is_some_and(|extension| extension == "gguf")
                {
                    return Err("Qwen3 GGUF 模型需要 .gguf 文件。".into());
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
        if id != resources::OPENVINO_ENGINE_ID {
            resources::engine_spec(id)?;
        }
        let expected = if id == resources::OPENVINO_ENGINE_ID {
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
        if id == resources::OPENVINO_ENGINE_ID
            && !hidden_command(path)
                .args(["-c", "import openvino, onnxruntime, numpy"])
                .output()
                .is_ok_and(|output| output.status.success())
        {
            return Err("此 Python 环境缺少 OpenVINO、ONNX Runtime 或 NumPy。".into());
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
        if descriptor.capabilities.is_empty()
            || !descriptor
                .capabilities
                .iter()
                .all(|item| item == TRANSCRIPTION)
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
        } else if !["whisper-ggml", "qwen3-gguf", "qwen3-openvino"]
            .contains(&descriptor.format.as_str())
        {
            return Err(format!("尚未支持的模型格式: {}", descriptor.format));
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
        let format_ok = match engine_id {
            resources::OPENVINO_ENGINE_ID => descriptor.format == "qwen3-openvino",
            _ => {
                resources::engine_spec(engine_id)?;
                descriptor.format == "whisper-ggml" || descriptor.format == "qwen3-gguf"
            }
        };
        if !format_ok
            || !descriptor
                .capabilities
                .iter()
                .any(|item| item == capability)
        {
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
        if self.model_path_with(&registry, &descriptor).is_none() {
            return Ok(Readiness {
                ready: false,
                reason: Some("模型尚未安装或登记。".into()),
            });
        }
        Ok(Readiness {
            ready: true,
            reason: None,
        })
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
        sdk.register_local_engine("vulkan", &engine).unwrap();
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
            sdk.check("vulkan", "qwen3-custom", TRANSCRIPTION)
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
}
