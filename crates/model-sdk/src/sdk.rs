use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use serde::{Deserialize, Serialize};

use crate::{
    catalog::{ArtifactFormat, ModelProfile, TaskKind},
    diagnostics,
    engines::{self, Family},
    resources::{self, Selection},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDescriptor {
    pub id: String,
    pub label: String,
    pub profile: ModelProfile,
    pub artifact_format: ArtifactFormat,
    pub task: TaskKind,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    pub descriptor: ModelDescriptor,
    pub path: Option<PathBuf>,
    pub size_bytes: Option<u64>,
    pub downloadable: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineInfo {
    pub id: String,
    pub family: String,
    pub variant: String,
    pub label: String,
    pub version: String,
    pub download_mb: u64,
    pub path: Option<PathBuf>,
    pub suitable: bool,
    pub reason: Option<String>,
    pub tasks: Vec<TaskKind>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Readiness {
    pub ready: bool,
    pub reason: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Registry {
    version: u32,
    engine_paths: HashMap<String, String>,
    model_paths: HashMap<String, String>,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            version: 1,
            engine_paths: HashMap::new(),
            model_paths: HashMap::new(),
        }
    }
}

pub struct ModelSdk {
    root: PathBuf,
    pub(crate) llama_server: Mutex<Option<crate::generate::LlamaServer>>,
}

impl ModelSdk {
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
        let source = if path.is_file() {
            path
        } else if backup.is_file() {
            backup
        } else {
            return Ok(Registry::default());
        };
        let bytes = fs::read(source).map_err(|error| error.to_string())?;
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|error| format!("SDK 资源记录无效: {error}"))?;
        if value.get("version").and_then(|v| v.as_u64()) != Some(1) {
            return Err("SDK 资源记录版本不兼容；旧格式不会自动迁移，请重新登记资源。".into());
        }
        serde_json::from_value(value).map_err(|error| format!("SDK 资源记录无效: {error}"))
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

    pub(crate) fn descriptor(id: &str) -> Result<ModelDescriptor, String> {
        if let Some(spec) = resources::wd14_spec(id) {
            return Ok(ModelDescriptor {
                id: spec.id.into(),
                label: spec.label.into(),
                profile: spec.profile,
                artifact_format: ArtifactFormat::Onnx,
                task: TaskKind::ImageTagging,
            });
        }
        if id == resources::OPENVINO_MODEL_ID {
            return Ok(ModelDescriptor {
                id: id.into(),
                label: "Qwen3-ASR 0.6B INT8".into(),
                profile: ModelProfile::Qwen3Asr06b,
                artifact_format: ArtifactFormat::OpenVinoIr,
                task: TaskKind::Transcription,
            });
        }
        let spec = resources::model_spec(id)?;
        Ok(ModelDescriptor {
            id: spec.id.into(),
            label: spec.label.into(),
            profile: spec.profile,
            artifact_format: spec.format,
            task: spec.task,
        })
    }

    pub fn models(&self) -> Result<Vec<InstalledModel>, String> {
        let registry = self.registry()?;
        let mut models = Vec::new();
        for spec in resources::MODELS {
            let descriptor = Self::descriptor(spec.id)?;
            models.push(InstalledModel {
                path: self.model_path_with(&registry, spec.id),
                descriptor,
                size_bytes: Some(spec.size),
                downloadable: true,
            });
        }
        for spec in resources::WD14_MODELS {
            models.push(InstalledModel {
                descriptor: Self::descriptor(spec.id)?,
                path: self.model_path_with(&registry, spec.id),
                size_bytes: Some(spec.size + spec.csv_size),
                downloadable: true,
            });
        }
        models.push(InstalledModel {
            descriptor: Self::descriptor(resources::OPENVINO_MODEL_ID)?,
            path: self.model_path_with(&registry, resources::OPENVINO_MODEL_ID),
            size_bytes: Some(1_281_675_864),
            downloadable: true,
        });
        Ok(models)
    }

    pub fn engines(&self) -> Result<Vec<EngineInfo>, String> {
        let registry = self.registry()?;
        let system = diagnostics::system_check(&Selection::default());
        engines::ADAPTERS
            .iter()
            .map(|adapter| {
                let path = self.engine_path_with(&registry, adapter.id)?;
                let (suitable, reason) = adapter.suitability(&system, path.as_deref())?;
                let (label, version, download_mb) = match adapter.family {
                    Family::CrispAsr => {
                        let spec = resources::engine_spec(adapter.id)?;
                        (spec.label, resources::ENGINE_VERSION, spec.size_mb)
                    }
                    Family::OpenVino => ("OpenVINO CPU", "2026.4.1", 0),
                    Family::OnnxRuntime => {
                        ("ONNX Runtime CPU", resources::ONNXRUNTIME_ENGINE_VERSION, 0)
                    }
                    Family::Llama => {
                        let variant = resources::llama_variant(adapter.id)?;
                        let mb = (variant.assets.iter().map(|asset| asset.size).sum::<u64>()
                            + 999_999)
                            / 1_000_000;
                        (variant.label, resources::LLAMA_ENGINE_VERSION, mb)
                    }
                };
                Ok(EngineInfo {
                    id: adapter.id.into(),
                    family: match adapter.family {
                        Family::CrispAsr => "crispasr",
                        Family::OpenVino => "openvino",
                        Family::Llama => "llama",
                        Family::OnnxRuntime => "onnxruntime",
                    }
                    .into(),
                    variant: adapter.variant.into(),
                    label: label.into(),
                    version: version.into(),
                    download_mb,
                    path,
                    suitable,
                    reason,
                    tasks: adapter.tasks.to_vec(),
                })
            })
            .collect()
    }

    pub fn engine_path(&self, id: &str) -> Result<Option<PathBuf>, String> {
        self.engine_path_with(&self.registry()?, id)
    }

    fn engine_path_with(&self, registry: &Registry, id: &str) -> Result<Option<PathBuf>, String> {
        let adapter = engines::get(id)?;
        let override_path = registry
            .engine_paths
            .get(id)
            .map(PathBuf::from)
            .filter(|path| path.is_file());
        let managed = adapter.managed_path(&self.root);
        Ok(override_path.or_else(|| {
            (managed.is_file()
                && match adapter.family {
                    Family::OpenVino => self.root.join("engines/openvino/ready.txt").is_file(),
                    Family::OnnxRuntime => self
                        .root
                        .join("engines/onnxruntime-cpu/ready.txt")
                        .is_file(),
                    _ => true,
                })
            .then_some(managed)
        }))
    }

    pub fn model_path(&self, id: &str) -> Result<Option<PathBuf>, String> {
        Self::descriptor(id)?;
        Ok(self.model_path_with(&self.registry()?, id))
    }

    fn model_path_with(&self, registry: &Registry, id: &str) -> Option<PathBuf> {
        let registered = registry.model_paths.get(id).map(PathBuf::from);
        let managed = if id == resources::OPENVINO_MODEL_ID {
            self.root.join("models/openvino/qwen3_asr_int8")
        } else if resources::wd14_spec(id).is_some() {
            self.root.join("models/wd14").join(id)
        } else {
            self.root
                .join("models")
                .join(resources::model_spec(id).ok()?.file)
        };
        registered
            .into_iter()
            .chain(std::iter::once(managed))
            .find(|path| Self::validate_model(id, path).is_ok())
    }

    fn validate_model(id: &str, path: &Path) -> Result<(), String> {
        if let Some(spec) = resources::wd14_spec(id) {
            resources::verify_wd14_model(path, spec)
        } else if id == resources::OPENVINO_MODEL_ID {
            resources::verify_openvino_model(path)?;
            for (name, hash) in resources::OPENVINO_MODEL_HASHES {
                crate::install::verify_sha256(&path.join(name), hash)?;
            }
            Ok(())
        } else {
            resources::verify_model(path, resources::model_spec(id)?)
        }
    }

    pub fn register_local_model(&self, id: &str, path: &Path) -> Result<(), String> {
        Self::descriptor(id)?;
        if !path.is_absolute() {
            return Err("请选择模型的绝对路径。".into());
        }
        Self::validate_model(id, path)?;
        let mut registry = self.registry()?;
        registry
            .model_paths
            .insert(id.into(), path.to_string_lossy().into_owned());
        self.save_registry(&registry)
    }

    pub fn register_local_engine(&self, id: &str, path: &Path) -> Result<(), String> {
        engines::get(id)?.verify_local(path)?;
        let mut registry = self.registry()?;
        registry
            .engine_paths
            .insert(id.into(), path.to_string_lossy().into_owned());
        self.save_registry(&registry)
    }

    pub fn validate_pair_for(
        &self,
        engine_id: &str,
        model_id: &str,
        task: TaskKind,
    ) -> Result<(), String> {
        let adapter = engines::get(engine_id)?;
        let model = Self::descriptor(model_id)?;
        if model.task == task && adapter.supports(model.profile, model.artifact_format, task) {
            Ok(())
        } else {
            Err("引擎与模型描述或任务类型不兼容。".into())
        }
    }

    pub fn check(
        &self,
        engine_id: &str,
        model_id: &str,
        task: TaskKind,
    ) -> Result<Readiness, String> {
        if let Err(error) = self.validate_pair_for(engine_id, model_id, task) {
            return Ok(Readiness {
                ready: false,
                reason: Some(error),
            });
        }
        let engine = self
            .engines()?
            .into_iter()
            .find(|item| item.id == engine_id)
            .expect("catalog engine");
        if !engine.suitable {
            return Ok(Readiness {
                ready: false,
                reason: engine.reason,
            });
        }
        let Some(path) = engine.path else {
            return Ok(Readiness {
                ready: false,
                reason: Some("推理引擎尚未安装或登记。".into()),
            });
        };
        let Some(model) = self.model_path(model_id)? else {
            return Ok(Readiness {
                ready: false,
                reason: Some("模型尚未安装或登记。".into()),
            });
        };
        let profile = Self::descriptor(model_id)?.profile;
        let probe = engines::get(engine_id)?.probe(&path, Some((&model, profile)));
        if let Err(error) = probe {
            return Ok(Readiness {
                ready: false,
                reason: Some(error),
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

    fn temp_sdk(name: &str) -> ModelSdk {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        ModelSdk::new(std::env::temp_dir().join(format!("model-sdk-{name}-{nonce}"))).unwrap()
    }

    #[test]
    fn catalog_compatibility_is_profile_format_and_task_specific() {
        let sdk = temp_sdk("compatibility");
        assert_eq!(engines::ADAPTERS.len(), 11);
        assert_eq!(
            sdk.models().unwrap().len(),
            resources::MODELS.len() + resources::WD14_MODELS.len() + 1
        );
        for model in sdk.models().unwrap() {
            for engine in engines::ADAPTERS {
                for task in [
                    TaskKind::Transcription,
                    TaskKind::TextGeneration,
                    TaskKind::ObjectDetection,
                    TaskKind::ImageTagging,
                ] {
                    let expected = match engine.family {
                        Family::CrispAsr => {
                            task == TaskKind::Transcription
                                && [
                                    "tiny",
                                    "base",
                                    "small",
                                    "medium",
                                    "turbo",
                                    "large-v2",
                                    "qwen3-1.7b-q4",
                                    "qwen3-1.7b-q8",
                                    "qwen3-1.7b-ja-q4",
                                    "qwen3-1.7b-ja-q8",
                                ]
                                .contains(&model.descriptor.id.as_str())
                        }
                        Family::OpenVino => {
                            task == TaskKind::Transcription
                                && model.descriptor.id == resources::OPENVINO_MODEL_ID
                        }
                        Family::Llama => {
                            task == TaskKind::TextGeneration
                                && model.descriptor.id == resources::HY_MT_MODEL_ID
                        }
                        Family::OnnxRuntime => {
                            (task == TaskKind::ObjectDetection
                                && [
                                    "yolov8n-det-fp32",
                                    "yolov8s-det-fp32",
                                    "yolo11n-det-fp32",
                                    "yolo11s-det-fp32",
                                ]
                                .contains(&model.descriptor.id.as_str()))
                                || (task == TaskKind::ImageTagging
                                    && resources::wd14_spec(&model.descriptor.id).is_some())
                        }
                    };
                    assert_eq!(
                        sdk.validate_pair_for(engine.id, &model.descriptor.id, task)
                            .is_ok(),
                        expected,
                        "{} / {} / {task:?}",
                        engine.id,
                        model.descriptor.id,
                    );
                }
            }
        }
        assert!(sdk
            .validate_pair_for("missing", "base", TaskKind::Transcription)
            .is_err());
        assert!(sdk
            .validate_pair_for("vulkan", "base", TaskKind::Transcription)
            .is_err());
        assert!(sdk
            .validate_pair_for("crispasr-cpu", "missing", TaskKind::Transcription)
            .is_err());
        assert!(sdk
            .validate_pair_for("onnxruntime-cpu", "missing", TaskKind::ObjectDetection)
            .is_err());
        let _ = fs::remove_dir_all(sdk.root());
    }

    #[test]
    fn multiple_runtime_adapters_can_accept_the_same_model() {
        let model = ModelSdk::descriptor("base").unwrap();
        assert!(engines::get("crispasr-cpu").unwrap().supports(
            model.profile,
            model.artifact_format,
            model.task
        ));
        assert!(engines::get("crispasr-vulkan").unwrap().supports(
            model.profile,
            model.artifact_format,
            model.task
        ));
        assert!(!engines::get("llama-cpu").unwrap().supports(
            model.profile,
            model.artifact_format,
            model.task
        ));
        assert!(!engines::get("crispasr-cpu").unwrap().supports(
            ModelProfile::HyMt15,
            ArtifactFormat::Gguf,
            TaskKind::Transcription
        ));
        let yolo = ModelSdk::descriptor("yolo11n-det-fp32").unwrap();
        assert!(engines::get("onnxruntime-cpu").unwrap().supports(
            yolo.profile,
            yolo.artifact_format,
            yolo.task
        ));
        assert!(!engines::get("onnxruntime-cpu").unwrap().supports(
            yolo.profile,
            yolo.artifact_format,
            TaskKind::TextGeneration
        ));
        let wd14 = ModelSdk::descriptor("wd-swinv2-tagger-v3").unwrap();
        assert!(engines::get("onnxruntime-cpu").unwrap().supports(
            wd14.profile,
            wd14.artifact_format,
            wd14.task
        ));
        assert!(!engines::get("onnxruntime-cpu").unwrap().supports(
            wd14.profile,
            wd14.artifact_format,
            TaskKind::ObjectDetection
        ));
    }

    #[test]
    fn compatibility_does_not_imply_runtime_readiness() {
        let sdk = temp_sdk("readiness");
        sdk.validate_pair_for(
            "llama-cpu",
            resources::HY_MT_MODEL_ID,
            TaskKind::TextGeneration,
        )
        .unwrap();
        let readiness = sdk
            .check(
                "llama-cpu",
                resources::HY_MT_MODEL_ID,
                TaskKind::TextGeneration,
            )
            .unwrap();
        assert!(!readiness.ready);
        assert!(readiness.reason.is_some());
        assert!(
            !sdk.check("llama-cpu", "base", TaskKind::TextGeneration)
                .unwrap()
                .ready
        );
        let _ = fs::remove_dir_all(sdk.root());
    }

    #[test]
    fn local_models_require_a_catalog_id_and_verified_weight() {
        let sdk = temp_sdk("offline");
        let model = sdk.root().join("fake.gguf");
        fs::write(&model, b"GGUF").unwrap();
        assert!(sdk
            .register_local_model("unknown", &model)
            .unwrap_err()
            .contains("未知模型"));
        assert!(sdk
            .register_local_model(resources::HY_MT_MODEL_ID, &model)
            .unwrap_err()
            .contains("模型大小不符"));
        assert!(sdk.model_path(resources::HY_MT_MODEL_ID).unwrap().is_none());
        let _ = fs::remove_dir_all(sdk.root());
    }

    #[test]
    fn versioned_registry_rejects_old_data_without_reading_application_selection() {
        let sdk = temp_sdk("registry");
        fs::write(
            sdk.root().join("resources.json"),
            b"{ invalid application data",
        )
        .unwrap();
        assert_eq!(sdk.models().unwrap().len(), 25);
        fs::write(
            sdk.root().join("sdk-resources.json"),
            br#"{"enginePaths":{},"modelPaths":{}}"#,
        )
        .unwrap();
        assert!(sdk.models().unwrap_err().contains("版本不兼容"));
        fs::write(
            sdk.root().join("sdk-resources.json"),
            br#"{"version":2,"enginePaths":{},"modelPaths":{}}"#,
        )
        .unwrap();
        assert!(sdk.models().unwrap_err().contains("版本不兼容"));
        sdk.save_registry(&Registry::default()).unwrap();
        assert_eq!(sdk.models().unwrap().len(), 25);
        let _ = fs::remove_dir_all(sdk.root());
    }

    #[test]
    fn llama_variants_resolve_independent_managed_paths() {
        let sdk = temp_sdk("llama-variants");
        for id in ["llama-cpu", "llama-vulkan", "llama-cuda12", "llama-cuda13"] {
            let path = sdk.root().join("engines").join(id).join("llama-server.exe");
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, id).unwrap();
            assert_eq!(sdk.engine_path(id).unwrap(), Some(path));
        }
        let _ = fs::remove_dir_all(sdk.root());
    }
}
