use std::{
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
};

use crate::{
    catalog::{ArtifactFormat, ModelProfile, TaskKind},
    diagnostics,
    infer::TranscriptionOptions,
    process::hidden_command,
    provider::{CrispAsr, OpenVino, Transcriber},
    resources,
    sdk::ModelSdk,
    subtitle::Segment,
    task::TaskControl,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Family {
    CrispAsr,
    OpenVino,
    Llama,
}

pub(crate) struct Adapter {
    pub id: &'static str,
    pub family: Family,
    pub variant: &'static str,
    pub task: TaskKind,
}

pub(crate) const ADAPTERS: &[Adapter] = &[
    Adapter {
        id: "crispasr-vulkan",
        family: Family::CrispAsr,
        variant: "vulkan",
        task: TaskKind::Transcription,
    },
    Adapter {
        id: "crispasr-cuda12",
        family: Family::CrispAsr,
        variant: "cuda12",
        task: TaskKind::Transcription,
    },
    Adapter {
        id: "crispasr-cuda13",
        family: Family::CrispAsr,
        variant: "cuda13",
        task: TaskKind::Transcription,
    },
    Adapter {
        id: "crispasr-cpu",
        family: Family::CrispAsr,
        variant: "cpu",
        task: TaskKind::Transcription,
    },
    Adapter {
        id: "crispasr-cpu-legacy",
        family: Family::CrispAsr,
        variant: "cpu-legacy",
        task: TaskKind::Transcription,
    },
    Adapter {
        id: resources::OPENVINO_ENGINE_ID,
        family: Family::OpenVino,
        variant: "cpu",
        task: TaskKind::Transcription,
    },
    Adapter {
        id: "llama-cpu",
        family: Family::Llama,
        variant: "cpu",
        task: TaskKind::TextGeneration,
    },
    Adapter {
        id: "llama-vulkan",
        family: Family::Llama,
        variant: "vulkan",
        task: TaskKind::TextGeneration,
    },
    Adapter {
        id: "llama-cuda12",
        family: Family::Llama,
        variant: "cuda12",
        task: TaskKind::TextGeneration,
    },
    Adapter {
        id: "llama-cuda13",
        family: Family::Llama,
        variant: "cuda13",
        task: TaskKind::TextGeneration,
    },
];

pub(crate) struct AudioRequest<'a> {
    pub sdk: &'a ModelSdk,
    pub engine: &'a Path,
    pub model: &'a Path,
    pub wav: &'a Path,
    pub output_base: &'a Path,
    pub profile: ModelProfile,
    pub format: ArtifactFormat,
    pub language: Option<&'a str>,
    pub openvino_language: Option<&'a str>,
    pub vad_enabled: bool,
    pub options: &'a TranscriptionOptions,
    pub task: &'a TaskControl,
    pub run: &'a dyn Fn(Command) -> Result<ExitStatus, String>,
}

pub(crate) fn get(id: &str) -> Result<&'static Adapter, String> {
    ADAPTERS
        .iter()
        .find(|item| item.id == id)
        .ok_or_else(|| format!("未知引擎: {id}"))
}

impl Adapter {
    pub fn supports(&self, profile: ModelProfile, format: ArtifactFormat, task: TaskKind) -> bool {
        self.task == task
            && match self.family {
                Family::CrispAsr => crispasr::supports(profile, format),
                Family::OpenVino => openvino::supports(profile, format),
                Family::Llama => llama::supports(profile, format),
            }
    }

    pub fn managed_path(&self, root: &Path) -> PathBuf {
        match self.family {
            Family::CrispAsr => crispasr::managed_path(root, self.id),
            Family::OpenVino => openvino::managed_path(root),
            Family::Llama => llama::managed_path(root, self.id),
        }
    }

    pub fn suitability(
        &self,
        system: &diagnostics::SystemCheck,
        path: Option<&Path>,
    ) -> Result<(bool, Option<String>), String> {
        Ok(match self.family {
            Family::CrispAsr => crispasr::suitability(self.id, system, path)?,
            Family::OpenVino => openvino::suitability(system, path),
            Family::Llama => llama::suitability(self.variant, system),
        })
    }

    pub fn verify_local(&self, path: &Path) -> Result<(), String> {
        let expected = match self.family {
            Family::CrispAsr => "crispasr.exe",
            Family::OpenVino => {
                if cfg!(windows) {
                    "python.exe"
                } else {
                    "python3"
                }
            }
            Family::Llama => "llama-server.exe",
        };
        let valid_name = path.file_name().is_some_and(|name| {
            name.to_string_lossy().eq_ignore_ascii_case(expected)
                || (self.family == Family::OpenVino && !cfg!(windows) && name == "python")
        });
        if !path.is_absolute() || !path.is_file() || !valid_name {
            return Err(format!("请选择有效的 {expected} 完整路径。"));
        }
        let system = diagnostics::system_check(&resources::Selection::default());
        let (suitable, reason) = self.suitability(&system, Some(path))?;
        if !suitable {
            return Err(reason.unwrap_or_else(|| "此引擎不适合当前设备。".into()));
        }
        self.probe(path, None)
    }

    pub fn probe(&self, path: &Path, model: Option<(&Path, ModelProfile)>) -> Result<(), String> {
        match self.family {
            Family::CrispAsr => {
                let spec = resources::engine_spec(self.id)?;
                let system = diagnostics::system_check(&resources::Selection::default());
                diagnostics::probe_crispasr(spec, path, &system, model)
            }
            Family::OpenVino => {
                if hidden_command(path)
                    .args(["-c", "import openvino, onnxruntime, numpy"])
                    .output()
                    .is_ok_and(|output| output.status.success())
                {
                    Ok(())
                } else {
                    Err("OpenVINO Python 环境无法运行或依赖缺失。".into())
                }
            }
            Family::Llama => {
                if hidden_command(path)
                    .arg("--version")
                    .output()
                    .is_ok_and(|output| output.status.success())
                {
                    Ok(())
                } else {
                    Err("llama.cpp 运行时无法启动。".into())
                }
            }
        }
    }

    pub fn transcribe(&self, request: &AudioRequest<'_>) -> Result<Vec<Segment>, String> {
        match self.family {
            Family::CrispAsr => crispasr::transcribe(self.id, request),
            Family::OpenVino => openvino::transcribe(request),
            Family::Llama => Err("文本生成引擎不能执行语音转写。".into()),
        }
    }
}

mod crispasr {
    use super::*;

    pub fn supports(profile: ModelProfile, format: ArtifactFormat) -> bool {
        matches!(
            (profile, format),
            (ModelProfile::Whisper, ArtifactFormat::Ggml)
                | (ModelProfile::Qwen3Asr17b, ArtifactFormat::Gguf)
        )
    }

    pub fn managed_path(root: &Path, id: &str) -> PathBuf {
        root.join("engines/crispasr").join(id).join("crispasr.exe")
    }

    pub fn suitability(
        id: &str,
        system: &diagnostics::SystemCheck,
        path: Option<&Path>,
    ) -> Result<(bool, Option<String>), String> {
        let spec = resources::engine_spec(id)?;
        let (mut suitable, mut reason) = diagnostics::suitability(spec, system);
        if suitable {
            if let Some(path) = path {
                if let Err(error) = diagnostics::probe_crispasr(spec, path, system, None) {
                    suitable = false;
                    reason = Some(error);
                }
            }
        }
        Ok((suitable, reason))
    }

    pub fn transcribe(id: &str, request: &AudioRequest<'_>) -> Result<Vec<Segment>, String> {
        let backend = resources::engine_spec(id)?.gpu_backend;
        let aligner = (request.format == ArtifactFormat::Gguf && request.options.align_timestamps)
            .then(|| request.sdk.ensure_aligner_model(request.task))
            .transpose()?;
        let transcriber = CrispAsr::new(
            request.engine,
            request.model,
            request.output_base,
            backend,
            request.profile,
            request.run,
        )
        .with_language(request.language);
        if let Some(ref aligner) = aligner {
            transcriber.with_aligner(aligner).transcribe(request.wav)
        } else {
            transcriber.transcribe(request.wav)
        }
    }
}

mod openvino {
    use super::*;

    pub fn supports(profile: ModelProfile, format: ArtifactFormat) -> bool {
        profile == ModelProfile::Qwen3Asr06b && format == ArtifactFormat::OpenVinoIr
    }

    pub fn managed_path(root: &Path) -> PathBuf {
        if cfg!(windows) {
            root.join("engines/openvino/venv/Scripts/python.exe")
        } else {
            root.join("engines/openvino/venv/bin/python3")
        }
    }

    pub fn suitability(
        system: &diagnostics::SystemCheck,
        path: Option<&Path>,
    ) -> (bool, Option<String>) {
        let suitable =
            (system.platform == "windows" && system.architecture == "x86_64") || path.is_some();
        (
            suitable,
            (!suitable).then_some("当前自动安装仅支持 Windows x86-64。".into()),
        )
    }

    pub fn transcribe(request: &AudioRequest<'_>) -> Result<Vec<Segment>, String> {
        let worker = request.sdk.ensure_openvino_worker()?;
        let vad_model = if request.vad_enabled {
            request.sdk.ensure_vad_model(request.task)?
        } else {
            worker.join("silero_vad_v4.onnx")
        };
        OpenVino {
            python: request.engine,
            worker: &worker.join("worker.py"),
            model: request.model,
            vad_model: &vad_model,
            vad_enabled: request.vad_enabled,
            language: request.openvino_language,
            output_base: request.output_base,
            run: request.run,
        }
        .transcribe(request.wav)
    }
}

mod llama {
    use super::*;

    pub fn supports(profile: ModelProfile, format: ArtifactFormat) -> bool {
        profile == ModelProfile::HyMt15 && format == ArtifactFormat::Gguf
    }

    pub fn managed_path(root: &Path, id: &str) -> PathBuf {
        root.join("engines").join(id).join("llama-server.exe")
    }

    pub fn suitability(variant: &str, system: &diagnostics::SystemCheck) -> (bool, Option<String>) {
        if system.platform != "windows" || system.architecture != "x86_64" {
            return (
                false,
                Some("llama.cpp 接入目前仅支持 Windows x86-64。".into()),
            );
        }
        let gpu = system
            .gpus
            .iter()
            .any(|item| ["nvidia", "amd", "intel"].contains(&item.vendor.as_str()));
        let nvidia = system.gpus.iter().any(|item| item.vendor == "nvidia");
        let driver = system
            .nvidia_driver
            .as_deref()
            .and_then(diagnostics::driver_major)
            .unwrap_or(0);
        let compute = system
            .nvidia_compute_capability
            .as_deref()
            .and_then(diagnostics::compute_capability)
            .unwrap_or(0);
        let reason = match variant {
            "vulkan" if !gpu => Some("未检测到可用显卡。".into()),
            "cuda12" | "cuda13" if !nvidia => Some("需要 NVIDIA 显卡。".into()),
            "cuda12" if driver < 551 || compute < 50 => {
                Some("需要 551 以上驱动和计算能力 5.0 以上的 NVIDIA 显卡。".into())
            }
            "cuda13" if driver < 580 || compute < 75 => {
                Some("需要 580 以上驱动和计算能力 7.5 以上的 NVIDIA 显卡。".into())
            }
            _ => None,
        };
        (reason.is_none(), reason)
    }
}
