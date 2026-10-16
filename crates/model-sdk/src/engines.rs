use std::{
    path::{Path, PathBuf},
    process::{Command, ExitStatus},
};

use crate::{
    catalog::{ArtifactFormat, ModelProfile, TaskKind},
    detect::{DetectionOptions, DetectionResult},
    diagnostics,
    infer::TranscriptionOptions,
    process::hidden_command,
    provider::{CrispAsr, OpenVino, Transcriber},
    resources,
    sdk::ModelSdk,
    subtitle::Segment,
    tag::TagResult,
    task::TaskControl,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Family {
    CrispAsr,
    OpenVino,
    Llama,
    OnnxRuntime,
}

pub(crate) struct Adapter {
    pub id: &'static str,
    pub family: Family,
    pub variant: &'static str,
    pub tasks: &'static [TaskKind],
}

pub(crate) const ADAPTERS: &[Adapter] = &[
    Adapter {
        id: "crispasr-vulkan",
        family: Family::CrispAsr,
        variant: "vulkan",
        tasks: &[TaskKind::Transcription],
    },
    Adapter {
        id: "crispasr-cuda12",
        family: Family::CrispAsr,
        variant: "cuda12",
        tasks: &[TaskKind::Transcription],
    },
    Adapter {
        id: "crispasr-cuda13",
        family: Family::CrispAsr,
        variant: "cuda13",
        tasks: &[TaskKind::Transcription],
    },
    Adapter {
        id: "crispasr-cpu",
        family: Family::CrispAsr,
        variant: "cpu",
        tasks: &[TaskKind::Transcription],
    },
    Adapter {
        id: "crispasr-cpu-legacy",
        family: Family::CrispAsr,
        variant: "cpu-legacy",
        tasks: &[TaskKind::Transcription],
    },
    Adapter {
        id: resources::OPENVINO_ENGINE_ID,
        family: Family::OpenVino,
        variant: "cpu",
        tasks: &[TaskKind::Transcription],
    },
    Adapter {
        id: "llama-cpu",
        family: Family::Llama,
        variant: "cpu",
        tasks: &[TaskKind::TextGeneration],
    },
    Adapter {
        id: "llama-vulkan",
        family: Family::Llama,
        variant: "vulkan",
        tasks: &[TaskKind::TextGeneration],
    },
    Adapter {
        id: "llama-cuda12",
        family: Family::Llama,
        variant: "cuda12",
        tasks: &[TaskKind::TextGeneration],
    },
    Adapter {
        id: "llama-cuda13",
        family: Family::Llama,
        variant: "cuda13",
        tasks: &[TaskKind::TextGeneration],
    },
    Adapter {
        id: "onnxruntime-cpu",
        family: Family::OnnxRuntime,
        variant: "cpu",
        tasks: &[
            TaskKind::ObjectDetection,
            TaskKind::ImageTagging,
            TaskKind::SpeechSynthesis,
        ],
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
        self.tasks.contains(&task)
            && match self.family {
                Family::CrispAsr => crispasr::supports(profile, format),
                Family::OpenVino => openvino::supports(profile, format),
                Family::Llama => llama::supports(profile, format),
                Family::OnnxRuntime => onnxruntime::supports(profile, format, task),
            }
    }

    pub fn managed_path(&self, root: &Path) -> PathBuf {
        match self.family {
            Family::CrispAsr => crispasr::managed_path(root, self.id),
            Family::OpenVino => openvino::managed_path(root),
            Family::Llama => llama::managed_path(root, self.id),
            Family::OnnxRuntime => onnxruntime::managed_path(root),
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
            Family::OnnxRuntime => onnxruntime::suitability(system, path),
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
            Family::OnnxRuntime => {
                if cfg!(windows) {
                    "python.exe"
                } else {
                    "python3"
                }
            }
        };
        let valid_name = path.file_name().is_some_and(|name| {
            name.to_string_lossy().eq_ignore_ascii_case(expected)
                || (matches!(self.family, Family::OpenVino | Family::OnnxRuntime)
                    && !cfg!(windows)
                    && name == "python")
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
            Family::OnnxRuntime => onnxruntime::probe(path, model),
        }
    }

    pub fn transcribe(&self, request: &AudioRequest<'_>) -> Result<Vec<Segment>, String> {
        match self.family {
            Family::CrispAsr => crispasr::transcribe(self.id, request),
            Family::OpenVino => openvino::transcribe(request),
            Family::Llama => Err("文本生成引擎不能执行语音转写。".into()),
            Family::OnnxRuntime => Err("视觉检测引擎不能执行语音转写。".into()),
        }
    }

    pub fn detect(
        &self,
        sdk: &ModelSdk,
        engine: &Path,
        model: &Path,
        image: &Path,
        options: &DetectionOptions,
        task: &TaskControl,
    ) -> Result<DetectionResult, String> {
        match self.family {
            Family::OnnxRuntime => onnxruntime::detect(sdk, engine, model, image, options, task),
            _ => Err("所选引擎不支持目标检测。".into()),
        }
    }

    pub fn tag(
        &self,
        sdk: &ModelSdk,
        engine: &Path,
        model: &Path,
        image: &Path,
        task: &TaskControl,
    ) -> Result<TagResult, String> {
        match self.family {
            Family::OnnxRuntime => onnxruntime::tag(sdk, engine, model, image, task),
            _ => Err("所选引擎不支持图片打标。".into()),
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

pub(crate) mod onnxruntime {
    use super::*;
    use std::{
        fs::{self, File},
        process::Stdio,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(0);

    pub fn supports(profile: ModelProfile, format: ArtifactFormat, task: TaskKind) -> bool {
        format == ArtifactFormat::Onnx
            && matches!(
                (profile, task),
                (
                    ModelProfile::YoloV8Detect | ModelProfile::Yolo11Detect,
                    TaskKind::ObjectDetection
                ) | (
                    ModelProfile::Wd14V2 | ModelProfile::Wd14V3,
                    TaskKind::ImageTagging
                ) | (ModelProfile::KokoroZh, TaskKind::SpeechSynthesis)
            )
    }

    pub fn managed_path(root: &Path) -> PathBuf {
        if cfg!(windows) {
            root.join("engines/onnxruntime-cpu/venv/Scripts/python.exe")
        } else {
            root.join("engines/onnxruntime-cpu/venv/bin/python3")
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
            (!suitable).then_some("ONNX Runtime 自动安装目前仅支持 Windows x86-64。".into()),
        )
    }

    pub fn probe(python: &Path, model: Option<(&Path, ModelProfile)>) -> Result<(), String> {
        let mut command = hidden_command(python);
        if let Some((model, profile)) = model {
            if profile == ModelProfile::KokoroZh {
                command.args(["-c", "import misaki.zh; import json,onnxruntime as o,sys; from pathlib import Path; p=Path(sys.argv[1]); s=o.InferenceSession(str(p/'model.onnx'),providers=['CPUExecutionProvider']); assert {i.name for i in s.get_inputs()} == {'input_ids','style','speed'}; assert len(s.get_outputs()) >= 1; assert json.loads((p/'tokenizer.json').read_text(encoding='utf-8'))['model']['vocab']['$'] == 0"]).arg(model);
            } else if matches!(profile, ModelProfile::Wd14V2 | ModelProfile::Wd14V3) {
                command.args(["-c", "import csv,onnxruntime as o,sys; from pathlib import Path; p=Path(sys.argv[1]); s=o.InferenceSession(str(p/'model.onnx'),providers=['CPUExecutionProvider']); i=s.get_inputs()[0].shape; rows=list(csv.DictReader((p/'selected_tags.csv').open(encoding='utf-8'))); assert len(i)==4 and i[1]==i[2] and i[3]==3; assert len(s.get_outputs())==1 and s.get_outputs()[0].shape[-1]==len(rows); assert rows and all(k in rows[0] for k in ('name','category'))"]).arg(model);
            } else {
                command.args(["-c", "import json,onnxruntime as o,sys; s=o.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider']); assert s.get_inputs()[0].shape == [1,3,640,640]; assert s.get_outputs()[0].shape == [1,84,8400]; assert len(json.loads(s.get_modelmeta().custom_metadata_map['labels'])) == 80"]).arg(model);
            }
        } else {
            command.args(["-c", "import onnxruntime, numpy, PIL"]);
        }
        if let Ok(output) = command.output() {
            if output.status.success() {
                return Ok(());
            }
            if model.is_some_and(|(_, profile)| profile == ModelProfile::KokoroZh)
                && String::from_utf8_lossy(&output.stderr).contains("ModuleNotFoundError")
            {
                return Err(
                    "中文语音依赖缺失，请在资源与兼容性中重新安装 ONNX Runtime 引擎。".into(),
                );
            }
        }
        Err("ONNX Runtime 环境或模型输出协议无法运行。".into())
    }

    pub fn synthesize(
        sdk: &ModelSdk,
        python: &Path,
        model: &Path,
        text: &str,
        speed: f32,
        output: &Path,
        task: &TaskControl,
    ) -> Result<(), String> {
        let worker_dir = sdk.root().join("workers/onnxruntime");
        fs::create_dir_all(&worker_dir).map_err(|error| error.to_string())?;
        let worker = worker_dir.join("synthesize_worker.py");
        let contents = include_bytes!("../workers/onnxruntime/synthesize_worker.py");
        if !fs::read(&worker).is_ok_and(|current| current == contents) {
            fs::write(&worker, contents).map_err(|error| error.to_string())?;
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let job = sdk.root().join("jobs").join(format!(
            "synthesize-{}-{nonce}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&job).map_err(|error| error.to_string())?;
        let result = (|| {
            let input = job.join("text.txt");
            fs::write(&input, text).map_err(|error| error.to_string())?;
            let wav = job.join("speech.wav");
            let log = job.join("worker.log");
            let mut command = hidden_command(python);
            command
                .arg(&worker)
                .arg("--model")
                .arg(model)
                .arg("--text-file")
                .arg(&input)
                .arg("--output")
                .arg(&wav)
                .arg("--speed")
                .arg(speed.to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::from(
                    File::create(&log).map_err(|error| error.to_string())?,
                ));
            task.progress("synthesize", 0, "正在合成语音");
            let status = task.run(command)?;
            if !status.success() {
                let detail = fs::read_to_string(log).unwrap_or_default();
                return Err(format!(
                    "语音合成失败 ({status}): {}",
                    detail
                        .chars()
                        .rev()
                        .take(1000)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect::<String>()
                ));
            }
            task.check_cancelled()?;
            let mut source = File::open(wav).map_err(|error| error.to_string())?;
            let mut target = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)
                .map_err(|error| format!("无法创建输出文件 {}: {error}", output.display()))?;
            if let Err(error) = std::io::copy(&mut source, &mut target) {
                drop(target);
                let _ = fs::remove_file(output);
                return Err(error.to_string());
            }
            if let Err(error) = task.check_cancelled() {
                drop(target);
                let _ = fs::remove_file(output);
                return Err(error);
            }
            task.progress("synthesize", 100, "语音合成完成");
            Ok(())
        })();
        let _ = fs::remove_dir_all(job);
        result
    }

    pub fn detect(
        sdk: &ModelSdk,
        python: &Path,
        model: &Path,
        image: &Path,
        options: &DetectionOptions,
        task: &TaskControl,
    ) -> Result<DetectionResult, String> {
        let worker_dir = sdk.root().join("workers/onnxruntime");
        fs::create_dir_all(&worker_dir).map_err(|error| error.to_string())?;
        let worker = worker_dir.join("worker.py");
        let contents = include_bytes!("../workers/onnxruntime/worker.py");
        if !fs::read(&worker).is_ok_and(|current| current == contents) {
            fs::write(&worker, contents).map_err(|error| error.to_string())?;
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let job = sdk.root().join("jobs").join(format!(
            "detect-{}-{nonce}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&job).map_err(|error| error.to_string())?;
        let output = job.join("result.json");
        let log = job.join("worker.log");
        let result = (|| {
            let mut command = hidden_command(python);
            command
                .arg(&worker)
                .arg("--model")
                .arg(model)
                .arg("--image")
                .arg(image)
                .arg("--output")
                .arg(&output)
                .arg("--confidence")
                .arg(options.confidence_threshold.to_string())
                .arg("--iou")
                .arg(options.iou_threshold.to_string())
                .arg("--max-detections")
                .arg(options.max_detections.to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::from(
                    File::create(&log).map_err(|error| error.to_string())?,
                ));
            task.progress("detect", 0, "正在检测图片");
            let status = task.run(command)?;
            if !status.success() {
                let detail = fs::read_to_string(&log).unwrap_or_default();
                return Err(format!(
                    "目标检测失败 ({status}): {}",
                    detail
                        .chars()
                        .rev()
                        .take(1000)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect::<String>()
                ));
            }
            task.check_cancelled()?;
            let result =
                serde_json::from_slice(&fs::read(output).map_err(|error| error.to_string())?)
                    .map_err(|error| format!("检测结果无效: {error}"))?;
            task.progress("detect", 100, "图片检测完成");
            Ok(result)
        })();
        let _ = fs::remove_dir_all(job);
        result
    }

    pub fn tag(
        sdk: &ModelSdk,
        python: &Path,
        model: &Path,
        image: &Path,
        task: &TaskControl,
    ) -> Result<TagResult, String> {
        let worker_dir = sdk.root().join("workers/onnxruntime");
        fs::create_dir_all(&worker_dir).map_err(|error| error.to_string())?;
        let worker = worker_dir.join("tag_worker.py");
        let contents = include_bytes!("../workers/onnxruntime/tag_worker.py");
        if !fs::read(&worker).is_ok_and(|current| current == contents) {
            fs::write(&worker, contents).map_err(|error| error.to_string())?;
        }
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let job = sdk.root().join("jobs").join(format!(
            "tag-{}-{nonce}-{}",
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&job).map_err(|error| error.to_string())?;
        let output = job.join("result.json");
        let log = job.join("worker.log");
        let result = (|| {
            let mut command = hidden_command(python);
            command
                .arg(&worker)
                .arg("--model")
                .arg(model)
                .arg("--image")
                .arg(image)
                .arg("--output")
                .arg(&output)
                .stdout(Stdio::null())
                .stderr(Stdio::from(
                    File::create(&log).map_err(|error| error.to_string())?,
                ));
            task.progress("tag", 0, "正在分析图片标签");
            let status = task.run(command)?;
            if !status.success() {
                let detail = fs::read_to_string(&log).unwrap_or_default();
                return Err(format!(
                    "图片打标失败 ({status}): {}",
                    detail
                        .chars()
                        .rev()
                        .take(1000)
                        .collect::<String>()
                        .chars()
                        .rev()
                        .collect::<String>()
                ));
            }
            task.check_cancelled()?;
            let result =
                serde_json::from_slice(&fs::read(output).map_err(|error| error.to_string())?)
                    .map_err(|error| format!("打标结果无效: {error}"))?;
            task.progress("tag", 100, "图片打标完成");
            Ok(result)
        })();
        let _ = fs::remove_dir_all(job);
        result
    }
}
