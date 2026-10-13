use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
};

use crate::{
    process::hidden_command,
    subtitle::{parse_srt, split_aligned_srt, Segment},
};

pub trait Transcriber {
    fn transcribe(&self, audio: &Path) -> Result<Vec<Segment>, String>;
}

pub struct OpenVino<'a> {
    pub python: &'a Path,
    pub worker: &'a Path,
    pub model: &'a Path,
    pub vad_model: &'a Path,
    pub vad_enabled: bool,
    pub language: Option<&'a str>,
    pub output_base: &'a Path,
    pub run: &'a dyn Fn(Command) -> Result<ExitStatus, String>,
}

impl Transcriber for OpenVino<'_> {
    fn transcribe(&self, audio: &Path) -> Result<Vec<Segment>, String> {
        let output = self.output_base.with_extension("json");
        let log_path = self.output_base.with_extension("log");
        let log = File::create(&log_path).map_err(|error| error.to_string())?;
        let mut command = hidden_command(self.python);
        command
            .arg(self.worker)
            .arg("--audio")
            .arg(audio)
            .arg("--model")
            .arg(self.model)
            .arg("--vad-model")
            .arg(self.vad_model)
            .arg("--vad")
            .arg(if self.vad_enabled { "on" } else { "off" })
            .arg("--output")
            .arg(&output);
        if let Some(language) = self.language {
            command.arg("--language").arg(language);
        }
        command.stdout(Stdio::null()).stderr(Stdio::from(log));
        let status = (self.run)(command)?;
        if !status.success() {
            let detail = fs::read_to_string(&log_path).unwrap_or_default();
            return Err(format!(
                "OpenVINO 推理失败 ({status}): {}",
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
        let segments: Vec<Segment> =
            serde_json::from_slice(&fs::read(output).map_err(|error| error.to_string())?)
                .map_err(|error| format!("OpenVINO 输出无效: {error}"))?;
        if segments.is_empty() {
            return Err("未检测到可转写语音。".into());
        }
        Ok(segments)
    }
}

pub struct CrispAsr<'a> {
    engine: &'a Path,
    model: &'a Path,
    output_base: &'a Path,
    gpu_backend: &'a str,
    aligner: Option<&'a Path>,
    language: Option<&'a str>,
    run: &'a dyn Fn(Command) -> Result<ExitStatus, String>,
}

impl<'a> CrispAsr<'a> {
    pub fn new(
        engine: &'a Path,
        model: &'a Path,
        output_base: &'a Path,
        gpu_backend: &'a str,
        run: &'a dyn Fn(Command) -> Result<ExitStatus, String>,
    ) -> Self {
        Self {
            engine,
            model,
            output_base,
            gpu_backend,
            aligner: None,
            language: None,
            run,
        }
    }

    pub fn with_aligner(mut self, aligner: &'a Path) -> Self {
        self.aligner = Some(aligner);
        self
    }

    pub fn with_language(mut self, language: Option<&'a str>) -> Self {
        self.language = language;
        self
    }

    fn model_backend(&self) -> Option<&'static str> {
        if self
            .model
            .extension()
            .is_some_and(|extension| extension == "gguf")
        {
            let name = self.model.file_name()?.to_str()?.to_ascii_lowercase();
            return Some(if name.contains("1.7b") {
                "qwen3-1.7b"
            } else {
                "qwen3"
            });
        }
        None
    }
}

impl Transcriber for CrispAsr<'_> {
    fn transcribe(&self, audio: &Path) -> Result<Vec<Segment>, String> {
        let mut command = hidden_command(self.engine);
        command.arg("-m").arg(self.model);
        if let Some(language) = self.language {
            command.arg("-l").arg(language);
        }
        if let Some(backend) = self.model_backend() {
            command.args(["--backend", backend]);
        }
        match self.gpu_backend {
            "vulkan" | "cuda" => {
                command.args(["--gpu-backend", self.gpu_backend]);
            }
            "cuda12" | "cuda13" => {
                command.args(["--gpu-backend", "cuda"]);
            }
            "cpu" | "cpu-legacy" => {
                command.arg("-ng");
            }
            other => return Err(format!("不支持的推理加速方式: {other}")),
        }
        let log_path = self.output_base.with_extension("log");
        let log = File::create(&log_path).map_err(|error| error.to_string())?;
        command
            .arg("-of")
            .arg(self.output_base)
            .args(["-np", "-osrt", "-fa", "-nf"]);
        if let Some(aligner) = self.aligner {
            command.args(["-ml", "1"]);
            command.arg("-am").arg(aligner).arg("-falign");
        }
        command
            .arg(audio)
            .stdout(Stdio::null())
            .stderr(Stdio::from(log));
        let status = (self.run)(command)?;
        if !status.success() {
            let detail = fs::read_to_string(log_path).unwrap_or_default();
            let tail: String = detail
                .chars()
                .rev()
                .take(500)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            return Err(format!("推理引擎退出 ({status}): {tail}"));
        }
        let srt_path = PathBuf::from(format!("{}.srt", self.output_base.display()));
        let bytes = fs::read(&srt_path)
            .map_err(|error| format!("读取引擎输出失败 ({}): {error}", srt_path.display()))?;
        let content =
            String::from_utf8(bytes).map_err(|_| "引擎输出不是 UTF-8 字幕。".to_string())?;
        if self.aligner.is_some() {
            split_aligned_srt(&content)
        } else {
            parse_srt(&content)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crispasr_uses_qwen_backend_for_gguf_models() {
        let run = |_| unreachable!();
        let engine = Path::new("crispasr.exe");
        let output = Path::new("out");
        let qwen = CrispAsr::new(
            engine,
            Path::new("qwen3-asr-1.7b-q4_k.gguf"),
            output,
            "cpu",
            &run,
        );
        assert_eq!(qwen.model_backend(), Some("qwen3-1.7b"));
        let whisper = CrispAsr::new(engine, Path::new("ggml-base.bin"), output, "cpu", &run);
        assert_eq!(whisper.model_backend(), None);
    }
}
