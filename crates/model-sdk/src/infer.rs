use std::{
    fs::{self, File},
    path::Path,
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    process::hidden_command,
    provider::{CrispAsr, OpenVino, Transcriber},
    resources,
    sdk::{ModelSdk, TRANSCRIPTION},
    subtitle::whole_transcript,
    subtitle::Segment,
    task::TaskControl,
};

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct TranscriptionOptions {
    pub align_timestamps: bool,
    pub language: Option<String>,
}

impl Default for TranscriptionOptions {
    fn default() -> Self {
        Self {
            align_timestamps: true,
            language: None,
        }
    }
}

fn language_name(code: Option<&str>) -> Result<Option<&'static str>, String> {
    Ok(match code {
        None => None,
        Some("zh") => Some("Chinese"),
        Some("en") => Some("English"),
        Some("ja") => Some("Japanese"),
        Some("ko") => Some("Korean"),
        Some("yue") => Some("Cantonese"),
        Some("fr") => Some("French"),
        Some("de") => Some("German"),
        Some("es") => Some("Spanish"),
        Some("pt") => Some("Portuguese"),
        Some("ru") => Some("Russian"),
        Some("ar") => Some("Arabic"),
        Some("th") => Some("Thai"),
        Some("vi") => Some("Vietnamese"),
        Some("id") => Some("Indonesian"),
        Some("ms") => Some("Malay"),
        Some(other) => return Err(format!("不支持的识别语言: {other}")),
    })
}

impl ModelSdk {
    pub fn transcribe(
        &self,
        engine_id: &str,
        model_id: &str,
        media: &Path,
        ffmpeg_path: &Path,
        vad_enabled: bool,
        task: &TaskControl,
    ) -> Result<Vec<Segment>, String> {
        self.transcribe_with_options(
            engine_id,
            model_id,
            media,
            ffmpeg_path,
            vad_enabled,
            &TranscriptionOptions::default(),
            task,
        )
    }

    pub fn transcribe_with_options(
        &self,
        engine_id: &str,
        model_id: &str,
        media: &Path,
        ffmpeg_path: &Path,
        vad_enabled: bool,
        options: &TranscriptionOptions,
        task: &TaskControl,
    ) -> Result<Vec<Segment>, String> {
        if !media.is_file() {
            return Err(format!("找不到文件: {}", media.display()));
        }
        let ready = self.check(engine_id, model_id, TRANSCRIPTION)?;
        if !ready.ready {
            return Err(ready.reason.unwrap_or_else(|| "所选资源尚未就绪。".into()));
        }
        let engine = self.engine_path(engine_id)?.ok_or("所选推理引擎不存在。")?;
        let model = self.model_path(model_id)?.ok_or("所选模型不存在。")?;
        let backend = if engine_id == resources::OPENVINO_ENGINE_ID {
            "cpu"
        } else {
            resources::engine_spec(engine_id)?.gpu_backend
        };
        self.transcribe_paths_with_options(
            media,
            &engine,
            &model,
            ffmpeg_path,
            backend,
            engine_id == resources::OPENVINO_ENGINE_ID,
            vad_enabled,
            options,
            task,
        )
    }

    pub fn transcribe_paths(
        &self,
        media: &Path,
        engine: &Path,
        model: &Path,
        ffmpeg_path: &Path,
        gpu_backend: &str,
        openvino: bool,
        vad_enabled: bool,
        task: &TaskControl,
    ) -> Result<Vec<Segment>, String> {
        self.transcribe_paths_with_options(
            media,
            engine,
            model,
            ffmpeg_path,
            gpu_backend,
            openvino,
            vad_enabled,
            &TranscriptionOptions::default(),
            task,
        )
    }

    pub fn transcribe_paths_with_options(
        &self,
        media: &Path,
        engine: &Path,
        model: &Path,
        ffmpeg_path: &Path,
        gpu_backend: &str,
        openvino: bool,
        vad_enabled: bool,
        options: &TranscriptionOptions,
        task: &TaskControl,
    ) -> Result<Vec<Segment>, String> {
        let openvino_language = language_name(options.language.as_deref())?;
        if !media.is_file() {
            return Err(format!("找不到文件: {}", media.display()));
        }
        if !engine.is_file() {
            return Err(format!("找不到文件: {}", engine.display()));
        }
        if (openvino && !model.is_dir()) || (!openvino && !model.is_file()) {
            return Err(format!("找不到模型: {}", model.display()));
        }
        let jobs_dir = self.root().join("jobs");
        fs::create_dir_all(&jobs_dir).map_err(|error| error.to_string())?;
        let job_id = format!(
            "job-{}-{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos(),
            std::process::id(),
            NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed)
        );
        let job_dir = jobs_dir.join(job_id);
        fs::create_dir(&job_dir).map_err(|error| error.to_string())?;
        let wav = job_dir.join("audio.wav");
        let output_base = job_dir.join("result");
        task.progress("prepare", 0, "正在提取音频");
        let ffmpeg_log = job_dir.join("ffmpeg.log");
        let mut ffmpeg = hidden_command(ffmpeg_path);
        ffmpeg
            .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
            .arg(media)
            .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
            .arg(&wav)
            .stdout(Stdio::null())
            .stderr(Stdio::from(
                File::create(&ffmpeg_log).map_err(|error| error.to_string())?,
            ));
        let result = (|| {
            let status = task.run(ffmpeg).map_err(|error| {
                if error.contains("No such file") || error.contains("找不到") {
                    "找不到 FFmpeg，请先安装或加入 PATH。".to_string()
                } else {
                    error
                }
            })?;
            if !status.success() {
                return Err(format!(
                    "音频提取失败: {}",
                    fs::read_to_string(&ffmpeg_log).unwrap_or_default()
                ));
            }
            task.progress("transcribe", 15, "正在转写音频");
            let runner = |command| task.run(command);
            let segments = if openvino {
                let worker = self.ensure_openvino_worker()?;
                let vad_model = if vad_enabled {
                    self.ensure_vad_model(task)?
                } else {
                    worker.join("silero_vad_v4.onnx")
                };
                OpenVino {
                    python: &engine,
                    worker: &worker.join("worker.py"),
                    model: &model,
                    vad_model: &vad_model,
                    vad_enabled,
                    language: openvino_language,
                    output_base: &output_base,
                    run: &runner,
                }
                .transcribe(&wav)?
            } else {
                let aligner = model
                    .extension()
                    .is_some_and(|extension| extension == "gguf" && options.align_timestamps)
                    .then(|| self.ensure_aligner_model(task))
                    .transpose()?;
                let transcriber = CrispAsr::new(engine, model, &output_base, gpu_backend, &runner)
                    .with_language(options.language.as_deref());
                if let Some(ref aligner) = aligner {
                    transcriber.with_aligner(aligner).transcribe(&wav)?
                } else {
                    transcriber.transcribe(&wav)?
                }
            };
            let segments = if options.align_timestamps {
                segments
            } else {
                whole_transcript(segments)?
            };
            task.check_cancelled()?;
            task.progress("transcribe", 100, "转写完成");
            Ok(segments)
        })();
        let _ = fs::remove_file(wav);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_codes_are_validated_before_inference() {
        assert_eq!(language_name(None).unwrap(), None);
        assert_eq!(language_name(Some("zh")).unwrap(), Some("Chinese"));
        assert_eq!(language_name(Some("yue")).unwrap(), Some("Cantonese"));
        assert!(language_name(Some("invalid")).is_err());
    }

    #[test]
    fn concurrent_transcriptions_keep_job_files_separate() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("model-sdk-jobs-{}-{nonce}", std::process::id()));
        let sdk = ModelSdk::new(&root).unwrap();
        let media = root.join("media.wav");
        let engine = root.join("engine.exe");
        let model = root.join("model.bin");
        for path in [&media, &engine, &model] {
            fs::write(path, b"test").unwrap();
        }
        std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        assert!(sdk
                            .transcribe_paths_with_options(
                                &media,
                                &engine,
                                &model,
                                &root.join("missing-ffmpeg"),
                                "cpu",
                                false,
                                false,
                                &TranscriptionOptions::default(),
                                &TaskControl::default(),
                            )
                            .is_err());
                    })
                })
                .collect();
            for job in jobs {
                job.join().unwrap();
            }
        });
        let job_dirs: Vec<_> = fs::read_dir(root.join("jobs"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(job_dirs.len(), 2);
        assert!(job_dirs.iter().all(|dir| dir.join("ffmpeg.log").is_file()));
        fs::remove_dir_all(root).unwrap();
    }
}
