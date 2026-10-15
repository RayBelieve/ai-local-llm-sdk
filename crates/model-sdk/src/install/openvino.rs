use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::Stdio,
};

use crate::{network, process::hidden_command, resources, sdk::ModelSdk, task::TaskControl};

use super::{download_to, replace_file, verify_sha256};

const VAD_SHA256: &str = "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28";
const VAD_URL: &str =
    "https://raw.githubusercontent.com/snakers4/silero-vad/v4.0/files/silero_vad.onnx";
const ALIGNER_FILE: &str = "qwen3-forced-aligner-0.6b-q5_0.gguf";
const ALIGNER_SIZE: u64 = 643_149_568;
const ALIGNER_SHA256: &str = "d76ce68f71a241356d74f9148ad0dc29a0a92c00647bc27f5d166fbae8f78bf4";
const ALIGNER_URL: &str = "https://huggingface.co/cstr/qwen3-forced-aligner-0.6b-GGUF/resolve/main/qwen3-forced-aligner-0.6b-q5_0.gguf";

fn run_install_step(
    program: &Path,
    args: &[&str],
    log_path: &Path,
    dir: &Path,
    task: &TaskControl,
) -> Result<(), String> {
    let log = File::create(log_path).map_err(|error| error.to_string())?;
    let stderr = log.try_clone().map_err(|error| error.to_string())?;
    let mut command = hidden_command(program);
    command
        .args(args)
        .env("UV_PYTHON_INSTALL_DIR", dir.join("python"))
        .env("UV_CACHE_DIR", dir.join("cache"))
        .env("UV_NO_PROGRESS", "1")
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr));
    network::apply_proxy_to_command(&mut command)?;
    let status = task.run(command)?;
    if !status.success() {
        let log = fs::read_to_string(log_path).unwrap_or_default();
        return Err(format!(
            "OpenVINO 环境安装失败 ({status}): {}",
            log.chars()
                .rev()
                .take(1000)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>()
        ));
    }
    Ok(())
}

impl ModelSdk {
    pub(super) fn install_openvino_runtime(&self, task: &TaskControl) -> Result<PathBuf, String> {
        if !cfg!(target_os = "windows") || !cfg!(target_arch = "x86_64") {
            return Err("OpenVINO 自动安装目前支持 Windows x86-64。".into());
        }
        let dir = self.root().join("engines/openvino");
        fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let python = dir.join("venv/Scripts/python.exe");
        if dir.join("ready.txt").is_file()
            && python.is_file()
            && hidden_command(&python)
                .args(["-c", "import openvino, onnxruntime, numpy"])
                .output()
                .is_ok_and(|output| output.status.success())
        {
            self.clear_engine_override(resources::OPENVINO_ENGINE_ID)?;
            return Ok(python);
        }
        let _ = fs::remove_file(dir.join("ready.txt"));
        let uv = dir.join("uv.exe");
        if !uv.is_file() {
            task.progress("engine", 0, "正在获取运行时安装器");
            let archive = dir.join("uv.zip");
            download_to("https://github.com/astral-sh/uv/releases/download/0.11.3/uv-x86_64-pc-windows-msvc.zip", &archive, "engine", task)?;
            let mut zip =
                zip::ZipArchive::new(File::open(&archive).map_err(|error| error.to_string())?)
                    .map_err(|error| format!("运行时安装器压缩包无效: {error}"))?;
            let mut found = false;
            for index in 0..zip.len() {
                let mut entry = zip.by_index(index).map_err(|error| error.to_string())?;
                if Path::new(entry.name())
                    .file_name()
                    .is_some_and(|name| name == "uv.exe")
                {
                    let mut output = File::create(&uv).map_err(|error| error.to_string())?;
                    std::io::copy(&mut entry, &mut output).map_err(|error| error.to_string())?;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err("安装器压缩包缺少 uv.exe。".into());
            }
            let _ = fs::remove_file(archive);
        }
        let log = dir.join("install.log");
        task.progress("engine", 15, "正在安装便携 Python");
        run_install_step(&uv, &["python", "install", "3.12"], &log, &dir, task)?;
        task.progress("engine", 40, "正在创建独立推理环境");
        let venv = dir.join("venv");
        let venv_arg = venv.to_str().ok_or("运行时路径编码无效。")?;
        run_install_step(
            &uv,
            &["venv", "--python", "3.12", "--managed-python", venv_arg],
            &log,
            &dir,
            task,
        )?;
        task.progress("engine", 55, "正在安装 OpenVINO 与 ONNX Runtime");
        let python_arg = python.to_str().ok_or("Python 路径编码无效。")?;
        run_install_step(
            &uv,
            &[
                "pip",
                "install",
                "--python",
                python_arg,
                "openvino==2026.4.1",
                "onnxruntime==1.30.0",
                "numpy==2.5.3",
            ],
            &log,
            &dir,
            task,
        )?;
        run_install_step(
            &python,
            &["-c", "import openvino, onnxruntime, numpy"],
            &log,
            &dir,
            task,
        )?;
        fs::write(
            dir.join("ready.txt"),
            "openvino=2026.4.1\nonnxruntime=1.30.0\n",
        )
        .map_err(|error| error.to_string())?;
        self.ensure_openvino_worker()?;
        self.clear_engine_override(resources::OPENVINO_ENGINE_ID)?;
        task.progress("engine", 100, "OpenVINO 运行时已就绪");
        Ok(python)
    }

    pub(super) fn download_openvino_model(
        &self,
        mirror: bool,
        task: &TaskControl,
    ) -> Result<PathBuf, String> {
        let dir = self.root().join("models/openvino/qwen3_asr_int8");
        fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let host = if mirror {
            "https://hf-mirror.com"
        } else {
            "https://huggingface.co"
        };
        for (index, name) in resources::OPENVINO_MODEL_FILES.iter().enumerate() {
            task.check_cancelled()?;
            let path = dir.join(name);
            let expected = resources::OPENVINO_MODEL_HASHES
                .iter()
                .find(|(file, _)| file == name)
                .map(|(_, hash)| *hash);
            let valid = if let Some(hash) = expected {
                path.is_file() && verify_sha256(&path, hash).is_ok()
            } else {
                path.metadata().is_ok_and(|metadata| metadata.len() >= 100)
            };
            if valid {
                continue;
            }
            task.progress(
                "model",
                (index * 100 / resources::OPENVINO_MODEL_FILES.len()) as u8,
                &format!("正在获取 {name}"),
            );
            let url =
                format!("{host}/dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO/resolve/main/{name}");
            if let Err(primary) = download_to(&url, &path, "model", task) {
                task.check_cancelled()?;
                let fallback = format!(
                    "{host}/Echo9Zulu/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO/resolve/main/{name}"
                );
                download_to(&fallback, &path, "model", task)
                    .map_err(|error| format!("模型下载失败: {primary}; 备用源: {error}"))?;
            }
            if let Some(hash) = expected {
                verify_sha256(&path, hash)?;
            }
        }
        resources::verify_openvino_model(&dir)?;
        self.ensure_openvino_worker()?;
        self.clear_model_override(resources::OPENVINO_MODEL_ID)?;
        task.progress("model", 100, "Qwen3-ASR 模型已就绪");
        Ok(dir)
    }

    pub fn ensure_openvino_worker(&self) -> Result<PathBuf, String> {
        let worker = self.root().join("workers/openvino");
        fs::create_dir_all(&worker).map_err(|error| error.to_string())?;
        for (name, contents) in [
            (
                "worker.py",
                include_bytes!("../../workers/openvino/worker.py").as_slice(),
            ),
            (
                "processor_numpy.py",
                include_bytes!("../../workers/openvino/processor_numpy.py").as_slice(),
            ),
            (
                "prompt_template.json",
                include_bytes!("../../workers/openvino/prompt_template.json").as_slice(),
            ),
            (
                "mel_filters.npy",
                include_bytes!("../../workers/openvino/mel_filters.npy").as_slice(),
            ),
            (
                "LICENSE-STTMiniTool",
                include_bytes!("../../workers/openvino/LICENSE-STTMiniTool").as_slice(),
            ),
            (
                "LICENSE-Silero",
                include_bytes!("../../workers/openvino/LICENSE-Silero").as_slice(),
            ),
        ] {
            let target = worker.join(name);
            if !fs::read(&target).is_ok_and(|current| current == contents) {
                fs::write(target, contents).map_err(|error| error.to_string())?;
            }
        }
        Ok(worker)
    }

    pub fn ensure_vad_model(&self, task: &TaskControl) -> Result<PathBuf, String> {
        let path = self.ensure_openvino_worker()?.join("silero_vad_v4.onnx");
        if !path.is_file() || verify_sha256(&path, VAD_SHA256).is_err() {
            download_to(VAD_URL, &path, "model", task)?;
            verify_sha256(&path, VAD_SHA256)?;
        }
        Ok(path)
    }

    pub fn register_local_vad_model(&self, source: &Path) -> Result<PathBuf, String> {
        if !source.is_absolute() || !source.is_file() {
            return Err("请选择有效的 Silero VAD 模型完整路径。".into());
        }
        verify_sha256(source, VAD_SHA256)?;
        let target = self.ensure_openvino_worker()?.join("silero_vad_v4.onnx");
        if fs::canonicalize(source).map_err(|error| error.to_string())? == target {
            return Ok(target);
        }
        let staged = target.with_extension("import");
        fs::copy(source, &staged).map_err(|error| error.to_string())?;
        if let Err(error) =
            verify_sha256(&staged, VAD_SHA256).and_then(|_| replace_file(&staged, &target))
        {
            let _ = fs::remove_file(staged);
            return Err(error);
        }
        Ok(target)
    }

    pub fn aligner_path(&self) -> Option<PathBuf> {
        let path = self.root().join("models/aligner").join(ALIGNER_FILE);
        path.metadata()
            .ok()
            .filter(|metadata| metadata.is_file() && metadata.len() == ALIGNER_SIZE)
            .map(|_| path)
    }

    pub fn ensure_aligner_model(&self, task: &TaskControl) -> Result<PathBuf, String> {
        let path = self.root().join("models/aligner").join(ALIGNER_FILE);
        if self.aligner_path().is_some() && verify_sha256(&path, ALIGNER_SHA256).is_ok() {
            return Ok(path);
        }
        fs::create_dir_all(path.parent().ok_or("对齐器目标路径无效。")?)
            .map_err(|error| error.to_string())?;
        let staged = path.with_extension("download");
        let result = (|| {
            download_to(ALIGNER_URL, &staged, "aligner", task)?;
            verify_sha256(&staged, ALIGNER_SHA256)?;
            replace_file(&staged, &path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(staged);
        }
        result?;
        Ok(path)
    }

    pub fn register_local_aligner_model(&self, source: &Path) -> Result<PathBuf, String> {
        if !source.is_absolute() || !source.is_file() {
            return Err("请选择有效的 Qwen3 对齐器完整路径。".into());
        }
        if source.metadata().map_err(|error| error.to_string())?.len() != ALIGNER_SIZE {
            return Err("Qwen3 对齐器文件大小不符。".into());
        }
        verify_sha256(source, ALIGNER_SHA256)?;
        let target = self.root().join("models/aligner").join(ALIGNER_FILE);
        fs::create_dir_all(target.parent().ok_or("对齐器目标路径无效。")?)
            .map_err(|error| error.to_string())?;
        if fs::canonicalize(source).map_err(|error| error.to_string())? == target {
            return Ok(target);
        }
        let staged = target.with_extension("import");
        fs::copy(source, &staged).map_err(|error| error.to_string())?;
        if let Err(error) =
            verify_sha256(&staged, ALIGNER_SHA256).and_then(|_| replace_file(&staged, &target))
        {
            let _ = fs::remove_file(staged);
            return Err(error);
        }
        Ok(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires access to the tagged Silero VAD release"]
    fn downloads_and_checks_vad_weight() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("model-sdk-vad-{nonce}"));
        let sdk = ModelSdk::new(&root).unwrap();
        let model = sdk.ensure_vad_model(&TaskControl::default()).unwrap();
        verify_sha256(&model, VAD_SHA256).unwrap();
        let imported_root = std::env::temp_dir().join(format!("model-sdk-vad-import-{nonce}"));
        let imported = ModelSdk::new(&imported_root)
            .unwrap()
            .register_local_vad_model(&model)
            .unwrap();
        verify_sha256(&imported, VAD_SHA256).unwrap();
        fs::remove_dir_all(imported_root).unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
