use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::Stdio,
};

use crate::{engines, network, process::hidden_command, sdk::ModelSdk, task::TaskControl};

use super::download_to;

fn install_step(
    program: &Path,
    args: &[&str],
    dir: &Path,
    task: &TaskControl,
) -> Result<(), String> {
    let log_path = dir.join("install.log");
    let log = File::create(&log_path).map_err(|error| error.to_string())?;
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
            "ONNX Runtime 环境安装失败 ({status}): {}",
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
    pub(super) fn install_onnxruntime_runtime(
        &self,
        task: &TaskControl,
    ) -> Result<PathBuf, String> {
        if !cfg!(all(windows, target_arch = "x86_64")) {
            return Err("ONNX Runtime 自动安装目前仅支持 Windows x86-64。".into());
        }
        let dir = self.root().join("engines/onnxruntime-cpu");
        fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let python = engines::onnxruntime::managed_path(self.root());
        if dir.join("ready.txt").is_file()
            && python.is_file()
            && engines::onnxruntime::probe(&python, None).is_ok()
        {
            self.clear_engine_override("onnxruntime-cpu")?;
            return Ok(python);
        }
        let _ = fs::remove_file(dir.join("ready.txt"));
        let uv = dir.join("uv.exe");
        if !uv.is_file() {
            task.progress("engine", 0, "正在获取运行时安装器");
            let archive = dir.join("uv.zip");
            download_to(
                "https://github.com/astral-sh/uv/releases/download/0.11.3/uv-x86_64-pc-windows-msvc.zip",
                &archive,
                "engine",
                task,
            )?;
            let mut zip =
                zip::ZipArchive::new(File::open(&archive).map_err(|error| error.to_string())?)
                    .map_err(|error| format!("运行时安装器压缩包无效: {error}"))?;
            let mut found = false;
            for index in 0..zip.len() {
                task.check_cancelled()?;
                let mut entry = zip.by_index(index).map_err(|error| error.to_string())?;
                if Path::new(entry.name())
                    .file_name()
                    .is_some_and(|name| name == "uv.exe")
                {
                    std::io::copy(
                        &mut entry,
                        &mut File::create(&uv).map_err(|error| error.to_string())?,
                    )
                    .map_err(|error| error.to_string())?;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err("安装器压缩包缺少 uv.exe。".into());
            }
            let _ = fs::remove_file(archive);
        }
        task.progress("engine", 15, "正在安装便携 Python");
        install_step(&uv, &["python", "install", "3.12"], &dir, task)?;
        task.progress("engine", 40, "正在创建独立推理环境");
        let venv = dir.join("venv");
        install_step(
            &uv,
            &[
                "venv",
                "--python",
                "3.12",
                "--managed-python",
                venv.to_str().ok_or("运行时路径编码无效。")?,
            ],
            &dir,
            task,
        )?;
        task.progress("engine", 55, "正在安装 ONNX Runtime");
        install_step(
            &uv,
            &[
                "pip",
                "install",
                "--python",
                python.to_str().ok_or("Python 路径编码无效。")?,
                "onnxruntime==1.30.0",
                "numpy==2.5.3",
                "pillow==12.0.0",
            ],
            &dir,
            task,
        )?;
        engines::onnxruntime::probe(&python, None)?;
        fs::write(dir.join("ready.txt"), "onnxruntime=1.30.0\n")
            .map_err(|error| error.to_string())?;
        self.clear_engine_override("onnxruntime-cpu")?;
        task.progress("engine", 100, "ONNX Runtime 已就绪");
        Ok(python)
    }
}
