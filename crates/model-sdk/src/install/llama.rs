use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    process::hidden_command,
    resources::{self, LlamaVariant},
    sdk::{ModelSdk, LLAMA_ENGINE_ID},
    task::TaskControl,
};

use super::download_to_range;

fn extract_runtime(archive: &Path, runtime: &Path, task: &TaskControl) -> Result<(), String> {
    let mut zip = zip::ZipArchive::new(File::open(archive).map_err(|error| error.to_string())?)
        .map_err(|error| format!("llama.cpp 压缩包无效: {error}"))?;
    for index in 0..zip.len() {
        task.check_cancelled()?;
        let mut entry = zip.by_index(index).map_err(|error| error.to_string())?;
        let Some(name) = Path::new(entry.name()).file_name() else {
            continue;
        };
        let filename = name.to_string_lossy();
        if filename != "llama-server.exe" && !filename.to_ascii_lowercase().ends_with(".dll") {
            continue;
        }
        let mut output = File::create(runtime.join(name)).map_err(|error| error.to_string())?;
        io::copy(&mut entry, &mut output).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn install_variant(
    sdk: &ModelSdk,
    variant: &LlamaVariant,
    task: &TaskControl,
) -> Result<PathBuf, String> {
    let parent = sdk.root().join("engines");
    fs::create_dir_all(&parent).map_err(|error| error.to_string())?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let work = parent.join(format!("llama-staging-{nonce}"));
    let runtime = work.join("runtime");
    fs::create_dir_all(&runtime).map_err(|error| error.to_string())?;
    let result = (|| {
        let total = variant.assets.iter().map(|asset| asset.size).sum::<u64>();
        let mut completed = 0;
        for asset in variant.assets {
            let archive = work.join(asset.file);
            let from = (completed * 90 / total) as u8;
            task.progress("engine", from, &format!("正在下载 {}", asset.file));
            download_to_range(
                &format!("{}/{}", resources::LLAMA_ENGINE_BASE, asset.file),
                &archive,
                "engine",
                from,
                ((completed + asset.size) * 90 / total) as u8,
                task,
            )?;
            let received = archive.metadata().map_err(|error| error.to_string())?.len();
            if received != asset.size {
                return Err(format!("引擎压缩包大小不符: {}", asset.file));
            }
            extract_runtime(&archive, &runtime, task)?;
            completed += asset.size;
        }
        let executable = runtime.join("llama-server.exe");
        if !executable.is_file() {
            return Err("压缩包内没有 llama-server.exe。".into());
        }
        task.progress("engine", 94, "正在验证 llama.cpp 运行时");
        if !hidden_command(&executable)
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success())
        {
            return Err("下载的 llama-server.exe 无法运行。".into());
        }
        task.check_cancelled()?;
        let target = parent.join("llama");
        let backup = parent.join(format!("llama-backup-{nonce}"));
        if target.exists() {
            fs::rename(&target, &backup).map_err(|error| error.to_string())?;
        }
        if let Err(error) = fs::rename(&runtime, &target) {
            if backup.exists() {
                fs::rename(&backup, &target)
                    .map_err(|rollback| format!("安装失败: {error}; 恢复旧版失败: {rollback}"))?;
            }
            return Err(error.to_string());
        }
        if let Err(error) = sdk.clear_engine_override(LLAMA_ENGINE_ID) {
            fs::rename(&target, &runtime)
                .map_err(|rollback| format!("资源登记失败: {error}; 恢复旧版失败: {rollback}"))?;
            if backup.exists() {
                fs::rename(&backup, &target).map_err(|rollback| {
                    format!("资源登记失败: {error}; 恢复旧版失败: {rollback}")
                })?;
            }
            return Err(error);
        }
        if backup.exists() {
            let _ = fs::remove_dir_all(backup);
        }
        task.progress("engine", 100, "llama.cpp 运行时已就绪");
        Ok(target.join("llama-server.exe"))
    })();
    let _ = fs::remove_dir_all(work);
    result
}

impl ModelSdk {
    pub fn download_llama_engine(
        &self,
        variant_id: &str,
        task: &TaskControl,
    ) -> Result<PathBuf, String> {
        let variant = resources::llama_variant(variant_id)?;
        let option = self
            .llama_download_options()
            .into_iter()
            .find(|option| option.id == variant_id)
            .expect("known llama variant");
        if !option.suitable {
            return Err(option.reason.unwrap_or("此运行时不适合当前设备。".into()));
        }
        install_variant(self, variant, task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn extracts_server_and_dependencies_without_archive_paths() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("llama-archive-{nonce}"));
        fs::create_dir_all(root.join("runtime")).unwrap();
        let archive = root.join("engine.zip");
        let mut zip = zip::ZipWriter::new(File::create(&archive).unwrap());
        for (name, contents) in [
            ("bin/llama-server.exe", b"server".as_slice()),
            ("bin/ggml-vulkan.dll", b"vulkan".as_slice()),
            ("other/readme.txt", b"ignored".as_slice()),
        ] {
            zip.start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(contents).unwrap();
        }
        zip.finish().unwrap();
        extract_runtime(&archive, &root.join("runtime"), &TaskControl::default()).unwrap();
        assert_eq!(
            fs::read(root.join("runtime/llama-server.exe")).unwrap(),
            b"server"
        );
        assert_eq!(
            fs::read(root.join("runtime/ggml-vulkan.dll")).unwrap(),
            b"vulkan"
        );
        assert!(!root.join("runtime/readme.txt").exists());
        fs::remove_dir_all(root).unwrap();
    }
}
