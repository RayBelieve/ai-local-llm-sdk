use std::{
    fs::{self, File},
    path::{Path, PathBuf},
};

use crate::{
    catalog::TaskKind,
    diagnostics,
    engines::{self, Family},
    resources,
    sdk::ModelSdk,
    task::TaskControl,
};

use super::{download_to, replace_file};

impl ModelSdk {
    pub fn download_engine(&self, id: &str, task: &TaskControl) -> Result<PathBuf, String> {
        match engines::get(id)?.family {
            Family::Llama => return self.install_llama_variant(id, task),
            Family::OpenVino => return self.install_openvino_runtime(task),
            Family::OnnxRuntime => return self.install_onnxruntime_runtime(task),
            Family::CrispAsr => {}
        }
        if !cfg!(target_os = "windows") || !cfg!(target_arch = "x86_64") {
            return Err("自动安装目前仅支持 Windows x86-64。".into());
        }
        let spec = resources::engine_spec(id)?;
        let system = diagnostics::system_check(&resources::Selection::default());
        let (suitable, reason) = diagnostics::suitability(spec, &system);
        if !suitable {
            return Err(reason.unwrap_or_else(|| "此引擎不适合当前设备。".into()));
        }
        let parent = self.root().join("engines/crispasr");
        fs::create_dir_all(&parent).map_err(|error| error.to_string())?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos();
        let staging = parent.join(format!("{}-staging-{nonce}", spec.id));
        fs::create_dir(&staging).map_err(|error| error.to_string())?;
        let result = (|| {
            let archive = staging.join(spec.asset);
            // shortcut: engine release archives have no pinned digest here; add one before supporting new sources.
            download_to(
                &format!("{}/{}", resources::ENGINE_BASE, spec.asset),
                &archive,
                "engine",
                task,
            )?;
            let mut zip =
                zip::ZipArchive::new(File::open(&archive).map_err(|error| error.to_string())?)
                    .map_err(|error| format!("引擎压缩包无效: {error}"))?;
            for index in 0..zip.len() {
                task.check_cancelled()?;
                let mut entry = zip.by_index(index).map_err(|error| error.to_string())?;
                let Some(name) = Path::new(entry.name())
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_owned)
                else {
                    continue;
                };
                let extension = Path::new(&name)
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if extension != "exe" && extension != "dll" {
                    continue;
                }
                let mut output =
                    File::create(staging.join(name)).map_err(|error| error.to_string())?;
                std::io::copy(&mut entry, &mut output).map_err(|error| error.to_string())?;
            }
            drop(zip);
            fs::remove_file(&archive).map_err(|error| error.to_string())?;
            if !staging.join("crispasr.exe").is_file() {
                return Err("压缩包内没有 crispasr.exe。".into());
            }
            fs::write(staging.join("backend.txt"), spec.id).map_err(|error| error.to_string())?;
            let model = self
                .models()?
                .into_iter()
                .filter(|item| {
                    item.descriptor.task == TaskKind::Transcription
                        && engines::get(id).is_ok_and(|adapter| {
                            adapter.supports(
                                item.descriptor.profile,
                                item.descriptor.artifact_format,
                                item.descriptor.task,
                            )
                        })
                })
                .find_map(|item| item.path.map(|path| (path, item.descriptor.profile)));
            diagnostics::probe_crispasr(
                spec,
                &staging.join("crispasr.exe"),
                &system,
                model
                    .as_ref()
                    .map(|(path, profile)| (path.as_path(), *profile)),
            )?;
            let target = parent.join(spec.id);
            let backup = parent.join(format!("{}-backup-{nonce}", spec.id));
            if target.exists() {
                fs::rename(&target, &backup).map_err(|error| error.to_string())?;
            }
            if let Err(error) = fs::rename(&staging, &target) {
                if backup.exists() {
                    let _ = fs::rename(&backup, &target);
                }
                return Err(error.to_string());
            }
            if backup.exists() {
                let _ = fs::remove_dir_all(backup);
            }
            diagnostics::clear_probe_cache();
            self.clear_engine_override(id)?;
            task.progress("engine", 100, "推理引擎已就绪");
            Ok(target.join("crispasr.exe"))
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(staging);
        }
        result
    }

    pub fn download_model(
        &self,
        id: &str,
        mirror: bool,
        task: &TaskControl,
    ) -> Result<PathBuf, String> {
        if id == resources::OPENVINO_MODEL_ID {
            return self.download_openvino_model(mirror, task);
        }
        let spec = resources::model_spec(id)?;
        let dir = self.root().join("models");
        fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let path = dir.join(spec.file);
        if !path.is_file() || resources::verify_model(&path, spec).is_err() {
            let staged = dir.join(format!("{}.download", spec.file));
            let result = (|| {
                download_to(&resources::model_url(spec, mirror), &staged, "model", task)?;
                resources::verify_model(&staged, spec)?;
                replace_file(&staged, &path)
            })();
            if result.is_err() {
                let _ = fs::remove_file(staged);
            }
            result?;
        }
        self.clear_model_override(id)?;
        task.progress("model", 100, "模型已就绪");
        Ok(path)
    }
}
