use std::{
    fs::{self, File},
    path::PathBuf,
};

use crate::{process::hidden_command, sdk::ModelSdk, task::TaskControl};

use super::{download_to, replace_file, verify_sha256};

impl ModelSdk {
    pub fn install_ffmpeg(&self, task: &TaskControl) -> Result<PathBuf, String> {
        let (url, digest) = match (std::env::consts::OS, std::env::consts::ARCH) {
            ("windows", "x86_64") => (
                "https://files.pythonhosted.org/packages/2c/c6/fa760e12a2483469e2bf5058c5faff664acf66cadb4df2ad6205b016a73d/imageio_ffmpeg-0.6.0-py3-none-win_amd64.whl",
                "02fa47c83703c37df6bfe4896aab339013f62bf02c5ebf2dce6da56af04ffc0a",
            ),
            ("macos", "x86_64") => (
                "https://files.pythonhosted.org/packages/da/58/87ef68ac83f4c7690961bce288fd8e382bc5f1513860fc7f90a9c1c1c6bf/imageio_ffmpeg-0.6.0-py3-none-macosx_10_9_intel.macosx_10_9_x86_64.whl",
                "9d2baaf867088508d4a3458e61eeb30e945c4ad8016025545f66c4b5aaef0a61",
            ),
            ("macos", "aarch64") => (
                "https://files.pythonhosted.org/packages/40/5c/f3d8a657d362cc93b81aab8feda487317da5b5d31c0e1fdfd5e986e55d17/imageio_ffmpeg-0.6.0-py3-none-macosx_11_0_arm64.whl",
                "b1ae3173414b5fc5f538a726c4e48ea97edc0d2cdc11f103afee655c463fa742",
            ),
            ("linux", "x86_64") => (
                "https://files.pythonhosted.org/packages/a0/2d/43c8522a2038e9d0e7dbdf3a61195ecc31ca576fb1527a528c877e87d973/imageio_ffmpeg-0.6.0-py3-none-manylinux2014_x86_64.whl",
                "c7e46fcec401dd990405049d2e2f475e2b397779df2519b544b8aab515195282",
            ),
            ("linux", "aarch64") => (
                "https://files.pythonhosted.org/packages/33/e7/1925bfbc563c39c1d2e82501d8372734a5c725e53ac3b31b4c2d081e895b/imageio_ffmpeg-0.6.0-py3-none-manylinux2014_aarch64.whl",
                "1d47bebd83d2c5fc770720d211855f208af8a596c82d17730aa51e815cdee6dc",
            ),
            _ => return Err("当前平台没有可自动安装的 FFmpeg，请指定本机路径。".into()),
        };
        let dir = self.root().join("tools/ffmpeg");
        fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        let executable = dir.join(if cfg!(windows) {
            "ffmpeg.exe"
        } else {
            "ffmpeg"
        });
        if executable.is_file()
            && hidden_command(&executable)
                .arg("-version")
                .output()
                .is_ok_and(|output| output.status.success())
        {
            task.progress("ffmpeg", 100, "FFmpeg 已就绪");
            return Ok(executable);
        }
        let archive = dir.join("imageio-ffmpeg-0.6.0.whl");
        let staged = dir.join(if cfg!(windows) {
            "ffmpeg.install.exe"
        } else {
            "ffmpeg.install"
        });
        let result = (|| {
            download_to(url, &archive, "ffmpeg", task)?;
            verify_sha256(&archive, digest)?;
            task.check_cancelled()?;
            let mut wheel =
                zip::ZipArchive::new(File::open(&archive).map_err(|error| error.to_string())?)
                    .map_err(|error| format!("FFmpeg 安装包无效: {error}"))?;
            let binaries = (0..wheel.len())
                .filter(|&index| {
                    wheel.by_index(index).is_ok_and(|entry| {
                        entry
                            .name()
                            .strip_prefix("imageio_ffmpeg/binaries/")
                            .is_some_and(|name| {
                                name.starts_with("ffmpeg-")
                                    && !name.contains('/')
                                    && name.ends_with(if cfg!(windows) { ".exe" } else { "" })
                            })
                    })
                })
                .collect::<Vec<_>>();
            if binaries.len() != 1 {
                return Err("FFmpeg 安装包未包含唯一的可执行文件。".into());
            }
            let mut binary = wheel
                .by_index(binaries[0])
                .map_err(|error| error.to_string())?;
            let mut output = File::create(&staged).map_err(|error| error.to_string())?;
            std::io::copy(&mut binary, &mut output).map_err(|error| error.to_string())?;
            output.sync_all().map_err(|error| error.to_string())?;
            drop(output);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&staged, fs::Permissions::from_mode(0o755))
                    .map_err(|error| error.to_string())?;
            }
            task.check_cancelled()?;
            if !hidden_command(&staged)
                .arg("-version")
                .output()
                .is_ok_and(|output| output.status.success())
            {
                return Err("下载的 FFmpeg 无法运行。".into());
            }
            replace_file(&staged, &executable)?;
            task.progress("ffmpeg", 100, "FFmpeg 已就绪");
            Ok(executable)
        })();
        let _ = fs::remove_file(archive);
        if result.is_err() {
            let _ = fs::remove_file(staged);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "downloads the pinned FFmpeg wheel"]
    fn downloads_and_runs_ffmpeg() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("model-sdk-ffmpeg-{nonce}"));
        let sdk = ModelSdk::new(&root).unwrap();
        let executable = sdk.install_ffmpeg(&TaskControl::default()).unwrap();
        assert!(hidden_command(&executable)
            .arg("-version")
            .output()
            .unwrap()
            .status
            .success());
        assert_eq!(
            sdk.install_ffmpeg(&TaskControl::default()).unwrap(),
            executable
        );
        fs::remove_dir_all(root).unwrap();
    }
}
