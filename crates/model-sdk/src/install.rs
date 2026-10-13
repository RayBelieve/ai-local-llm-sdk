mod crispasr;
mod ffmpeg;
mod llama;
mod openvino;

use std::{
    fs::{self, File},
    io::{Read, Write},
    path::Path,
};

use sha2::{Digest, Sha256};

use crate::{network, task::TaskControl};

pub(crate) fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    if format!("{:x}", digest.finalize()) != expected {
        return Err(format!("资源校验失败: {}", path.display()));
    }
    Ok(())
}

fn replace_file(staged: &Path, target: &Path) -> Result<(), String> {
    let backup = target.with_extension("backup");
    if target.exists() {
        if backup.exists() {
            fs::remove_file(&backup).map_err(|error| error.to_string())?;
        }
        fs::rename(target, &backup).map_err(|error| error.to_string())?;
    }
    if let Err(error) = fs::rename(staged, target) {
        if backup.exists() {
            let _ = fs::rename(&backup, target);
        }
        return Err(error.to_string());
    }
    if backup.exists() {
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

fn download_to(url: &str, target: &Path, stage: &str, task: &TaskControl) -> Result<(), String> {
    download_to_range(url, target, stage, 0, 100, task)
}

fn download_to_range(
    url: &str,
    target: &Path,
    stage: &str,
    from: u8,
    to: u8,
    task: &TaskControl,
) -> Result<(), String> {
    let mut response = network::download_client()?
        .get(url)
        .send()
        .map_err(|error| format!("下载连接失败: {error}"))?
        .error_for_status()
        .map_err(|error| format!("下载失败: {error}"))?;
    let expected = response.content_length().unwrap_or(0);
    let part = target.with_extension("part");
    let result = (|| {
        let mut file = File::create(&part).map_err(|error| error.to_string())?;
        let mut received = 0_u64;
        let mut previous = 255;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            task.check_cancelled()?;
            let count = response
                .read(&mut buffer)
                .map_err(|error| error.to_string())?;
            if count == 0 {
                break;
            }
            file.write_all(&buffer[..count])
                .map_err(|error| error.to_string())?;
            received += count as u64;
            let percent = if expected == 0 {
                from
            } else {
                from + ((received * u64::from(to - from) / expected).min(u64::from(to - from))
                    as u8)
            };
            if percent != previous {
                task.progress(
                    stage,
                    percent,
                    &format!("已下载 {:.1} MB", received as f64 / 1_048_576.0),
                );
                previous = percent;
            }
        }
        file.sync_all().map_err(|error| error.to_string())?;
        if expected != 0 && received != expected {
            return Err(format!("下载不完整: 收到 {received} / {expected} 字节"));
        }
        Ok::<(), String>(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result?;
    if let Err(error) = task.check_cancelled() {
        let _ = fs::remove_file(&part);
        return Err(error);
    }
    replace_file(&part, target)
}
