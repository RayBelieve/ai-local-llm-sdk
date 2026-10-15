use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::Path,
    process::{Command, Stdio},
    sync::{Mutex, OnceLock},
    thread,
    time::{Duration, Instant},
};

use serde::Serialize;

use crate::{
    process::hidden_command,
    resources::{EngineSpec, Selection},
};

static PROBES: OnceLock<Mutex<HashMap<String, Result<(), String>>>> = OnceLock::new();

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub vendor: String,
    pub driver: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemCheck {
    pub platform: String,
    pub architecture: String,
    pub windows_build: Option<u32>,
    pub cpu: String,
    pub avx2: bool,
    pub gpus: Vec<GpuInfo>,
    pub nvidia_driver: Option<String>,
    pub nvidia_compute_capability: Option<String>,
    pub available_memory_bytes: Option<u64>,
    pub nvidia_free_vram_bytes: Option<u64>,
    pub ffmpeg_path: Option<String>,
    pub ffmpeg_available: bool,
    pub diagnostics: Vec<String>,
}

fn gpu_vendor(name: &str) -> &'static str {
    let name = name.to_ascii_lowercase();
    if name.contains("nvidia") {
        "nvidia"
    } else if name.contains("amd") || name.contains("radeon") {
        "amd"
    } else if name.contains("intel") {
        "intel"
    } else {
        "other"
    }
}

pub(crate) fn driver_major(version: &str) -> Option<u32> {
    version.trim().split('.').next()?.parse().ok()
}

pub(crate) fn compute_capability(version: &str) -> Option<u32> {
    let (major, minor) = version.trim().split_once('.')?;
    Some(major.parse::<u32>().ok()? * 10 + minor.parse::<u32>().ok()?)
}

#[cfg(windows)]
fn windows_build() -> Option<u32> {
    use winreg::{enums::HKEY_LOCAL_MACHINE, RegKey};
    RegKey::predef(HKEY_LOCAL_MACHINE)
        .open_subkey("SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion")
        .ok()?
        .get_value::<String, _>("CurrentBuildNumber")
        .ok()?
        .parse()
        .ok()
}

fn ffmpeg_works(path: &str) -> bool {
    hidden_command(path)
        .arg("-version")
        .output()
        .map(|result| result.status.success())
        .unwrap_or(false)
}

pub fn system_check(selection: &Selection) -> SystemCheck {
    let platform = std::env::consts::OS.to_string();
    let architecture = std::env::consts::ARCH.to_string();
    #[cfg(windows)]
    let windows_build = windows_build();
    #[cfg(not(windows))]
    let windows_build = None;
    let avx2 = if cfg!(target_arch = "x86_64") || cfg!(target_arch = "x86") {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        {
            false
        }
    } else {
        false
    };
    let mut diagnostics = Vec::new();
    let mut gpus = Vec::new();
    if platform == "windows" {
        let result = hidden_command("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"])
            .output();
        match result {
            Ok(output) if output.status.success() => {
                let value = serde_json::from_slice::<serde_json::Value>(&output.stdout);
                match value {
                    Ok(value) => {
                        let entries: Vec<&serde_json::Value> = match &value {
                            serde_json::Value::Array(items) => items.iter().collect(),
                            serde_json::Value::Object(_) => vec![&value],
                            _ => Vec::new(),
                        };
                        for entry in entries {
                            let name = entry["Name"].as_str().unwrap_or("").trim();
                            let lower = name.to_ascii_lowercase();
                            if name.is_empty()
                                || ["microsoft basic", "virtual", "remote", "todesk"]
                                    .iter()
                                    .any(|part| lower.contains(part))
                            {
                                continue;
                            }
                            gpus.push(GpuInfo {
                                name: name.into(),
                                vendor: gpu_vendor(name).into(),
                                driver: entry["DriverVersion"].as_str().unwrap_or("").into(),
                            });
                        }
                    }
                    Err(_) => diagnostics.push("显卡自检结果无法解析。".into()),
                }
            }
            _ => diagnostics.push("无法读取显卡信息。".into()),
        }
    } else {
        diagnostics.push("当前版本尚未提供此平台的自动引擎安装。".into());
    }
    let nvidia_gpu = if platform == "windows" {
        hidden_command("nvidia-smi")
            .args([
                "--query-gpu=driver_version,compute_cap,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|output| output.lines().next().map(str::to_owned))
    } else {
        None
    };
    let (nvidia_driver, nvidia_compute_capability, nvidia_free_vram_bytes) = nvidia_gpu
        .as_deref()
        .and_then(|line| {
            let mut parts = line.split(',').map(str::trim);
            Some((parts.next()?, parts.next()?, parts.next()?))
        })
        .map(|(driver, compute, free_mib)| {
            (
                Some(driver.to_owned()),
                Some(compute.to_owned()),
                free_mib
                    .parse::<u64>()
                    .ok()
                    .and_then(|mib| mib.checked_mul(1024 * 1024)),
            )
        })
        .unwrap_or((None, None, None));
    let available_memory_bytes = (platform == "windows")
        .then(|| {
            hidden_command("powershell")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
                ])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .and_then(|value| value.trim().parse::<u64>().ok())
                .and_then(|kib| kib.checked_mul(1024))
        })
        .flatten();
    let ffmpeg_path = selection.ffmpeg_path.clone().or_else(|| {
        let locator = if platform == "windows" {
            "where.exe"
        } else {
            "which"
        };
        hidden_command(locator)
            .arg("ffmpeg")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|output| output.lines().next().map(str::trim).map(str::to_owned))
    });
    let ffmpeg_available = ffmpeg_path.as_deref().is_some_and(ffmpeg_works);
    if !ffmpeg_available {
        diagnostics.push("未找到可运行的 FFmpeg，请安装或登记本机路径。".into());
    }
    SystemCheck {
        platform,
        architecture,
        windows_build,
        cpu: std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "未知处理器".into()),
        avx2,
        gpus,
        nvidia_driver,
        nvidia_compute_capability,
        available_memory_bytes,
        nvidia_free_vram_bytes,
        ffmpeg_path,
        ffmpeg_available,
        diagnostics,
    }
}

pub fn suitability(spec: &EngineSpec, check: &SystemCheck) -> (bool, Option<String>) {
    if check.platform != "windows" || check.architecture != "x86_64" {
        return (false, Some("自动安装目前仅支持 Windows x86-64。".into()));
    }
    if check.windows_build.is_some_and(|build| build < 10240) {
        return (false, Some("此构建需要 Windows 10 或更新版本。".into()));
    }
    if spec.requires_avx2 && !check.avx2 {
        return (false, Some("处理器缺少 AVX2，请选 CPU 兼容版。".into()));
    }
    if spec.id == "crispasr-vulkan" && check.gpus.is_empty() {
        return (false, Some("未检测到可用于 Vulkan 的实体显卡。".into()));
    }
    if let Some(min_driver) = spec.min_nvidia_driver {
        if !check.gpus.iter().any(|gpu| gpu.vendor == "nvidia") && check.nvidia_driver.is_none() {
            return (false, Some("需要 NVIDIA 显卡。".into()));
        }
        let Some(driver) = check.nvidia_driver.as_deref().and_then(driver_major) else {
            return (
                false,
                Some("无法确认 NVIDIA 驱动版本，请选 Vulkan 或 CPU。".into()),
            );
        };
        if driver < min_driver {
            return (
                false,
                Some(format!(
                    "{} 需要 NVIDIA 驱动 {min_driver} 或更新版本。",
                    spec.label
                )),
            );
        }
        let Some(compute) = check
            .nvidia_compute_capability
            .as_deref()
            .and_then(compute_capability)
        else {
            return (
                false,
                Some("无法确认 NVIDIA GPU 架构，请选 Vulkan 或 CPU。".into()),
            );
        };
        if compute < spec.min_compute_capability.unwrap_or(0) {
            return (
                false,
                Some(format!("{} 不支持此显卡的计算能力。", spec.label)),
            );
        }
    }
    (true, None)
}

pub fn recommended_engine_id(available: impl Fn(&str) -> bool) -> Option<&'static str> {
    [
        "crispasr-cuda13",
        "crispasr-cuda12",
        "crispasr-vulkan",
        "crispasr-cpu",
        "crispasr-cpu-legacy",
    ]
    .into_iter()
    .find(|id| available(id))
}

fn probe_command(mut command: Command, timeout: Duration) -> Result<String, String> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("无法启动引擎兼容性测试: {error}"))?;
    let mut stdout = child.stdout.take().ok_or("无法读取引擎自检输出。")?;
    let mut stderr = child.stderr.take().ok_or("无法读取引擎自检错误。")?;
    let output = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stdout.read_to_end(&mut bytes);
        bytes
    });
    let errors = thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = stderr.read_to_end(&mut bytes);
        bytes
    });
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < timeout => thread::sleep(Duration::from_millis(100)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("引擎兼容性测试超时。".into());
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("引擎兼容性测试失败: {error}"));
            }
        }
    };
    let mut text = output.join().unwrap_or_default();
    text.extend(errors.join().unwrap_or_default());
    if !status.success() {
        let detail = String::from_utf8_lossy(&text);
        let tail = detail
            .lines()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join(" ");
        return Err(format!(
            "引擎兼容性测试退出状态异常: {status}。{}",
            if tail.is_empty() {
                String::new()
            } else {
                format!(" {tail}")
            }
        ));
    }
    Ok(String::from_utf8_lossy(&text).into_owned())
}

fn backend_devices(diagnostics: &str, backend: &str) -> Option<u32> {
    diagnostics
        .split("=== ggml backends + devices ===")
        .nth(1)?
        .split("registered devices")
        .next()?
        .lines()
        .find_map(|line| {
            line.split_once(&format!("] {backend} (devices: "))?
                .1
                .split_once(')')?
                .0
                .parse()
                .ok()
        })
}

fn cuda_arch_supported(diagnostics: &str) -> bool {
    let Some(compute) = diagnostics
        .lines()
        .find(|line| line.contains("Device 0:") && line.contains("compute capability "))
        .and_then(|line| line.split_once("compute capability "))
        .and_then(|(_, value)| value.split(',').next())
        .and_then(compute_capability)
    else {
        return false;
    };
    diagnostics
        .lines()
        .find(|line| line.trim_start().starts_with("cuda archs"))
        .and_then(|line| line.split_once(':'))
        .is_some_and(|(_, archs)| {
            archs.split(',').any(|entry| {
                let Some((arch, kind)) = entry.trim().split_once('-') else {
                    return false;
                };
                let Ok(arch) = arch.trim_end_matches('a').parse::<u32>() else {
                    return false;
                };
                (kind == "real" && arch == compute) || (kind == "virtual" && arch <= compute)
            })
        })
}

fn probe_key(path: &Path, suffix: &str, check: &SystemCheck) -> String {
    let metadata = fs::metadata(path).ok();
    format!(
        "{}|{}|{:?}|{}|{:?}|{:?}|{}",
        path.display(),
        metadata.as_ref().map_or(0, |value| value.len()),
        metadata.and_then(|value| value.modified().ok()),
        check.nvidia_driver.as_deref().unwrap_or(""),
        check.nvidia_compute_capability,
        check
            .gpus
            .iter()
            .map(|gpu| (&gpu.name, &gpu.driver))
            .collect::<Vec<_>>(),
        suffix
    )
}

fn cached_probe(key: String, run: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    let probes = PROBES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(result) = probes.lock().map_err(|error| error.to_string())?.get(&key) {
        return result.clone();
    }
    let result = run();
    probes
        .lock()
        .map_err(|error| error.to_string())?
        .insert(key, result.clone());
    result
}

pub(crate) fn clear_probe_cache() {
    if let Some(probes) = PROBES.get() {
        if let Ok(mut probes) = probes.lock() {
            probes.clear();
        }
    }
}

pub fn probe_crispasr(
    spec: &EngineSpec,
    path: &Path,
    check: &SystemCheck,
    model: Option<(&Path, crate::ModelProfile)>,
) -> Result<(), String> {
    let backend = match spec.gpu_backend {
        "cuda" => "CUDA",
        "vulkan" => "Vulkan",
        _ => "CPU",
    };
    cached_probe(probe_key(path, "diagnostics", check), || {
        let mut command = hidden_command(path);
        command.arg("--diagnostics");
        let diagnostics = probe_command(command, Duration::from_secs(15))?;
        if backend_devices(&diagnostics, backend).unwrap_or(0) == 0 {
            return Err(format!("{backend} 后端未检测到可用设备，请选择其他引擎。"));
        }
        if backend == "CUDA" && !cuda_arch_supported(&diagnostics) {
            return Err("引擎未包含当前 GPU 可运行的 CUDA 架构，请选择 Vulkan 或 CPU。".into());
        }
        Ok(())
    })?;
    if let Some((model, profile)) = model {
        let model_key = probe_key(model, "model", check);
        cached_probe(probe_key(path, &model_key, check), || {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos();
            let temporary = std::env::temp_dir()
                .join(format!("model-sdk-smoke-{}-{nonce}", std::process::id()));
            fs::create_dir(&temporary).map_err(|error| error.to_string())?;
            let audio = temporary.join("sample.wav");
            let mut wav = Vec::with_capacity(12_844);
            wav.extend_from_slice(b"RIFF");
            wav.extend_from_slice(&12_836_u32.to_le_bytes());
            wav.extend_from_slice(b"WAVEfmt ");
            wav.extend_from_slice(&16_u32.to_le_bytes());
            wav.extend_from_slice(&1_u16.to_le_bytes());
            wav.extend_from_slice(&1_u16.to_le_bytes());
            wav.extend_from_slice(&16_000_u32.to_le_bytes());
            wav.extend_from_slice(&32_000_u32.to_le_bytes());
            wav.extend_from_slice(&2_u16.to_le_bytes());
            wav.extend_from_slice(&16_u16.to_le_bytes());
            wav.extend_from_slice(b"data");
            wav.extend_from_slice(&12_800_u32.to_le_bytes());
            for index in 0..6_400 {
                let sample = ((index as f32 * std::f32::consts::TAU * 440.0 / 16_000.0).sin()
                    * 5_000.0) as i16;
                wav.extend_from_slice(&sample.to_le_bytes());
            }
            if let Err(error) = fs::write(&audio, wav) {
                let _ = fs::remove_dir_all(&temporary);
                return Err(error.to_string());
            }
            let mut command = hidden_command(path);
            command
                .arg("-m")
                .arg(model)
                .arg("-of")
                .arg(temporary.join("result"))
                .args(["-np", "-osrt", "-nf"]);
            if profile == crate::ModelProfile::Qwen3Asr17b {
                command.args(["--backend", "qwen3-1.7b"]);
            }
            match spec.gpu_backend {
                "cuda" | "vulkan" => {
                    command.args(["--gpu-backend", spec.gpu_backend]);
                }
                _ => {
                    command.arg("-ng");
                }
            }
            command.arg(&audio);
            let timeout = if profile == crate::ModelProfile::Qwen3Asr17b {
                180
            } else {
                90
            };
            let result = probe_command(command, Duration::from_secs(timeout)).map(|_| ());
            let _ = fs::remove_dir_all(temporary);
            result.map_err(|error| format!("短音频推理测试失败: {error}"))
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::resources::engine_spec;
    fn check(gpus: Vec<GpuInfo>, driver: Option<&str>, avx2: bool) -> SystemCheck {
        SystemCheck {
            platform: "windows".into(),
            architecture: "x86_64".into(),
            windows_build: Some(22631),
            cpu: "CPU".into(),
            avx2,
            gpus,
            nvidia_driver: driver.map(str::to_owned),
            nvidia_compute_capability: Some("8.6".into()),
            available_memory_bytes: None,
            nvidia_free_vram_bytes: None,
            ffmpeg_path: None,
            ffmpeg_available: false,
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn checks_cuda_driver_and_gpu_architecture() {
        let nvidia = vec![GpuInfo {
            name: "NVIDIA GPU".into(),
            vendor: "nvidia".into(),
            driver: "".into(),
        }];
        let cuda12 = engine_spec("crispasr-cuda12").unwrap();
        let cuda13 = engine_spec("crispasr-cuda13").unwrap();
        let eligible = check(nvidia.clone(), Some("581.15"), true);
        assert!(suitability(cuda13, &eligible).0);
        assert!(!suitability(cuda13, &check(nvidia.clone(), Some("576.88"), true)).0);
        assert!(suitability(cuda12, &check(nvidia.clone(), Some("576.88"), true)).0);
        assert!(!suitability(cuda12, &check(nvidia.clone(), None, true)).0);
        assert!(!suitability(cuda12, &check(nvidia.clone(), Some("527.99"), true)).0);
        let mut old_gpu = check(nvidia, Some("581.15"), true);
        old_gpu.nvidia_compute_capability = Some("6.1".into());
        assert!(!suitability(cuda13, &old_gpu).0);
        assert!(suitability(cuda12, &old_gpu).0);
        old_gpu.nvidia_compute_capability = None;
        assert!(!suitability(cuda12, &old_gpu).0);
        assert_eq!(
            recommended_engine_id(|id| suitability(engine_spec(id).unwrap(), &eligible).0),
            Some("crispasr-cuda13")
        );
    }

    #[test]
    fn diagnostics_require_a_device_and_supported_cuda_architecture() {
        let diagnostics = "=== ggml backends + devices ===\n  Device 0: GPU, compute capability 12.0, VRAM: 8 GiB\n    [0] CUDA (devices: 1)\n    [1] CPU (devices: 1)\n  registered devices : 2\n  cuda archs : 75-real,120-virtual";
        assert_eq!(backend_devices(diagnostics, "CUDA"), Some(1));
        assert_eq!(backend_devices(diagnostics, "Vulkan"), None);
        assert!(cuda_arch_supported(diagnostics));
        assert!(!cuda_arch_supported(
            &diagnostics.replace("120-virtual", "89-real")
        ));
    }

    #[test]
    #[ignore = "requires an installed CrispASR engine and model"]
    fn probes_installed_engine_with_short_audio() {
        let engine = std::env::var("MODEL_SDK_PROBE_ENGINE").unwrap();
        let model = std::env::var("MODEL_SDK_PROBE_MODEL").unwrap();
        let id = std::env::var("MODEL_SDK_PROBE_VARIANT").unwrap();
        let spec = engine_spec(&id).unwrap();
        let check = system_check(&Selection::default());
        let profile = if std::env::var("MODEL_SDK_PROBE_MODEL_ID")
            .unwrap_or_default()
            .starts_with("qwen3-")
        {
            crate::ModelProfile::Qwen3Asr17b
        } else {
            crate::ModelProfile::Whisper
        };
        probe_crispasr(
            spec,
            Path::new(&engine),
            &check,
            Some((Path::new(&model), profile)),
        )
        .unwrap();
    }

    #[test]
    fn old_cpu_uses_legacy_variant() {
        assert!(
            !suitability(
                engine_spec("crispasr-cpu").unwrap(),
                &check(Vec::new(), None, false)
            )
            .0
        );
        assert!(
            suitability(
                engine_spec("crispasr-cpu-legacy").unwrap(),
                &check(Vec::new(), None, false)
            )
            .0
        );
    }
}
