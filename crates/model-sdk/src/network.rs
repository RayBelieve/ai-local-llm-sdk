use std::process::Command;
use std::time::Duration;

#[cfg(windows)]
fn has_proxy_environment() -> bool {
    ["HTTPS_PROXY", "ALL_PROXY", "https_proxy", "all_proxy"]
        .iter()
        .any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()))
}

#[cfg(windows)]
fn parse_windows_proxy(raw: &str) -> Option<String> {
    let (value, scheme) = if raw.contains('=') {
        let entries = raw
            .split(';')
            .filter_map(|entry| entry.trim().split_once('='));
        let mut https = None;
        let mut socks = None;
        for (scheme, address) in entries {
            if scheme.eq_ignore_ascii_case("https") {
                https = Some(address.trim()).filter(|value| !value.is_empty());
            } else if scheme.eq_ignore_ascii_case("socks") {
                socks = Some(address.trim()).filter(|value| !value.is_empty());
            }
        }
        if let Some(address) = https {
            (address, "http")
        } else {
            (socks?, "socks5h")
        }
    } else {
        (raw.trim(), "http")
    };
    if value.is_empty() {
        return None;
    }
    if value.contains("://") {
        return Some(value.to_owned());
    }
    Some(format!("{scheme}://{value}"))
}

#[cfg(windows)]
fn windows_proxy() -> Result<Option<String>, String> {
    use std::io::ErrorKind;
    use winreg::{enums::HKEY_CURRENT_USER, RegKey};

    let settings = match RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey("Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings")
    {
        Ok(settings) => settings,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("读取 Windows 代理设置失败: {error}")),
    };
    let enabled: u32 = settings.get_value("ProxyEnable").unwrap_or(0);
    if enabled == 0 {
        return Ok(None);
    }
    let server: String = settings.get_value("ProxyServer").unwrap_or_default();
    Ok(parse_windows_proxy(&server))
}

pub fn apply_proxy_to_command(command: &mut Command) -> Result<(), String> {
    #[cfg(windows)]
    if !has_proxy_environment() {
        if let Some(proxy) = windows_proxy()? {
            command
                .env("HTTPS_PROXY", proxy.clone())
                .env("HTTP_PROXY", proxy);
        }
    }
    #[cfg(not(windows))]
    let _ = command;
    Ok(())
}

pub fn download_client() -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder().timeout(Duration::from_secs(1800));
    #[cfg(windows)]
    if !has_proxy_environment() {
        if let Some(address) = windows_proxy()? {
            let proxy = reqwest::Proxy::https(&address)
                .map_err(|error| format!("Windows 代理地址无效: {error}"))?;
            let proxy = if let Some(exclusions) = reqwest::NoProxy::from_env() {
                proxy.no_proxy(Some(exclusions))
            } else {
                proxy
            };
            builder = builder.proxy(proxy);
        }
    }
    builder.build().map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use super::parse_windows_proxy;

    #[cfg(windows)]
    #[test]
    fn reads_global_and_protocol_proxy_formats() {
        assert_eq!(
            parse_windows_proxy("127.0.0.1:3067"),
            Some("http://127.0.0.1:3067".into())
        );
        assert_eq!(
            parse_windows_proxy("http=localhost:8000;https=localhost:9000"),
            Some("http://localhost:9000".into())
        );
        assert_eq!(
            parse_windows_proxy("socks=localhost:1080"),
            Some("socks5h://localhost:1080".into())
        );
        assert_eq!(parse_windows_proxy("http=localhost:8000"), None);
    }

    #[test]
    #[ignore = "requires access to GitHub Releases"]
    fn release_connection_uses_configured_proxy() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let response = super::download_client()
                .unwrap()
                .head("https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.33/crispasr-windows-x86_64-cuda13.zip")
                .send()
                .await
                .unwrap();
            assert!(response.status().is_success());
        });
    }
}
