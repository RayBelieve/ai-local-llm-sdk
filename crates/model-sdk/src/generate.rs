use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use serde_json::{json, Value};

use crate::{
    diagnostics,
    process::hidden_command,
    resources::Selection,
    sdk::{ModelSdk, LLAMA_ENGINE_ID, TEXT_GENERATION},
    task::TaskControl,
};

pub(crate) struct LlamaServer {
    child: Child,
    engine: PathBuf,
    model: PathBuf,
    base: String,
    log: PathBuf,
    context: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerationOptions {
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            seed: None,
        }
    }
}

impl GenerationOptions {
    fn validate(&self) -> Result<(), String> {
        if !(1..=1024).contains(&self.max_tokens)
            || self
                .temperature
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            || self.top_p.is_some_and(|value| {
                !value.is_finite() || !(0.0..=1.0).contains(&value) || value == 0.0
            })
            || self
                .min_p
                .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
            || self
                .repeat_penalty
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err("生成参数无效：max_tokens 需为 1-1024，temperature >= 0，top_p 为 (0,1]，min_p 为 [0,1]，repeat_penalty > 0。".into());
        }
        Ok(())
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl ModelSdk {
    pub fn stop_text_generation(&self) -> Result<(), String> {
        *self
            .llama_server
            .lock()
            .map_err(|error| error.to_string())? = None;
        Ok(())
    }

    pub fn generate_text(
        &self,
        engine_id: &str,
        model_id: &str,
        prompt: &str,
        max_tokens: u32,
        task: &TaskControl,
    ) -> Result<String, String> {
        self.generate_chat(
            engine_id,
            model_id,
            &[ChatMessage {
                role: ChatRole::User,
                content: prompt.into(),
            }],
            &GenerationOptions {
                max_tokens,
                ..GenerationOptions::default()
            },
            task,
        )
    }

    pub fn generate_chat(
        &self,
        engine_id: &str,
        model_id: &str,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        task: &TaskControl,
    ) -> Result<String, String> {
        self.generate_chat_inner(engine_id, model_id, messages, options, task, None)
    }

    pub fn generate_chat_stream(
        &self,
        engine_id: &str,
        model_id: &str,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        task: &TaskControl,
        mut on_delta: impl FnMut(&str),
    ) -> Result<String, String> {
        self.generate_chat_inner(
            engine_id,
            model_id,
            messages,
            options,
            task,
            Some(&mut on_delta),
        )
    }

    fn generate_chat_inner(
        &self,
        engine_id: &str,
        model_id: &str,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        task: &TaskControl,
        on_delta: Option<&mut dyn FnMut(&str)>,
    ) -> Result<String, String> {
        if messages.is_empty()
            || messages
                .iter()
                .any(|message| message.content.trim().is_empty())
        {
            return Err("消息列表不能为空，且每条消息必须有内容。".into());
        }
        options.validate()?;
        if engine_id != LLAMA_ENGINE_ID {
            return Err(format!("不支持的文本生成引擎: {engine_id}"));
        }
        self.validate_pair_for(engine_id, model_id, TEXT_GENERATION)?;
        if !cfg!(all(windows, target_arch = "x86_64")) {
            return Err("llama.cpp 接入目前仅支持 Windows x86-64。".into());
        }
        let engine = self.engine_path(engine_id)?.ok_or("llama.cpp 尚未登记。")?;
        let model = self.model_path(model_id)?.ok_or("文本模型尚未登记。")?;
        task.check_cancelled()?;
        let mut active = self
            .llama_server
            .lock()
            .map_err(|error| error.to_string())?;
        task.check_cancelled()?;
        let stale = active.as_mut().is_some_and(|server| {
            server.engine != engine || server.model != model || !server.healthy()
        });
        if stale {
            *active = None;
        }
        if active.is_none() {
            let check = diagnostics::system_check(&Selection::default());
            let model_bytes = fs::metadata(&model)
                .map_err(|error| error.to_string())?
                .len();
            let cpu_context = context_size(check.available_memory_bytes, model_bytes);
            let cuda = check.nvidia_free_vram_bytes.is_some_and(|free| {
                engine
                    .parent()
                    .is_some_and(|dir| dir.join("ggml-cuda.dll").is_file())
                    && free > (model_bytes.saturating_mul(3) / 2).saturating_add(1024 * 1024 * 1024)
            });
            let vulkan = engine
                .parent()
                .is_some_and(|dir| dir.join("ggml-vulkan.dll").is_file())
                && !check.gpus.is_empty();
            let gpu = cuda || vulkan;
            let context = if cuda {
                context_size(check.nvidia_free_vram_bytes, model_bytes)
            } else {
                cpu_context
            };
            let started = start_server(self.root(), &engine, &model, context, gpu, task);
            *active = Some(match started {
                Ok(server) => server,
                Err(error) if gpu && task.check_cancelled().is_ok() => {
                    task.progress("generate", 0, "显卡加载失败，改用 CPU");
                    start_server(self.root(), &engine, &model, cpu_context, false, task).map_err(
                        |cpu_error| format!("显卡加载失败: {error}; CPU 加载失败: {cpu_error}"),
                    )?
                }
                Err(error) => return Err(error),
            });
        }
        let server = active.as_mut().expect("server started");
        task.progress("generate", 30, "正在生成文本");
        let result = server.generate(messages, options, task, on_delta);
        if result.is_err() {
            *active = None;
        }
        if result.is_ok() {
            task.progress("generate", 100, "文本生成完成");
        }
        result
    }
}

fn context_size(available_memory: Option<u64>, model_bytes: u64) -> u32 {
    let remaining = available_memory
        .unwrap_or(0)
        .saturating_sub(model_bytes.saturating_mul(2));
    if remaining >= 16 * 1024 * 1024 * 1024 {
        16384
    } else if remaining >= 8 * 1024 * 1024 * 1024 {
        8192
    } else if remaining >= 4 * 1024 * 1024 * 1024 {
        4096
    } else {
        2048
    }
}

fn server_error(child: &mut Child, log: &Path) -> String {
    let status = child.try_wait().ok().flatten();
    let detail = fs::read_to_string(log).unwrap_or_default();
    let tail = detail.chars().rev().take(800).collect::<String>();
    format!(
        "llama.cpp 服务异常{}: {}",
        status.map_or(String::new(), |value| format!(" ({value})")),
        tail.chars().rev().collect::<String>()
    )
}

enum ServerEvent {
    Delta(String),
    Done(Result<String, String>),
}

fn completion_body(messages: &[ChatMessage], options: &GenerationOptions, stream: bool) -> Value {
    let mut body = serde_json::to_value(options).expect("generation options serialize");
    body["messages"] = json!(messages);
    body["stream"] = json!(stream);
    body
}

fn stream_event(
    data: &str,
    sender: &mpsc::SyncSender<ServerEvent>,
    output: &mut String,
) -> Result<bool, String> {
    if data == "[DONE]" {
        return Ok(true);
    }
    let value: Value =
        serde_json::from_str(data).map_err(|error| format!("引擎流式响应无效: {error}"))?;
    if let Some(error) = value.get("error") {
        return Err(format!("文本生成失败: {error}"));
    }
    let choice = value["choices"]
        .as_array()
        .and_then(|items| items.first())
        .ok_or("引擎流式响应缺少 choices。")?;
    if choice["finish_reason"] == "length" {
        return Err("生成文本达到输出长度上限，结果可能不完整。".into());
    }
    if let Some(delta) = choice["delta"]["content"].as_str() {
        output.push_str(delta);
        sender
            .send(ServerEvent::Delta(delta.into()))
            .map_err(|_| "文本生成请求已取消。".to_string())?;
    }
    Ok(false)
}

fn stream_response(
    response: impl std::io::Read,
    sender: &mpsc::SyncSender<ServerEvent>,
) -> Result<String, String> {
    let mut reader = BufReader::new(response);
    let mut line = String::new();
    let mut data = String::new();
    let mut output = String::new();
    loop {
        line.clear();
        let count = reader
            .read_line(&mut line)
            .map_err(|error| format!("读取流式响应失败: {error}"))?;
        if count == 0 {
            return Err("流式响应在结束标记前中断。".into());
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            if !data.is_empty() {
                if stream_event(&data, sender, &mut output)? {
                    return if output.trim().is_empty() {
                        Err("引擎未生成文本。".into())
                    } else {
                        Ok(output)
                    };
                }
                data.clear();
            }
        } else if let Some(value) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(value.trim_start());
        }
    }
}

fn request(
    client: reqwest::blocking::Client,
    base: String,
    messages: Vec<ChatMessage>,
    options: GenerationOptions,
    context: u32,
    stream: bool,
    sender: &mpsc::SyncSender<ServerEvent>,
) -> Result<String, String> {
    let content = messages
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let response = client
        .post(format!("{base}/tokenize"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(json!({"content": content, "add_special": true}).to_string())
        .send()
        .map_err(|error| format!("无法计算提示词长度: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("引擎分词失败: {}", response.status()));
    }
    let tokens: Value = serde_json::from_str(&response.text().map_err(|error| error.to_string())?)
        .map_err(|error| format!("引擎分词结果无效: {error}"))?;
    let input_tokens = tokens["tokens"]
        .as_array()
        .ok_or("引擎分词结果缺少 tokens。")?
        .len();
    // Reserve the existing template margin plus one role margin per additional message.
    if input_tokens
        .saturating_add(messages.len().saturating_sub(1).saturating_mul(32))
        .saturating_add(options.max_tokens as usize)
        .saturating_add(128)
        > context as usize
    {
        return Err(format!("提示词约 {input_tokens} token，加上输出预算 {} token 超过当前 {context} token 上下文；请缩短提示词或输出长度。", options.max_tokens));
    }
    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(completion_body(&messages, &options, stream).to_string())
        .send()
        .map_err(|error| format!("文本生成请求失败: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().map_err(|error| error.to_string())?;
        return Err(format!(
            "文本生成失败 ({status}): {}",
            body.chars().take(500).collect::<String>()
        ));
    }
    if stream {
        return stream_response(response, sender);
    }
    let body = response.text().map_err(|error| error.to_string())?;
    let value: Value =
        serde_json::from_str(&body).map_err(|error| format!("引擎响应无效: {error}"))?;
    if value["choices"][0]["finish_reason"] == "length" {
        return Err("生成文本达到输出长度上限，结果可能不完整。".into());
    }
    let output = value["choices"][0]["message"]["content"]
        .as_str()
        .ok_or("引擎响应缺少生成文本。")?;
    if output.trim().is_empty() {
        return Err("引擎未生成文本。".into());
    }
    Ok(output.to_owned())
}

impl LlamaServer {
    fn healthy(&mut self) -> bool {
        self.child.try_wait().is_ok_and(|status| status.is_none())
            && reqwest::blocking::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(1))
                .build()
                .is_ok_and(|client| {
                    client
                        .get(format!("{}/health", self.base))
                        .send()
                        .is_ok_and(|response| response.status().is_success())
                })
    }

    fn generate(
        &mut self,
        messages: &[ChatMessage],
        options: &GenerationOptions,
        task: &TaskControl,
        mut on_delta: Option<&mut dyn FnMut(&str)>,
    ) -> Result<String, String> {
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|error| error.to_string())?;
        let (sender, receiver) = mpsc::sync_channel(16);
        let base = self.base.clone();
        let messages = messages.to_vec();
        let options = options.clone();
        let context = self.context;
        let stream = on_delta.is_some();
        thread::spawn(move || {
            let result = request(client, base, messages, options, context, stream, &sender);
            let _ = sender.send(ServerEvent::Done(result));
        });
        let started = Instant::now();
        loop {
            task.check_cancelled()?;
            match receiver.recv_timeout(Duration::from_millis(250)) {
                Ok(ServerEvent::Delta(delta)) => {
                    if let Some(callback) = on_delta.as_mut() {
                        callback(&delta);
                    }
                }
                Ok(ServerEvent::Done(result)) => return result,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("文本生成请求意外中断。".into())
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            if self
                .child
                .try_wait()
                .map_err(|error| error.to_string())?
                .is_some()
            {
                return Err(server_error(&mut self.child, &self.log));
            }
            if started.elapsed() >= Duration::from_secs(300) {
                return Err("文本生成超时。".into());
            }
        }
    }
}

fn start_server(
    root: &Path,
    engine: &Path,
    model: &Path,
    context: u32,
    gpu: bool,
    task: &TaskControl,
) -> Result<LlamaServer, String> {
    task.check_cancelled()?;
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|error| error.to_string())?;
    let port = listener
        .local_addr()
        .map_err(|error| error.to_string())?
        .port();
    drop(listener);
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let job_dir = root
        .join("jobs")
        .join(format!("llama-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&job_dir).map_err(|error| error.to_string())?;
    let log = job_dir.join("server.log");
    let mut command = hidden_command(engine);
    command
        .arg("--model")
        .arg(model)
        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
        .args([
            "--ctx-size",
            &context.to_string(),
            "--gpu-layers",
            if gpu { "-1" } else { "0" },
            "--no-ui",
        ])
        .current_dir(engine.parent().ok_or("引擎路径无效。")?)
        .stdout(Stdio::null())
        .stderr(Stdio::from(
            File::create(&log).map_err(|error| error.to_string())?,
        ));
    task.progress("generate", 0, "正在加载文本模型");
    let child = command
        .spawn()
        .map_err(|error| format!("启动 llama.cpp 失败: {error}"))?;
    let mut server = LlamaServer {
        child,
        engine: engine.to_owned(),
        model: model.to_owned(),
        base: format!("http://127.0.0.1:{port}"),
        log,
        context,
    };
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(1))
        .build()
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    loop {
        task.check_cancelled()?;
        if server
            .child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some()
        {
            return Err(server_error(&mut server.child, &server.log));
        }
        if client
            .get(format!("{}/health", server.base))
            .send()
            .is_ok_and(|response| response.status().is_success())
        {
            return Ok(server);
        }
        if started.elapsed() >= Duration::from_secs(180) {
            return Err(format!(
                "llama.cpp 模型加载超时。{}",
                server_error(&mut server.child, &server.log)
            ));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn context_uses_available_memory_conservatively() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(context_size(None, gib), 2048);
        assert_eq!(context_size(Some(6 * gib), gib), 4096);
        assert_eq!(context_size(Some(12 * gib), gib), 8192);
        assert_eq!(context_size(Some(20 * gib), gib), 16384);
    }

    #[test]
    fn chat_request_serializes_roles_and_sampling_options() {
        let messages = [
            ChatMessage {
                role: ChatRole::System,
                content: "Be concise".into(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hello".into(),
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "Hi".into(),
            },
        ];
        let options = GenerationOptions {
            max_tokens: 64,
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            repeat_penalty: Some(1.1),
            seed: Some(42),
        };
        assert!(options.validate().is_ok());
        let body = completion_body(&messages, &options, true);
        assert_eq!(
            body["messages"][0],
            json!({"role": "system", "content": "Be concise"})
        );
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][2]["role"], "assistant");
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["seed"], 42);
        assert_eq!(body["stream"], true);
        assert!(GenerationOptions {
            top_p: Some(f32::NAN),
            ..options
        }
        .validate()
        .is_err());
    }

    #[test]
    fn stream_collects_deltas_and_rejects_truncated_response() {
        let (sender, receiver) = mpsc::sync_channel(16);
        let response = concat!(
            ": keepalive\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\r\n\r\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\r\n\r\n",
            "data: [DONE]\r\n\r\n"
        );
        assert_eq!(
            stream_response(response.as_bytes(), &sender).unwrap(),
            "Hello world"
        );
        let deltas = receiver
            .try_iter()
            .filter_map(|event| match event {
                ServerEvent::Delta(delta) => Some(delta),
                ServerEvent::Done(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(deltas, ["Hello", " world"]);
        assert!(stream_response("data: {}\n\n".as_bytes(), &sender).is_err());
        assert!(stream_response("data: [DONE]".as_bytes(), &sender).is_err());
        assert!(stream_response(
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n".as_bytes(),
            &sender,
        )
        .unwrap_err()
        .contains("输出长度上限"));
    }

    #[test]
    fn streaming_request_sends_chat_messages_and_sampling_options() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            for (index, path) in [
                "/tokenize",
                "/v1/chat/completions",
                "/tokenize",
                "/v1/chat/completions",
            ]
            .into_iter()
            .enumerate()
            {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                assert!(line.starts_with(&format!("POST {path} ")));
                let mut length = 0;
                loop {
                    line.clear();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse().unwrap();
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let body: Value = serde_json::from_slice(&body).unwrap();
                let reply = if path == "/tokenize" {
                    assert_eq!(body["content"], "Be brief\nHello");
                    "{\"tokens\":[1,2]}"
                } else {
                    assert_eq!(body["messages"][0]["role"], "system");
                    assert_eq!(body["messages"][1]["role"], "user");
                    assert_eq!(body["temperature"], 0.5);
                    assert_eq!(body["stream"], index == 1);
                    if index == 1 {
                        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\n"
                    } else {
                        "{\"choices\":[{\"message\":{\"content\":\"partial\"},\"finish_reason\":\"length\"}]}"
                    }
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                )
                .unwrap();
            }
        });
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .build()
            .unwrap();
        let (sender, receiver) = mpsc::sync_channel(16);
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "Be brief".into(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Hello".into(),
            },
        ];
        let options = GenerationOptions {
            temperature: Some(0.5),
            ..GenerationOptions::default()
        };
        assert_eq!(
            request(
                client.clone(),
                base.clone(),
                messages.clone(),
                options.clone(),
                2048,
                true,
                &sender
            )
            .unwrap(),
            "Hi"
        );
        assert!(matches!(receiver.try_recv(), Ok(ServerEvent::Delta(delta)) if delta == "Hi"));
        assert!(
            request(client, base, messages, options, 2048, false, &sender)
                .unwrap_err()
                .contains("输出长度上限")
        );
        server.join().unwrap();
    }
}
