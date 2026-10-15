# model-sdk 接口与兼容性模型

应用显式传入引擎 ID、内置模型 ID 和任务类型。模型 ID 对应目录中的一份具体权重文件或目录。SDK 不自动选引擎，也不接受任意自定义模型描述。

## 目录类型

- `TaskKind::Transcription`：语音转写；`TaskKind::TextGeneration`：文本生成。
- `ArtifactFormat::Ggml`、`Gguf`、`OpenVinoIr`：权重的文件格式，不编码 provider。
- `ModelProfile::Whisper`、`Qwen3Asr17b`、`Qwen3Asr06b`、`HyMt15`：模型架构及推理协议。格式相同不代表适配器一定支持。
- `ModelDescriptor`：`id`、`label`、`profile`、`artifact_format`、`task`。
- `InstalledModel`：`descriptor`、已验证的 `path`、`size_bytes`、`downloadable`；没有唯一 provider 字段。
- `EngineInfo`：`id`、`family`、`variant`、`label`、`version`、`download_mb`、`path`、`suitable`、`reason`、`task`。

现有模型为 11 个文件模型加 1 个 OpenVINO IR 目录模型。Whisper GGML 共 6 份；Qwen3-ASR 1.7B GGUF 共 4 份；HY-MT1.5 GGUF 共 1 份；Qwen3-ASR 0.6B INT8 OpenVINO IR 共 1 份。

## 引擎 ID

- CrispASR：`crispasr-vulkan`、`crispasr-cuda12`、`crispasr-cuda13`、`crispasr-cpu`、`crispasr-cpu-legacy`。
- OpenVINO：`openvino-cpu`。
- llama.cpp：`llama-cpu`、`llama-vulkan`、`llama-cuda12`、`llama-cuda13`。

每个 ID 代表可独立安装的具体运行时。llama.cpp 变体保存在 `engines/<engine_id>/` 下，可以并存。引擎家族与变体也作为 `EngineInfo` 的独立字段返回。

内部静态适配器注册表位于 `src/engines.rs`；其中的引擎家族模块负责支持规则、适用性、路径解析、运行探测和转写执行。目录元数据位于 `src/catalog.rs` 与 `src/resources.rs`。安装实现分别位于 `src/install/crispasr.rs`、`openvino.rs`、`llama.rs`；`src/infer.rs` 负责通用音频准备和任务分发，`src/generate.rs` 负责 llama.cpp 文本生成。SDK 没有外部适配器注册或通用 JSON 推理接口。

## ModelSdk

- `new(root) -> Result<ModelSdk, String>`：设置 SDK 资源根目录。
- `root() -> &Path`：获取规范化后的资源根目录。
- `engines() -> Result<Vec<EngineInfo>, String>`：列出全部运行时和本机适用性、路径。
- `models() -> Result<Vec<InstalledModel>, String>`：列出全部内置权重及验证后的本地路径。
- `engine_path(id) -> Result<Option<PathBuf>, String>` 与 `model_path(id) -> Result<Option<PathBuf>, String>`：按目录 ID 查询资源；未知 ID 返回错误。
- `validate_pair_for(engine_id, model_id, TaskKind) -> Result<(), String>`：只检查目录元数据和适配器支持规则，不要求安装资源。
- `check(engine_id, model_id, TaskKind) -> Result<Readiness, String>`：在静态兼容基础上检查适用性、安装路径、已验证权重和运行探测。已知但不兼容的组合返回 `ready: false`；未知 ID 返回错误。
- `download_engine(engine_id, task) -> Result<PathBuf, String>`：精确安装指定运行时，包括 llama.cpp 变体。
- `download_model(model_id, mirror, task) -> Result<PathBuf, String>`：按内置目录下载并验证权重。
- `register_local_engine(engine_id, path) -> Result<(), String>`：登记目录内运行时的本地绝对路径，并测试可运行性。
- `register_local_model(model_id, path) -> Result<(), String>`：只登记内置模型 ID；校验路径、文件大小与 SHA-256，OpenVINO IR 还检查目录结构和已知权重哈希。

资源记录写入 `root/sdk-resources.json`，格式包含 `version: 1`、`enginePaths` 和 `modelPaths`。SDK 不从应用的 `resources.json` 读取自身配置。遇到无版本的旧 SDK 记录或不支持的版本会报“版本不兼容”，不会丢弃、迁移或覆盖旧数据。旧资源 ID 不可直接用于新 API。

## 推理

- `transcribe(engine_id, model_id, media, ffmpeg_path, vad_enabled, task)`：语音转写，返回 `Vec<subtitle::Segment>`。
- `transcribe_with_options(..., &TranscriptionOptions, task)`：额外控制 `align_timestamps` 和 `language`。对齐默认开启；关闭时合并为完整文本段。Qwen 后端由 `ModelProfile` 决定，离线文件名不影响选择。
- `generate_text(engine_id, model_id, prompt, max_tokens, task)`：单条用户消息文本生成。
- `generate_chat(engine_id, model_id, messages, options, task)`：完整对话文本生成。
- `generate_chat_stream(engine_id, model_id, messages, options, task, on_delta)`：逐段回调，成功时也返回完整文本。
- `stop_text_generation()`：关闭当前 SDK 实例持有的 llama.cpp 服务。

`TaskControl` 提供进度回调和取消；下载、音频处理、模型探测和生成调用会检查取消状态。直接传入任意引擎及模型路径进行推理的公开入口已移除。

OpenVINO worker、VAD 与 ForcedAligner 的安装和离线导入辅助接口仍可由 `ModelSdk` 使用；FFmpeg 可用 `install_ffmpeg(task)` 安装。真实推理需本机资源，见忽略的 `tests/smoke.rs`。

## 示例

```rust
use std::path::Path;
use model_sdk::{ModelSdk, TaskKind, TranscriptionOptions, task::TaskControl};

let sdk = ModelSdk::new("app-resources")?;
sdk.validate_pair_for("crispasr-cpu", "base", TaskKind::Transcription)?;
let ready = sdk.check("crispasr-cpu", "base", TaskKind::Transcription)?;
if ready.ready {
    let segments = sdk.transcribe_with_options(
        "crispasr-cpu", "base", Path::new("audio.mp3"), Path::new("ffmpeg"), false,
        &TranscriptionOptions::default(), &TaskControl::default(),
    )?;
    println!("{} segments", segments.len());
}
```
