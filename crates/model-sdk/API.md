# model-sdk 接口与兼容性模型

应用显式传入引擎 ID、内置模型 ID 和任务类型。模型 ID 对应目录中的一份具体权重文件或目录。SDK 不自动选引擎，也不接受任意自定义模型描述。

## 目录类型

- `TaskKind::Transcription`：语音转写；`TaskKind::TextGeneration`：文本生成；`TaskKind::ObjectDetection`：图片目标检测；`TaskKind::ImageTagging`：图片打标；`TaskKind::SpeechSynthesis`：语音合成。
- `ArtifactFormat::Ggml`、`Gguf`、`OpenVinoIr`、`Onnx`：权重的文件格式，不编码 provider。
- `ModelProfile::Whisper`、`Qwen3Asr17b`、`Qwen3Asr06b`、`HyMt15`、`YoloV8Detect`、`Yolo11Detect`、`Wd14V2`、`Wd14V3`、`KokoroZh`：模型架构及推理协议。格式相同不代表适配器一定支持。
- `ModelDescriptor`：`id`、`label`、`profile`、`artifact_format`、`task`。
- `InstalledModel`：`descriptor`、已验证的 `path`、`size_bytes`、`downloadable`；没有唯一 provider 字段。
- `EngineInfo`：`id`、`family`、`variant`、`label`、`version`、`download_mb`、`path`、`suitable`、`reason`、`tasks`。一个运行时可以支持多个任务。

现有模型为 15 个单文件模型、1 个 OpenVINO IR 目录模型、9 个 WD-14 双文件目录模型及 1 个 Kokoro 三文件目录模型。Whisper GGML 共 6 份；Qwen3-ASR 1.7B GGUF 共 4 份；HY-MT1.5 GGUF 共 1 份；Qwen3-ASR 0.6B INT8 OpenVINO IR 共 1 份；YOLOv8/YOLO11 检测 ONNX 各有 Nano、Small 两份；WD-14 提供 5 款 v2 和 4 款 v3。

## 引擎 ID

- CrispASR：`crispasr-vulkan`、`crispasr-cuda12`、`crispasr-cuda13`、`crispasr-cpu`、`crispasr-cpu-legacy`。
- OpenVINO：`openvino-cpu`。
- llama.cpp：`llama-cpu`、`llama-vulkan`、`llama-cuda12`、`llama-cuda13`。
- ONNX Runtime：`onnxruntime-cpu`。

每个 ID 代表可独立安装的具体运行时。llama.cpp 变体保存在 `engines/<engine_id>/` 下，可以并存。引擎家族与变体也作为 `EngineInfo` 的独立字段返回。

内部静态适配器注册表位于 `src/engines.rs`；其中的引擎家族模块负责支持规则、适用性、路径解析、运行探测和执行。目录元数据位于 `src/catalog.rs` 与 `src/resources.rs`。安装实现分别位于 `src/install/crispasr.rs`、`openvino.rs`、`llama.rs`、`onnxruntime.rs`；`src/infer.rs` 负责音频转写，`src/generate.rs` 负责文本生成，`src/detect.rs` 定义视觉检测接口。SDK 没有外部适配器注册或通用 JSON 推理接口。

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
- `detect_objects(engine_id, model_id, image, &DetectionOptions, task) -> DetectionResult`：单张图片目标检测。`DetectionOptions` 包含置信度阈值（默认 0.25）、NMS IoU 阈值（默认 0.45）和最大目标数（默认 100）。结果含图片宽高，以及类别 ID/名称、置信度与原图像素坐标 `x1,y1,x2,y2`。
- `tag_image(engine_id, model_id, image, task) -> TagResult`：单张图片 WD-14 打标。`ratings` 返回四项评级；`tags` 返回 CSV 中其余全部标签。每项包含原始英文 `name`、数值 `category`、未经阈值筛选的 `score`。SDK 不翻译、不拼接提示词，也不筛选或排序标签。
- `synthesize_speech(engine_id, model_id, text, speed, output, task) -> Result<(), String>`：Kokoro 中文语音合成，输出 24 kHz 单声道 PCM WAV。`speed` 范围为 0.5 到 2.0；输出路径须以 `.wav` 结尾且不能已存在，避免覆盖现有文件。

Kokoro 模型 ID 为 `kokoro-82m-v1.1-zh`，与 `onnxruntime-cpu` 配对。下载固定提交中的 FP32 `model.onnx`、`tokenizer.json` 和中文音色 `zf_001.bin`，离线登记时传包含这三个文件的目录。运行时额外安装 `misaki[zh]` 完成中文音素化；当前接口使用 `zf_001` 音色。模型文件按大小和 SHA-256 校验。

`TaskControl` 提供进度回调和取消；`cancel()` 会终止正在运行的子进程，并中断进行中的下载连接。音频处理、模型探测和生成调用也会检查取消状态。直接传入任意引擎及模型路径进行推理的公开入口已移除。

OpenVINO worker、VAD 与 ForcedAligner 的安装和离线导入辅助接口仍可由 `ModelSdk` 使用；FFmpeg 可用 `install_ffmpeg(task)` 安装。真实推理需本机资源，见忽略的 `tests/smoke.rs`。

首批检测模型 ID 为 `yolov8n-det-fp32`、`yolov8s-det-fp32`、`yolo11n-det-fp32`、`yolo11s-det-fp32`。权重来自 EdgeFirst Model Zoo 的固定提交，按大小和 SHA-256 验证，包含 COCO 80 类标签元数据；输入为 640×640 RGB FP32，输出为归一化检测框，SDK 执行 letterbox、坐标还原和按类别 NMS。`onnxruntime-cpu` 在 Windows x86-64 可自动安装独立的 Python/ONNX Runtime 环境，其他平台可登记已装齐 `onnxruntime`、`numpy`、`Pillow` 的 Python。模型标记为 AGPL-3.0，分发前需核对其许可证要求。桌面调试台提供独立的图片检测调用。

WD-14 模型 ID 与 SmilingWolf 仓库名相同：`wd-v1-4-moat-tagger-v2`、`wd-v1-4-swinv2-tagger-v2`、`wd-v1-4-convnext-tagger-v2`、`wd-v1-4-vit-tagger-v2`、`wd-v1-4-convnextv2-tagger-v2`、`wd-swinv2-tagger-v3`、`wd-convnext-tagger-v3`、`wd-vit-tagger-v3`、`wd-eva02-large-tagger-v3`。每款从固定提交下载 `model.onnx` 和 `selected_tags.csv`，两者都按大小与 SHA-256 验证；离线登记时传包含这两个文件的目录。共用 `onnxruntime-cpu`，输入图片按 EXIF 方向处理，透明区域铺白、补方形后以 BGR FP32 推理。桌面调试台展示评级、最高分标签预览和完整原始 JSON。

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
