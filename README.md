# 字幕工作台（最小闭环）

导入音视频，使用本地 CrispASR + Whisper/Qwen3-ASR 或 OpenVINO + Qwen3-ASR 转写，校对时间与文字；可用 llama.cpp 和 HY-MT 翻译 GGUF 逐条翻译字幕，分别导出原文或译文 SRT。
界面和任务逻辑是新实现。独立 Rust SDK 位于 `crates/model-sdk/`，当前桌面应用是它的第一个调用方。OpenVINO worker 复用了 STTMiniTool 的 MIT 许可 NumPy 处理器和小型预处理文件，许可文本见 `crates/model-sdk/workers/openvino/LICENSE-STTMiniTool`。

## 运行

开发环境：Node.js 20+、Rust stable、Tauri 2 所需系统构建工具。FFmpeg 可从资源页按需安装，也可加入 PATH 或指定本机路径。

```powershell
npm install
npm run tauri dev
```

当前界面的「资源与设备」页可查看系统自检、选择 CrispASR 或 OpenVINO 运行时与对应模型、
下载或登记本机文件，以及指定 FFmpeg 路径。字幕翻译区展示 llama.cpp CPU、Vulkan 或 CUDA 版本，支持下载各自运行时与内置 HY-MT1.5-1.8B Q4_K_M GGUF，也可按该内置模型 ID 登记已有权重；任意翻译模型导入暂不支持。「转写测试」页可选择目标语言并翻译已转写的字幕，原文和译文分别编辑与导出。
模型、推理引擎和转写临时文件存放在 EXE 旁的 `resources/` 目录，
不随界面安装包分发；请将程序放在可写目录，不要放在受保护的系统安装目录。
macOS 使用 `.app` 所在目录旁的 `resources/`。旧版资源 ID 和 SDK 记录不自动迁移，
本地权重可按当前内置模型 ID 重新登记；WebView 缓存仍由系统管理。
中国大陆网络可打开「镜像源」下载 Whisper 或 Qwen3-ASR 模型。
下载器优先使用 `HTTPS_PROXY` / `ALL_PROXY` 环境变量；Windows 上未设置这些变量时，
会读取系统手动代理设置。PAC 自动配置未接入，可通过环境变量指定代理。

转写路径：音视频经 FFmpeg 转为 16 kHz 单声道 WAV，CrispASR 生成 SRT，
SDK 解析为统一的 `Segment { start, end, text }`，编辑后重新写出 SRT。
Qwen3 GGUF 可按需加载 Qwen3 ForcedAligner，得到字级时间轴再分行；关闭时间轴对齐时返回一段完整转录文本，不下载对齐器。
识别语言可自动检测或从 15 种常用语言中指定，SDK 通过 `TranscriptionOptions` 接收每次转写的设置，旧接口默认开启对齐并自动识别。
OpenVINO 路径由独立 Python worker 运行 Qwen3-ASR 0.6B INT8；默认先用 Silero VAD 分段，保留原始音频偏移。它目前按句长估算字幕内时间，不提供字级对齐。Silero VAD 权重首次使用时按需下载并校验 SHA-256，已有本地权重继续使用。许可见 `crates/model-sdk/workers/openvino/LICENSE-Silero`。
100 MB 以内的素材可在窗口中预览；较大的素材仍可转写与导出。

## 架构边界

- `src/`：桌面界面、字幕校对、进度和错误展示。
- `crates/model-sdk/`：无 Tauri 依赖的资源目录、下载安装、本地登记、兼容性检查、任务控制及 CrispASR/OpenVINO 转写适配。
- `src-tauri/src/lib.rs`：Tauri 命令、应用选择及进度事件适配。

当前自动下载的 CrispASR 运行时固定为上游 `v0.8.33` 的 Windows Vulkan、CUDA 12、
CUDA 13、CPU 和 CPU 兼容构建。模型目录包含 Whisper Tiny、Base、Small、Medium、
Large V3 Turbo 和 Large V2；每个下载项均有固定大小与 SHA-256 校验。
自动安装引擎目前只支持 Windows x86-64；macOS 可手动登记 OpenVINO Python 环境，
但 macOS 构建与转写尚未实测。

## 后端资源接口

Tauri 命令 `get_resource_status` 返回系统自检、引擎和模型清单、每个引擎兼容的模型 ID、已安装路径、适用性、
推荐引擎、当前选择及 `ready` 状态。显卡由 Windows CIM 枚举，NVIDIA 驱动版本由
`nvidia-smi` 读取，并同时检查 GPU 算力。CUDA 12 至少需要驱动 528、算力 5.0；
CUDA 13 至少需要驱动 580、算力 7.5。驱动或算力未知时推荐 Vulkan 或 CPU。
已安装的 CrispASR 还会运行设备诊断；所选引擎和模型首次使用前会执行短音频推理。
诊断失败时状态显示原因并推荐其他适用引擎，下载的新引擎通过验证后才会替换旧版。
状态中还包括资源目录；旧资源不自动迁移，可按内置 ID 重新登记。
FFmpeg 可从「共享组件」下载到 `resources/tools/ffmpeg/`，安装后自动选用；也可通过
`set_ffmpeg_path({ path })` 登记本机路径，传入 `null` 恢复 PATH。自动安装支持 Windows x86-64、macOS x86-64/ARM64 和 Linux x86-64/ARM64，下载固定的 `imageio-ffmpeg 0.6.0` 平台包并校验 SHA-256；当前仅在 Windows 实测。

- `download_engine_variant({ engineId })`、`download_model_variant({ modelId, mirror })`：按 ID 下载，不修改当前选择。
- llama.cpp 的 CPU、Vulkan、CUDA 变体均使用 `download_engine_variant({ engineId })` 下载；`download_model_variant` 可使用模型 ID `hy-mt-1.5-1.8b-q4`。
- `register_local_engine({ engineId, path })`、`register_local_model({ modelId, path })`：登记本机已有文件；模型登记时验证完整 SHA-256。
- `select_resources({ engineId, modelId })`、`select_translation_engine({ engineId })`、`select_translation_model({ modelId })`：校验兼容性并保存选择，允许先选后装；返回新的资源状态。
- `transcribe_selected({ mediaPath, alignTimestamps?, language? })`：检查所选资源是否就绪，再按本次选项执行转写；语言使用 `zh`、`en` 等代码，省略时自动识别。
- `register_translation_model({ path })`：旧命令返回不支持任意模型导入；内置 HY-MT 权重使用 `register_local_model` 登记。
- `translate_subtitles({ segments, modelId, targetLanguage })`：用已登记的 llama.cpp 模型逐条翻译，返回保持原时间轴的字幕；可用现有取消命令停止。

旧的 `get_environment`、`download_engine`、`download_model` 命令仍存在；直接路径 `transcribe` 命令明确返回不支持，请使用按 ID 选择的 `transcribe_selected`。
旧版下载命令会同步更新选择；旧引擎 ID 与无版本的 SDK 资源记录不自动迁移。
新增资源按引擎 ID 分目录保存，切换时不删除其他版本。OpenVINO 自动安装目前支持 Windows x86-64：SDK 先下载 `uv` 到 `resources/engines/openvino/`，再安装便携 Python 3.12、OpenVINO、ONNX Runtime 和 NumPy。Qwen3-ASR 模型下载到 `resources/models/openvino/qwen3_asr_int8/`，三个大权重按上游 SHA-256 校验；已安装资源也可手动登记。worker 代码与小型预处理文件由 SDK 提供，Silero VAD ONNX 权重按需下载到 `resources/workers/openvino/`。
OpenVINO 需要 CPU 推理，模型加载和首次编译可能较慢；已在 Windows 上用本机完整 Qwen 权重通过一段短音频的 SDK 转写测试，长音频与 macOS 尚未验收。没有权重时资源状态不会显示为就绪。
macOS 可登记已装好这些 Python 依赖的 `python`/`python3` 和模型目录；自动安装与 macOS 实机转写尚未验证。

## 阶段验收（2026-10-12）

Windows 上已验证 CrispASR + Qwen3-ASR 1.7B Q4：同一 90 秒视频关闭对齐返回一段全文，开启对齐得到 49 条字幕；SDK Rust 测试和 OpenVINO worker 单元测试通过。
2026-10-13 使用已有的 HY-MT1.5-1.8B Q4 GGUF 和 llama-server.exe 实测两条中译英字幕，译文与原文时间轴一致；桌面界面中的完整操作流程仍需实机验收。
跨平台构建和 macOS 推理尚未验证；OpenVINO 仍使用估算的句级时间，CrispASR 引擎压缩包尚无固定哈希。
开发资源和可直接运行的 Release EXE 位于仓库外的 `../LLM-local-runtime/`（`resources/` 与 EXE 同级），旧版 EXE 保存在其 `portable-archive/`。重新执行 `npm run tauri dev` 会生成新的 EXE 和空资源目录；可在「资源与设备」页登记仓库外已有的引擎与模型文件。

## 下一步

为 macOS 接入支持 Metal 的本地引擎；
给运行时下载增加 SHA-256 校验；云端模型接入保留在同一能力
边界内，不影响字幕编辑和导出。

## 上游资源

- [CrispASR](https://github.com/CrispStrobe/CrispASR)
- [Whisper GGML 模型](https://huggingface.co/ggerganov/whisper.cpp)
- [FFmpeg](https://ffmpeg.org/)
- [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/)（按需下载的平台二进制来源）
- STTMiniTool（OpenVINO 处理器参考，MIT；项目副本许可见上述路径）
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)

分发产品时需分别核对推理引擎、模型权重和 FFmpeg 的许可证与分发要求。
