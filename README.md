# 字幕工作台（最小闭环）

导入音视频，使用本地 CrispASR + Whisper/Qwen3-ASR 或 OpenVINO + Qwen3-ASR 转写，校对时间与文字，导出 SRT。
界面和任务逻辑是新实现。独立 Rust SDK 位于 `crates/model-sdk/`，当前桌面应用是它的第一个调用方。OpenVINO worker 复用了 STTMiniTool 的 MIT 许可 NumPy 处理器和小型预处理文件，许可文本见 `crates/model-sdk/workers/openvino/LICENSE-STTMiniTool`。

## 运行

开发环境：Node.js 20+、Rust stable、Tauri 2 所需系统构建工具。FFmpeg 可从资源页按需安装，也可加入 PATH 或指定本机路径。

```powershell
npm install
npm run tauri dev
```

当前界面的「资源与设备」页可查看系统自检、选择 CrispASR 或 OpenVINO 运行时与对应模型、
下载或登记本机文件，以及指定 FFmpeg 路径。「转写测试」页使用当前已保存的资源选择
运行转写。也可以登记已有的 `crispasr.exe` 与目录内列出的 GGML Whisper 模型。
模型、推理引擎和转写临时文件存放在 EXE 旁的 `resources/` 目录，
不随界面安装包分发；请将程序放在可写目录，不要放在受保护的系统安装目录。
macOS 使用 `.app` 所在目录旁的 `resources/`。旧版已下载到用户本地数据目录的资源，
可在「资源与设备」页点击「迁移旧资源」搬到新目录；WebView 缓存仍由系统管理。
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
- `src-tauri/src/lib.rs`：Tauri 命令、应用选择、旧资源迁移及进度事件适配。

当前自动下载的 CrispASR 运行时固定为上游 `v0.8.33` 的 Windows Vulkan、CUDA 12、
CUDA 13、CPU 和 CPU 兼容构建。模型目录包含 Whisper Tiny、Base、Small、Medium、
Large V3 Turbo 和 Large V2；每个下载项均有固定大小与 SHA-256 校验。
自动安装引擎目前只支持 Windows x86-64；macOS 可手动登记 OpenVINO Python 环境，
但 macOS 构建与转写尚未实测。

## 后端资源接口

Tauri 命令 `get_resource_status` 返回系统自检、引擎和模型清单、已安装路径、适用性、
推荐引擎、当前选择及 `ready` 状态。显卡由 Windows CIM 枚举，NVIDIA 驱动版本由
`nvidia-smi` 读取；驱动版本未知时推荐 Vulkan。状态中还包括资源目录及旧资源迁移入口。
FFmpeg 可从「共享组件」下载到 `resources/tools/ffmpeg/`，安装后自动选用；也可通过
`set_ffmpeg_path({ path })` 登记本机路径，传入 `null` 恢复 PATH。自动安装支持 Windows x86-64、macOS x86-64/ARM64 和 Linux x86-64/ARM64，下载固定的 `imageio-ffmpeg 0.6.0` 平台包并校验 SHA-256；当前仅在 Windows 实测。

- `download_engine_variant({ engineId })`、`download_model_variant({ modelId, mirror })`：按 ID 下载，不修改当前选择。
- `register_local_engine({ engineId, path })`、`register_local_model({ modelId, path })`：登记本机已有文件；模型登记时验证完整 SHA-256。
- `select_resources({ engineId, modelId })`：保存选择，允许先选后装；返回新的资源状态。
- `transcribe_selected({ mediaPath, alignTimestamps?, language? })`：检查所选资源是否就绪，再按本次选项执行转写；语言使用 `zh`、`en` 等代码，省略时自动识别。

旧的 `get_environment`、`download_engine`、`download_model`、`transcribe` 命令保留兼容。
旧版下载命令会同步更新选择；原有 `engines/crispasr` 目录仍可识别。
新增资源按 provider 分目录保存，切换时不删除其他版本。OpenVINO 自动安装目前支持 Windows x86-64：SDK 先下载 `uv` 到 `resources/engines/openvino/`，再安装便携 Python 3.12、OpenVINO、ONNX Runtime 和 NumPy。Qwen3-ASR 模型下载到 `resources/models/openvino/qwen3_asr_int8/`，三个大权重按上游 SHA-256 校验；已安装资源也可手动登记。worker 代码与小型预处理文件由 SDK 提供，Silero VAD ONNX 权重按需下载到 `resources/workers/openvino/`。
OpenVINO 需要 CPU 推理，模型加载和首次编译可能较慢；已在 Windows 上用本机完整 Qwen 权重通过一段短音频的 SDK 转写测试，长音频与 macOS 尚未验收。没有权重时资源状态不会显示为就绪。
macOS 可登记已装好这些 Python 依赖的 `python`/`python3` 和模型目录；自动安装与 macOS 实机转写尚未验证。

## 阶段验收（2026-10-12）

Windows 上已验证 CrispASR + Qwen3-ASR 1.7B Q4：同一 90 秒视频关闭对齐返回一段全文，开启对齐得到 49 条字幕；SDK Rust 测试和 OpenVINO worker 单元测试通过。SDK 与 Tauri 分层可继续使用，目前仍只实现语音转写能力。
跨平台构建和 macOS 推理尚未验证；OpenVINO 仍使用估算的句级时间，CrispASR 引擎压缩包尚无固定哈希。
开发资源和现有调试程序已移到仓库外的 `../LLM-local-runtime/`（`resources/` 与 EXE 同级），旧版 EXE 保存在其 `portable-archive/`。该调试程序需配合 Vite 开发服务器运行。重新执行 `npm run tauri dev` 会生成新的 EXE 和空资源目录；可在「资源与设备」页登记仓库外已有的引擎与模型文件。

## 下一步

为 macOS 接入支持 Metal 的本地引擎；
给运行时下载增加 SHA-256 校验；再扩展翻译 provider。云端模型接入保留在同一能力
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
