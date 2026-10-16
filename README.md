# Model SDK 调试台

这个仓库以 `crates/model-sdk/` 为核心。Tauri 桌面端仅用于逐项验证 SDK：查看内置目录、安装或登记资源、检查引擎与模型兼容性，然后独立调用转写、文本生成和图片目标检测接口。桌面端不组合这些任务，也不保存工作流选择。

## 运行

需要 Node.js 20+、Rust stable 和 Tauri 2 的系统构建依赖。

```powershell
npm install
npm run tauri dev
```

资源保存在可执行文件旁的 `resources/` 目录。运行程序的位置必须可写；已有 SDK 资源记录继续使用，旧应用的 `resources.json` 不再读取。旧版资源 ID 与无版本 SDK 记录不会自动迁移，可按当前内置 ID 重新登记本地权重。SDK 目录与接口见 [API.md](crates/model-sdk/API.md)。

## 调试页

- **资源与兼容性**：按任务选择具体引擎和内置模型，分别调用 `validate_pair_for` 与 `check`，查看静态兼容及运行就绪；可下载或登记运行时、模型、FFmpeg 与转写对齐模型。
- **语音转写**：选择音视频，分别调用 `transcribe` 和 `transcribe_with_options`；后者可设置识别语言与时间轴对齐，两种接口均可设置 VAD 和 FFmpeg。查看分段、时间戳与原始返回值。
- **文本生成**：分别调用 `generate_text`、`generate_chat`、`generate_chat_stream`；对话模式可编辑消息角色和内容，流式模式即时显示增量。
- **目标检测**：选择图片和 YOLO 权重，调用 `detect_objects`；在原图上查看检测框，并查看类别、置信度、坐标和原始返回值。

任务进度与取消均通过 `TaskControl`。界面选择只作用于当前调用，不改写 SDK 资源目录之外的应用配置。FFmpeg 可使用 PATH、手动路径或 SDK 安装版本。

中国大陆网络可为模型下载启用镜像源。下载器优先使用 `HTTPS_PROXY` / `ALL_PROXY`；Windows 未设置时读取系统手动代理设置。PAC 自动配置需通过环境变量指定代理。

## 代码边界

- `crates/model-sdk/`：模型目录、安装与本地登记、兼容性和就绪检查、任务控制及各推理适配器，不依赖 Tauri。
- `src-tauri/src/lib.rs`：薄命令层，将界面输入映射为 SDK 的显式引擎 ID、模型 ID 和参数，转发进度与流式输出。
- `src/`：参数输入、文件预览和结果展示，不实现推理或应用工作流。

模型权重按内置目录中的大小和 SHA-256 校验。YOLO 首批权重来自 EdgeFirst Model Zoo 的固定提交，标记为 AGPL-3.0；分发前需核对各运行时、权重及 FFmpeg 的许可证。SDK 的真实推理验收使用 `crates/model-sdk/tests/smoke.rs`，需要本地资源。
