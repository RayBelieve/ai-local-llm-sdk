import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { convertFileSrc, invoke, isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AudioLines, Box, Check, CircleAlert, Cpu, Download, FileImage, FolderOpen,
  LoaderCircle, MessageSquareText, Play, Plus, RefreshCw,
  Square, Trash2, X,
} from "lucide-react";
import "./style.css";

type TaskKind = "transcription" | "text-generation" | "object-detection";
type View = "resources" | TaskKind;
type Engine = {
  id: string; family: string; variant: string; label: string; version: string;
  downloadMb: number; path: string | null; suitable: boolean; reason: string | null; task: TaskKind;
};
type Model = {
  descriptor: { id: string; label: string; profile: string; artifactFormat: string; task: TaskKind };
  path: string | null; sizeBytes: number | null; downloadable: boolean;
};
type Catalog = { storageDir: string; ffmpegPath: string; engines: Engine[]; models: Model[] };
type Pair = { engineId: string; modelId: string };
type PairCheck = { compatible?: boolean; ready?: boolean; reason?: string | null };
type Progress = { stage: string; percent: number; message: string };
type Segment = { start: number; end: number; text: string };
type Detection = { classId: number; label: string; confidence: number; x1: number; y1: number; x2: number; y2: number };
type DetectionResult = { imageWidth: number; imageHeight: number; detections: Detection[] };
type Message = { role: "system" | "user" | "assistant"; content: string };
type GenerateMode = "text" | "chat" | "stream";

const tasks: { id: TaskKind; label: string; method: string; icon: typeof AudioLines }[] = [
  { id: "transcription", label: "语音转写", method: "transcribe_with_options", icon: AudioLines },
  { id: "text-generation", label: "文本生成", method: "generate_text / generate_chat", icon: MessageSquareText },
  { id: "object-detection", label: "目标检测", method: "detect_objects", icon: FileImage },
];
const emptyPairs: Record<TaskKind, Pair> = {
  transcription: { engineId: "", modelId: "" },
  "text-generation": { engineId: "", modelId: "" },
  "object-detection": { engineId: "", modelId: "" },
};
const languages = [["", "自动识别"], ["zh", "中文"], ["en", "English"], ["ja", "日本語"], ["ko", "한국어"], ["fr", "Français"], ["de", "Deutsch"], ["es", "Español"]];

function fileName(path: string) { return path.split(/[\\/]/).pop() || path; }
function time(seconds: number) {
  const value = Math.round(seconds * 1000);
  const h = Math.floor(value / 3600000);
  const m = Math.floor(value / 60000) % 60;
  const s = Math.floor(value / 1000) % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(value % 1000).padStart(3, "0")}`;
}
function size(bytes: number | null) {
  return bytes == null ? "" : `${(bytes / 1_000_000).toFixed(bytes > 100_000_000 ? 0 : 1)} MB`;
}

function App() {
  const desktop = isTauri();
  const [view, setView] = useState<View>("resources");
  const [resourceTask, setResourceTask] = useState<TaskKind>("transcription");
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [pairs, setPairs] = useState<Record<TaskKind, Pair>>(emptyPairs);
  const [checks, setChecks] = useState<Partial<Record<TaskKind, PairCheck>>>({});
  const [busy, setBusy] = useState("");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [mirror, setMirror] = useState(false);
  const [ffmpegPath, setFfmpegPath] = useState("");
  const [mediaPath, setMediaPath] = useState("");
  const [imagePath, setImagePath] = useState("");
  const [segments, setSegments] = useState<Segment[] | null>(null);
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [textOutput, setTextOutput] = useState<string | null>(null);
  const [language, setLanguage] = useState("");
  const [alignTimestamps, setAlignTimestamps] = useState(true);
  const [transcribeOptions, setTranscribeOptions] = useState(true);
  const [vadEnabled, setVadEnabled] = useState(true);
  const [generateMode, setGenerateMode] = useState<GenerateMode>("text");
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([{ role: "user", content: "" }]);
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [maximum, setMaximum] = useState(100);
  const activeRun = useRef(0);

  async function refreshCatalog() {
    const next = await invoke<Catalog>("get_catalog");
    setCatalog(next);
    setFfmpegPath((current) => current || next.ffmpegPath);
    setPairs((current) => {
      const updated = { ...current };
      for (const { id } of tasks) {
        const engines = next.engines.filter((item) => item.task === id);
        const models = next.models.filter((item) => item.descriptor.task === id);
        updated[id] = {
          engineId: engines.some((item) => item.id === current[id].engineId) ? current[id].engineId : engines[0]?.id || "",
          modelId: models.some((item) => item.descriptor.id === current[id].modelId) ? current[id].modelId : models[0]?.descriptor.id || "",
        };
      }
      return updated;
    });
    setChecks({});
    return next;
  }

  useEffect(() => {
    if (!desktop) return;
    void refreshCatalog().catch((reason) => setError(String(reason)));
    const progressListener = listen<Progress>("task-progress", (event) => setProgress(event.payload));
    const deltaListener = listen<{ runId: number; delta: string }>("text-delta", (event) => {
      if (event.payload.runId === activeRun.current) setTextOutput((value) => (value || "") + event.payload.delta);
    });
    return () => {
      void progressListener.then((unlisten) => unlisten());
      void deltaListener.then((unlisten) => unlisten());
    };
  }, [desktop]);

  async function act<T>(name: string, action: () => Promise<T>, done?: (result: T) => void) {
    setBusy(name); setProgress(null); setError(""); setNotice("");
    try { const result = await action(); done?.(result); return result; }
    catch (reason) { setError(String(reason)); return undefined; }
    finally { setBusy(""); }
  }

  function updatePair(kind: TaskKind, patch: Partial<Pair>) {
    setPairs((current) => ({ ...current, [kind]: { ...current[kind], ...patch } }));
    setChecks((current) => ({ ...current, [kind]: undefined }));
  }

  async function validate(kind: TaskKind) {
    const { engineId, modelId } = pairs[kind];
    if (!engineId || !modelId) return;
    await act("validate", async () => {
      try { await invoke("validate_pair", { engineId, modelId, taskKind: kind }); return { compatible: true, reason: null }; }
      catch (reason) { return { compatible: false, reason: String(reason) }; }
    }, (result) => setChecks((current) => ({ ...current, [kind]: result })));
  }

  async function check(kind: TaskKind) {
    const { engineId, modelId } = pairs[kind];
    if (!engineId || !modelId) return;
    await act("check", () => invoke<{ ready: boolean; reason: string | null }>("check_pair", { engineId, modelId, taskKind: kind }),
      (result) => setChecks((current) => ({ ...current, [kind]: { ...current[kind], ...result } })));
  }

  async function resourceAction(command: string, args: Record<string, unknown>, label: string) {
    await act(label, async () => {
      const result = await invoke<string | void>(command, args);
      await refreshCatalog();
      return result;
    }, () => setNotice(`${label}完成`));
  }

  async function register(kind: "engine" | "model" | "aligner") {
    try {
      const model = catalog?.models.find((item) => item.descriptor.id === pairs[resourceTask].modelId);
      const path = await open({ directory: kind === "model" && model?.descriptor.artifactFormat === "openvino-ir", multiple: false });
      if (typeof path !== "string") return;
      if (kind === "aligner") await resourceAction("register_local_aligner", { path }, "登记对齐模型");
      else if (kind === "engine") await resourceAction("register_local_engine", { engineId: pairs[resourceTask].engineId, path }, "登记引擎");
      else await resourceAction("register_local_model", { modelId: pairs[resourceTask].modelId, path }, "登记模型");
    } catch (reason) { setError(String(reason)); }
  }

  async function pickInput(kind: "media" | "image") {
    try {
      const path = await open({ multiple: false, filters: kind === "image"
        ? [{ name: "图片", extensions: ["jpg", "jpeg", "png", "webp", "bmp"] }]
        : [{ name: "音视频", extensions: ["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "aac", "flac", "ogg"] }] });
      if (typeof path !== "string") return;
      await invoke("allow_preview", { path });
      if (kind === "image") { setImagePath(path); setDetection(null); }
      else { setMediaPath(path); setSegments(null); }
      setError("");
    } catch (reason) { setError(String(reason)); }
  }

  async function runTranscription() {
    const pair = pairs.transcription;
    setSegments(null);
    await act("transcribe", () => invoke<Segment[]>("test_transcribe", {
      engineId: pair.engineId, modelId: pair.modelId, mediaPath, ffmpegPath: ffmpegPath || "ffmpeg",
      vadEnabled, withOptions: transcribeOptions, alignTimestamps, language: language || null,
    }), setSegments);
  }

  async function runGeneration() {
    const pair = pairs["text-generation"];
    const runId = ++activeRun.current;
    setTextOutput(null);
    await act("generate", () => invoke<string>("test_generate", { input: {
      engineId: pair.engineId, modelId: pair.modelId, mode: generateMode, prompt, messages,
      maxTokens, temperature, topP, runId,
    } }), setTextOutput);
  }

  async function runDetection() {
    const pair = pairs["object-detection"];
    setDetection(null);
    await act("detect", () => invoke<DetectionResult>("test_detect", {
      engineId: pair.engineId, modelId: pair.modelId, imagePath,
      confidenceThreshold: confidence, iouThreshold: iou, maxDetections: maximum,
    }), setDetection);
  }

  function pairFields(kind: TaskKind) {
    const engines = catalog?.engines.filter((item) => item.task === kind) || [];
    const models = catalog?.models.filter((item) => item.descriptor.task === kind) || [];
    const pair = pairs[kind];
    const engine = engines.find((item) => item.id === pair.engineId);
    const model = models.find((item) => item.descriptor.id === pair.modelId);
    const pairCheck = checks[kind];
    return <div className="pair-section">
      <div className="section-title"><h2>运行资源</h2><span>{kind}</span></div>
      <div className="field-grid">
        <label className="field"><span>引擎</span><select value={pair.engineId} disabled={!!busy} onChange={(event) => updatePair(kind, { engineId: event.target.value })}>
          {engines.map((item) => <option key={item.id} value={item.id}>{item.label} ({item.id})</option>)}
        </select></label>
        <label className="field"><span>模型</span><select value={pair.modelId} disabled={!!busy} onChange={(event) => updatePair(kind, { modelId: event.target.value })}>
          {models.map((item) => <option key={item.descriptor.id} value={item.descriptor.id}>{item.descriptor.label}</option>)}
        </select></label>
      </div>
      <div className="pair-meta"><span className={engine?.path ? "ok" : ""}>{engine?.path ? "引擎已安装" : "引擎未安装"}</span><span className={model?.path ? "ok" : ""}>{model?.path ? "模型已登记" : "模型未登记"}</span></div>
      <div className="pair-check"><button className="button" disabled={!desktop || !!busy || !pair.engineId || !pair.modelId} onClick={() => void validate(kind)}><Check size={15} />validate_pair_for</button>
        <button className="button" disabled={!desktop || !!busy || !pair.engineId || !pair.modelId} onClick={() => void check(kind)}><Check size={15} />check</button>
        {pairCheck && <span className={`check-state ${pairCheck.ready || pairCheck.compatible ? "ok" : "warn"}`}>{pairCheck.compatible === undefined ? "" : pairCheck.compatible ? "静态兼容" : "不兼容"}{pairCheck.ready === undefined ? "" : pairCheck.ready ? " · 运行就绪" : " · 未就绪"}{pairCheck.reason ? ` · ${pairCheck.reason}` : ""}</span>}</div>
    </div>;
  }

  const selectedEngine = catalog?.engines.find((item) => item.id === pairs[resourceTask].engineId);
  const selectedModel = catalog?.models.find((item) => item.descriptor.id === pairs[resourceTask].modelId);
  const imageUrl = imagePath && desktop ? convertFileSrc(imagePath) : "";
  const mediaUrl = mediaPath && desktop ? convertFileSrc(mediaPath) : "";
  const isAudio = /\.(mp3|wav|m4a|aac|flac|ogg)$/i.test(mediaPath);

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-icon"><Box size={20} /></span><div><strong>Model SDK</strong><small>调试台</small></div></div>
      <nav aria-label="SDK 接口">
        <button className={view === "resources" ? "nav-item active" : "nav-item"} onClick={() => { setView("resources"); setError(""); }}><Cpu size={18} />资源与兼容性</button>
        {tasks.map(({ id, label, icon: Icon }) => <button key={id} className={view === id ? "nav-item active" : "nav-item"} onClick={() => { setView(id); setError(""); }}><Icon size={18} />{label}</button>)}
      </nav>
      <div className="sidebar-footer"><span>资源目录</span><code title={catalog?.storageDir}>{catalog?.storageDir || "未连接"}</code></div>
    </aside>

    <div className="workspace">
      <header className="topbar"><div><h1>{view === "resources" ? "资源与兼容性" : tasks.find((item) => item.id === view)?.label}</h1><span>{view === "resources" ? "engines / models / validate_pair_for / check" : tasks.find((item) => item.id === view)?.method}</span></div>
        <button className="icon-button" aria-label="刷新资源目录" title="刷新资源目录" disabled={!!busy || !desktop} onClick={() => void act("refresh", refreshCatalog)}><RefreshCw size={18} /></button>
      </header>
      {!desktop && <div className="banner warning"><CircleAlert size={17} />请在 Tauri 桌面应用中运行，浏览器页面无法调用本地 SDK。</div>}
      {error && <div className="banner error" role="alert"><CircleAlert size={17} /><span>{error}</span><button className="icon-button" aria-label="关闭错误" onClick={() => setError("")}><X size={16} /></button></div>}
      {notice && <div className="banner success" role="status"><Check size={17} />{notice}</div>}
      <div className="content-grid"><main className="controls">
        {view === "resources" && <>
          <div className="task-tabs" role="tablist" aria-label="任务类型">{tasks.map(({ id, label }) => <button key={id} role="tab" aria-selected={resourceTask === id} className={resourceTask === id ? "active" : ""} onClick={() => setResourceTask(id)}>{label}</button>)}</div>
          {pairFields(resourceTask)}
          <section className="work-section"><div className="section-title"><h2>引擎</h2><code>{selectedEngine?.id}</code></div>
            <div className="resource-line"><strong>{selectedEngine?.label}</strong><span>{selectedEngine?.path || selectedEngine?.reason || `未安装 · ${selectedEngine?.version || ""}`}</span></div>
            <div className="button-row"><button className="button" disabled={!!busy || !selectedEngine} onClick={() => void register("engine")}><FolderOpen size={16} />登记本机运行时</button>
              <button className="button primary" disabled={!!busy || !selectedEngine?.suitable} onClick={() => void resourceAction("download_engine", { engineId: selectedEngine?.id }, "安装引擎")}><Download size={16} />安装指定版本</button></div>
          </section>
          <section className="work-section"><div className="section-title"><h2>模型权重</h2><code>{selectedModel?.descriptor.id}</code></div>
            <div className="resource-line"><strong>{selectedModel?.descriptor.label}</strong><span>{selectedModel?.path || `${selectedModel?.descriptor.artifactFormat || ""} · ${size(selectedModel?.sizeBytes ?? null)}`}</span></div>
            <div className="button-row"><button className="button" disabled={!!busy || !selectedModel} onClick={() => void register("model")}><FolderOpen size={16} />登记本地权重</button>
              <button className="button primary" disabled={!!busy || !selectedModel?.downloadable} onClick={() => void resourceAction("download_model", { modelId: selectedModel?.descriptor.id, mirror }, "下载模型")}><Download size={16} />下载并校验</button>
              <label className="check-label"><input type="checkbox" checked={mirror} disabled={!!busy} onChange={(event) => setMirror(event.target.checked)} />镜像源</label></div>
          </section>
          <section className="work-section"><div className="section-title"><h2>转写依赖</h2></div>
            <div className="field"><span>FFmpeg 路径</span><div className="input-action"><input value={ffmpegPath} onChange={(event) => setFfmpegPath(event.target.value)} placeholder="ffmpeg" disabled={!!busy} /><button className="icon-button" title="选择 FFmpeg" aria-label="选择 FFmpeg" disabled={!desktop || !!busy} onClick={() => void open({ multiple: false }).then((path) => { if (typeof path === "string") setFfmpegPath(path); })}><FolderOpen size={17} /></button></div></div>
            <div className="button-row"><button className="button" disabled={!desktop || !!busy} onClick={() => void act("install-ffmpeg", () => invoke<string>("install_ffmpeg"), (path) => { setFfmpegPath(path); setNotice("FFmpeg 已安装"); })}><Download size={16} />安装 FFmpeg</button>
              <button className="button" disabled={!desktop || !!busy} onClick={() => void register("aligner")}><FolderOpen size={16} />登记对齐模型</button>
              <button className="button" disabled={!desktop || !!busy} onClick={() => void resourceAction("download_aligner", {}, "下载对齐模型")}><Download size={16} />下载对齐模型</button></div>
          </section>
        </>}

        {view !== "resources" && pairFields(view)}
        {view === "transcription" && <section className="work-section"><div className="section-title"><h2>输入文件</h2></div>
          <div className="segmented" role="tablist" aria-label="转写接口"><button role="tab" aria-selected={!transcribeOptions} className={!transcribeOptions ? "active" : ""} onClick={() => setTranscribeOptions(false)}>transcribe</button><button role="tab" aria-selected={transcribeOptions} className={transcribeOptions ? "active" : ""} onClick={() => setTranscribeOptions(true)}>transcribe_with_options</button></div>
          <button className="file-picker" disabled={!!busy || !desktop} onClick={() => void pickInput("media")}><FolderOpen size={19} /><span>{mediaPath ? fileName(mediaPath) : "选择音频或视频"}</span><small>{mediaPath || "MP3、WAV、MP4 等"}</small></button>
          <div className="field-grid options"><label className="field"><span>识别语言</span><select value={language} disabled={!!busy || !transcribeOptions} onChange={(event) => setLanguage(event.target.value)}>{languages.map(([code, name]) => <option key={code} value={code}>{name}</option>)}</select></label>
            <label className="field"><span>FFmpeg</span><input value={ffmpegPath} disabled={!!busy} onChange={(event) => setFfmpegPath(event.target.value)} /></label></div>
          <div className="button-row"><label className="check-label"><input type="checkbox" checked={alignTimestamps} disabled={!!busy || !transcribeOptions} onChange={(event) => setAlignTimestamps(event.target.checked)} />时间轴对齐</label>
            <label className="check-label"><input type="checkbox" checked={vadEnabled} disabled={!!busy} onChange={(event) => setVadEnabled(event.target.checked)} />VAD 分段</label></div>
          <button className="button primary run-button" disabled={!!busy || !mediaPath || !pairs.transcription.engineId || !pairs.transcription.modelId} onClick={() => void runTranscription()}><Play size={16} />调用转写接口</button>
        </section>}

        {view === "text-generation" && <section className="work-section"><div className="section-title"><h2>请求内容</h2></div>
          <div className="segmented" role="tablist" aria-label="生成接口">{(["text", "chat", "stream"] as GenerateMode[]).map((mode) => <button key={mode} className={generateMode === mode ? "active" : ""} role="tab" aria-selected={generateMode === mode} onClick={() => setGenerateMode(mode)}>{mode === "text" ? "generate_text" : mode === "chat" ? "generate_chat" : "generate_chat_stream"}</button>)}</div>
          {generateMode === "text" ? <label className="field"><span>Prompt</span><textarea rows={8} value={prompt} disabled={!!busy} onChange={(event) => setPrompt(event.target.value)} placeholder="输入测试提示词" /></label>
            : <div className="messages">{messages.map((message, index) => <div className="message-row" key={index}><select value={message.role} disabled={!!busy} aria-label={`消息 ${index + 1} 角色`} onChange={(event) => setMessages((current) => current.map((item, at) => at === index ? { ...item, role: event.target.value as Message["role"] } : item))}><option value="system">system</option><option value="user">user</option><option value="assistant">assistant</option></select><textarea rows={3} value={message.content} disabled={!!busy} aria-label={`消息 ${index + 1} 内容`} onChange={(event) => setMessages((current) => current.map((item, at) => at === index ? { ...item, content: event.target.value } : item))} /><button className="icon-button" title="删除消息" aria-label="删除消息" disabled={!!busy || messages.length === 1} onClick={() => setMessages((current) => current.filter((_, at) => at !== index))}><Trash2 size={17} /></button></div>)}<button className="button" disabled={!!busy} onClick={() => setMessages((current) => [...current, { role: "user", content: "" }])}><Plus size={16} />添加消息</button></div>}
          <div className="field-grid options"><label className="field"><span>最大 token</span><input type="number" min={1} max={1024} value={maxTokens} disabled={!!busy} onChange={(event) => setMaxTokens(Number(event.target.value))} /></label>
            {generateMode !== "text" && <><label className="field"><span>Temperature</span><input type="number" step="0.1" min={0} value={temperature} disabled={!!busy} onChange={(event) => setTemperature(Number(event.target.value))} /></label><label className="field"><span>Top P</span><input type="number" step="0.05" min={0.01} max={1} value={topP} disabled={!!busy} onChange={(event) => setTopP(Number(event.target.value))} /></label></>}</div>
          <div className="button-row"><button className="button primary run-button" disabled={!!busy || !pairs["text-generation"].engineId || !pairs["text-generation"].modelId || (generateMode === "text" ? !prompt.trim() : messages.some((item) => !item.content.trim()))} onClick={() => void runGeneration()}><Play size={16} />调用生成接口</button>
            <button className="button" disabled={!desktop || !!busy} onClick={() => void act("stop", () => invoke<void>("stop_generation"), () => setNotice("生成服务已停止"))}><Square size={15} />停止服务</button></div>
        </section>}

        {view === "object-detection" && <section className="work-section"><div className="section-title"><h2>输入图片</h2></div>
          <button className="file-picker" disabled={!!busy || !desktop} onClick={() => void pickInput("image")}><FolderOpen size={19} /><span>{imagePath ? fileName(imagePath) : "选择图片"}</span><small>{imagePath || "JPG、PNG、WebP、BMP"}</small></button>
          <div className="field-grid options"><label className="field"><span>置信度阈值</span><input type="number" min={0} max={1} step={0.05} value={confidence} disabled={!!busy} onChange={(event) => setConfidence(Number(event.target.value))} /></label>
            <label className="field"><span>NMS IoU</span><input type="number" min={0} max={1} step={0.05} value={iou} disabled={!!busy} onChange={(event) => setIou(Number(event.target.value))} /></label>
            <label className="field"><span>最多目标</span><input type="number" min={1} max={1000} value={maximum} disabled={!!busy} onChange={(event) => setMaximum(Number(event.target.value))} /></label></div>
          <button className="button primary run-button" disabled={!!busy || !imagePath || !pairs["object-detection"].engineId || !pairs["object-detection"].modelId} onClick={() => void runDetection()}><Play size={16} />调用检测接口</button>
        </section>}
      </main>

      <aside className="results"><div className="result-heading"><div><h2>{view === "resources" ? "目录概览" : "调用结果"}</h2><span>{view === "resources" ? "SDK catalog" : tasks.find((item) => item.id === view)?.method}</span></div>{busy && <LoaderCircle size={18} className="spin" />}</div>
        {view === "resources" && <div className="inventory"><div className="summary-row"><div><strong>{catalog?.engines.filter((item) => item.task === resourceTask).length || 0}</strong><span>引擎版本</span></div><div><strong>{catalog?.models.filter((item) => item.descriptor.task === resourceTask).length || 0}</strong><span>内置模型</span></div></div><h3>引擎</h3>{catalog?.engines.filter((item) => item.task === resourceTask).map((item) => <div className="inventory-row" key={item.id}><div><strong>{item.label}</strong><code>{item.id}</code></div><span className={item.path ? "state ok" : "state"}>{item.path ? "已安装" : item.suitable ? "未安装" : "不可用"}</span></div>)}<h3>模型</h3>{catalog?.models.filter((item) => item.descriptor.task === resourceTask).map((item) => <div className="inventory-row" key={item.descriptor.id}><div><strong>{item.descriptor.label}</strong><code>{item.descriptor.id}</code></div><span className={item.path ? "state ok" : "state"}>{item.path ? "已登记" : "未登记"}</span></div>)}</div>}
        {view === "transcription" && <div className="result-body">{mediaUrl && (isAudio ? <audio controls src={mediaUrl} className="media-preview" /> : <video controls src={mediaUrl} className="media-preview" />)}{segments ? <><div className="result-count">{segments.length} 条结果</div><div className="segments">{segments.map((item, index) => <div className="segment" key={index}><time>{time(item.start)} – {time(item.end)}</time><p>{item.text}</p></div>)}</div><details><summary>原始返回值</summary><pre>{JSON.stringify(segments, null, 2)}</pre></details></> : <div className="empty-result"><AudioLines size={30} /><span>选择文件并调用转写接口</span></div>}</div>}
        {view === "text-generation" && <div className="result-body">{textOutput !== null ? <><pre className="generated-text">{textOutput || (busy ? "正在等待首段输出…" : "模型返回空文本")}</pre><details><summary>原始返回值</summary><pre>{JSON.stringify(textOutput)}</pre></details></> : <div className="empty-result"><MessageSquareText size={30} /><span>输入内容并调用生成接口</span></div>}</div>}
        {view === "object-detection" && <div className="result-body">{imageUrl && <div className="detection-image"><img src={imageUrl} alt={fileName(imagePath)} />{detection?.detections.map((item, index) => <div key={index} className="detection-box" style={{ left: `${item.x1 / detection.imageWidth * 100}%`, top: `${item.y1 / detection.imageHeight * 100}%`, width: `${(item.x2 - item.x1) / detection.imageWidth * 100}%`, height: `${(item.y2 - item.y1) / detection.imageHeight * 100}%` }}><span>{item.label} {(item.confidence * 100).toFixed(0)}%</span></div>)}</div>}{detection ? <><div className="result-count">{detection.detections.length} 个目标 · {detection.imageWidth} × {detection.imageHeight}</div><div className="detections">{detection.detections.map((item, index) => <div className="detection-row" key={index}><strong>{item.label}</strong><span>{(item.confidence * 100).toFixed(1)}%</span><code>{Math.round(item.x1)}, {Math.round(item.y1)} → {Math.round(item.x2)}, {Math.round(item.y2)}</code></div>)}</div><details><summary>原始返回值</summary><pre>{JSON.stringify(detection, null, 2)}</pre></details></> : <div className="empty-result"><FileImage size={30} /><span>选择图片并调用检测接口</span></div>}</div>}
      </aside></div>
      {busy && <footer className="taskbar"><div><LoaderCircle size={16} className="spin" /><span>{progress?.message || "正在执行 SDK 调用"}</span><strong>{progress ? `${progress.percent}%` : ""}</strong></div>{["安装引擎", "下载模型", "install-ffmpeg", "下载对齐模型", "transcribe", "generate", "detect"].includes(busy) && <button className="button danger" onClick={() => void invoke("cancel_task").catch((reason) => setError(String(reason)))}><Square size={14} />取消</button>}</footer>}
    </div>
  </div>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
