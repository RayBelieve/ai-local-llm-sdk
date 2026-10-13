import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AudioLines,
  Check,
  CircleAlert,
  Cpu,
  Download,
  FolderOpen,
  HardDriveDownload,
  LoaderCircle,
  MonitorCog,
  RefreshCw,
  Languages,
} from "lucide-react";

export type Progress = { stage: string; percent: number; message: string };
export type ResourceStatus = {
  storageDir: string;
  alignerPath: string | null;
  legacyResourcesAvailable: boolean;
  system: {
    platform: string;
    architecture: string;
    windowsBuild: number | null;
    cpu: string;
    avx2: boolean;
    gpus: { name: string; vendor: string; driver: string }[];
    nvidiaDriver: string | null;
    nvidiaComputeCapability: string | null;
    ffmpegPath: string | null;
    ffmpegAvailable: boolean;
    diagnostics: string[];
  };
  engines: {
    id: string;
    provider: string;
    label: string;
    version: string;
    downloadMb: number;
    installedPath: string | null;
    suitable: boolean;
    reason: string | null;
    recommended: boolean;
  }[];
  models: {
    id: string;
    provider: string;
    label: string;
    format: string;
    sizeBytes: number | null;
    downloadable: boolean;
    installedPath: string | null;
  }[];
  selectedEngineId: string;
  selectedModelId: string;
  selectedTranslationModelId: string | null;
  llamaDownloadOptions: { id: string; label: string; downloadMb: number; suitable: boolean; recommended: boolean; reason: string | null }[];
  translationReady: boolean;
  translationReason: string | null;
  vadEnabled: boolean;
  recommendedEngineId: string;
  ready: boolean;
};

type Props = {
  status: ResourceStatus | null;
  error: string;
  busy: string | null;
  progress: Progress | null;
  onStatus: (status: ResourceStatus) => void;
  onBusy: (busy: string | null) => void;
  onProgress: (progress: Progress | null) => void;
  onError: (message: string) => void;
  onNotice: (message: string) => void;
};

function sizeLabel(bytes: number) {
  return bytes >= 1_073_741_824
    ? `${(bytes / 1_073_741_824).toFixed(1)} GB`
    : `${Math.round(bytes / 1_048_576)} MB`;
}

function ResourcePanel({ status, error, busy, progress, onStatus, onBusy, onProgress, onError, onNotice }: Props) {
  const engine = status?.engines.find((item) => item.id === status.selectedEngineId);
  const model = status?.models.find((item) => item.id === status.selectedModelId);
  const transcriptionEngines = status?.engines.filter((item) => item.provider !== "llama") || [];
  const transcriptionModels = status?.models.filter((item) => item.provider !== "llama") || [];
  const providerModels = transcriptionModels.filter((item) => item.provider === engine?.provider);
  const translationEngine = status?.engines.find((item) => item.provider === "llama");
  const translationModels = status?.models.filter((item) => item.provider === "llama") || [];
  const translationModel = translationModels.find((item) => item.id === status?.selectedTranslationModelId);
  const [llamaVariantId, setLlamaVariantId] = useState("");
  const llamaVariant = status?.llamaDownloadOptions.find((item) => item.id === llamaVariantId)
    || status?.llamaDownloadOptions.find((item) => item.recommended);
  const recommended = status?.engines.find((item) => item.recommended);
  const locked = Boolean(busy);
  const ffmpegInstallSupported = status?.system && [
    "windows:x86_64", "macos:x86_64", "macos:aarch64", "linux:x86_64", "linux:aarch64",
  ].includes(`${status.system.platform}:${status.system.architecture}`);

  async function refresh() {
    onBusy("scan");
    onError("");
    onNotice("");
    try {
      onStatus(await invoke<ResourceStatus>("get_resource_status"));
      onNotice("自检已更新");
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function select(engineId: string, modelId: string) {
    onBusy("selection");
    onError("");
    onNotice("");
    try {
      onStatus(await invoke<ResourceStatus>("select_resources", { engineId, modelId }));
      onNotice("资源选择已保存");
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function setVad(enabled: boolean) {
    onBusy("vad");
    onError("");
    try {
      onStatus(await invoke<ResourceStatus>("set_vad_enabled", { enabled }));
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function changeAligner(local: boolean) {
    let path: string | null = null;
    if (local) {
      const result = await open({ multiple: false, filters: [{ name: "Qwen3 ForcedAligner GGUF", extensions: ["gguf"] }] });
      if (typeof result !== "string") return;
      path = result;
    }
    onBusy("aligner");
    onProgress(null);
    onError("");
    onNotice("");
    try {
      if (path) await invoke("register_local_aligner", { path });
      else await invoke("download_aligner");
      onStatus(await invoke<ResourceStatus>("get_resource_status"));
      onNotice("时间轴对齐器已就绪");
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function migrate() {
    onBusy("migration");
    onError("");
    onNotice("");
    try {
      onStatus(await invoke<ResourceStatus>("migrate_legacy_resources"));
      onNotice("旧资源已迁移到程序目录");
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function changeResource(kind: "engine" | "model", source: "download" | "local") {
    if (!status) return;
    const id = kind === "engine" ? status.selectedEngineId : status.selectedModelId;
    let path: string | null = null;
    if (source === "local") {
      const result = await open({
        multiple: false,
        directory: kind === "model" && engine?.provider === "openvino",
        filters: kind === "engine" && engine?.provider === "openvino" && status.system.platform !== "windows" ? undefined : kind === "engine"
          ? [{ name: engine?.provider === "openvino" ? "Python" : "CrispASR", extensions: ["exe"] }]
          : engine?.provider === "openvino" ? undefined : model?.format === "qwen3-gguf"
            ? [{ name: "Qwen3 GGUF", extensions: ["gguf"] }]
            : [{ name: "Whisper GGML", extensions: ["bin"] }],
      });
      if (typeof result !== "string") return;
      path = result;
    }
    onBusy(kind);
    onProgress(null);
    onError("");
    onNotice("");
    try {
      if (kind === "engine") {
        if (path) await invoke("register_local_engine", { engineId: id, path });
        else await invoke("download_engine_variant", { engineId: id });
      } else if (path) {
        await invoke("register_local_model", { modelId: id, path });
      } else {
        await invoke("download_model_variant", { modelId: id, mirror });
      }
      onStatus(await invoke<ResourceStatus>("get_resource_status"));
      onNotice(kind === "engine" ? "推理引擎已就绪" : "转写模型已就绪");
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function registerTranslation(kind: "engine" | "model") {
    const result = await open({
      multiple: false,
      filters: [{ name: kind === "engine" ? "llama-server" : "翻译 GGUF", extensions: [kind === "engine" ? "exe" : "gguf"] }],
    });
    if (typeof result !== "string") return;
    onBusy("translation-resource");
    onProgress(null);
    onError("");
    onNotice("");
    try {
      if (kind === "engine") {
        await invoke("register_local_engine", { engineId: "llama-server", path: result });
        onStatus(await invoke<ResourceStatus>("get_resource_status"));
      } else {
        onStatus(await invoke<ResourceStatus>("register_translation_model", { path: result }));
      }
      onNotice(kind === "engine" ? "翻译引擎已登记" : "翻译模型已登记");
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function selectTranslationModel(modelId: string) {
    onBusy("translation-resource");
    onProgress(null);
    onError("");
    try {
      onStatus(await invoke<ResourceStatus>("select_translation_model", { modelId }));
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function downloadTranslation(kind: "engine" | "model") {
    onBusy("translation-resource");
    onProgress(null);
    onError("");
    onNotice("");
    try {
      if (kind === "engine") {
        if (!llamaVariant) return;
        await invoke("download_llama_variant", { variantId: llamaVariant.id });
        onStatus(await invoke<ResourceStatus>("get_resource_status"));
      } else {
        await invoke("download_model_variant", { modelId: "hy-mt-1.5-1.8b-q4", mirror });
        onStatus(await invoke<ResourceStatus>("select_translation_model", { modelId: "hy-mt-1.5-1.8b-q4" }));
      }
      onNotice(kind === "engine" ? "翻译引擎已安装" : "HY-MT 翻译模型已安装");
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function setFfmpeg(local: boolean) {
    let path: string | null = null;
    if (local) {
      const result = await open({
        multiple: false,
        filters: status?.system.platform === "windows"
          ? [{ name: "FFmpeg", extensions: ["exe"] }]
          : undefined,
      });
      if (typeof result !== "string") return;
      path = result;
    }
    onBusy("ffmpeg");
    onProgress(null);
    onError("");
    onNotice("");
    try {
      await invoke("set_ffmpeg_path", { path });
      onStatus(await invoke<ResourceStatus>("get_resource_status"));
      onNotice(local ? "FFmpeg 路径已保存" : "已改用系统 PATH 中的 FFmpeg");
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  async function installFfmpeg() {
    onBusy("ffmpeg");
    onProgress(null);
    onError("");
    onNotice("");
    try {
      await invoke("install_ffmpeg");
      onStatus(await invoke<ResourceStatus>("get_resource_status"));
      onNotice("FFmpeg 已就绪");
    } catch (reason) {
      onError(String(reason));
    } finally {
      onBusy(null);
    }
  }

  const [mirror, setMirror] = useState(true);

  return (
    <div className="resource-panel">
      <div className="resource-panel-heading">
        <div>
          <h1>资源与设备</h1>
          <span className={`readiness ${status?.ready ? "is-ready" : ""}`}>
            <span className="status-dot" />
            {status ? (status.ready ? "所选资源已就绪" : "所选资源未就绪") : error ? "自检失败" : "正在自检"}
          </span>
        </div>
        <button className="button button-quiet" onClick={refresh} disabled={locked} title="重新自检">
          {busy === "scan" ? <LoaderCircle size={16} className="spin" /> : <RefreshCw size={16} />}
          重新自检
        </button>
      </div>

      {!status ? (
        <div className="resource-loading">{error ? <CircleAlert size={20} /> : <LoaderCircle size={20} className="spin" />}{error ? "无法读取设备与资源，请重新自检" : "正在读取设备与资源"}</div>
      ) : (
        <div className="resource-panel-body">
          <section className="inspect-section storage-section">
            <div className="inspect-heading"><HardDriveDownload size={16} /><h2>资源存储目录</h2></div>
            <code className="storage-path" title={status.storageDir}>{status.storageDir}</code>
            {status.legacyResourcesAvailable && (
              <button className="button storage-migrate" onClick={migrate} disabled={locked}>
                {busy === "migration" ? <LoaderCircle size={15} className="spin" /> : <HardDriveDownload size={15} />}
                迁移旧资源
              </button>
            )}
          </section>

          <section className="inspect-section">
            <div className="inspect-heading"><MonitorCog size={16} /><h2>系统自检</h2></div>
            <div className="inspect-grid">
              <div><span>平台 / 架构</span><strong>{status.system.platform} / {status.system.architecture}{status.system.windowsBuild ? ` · ${status.system.windowsBuild}` : ""}</strong></div>
              <div><span>处理器</span><strong title={status.system.cpu}>{status.system.cpu}</strong></div>
              <div><span>AVX2</span><strong>{status.system.avx2 ? "可用" : "不支持"}</strong></div>
              <div><span>NVIDIA 驱动 / 算力</span><strong>{status.system.nvidiaDriver || "未检测到"}{status.system.nvidiaComputeCapability ? ` / ${status.system.nvidiaComputeCapability}` : ""}</strong></div>
            </div>
            <div className="detected-gpus">
              <span>显卡</span>
              <strong>{status.system.gpus.length ? status.system.gpus.map((gpu) => gpu.name).join(" / ") : "未检测到实体显卡"}</strong>
            </div>
            {status.system.diagnostics.map((item) => (
              <p className="diagnostic" key={item}><CircleAlert size={15} />{item}</p>
            ))}
          </section>

          <section className="inspect-section">
            <div className="inspect-heading"><Cpu size={16} /><h2>推理引擎</h2><span>{engine?.provider === "openvino" ? "OpenVINO" : "CrispASR"} {engine?.version}</span></div>
            <div className="selection-field">
              <label htmlFor="engine-select">运行时版本</label>
              <select id="engine-select" value={status.selectedEngineId} disabled={locked}
                onChange={(event) => {
                  const chosen = status.engines.find((item) => item.id === event.target.value);
                   const matching = transcriptionModels.find((item) => item.provider === chosen?.provider);
                  if (matching) void select(event.target.value, chosen?.provider === model?.provider ? status.selectedModelId : matching.id);
                }}>
                 {transcriptionEngines.map((item) => (
                  <option key={item.id} value={item.id}>{item.label}{item.recommended ? " · 推荐" : ""}{item.installedPath ? " · 已安装" : ""}</option>
                ))}
              </select>
            </div>
            <div className="resource-detail">
              <div className="resource-detail-title"><strong>{engine?.label}</strong><span>{engine?.installedPath ? "已安装" : "未安装"}</span></div>
              <span className="resource-path" title={engine?.installedPath || undefined}>{engine?.installedPath || (engine?.provider === "openvino" ? "按需安装便携 Python 与推理依赖" : `下载约 ${engine?.downloadMb ?? 0} MB`)}</span>
              {engine?.reason && <p className="resource-note"><CircleAlert size={14} />{engine.reason}</p>}
              {recommended && engine?.provider === "crispasr" && recommended.id !== engine?.id && (
                <button className="text-action" onClick={() => select(recommended.id, status.selectedModelId)} disabled={locked}>
                  使用推荐版本：{recommended.label}
                </button>
              )}
            </div>
            <div className="resource-controls">
              <button className="button" onClick={() => changeResource("engine", "local")} disabled={locked}>
                <FolderOpen size={15} />{engine?.provider === "openvino" ? "登记 Python 环境" : "登记本机文件"}
              </button>
              <button className="button button-primary" onClick={() => changeResource("engine", "download")}
                disabled={locked || (!engine?.suitable && !engine?.installedPath)}>
                {busy === "engine" ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
                {engine?.installedPath ? "重新安装" : "下载安装"}
              </button>
            </div>
          </section>

          <section className="inspect-section">
            <div className="inspect-heading"><HardDriveDownload size={16} /><h2>转写模型</h2><span>{engine?.provider === "openvino" ? "Qwen3-ASR" : "Whisper GGML / Qwen3 GGUF"}</span></div>
            <div className="selection-field">
              <label htmlFor="model-select">模型规格</label>
              <select id="model-select" value={status.selectedModelId} disabled={locked}
                onChange={(event) => select(status.selectedEngineId, event.target.value)}>
                {providerModels.map((item) => (
                  <option key={item.id} value={item.id}>{item.label}{item.installedPath ? " · 已安装" : ""}{item.sizeBytes === null ? "" : ` · ${sizeLabel(item.sizeBytes)}`}</option>
                ))}
              </select>
            </div>
            <div className="resource-detail">
              <div className="resource-detail-title"><strong>{model?.label}</strong><span>{model?.installedPath ? "已安装" : "未安装"}</span></div>
              <span className="resource-path" title={model?.installedPath || undefined}>{model?.installedPath || (model?.sizeBytes == null ? "请登记本机模型" : `下载大小 ${sizeLabel(model.sizeBytes)}`)}</span>
            </div>
            <div className="resource-controls">
              {model?.downloadable && <label className="mirror-toggle"><input type="checkbox" checked={mirror} onChange={(event) => setMirror(event.target.checked)} disabled={locked} />镜像源</label>}
              <button className="button" onClick={() => changeResource("model", "local")} disabled={locked}>
                <FolderOpen size={15} />{engine?.provider === "openvino" ? "登记模型目录" : "登记本机文件"}
              </button>
              {model?.downloadable && <button className="button button-primary" onClick={() => changeResource("model", "download")} disabled={locked}>
                {busy === "model" ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
                {model?.installedPath ? "下载并校验" : "下载安装"}
              </button>}
            </div>
          </section>

          {model?.format === "qwen3-gguf" && (
            <section className="inspect-section">
              <div className="inspect-heading"><AudioLines size={16} /><h2>时间轴对齐</h2><span>Qwen3 ForcedAligner Q5</span></div>
              <div className="resource-detail">
                <div className="resource-detail-title"><strong>字级时间轴</strong><span>{status.alignerPath ? "已安装" : "未安装"}</span></div>
                <span className="resource-path" title={status.alignerPath || undefined}>{status.alignerPath || "下载大小约 613 MB"}</span>
              </div>
              <div className="resource-controls">
                <button className="button" onClick={() => changeAligner(true)} disabled={locked}><FolderOpen size={15} />登记本机文件</button>
                <button className="button button-primary" onClick={() => changeAligner(false)} disabled={locked}>
                  {busy === "aligner" ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
                  {status.alignerPath ? "下载并校验" : "下载安装"}
                </button>
              </div>
            </section>
          )}

          {engine?.provider === "openvino" && (
            <section className="inspect-section">
              <div className="inspect-heading"><AudioLines size={16} /><h2>语音分段</h2><span>Silero VAD</span></div>
              <label className="mirror-toggle"><input type="checkbox" checked={status.vadEnabled} onChange={(event) => void setVad(event.target.checked)} disabled={locked} />启用 VAD</label>
            </section>
          )}

          <section className="inspect-section">
            <div className="inspect-heading"><Languages size={16} /><h2>字幕翻译</h2><span>llama.cpp / GGUF</span></div>
            <div className="selection-field">
              <label htmlFor="llama-variant-select">运行版本</label>
              <select id="llama-variant-select" value={llamaVariant?.id || ""} disabled={locked || !translationEngine?.suitable}
                onChange={(event) => setLlamaVariantId(event.target.value)}>
                {status.llamaDownloadOptions.map((item) => <option key={item.id} value={item.id} disabled={!item.suitable}>{item.label} · {item.downloadMb} MB{item.recommended ? " · 推荐" : ""}</option>)}
              </select>
            </div>
            <div className="resource-detail">
              <div className="resource-detail-title"><strong>llama-server.exe</strong><span>{translationEngine?.installedPath ? "已登记" : "未登记"}</span></div>
              <span className="resource-path" title={translationEngine?.installedPath || undefined}>{translationEngine?.installedPath || "选择本机 llama-server.exe"}</span>
            </div>
            <div className="resource-controls">
              <button className="button" onClick={() => registerTranslation("engine")} disabled={locked || !translationEngine?.suitable}><FolderOpen size={15} />登记翻译引擎</button>
              <button className="button button-primary" onClick={() => downloadTranslation("engine")} disabled={locked || !llamaVariant?.suitable} title={llamaVariant?.reason || undefined}>
                {busy === "translation-resource" ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}{translationEngine?.installedPath ? "重新安装" : "下载安装"}
              </button>
            </div>
            <div className="selection-field translation-model-select">
              <label htmlFor="translation-model-select">翻译模型</label>
              <select id="translation-model-select" value={status.selectedTranslationModelId || ""} disabled={locked || !translationModels.length}
                onChange={(event) => void selectTranslationModel(event.target.value)}>
                {!translationModels.length && <option value="">未登记模型</option>}
                {translationModels.map((item) => <option key={item.id} value={item.id}>{item.label}{item.installedPath ? " · 已登记" : " · 文件缺失"}</option>)}
              </select>
            </div>
            <div className="resource-detail">
              <div className="resource-detail-title"><strong>{translationModel?.label || "翻译 GGUF"}</strong><span>{status.translationReady ? "可用" : "未就绪"}</span></div>
              <span className="resource-path" title={translationModel?.installedPath || undefined}>{translationModel?.installedPath || "选择本机翻译模型"}</span>
              {status.translationReason && <p className="resource-note"><CircleAlert size={14} />{status.translationReason}</p>}
            </div>
            <div className="resource-controls">
              <label className="mirror-toggle"><input type="checkbox" checked={mirror} onChange={(event) => setMirror(event.target.checked)} disabled={locked} />镜像源</label>
              <button className="button" onClick={() => registerTranslation("model")} disabled={locked || !translationEngine?.suitable}><FolderOpen size={15} />登记翻译 GGUF</button>
              <button className="button button-primary" onClick={() => downloadTranslation("model")} disabled={locked}>
                {busy === "translation-resource" ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}下载 HY-MT
              </button>
            </div>
          </section>

          <section className="inspect-section">
            <div className="inspect-heading"><Check size={16} /><h2>共享组件</h2></div>
            <div className="ffmpeg-row">
              <div><strong>FFmpeg</strong><span className="resource-path" title={status.system.ffmpegPath || undefined}>{status.system.ffmpegPath || "未找到"}</span></div>
              <span className={`availability ${status.system.ffmpegAvailable ? "available" : ""}`}>{status.system.ffmpegAvailable ? "可用" : "缺失"}</span>
            </div>
            <div className="resource-controls">
              <button className="button" onClick={() => setFfmpeg(true)} disabled={locked}><FolderOpen size={15} />指定路径</button>
              <button className="button" onClick={() => setFfmpeg(false)} disabled={locked}><RefreshCw size={15} />使用系统 PATH</button>
              <button className="button button-primary" onClick={installFfmpeg} disabled={locked || !ffmpegInstallSupported} title={!ffmpegInstallSupported ? "当前平台暂不支持自动安装" : undefined}>
                {busy === "ffmpeg" ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}
                下载到应用
              </button>
            </div>
          </section>

          {(busy === "engine" || busy === "model" || busy === "aligner" || busy === "ffmpeg" || busy === "translation-resource") && progress && (
            <div className="download-progress" role="status">
              <span>{progress.message}</span><strong>{progress.percent}%</strong>
              <div><span style={{ width: `${progress.percent}%` }} /></div>
            </div>
          )}
          <details className="raw-status"><summary>原始自检结果</summary><pre>{JSON.stringify(status, null, 2)}</pre></details>
        </div>
      )}
    </div>
  );
}

export default ResourcePanel;
