import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { convertFileSrc, isTauri } from "@tauri-apps/api/core";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  AudioLines,
  FileAudio,
  FileDown,
  FileVideo,
  FolderOpen,
  LoaderCircle,
  Languages,
  Play,
  Plus,
  Scissors,
  SlidersHorizontal,
  Square,
  Trash2,
} from "lucide-react";
import ResourcePanel, { type Progress, type ResourceStatus } from "./ResourcePanel";
import "./style.css";

type Segment = { start: number; end: number; text: string; translation?: string };

const languages = [
  ["zh", "中文"], ["en", "English"], ["ja", "日本語"], ["ko", "한국어"],
  ["yue", "粤语"], ["fr", "Français"], ["de", "Deutsch"], ["es", "Español"],
  ["pt", "Português"], ["ru", "Русский"], ["ar", "العربية"], ["th", "ไทย"],
  ["vi", "Tiếng Việt"], ["id", "Bahasa Indonesia"], ["ms", "Bahasa Melayu"],
] as const;

function fileName(path: string) {
  return path.split(/[\\/]/).pop() || path;
}

function displayTime(seconds: number) {
  const total = Math.max(0, Math.round(seconds * 1000));
  return `${String(Math.floor(total / 3600000)).padStart(2, "0")}:${String(Math.floor(total / 60000) % 60).padStart(2, "0")}:${String(Math.floor(total / 1000) % 60).padStart(2, "0")}.${String(total % 1000).padStart(3, "0")}`;
}

function parseTime(value: string) {
  const match = value.trim().match(/^(\d+):(\d{2}):(\d{2})[.,](\d{1,3})$/);
  if (!match) return null;
  const [, hours, minutes, seconds, millis] = match;
  if (+minutes > 59 || +seconds > 59) return null;
  return (
    +hours * 3600 + +minutes * 60 + +seconds + +millis.padEnd(3, "0") / 1000
  );
}

function App() {
  const desktop = isTauri();
  const [resources, setResources] = useState<ResourceStatus | null>(null);
  const [view, setView] = useState<"transcribe" | "resources">("resources");
  const [mediaPath, setMediaPath] = useState("");
  const [segments, setSegments] = useState<Segment[]>([]);
  const [resultAligned, setResultAligned] = useState(true);
  const [lastRun, setLastRun] = useState("");
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [previewFailed, setPreviewFailed] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");
  const [alignTimestamps, setAlignTimestamps] = useState(true);
  const [language, setLanguage] = useState("");
  const [targetLanguage, setTargetLanguage] = useState("en");
  const [translatedLanguage, setTranslatedLanguage] = useState("");
  const [exportMode, setExportMode] = useState<"source" | "translation">("source");

  useEffect(() => {
    if (!desktop) return;
    invoke<ResourceStatus>("get_resource_status")
      .then(setResources)
      .catch((reason) => setError(String(reason)));
    const pending = listen<Progress>("task-progress", (event) => {
      if (event.payload.stage !== "generate") setProgress(event.payload);
    });
    return () => {
      void pending.then((unlisten) => unlisten());
    };
  }, [desktop]);

  const mediaUrl = useMemo(
    () => (mediaPath && desktop ? convertFileSrc(mediaPath) : ""),
    [mediaPath, desktop],
  );
  const isAudio = /\.(mp3|wav|m4a|aac|flac|ogg)$/i.test(mediaPath);
  useEffect(() => {
    if (!mediaUrl) return;
    let cancelled = false;
    let objectUrl = "";
    setPreviewUrl("");
    void (async () => {
      try {
        const head = await fetch(mediaUrl, { method: "HEAD" });
        if (
          !head.ok ||
          Number(head.headers.get("content-length")) > 100 * 1024 * 1024
        ) {
          throw new Error("preview unavailable");
        }
        const response = await fetch(mediaUrl);
        if (!response.ok) throw new Error("preview unavailable");
        objectUrl = URL.createObjectURL(await response.blob());
        if (!cancelled) setPreviewUrl(objectUrl);
        else URL.revokeObjectURL(objectUrl);
      } catch {
        if (!cancelled) setPreviewFailed(true);
      }
    })();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [mediaUrl]);
  const ready = Boolean(mediaPath && resources?.ready);
  const selectedEngine = resources?.engines.find((item) => item.id === resources.selectedEngineId);
  const selectedModel = resources?.models.find((item) => item.id === resources.selectedModelId);
  const translationModel = resources?.models.find((item) => item.id === resources.selectedTranslationModelId);
  const translationComplete = Boolean(segments.length && translatedLanguage && segments.every((segment) => segment.translation?.trim()));

  function updateResources(next: ResourceStatus) {
    if (resources && (resources.selectedEngineId !== next.selectedEngineId || resources.selectedModelId !== next.selectedModelId)) {
      setSegments([]);
      setLastRun("");
      setTranslatedLanguage("");
      setExportMode("source");
    }
    setResources(next);
  }

  async function pickMedia() {
    try {
      const path = await open({
        multiple: false,
        filters: [
          {
            name: "音视频",
            extensions: [
              "mp4",
              "mov",
              "mkv",
              "webm",
              "mp3",
              "wav",
              "m4a",
              "aac",
              "flac",
              "ogg",
            ],
          },
        ],
      });
      if (typeof path === "string") {
        await invoke("allow_media_preview", { path });
        setMediaPath(path);
        setPreviewFailed(false);
        setSegments([]);
        setLastRun("");
        setTranslatedLanguage("");
        setExportMode("source");
        setError("");
        setNotice("");
        setView("transcribe");
      }
    } catch (reason) {
      setError(String(reason));
    }
  }

  async function run() {
    if (!ready) return;
    setBusy("transcribe");
    setProgress(null);
    setError("");
    setNotice("");
    try {
      const result = await invoke<Segment[]>("transcribe_selected", {
        mediaPath, alignTimestamps, language: language || null,
      });
      setSegments(result);
      setTranslatedLanguage("");
      setExportMode("source");
      setResultAligned(alignTimestamps);
      setLastRun(`${selectedModel?.label || resources?.selectedModelId} / ${selectedEngine?.label || resources?.selectedEngineId} / ${alignTimestamps ? "时间轴" : "全文"} / ${languages.find(([code]) => code === language)?.[1] || "自动"}`);
      setSelected(0);
      setNotice(alignTimestamps ? `转写完成，共 ${result.length} 条字幕` : "转写完成，已合并为全文");
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(null);
    }
  }

  async function cancel() {
    await invoke("cancel_transcription");
  }

  async function translate() {
    if (!segments.length || !resources?.translationReady || !resources.selectedTranslationModelId) return;
    setBusy("translate");
    setProgress(null);
    setError("");
    setNotice("");
    try {
      const result = await invoke<Segment[]>("translate_subtitles", {
        segments: segments.map(({ start, end, text }) => ({ start, end, text })),
        modelId: resources.selectedTranslationModelId,
        targetLanguage,
      });
      if (result.length !== segments.length) throw new Error("翻译结果与字幕数量不一致。");
      setSegments((current) => current.map((segment, index) => ({ ...segment, translation: result[index].text })));
      setTranslatedLanguage(targetLanguage);
      setExportMode("translation");
      setNotice(`已翻译 ${result.length} 条字幕`);
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(null);
    }
  }

  async function exportFile() {
    if (!segments.length) return;
    if (exportMode === "translation" && !translationComplete) return;
    const initial = mediaPath.replace(/\.[^.\\/]+$/, "") + (exportMode === "translation" ? `.${translatedLanguage}.srt` : ".srt");
    const path = await save({
      defaultPath: initial,
      filters: [{ name: "SubRip 字幕", extensions: ["srt"] }],
    });
    if (!path) return;
    try {
      await invoke("export_srt", { path, segments: segments.map(({ start, end, text, translation }) => ({ start, end, text: exportMode === "translation" ? translation : text })) });
      setNotice(`已导出 ${fileName(path)}`);
      setError("");
    } catch (reason) {
      setError(String(reason));
    }
  }

  function updateSegment(index: number, patch: Partial<Segment>) {
    setSegments((current) =>
      current.map((item, i) => (i === index ? { ...item, ...patch, translation: patch.text !== undefined && patch.text !== item.text ? undefined : item.translation } : item)),
    );
  }

  function addSegment() {
    const last = segments.at(-1);
    setSegments((current) => [
      ...current,
      { start: last?.end || 0, end: (last?.end || 0) + 2, text: "" },
    ]);
    setSelected(segments.length);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <AudioLines size={19} strokeWidth={2.3} />
          </span>
          <div>
            <strong>字幕工作台</strong>
            <span>本地推理调试</span>
          </div>
        </div>
        <div className="top-actions">
          <span className="platform-chip">
            {resources?.system.platform === "windows"
              ? "Windows"
              : resources?.system.platform || "桌面应用"}
          </span>
          <button
            className="button button-quiet"
            onClick={pickMedia}
            disabled={!desktop || Boolean(busy)}
          >
            <FolderOpen size={17} />
            导入音视频
          </button>
          <select className="export-mode" aria-label="导出内容" value={exportMode} disabled={Boolean(busy)} onChange={(event) => setExportMode(event.target.value as "source" | "translation")}>
            <option value="source">原文 SRT</option>
            <option value="translation" disabled={!translationComplete}>译文 SRT</option>
          </select>
          <button
            className="button button-primary"
            onClick={exportFile}
            disabled={!segments.length || Boolean(busy) || (exportMode === "translation" && !translationComplete)}
          >
            <FileDown size={17} />
            导出
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className={`left-pane ${view === "resources" ? "resource-view" : "transcribe-view"}`}>
          <div className="work-tabs" role="tablist" aria-label="工作视图">
            <button role="tab" aria-selected={view === "resources"} className={view === "resources" ? "active" : ""}
              onClick={() => setView("resources")}><SlidersHorizontal size={16} />资源与设备</button>
            <button role="tab" aria-selected={view === "transcribe"} className={view === "transcribe" ? "active" : ""}
              onClick={() => setView("transcribe")}><AudioLines size={16} />转写测试</button>
          </div>
          {view === "resources" ? (
            <ResourcePanel status={resources} error={error} busy={busy} progress={progress}
              onStatus={updateResources} onBusy={setBusy} onProgress={setProgress} onError={setError} onNotice={setNotice} />
          ) : (
            <>
              <div className="pane-heading">
                <div><h1>素材与转写</h1></div>
                <span className="subtle">{mediaPath ? fileName(mediaPath) : "未导入素材"}</span>
              </div>
              <div className={`media-stage ${mediaPath ? "has-media" : ""}`}>
                {mediaPath ? (
                  isAudio ? (
                    <div className="audio-stage">
                      <FileAudio size={62} strokeWidth={1.3} />
                      <strong>{fileName(mediaPath)}</strong>
                      {previewFailed ? <span>此音频无法在窗口中预览，仍可继续转写</span>
                        : !previewUrl ? <span>正在准备音频预览</span>
                        : <audio controls src={previewUrl} onError={() => setPreviewFailed(true)} />}
                    </div>
                  ) : previewFailed ? (
                    <div className="preview-fallback"><FileVideo size={38} /><strong>{fileName(mediaPath)}</strong><span>当前素材无法在窗口中预览，仍可继续转写</span></div>
                  ) : !previewUrl ? (
                    <div className="preview-fallback"><FileVideo size={38} /><span>正在准备视频预览</span></div>
                  ) : <video controls src={previewUrl} onError={() => setPreviewFailed(true)} />
                ) : (
                  <button className="empty-media" onClick={pickMedia} disabled={!desktop}>
                    <span className="empty-icon"><FileVideo size={31} strokeWidth={1.5} /></span>
                    <strong>导入音视频</strong><small>MP4、MOV、MKV、MP3、WAV 等格式</small>
                  </button>
                )}
              </div>
              <div className="active-resources">
                <div className="section-heading"><h2>当前资源</h2><button className="text-action" onClick={() => setView("resources")}>配置资源</button></div>
                <div className="active-resource-line"><span>推理引擎</span><strong>{selectedEngine?.label || "未选择"}</strong><small>{selectedEngine?.installedPath ? "已安装" : "未安装"}</small></div>
                <div className="active-resource-line"><span>转写模型</span><strong>{selectedModel?.label || "未选择"}</strong><small>{selectedModel?.installedPath ? "已安装" : "未安装"}</small></div>
                <div className="active-resource-line"><span>翻译模型</span><strong>{translationModel?.label || "未登记"}</strong><small>{resources?.translationReady ? "可用" : "未就绪"}</small></div>
                <div className="active-resource-line"><span>FFmpeg</span><strong>{resources?.system.ffmpegAvailable ? "可用" : "缺失"}</strong></div>
              </div>
              <div className="transcription-options">
                <label className="transcription-toggle">
                  <input type="checkbox" checked={alignTimestamps} disabled={Boolean(busy)}
                    onChange={(event) => setAlignTimestamps(event.target.checked)} />
                  时间轴对齐
                </label>
                <label className="transcription-language">
                  <span>识别语言</span>
                  <select value={language} disabled={Boolean(busy)} onChange={(event) => setLanguage(event.target.value)}>
                    <option value="">自动识别</option>
                    {languages.map(([code, name]) => <option value={code} key={code}>{name}</option>)}
                  </select>
                </label>
              </div>
              <div className="task-footer">
                <div className="task-status">
                  {busy ? <><LoaderCircle size={17} className="spin" /><span>{progress?.message || "正在准备任务"}</span>{progress && <b>{progress.percent}%</b>}</>
                    : <><span className={`status-dot ${ready ? "ready" : ""}`} />{ready ? "可以开始转写" : "请准备素材与本地资源"}</>}
                </div>
                {busy === "transcribe" ? (
                  <button className="button button-stop" onClick={cancel}><Square size={15} />取消</button>
                ) : (
                  <button className="button button-primary run-button" onClick={run} disabled={!ready || Boolean(busy)}>
                    <Play size={17} fill="currentColor" />开始转写
                  </button>
                )}
              </div>
            </>
          )}
        </section>

        <section className="right-pane">
          <div className="pane-heading transcript-heading">
            <div>
              <h2>{segments.length && !resultAligned ? "转录全文" : "字幕"}</h2>
              {resultAligned && <span className="count">{segments.length}</span>}
            </div>
            <div className="subtitle-heading-actions">
              <label className="translation-target">
                翻译为
                <select value={targetLanguage} disabled={Boolean(busy)} onChange={(event) => setTargetLanguage(event.target.value)}>
                  {languages.map(([code, name]) => <option value={code} key={code}>{name}</option>)}
                </select>
              </label>
              {busy === "translate" ? (
                <button className="button button-stop" onClick={cancel}><Square size={15} />取消</button>
              ) : (
                <button className="button" onClick={translate} disabled={!segments.length || !resources?.translationReady || Boolean(busy)} title={!resources?.translationReady ? resources?.translationReason || "请先配置翻译资源" : undefined}>
                  <Languages size={16} />翻译
                </button>
              )}
              <button
                className="icon-button"
                title="添加字幕行"
                aria-label="添加字幕行"
                onClick={addSegment}
                disabled={!segments.length || !resultAligned || Boolean(busy)}
              >
                <Plus size={18} />
              </button>
            </div>
          </div>
          <div className="transcript-body">
            {segments.length ? (
              segments.map((segment, index) => (
                <div
                  className={`subtitle-row ${selected === index ? "selected" : ""} ${resultAligned ? "" : "full-text"}`}
                  key={index}
                  onClick={() => setSelected(index)}
                >
                  {resultAligned && <div className="row-index">{String(index + 1).padStart(2, "0")}</div>}
                  <div className="row-main">
                    {resultAligned && <div className="time-line">
                      <input
                        aria-label={`第 ${index + 1} 行开始时间`}
                        disabled={Boolean(busy)}
                        defaultValue={displayTime(segment.start)}
                        key={`start-${index}-${segment.start}`}
                        onBlur={(event) => {
                          const value = parseTime(event.target.value);
                          if (value !== null)
                            updateSegment(index, { start: value });
                          else event.target.value = displayTime(segment.start);
                        }}
                      />
                      <span>至</span>
                      <input
                        aria-label={`第 ${index + 1} 行结束时间`}
                        disabled={Boolean(busy)}
                        defaultValue={displayTime(segment.end)}
                        key={`end-${index}-${segment.end}`}
                        onBlur={(event) => {
                          const value = parseTime(event.target.value);
                          if (value !== null)
                            updateSegment(index, { end: value });
                          else event.target.value = displayTime(segment.end);
                        }}
                      />
                    </div>}
                    <textarea
                      aria-label={resultAligned ? `第 ${index + 1} 行字幕` : "转录全文"}
                      disabled={Boolean(busy)}
                      value={segment.text}
                      onChange={(event) =>
                        updateSegment(index, { text: event.target.value })
                      }
                      rows={resultAligned ? 2 : 18}
                    />
                    {translatedLanguage && <div className="translation-line">
                      <span>译文 · {languages.find(([code]) => code === translatedLanguage)?.[1]}</span>
                      <textarea
                        aria-label={`第 ${index + 1} 行译文`}
                        value={segment.translation || ""}
                        disabled={Boolean(busy)}
                        onChange={(event) => setSegments((current) => current.map((item, i) => i === index ? { ...item, translation: event.target.value } : item))}
                        rows={resultAligned ? 2 : 8}
                      />
                    </div>}
                  </div>
                  {resultAligned && <button
                    className="icon-button delete-button"
                    title="删除字幕行"
                    aria-label={`删除第 ${index + 1} 行`}
                    disabled={Boolean(busy)}
                    onClick={(event) => {
                      event.stopPropagation();
                      setSegments((current) =>
                        current.filter((_, i) => i !== index),
                      );
                    }}
                  >
                    <Trash2 size={16} />
                  </button>}
                </div>
              ))
            ) : (
              <div className="empty-transcript">
                <Scissors size={35} strokeWidth={1.3} />
                <strong>字幕会显示在这里</strong>
                <span>完成转写后，可逐条修改时间和文字</span>
              </div>
            )}
          </div>
          <div className="transcript-footer">
            <span>
              {busy === "translate" ? `${progress?.message || "正在准备翻译"}${progress ? ` · ${progress.percent}%` : ""}`
                : segments.length ? lastRun : "支持逐条校对与导出"}
            </span>
            <span>
              {segments.length
                ? `${displayTime(segments.at(-1)?.end || 0)} 结束`
                : "SRT"}
            </span>
          </div>
        </section>
      </main>
      {!desktop && (
        <div className="browser-note">
          此页面需要在 Tauri 桌面窗口中运行，浏览器仅用于界面预览。
        </div>
      )}
      {(error || notice) && (
        <div className={`toast ${error ? "toast-error" : ""}`} role="status">
          <span>{error || notice}</span>
          <button
            aria-label="关闭提示"
            onClick={() => {
              setError("");
              setNotice("");
            }}
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
