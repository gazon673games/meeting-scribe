import React from "react";
import { Activity, AlertTriangle, Cpu, Database, Trash2 } from "lucide-react";

import { meetingScribeClient } from "../shared/api/meetingScribeClient";
import { formatBytes } from "../shared/lib/format";
import "./styles.css";

const MAX_EVENTS = 600;
const ASR_EVENT_HANDLERS = {
  asr_init_start: (event) => statusRecord("starting", `${event.model || "model"} on ${event.device || "-"}`, "warn"),
  asr_init_ok: (event) => statusRecord("ready", event.model || "ASR initialized", "good"),
  asr_started: (event) => statusRecord("ready", event.model || "ASR initialized", "good"),
  asr_segment_processing: (event) => statusRecord("processing", segmentDetail(event), "warn"),
  asr_segment_done: (event) => statusRecord("idle", `${segmentDetail(event)} latency ${seconds(event.latency_s)}`, "good"),
  segment: (event) => statusRecord("idle", `${segmentDetail(event)} latency ${seconds(event.latency_s)}`, "good"),
  asr_stopped: () => statusRecord("stopped", "Session stopped", "idle"),
  session_stopped: () => statusRecord("stopped", "Session stopped", "idle"),
};
const DIAR_EVENT_HANDLERS = {
  diar_init_ok: (event) => statusRecord("ready", `backend ${event.backend || "-"}`, "good"),
  diar_sidecar_started: () => statusRecord("ready", "sidecar started", "good"),
  diar_segment_processing: (event) => statusRecord("processing", segmentDetail(event), "warn"),
  diar_debug: (event) => statusRecord("idle", diarDebugDetail(event), "good"),
  diar_segment_done: (event) => statusRecord("idle", `${event.speaker || "no speaker"} in ${seconds(event.latency_s)}`, event.speaker ? "good" : "idle"),
  transcript_speaker_update: (event) => statusRecord("updated", `${event.stream || "-"} -> ${event.speaker || "-"}`, "good"),
};
const PROCESSING_TYPES = new Set(["asr_segment_processing", "diar_segment_processing"]);
const DONE_TYPES = new Set(["asr_segment_done", "diar_segment_done"]);
const SUMMARY_HANDLERS = {
  utterance: transcriptSummary,
  transcript_line: transcriptSummary,
  transcript_speaker_update: (event) => `${event.stream || "-"} ${event.t_start ?? ""}-${event.t_end ?? ""} -> ${event.speaker || "-"}`,
  diar_debug: diarDebugDetail,
  backend_stderr: (event) => event.text || "",
};

export function DebugConsoleApp() {
  const [events, setEvents] = React.useState([]);
  const [backendStatus, setBackendStatus] = React.useState({ ready: false, running: false, lastError: "" });
  const [resourceUsage, setResourceUsage] = React.useState({ app: {}, system: {}, gpus: [] });

  React.useEffect(() => {
    let cancelled = false;
    meetingScribeClient.recentBackendEvents().then((result) => {
      if (!cancelled) {
        setEvents((current) => mergeEventLists((result || []).slice(-MAX_EVENTS).reverse(), current));
      }
    }).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  React.useEffect(() => {
    return meetingScribeClient.onBackendEvent((event) => {
      setEvents((current) => mergeEventLists([event], current));
    });
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    const refreshRuntime = () => {
      Promise.all([meetingScribeClient.status(), meetingScribeClient.resourceUsage()])
        .then(([status, usage]) => {
          if (cancelled) {
            return;
          }
          setBackendStatus(status || { ready: false, running: false, lastError: "" });
          setResourceUsage(usage || { app: {}, system: {}, gpus: [] });
        })
        .catch(() => {});
    };
    refreshRuntime();
    const handle = window.setInterval(refreshRuntime, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, []);

  const processes = React.useMemo(
    () => buildProcessRows({ backendStatus, events, resourceUsage }),
    [backendStatus, events, resourceUsage]
  );

  return (
    <main className="debug-console-shell" data-theme="dark">
      <header className="debug-console-head">
        <div>
          <span>Diagnostics</span>
          <h1>ASR / Speaker ID Console</h1>
        </div>
        <button className="icon-button" title="Clear visible logs" type="button" onClick={() => setEvents([])}>
          <Trash2 size={15} />
        </button>
      </header>

      <section className="debug-process-grid">
        {processes.map((process) => (
          <article key={process.id} className={`debug-process-card ${process.tone}`}>
            <div className="debug-process-icon">{process.icon}</div>
            <div className="debug-process-copy">
              <div>
                <strong>{process.label}</strong>
                <span>{process.status}</span>
              </div>
              <p title={process.detail}>{process.detail || "-"}</p>
            </div>
          </article>
        ))}
      </section>

      <section className="debug-log-panel">
        <div className="debug-log-head">
          <strong>Live Logs</strong>
          <span>{events.length} events</span>
        </div>
        <div className="debug-log-list">
          {events.length ? events.map((event, index) => <LogRow event={event} index={index} key={`${event.ts || index}:${event.type}:${index}`} />) : (
            <div className="debug-log-empty">No backend events yet.</div>
          )}
        </div>
      </section>
    </main>
  );
}

function LogRow({ event }) {
  const tone = eventTone(event);
  return (
    <div className={`debug-log-row ${tone}`}>
      <time>{formatTime(event.ts)}</time>
      <strong>{event.type || "event"}</strong>
      <span title={eventTitle(event)}>{eventSummary(event)}</span>
    </div>
  );
}

function mergeEventLists(primary, secondary) {
  const merged = [];
  const seen = new Set();
  const shouldSort = (primary || []).length > 1;
  for (const event of [...(primary || []), ...(secondary || [])]) {
    const key = eventKey(event, merged.length);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(event);
    if (merged.length >= MAX_EVENTS) {
      break;
    }
  }
  return shouldSort ? merged.sort((left, right) => Number(right?.ts || 0) - Number(left?.ts || 0)) : merged;
}

function eventKey(event, fallbackIndex) {
  const ts = Number(valueOrDefault(event?.ts, 0)).toFixed(6);
  const type = String(valueOrDefault(event?.type, ""));
  const id = String(firstDefined([event?.id, event?.line?.id], ""));
  const stream = String(firstDefined([event?.stream, event?.line?.stream], ""));
  const span = `${valueOrDefault(firstDefined([event?.t_start, event?.line?.t_start], ""), "")}:${valueOrDefault(firstDefined([event?.t_end, event?.line?.t_end], ""), "")}`;
  const text = String(firstDefined([event?.text, event?.line?.text, event?.message, event?.error], "")).slice(0, 80);
  return `${ts}:${type}:${id}:${stream}:${span}:${text || fallbackIndex}`;
}

function buildProcessRows({ backendStatus, events, resourceUsage }) {
  const asr = buildAsrStatus(events);
  const diar = buildDiarStatus(events);
  const gpu = buildGpuStatus(resourceUsage);
  return [
    {
      id: "backend",
      label: "Backend",
      status: backendStatus.running ? (backendStatus.ready ? "ready" : "starting") : "stopped",
      detail: backendStatus.lastError || `PID ${resourceUsage.app?.backendPid || "-"}`,
      tone: backendStatus.running ? "good" : "error",
      icon: <Database size={18} />
    },
    { id: "asr", label: "ASR", icon: <Activity size={18} />, ...asr },
    { id: "diar", label: "Speaker ID", icon: <AlertTriangle size={18} />, ...diar },
    { id: "gpu", label: "GPU", icon: <Cpu size={18} />, ...gpu }
  ];
}

function buildAsrStatus(events) {
  const initial = statusRecord("idle", "Waiting for ASR events", "idle");
  return foldStatusEvents(events, initial, applyAsrEvent);
}

function buildDiarStatus(events) {
  const initial = statusRecord("idle", "Waiting for Speaker ID events", "idle");
  return foldStatusEvents(events, initial, applyDiarEvent);
}

function buildGpuStatus(resourceUsage) {
  const gpu = resourceUsage.gpus?.[0];
  if (!gpu) {
    return { status: "unknown", detail: "No GPU metrics", tone: "idle" };
  }
  const util = Number(gpu.gpuUtilizationPct || 0);
  const memory = `${gpu.memoryUsedMiB || 0}/${gpu.memoryTotalMiB || 0} MiB`;
  return {
    status: `${Math.round(util)}%`,
    detail: `${gpu.name || "GPU"} memory ${memory}, app ${formatBytes(resourceUsage.app?.memoryBytes || 0)}`,
    tone: util >= 90 ? "warn" : "good"
  };
}

function eventSummary(event) {
  const type = String(event?.type || "");
  if (type === "error") return errorSummary(event);
  const handler = SUMMARY_HANDLERS[type];
  if (handler) return handler(event);
  if (PROCESSING_TYPES.has(type)) return `processing ${segmentDetail(event)}`;
  if (DONE_TYPES.has(type)) return `done ${segmentDetail(event)} in ${seconds(event.latency_s)} ${event.speaker || ""}`.trim();
  return compactEventText(event);
}

function eventTitle(event) {
  return compactEventText(event, 300);
}

function compactEventText(event, maxLength = 180) {
  if (!event || typeof event !== "object") {
    return "";
  }
  const parts = compactEventParts(event);
  const text = parts.join(" ");
  if (text) {
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}...` : text;
  }
  return String(event.type || "event");
}

function foldStatusEvents(events, initialStatus, reducer) {
  let status = initialStatus;
  for (const event of [...events].reverse()) {
    status = reducer(status, event);
  }
  return status;
}

function applyAsrEvent(status, event) {
  if (event.type === "error") {
    return String(event.where || "").startsWith("diar")
      ? status
      : statusRecord("error", event.error || event.message || "ASR error", "error");
  }
  const handler = ASR_EVENT_HANDLERS[event.type];
  return handler ? handler(event, status) : status;
}

function applyDiarEvent(status, event) {
  if (event.type === "error") {
    return String(event.where || "").startsWith("diar")
      ? statusRecord("error", event.error || "Speaker ID error", "error")
      : status;
  }
  const handler = DIAR_EVENT_HANDLERS[event.type];
  return handler ? handler(event, status) : status;
}

function statusRecord(status, detail, tone) {
  return { status, detail, tone };
}

function transcriptSummary(event) {
  return `${event.speaker || ""} ${event.stream || ""}: ${event.text || ""}`.trim();
}

function errorSummary(event) {
  return `${event.where || "error"}: ${event.error || event.message || ""}`;
}

function compactEventParts(event) {
  const keys = ["type", "where", "stream", "speaker", "model", "message", "error", "text"];
  const parts = [];
  for (const key of keys) {
    const value = event[key];
    if (isBlankValue(value)) {
      continue;
    }
    parts.push(`${key}=${String(value).replace(/\s+/g, " ").slice(0, 120)}`);
  }
  return parts;
}

function isBlankValue(value) {
  return value === undefined || value === null || value === "";
}

function firstDefined(values, fallback) {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== "") {
      return value;
    }
  }
  return fallback;
}

function valueOrDefault(value, fallback) {
  return value === undefined || value === null ? fallback : value;
}

function diarDebugDetail(event) {
  return `${event.stream || "-"} speaker ${event.speaker || "-"} sim ${number(event.best_sim)} speakers ${event.n_speakers_window ?? "-"}`;
}

function segmentDetail(event) {
  return `${event.stream || "-"} ${number(event.t_start)}-${number(event.t_end)}s`;
}

function eventTone(event) {
  if (event.type === "error" || event.type === "backend_stderr" || event.type === "backend_exit") {
    return "error";
  }
  if (String(event.type || "").includes("processing")) {
    return "warn";
  }
  if (event.type === "transcript_speaker_update" || event.type === "diar_debug") {
    return "good";
  }
  return "";
}

function formatTime(ts) {
  const value = Number(ts || Date.now() / 1000);
  return new Date(value * 1000).toLocaleTimeString();
}

function seconds(value) {
  const numberValue = Number(value || 0);
  return Number.isFinite(numberValue) ? `${numberValue.toFixed(2)}s` : "-";
}

function number(value) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue.toFixed(2) : "-";
}
