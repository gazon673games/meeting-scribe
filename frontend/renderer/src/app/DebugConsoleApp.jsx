import React from "react";
import { Activity, AlertTriangle, Cpu, Database, Trash2 } from "lucide-react";

import { meetingScribeClient } from "../shared/api/meetingScribeClient";
import { formatBytes } from "../shared/lib/format";
import "./styles.css";

const MAX_EVENTS = 600;

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
  const ts = Number(event?.ts || 0).toFixed(6);
  const type = String(event?.type || "");
  const id = event?.id || event?.line?.id || "";
  const stream = event?.stream || event?.line?.stream || "";
  const span = `${event?.t_start ?? event?.line?.t_start ?? ""}:${event?.t_end ?? event?.line?.t_end ?? ""}`;
  const text = String(event?.text || event?.line?.text || event?.message || event?.error || "").slice(0, 80);
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
  const status = { status: "idle", detail: "Waiting for ASR events", tone: "idle" };
  for (const event of [...events].reverse()) {
    if (event.type === "asr_init_start") {
      Object.assign(status, { status: "starting", detail: `${event.model || "model"} on ${event.device || "-"}`, tone: "warn" });
    }
    if (event.type === "asr_init_ok" || event.type === "asr_started") {
      Object.assign(status, { status: "ready", detail: event.model || "ASR initialized", tone: "good" });
    }
    if (event.type === "asr_segment_processing") {
      Object.assign(status, { status: "processing", detail: segmentDetail(event), tone: "warn" });
    }
    if (event.type === "asr_segment_done" || event.type === "segment") {
      Object.assign(status, { status: "idle", detail: `${segmentDetail(event)} latency ${seconds(event.latency_s)}`, tone: "good" });
    }
    if (event.type === "error" && !String(event.where || "").startsWith("diar")) {
      Object.assign(status, { status: "error", detail: event.error || event.message || "ASR error", tone: "error" });
    }
    if (event.type === "asr_stopped" || event.type === "session_stopped") {
      Object.assign(status, { status: "stopped", detail: "Session stopped", tone: "idle" });
    }
  }
  return status;
}

function buildDiarStatus(events) {
  const status = { status: "idle", detail: "Waiting for Speaker ID events", tone: "idle" };
  for (const event of [...events].reverse()) {
    if (event.type === "diar_init_ok") {
      Object.assign(status, { status: "ready", detail: `backend ${event.backend || "-"}`, tone: "good" });
    }
    if (event.type === "diar_sidecar_started") {
      Object.assign(status, { status: "ready", detail: "sidecar started", tone: "good" });
    }
    if (event.type === "diar_segment_processing") {
      Object.assign(status, { status: "processing", detail: segmentDetail(event), tone: "warn" });
    }
    if (event.type === "diar_debug") {
      Object.assign(status, { status: "idle", detail: diarDebugDetail(event), tone: "good" });
    }
    if (event.type === "diar_segment_done") {
      Object.assign(status, { status: "idle", detail: `${event.speaker || "no speaker"} in ${seconds(event.latency_s)}`, tone: event.speaker ? "good" : "idle" });
    }
    if (event.type === "transcript_speaker_update") {
      Object.assign(status, { status: "updated", detail: `${event.stream || "-"} -> ${event.speaker || "-"}`, tone: "good" });
    }
    if (event.type === "error" && String(event.where || "").startsWith("diar")) {
      Object.assign(status, { status: "error", detail: event.error || "Speaker ID error", tone: "error" });
    }
  }
  return status;
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
  if (event.type === "error") {
    return `${event.where || "error"}: ${event.error || event.message || ""}`;
  }
  if (event.type === "utterance" || event.type === "transcript_line") {
    return `${event.speaker || ""} ${event.stream || ""}: ${event.text || ""}`.trim();
  }
  if (event.type === "transcript_speaker_update") {
    return `${event.stream || "-"} ${event.t_start ?? ""}-${event.t_end ?? ""} -> ${event.speaker || "-"}`;
  }
  if (event.type === "diar_debug") {
    return diarDebugDetail(event);
  }
  if (event.type === "asr_segment_processing" || event.type === "diar_segment_processing") {
    return `processing ${segmentDetail(event)}`;
  }
  if (event.type === "asr_segment_done" || event.type === "diar_segment_done") {
    return `done ${segmentDetail(event)} in ${seconds(event.latency_s)} ${event.speaker || ""}`.trim();
  }
  if (event.type === "backend_stderr") {
    return event.text || "";
  }
  return compactEventText(event);
}

function eventTitle(event) {
  return compactEventText(event, 300);
}

function compactEventText(event, maxLength = 180) {
  if (!event || typeof event !== "object") {
    return "";
  }
  const parts = [];
  for (const key of ["type", "where", "stream", "speaker", "model", "message", "error", "text"]) {
    const value = event[key];
    if (value === undefined || value === null || value === "") {
      continue;
    }
    parts.push(`${key}=${String(value).replace(/\s+/g, " ").slice(0, 120)}`);
  }
  const text = parts.join(" ");
  if (text) {
    return text.length > maxLength ? `${text.slice(0, maxLength - 1)}...` : text;
  }
  return String(event.type || "event");
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
