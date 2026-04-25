import React from "react";
import { createRoot } from "react-dom/client";
import { Activity, Bot, Mic, RefreshCw, Settings2, Square, Waves } from "lucide-react";
import "./styles.css";

const api = window.meetingScribe ?? {
  status: async () => ({ ready: false, running: false, lastError: "Electron preload unavailable" }),
  request: async () => {
    throw new Error("Electron preload unavailable");
  },
  onBackendEvent: () => () => {}
};

function App() {
  const [backendStatus, setBackendStatus] = React.useState({ ready: false, running: false, lastError: "" });
  const [state, setState] = React.useState(null);
  const [config, setConfig] = React.useState(null);
  const [devices, setDevices] = React.useState({ loopback: [], input: [], errors: [] });
  const [events, setEvents] = React.useState([]);
  const [error, setError] = React.useState("");
  const [loading, setLoading] = React.useState(true);

  const refresh = React.useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [statusResult, stateResult, configResult, devicesResult] = await Promise.all([
        api.status(),
        api.request("get_state"),
        api.request("get_config"),
        api.request("list_devices")
      ]);
      setBackendStatus(statusResult);
      setState(stateResult);
      setConfig(configResult);
      setDevices(devicesResult);
    } catch (requestError) {
      setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`);
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    refresh();
    return api.onBackendEvent((event) => {
      setEvents((current) => [event, ...current].slice(0, 6));
      if (event.type === "backend_ready") {
        setBackendStatus((current) => ({ ...current, ready: true, running: true }));
      }
      if (
        [
          "session_started",
          "session_stopped",
          "source_added",
          "source_updated",
          "source_error",
          "assistant_started",
          "assistant_fallback",
          "assistant_result",
          "offline_pass_started",
          "offline_pass_done",
          "offline_pass_error"
        ].includes(event.type)
      ) {
        refresh();
      }
    });
  }, [refresh]);

  React.useEffect(() => {
    if (!state?.session?.running) {
      return undefined;
    }
    const handle = window.setInterval(() => {
      api.request("get_state")
        .then(setState)
        .catch((requestError) => setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`));
    }, 600);
    return () => window.clearInterval(handle);
  }, [state?.session?.running]);

  const runBackendAction = React.useCallback(
    async (method, params = {}) => {
      setError("");
      try {
        const result = await api.request(method, params);
        if (method === "start_session" || method === "stop_session") {
          setState((current) => ({ ...(current || {}), session: result }));
        } else {
          await refresh();
        }
        if (result?.warning) {
          setError(result.warning);
        }
      } catch (requestError) {
        setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`);
      }
    },
    [refresh]
  );

  const summary = state?.configSummary || {};
  const session = state?.session || {};
  const assistant = state?.assistant || {};
  const capabilities = state?.capabilities || {};
  const codexProfiles = assistant.profiles || config?.codex?.profiles || [];
  const sources = session.sources || [];
  const transcript = session.transcript || [];
  const asrMetrics = session.asrMetrics || {};
  const offlinePass = session.offlinePass || {};
  const canStart = Boolean(capabilities.sessionControl && sources.length > 0 && !session.running);
  const canStop = Boolean(capabilities.sessionControl && session.running);

  return (
    <main className="app-shell">
      <header className="top-bar">
        <div>
          <div className="eyebrow">Meeting Scribe</div>
          <h1>Workspace</h1>
        </div>
        <div className="top-actions">
          <StatusPill tone={backendStatus.ready ? "good" : "warn"} label={backendStatus.ready ? "Python ready" : "Connecting"} />
          <StatusPill tone={session.running ? "live" : "idle"} label={session.running ? "Recording" : "Idle"} />
          <StatusPill tone={session.asrRunning ? "good" : "idle"} label={session.asrRunning ? "ASR live" : "ASR idle"} />
          <button className="icon-button" onClick={refresh} disabled={loading} title="Refresh">
            <RefreshCw size={17} />
          </button>
        </div>
      </header>

      {error ? <div className="error-strip">{error}</div> : null}

      <section className="toolbar">
        <button className="primary-action" disabled={!canStart} onClick={() => runBackendAction("start_session")}>
          <Activity size={17} />
          Start
        </button>
        <button className="danger-action" disabled={!canStop} onClick={() => runBackendAction("stop_session")}>
          <Square size={16} />
          Stop
        </button>
        <div className="toolbar-spacer" />
        <Metric label="Language" value={summary.language || "-"} />
        <Metric label="Model" value={summary.model || "-"} wide />
        <Metric label="Mode" value={summary.asrMode || "-"} />
        <Metric label="Master" value={`${session.master?.level ?? 0}%`} />
      </section>

      <section className="workspace-grid">
        <Panel title="Sources" icon={<Mic size={18} />} meta={`${devices.loopback.length + devices.input.length} devices`}>
          <SourceList sources={sources} disabled={session.running} onToggle={(source) => runBackendAction("set_source_enabled", { name: source.name, enabled: !source.enabled })} />
          <DeviceGroup title="System" devices={devices.loopback} disabled={session.running} onAdd={(device) => runBackendAction("add_source", { deviceId: device.id })} />
          <DeviceGroup title="Microphones" devices={devices.input} disabled={session.running} onAdd={(device) => runBackendAction("add_source", { deviceId: device.id })} />
          {devices.errors?.length ? (
            <div className="panel-note">{devices.errors.join(" | ")}</div>
          ) : null}
        </Panel>

        <Panel title="Processing" icon={<Settings2 size={18} />} meta={summary.asrEnabled ? "ASR on" : "ASR off"}>
          <dl className="kv-list">
            <KeyValue label="Profile" value={summary.profile || "-"} />
            <KeyValue label="Compute" value={summary.computeType || "-"} />
            <KeyValue label="WAV" value={summary.wavEnabled ? "on" : "off"} />
            <KeyValue label="Audio drops" value={`${session.drops?.droppedOutBlocks ?? 0}/${session.drops?.droppedTapBlocks ?? 0}`} />
            <KeyValue label="ASR drops" value={`${asrMetrics.segDroppedTotal ?? 0}/${asrMetrics.segSkippedTotal ?? 0}`} />
            <KeyValue label="ASR latency" value={`${formatNumber(asrMetrics.avgLatencyS)}s avg`} />
            <KeyValue label="Offline pass" value={offlinePass.running ? "running" : offlinePass.result?.status || (summary.offlineOnStop ? "armed" : "off")} />
            <KeyValue label="Warning" value={session.lastWarning || "-"} />
          </dl>
        </Panel>

        <Panel title="Transcript" icon={<Waves size={18} />} meta={session.asrRunning ? "live" : "standby"} large>
          <TranscriptView lines={transcript} />
        </Panel>

        <Panel title="Assistant" icon={<Bot size={18} />} meta={assistant.busy ? "busy" : assistant.enabled ? "enabled" : "disabled"}>
          <AssistantPanel
            assistant={assistant}
            profiles={codexProfiles}
            disabled={!capabilities.assistant || !assistant.enabled || assistant.busy}
            onInvoke={(params) => runBackendAction("invoke_assistant", params)}
          />
        </Panel>
      </section>

      <footer className="event-bar">
        {events.length ? events.map((event, index) => <span key={`${event.type}-${index}`}>{event.type}</span>) : <span>backend pending</span>}
      </footer>
    </main>
  );
}

function StatusPill({ tone, label }) {
  return <span className={`status-pill ${tone}`}>{label}</span>;
}

function Metric({ label, value, wide = false }) {
  return (
    <div className={`metric ${wide ? "wide" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Panel({ title, icon, meta, children, large = false }) {
  return (
    <article className={`panel ${large ? "large" : ""}`}>
      <div className="panel-header">
        <div className="panel-title">
          {icon}
          <h2>{title}</h2>
        </div>
        <span>{meta}</span>
      </div>
      <div className="panel-body">{children}</div>
    </article>
  );
}

function SourceList({ sources, disabled, onToggle }) {
  return (
    <div className="source-list">
      <h3>Selected</h3>
      {sources.length ? (
        sources.map((source) => (
          <div className="source-row" key={source.name}>
            <button className={`toggle-dot ${source.enabled ? "on" : "off"}`} disabled={disabled} onClick={() => onToggle(source)} title={source.enabled ? "Mute source" : "Enable source"} />
            <div className="source-main">
              <strong>{source.name}</strong>
              <span>{source.label}</span>
            </div>
            <div className="source-meter">
              <div style={{ width: `${Math.max(0, Math.min(100, source.level || 0))}%` }} />
            </div>
            <code>{source.status}</code>
          </div>
        ))
      ) : (
        <div className="empty-row">Add a source to start</div>
      )}
    </div>
  );
}

function DeviceGroup({ title, devices, disabled, onAdd }) {
  return (
    <div className="device-group">
      <h3>{title}</h3>
      {devices.length ? (
        devices.map((device) => (
          <div className="device-row" key={device.id}>
            <span>{device.label}</span>
            <button className="small-action" disabled={disabled} onClick={() => onAdd(device)}>
              Add
            </button>
          </div>
        ))
      ) : (
        <div className="empty-row">None</div>
      )}
    </div>
  );
}

function KeyValue({ label, value }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}

function AssistantPanel({ assistant, profiles, disabled, onInvoke }) {
  const [text, setText] = React.useState("");
  const [profileId, setProfileId] = React.useState(assistant.selectedProfileId || profiles?.[0]?.id || "");
  React.useEffect(() => {
    setProfileId((current) => current || assistant.selectedProfileId || profiles?.[0]?.id || "");
  }, [assistant.selectedProfileId, profiles]);

  const response = assistant.lastResponse || {};
  return (
    <div className="assistant-panel">
      <div className="assistant-controls">
        <select value={profileId} disabled={disabled || !profiles.length} onChange={(event) => setProfileId(event.target.value)}>
          {profiles.length ? (
            profiles.map((profile) => (
              <option key={profile.id || profile.label} value={profile.id}>
                {profile.label || profile.id}
              </option>
            ))
          ) : (
            <option value="">No profiles</option>
          )}
        </select>
        <div className="assistant-actions">
          <button className="small-action" disabled={disabled} onClick={() => onInvoke({ action: "answer", profileId })}>
            Answer
          </button>
          <button className="small-action" disabled={disabled} onClick={() => onInvoke({ action: "summary", profileId })}>
            Summary
          </button>
        </div>
      </div>
      <textarea value={text} onChange={(event) => setText(event.target.value)} placeholder="Ask the assistant" disabled={disabled} />
      <button
        className="primary-action assistant-send"
        disabled={disabled || !text.trim()}
        onClick={() => {
          onInvoke({ requestText: text, profileId });
          setText("");
        }}
      >
        Send
      </button>
      <div className="assistant-response">
        {response.text ? (
          <>
            <strong>{response.ok ? response.profile || "Assistant" : "Assistant error"}</strong>
            <p>{response.text}</p>
          </>
        ) : (
          <div className="empty-row">{assistant.busy ? "Working..." : "No response yet"}</div>
        )}
      </div>
    </div>
  );
}

function TranscriptView({ lines }) {
  if (!lines.length) {
    return (
      <div className="transcript-empty">
        <span>Transcript stream</span>
      </div>
    );
  }
  return (
    <div className="transcript-lines">
      {lines.map((line, index) => (
        <div className="transcript-line" key={`${line.ts}-${index}`}>
          <time>{formatTime(line.ts)}</time>
          <strong>{line.stream || "mix"}</strong>
          <span>{line.text}</span>
        </div>
      ))}
    </div>
  );
}

function formatNumber(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toFixed(2) : "0.00";
}

function formatTime(ts) {
  const date = new Date(Number(ts || 0) * 1000);
  if (Number.isNaN(date.getTime())) {
    return "--:--:--";
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

createRoot(document.getElementById("root")).render(<App />);
