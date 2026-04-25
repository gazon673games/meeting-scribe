import { Activity, Play, RefreshCw, Square } from "lucide-react";

const MODE_TABS = [
  { id: "recording", label: "Recording" },
  { id: "live-assist", label: "Live Assist" },
  { id: "analysis", label: "Analysis" }
];

export function TopBar({ canStart, canStop, loading, mode, status, session, backendStatus, asrMetrics, onModeChange, onRefresh, onStartStop }) {
  const running = status === "recording";
  const disabled = running ? !canStop : !canStart;
  const latencyMs = Math.max(0, Math.round(Number(asrMetrics.avgLatencyS || 0) * 1000));

  return (
    <header className="control-bar">
      <div className="control-left">
        <button className={`record-button ${running ? "stop" : ""}`} disabled={disabled} onClick={onStartStop}>
          {running ? <Square size={16} /> : <Play size={16} />}
          {running ? "Stop Recording" : "Start Recording"}
        </button>

        <div className="status-line">
          <span className={`status-dot ${status}`} />
          <span>{status}</span>
        </div>

        <div className="latency-pill">
          <Activity size={14} />
          <span>{latencyMs > 0 ? `~${latencyMs}ms delay` : "~0ms delay"}</span>
        </div>

        <button className="icon-button" onClick={onRefresh} disabled={loading} title="Refresh backend state">
          <RefreshCw size={16} />
        </button>

        <div className={`backend-pill ${backendStatus.ready ? "ready" : "pending"}`}>
          {backendStatus.ready ? "Python ready" : "Connecting"}
        </div>
      </div>

      <div className="mode-switcher">
        {MODE_TABS.map((tab) => (
          <button key={tab.id} className={mode === tab.id ? "active" : ""} onClick={() => onModeChange(tab.id)}>
            {tab.label}
          </button>
        ))}
      </div>
    </header>
  );
}
