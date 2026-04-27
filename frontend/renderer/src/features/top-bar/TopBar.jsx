import { Activity, Play, RefreshCw, Square } from "lucide-react";

import { SettingsDialogButton } from "../settings/SettingsDialogButton";
import { ColumnLayoutMenu } from "./ColumnLayoutMenu";

export function TopBar({ canStart, canStop, loading, pipelineLayout, settingsPanel, status, asrMetrics, onRefresh, onStartStop }) {
  const running = status === "recording";
  const disabled = running ? !canStop : !canStart;
  const latencyMs = Math.max(0, Math.round(Number(asrMetrics.avgLatencyS || 0) * 1000));
  const statusView = loading
    ? { label: "Loading", tone: "muted" }
    : status === "recording"
      ? { label: "Recording", tone: "good" }
      : status === "processing"
        ? { label: "Processing", tone: "warn" }
        : { label: "Ready", tone: "idle" };

  return (
    <header className="control-bar">
      <div className="control-left">
        <button className={`record-button ${running ? "stop" : ""}`} disabled={disabled} onClick={onStartStop}>
          {running ? <Square size={16} /> : <Play size={16} />}
          {running ? "Stop" : "Start"}
        </button>

        <div className="latency-pill">
          <Activity size={14} />
          <span>{latencyMs > 0 ? `~${latencyMs}ms delay` : "~0ms delay"}</span>
        </div>

        <button className="icon-button" onClick={onRefresh} disabled={loading} title="Refresh backend state">
          <RefreshCw size={16} />
        </button>
      </div>

      <div className="control-right">
        <div className={`top-status ${statusView.tone}`} title="Application status">
          <span />
          <strong>{statusView.label}</strong>
        </div>
        <SettingsDialogButton {...settingsPanel} />
        <ColumnLayoutMenu layout={pipelineLayout} />
      </div>
    </header>
  );
}
