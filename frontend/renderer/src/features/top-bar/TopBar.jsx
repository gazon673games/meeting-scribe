import { Activity, Moon, Palette, Play, RefreshCw, Square, SquareTerminal, Sun } from "lucide-react";

import { THEME_OPTIONS } from "../../entities/settings/model";
import { SettingsDialogButton } from "../settings/SettingsDialogButton";
import { ColumnLayoutMenu } from "./ColumnLayoutMenu";

const THEME_ICONS = { dark: Moon, light: Sun, grimdark: Palette };

export function TopBar({ backendStatus, canStart, canStop, loading, pipelineLayout, sessionActionPending, settingsPanel, startUnavailableReason, status, asrMetrics, onOpenDebugConsole, onRefresh, onStartStop }) {
  const running = status === "recording";
  const disabled = loading || Boolean(backendStatus && !backendStatus.ready) || Boolean(sessionActionPending) || (running ? !canStop : !canStart);
  const actionLabel = sessionActionPending
    ? running ? "Stopping..." : "Starting..."
    : running ? "Stop" : "Start";
  const latencyMs = Math.max(0, Math.round(Number(asrMetrics.avgLatencyS || 0) * 1000));
  const statusView = loading
    ? { label: "Loading", tone: "muted" }
    : status === "recording"
      ? { label: "Recording", tone: "recording" }
      : status === "processing"
        ? { label: "Processing", tone: "warn" }
        : backendStatus && !backendStatus.ready
          ? { label: backendStatus.running ? "Backend starting" : "Backend unavailable", tone: "warn" }
          : canStart ? { label: "Capture ready", tone: "idle" } : { label: "Capture paused", tone: "muted" };
  const startHint = loading ? "Loading application state..." : backendStatus && !backendStatus.ready ? backendStatus.lastError || "Wait for the backend to become ready." : startUnavailableReason;

  const theme = settingsPanel.draft?.theme || "dark";
  const themeIndex = THEME_OPTIONS.findIndex((option) => option.id === theme);
  const nextTheme = THEME_OPTIONS[(themeIndex + 1) % THEME_OPTIONS.length];
  const ThemeIcon = THEME_ICONS[theme] || Moon;
  const handleThemeToggle = () => {
    settingsPanel.onChange?.({ theme: nextTheme.id });
  };

  return (
    <header className="control-bar">
      <div className="control-left">
        <div className="record-control">
          <button className={`record-button ${running ? "stop" : ""}`} disabled={disabled} title={!running ? startHint : undefined} onClick={onStartStop}>
            {running ? <Square size={16} /> : <Play size={16} />}
            {actionLabel}
          </button>
          {disabled && !running && !sessionActionPending && startHint ? <small className="start-hint" role="status">{startHint}</small> : null}
        </div>

        <div className="latency-pill">
          <Activity size={14} />
          <span>{latencyMs > 0 ? `~${latencyMs}ms delay` : "No latency data"}</span>
        </div>

        <button className="icon-button" onClick={onRefresh} disabled={loading} title="Refresh backend state">
          <RefreshCw size={16} />
        </button>
      </div>

      <div className="control-right">
        <div className={`top-status ${statusView.tone}`} title="Audio capture status">
          <span />
          <strong>{statusView.label}</strong>
        </div>
        <button
          className="icon-button"
          title={`Switch to ${nextTheme.label.toLowerCase()} theme`}
          type="button"
          onClick={handleThemeToggle}
        >
          <ThemeIcon size={16} />
        </button>
        <SettingsDialogButton {...settingsPanel} />
        <button className="icon-button" title="Open diagnostics console" type="button" onClick={onOpenDebugConsole}>
          <SquareTerminal size={16} />
        </button>
        <ColumnLayoutMenu layout={pipelineLayout} />
      </div>
    </header>
  );
}
