import {
  Activity,
  Moon,
  Palette,
  Play,
  RefreshCw,
  Square,
  SquareTerminal,
  Sun,
} from 'lucide-react';

import { THEME_OPTIONS } from '../../entities/settings/model';
import { SettingsDialogButton } from '../settings/SettingsDialogButton';
import { ColumnLayoutMenu } from './ColumnLayoutMenu';

const THEME_ICONS = { dark: Moon, light: Sun, grimdark: Palette };

const captureStatus = ({ loading, status, backendStatus, canStart }) => {
  if (loading) return { label: 'Loading', tone: 'muted' };
  if (status === 'recording') {
    return { label: 'Recording', tone: 'recording' };
  }
  if (status === 'processing') {
    return { label: 'Processing', tone: 'warn' };
  }
  if (backendStatus && !backendStatus.ready) {
    const label = backendStatus.running
      ? 'Backend starting'
      : 'Backend unavailable';
    return { label, tone: 'warn' };
  }
  return canStart
    ? { label: 'Capture ready', tone: 'idle' }
    : { label: 'Capture paused', tone: 'muted' };
};

const captureHint = ({ loading, backendStatus, startUnavailableReason }) => {
  if (loading) return 'Loading application state...';
  if (backendStatus && !backendStatus.ready) {
    return backendStatus.lastError || 'Wait for the backend to become ready.';
  }
  return startUnavailableReason;
};

export const TopBar = ({
  backendStatus,
  canStart,
  canStop,
  loading,
  pipelineLayout,
  sessionActionPending,
  settingsPanel,
  startUnavailableReason,
  status,
  asrMetrics,
  onOpenDebugConsole,
  onRefresh,
  onStartStop,
}) => {
  const running = status === 'recording';
  const disabled =
    loading ||
    Boolean(backendStatus && !backendStatus.ready) ||
    Boolean(sessionActionPending) ||
    (running ? !canStop : !canStart);
  const idleLabel = running ? 'Stop' : 'Start';
  const pendingLabel = running ? 'Stopping...' : 'Starting...';
  const actionLabel = sessionActionPending ? pendingLabel : idleLabel;
  const latencyMs = Math.max(
    0,
    Math.round(Number(asrMetrics.avgLatencyS || 0) * 1000),
  );
  const statusView = captureStatus({
    loading,
    status,
    backendStatus,
    canStart,
  });
  const startHint = captureHint({
    loading,
    backendStatus,
    startUnavailableReason,
  });

  const theme = settingsPanel.draft?.theme || 'dark';
  const themeIndex = THEME_OPTIONS.findIndex((option) => option.id === theme);
  const nextTheme = THEME_OPTIONS[(themeIndex + 1) % THEME_OPTIONS.length];
  const ThemeIcon = Object.hasOwn(THEME_ICONS, theme)
    ? THEME_ICONS[theme]
    : Moon;
  const handleThemeToggle = () => {
    settingsPanel.onChange?.({ theme: nextTheme.id });
  };

  return (
    <header className="control-bar">
      <div className="control-left">
        <div className="record-control">
          <button
            className={`record-button ${running ? 'stop' : ''}`}
            disabled={disabled}
            title={!running ? startHint : undefined}
            onClick={onStartStop}
          >
            {running ? <Square size={16} /> : <Play size={16} />}
            {actionLabel}
          </button>
          {disabled && !running && !sessionActionPending && startHint ? (
            <small className="start-hint" role="status">
              {startHint}
            </small>
          ) : null}
        </div>

        <div className="latency-pill">
          <Activity size={14} />
          <span>
            {latencyMs > 0 ? `~${latencyMs}ms delay` : 'No latency data'}
          </span>
        </div>

        <button
          className="icon-button"
          onClick={onRefresh}
          disabled={loading}
          title="Refresh backend state"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      <div className="control-right">
        <div
          className={`top-status ${statusView.tone}`}
          title="Audio capture status"
        >
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
        <button
          className="icon-button"
          title="Open diagnostics console"
          type="button"
          onClick={onOpenDebugConsole}
        >
          <SquareTerminal size={16} />
        </button>
        <ColumnLayoutMenu layout={pipelineLayout} />
      </div>
    </header>
  );
};
