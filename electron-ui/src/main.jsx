import React from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Check,
  ChevronDown,
  FileText,
  ListChecks,
  MessageSquare,
  Mic,
  Monitor,
  Plus,
  Play,
  RefreshCw,
  Save,
  Send,
  Settings,
  Sparkles,
  Square,
  Trash2,
  Zap
} from "lucide-react";
import "./styles.css";

const api = window.meetingScribe ?? {
  status: async () => ({ ready: false, running: false, lastError: "Electron preload unavailable" }),
  request: async () => {
    throw new Error("Electron preload unavailable");
  },
  onBackendEvent: () => () => {}
};

const FALLBACK_OPTIONS = {
  languages: ["ru", "en", "auto"],
  asrProfiles: ["Realtime", "Balanced", "Quality", "Custom"],
  asrModels: [
    "large-v3",
    "large-v3-turbo",
    "bzikst/faster-whisper-large-v3-russian",
    "bzikst/faster-whisper-podlodka-turbo",
    "medium",
    "small"
  ],
  asrModes: [
    { id: "mix", label: "MIX (master)" },
    { id: "split", label: "SPLIT (all sources)" }
  ],
  computeTypes: ["int8_float16", "float16", "int8", "int8_float32", "float32"],
  overloadStrategies: ["drop_old", "keep_all"],
  profileDefaults: {}
};

const ASR_FIELDS = [
  { key: "beam_size", label: "Beam", defaultValue: 5, step: 1, min: 1, max: 20, integer: true },
  { key: "endpoint_silence_ms", label: "Endpoint ms", defaultValue: 650, step: 10, min: 50, max: 5000 },
  { key: "max_segment_s", label: "Segment s", defaultValue: 7, step: 0.5, min: 1, max: 60 },
  { key: "overlap_ms", label: "Overlap ms", defaultValue: 200, step: 10, min: 0, max: 2000 },
  { key: "vad_energy_threshold", label: "VAD", defaultValue: 0.0055, step: 0.0001, min: 0.00001, max: 1 },
  { key: "overload_enter_qsize", label: "Overload in", defaultValue: 18, step: 1, min: 1, max: 999, integer: true },
  { key: "overload_exit_qsize", label: "Overload out", defaultValue: 6, step: 1, min: 1, max: 999, integer: true },
  { key: "overload_hard_qsize", label: "Hard q", defaultValue: 28, step: 1, min: 1, max: 999, integer: true },
  { key: "overload_beam_cap", label: "Beam cap", defaultValue: 2, step: 1, min: 1, max: 20, integer: true },
  { key: "overload_max_segment_s", label: "Overload seg", defaultValue: 5, step: 0.5, min: 0.5, max: 60 },
  { key: "overload_overlap_ms", label: "Overload overlap", defaultValue: 120, step: 10, min: 0, max: 2000 }
];

const QUALITY_PROFILES = [
  { profile: "Realtime", label: "Fast", icon: Zap },
  { profile: "Balanced", label: "Balanced" },
  { profile: "Quality", label: "High Quality" }
];

const MODE_TABS = [
  { id: "recording", label: "Recording" },
  { id: "live-assist", label: "Live Assist" },
  { id: "analysis", label: "Analysis" }
];

function App() {
  const [backendStatus, setBackendStatus] = React.useState({ ready: false, running: false, lastError: "" });
  const [state, setState] = React.useState(null);
  const [config, setConfig] = React.useState(null);
  const [settingsDraft, setSettingsDraft] = React.useState(() => makeSettingsDraft(null));
  const [settingsDirty, setSettingsDirty] = React.useState(false);
  const [savingSettings, setSavingSettings] = React.useState(false);
  const [devices, setDevices] = React.useState({ loopback: [], input: [], errors: [] });
  const [mode, setMode] = React.useState("recording");
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
      setEvents((current) => [event, ...current].slice(0, 8));
      if (event.type === "backend_ready") {
        setBackendStatus((current) => ({ ...current, ready: true, running: true }));
      }
      if (
        [
          "session_started",
          "session_stopped",
          "source_added",
          "source_removed",
          "source_updated",
          "source_error",
          "transcript_cleared",
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
    if (!settingsDirty) {
      setSettingsDraft(makeSettingsDraft(config));
    }
  }, [config, settingsDirty]);

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

  const options = state?.options || FALLBACK_OPTIONS;
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

  const updateSettings = React.useCallback((patch) => {
    setSettingsDraft((current) => ({ ...current, ...patch }));
    setSettingsDirty(true);
  }, []);

  const updateAsrSetting = React.useCallback((key, value) => {
    setSettingsDraft((current) => ({ ...current, asr: { ...current.asr, [key]: value } }));
    setSettingsDirty(true);
  }, []);

  const applyProfile = React.useCallback(
    (profile) => {
      setSettingsDraft((current) => {
        const defaults = options.profileDefaults?.[profile];
        if (!defaults || profile === "Custom") {
          return { ...current, profile };
        }
        return {
          ...current,
          profile,
          computeType: String(defaults.compute_type || current.computeType),
          overloadStrategy: String(defaults.overload_strategy || current.overloadStrategy),
          asr: {
            ...current.asr,
            ...Object.fromEntries(ASR_FIELDS.map((field) => [field.key, defaults[field.key] ?? current.asr[field.key]]))
          }
        };
      });
      setSettingsDirty(true);
    },
    [options.profileDefaults]
  );

  const saveSettings = React.useCallback(async () => {
    setSavingSettings(true);
    setError("");
    try {
      const result = await api.request("save_config", { config: applySettingsToConfig(config, settingsDraft) });
      setSettingsDirty(false);
      setConfig(result.config);
      const stateResult = await api.request("get_state");
      setState(stateResult);
    } catch (requestError) {
      setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`);
    } finally {
      setSavingSettings(false);
    }
  }, [config, settingsDraft]);

  const runBackendAction = React.useCallback(
    async (method, params = {}) => {
      setError("");
      try {
        const result = await api.request(method, params);
        if (method === "start_session" || method === "stop_session" || method === "clear_transcript") {
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

  const startOrStop = React.useCallback(() => {
    if (session.running) {
      runBackendAction("stop_session", {
        runOfflinePass: settingsDraft.offlineOnStop,
        outputFile: settingsDraft.outputFile,
        language: settingsDraft.language,
        model: settingsDraft.model
      });
      return;
    }
    runBackendAction("start_session", draftToStartParams(settingsDraft));
  }, [runBackendAction, session.running, settingsDraft]);

  const status = session.running ? "recording" : offlinePass.running ? "processing" : "idle";

  return (
    <main className="app-shell">
      <TopBar
        canStart={canStart}
        canStop={canStop}
        loading={loading}
        mode={mode}
        status={status}
        session={session}
        backendStatus={backendStatus}
        asrMetrics={asrMetrics}
        onModeChange={setMode}
        onRefresh={refresh}
        onStartStop={startOrStop}
      />

      {error ? <div className="error-strip">{error}</div> : null}

      <section className="pipeline">
        <AudioInputs
          devices={devices}
          disabled={session.running}
          sources={sources}
          onAdd={(device) => runBackendAction("add_source", { deviceId: device.id })}
          onDelay={(source, delayMs) => runBackendAction("set_source_delay", { name: source.name, delayMs })}
          onRemove={(source) => runBackendAction("remove_source", { name: source.name })}
          onToggle={(source) => runBackendAction("set_source_enabled", { name: source.name, enabled: !source.enabled })}
        />

        <ProcessingColumn
          asrMetrics={asrMetrics}
          dirty={settingsDirty}
          events={events}
          locked={session.running}
          offlinePass={offlinePass}
          options={options}
          saving={savingSettings}
          session={session}
          summary={summary}
          draft={settingsDraft}
          onAsrChange={updateAsrSetting}
          onChange={updateSettings}
          onProfileChange={applyProfile}
          onSave={saveSettings}
        />

        <TranscriptColumn
          lines={transcript}
          session={session}
          status={status}
          onClear={() => runBackendAction("clear_transcript")}
        />

        <AssistantColumn
          assistant={assistant}
          disabled={!capabilities.assistant || !assistant.enabled || assistant.busy}
          profiles={codexProfiles}
          onInvoke={(params) => runBackendAction("invoke_assistant", params)}
        />
      </section>
    </main>
  );
}

function TopBar({ canStart, canStop, loading, mode, status, session, backendStatus, asrMetrics, onModeChange, onRefresh, onStartStop }) {
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

function AudioInputs({ devices, disabled, sources, onAdd, onDelay, onRemove, onToggle }) {
  const active = sources.some((source) => source.enabled);
  const inputSources = sources.filter((source) => source.kind === "input");
  const loopbackSources = sources.filter((source) => source.kind === "loopback");
  const otherSources = sources.filter((source) => !["input", "loopback"].includes(source.kind));

  return (
    <PipelineColumn title="Audio Inputs" active={active} className="inputs-column">
      <div className="source-stack">
        {inputSources.map((source) => (
          <SourceCard
            key={source.name}
            disabled={disabled}
            icon={<Mic size={16} />}
            source={source}
            title="Microphone"
            onDelay={onDelay}
            onRemove={onRemove}
            onToggle={onToggle}
          />
        ))}
        {loopbackSources.map((source) => (
          <SourceCard
            key={source.name}
            disabled={disabled}
            icon={<Monitor size={16} />}
            source={source}
            title="System Audio"
            onDelay={onDelay}
            onRemove={onRemove}
            onToggle={onToggle}
          />
        ))}
        {otherSources.map((source) => (
          <SourceCard
            key={source.name}
            disabled={disabled}
            icon={<Activity size={16} />}
            source={source}
            title={source.name}
            onDelay={onDelay}
            onRemove={onRemove}
            onToggle={onToggle}
          />
        ))}
      </div>

      <AddDevicePicker disabled={disabled} devices={devices.input} label="Add Microphone" onAdd={onAdd} />
      <AddDevicePicker disabled={disabled} devices={devices.loopback} label="Add System Audio" onAdd={onAdd} />

      {devices.errors?.length ? <div className="panel-note">{devices.errors.join(" | ")}</div> : null}
    </PipelineColumn>
  );
}

function SourceCard({ disabled, icon, source, title, onDelay, onRemove, onToggle }) {
  const level = Math.max(0, Math.min(100, Number(source.level || 0)));
  return (
    <article className="source-card">
      <div className="source-head">
        <div className="source-title">
          {icon}
          <strong>{title}</strong>
        </div>
        <SwitchButton checked={source.enabled} disabled={disabled} onClick={() => onToggle(source)} />
      </div>

      <div className="audio-meter">
        <div className={level > 80 ? "hot" : level > 55 ? "warm" : ""} style={{ width: `${source.enabled ? level : 0}%` }} />
      </div>

      <div className="select-shell">
        <select value={source.label || source.name} disabled>
          <option>{source.label || source.name}</option>
        </select>
        <ChevronDown size={14} />
      </div>

      <div className="source-actions">
        <SourceDelayControl disabled={disabled} source={source} onDelay={onDelay} />
        <span className="source-status">{source.status}</span>
        <button className="icon-action" disabled={disabled} onClick={() => onRemove(source)} title="Remove source">
          <Trash2 size={14} />
        </button>
      </div>
    </article>
  );
}

function SourceDelayControl({ disabled, source, onDelay }) {
  const [value, setValue] = React.useState(formatDelay(source.delayMs));
  React.useEffect(() => {
    setValue(formatDelay(source.delayMs));
  }, [source.delayMs]);

  const commit = React.useCallback(() => {
    const parsed = Number(String(value || "0").replace(",", "."));
    const delayMs = Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
    setValue(formatDelay(delayMs));
    onDelay(source, delayMs);
  }, [onDelay, source, value]);

  return (
    <label className="delay-control">
      <span>delay</span>
      <input
        disabled={disabled}
        min="0"
        onBlur={commit}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
        step="10"
        type="number"
        value={value}
      />
    </label>
  );
}

function AddDevicePicker({ devices, disabled, label, onAdd }) {
  const [selectedId, setSelectedId] = React.useState("");

  React.useEffect(() => {
    setSelectedId((current) => current || devices?.[0]?.id || "");
  }, [devices]);

  if (!devices?.length) {
    return (
      <div className="add-source empty">
        <Plus size={16} />
        <span>{label}</span>
      </div>
    );
  }

  const selected = devices.find((device) => device.id === selectedId) || devices[0];
  return (
    <div className="add-source">
      <div className="select-shell">
        <select disabled={disabled} value={selected.id} onChange={(event) => setSelectedId(event.target.value)}>
          {devices.map((device) => (
            <option key={device.id} value={device.id}>
              {device.label}
            </option>
          ))}
        </select>
        <ChevronDown size={14} />
      </div>
      <button disabled={disabled} onClick={() => onAdd(selected)}>
        <Plus size={15} />
        {label}
      </button>
    </div>
  );
}

function ProcessingColumn({ asrMetrics, dirty, draft, events, locked, offlinePass, options, saving, session, summary, onAsrChange, onChange, onProfileChange, onSave }) {
  const languageOptions = uniqueOptions(options.languages?.length ? options.languages : FALLBACK_OPTIONS.languages, draft.language);
  const modelOptions = uniqueOptions(options.asrModels?.length ? options.asrModels : FALLBACK_OPTIONS.asrModels, draft.model);
  const computeOptions = uniqueOptions(options.computeTypes?.length ? options.computeTypes : FALLBACK_OPTIONS.computeTypes, draft.computeType);
  const overloadOptions = uniqueOptions(options.overloadStrategies?.length ? options.overloadStrategies : FALLBACK_OPTIONS.overloadStrategies, draft.overloadStrategy);

  return (
    <PipelineColumn title="Processing" active={session.running || offlinePass.running} activeTone={offlinePass.running ? "warn" : "muted"}>
      <SectionLabel>Speed vs Quality</SectionLabel>
      <div className="quality-stack">
        {QUALITY_PROFILES.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.profile}
              className={draft.profile === item.profile ? "selected" : ""}
              disabled={locked}
              onClick={() => onProfileChange(item.profile)}
            >
              <span>{item.label}</span>
              {Icon ? <Icon size={15} /> : null}
            </button>
          );
        })}
      </div>

      <SectionLabel>Language</SectionLabel>
      <div className="select-shell">
        <select disabled={locked} value={draft.language} onChange={(event) => onChange({ language: event.target.value })}>
          {languageOptions.map((language) => (
            <option key={language} value={language}>
              {languageLabel(language)}
            </option>
          ))}
        </select>
        <ChevronDown size={15} />
      </div>

      <SectionLabel>Features</SectionLabel>
      <div className="feature-grid">
        <FeatureToggle checked={draft.asrEnabled} disabled={locked} label="Live ASR" onClick={() => onChange({ asrEnabled: !draft.asrEnabled })} />
        <FeatureToggle checked={draft.asrMode === "split"} disabled={locked} label="Speaker Separation" onClick={() => onChange({ asrMode: draft.asrMode === "split" ? "mix" : "split" })} />
        <FeatureToggle checked={draft.wavEnabled} disabled={locked} label="Write WAV" onClick={() => onChange({ wavEnabled: !draft.wavEnabled })} />
        <FeatureToggle checked={draft.offlineOnStop} disabled={locked} label="Offline Pass" onClick={() => onChange({ offlineOnStop: !draft.offlineOnStop })} />
      </div>

      <div className="settings-strip">
        <StatLine label="Active Model" value={summary.model || draft.model || "-"} />
        <StatLine label="ASR Latency" value={`${formatNumber(asrMetrics.avgLatencyS)}s`} accent="green" />
        <StatLine label="ASR Drops" value={`${asrMetrics.segDroppedTotal ?? 0}/${asrMetrics.segSkippedTotal ?? 0}`} accent="blue" />
        <StatLine label="Audio Drops" value={`${session.drops?.droppedOutBlocks ?? 0}/${session.drops?.droppedTapBlocks ?? 0}`} />
      </div>

      <details className="advanced-settings">
        <summary>Advanced ASR</summary>
        <div className="advanced-grid">
          <Field label="Model">
            <select disabled={locked} value={draft.model} onChange={(event) => onChange({ model: event.target.value })}>
              {modelOptions.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Compute">
            <select disabled={locked} value={draft.computeType} onChange={(event) => onChange({ computeType: event.target.value })}>
              {computeOptions.map((computeType) => (
                <option key={computeType} value={computeType}>
                  {computeType}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Overload">
            <select disabled={locked} value={draft.overloadStrategy} onChange={(event) => onChange({ overloadStrategy: event.target.value })}>
              {overloadOptions.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Output">
            <input disabled={locked} type="text" value={draft.outputFile} onChange={(event) => onChange({ outputFile: event.target.value })} />
          </Field>
          {ASR_FIELDS.map((field) => (
            <Field key={field.key} label={field.label}>
              <input
                disabled={locked}
                max={field.max}
                min={field.min}
                step={field.step}
                type="number"
                value={draft.asr[field.key]}
                onChange={(event) => onAsrChange(field.key, event.target.value)}
              />
            </Field>
          ))}
        </div>
      </details>

      <button className="save-button" disabled={!dirty || saving} onClick={onSave}>
        {dirty ? <Save size={15} /> : <Check size={15} />}
        {saving ? "Saving" : dirty ? "Save Settings" : "Saved"}
      </button>

      <div className="state-card">
        <div>
          <Settings size={15} />
          <span>Current State</span>
        </div>
        <strong>{session.running ? "Processing audio stream..." : offlinePass.running ? "Finalizing transcription..." : "Ready to record"}</strong>
        {events.length ? <small>{events[0].type}</small> : null}
      </div>
    </PipelineColumn>
  );
}

function TranscriptColumn({ lines, session, status, onClear }) {
  const running = status === "recording";
  return (
    <PipelineColumn title="Live Transcript" active={running} className="transcript-column" rightMeta={<LatencyMeta session={session} />}>
      <div className="transcript-tools">
        <span>{session.realtimeTranscriptPath || session.humanLogPath || "memory"}</span>
        <button disabled={!lines.length} onClick={onClear}>
          Clear
        </button>
      </div>

      <div className="transcript-list">
        {lines.length ? (
          lines.map((line, index) => <TranscriptLine key={`${line.ts}-${index}`} line={line} index={index} />)
        ) : (
          <div className="transcript-empty">Transcript stream</div>
        )}
        {running ? (
          <div className="typing-row">
            <div>
              <time>live</time>
              <span>ASR</span>
            </div>
            <p>
              <i />
              <i />
              <i />
            </p>
          </div>
        ) : null}
      </div>
    </PipelineColumn>
  );
}

function LatencyMeta({ session }) {
  const text = session.asrRunning ? "live" : "standby";
  return (
    <span className="column-meta">
      <Activity size={13} />
      {text}
    </span>
  );
}

function TranscriptLine({ line, index }) {
  const speaker = line.stream || "mix";
  return (
    <div className="transcript-line">
      <div className="speaker-block">
        <time>{formatTime(line.ts)}</time>
        <span className={index % 2 ? "speaker-two" : ""}>{speaker}</span>
      </div>
      <p>{line.text}</p>
    </div>
  );
}

function AssistantColumn({ assistant, disabled, profiles, onInvoke }) {
  const [text, setText] = React.useState("");
  const [profileId, setProfileId] = React.useState(assistant.selectedProfileId || profiles?.[0]?.id || "");
  React.useEffect(() => {
    setProfileId((current) => current || assistant.selectedProfileId || profiles?.[0]?.id || "");
  }, [assistant.selectedProfileId, profiles]);

  const selectedProfile = profiles.find((profile) => profile.id === profileId) || profiles[0] || {};
  const response = assistant.lastResponse || {};

  const invokeCustom = (requestText, sourceLabel = "you") => {
    onInvoke({ requestText, profileId, sourceLabel });
  };

  return (
    <PipelineColumn title="AI Assistant" active={assistant.enabled} className="assistant-column">
      <SectionLabel>Profile</SectionLabel>
      <div className="select-shell">
        <select disabled={disabled || !profiles.length} value={profileId} onChange={(event) => setProfileId(event.target.value)}>
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
        <ChevronDown size={15} />
      </div>

      <SectionLabel>Quick Actions</SectionLabel>
      <div className="quick-grid">
        <ActionButton disabled={disabled} icon={<MessageSquare size={15} />} label="Quick Answer" onClick={() => onInvoke({ action: "answer", profileId })} />
        <ActionButton disabled={disabled} icon={<FileText size={15} />} label="Summarize" onClick={() => onInvoke({ action: "summary", profileId })} />
        <ActionButton
          disabled={disabled}
          icon={<ListChecks size={15} />}
          label="Action Items"
          onClick={() => invokeCustom("Extract concise action items from the current transcript.", "action items")}
        />
        <ActionButton
          disabled={disabled}
          icon={<Sparkles size={15} />}
          label="Interview Assist"
          onClick={() => invokeCustom("Help me answer the latest interview question using the current transcript.", "interview assist")}
        />
      </div>

      <div className="assistant-response">
        <span>Assistant Response</span>
        {response.text ? <p>{response.text}</p> : <p className="muted">{assistant.busy ? "Working..." : "No response yet"}</p>}
      </div>

      <div className="settings-strip">
        <StatLine label="Active Model" value={selectedProfile.model || "-"} />
        <StatLine label="Response Time" value={response.dtS ? `${formatNumber(response.dtS)}s` : assistant.busy ? "running" : "-"} accent="green" />
        <StatLine label="Last Action" value={assistant.lastRequest?.sourceLabel || "-"} accent="blue" />
      </div>

      <div className="assistant-input">
        <textarea disabled={disabled} placeholder="Ask the assistant anything..." value={text} onChange={(event) => setText(event.target.value)} />
        <button
          disabled={disabled || !text.trim()}
          onClick={() => {
            invokeCustom(text);
            setText("");
          }}
        >
          <Send size={15} />
          Send
        </button>
      </div>
    </PipelineColumn>
  );
}

function PipelineColumn({ active, activeTone = "good", children, className = "", rightMeta, title }) {
  return (
    <article className={`pipeline-column ${className}`}>
      <header className="column-header">
        <h2>{title}</h2>
        <div className="header-right">
          {rightMeta}
          <span className={`column-dot ${active ? activeTone : "idle"}`} />
        </div>
      </header>
      <div className="column-body">{children}</div>
    </article>
  );
}

function ActionButton({ disabled, icon, label, onClick }) {
  return (
    <button className="action-button" disabled={disabled} onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}

function FeatureToggle({ checked, disabled, label, onClick }) {
  return (
    <button className={`feature-toggle ${checked ? "selected" : ""}`} disabled={disabled} onClick={onClick}>
      <span>{label}</span>
      <b />
    </button>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function SectionLabel({ children }) {
  return <h3 className="section-label">{children}</h3>;
}

function StatLine({ accent = "", label, value }) {
  return (
    <div className={`stat-line ${accent}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SwitchButton({ checked, disabled, onClick }) {
  return (
    <button className={`switch-button ${checked ? "on" : ""}`} disabled={disabled} onClick={onClick}>
      <span />
    </button>
  );
}

function makeSettingsDraft(config) {
  const ui = objectSection(config?.ui);
  const asr = objectSection(config?.asr);
  return {
    asrEnabled: boolWithDefault(ui.asr_enabled, true),
    wavEnabled: boolWithDefault(ui.wav_enabled, false),
    offlineOnStop: boolWithDefault(ui.offline_on_stop, false),
    realtimeTranscriptToFile: boolWithDefault(ui.rt_transcript_to_file, false),
    language: String(ui.lang || "ru"),
    asrMode: Number(ui.asr_mode || 0) === 1 ? "split" : "mix",
    model: String(ui.model || "medium"),
    profile: String(ui.profile || "Balanced"),
    outputFile: String(ui.output_file || "capture_mix.wav"),
    computeType: String(asr.compute_type || "float16"),
    overloadStrategy: String(asr.overload_strategy || "drop_old"),
    asr: Object.fromEntries(ASR_FIELDS.map((field) => [field.key, asr[field.key] ?? field.defaultValue]))
  };
}

function applySettingsToConfig(config, draft) {
  const current = objectSection(config);
  return {
    ...current,
    version: current.version ?? 2,
    ui: {
      ...objectSection(current.ui),
      asr_enabled: Boolean(draft.asrEnabled),
      lang: String(draft.language || "ru"),
      asr_mode: draft.asrMode === "split" ? 1 : 0,
      model: String(draft.model || "medium"),
      profile: String(draft.profile || "Balanced"),
      wav_enabled: Boolean(draft.wavEnabled),
      output_file: String(draft.outputFile || "capture_mix.wav"),
      long_run: boolWithDefault(current.ui?.long_run, true),
      rt_transcript_to_file: Boolean(draft.realtimeTranscriptToFile),
      offline_on_stop: Boolean(draft.offlineOnStop),
      asr_settings_expanded: boolWithDefault(current.ui?.asr_settings_expanded, false)
    },
    asr: {
      ...objectSection(current.asr),
      compute_type: String(draft.computeType || "float16"),
      overload_strategy: String(draft.overloadStrategy || "drop_old"),
      ...Object.fromEntries(ASR_FIELDS.map((field) => [field.key, normalizeNumber(draft.asr[field.key], field)]))
    }
  };
}

function draftToStartParams(draft) {
  const config = applySettingsToConfig({}, draft);
  return {
    asrEnabled: config.ui.asr_enabled,
    wavEnabled: config.ui.wav_enabled,
    runOfflinePass: config.ui.offline_on_stop,
    realtimeTranscriptToFile: config.ui.rt_transcript_to_file,
    language: config.ui.lang,
    asrMode: draft.asrMode,
    profile: config.ui.profile,
    model: config.ui.model,
    outputFile: config.ui.output_file,
    ...config.asr
  };
}

function objectSection(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function uniqueOptions(options, currentValue) {
  const result = [];
  const seen = new Set();
  for (const value of [...(options || []), currentValue]) {
    const text = String(value || "").trim();
    if (!text || seen.has(text)) {
      continue;
    }
    seen.add(text);
    result.push(text);
  }
  return result;
}

function boolWithDefault(value, fallback) {
  return value === undefined || value === null ? Boolean(fallback) : Boolean(value);
}

function normalizeNumber(value, field) {
  const parsed = Number(String(value ?? field.defaultValue).replace(",", "."));
  let next = Number.isFinite(parsed) ? parsed : field.defaultValue;
  next = Math.max(Number(field.min), Math.min(Number(field.max), next));
  return field.integer ? Math.round(next) : next;
}

function formatDelay(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) {
    return "0";
  }
  return Math.abs(number - Math.round(number)) < 1e-6 ? String(Math.round(number)) : number.toFixed(2);
}

function formatNumber(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number.toFixed(2) : "0.00";
}

function formatTime(ts) {
  const date = new Date(Number(ts || 0) * 1000);
  if (Number.isNaN(date.getTime())) {
    return "--:--";
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function languageLabel(language) {
  const labels = { auto: "Auto", en: "English", ru: "Russian" };
  return labels[language] || language;
}

createRoot(document.getElementById("root")).render(<App />);
