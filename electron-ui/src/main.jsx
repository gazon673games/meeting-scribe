import React from "react";
import { createRoot } from "react-dom/client";
import { Activity, Bot, Check, Mic, RefreshCw, Save, Settings2, Square, Trash2, Waves } from "lucide-react";
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

function App() {
  const [backendStatus, setBackendStatus] = React.useState({ ready: false, running: false, lastError: "" });
  const [state, setState] = React.useState(null);
  const [config, setConfig] = React.useState(null);
  const [settingsDraft, setSettingsDraft] = React.useState(() => makeSettingsDraft(null));
  const [settingsDirty, setSettingsDirty] = React.useState(false);
  const [savingSettings, setSavingSettings] = React.useState(false);
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
        <button className="primary-action" disabled={!canStart} onClick={() => runBackendAction("start_session", draftToStartParams(settingsDraft))}>
          <Activity size={17} />
          Start
        </button>
        <button
          className="danger-action"
          disabled={!canStop}
          onClick={() =>
            runBackendAction("stop_session", {
              runOfflinePass: settingsDraft.offlineOnStop,
              outputFile: settingsDraft.outputFile,
              language: settingsDraft.language,
              model: settingsDraft.model
            })
          }
        >
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
          <SourceList
            sources={sources}
            disabled={session.running}
            onToggle={(source) => runBackendAction("set_source_enabled", { name: source.name, enabled: !source.enabled })}
            onDelay={(source, delayMs) => runBackendAction("set_source_delay", { name: source.name, delayMs })}
            onRemove={(source) => runBackendAction("remove_source", { name: source.name })}
          />
          <DeviceGroup title="System" devices={devices.loopback} disabled={session.running} onAdd={(device) => runBackendAction("add_source", { deviceId: device.id })} />
          <DeviceGroup title="Microphones" devices={devices.input} disabled={session.running} onAdd={(device) => runBackendAction("add_source", { deviceId: device.id })} />
          {devices.errors?.length ? <div className="panel-note">{devices.errors.join(" | ")}</div> : null}
        </Panel>

        <Panel title="Processing" icon={<Settings2 size={18} />} meta={settingsDraft.asrEnabled ? "ASR on" : "ASR off"}>
          <ProcessingPanel
            summary={summary}
            session={session}
            asrMetrics={asrMetrics}
            offlinePass={offlinePass}
            draft={settingsDraft}
            options={options}
            dirty={settingsDirty}
            saving={savingSettings}
            locked={session.running}
            onChange={updateSettings}
            onAsrChange={updateAsrSetting}
            onProfileChange={applyProfile}
            onSave={saveSettings}
          />
        </Panel>

        <Panel title="Transcript" icon={<Waves size={18} />} meta={session.asrRunning ? "live" : "standby"} large>
          <TranscriptPanel lines={transcript} session={session} onClear={() => runBackendAction("clear_transcript")} />
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

function SourceList({ sources, disabled, onToggle, onDelay, onRemove }) {
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
            <SourceDelayControl source={source} disabled={disabled} onDelay={onDelay} />
            <div className="source-meter">
              <div style={{ width: `${Math.max(0, Math.min(100, source.level || 0))}%` }} />
            </div>
            <code>{source.status}</code>
            <button className="icon-action" disabled={disabled} onClick={() => onRemove(source)} title="Remove source">
              <Trash2 size={15} />
            </button>
          </div>
        ))
      ) : (
        <div className="empty-row">Add a source to start</div>
      )}
    </div>
  );
}

function SourceDelayControl({ source, disabled, onDelay }) {
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
      <span>ms</span>
      <input
        type="number"
        min="0"
        step="10"
        value={value}
        disabled={disabled}
        onChange={(event) => setValue(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
      />
    </label>
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

function ProcessingPanel({ summary, session, asrMetrics, offlinePass, draft, options, dirty, saving, locked, onChange, onAsrChange, onProfileChange, onSave }) {
  const profileOptions = uniqueOptions(options.asrProfiles?.length ? options.asrProfiles : FALLBACK_OPTIONS.asrProfiles, draft.profile);
  const modelOptions = uniqueOptions(options.asrModels?.length ? options.asrModels : FALLBACK_OPTIONS.asrModels, draft.model);
  const languageOptions = uniqueOptions(options.languages?.length ? options.languages : FALLBACK_OPTIONS.languages, draft.language);
  const modeOptions = options.asrModes?.length ? options.asrModes : FALLBACK_OPTIONS.asrModes;
  const computeOptions = uniqueOptions(options.computeTypes?.length ? options.computeTypes : FALLBACK_OPTIONS.computeTypes, draft.computeType);
  const overloadOptions = uniqueOptions(options.overloadStrategies?.length ? options.overloadStrategies : FALLBACK_OPTIONS.overloadStrategies, draft.overloadStrategy);

  return (
    <div className="processing-panel">
      <div className="settings-grid">
        <label className="check-row">
          <input type="checkbox" checked={draft.asrEnabled} disabled={locked} onChange={(event) => onChange({ asrEnabled: event.target.checked })} />
          <span>ASR</span>
        </label>
        <label className="check-row">
          <input type="checkbox" checked={draft.wavEnabled} disabled={locked} onChange={(event) => onChange({ wavEnabled: event.target.checked })} />
          <span>WAV</span>
        </label>
        <label className="check-row">
          <input type="checkbox" checked={draft.offlineOnStop} disabled={locked} onChange={(event) => onChange({ offlineOnStop: event.target.checked })} />
          <span>Offline stop</span>
        </label>
        <label className="check-row">
          <input type="checkbox" checked={draft.realtimeTranscriptToFile} disabled={locked} onChange={(event) => onChange({ realtimeTranscriptToFile: event.target.checked })} />
          <span>RT file</span>
        </label>
        <Field label="Profile">
          <select value={draft.profile} disabled={locked} onChange={(event) => onProfileChange(event.target.value)}>
            {profileOptions.map((profile) => (
              <option key={profile} value={profile}>
                {profile}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Language">
          <select value={draft.language} disabled={locked} onChange={(event) => onChange({ language: event.target.value })}>
            {languageOptions.map((language) => (
              <option key={language} value={language}>
                {language}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Mode">
          <select value={draft.asrMode} disabled={locked} onChange={(event) => onChange({ asrMode: event.target.value })}>
            {modeOptions.map((mode) => (
              <option key={mode.id} value={mode.id}>
                {mode.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Model">
          <select value={draft.model} disabled={locked} onChange={(event) => onChange({ model: event.target.value })}>
            {modelOptions.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Output">
          <input type="text" value={draft.outputFile} disabled={locked} onChange={(event) => onChange({ outputFile: event.target.value })} />
        </Field>
        <button className="primary-action settings-save" disabled={!dirty || saving} onClick={onSave}>
          {dirty ? <Save size={16} /> : <Check size={16} />}
          {saving ? "Saving" : dirty ? "Save" : "Saved"}
        </button>
      </div>

      <details className="advanced-settings">
        <summary>ASR settings</summary>
        <div className="settings-grid advanced">
          <Field label="Compute">
            <select value={draft.computeType} disabled={locked} onChange={(event) => onChange({ computeType: event.target.value })}>
              {computeOptions.map((computeType) => (
                <option key={computeType} value={computeType}>
                  {computeType}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Overload">
            <select value={draft.overloadStrategy} disabled={locked} onChange={(event) => onChange({ overloadStrategy: event.target.value })}>
              {overloadOptions.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </select>
          </Field>
          {ASR_FIELDS.map((field) => (
            <Field key={field.key} label={field.label}>
              <input
                type="number"
                min={field.min}
                max={field.max}
                step={field.step}
                value={draft.asr[field.key]}
                disabled={locked}
                onChange={(event) => onAsrChange(field.key, event.target.value)}
              />
            </Field>
          ))}
        </div>
      </details>

      <dl className="kv-list">
        <KeyValue label="Profile" value={summary.profile || draft.profile || "-"} />
        <KeyValue label="Compute" value={draft.computeType || "-"} />
        <KeyValue label="WAV" value={summary.wavEnabled ? "on" : "off"} />
        <KeyValue label="Audio drops" value={`${session.drops?.droppedOutBlocks ?? 0}/${session.drops?.droppedTapBlocks ?? 0}`} />
        <KeyValue label="ASR drops" value={`${asrMetrics.segDroppedTotal ?? 0}/${asrMetrics.segSkippedTotal ?? 0}`} />
        <KeyValue label="ASR latency" value={`${formatNumber(asrMetrics.avgLatencyS)}s avg`} />
        <KeyValue label="Offline pass" value={offlinePass.running ? "running" : offlinePass.result?.status || (summary.offlineOnStop ? "armed" : "off")} />
        <KeyValue label="Warning" value={session.lastWarning || "-"} />
      </dl>
    </div>
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

function TranscriptPanel({ lines, session, onClear }) {
  return (
    <div className="transcript-panel">
      <div className="transcript-tools">
        <button className="small-action" disabled={!lines.length} onClick={onClear}>
          Clear
        </button>
        <span>{session.realtimeTranscriptPath || session.humanLogPath || "memory"}</span>
      </div>
      <TranscriptView lines={lines} />
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
    return "--:--:--";
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

createRoot(document.getElementById("root")).render(<App />);
