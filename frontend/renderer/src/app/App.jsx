import React from "react";

import { ASR_FIELDS, FALLBACK_OPTIONS, applySettingsToConfig, draftToStartParams, makeSettingsDraft } from "../entities/settings/model";
import { AssistantColumn } from "../features/assistant/AssistantColumn";
import { AudioInputs } from "../features/audio-inputs/AudioInputs";
import { ProcessingColumn } from "../features/processing/ProcessingColumn";
import { TopBar } from "../features/top-bar/TopBar";
import { TranscriptColumn } from "../features/transcript/TranscriptColumn";
import { meetingScribeClient } from "../shared/api/meetingScribeClient";
import "./styles.css";

export function App() {
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
        meetingScribeClient.status(),
        meetingScribeClient.request("get_state"),
        meetingScribeClient.request("get_config"),
        meetingScribeClient.request("list_devices")
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
    return meetingScribeClient.onBackendEvent((event) => {
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
      meetingScribeClient
        .request("get_state")
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
      const result = await meetingScribeClient.request("save_config", { config: applySettingsToConfig(config, settingsDraft) });
      setSettingsDirty(false);
      setConfig(result.config);
      const stateResult = await meetingScribeClient.request("get_state");
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
        const result = await meetingScribeClient.request(method, params);
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
