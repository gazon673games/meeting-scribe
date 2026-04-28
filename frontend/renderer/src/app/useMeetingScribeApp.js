import React from "react";

import { ASR_FIELDS, FALLBACK_OPTIONS, applySettingsToConfig, draftToStartParams, makeSettingsDraft } from "../entities/settings/model";
import { meetingScribeClient } from "../shared/api/meetingScribeClient";

const REFRESHING_EVENTS = [
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
];

export function useMeetingScribeApp() {
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
  const [resourceUsage, setResourceUsage] = React.useState({ app: {}, system: {}, gpus: [] });

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
      if (event.type === "transcript_line") {
        setState((current) => appendTranscriptLine(current, event));
      }
      if (event.type === "asr_metrics") {
        setState((current) => applyAsrMetrics(current, event));
      }
      if (event.type === "backend_ready") {
        setBackendStatus((current) => ({ ...current, ready: true, running: true }));
      }
      if (event.type === "backend_exit") {
        setBackendStatus((current) => ({ ...current, ready: false, running: false, lastError: "Python backend stopped" }));
        setError(`Python backend stopped (${event.code ?? "null"}, ${event.signal ?? "null"})`);
      }
      if (REFRESHING_EVENTS.includes(event.type)) {
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
    if (!config) {
      return;
    }
    const savedDraft = makeSettingsDraft(config);
    const enabled = settingsDirty ? settingsDraft.screenCaptureProtection : savedDraft.screenCaptureProtection;
    meetingScribeClient
      .setContentProtection(Boolean(enabled))
      .catch((requestError) => setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`));
  }, [config, settingsDirty, settingsDraft.screenCaptureProtection]);

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

  React.useEffect(() => {
    let cancelled = false;
    const loadResourceUsage = () => {
      meetingScribeClient
        .resourceUsage()
        .then((result) => {
          if (!cancelled) {
            setResourceUsage(result || { app: {}, system: {}, gpus: [] });
          }
        })
        .catch(() => {
          if (!cancelled) {
            setResourceUsage({ app: {}, system: {}, gpus: [] });
          }
        });
    };
    loadResourceUsage();
    const handle = window.setInterval(loadResourceUsage, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, []);

  const options = state?.options || FALLBACK_OPTIONS;
  const summary = state?.configSummary || {};
  const session = state?.session || {};
  const assistant = state?.assistant || {};
  const capabilities = state?.capabilities || {};
  const hardware = state?.hardware || {};
  const assistantProfiles = assistant.profiles || settingsDraft.assistantProfiles || config?.codex?.profiles || [];
  const sources = session.sources || [];
  const transcript = session.transcript || [];
  const assistantContextReady = transcript.some((line) => String(line?.text || "").trim());
  const asrMetrics = session.asrMetrics || {};
  const offlinePass = session.offlinePass || {};
  const canStart = Boolean(capabilities.sessionControl && sources.length > 0 && !session.running);
  const canStop = Boolean(capabilities.sessionControl && session.running);
  const status = session.running ? "recording" : offlinePass.running ? "processing" : "idle";

  const updateSettings = React.useCallback((patch) => {
    setSettingsDraft((current) => ({ ...current, ...patch }));
    setSettingsDirty(true);
  }, []);

  const clearError = React.useCallback(() => {
    setError("");
  }, []);

  const reloadApp = React.useCallback(async () => {
    setError("");
    try {
      await meetingScribeClient.reloadApp();
    } catch (requestError) {
      setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`);
    }
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
      return true;
    } catch (requestError) {
      setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`);
      return false;
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
        runOfflinePass: false,
        outputFile: settingsDraft.outputFile,
        language: settingsDraft.language,
        model: settingsDraft.model
      });
      return;
    }
    runBackendAction("start_session", draftToStartParams(settingsDraft));
  }, [runBackendAction, session.running, settingsDraft]);

  return {
    assistant,
    assistantContextReady,
    asrMetrics,
    backendStatus,
    canStart,
    canStop,
    capabilities,
    assistantProfiles,
    devices,
    error,
    events,
    hardware,
    loading,
    offlinePass,
    options,
    session,
    settingsDirty,
    settingsDraft,
    savingSettings,
    sources,
    status,
    summary,
    transcript,
    applyProfile,
    refresh,
    reloadApp,
    resourceUsage,
    runBackendAction,
    saveSettings,
    startOrStop,
    updateAsrSetting,
    clearError,
    updateSettings
  };
}

function appendTranscriptLine(state, event) {
  if (!state?.session || !String(event?.text || "").trim()) {
    return state;
  }
  const line = {
    ts: Number(event.ts || Date.now() / 1000),
    stream: String(event.stream || "mix"),
    text: String(event.text || ""),
    overload: Boolean(event.overload)
  };
  const transcript = state.session.transcript || [];
  const last = transcript[transcript.length - 1];
  if (last && last.ts === line.ts && last.stream === line.stream && last.text === line.text) {
    return state;
  }
  return {
    ...state,
    session: {
      ...state.session,
      transcript: [...transcript, line].slice(-200)
    }
  };
}

function applyAsrMetrics(state, event) {
  if (!state?.session) {
    return state;
  }
  return {
    ...state,
    session: {
      ...state.session,
      asrMetrics: {
        segDroppedTotal: Number(event.seg_dropped_total ?? state.session.asrMetrics?.segDroppedTotal ?? 0),
        segSkippedTotal: Number(event.seg_skipped_total ?? state.session.asrMetrics?.segSkippedTotal ?? 0),
        avgLatencyS: Number(event.avg_latency_s ?? state.session.asrMetrics?.avgLatencyS ?? 0),
        p95LatencyS: Number(event.p95_latency_s ?? state.session.asrMetrics?.p95LatencyS ?? 0),
        lagS: Number(event.lag_s ?? state.session.asrMetrics?.lagS ?? 0)
      }
    }
  };
}
