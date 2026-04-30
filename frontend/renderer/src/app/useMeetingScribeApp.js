import React from "react";

import { ASR_FIELDS, FALLBACK_OPTIONS, applySettingsToConfig, draftToStartParams, makeSettingsDraft } from "../entities/settings/model";
import { noticeFromBackendEvent, upsertRuntimeNotice } from "../entities/runtimeNotice/model";
import { meetingScribeClient } from "../shared/api/meetingScribeClient";

const REFRESHING_EVENTS = [
  "session_started",
  "session_stopped",
  "transcript_cleared",
  "assistant_started",
  "assistant_fallback",
  "assistant_result",
  "offline_pass_started",
  "offline_pass_done",
  "offline_pass_error"
];
const DEVICE_CACHE_KEY = "meeting-scribe.devices.v1";

export function useMeetingScribeApp() {
  const [backendStatus, setBackendStatus] = React.useState({ ready: false, running: false, lastError: "" });
  const [state, setState] = React.useState(null);
  const [config, setConfig] = React.useState(null);
  const [settingsDraft, setSettingsDraft] = React.useState(() => makeSettingsDraft(null));
  const [settingsDirty, setSettingsDirty] = React.useState(false);
  const [savingSettings, setSavingSettings] = React.useState(false);
  const [devices, setDevices] = React.useState(() => readCachedDevices());
  const [events, setEvents] = React.useState([]);
  const [error, setError] = React.useState("");
  const [loading, setLoading] = React.useState(true);
  const [resourceUsage, setResourceUsage] = React.useState({ app: {}, system: {}, gpus: [] });
  const [runtimeNotices, setRuntimeNotices] = React.useState([]);
  const [assistantPing, setAssistantPing] = React.useState({ busy: false });
  const [localLlmStatus, setLocalLlmStatus] = React.useState({});

  const refreshDevices = React.useCallback(async () => {
    try {
      const devicesResult = normalizeDevicesResult(await meetingScribeClient.request("list_devices"));
      setDevices(devicesResult);
      writeCachedDevices(devicesResult);
    } catch (requestError) {
      setDevices((current) => ({
        ...(current || { loopback: [], input: [] }),
        errors: [`devices: ${requestError.name || "Error"}: ${requestError.message || requestError}`]
      }));
    }
  }, []);

  const refresh = React.useCallback(async (options = {}) => {
    const includeDevices = options?.includeDevices !== false;
    const showLoading = options?.showLoading !== false;
    if (showLoading) {
      setLoading(true);
    }
    setError("");
    try {
      const [statusResult, stateResult, configResult] = await Promise.all([
        meetingScribeClient.status(),
        meetingScribeClient.request("get_state"),
        meetingScribeClient.request("get_config")
      ]);
      setBackendStatus(statusResult);
      setState((current) => mergeBackendStateSnapshot(current, stateResult));
      setConfig(configResult);
      if (includeDevices) {
        refreshDevices();
      }
    } catch (requestError) {
      setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`);
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  }, [refreshDevices]);

  React.useEffect(() => {
    refresh();
    return meetingScribeClient.onBackendEvent((event) => {
      setEvents((current) => [event, ...current].slice(0, 8));
      const runtimeNotice = noticeFromBackendEvent(event);
      if (runtimeNotice) {
        setRuntimeNotices((current) => upsertRuntimeNotice(current, runtimeNotice));
      }
      if (event.type === "transcript_line") {
        setState((current) => appendTranscriptLine(current, event));
      }
      if (event.type === "transcript_line_update") {
        setState((current) => updateTranscriptLine(current, event));
      }
      if (event.type === "asr_metrics") {
        setState((current) => applyAsrMetrics(current, event));
      }
      if (event.type === "assistant_ping_result") {
        setAssistantPing((current) => ({ ...current, busy: false, ...event, ts: Date.now() }));
      }
      if (event.type === "local_llm_status") {
        setLocalLlmStatus((current) => ({ ...current, [event.profileId]: { state: event.state, message: event.message || "" } }));
      }
      if (event.type === "backend_ready") {
        setBackendStatus((current) => ({ ...current, ready: true, running: true }));
      }
      if (event.type === "backend_exit") {
        setBackendStatus((current) => ({ ...current, ready: false, running: false, lastError: "Python backend stopped" }));
        setError(`Python backend stopped (${event.code ?? "null"}, ${event.signal ?? "null"})`);
      }
      if (event.type === "source_added" && event.source) {
        setState((current) => upsertSessionSource(current, event.source));
      }
      if (event.type === "source_removed" && event.source) {
        setState((current) => removeSessionSource(current, event.source));
      }
      if (event.type === "source_updated" && event.source) {
        setState((current) => upsertSessionSource(current, event.source));
      }
      if (REFRESHING_EVENTS.includes(event.type)) {
        refresh({ includeDevices: false, showLoading: false });
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
        .request("get_runtime_state")
        .then((result) => setState((current) => mergeBackendStateSnapshot(current, result)))
        .catch((requestError) => setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`));
    }, 600);
    return () => window.clearInterval(handle);
  }, [state?.session?.running]);

  const fastResourcePolling = Boolean(state?.session?.running || state?.session?.offlinePass?.running);

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
    const handle = window.setInterval(loadResourceUsage, fastResourcePolling ? 2000 : 5000);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [fastResourcePolling]);

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

  const dismissRuntimeNotice = React.useCallback((key) => {
    setRuntimeNotices((current) => current.filter((notice) => notice.key !== key));
  }, []);

  const reloadApp = React.useCallback(async () => {
    setError("");
    try {
      await meetingScribeClient.reloadApp();
    } catch (requestError) {
      setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`);
    }
  }, []);

  const openDebugConsole = React.useCallback(async () => {
    try {
      await meetingScribeClient.showDebugConsole();
    } catch (requestError) {
      setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`);
    }
  }, []);

  const startAssistantLogin = React.useCallback(
    async (providerId = "codex") => {
      setError("");
      try {
        const result = await meetingScribeClient.request("start_assistant_login", { providerId });
        await refresh();
        if (result?.errorCode) {
          setError(result.suggestion || result.message || result.errorCode);
        }
      } catch (requestError) {
        setError(`${requestError.name || "Error"}: ${requestError.message || requestError}`);
      }
    },
    [refresh]
  );

  const pingAssistantProvider = React.useCallback((providerId = "codex", profileId = "") => {
    setError("");
    setAssistantPing({ busy: true, providerId, profileId });
    meetingScribeClient
      .request("ping_assistant_provider", { providerId, profileId })
      .catch((requestError) => {
        setAssistantPing({
          busy: false,
          providerId,
          profileId,
          ok: false,
          message: `${requestError.name || "Error"}: ${requestError.message || requestError}`,
          errorCode: "ping_failed",
          retryable: true,
          ts: Date.now()
        });
      });
  }, []);

  const startLocalModel = React.useCallback((profileId) => {
    setLocalLlmStatus((s) => ({ ...s, [profileId]: { state: "starting", message: "" } }));
    meetingScribeClient.request("start_local_llm", { profileId }).catch(() => {
      setLocalLlmStatus((s) => ({ ...s, [profileId]: { state: "error", message: "Failed to start" } }));
    });
  }, []);

  const stopLocalModel = React.useCallback((profileId) => {
    meetingScribeClient.request("stop_local_llm", { profileId }).then(() => {
      setLocalLlmStatus((s) => ({ ...s, [profileId]: { state: "stopped", message: "" } }));
    }).catch(() => {});
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
      setState((current) => mergeBackendStateSnapshot(current, stateResult));
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
        } else if (method === "add_source" || method === "set_source_enabled") {
          setState((current) => upsertSessionSource(current, result));
        } else if (method === "remove_source") {
          setState((current) => removeSessionSource(current, result));
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
    assistantPing,
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
    openDebugConsole,
    resourceUsage,
    runtimeNotices,
    localLlmStatus,
    pingAssistantProvider,
    startLocalModel,
    stopLocalModel,
    runBackendAction,
    saveSettings,
    startAssistantLogin,
    startOrStop,
    updateAsrSetting,
    clearError,
    dismissRuntimeNotice,
    updateSettings
  };
}

function appendTranscriptLine(state, event) {
  if (!state?.session || !String(event?.text || "").trim()) {
    return state;
  }
  const line = {
    id: String(event.id || transcriptLineId(event)),
    ts: Number(event.ts || Date.now() / 1000),
    stream: String(event.stream || "mix"),
    speaker: String(event.speaker || ""),
    t_start: optionalNumber(event.t_start),
    t_end: optionalNumber(event.t_end),
    text: String(event.text || ""),
    overload: Boolean(event.overload)
  };
  const transcript = state.session.transcript || [];
  const existingIndex = transcript.findIndex((item) => transcriptLineKey(item) === transcriptLineKey(line));
  if (existingIndex >= 0) {
    const existing = transcript[existingIndex];
    if (existing.ts === line.ts && existing.stream === line.stream && existing.speaker === line.speaker && existing.text === line.text) {
      return state;
    }
    return {
      ...state,
      session: {
        ...state.session,
        transcript: transcript.map((item, index) => (index === existingIndex ? { ...item, ...line } : item))
      }
    };
  }
  return {
    ...state,
    session: {
      ...state.session,
      transcript: [...transcript, line].slice(-200)
    }
  };
}

function updateTranscriptLine(state, event) {
  if (!state?.session) {
    return state;
  }
  const id = String(event?.id || "");
  const transcript = state.session.transcript || [];
  let changed = false;
  const updated = transcript.map((line) => {
    if (!sameTranscriptLine(line, event, id)) {
      return line;
    }
    changed = true;
    return {
      ...line,
      speaker: String(event.speaker || line.speaker || ""),
      speakerSource: String(event.speakerSource || line.speakerSource || ""),
      speakerConfidence: event.speakerConfidence ?? line.speakerConfidence
    };
  });
  if (!changed) {
    return state;
  }
  return {
    ...state,
    session: {
      ...state.session,
      transcript: updated
    }
  };
}

function sameTranscriptLine(line, event, id) {
  if (id && String(line.id || "") === id) {
    return true;
  }
  return (
    String(line.stream || "") === String(event?.stream || "") &&
    Number(line.t_start ?? -1) === Number(event?.t_start ?? -2) &&
    Number(line.t_end ?? -1) === Number(event?.t_end ?? -2)
  );
}

function transcriptLineId(event) {
  const stream = String(event?.stream || "mix").replace(/[^a-zA-Z0-9_-]+/g, "_") || "mix";
  const ts = Math.round(Number(event?.ts || Date.now() / 1000) * 1000);
  const start = optionalNumber(event?.t_start);
  const end = optionalNumber(event?.t_end);
  return `${stream}:${Math.round((start ?? event?.ts ?? 0) * 1000) || ts}:${Math.round((end ?? event?.ts ?? 0) * 1000) || ts}`;
}

function optionalNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
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

function mergeBackendStateSnapshot(current, next) {
  if (!next) {
    return current;
  }
  if (!current) {
    return next;
  }
  const merged = { ...current, ...next };
  if (!current?.session || !next?.session) {
    return merged;
  }
  if (!current.session.running || !next.session.running) {
    return merged;
  }
  return {
    ...merged,
    session: {
      ...next.session,
      transcript: mergeTranscriptLines(current.session.transcript || [], next.session.transcript || [])
    }
  };
}

function mergeTranscriptLines(currentLines, nextLines) {
  const merged = [];
  const indexByKey = new Map();
  const appendOrUpdate = (line) => {
    if (!line || !String(line.text || "").trim()) {
      return;
    }
    const key = transcriptLineKey(line);
    const existingIndex = indexByKey.get(key);
    if (existingIndex === undefined) {
      indexByKey.set(key, merged.length);
      merged.push(line);
      return;
    }
    merged[existingIndex] = { ...merged[existingIndex], ...line };
  };
  currentLines.forEach(appendOrUpdate);
  nextLines.forEach(appendOrUpdate);
  return merged.sort(compareTranscriptLines).slice(-200);
}

function transcriptLineKey(line) {
  if (line?.id) {
    return String(line.id);
  }
  return [
    line?.stream || "mix",
    optionalNumber(line?.t_start) ?? "",
    optionalNumber(line?.t_end) ?? "",
    optionalNumber(line?.ts) ?? "",
    String(line?.text || "")
  ].join(":");
}

function compareTranscriptLines(left, right) {
  const leftStart = optionalNumber(left?.t_start) ?? optionalNumber(left?.ts) ?? 0;
  const rightStart = optionalNumber(right?.t_start) ?? optionalNumber(right?.ts) ?? 0;
  if (leftStart !== rightStart) {
    return leftStart - rightStart;
  }
  return (optionalNumber(left?.ts) ?? 0) - (optionalNumber(right?.ts) ?? 0);
}

function normalizeDevicesResult(result) {
  return {
    loopback: Array.isArray(result?.loopback) ? result.loopback : [],
    input: Array.isArray(result?.input) ? result.input : [],
    errors: Array.isArray(result?.errors) ? result.errors : []
  };
}

function readCachedDevices() {
  try {
    const raw = window.localStorage?.getItem(DEVICE_CACHE_KEY);
    return normalizeDevicesResult(raw ? JSON.parse(raw) : null);
  } catch {
    return normalizeDevicesResult(null);
  }
}

function writeCachedDevices(devices) {
  try {
    window.localStorage?.setItem(DEVICE_CACHE_KEY, JSON.stringify(normalizeDevicesResult(devices)));
  } catch {
    // Device cache is only a startup convenience.
  }
}

function upsertSessionSource(state, source) {
  if (!state?.session || !source?.name) {
    return state;
  }
  const sources = state.session.sources || [];
  const index = sources.findIndex((item) => item.name === source.name);
  const nextSources =
    index >= 0
      ? sources.map((item, itemIndex) => (itemIndex === index ? { ...item, ...source } : item))
      : [...sources, source];
  return {
    ...state,
    session: {
      ...state.session,
      sources: nextSources
    }
  };
}

function removeSessionSource(state, source) {
  if (!state?.session) {
    return state;
  }
  const sourceName = String(source?.name || source || "");
  if (!sourceName) {
    return state;
  }
  return {
    ...state,
    session: {
      ...state.session,
      sources: (state.session.sources || []).filter((item) => item.name !== sourceName)
    }
  };
}
