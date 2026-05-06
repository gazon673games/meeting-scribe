import React from "react";

import {
  FALLBACK_OPTIONS,
  applySettingsToConfig,
  draftToStartParams,
  makeSettingsDraft
} from "../entities/settings/model";
import { meetingScribeClient } from "../shared/api/meetingScribeClient";
import { handleBackendEvent } from "./useMeetingScribeApp/backendEvents";
import { emptyResourceUsage } from "./useMeetingScribeApp/constants";
import { normalizeDevicesResult, readCachedDevices, writeCachedDevices } from "./useMeetingScribeApp/devices";
import { formatRequestError } from "./useMeetingScribeApp/errors";
import { removeSessionSource, upsertSessionSource } from "./useMeetingScribeApp/sessionState";
import { applyLockedProfileDefaults, applyProfileDefaults, mergeSettingsPatch } from "./useMeetingScribeApp/settingsDraft";
import { mergeBackendStateSnapshot } from "./useMeetingScribeApp/transcriptState";
import { buildAppViewModel } from "./useMeetingScribeApp/viewModel";

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
  const [resourceUsage, setResourceUsage] = React.useState(() => emptyResourceUsage());
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
        errors: [`devices: ${formatRequestError(requestError)}`]
      }));
    }
  }, []);

  const refresh = React.useCallback(
    async (options = {}) => {
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
        setError(formatRequestError(requestError));
      } finally {
        if (showLoading) {
          setLoading(false);
        }
      }
    },
    [refreshDevices]
  );

  React.useEffect(() => {
    refresh();
    return meetingScribeClient.onBackendEvent((event) => {
      handleBackendEvent(event, {
        refresh,
        setAssistantPing,
        setBackendStatus,
        setError,
        setEvents,
        setLocalLlmStatus,
        setRuntimeNotices,
        setState
      });
    });
  }, [refresh]);

  React.useEffect(() => {
    if (!settingsDirty) {
      setSettingsDraft(applyLockedProfileDefaults(makeSettingsDraft(config), state?.options || FALLBACK_OPTIONS));
    }
  }, [config, settingsDirty, state?.options]);

  React.useEffect(() => {
    if (!config) {
      return;
    }
    const savedDraft = makeSettingsDraft(config);
    const enabled = settingsDirty ? settingsDraft.screenCaptureProtection : savedDraft.screenCaptureProtection;
    meetingScribeClient
      .setContentProtection(Boolean(enabled))
      .catch((requestError) => setError(formatRequestError(requestError)));
  }, [config, settingsDirty, settingsDraft.screenCaptureProtection]);

  React.useEffect(() => {
    if (!state?.session?.running) {
      return undefined;
    }
    const handle = window.setInterval(() => {
      meetingScribeClient
        .request("get_runtime_state")
        .then((result) => setState((current) => mergeBackendStateSnapshot(current, result)))
        .catch((requestError) => setError(formatRequestError(requestError)));
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
            setResourceUsage(result || emptyResourceUsage());
          }
        })
        .catch(() => {
          if (!cancelled) {
            setResourceUsage(emptyResourceUsage());
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

  const appView = buildAppViewModel(state, settingsDraft, config);

  const updateSettings = React.useCallback(
    (patch) => {
      setSettingsDraft((current) => mergeSettingsPatch(current, patch, appView.options));
      setSettingsDirty(true);
    },
    [appView.options]
  );

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
      setError(formatRequestError(requestError));
    }
  }, []);

  const openDebugConsole = React.useCallback(async () => {
    try {
      await meetingScribeClient.showDebugConsole();
    } catch (requestError) {
      setError(formatRequestError(requestError));
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
        setError(formatRequestError(requestError));
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
          message: formatRequestError(requestError),
          errorCode: "ping_failed",
          retryable: true,
          ts: Date.now()
        });
      });
  }, []);

  const startLocalModel = React.useCallback((profileId) => {
    setLocalLlmStatus((current) => ({ ...current, [profileId]: { state: "starting", message: "" } }));
    meetingScribeClient.request("start_local_llm", { profileId }).catch(() => {
      setLocalLlmStatus((current) => ({ ...current, [profileId]: { state: "error", message: "Failed to start" } }));
    });
  }, []);

  const stopLocalModel = React.useCallback((profileId) => {
    meetingScribeClient
      .request("stop_local_llm", { profileId })
      .then(() => {
        setLocalLlmStatus((current) => ({ ...current, [profileId]: { state: "stopped", message: "" } }));
      })
      .catch(() => {});
  }, []);

  const updateAsrSetting = React.useCallback((key, value) => {
    setSettingsDraft((current) => ({ ...current, asr: { ...current.asr, [key]: value } }));
    setSettingsDirty(true);
  }, []);

  const applyProfile = React.useCallback(
    (profile) => {
      setSettingsDraft((current) => applyProfileDefaults(current, profile, appView.options));
      setSettingsDirty(true);
    },
    [appView.options]
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
      setError(formatRequestError(requestError));
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
        setError(formatRequestError(requestError));
      }
    },
    [refresh]
  );

  const startOrStop = React.useCallback(() => {
    if (appView.session.running) {
      runBackendAction("stop_session", {
        runOfflinePass: false,
        outputFile: settingsDraft.outputFile,
        language: settingsDraft.language,
        model: settingsDraft.model
      });
      return;
    }
    runBackendAction("start_session", draftToStartParams(settingsDraft));
  }, [appView.session.running, runBackendAction, settingsDraft]);

  return {
    ...appView,
    assistantPing,
    backendStatus,
    devices,
    error,
    events,
    loading,
    localLlmStatus,
    refresh,
    reloadApp,
    openDebugConsole,
    resourceUsage,
    runtimeNotices,
    settingsDirty,
    settingsDraft,
    savingSettings,
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
    updateSettings,
    applyProfile
  };
}
