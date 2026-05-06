import React from "react";

import { applySettingsToConfig, draftToStartParams } from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { formatRequestError } from "./errors";
import { removeSessionSource, upsertSessionSource } from "./sessionState";
import { applyProfileDefaults, mergeSettingsPatch } from "./settingsDraft";
import { mergeBackendStateSnapshot } from "./transcriptState";

export function useAppActions({
  appOptions,
  config,
  refresh,
  runState,
  setAssistantPing,
  setConfig,
  setError,
  setLocalLlmStatus,
  setRuntimeNotices,
  setSavingSettings,
  setSettingsDirty,
  setSettingsDraft,
  setState,
  settingsDraft,
}) {
  const updateSettings = React.useCallback((patch) => {
    setSettingsDraft((current) => mergeSettingsPatch(current, patch, appOptions));
    setSettingsDirty(true);
  }, [appOptions, setSettingsDirty, setSettingsDraft]);

  const clearError = React.useCallback(() => {
    setError("");
  }, [setError]);

  const dismissRuntimeNotice = React.useCallback((key) => {
    setRuntimeNotices((current) => current.filter((notice) => notice.key !== key));
  }, [setRuntimeNotices]);

  const reloadApp = React.useCallback(async () => {
    setError("");
    try {
      await meetingScribeClient.reloadApp();
    } catch (requestError) {
      setError(formatRequestError(requestError));
    }
  }, [setError]);

  const openDebugConsole = React.useCallback(async () => {
    try {
      await meetingScribeClient.showDebugConsole();
    } catch (requestError) {
      setError(formatRequestError(requestError));
    }
  }, [setError]);

  const startAssistantLogin = React.useCallback(async (providerId = "codex") => {
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
  }, [refresh, setError]);

  const pingAssistantProvider = React.useCallback((providerId = "codex", profileId = "") => {
    setError("");
    setAssistantPing({ busy: true, providerId, profileId });
    meetingScribeClient.request("ping_assistant_provider", { providerId, profileId }).catch((requestError) => {
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
  }, [setAssistantPing, setError]);

  const startLocalModel = React.useCallback((profileId) => {
    setLocalLlmStatus((current) => ({ ...current, [profileId]: { state: "starting", message: "" } }));
    meetingScribeClient.request("start_local_llm", { profileId }).catch(() => {
      setLocalLlmStatus((current) => ({ ...current, [profileId]: { state: "error", message: "Failed to start" } }));
    });
  }, [setLocalLlmStatus]);

  const stopLocalModel = React.useCallback((profileId) => {
    meetingScribeClient.request("stop_local_llm", { profileId }).then(() => {
      setLocalLlmStatus((current) => ({ ...current, [profileId]: { state: "stopped", message: "" } }));
    }).catch(() => {});
  }, [setLocalLlmStatus]);

  const updateAsrSetting = React.useCallback((key, value) => {
    setSettingsDraft((current) => ({ ...current, asr: { ...current.asr, [key]: value } }));
    setSettingsDirty(true);
  }, [setSettingsDirty, setSettingsDraft]);

  const applyProfile = React.useCallback((profile) => {
    setSettingsDraft((current) => applyProfileDefaults(current, profile, appOptions));
    setSettingsDirty(true);
  }, [appOptions, setSettingsDirty, setSettingsDraft]);

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
  }, [config, setConfig, setError, setSavingSettings, setSettingsDirty, setState, settingsDraft]);

  const runBackendAction = React.useCallback(async (method, params = {}) => {
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
  }, [refresh, setError, setState]);

  const startOrStop = React.useCallback(() => {
    if (runState) {
      runBackendAction("stop_session", {
        runOfflinePass: false,
        outputFile: settingsDraft.outputFile,
        language: settingsDraft.language,
        model: settingsDraft.model
      });
      return;
    }
    runBackendAction("start_session", draftToStartParams(settingsDraft));
  }, [runBackendAction, runState, settingsDraft]);

  return {
    applyProfile,
    clearError,
    dismissRuntimeNotice,
    openDebugConsole,
    pingAssistantProvider,
    reloadApp,
    runBackendAction,
    saveSettings,
    startAssistantLogin,
    startLocalModel,
    startOrStop,
    stopLocalModel,
    updateAsrSetting,
    updateSettings,
  };
}
