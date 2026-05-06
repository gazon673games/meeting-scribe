import React from "react";

import { FALLBACK_OPTIONS, makeSettingsDraft } from "../entities/settings/model";
import { meetingScribeClient } from "../shared/api/meetingScribeClient";
import { useAppActions } from "./useMeetingScribeApp/actions";
import { handleBackendEvent } from "./useMeetingScribeApp/backendEvents";
import { emptyResourceUsage } from "./useMeetingScribeApp/constants";
import { normalizeDevicesResult, readCachedDevices, writeCachedDevices } from "./useMeetingScribeApp/devices";
import {
  useBackendEventSubscription,
  useContentProtectionSync,
  useResourceUsagePolling,
  useRuntimeStatePolling,
  useSettingsDraftSync
} from "./useMeetingScribeApp/effects";
import { formatRequestError } from "./useMeetingScribeApp/errors";
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

  const refresh = React.useCallback(async (options = {}) => {
    const includeDevices = options?.includeDevices !== false;
    const showLoading = options?.showLoading !== false;
    if (showLoading) setLoading(true);
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
      if (includeDevices) refreshDevices();
    } catch (requestError) {
      setError(formatRequestError(requestError));
    } finally {
      if (showLoading) setLoading(false);
    }
  }, [refreshDevices]);

  const appView = buildAppViewModel(state, settingsDraft, config);

  const eventHandler = React.useCallback((event) => {
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
  }, [refresh]);

  useBackendEventSubscription({ refresh, handlers: eventHandler });
  useSettingsDraftSync({
    config,
    settingsDirty,
    stateOptions: state?.options,
    setSettingsDraft
  });
  useContentProtectionSync({ config, settingsDirty, settingsDraft, setError });
  useRuntimeStatePolling({ isRunning: Boolean(state?.session?.running), setState, setError });
  useResourceUsagePolling({
    fastPolling: Boolean(state?.session?.running || state?.session?.offlinePass?.running),
    setResourceUsage
  });

  const actions = useAppActions({
    appOptions: appView.options || FALLBACK_OPTIONS,
    config,
    refresh,
    runState: Boolean(appView.session.running),
    setAssistantPing,
    setConfig,
    setError,
    setLocalLlmStatus,
    setRuntimeNotices,
    setSavingSettings,
    setSettingsDirty,
    setSettingsDraft,
    setState,
    settingsDraft
  });

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
    resourceUsage,
    runtimeNotices,
    settingsDirty,
    settingsDraft,
    savingSettings,
    ...actions
  };
}
