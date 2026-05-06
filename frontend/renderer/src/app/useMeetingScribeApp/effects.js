import React from "react";

import { FALLBACK_OPTIONS, makeSettingsDraft } from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { emptyResourceUsage } from "./constants";
import { formatRequestError } from "./errors";
import { applyLockedProfileDefaults } from "./settingsDraft";
import { mergeBackendStateSnapshot } from "./transcriptState";

export function useBackendEventSubscription({ refresh, handlers }) {
  React.useEffect(() => {
    refresh();
    return meetingScribeClient.onBackendEvent((event) => handlers(event));
  }, [handlers, refresh]);
}

export function useSettingsDraftSync({ config, settingsDirty, stateOptions, setSettingsDraft }) {
  React.useEffect(() => {
    if (!settingsDirty) {
      const next = applyLockedProfileDefaults(makeSettingsDraft(config), stateOptions || FALLBACK_OPTIONS);
      setSettingsDraft(next);
    }
  }, [config, settingsDirty, stateOptions, setSettingsDraft]);
}

export function useContentProtectionSync({ config, settingsDirty, settingsDraft, setError }) {
  React.useEffect(() => {
    if (!config) return;
    const savedDraft = makeSettingsDraft(config);
    const enabled = settingsDirty ? settingsDraft.screenCaptureProtection : savedDraft.screenCaptureProtection;
    meetingScribeClient.setContentProtection(Boolean(enabled)).catch((requestError) => {
      setError(formatRequestError(requestError));
    });
  }, [config, settingsDirty, settingsDraft.screenCaptureProtection, setError]);
}

export function useRuntimeStatePolling({ isRunning, setState, setError }) {
  React.useEffect(() => {
    if (!isRunning) {
      return undefined;
    }
    const handle = window.setInterval(() => {
      meetingScribeClient
        .request("get_runtime_state")
        .then((result) => setState((current) => mergeBackendStateSnapshot(current, result)))
        .catch((requestError) => setError(formatRequestError(requestError)));
    }, 600);
    return () => window.clearInterval(handle);
  }, [isRunning, setError, setState]);
}

export function useResourceUsagePolling({ fastPolling, setResourceUsage }) {
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
    const handle = window.setInterval(loadResourceUsage, fastPolling ? 2000 : 5000);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [fastPolling, setResourceUsage]);
}
