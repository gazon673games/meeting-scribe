import React from 'react';

import {
  FALLBACK_OPTIONS,
  makeSettingsDraft,
} from '../../entities/settings/model';
import { meetingScribeClient } from '../../shared/api/meetingScribeClient';
import { emptyResourceUsage } from './constants';
import { formatRequestError } from './errors';
import { applyLockedProfileDefaults } from './settingsDraft';
import { mergeBackendStateSnapshot } from './transcriptState';

export const useBackendEventSubscription = ({ refresh, handler }) => {
  React.useEffect(() => {
    refresh();
    return meetingScribeClient.onBackendEvent(handler);
  }, [handler, refresh]);
};

export const useSettingsDraftSync = ({
  config,
  settingsDirty,
  stateOptions,
  setSettingsDraft,
}) => {
  React.useEffect(() => {
    if (!settingsDirty) {
      const next = applyLockedProfileDefaults(
        makeSettingsDraft(config),
        stateOptions || FALLBACK_OPTIONS,
      );
      setSettingsDraft(next);
    }
  }, [config, settingsDirty, stateOptions, setSettingsDraft]);
};

export const useContentProtectionSync = ({
  config,
  settingsDirty,
  settingsDraft,
  setError,
}) => {
  React.useEffect(() => {
    if (!config) return;
    const savedDraft = makeSettingsDraft(config);
    const enabled = settingsDirty
      ? settingsDraft.screenCaptureProtection
      : savedDraft.screenCaptureProtection;
    meetingScribeClient
      .setContentProtection(Boolean(enabled))
      .catch((requestError) => {
        setError(formatRequestError(requestError));
      });
  }, [config, settingsDirty, settingsDraft.screenCaptureProtection, setError]);
};

export const useRuntimeStatePolling = ({ isRunning, setState, setError }) => {
  React.useEffect(() => {
    if (!isRunning) {
      return undefined;
    }
    let isCancelled = false;
    let isPending = false;
    const loadRuntimeState = async () => {
      if (isPending) return;
      isPending = true;
      try {
        const result = await meetingScribeClient.request('get_runtime_state');
        if (!isCancelled) {
          setState((current) => mergeBackendStateSnapshot(current, result));
        }
      } catch (requestError) {
        if (!isCancelled) setError(formatRequestError(requestError));
      } finally {
        isPending = false;
      }
    };
    const handle = window.setInterval(loadRuntimeState, 600);
    return () => {
      isCancelled = true;
      window.clearInterval(handle);
    };
  }, [isRunning, setError, setState]);
};

export const useResourceUsagePolling = ({ fastPolling, setResourceUsage }) => {
  React.useEffect(() => {
    let isCancelled = false;
    let isPending = false;
    const loadResourceUsage = async () => {
      if (isPending) return;
      isPending = true;
      try {
        const result = await meetingScribeClient.resourceUsage();
        if (!isCancelled) setResourceUsage(result || emptyResourceUsage());
      } catch (requestError) {
        if (!isCancelled) {
          console.warn('[resources] resource polling failed', requestError);
          setResourceUsage(emptyResourceUsage());
        }
      } finally {
        isPending = false;
      }
    };
    loadResourceUsage();
    const handle = window.setInterval(
      loadResourceUsage,
      fastPolling ? 2000 : 5000,
    );
    return () => {
      isCancelled = true;
      window.clearInterval(handle);
    };
  }, [fastPolling, setResourceUsage]);
};
