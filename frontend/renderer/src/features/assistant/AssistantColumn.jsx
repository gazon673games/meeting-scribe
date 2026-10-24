import { classNames } from '../../shared/lib/classNames';
import React from 'react';
import { Play, Radio, Square } from 'lucide-react';

import { PipelinePanel } from '../../shared/ui/pipeline/PipelinePanel';
import { AssistantProfileSelect } from './AssistantProfileSelect';
import { AssistantPromptBox } from './AssistantPromptBox';
import { AssistantPingStatus } from './AssistantPingStatus';
import { AssistantQuickActions } from './AssistantQuickActions';
import { AssistantResponse } from './AssistantResponse';
import { AssistantStats } from './AssistantStats';

const ACTION_LABELS = {
  answer: 'Answer Latest',
  summary: 'Summarize',
  action_items: 'Action Items',
  risk_check: 'Risk Check',
};

const isAssistantResponseAppendable = (response, lastTs) => {
  if (!response || response.ts === null || response.ts === undefined) {
    return false;
  }
  if (response.ts === lastTs) return false;
  if (!response.text && response.ok !== false) return false;
  return true;
};

const useAssistantChat = (lastResponse) => {
  const [chatMessages, setChatMessages] = React.useState([]);
  const lastResponseRef = React.useRef(null);
  const nextMessageId = React.useRef(0);

  React.useEffect(() => {
    if (!isAssistantResponseAppendable(lastResponse, lastResponseRef.current)) {
      return;
    }
    lastResponseRef.current = lastResponse.ts;
    nextMessageId.current += 1;
    const message = {
      ...lastResponse,
      id: `assistant-${nextMessageId.current}`,
      role: 'assistant',
    };
    setChatMessages((prev) => [...prev, message]);
  }, [lastResponse]);

  const pushUserMessage = React.useCallback((text, sourceLabel) => {
    nextMessageId.current += 1;
    const message = {
      id: `user-${nextMessageId.current}`,
      role: 'user',
      text,
      sourceLabel,
    };
    setChatMessages((prev) => [...prev, message]);
  }, []);

  return { chatMessages, pushUserMessage };
};

const selectProfile = (profiles, profileId) =>
  profiles.find((profile) => profile.id === profileId) || {};

const LocalLlmStatusText = ({ status }) => {
  if (!status) {
    return (
      <span className="ping-status" style={{ color: 'var(--muted)' }}>
        not started
      </span>
    );
  }
  if (status.state === 'running') {
    return <span className="ping-status ping-ok">ready</span>;
  }
  if (status.state === 'starting') {
    return <span className="ping-status ping-busy">starting...</span>;
  }
  if (status.state === 'error') {
    return (
      <span className="ping-status ping-err" title={status.message}>
        error: {status.message || 'failed to start'}
      </span>
    );
  }
  if (status.state === 'stopped') {
    return (
      <span className="ping-status" style={{ color: 'var(--muted)' }}>
        stopped
      </span>
    );
  }
  return null;
};

const AssistantRuntimeBar = ({
  assistantPing,
  disabled,
  onPing,
  onStartLocalModel,
  onStopLocalModel,
  profileId,
  runtime,
}) => {
  if (!profileId) {
    return null;
  }
  if (!runtime.canManageModel) {
    return (
      <div className="local-model-bar">
        <button
          className="assistant-ping-btn"
          disabled={disabled || assistantPing?.busy}
          type="button"
          onClick={() => onPing?.(runtime.selectedProviderId, profileId)}
        >
          <Radio size={13} />
          Ping
        </button>
        <AssistantPingStatus idleRequiresTs ping={assistantPing} />
      </div>
    );
  }
  return (
    <div className="local-model-bar">
      {!runtime.llmRunning && !runtime.llmStarting ? (
        <button
          className="assistant-ping-btn local-start-btn"
          disabled={disabled}
          type="button"
          onClick={() => onStartLocalModel?.(profileId)}
        >
          <Play size={13} />
          Start Model
        </button>
      ) : (
        <button
          className="assistant-ping-btn local-stop-btn"
          disabled={disabled || runtime.llmStarting}
          type="button"
          onClick={() => onStopLocalModel?.(profileId)}
        >
          <Square size={13} />
          {runtime.llmStarting ? 'Starting...' : 'Stop Model'}
        </button>
      )}
      <LocalLlmStatusText status={runtime.llmStatus} />
      <button
        className="assistant-ping-btn"
        disabled={disabled || assistantPing?.busy}
        onClick={() => onPing?.(runtime.selectedProviderId, profileId)}
        type="button"
        title="Check the selected model connection"
      >
        <Radio size={13} />
        Ping
      </button>
      <AssistantPingStatus
        idleRequiresTs
        ping={assistantPing?.profileId === profileId ? assistantPing : null}
      />
    </div>
  );
};

const localModelMessage = (status) => {
  if (status.state === 'starting') return 'Model is starting...';
  if (status.state === 'error') {
    return status.message || 'Model failed to start.';
  }
  return 'Model is stopped.';
};

const providerSnapshot = (assistant, selectedProviderId) => {
  const provider = assistant.providers?.find(
    (item) => item.id === selectedProviderId,
  );
  if (provider) {
    return {
      ...assistant,
      providerAvailable: provider.available,
      providerMessage: provider.message,
      providerSuggestion: provider.suggestion,
      providerErrorCode: provider.errorCode,
      providerLoginSupported: provider.loginSupported,
    };
  }
  return {
    ...assistant,
    providerAvailable:
      assistant.providerId === selectedProviderId
        ? assistant.providerAvailable
        : undefined,
  };
};

const applyLocalModelStatus = (providerState, llmStatus) => {
  if (!llmStatus?.state) return;
  if (llmStatus.state === 'running') {
    providerState.providerAvailable = true;
    return;
  }
  providerState.providerAvailable = false;
  providerState.providerMessage = localModelMessage(llmStatus);
};

const applyProviderPing = (
  providerState,
  { assistantPing, llmStatus, selectedProviderId, profileId },
) => {
  if (!assistantPing?.ts || assistantPing.busy) return;
  if (!(assistantPing.ts >= (llmStatus?.ts || 0))) return;
  if (assistantPing.providerId !== selectedProviderId) return;
  if (assistantPing.profileId && assistantPing.profileId !== profileId) return;
  providerState.providerAvailable = assistantPing.ok;
  providerState.providerMessage = assistantPing.message;
  providerState.providerSuggestion = assistantPing.suggestion;
  providerState.providerErrorCode = assistantPing.errorCode;
};

const applyProviderHints = (providerState, isCodexProfile) => {
  if (providerState.providerAvailable === undefined) {
    providerState.providerMessage = 'Connection not checked';
    providerState.providerSuggestion = isCodexProfile
      ? 'Use Ping to check the selected provider, or send a request.'
      : 'Start the selected model or use Ping to check its connection.';
  }
  if (
    providerState.providerAvailable === false &&
    !isCodexProfile &&
    !providerState.providerSuggestion
  ) {
    providerState.providerSuggestion =
      'Start the selected model, or check its connection in Settings.';
  }
};

const buildAssistantContext = ({
  assistant,
  assistantPing,
  contextReady,
  disabled,
  localModelStatus,
  profileId,
  selectedProfile,
}) => {
  const noProfile = !profileId;
  const selectedProviderId =
    selectedProfile.providerId ||
    selectedProfile.provider ||
    assistant.providerId ||
    'codex';
  const isCodexProfile =
    (selectedProfile.provider || selectedProfile.providerId || 'codex') ===
    'codex';
  const canManageModel =
    selectedProviderId === 'local' || selectedProviderId === 'openai_local';
  const llmStatus = canManageModel ? localModelStatus : null;
  const providerState = providerSnapshot(assistant, selectedProviderId);
  applyLocalModelStatus(providerState, llmStatus);
  applyProviderPing(providerState, {
    assistantPing,
    llmStatus,
    selectedProviderId,
    profileId,
  });
  applyProviderHints(providerState, isCodexProfile);
  const submitDisabled =
    disabled || noProfile || providerState.providerAvailable === false;
  return {
    providerState,
    selectedProviderId,
    isCodexProfile,
    canManageModel,
    llmStatus,
    llmRunning: llmStatus?.state === 'running',
    llmStarting: llmStatus?.state === 'starting',
    actionsDisabled: submitDisabled || !contextReady,
    submitDisabled,
  };
};

export const AssistantColumn = ({
  assistant,
  assistantPing,
  contextReady,
  disabled,
  headerProps,
  layoutControls,
  localLlmStatus,
  profiles,
  onAuthorize,
  onInvoke,
  onPing,
  onStartLocalModel,
  onStopLocalModel,
}) => {
  const [text, setText] = React.useState('');
  const [preferredProfileId, setProfileId] = React.useState(
    assistant.selectedProfileId || '',
  );
  const profileId =
    profiles.find((profile) => profile.id === preferredProfileId)?.id ??
    profiles.find((profile) => profile.id === assistant.selectedProfileId)
      ?.id ??
    profiles[0]?.id ??
    '';
  const { chatMessages, pushUserMessage } = useAssistantChat(
    assistant.lastResponse,
  );

  const handleProfileChange = (id) => {
    setProfileId(id);
  };

  React.useEffect(() => {
    setProfileId(profileId);
  }, [profileId]);

  const selectedProfile = selectProfile(profiles, profileId);
  const localModelStatus = localLlmStatus?.[profileId] || null;
  const observedLocalStatus = React.useMemo(
    () => (localModelStatus ? { ...localModelStatus, ts: Date.now() } : null),
    [localModelStatus],
  );
  const response = assistant.lastResponse || {};
  const context = buildAssistantContext({
    assistant,
    assistantPing,
    contextReady,
    disabled,
    localModelStatus: observedLocalStatus,
    profileId,
    selectedProfile,
  });
  const providerUnavailable = context.providerState.providerAvailable === false;

  const handleInvoke = (params) => {
    const label = params.requestText || ACTION_LABELS[params.action] || '';
    if (label) {
      pushUserMessage(label, params.sourceLabel || 'you');
    }
    onInvoke(params);
  };

  const invokeCustom = (requestText, sourceLabel = 'you') => {
    pushUserMessage(requestText, sourceLabel);
    onInvoke({ requestText, profileId, sourceLabel });
  };

  return (
    <PipelinePanel
      title="AI Assistant"
      active={assistant.busy}
      className={classNames(
        'assistant-column',
        providerUnavailable && 'assistant-unavailable',
      )}
      headerControls={layoutControls}
      headerProps={headerProps}
      showIndicator
    >
      <div className="assistant-controls">
        <AssistantProfileSelect
          disabled={disabled}
          profileId={profileId}
          profiles={profiles}
          onProfileChange={handleProfileChange}
        />

        <AssistantRuntimeBar
          assistantPing={assistantPing}
          disabled={disabled}
          onPing={onPing}
          onStartLocalModel={onStartLocalModel}
          onStopLocalModel={onStopLocalModel}
          profileId={profileId}
          runtime={context}
        />

        {!providerUnavailable && contextReady ? (
          <AssistantQuickActions
            disabled={context.actionsDisabled}
            profileId={profileId}
            onInvoke={handleInvoke}
          />
        ) : null}
        {!providerUnavailable && !contextReady ? (
          <p className="control-hint">
            Quick actions become available when the transcript has text.
          </p>
        ) : null}
      </div>
      <AssistantResponse
        assistant={context.providerState}
        busy={assistant.busy}
        messages={chatMessages}
        onAuthorize={() => onAuthorize?.(context.selectedProviderId)}
      />
      <details className="panel-details assistant-details">
        <summary>Session details</summary>
        <AssistantStats
          assistant={assistant}
          response={response}
          selectedProfile={selectedProfile}
        />
      </details>
      <AssistantPromptBox
        disabled={disabled || providerUnavailable}
        submitDisabled={context.submitDisabled}
        text={text}
        onTextChange={setText}
        onSubmit={() => {
          invokeCustom(text);
          setText('');
        }}
      />
    </PipelinePanel>
  );
};
