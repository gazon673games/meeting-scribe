import React from "react";
import { Play, Radio, Square } from "lucide-react";

import { PipelinePanel } from "../../shared/ui/pipeline/PipelinePanel";
import { AssistantProfileSelect } from "./AssistantProfileSelect";
import { AssistantPromptBox } from "./AssistantPromptBox";
import { AssistantQuickActions } from "./AssistantQuickActions";
import { AssistantResponse } from "./AssistantResponse";
import { AssistantStats } from "./AssistantStats";

const ACTION_LABELS = {
  answer: "Answer Latest",
  summary: "Summarize",
  action_items: "Action Items",
  risk_check: "Risk Check",
};

export function AssistantColumn({
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
  onStopLocalModel
}) {
  const [text, setText] = React.useState("");
  const [profileId, setProfileId] = React.useState(assistant.selectedProfileId || "");
  const { chatMessages, pushUserMessage } = useAssistantChat(assistant.lastResponse);

  const handleProfileChange = (id) => {
    setProfileId(id);
  };

  React.useEffect(() => {
    setProfileId((current) => current || assistant.selectedProfileId || "");
  }, [assistant.selectedProfileId]);

  const selectedProfile = selectProfile(profiles, profileId);
  const response = assistant.lastResponse || {};
  const context = buildAssistantContext({
    assistant,
    contextReady,
    disabled,
    localLlmStatus,
    profileId,
    selectedProfile,
  });

  const handleInvoke = (params) => {
    const label = params.requestText || ACTION_LABELS[params.action] || "";
    if (label) {
      pushUserMessage(label, params.sourceLabel || "you");
    }
    onInvoke(params);
  };

  const invokeCustom = (requestText, sourceLabel = "you") => {
    pushUserMessage(requestText, sourceLabel);
    onInvoke({ requestText, profileId, sourceLabel });
  };

  return (
    <PipelinePanel title="AI Assistant" active={assistant.busy} className="assistant-column" headerControls={layoutControls} headerProps={headerProps} showIndicator>
      <AssistantProfileSelect disabled={disabled} profileId={profileId} profiles={profiles} onProfileChange={handleProfileChange} />

      <AssistantRuntimeBar
        assistantPing={assistantPing}
        disabled={disabled}
        onPing={onPing}
        onStartLocalModel={onStartLocalModel}
        onStopLocalModel={onStopLocalModel}
        profileId={profileId}
        runtime={context}
      />

      <AssistantQuickActions disabled={context.actionsDisabled} profileId={profileId} onInvoke={handleInvoke} />
      <AssistantResponse
        assistant={assistant}
        busy={assistant.busy}
        messages={chatMessages}
        onAuthorize={() => onAuthorize?.(context.selectedProviderId)}
      />
      <AssistantStats assistant={assistant} response={response} selectedProfile={selectedProfile} />
      <AssistantPromptBox
        disabled={disabled}
        submitDisabled={context.submitDisabled}
        text={text}
        onTextChange={setText}
        onSubmit={() => {
          invokeCustom(text);
          setText("");
        }}
      />
    </PipelinePanel>
  );
}

function useAssistantChat(lastResponse) {
  const [chatMessages, setChatMessages] = React.useState([]);
  const lastResponseRef = React.useRef(null);

  React.useEffect(() => {
    if (!isAssistantResponseAppendable(lastResponse, lastResponseRef.current)) {
      return;
    }
    lastResponseRef.current = lastResponse.ts;
    setChatMessages((prev) => [...prev, { id: Date.now(), role: "assistant", ...lastResponse }]);
  }, [lastResponse]);

  const pushUserMessage = React.useCallback((text, sourceLabel) => {
    setChatMessages((prev) => [...prev, { id: Date.now(), role: "user", text, sourceLabel }]);
  }, []);

  return { chatMessages, pushUserMessage };
}

function isAssistantResponseAppendable(response, lastTs) {
  if (!response || !response.ts) return false;
  if (response.ts === lastTs) return false;
  if (!response.text && response.ok !== false) return false;
  return true;
}

function selectProfile(profiles, profileId) {
  return profiles.find((profile) => profile.id === profileId) || {};
}

function buildAssistantContext({ assistant, contextReady, disabled, localLlmStatus, profileId, selectedProfile }) {
  const noProfile = !profileId;
  const selectedProviderId = selectedProfile.providerId || selectedProfile.provider || assistant.providerId || "codex";
  const isCodexProfile = (selectedProfile.provider || selectedProfile.providerId || "codex") === "codex";
  const llmStatus = !isCodexProfile ? (localLlmStatus?.[profileId] || null) : null;
  return {
    selectedProviderId,
    isCodexProfile,
    llmStatus,
    llmRunning: llmStatus?.state === "running",
    llmStarting: llmStatus?.state === "starting",
    actionsDisabled: disabled || !contextReady || noProfile,
    submitDisabled: disabled || noProfile,
  };
}

function AssistantRuntimeBar({
  assistantPing,
  disabled,
  onPing,
  onStartLocalModel,
  onStopLocalModel,
  profileId,
  runtime,
}) {
  if (!profileId) {
    return null;
  }
  if (runtime.isCodexProfile) {
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
        <PingStatus ping={assistantPing} />
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
          {runtime.llmStarting ? "Starting..." : "Stop Model"}
        </button>
      )}
      <LocalLlmStatusText status={runtime.llmStatus} />
    </div>
  );
}

function LocalLlmStatusText({ status }) {
  if (!status) return <span className="ping-status" style={{ color: "var(--muted)" }}>not started</span>;
  if (status.state === "running") return <span className="ping-status ping-ok">ready</span>;
  if (status.state === "starting") return <span className="ping-status ping-busy">starting...</span>;
  if (status.state === "error") return <span className="ping-status ping-err" title={status.message}>error: {status.message || "failed to start"}</span>;
  if (status.state === "stopped") return <span className="ping-status" style={{ color: "var(--muted)" }}>stopped</span>;
  return null;
}

function PingStatus({ ping }) {
  if (!ping || (!ping.busy && ping.ts == null)) return null;
  if (ping.busy) return <span className="ping-status ping-busy">pinging...</span>;
  const isAuthError = /auth|unauthorized|not_logged|login/i.test(ping.errorCode || "");
  if (ping.ok) return <span className="ping-status ping-ok">ok</span>;
  if (isAuthError) return <span className="ping-status ping-err">not authorized</span>;
  return <span className="ping-status ping-err">error</span>;
}
