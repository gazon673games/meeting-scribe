import React from "react";
import { ChevronDown, ChevronRight, Play, Radio, Square } from "lucide-react";

import { PipelinePanel } from "../../shared/ui/pipeline/PipelinePanel";
import { AssistantProfileSelect } from "./AssistantProfileSelect";
import { AssistantPromptBox } from "./AssistantPromptBox";
import { AssistantQuickActions } from "./AssistantQuickActions";
import { AssistantResponse } from "./AssistantResponse";
import { AssistantStats } from "./AssistantStats";

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
  const [optionsOpen, setOptionsOpen] = React.useState(false);
  const [profileId, setProfileId] = React.useState(assistant.selectedProfileId || profiles?.[0]?.id || "");

  React.useEffect(() => {
    setProfileId((current) => current || assistant.selectedProfileId || profiles?.[0]?.id || "");
  }, [assistant.selectedProfileId, profiles]);

  // Stop previous local model server when switching profiles
  const prevProfileRef = React.useRef(profileId);
  React.useEffect(() => {
    const prev = prevProfileRef.current;
    prevProfileRef.current = profileId;
    if (prev && prev !== profileId) {
      const prevProfile = profiles.find((p) => p.id === prev);
      const prevProvider = prevProfile?.provider || prevProfile?.providerId || "codex";
      if (prevProvider !== "codex") {
        onStopLocalModel?.(prev);
      }
    }
  }, [profileId, profiles, onStopLocalModel]);

  const selectedProfile = profiles.find((profile) => profile.id === profileId) || profiles[0] || {};
  const response = assistant.lastResponse || {};
  const actionsDisabled = disabled || !contextReady;
  const selectedProviderId = selectedProfile.providerId || selectedProfile.provider || assistant.providerId || "codex";
  const isCodexProfile = (selectedProfile.provider || selectedProfile.providerId || "codex") === "codex";
  const llmStatus = !isCodexProfile ? (localLlmStatus?.[profileId] || null) : null;
  const llmRunning = llmStatus?.state === "running";
  const llmStarting = llmStatus?.state === "starting";

  const invokeCustom = (requestText, sourceLabel = "you") => {
    if (!contextReady) return;
    onInvoke({ requestText, profileId, sourceLabel });
  };

  return (
    <PipelinePanel title="AI Assistant" active={assistant.busy} className="assistant-column" headerControls={layoutControls} headerProps={headerProps} showIndicator>
      <AssistantProfileSelect disabled={disabled} profileId={profileId} profiles={profiles} onProfileChange={setProfileId} />

      {!isCodexProfile ? (
        <div className="local-model-bar">
          {!llmRunning && !llmStarting ? (
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
              disabled={disabled || llmStarting}
              type="button"
              onClick={() => onStopLocalModel?.(profileId)}
            >
              <Square size={13} />
              {llmStarting ? "Starting..." : "Stop Model"}
            </button>
          )}
          <LocalLlmStatusText status={llmStatus} />
        </div>
      ) : null}

      <AssistantQuickActions disabled={actionsDisabled} profileId={profileId} onInvoke={onInvoke} />
      <AssistantResponse
        assistant={assistant}
        busy={assistant.busy}
        response={response}
        onAuthorize={() => onAuthorize?.(selectedProviderId)}
      />
      <div className="assistant-options">
        <button className="assistant-options-toggle" type="button" onClick={() => setOptionsOpen((v) => !v)}>
          {optionsOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          Profile Options
        </button>
        {optionsOpen && (
          <div className="assistant-options-body">
            <div className="assistant-options-meta">
              {selectedProfile.model ? <span className="profile-meta-model">{selectedProfile.model}</span> : null}
              <span className={`profile-mode-badge ${selectedProfile.offline ? "offline" : "online"}`}>
                {selectedProfile.offline ? "offline" : "online"}
              </span>
            </div>
            {isCodexProfile ? (
              <div className="assistant-options-row">
                <button
                  className="assistant-ping-btn"
                  disabled={disabled || assistantPing?.busy}
                  type="button"
                  onClick={() => onPing?.(selectedProviderId, profileId)}
                >
                  <Radio size={13} />
                  Ping
                </button>
                <PingStatus ping={assistantPing} />
              </div>
            ) : null}
          </div>
        )}
      </div>
      <AssistantStats assistant={assistant} response={response} selectedProfile={selectedProfile} />
      <AssistantPromptBox
        disabled={disabled}
        submitDisabled={actionsDisabled}
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

function LocalLlmBadge({ state }) {
  if (state === "running") return <span className="llm-badge llm-badge-running">ok</span>;
  if (state === "starting") return <span className="llm-badge llm-badge-starting">...</span>;
  if (state === "error") return <span className="llm-badge llm-badge-error">!</span>;
  return null;
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
