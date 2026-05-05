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
  const [chatMessages, setChatMessages] = React.useState([]);
  const lastResponseRef = React.useRef(null);

  React.useEffect(() => {
    setProfileId((current) => current || assistant.selectedProfileId || "");
  }, [assistant.selectedProfileId]);

  React.useEffect(() => {
    const r = assistant.lastResponse;
    if (!r || !r.ts) return;
    if (r.ts === lastResponseRef.current) return;
    if (!r.text && r.ok !== false) return;
    lastResponseRef.current = r.ts;
    setChatMessages((prev) => [...prev, { id: Date.now(), role: "assistant", ...r }]);
  }, [assistant.lastResponse]);

  const handleProfileChange = (id) => {
    setProfileId(id);
  };

  const selectedProfile = profiles.find((profile) => profile.id === profileId) || {};
  const response = assistant.lastResponse || {};
  const noProfile = !profileId;
  const actionsDisabled = disabled || !contextReady || noProfile;
  const selectedProviderId = selectedProfile.providerId || selectedProfile.provider || assistant.providerId || "codex";
  const isCodexProfile = (selectedProfile.provider || selectedProfile.providerId || "codex") === "codex";
  const llmStatus = !isCodexProfile ? (localLlmStatus?.[profileId] || null) : null;
  const llmRunning = llmStatus?.state === "running";
  const llmStarting = llmStatus?.state === "starting";

  const handleInvoke = (params) => {
    const label = params.requestText || ACTION_LABELS[params.action] || "";
    if (label) {
      setChatMessages((prev) => [...prev, { id: Date.now(), role: "user", text: label, sourceLabel: params.sourceLabel || "you" }]);
    }
    onInvoke(params);
  };

  const invokeCustom = (requestText, sourceLabel = "you") => {
    setChatMessages((prev) => [...prev, { id: Date.now(), role: "user", text: requestText, sourceLabel }]);
    onInvoke({ requestText, profileId, sourceLabel });
  };

  return (
    <PipelinePanel title="AI Assistant" active={assistant.busy} className="assistant-column" headerControls={layoutControls} headerProps={headerProps} showIndicator>
      <AssistantProfileSelect disabled={disabled} profileId={profileId} profiles={profiles} onProfileChange={handleProfileChange} />

      {!isCodexProfile && profileId ? (
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

      {isCodexProfile && profileId ? (
        <div className="local-model-bar">
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

      <AssistantQuickActions disabled={actionsDisabled} profileId={profileId} onInvoke={handleInvoke} />
      <AssistantResponse
        assistant={assistant}
        busy={assistant.busy}
        messages={chatMessages}
        onAuthorize={() => onAuthorize?.(selectedProviderId)}
      />
      <AssistantStats assistant={assistant} response={response} selectedProfile={selectedProfile} />
      <AssistantPromptBox
        disabled={disabled}
        submitDisabled={disabled || noProfile}
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
