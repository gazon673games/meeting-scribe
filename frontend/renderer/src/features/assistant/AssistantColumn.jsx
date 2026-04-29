import React from "react";
import { ChevronDown, ChevronRight, Radio } from "lucide-react";

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
  profiles,
  onAuthorize,
  onInvoke,
  onPing
}) {
  const [text, setText] = React.useState("");
  const [optionsOpen, setOptionsOpen] = React.useState(false);
  const [profileId, setProfileId] = React.useState(assistant.selectedProfileId || profiles?.[0]?.id || "");
  React.useEffect(() => {
    setProfileId((current) => current || assistant.selectedProfileId || profiles?.[0]?.id || "");
  }, [assistant.selectedProfileId, profiles]);

  const selectedProfile = profiles.find((profile) => profile.id === profileId) || profiles[0] || {};
  const response = assistant.lastResponse || {};
  const actionsDisabled = disabled || !contextReady;
  const selectedProviderId = selectedProfile.providerId || selectedProfile.provider || assistant.providerId || "codex";

  const invokeCustom = (requestText, sourceLabel = "you") => {
    if (!contextReady) {
      return;
    }
    onInvoke({ requestText, profileId, sourceLabel });
  };

  return (
    <PipelinePanel title="AI Assistant" active={assistant.busy} className="assistant-column" headerControls={layoutControls} headerProps={headerProps} showIndicator>
      <AssistantProfileSelect disabled={disabled} profileId={profileId} profiles={profiles} onProfileChange={setProfileId} />
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
            {!selectedProfile.offline && (
              <div className="assistant-options-row">
                <button
                  className="assistant-ping-btn"
                  disabled={disabled || assistantPing?.busy}
                  type="button"
                  onClick={() => onPing?.(selectedProviderId)}
                >
                  <Radio size={13} />
                  Ping
                </button>
                <PingStatus ping={assistantPing} />
              </div>
            )}
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

function PingStatus({ ping }) {
  if (!ping || (!ping.busy && ping.ts == null)) return null;
  if (ping.busy) return <span className="ping-status ping-busy">pinging…</span>;
  const isAuthError = /auth|unauthorized|not_logged|login/i.test(ping.errorCode || "");
  if (ping.ok) return <span className="ping-status ping-ok">ok</span>;
  if (isAuthError) return <span className="ping-status ping-err">not authorized</span>;
  return <span className="ping-status ping-err">error</span>;
}
