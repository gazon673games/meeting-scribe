import React from "react";

import { PipelinePanel } from "../../shared/ui/pipeline/PipelinePanel";
import { AssistantProfileSelect } from "./AssistantProfileSelect";
import { AssistantPromptBox } from "./AssistantPromptBox";
import { AssistantQuickActions } from "./AssistantQuickActions";
import { AssistantResponse } from "./AssistantResponse";
import { AssistantStats } from "./AssistantStats";

export function AssistantColumn({ assistant, contextReady, disabled, headerProps, layoutControls, profiles, onInvoke }) {
  const [text, setText] = React.useState("");
  const [profileId, setProfileId] = React.useState(assistant.selectedProfileId || profiles?.[0]?.id || "");
  React.useEffect(() => {
    setProfileId((current) => current || assistant.selectedProfileId || profiles?.[0]?.id || "");
  }, [assistant.selectedProfileId, profiles]);

  const selectedProfile = profiles.find((profile) => profile.id === profileId) || profiles[0] || {};
  const response = assistant.lastResponse || {};
  const actionsDisabled = disabled || !contextReady;

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
      <AssistantResponse assistant={assistant} busy={assistant.busy} response={response} />
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
