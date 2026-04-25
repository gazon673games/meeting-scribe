import React from "react";
import { ChevronDown, FileText, ListChecks, MessageSquare, Send, Sparkles } from "lucide-react";

import { formatNumber } from "../../shared/lib/format";
import { ActionButton } from "../../shared/ui/ActionButton";
import { PipelineColumn } from "../../shared/ui/PipelineColumn";
import { SectionLabel } from "../../shared/ui/SectionLabel";
import { StatLine } from "../../shared/ui/StatLine";

export function AssistantColumn({ assistant, disabled, profiles, onInvoke }) {
  const [text, setText] = React.useState("");
  const [profileId, setProfileId] = React.useState(assistant.selectedProfileId || profiles?.[0]?.id || "");
  React.useEffect(() => {
    setProfileId((current) => current || assistant.selectedProfileId || profiles?.[0]?.id || "");
  }, [assistant.selectedProfileId, profiles]);

  const selectedProfile = profiles.find((profile) => profile.id === profileId) || profiles[0] || {};
  const response = assistant.lastResponse || {};

  const invokeCustom = (requestText, sourceLabel = "you") => {
    onInvoke({ requestText, profileId, sourceLabel });
  };

  return (
    <PipelineColumn title="AI Assistant" active={assistant.enabled} className="assistant-column">
      <SectionLabel>Profile</SectionLabel>
      <div className="select-shell">
        <select disabled={disabled || !profiles.length} value={profileId} onChange={(event) => setProfileId(event.target.value)}>
          {profiles.length ? (
            profiles.map((profile) => (
              <option key={profile.id || profile.label} value={profile.id}>
                {profile.label || profile.id}
              </option>
            ))
          ) : (
            <option value="">No profiles</option>
          )}
        </select>
        <ChevronDown size={15} />
      </div>

      <SectionLabel>Quick Actions</SectionLabel>
      <div className="quick-grid">
        <ActionButton disabled={disabled} icon={<MessageSquare size={15} />} label="Quick Answer" onClick={() => onInvoke({ action: "answer", profileId })} />
        <ActionButton disabled={disabled} icon={<FileText size={15} />} label="Summarize" onClick={() => onInvoke({ action: "summary", profileId })} />
        <ActionButton
          disabled={disabled}
          icon={<ListChecks size={15} />}
          label="Action Items"
          onClick={() => invokeCustom("Extract concise action items from the current transcript.", "action items")}
        />
        <ActionButton
          disabled={disabled}
          icon={<Sparkles size={15} />}
          label="Interview Assist"
          onClick={() => invokeCustom("Help me answer the latest interview question using the current transcript.", "interview assist")}
        />
      </div>

      <div className="assistant-response">
        <span>Assistant Response</span>
        {response.text ? <p>{response.text}</p> : <p className="muted">{assistant.busy ? "Working..." : "No response yet"}</p>}
      </div>

      <div className="settings-strip">
        <StatLine label="Active Model" value={selectedProfile.model || "-"} />
        <StatLine label="Response Time" value={response.dtS ? `${formatNumber(response.dtS)}s` : assistant.busy ? "running" : "-"} accent="green" />
        <StatLine label="Last Action" value={assistant.lastRequest?.sourceLabel || "-"} accent="blue" />
      </div>

      <div className="assistant-input">
        <textarea disabled={disabled} placeholder="Ask the assistant anything..." value={text} onChange={(event) => setText(event.target.value)} />
        <button
          disabled={disabled || !text.trim()}
          onClick={() => {
            invokeCustom(text);
            setText("");
          }}
        >
          <Send size={15} />
          Send
        </button>
      </div>
    </PipelineColumn>
  );
}
