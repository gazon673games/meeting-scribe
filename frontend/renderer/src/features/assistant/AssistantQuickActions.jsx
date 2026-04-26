import { FileText, ListChecks, MessageSquare, Sparkles } from "lucide-react";

import { ActionButton } from "../../shared/ui/ActionButton";
import { SectionLabel } from "../../shared/ui/SectionLabel";

export function AssistantQuickActions({ disabled, profileId, onInvoke, onInvokeCustom }) {
  return (
    <>
      <SectionLabel>Quick Actions</SectionLabel>
      <div className="quick-grid">
        <ActionButton disabled={disabled} icon={<MessageSquare size={15} />} label="Quick Answer" onClick={() => onInvoke({ action: "answer", profileId })} />
        <ActionButton disabled={disabled} icon={<FileText size={15} />} label="Summarize" onClick={() => onInvoke({ action: "summary", profileId })} />
        <ActionButton
          disabled={disabled}
          icon={<ListChecks size={15} />}
          label="Action Items"
          onClick={() => onInvokeCustom("Extract concise action items from the current transcript.", "action items")}
        />
        <ActionButton
          disabled={disabled}
          icon={<Sparkles size={15} />}
          label="Interview Assist"
          onClick={() => onInvokeCustom("Help me answer the latest interview question using the current transcript.", "interview assist")}
        />
      </div>
    </>
  );
}
