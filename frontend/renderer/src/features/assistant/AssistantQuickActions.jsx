import { FileText, ListChecks, MessageSquare, TriangleAlert } from "lucide-react";

import { ActionButton } from "../../shared/ui/ActionButton";
import { SectionLabel } from "../../shared/ui/SectionLabel";

export function AssistantQuickActions({ disabled, profileId, onInvoke }) {
  return (
    <>
      <SectionLabel>Quick Actions</SectionLabel>
      <div className="quick-grid">
        <ActionButton disabled={disabled} icon={<MessageSquare size={15} />} label="Answer Latest" onClick={() => onInvoke({ action: "answer", profileId })} />
        <ActionButton disabled={disabled} icon={<FileText size={15} />} label="Summarize" onClick={() => onInvoke({ action: "summary", profileId })} />
        <ActionButton disabled={disabled} icon={<ListChecks size={15} />} label="Action Items" onClick={() => onInvoke({ action: "action_items", profileId })} />
        <ActionButton disabled={disabled} icon={<TriangleAlert size={15} />} label="Risk Check" onClick={() => onInvoke({ action: "risk_check", profileId })} />
      </div>
    </>
  );
}
