import { Trash2 } from "lucide-react";

import { SelectShell } from "../../shared/ui/forms/SelectShell";
import { SwitchButton } from "../../shared/ui/SwitchButton";
import { SourceDelayControl } from "./SourceDelayControl";

export function SourceCard({ disabled, icon, source, title, onDelay, onRemove, onToggle }) {
  const level = Math.max(0, Math.min(100, Number(source.level || 0)));
  return (
    <article className="source-card">
      <div className="source-head">
        <div className="source-title">
          {icon}
          <strong>{title}</strong>
        </div>
        <SwitchButton checked={source.enabled} disabled={disabled} onClick={() => onToggle(source)} />
      </div>

      <div className="audio-meter">
        <div className={level > 80 ? "hot" : level > 55 ? "warm" : ""} style={{ width: `${source.enabled ? level : 0}%` }} />
      </div>

      <SelectShell iconSize={14}>
        <select value={source.label || source.name} disabled>
          <option>{source.label || source.name}</option>
        </select>
      </SelectShell>

      <div className="source-actions">
        <SourceDelayControl disabled={disabled} source={source} onDelay={onDelay} />
        <span className="source-status">{source.status}</span>
        <button className="icon-action" disabled={disabled} onClick={() => onRemove(source)} title="Remove source">
          <Trash2 size={14} />
        </button>
      </div>
    </article>
  );
}
