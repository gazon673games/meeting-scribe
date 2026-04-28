import { ChevronDown } from "lucide-react";

export function SelectShell({ children, displayLabel = "", iconSize = 15 }) {
  return (
    <div className={`select-shell ${displayLabel ? "has-display-label" : ""}`}>
      {displayLabel ? (
        <span className="select-shell-label" title={displayLabel}>
          {displayLabel}
        </span>
      ) : null}
      {children}
      <ChevronDown size={iconSize} />
    </div>
  );
}
