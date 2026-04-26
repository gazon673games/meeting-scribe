import { ChevronDown } from "lucide-react";

export function SelectShell({ children, iconSize = 15 }) {
  return (
    <div className="select-shell">
      {children}
      <ChevronDown size={iconSize} />
    </div>
  );
}
