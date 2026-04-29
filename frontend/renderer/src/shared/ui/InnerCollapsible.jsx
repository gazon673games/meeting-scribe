import React from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

export function InnerCollapsible({ title, children, defaultOpen = false }) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div className={`inner-collapsible${open ? " open" : ""}`}>
      <div
        className="inner-collapsible-head"
        role="button"
        tabIndex={0}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen((o) => !o);
          }
        }}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <span>{title}</span>
      </div>
      {open ? <div className="inner-collapsible-body">{children}</div> : null}
    </div>
  );
}
