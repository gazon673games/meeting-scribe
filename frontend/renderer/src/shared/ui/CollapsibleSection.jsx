import React from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

export function CollapsibleSection({ title, children, defaultOpen = true, action }) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <section className="settings-section">
      <div
        className="settings-section-head collapsible-head"
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
        <div className="section-head-left">
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <h3>{title}</h3>
        </div>
        {action ? (
          <div
            role="presentation"
            onClick={(e) => e.stopPropagation()}
          >
            {action}
          </div>
        ) : null}
      </div>
      {open ? children : null}
    </section>
  );
}
