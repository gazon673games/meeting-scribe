import React from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

export const CollapsibleSection = ({
  title,
  children,
  defaultOpen = true,
  action,
  summary,
}) => {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <section className="settings-section">
      <div
        className="settings-section-head collapsible-head"
        role="button"
        aria-expanded={open}
        tabIndex={0}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (
            e.target === e.currentTarget &&
            (e.key === 'Enter' || e.key === ' ')
          ) {
            e.preventDefault();
            setOpen((o) => !o);
          }
        }}
      >
        <div className="section-head-left">
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          <h3>{title}</h3>
          {!open && summary ? (
            <span className="settings-section-summary">{summary}</span>
          ) : null}
        </div>
        {action ? (
          <div role="presentation" onClick={(e) => e.stopPropagation()}>
            {action}
          </div>
        ) : null}
      </div>
      {open ? children : null}
    </section>
  );
};
