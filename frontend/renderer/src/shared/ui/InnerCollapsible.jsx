import React from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

export const InnerCollapsible = ({ title, children, defaultOpen = false }) => {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div className={`inner-collapsible${open ? ' open' : ''}`}>
      <div
        className="inner-collapsible-head"
        role="button"
        aria-expanded={open}
        tabIndex={0}
        onClick={() => setOpen((current) => !current)}
        onKeyDown={(event) => {
          if (event.target !== event.currentTarget) return;
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            setOpen((current) => !current);
          }
        }}
      >
        {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        <span>{title}</span>
      </div>
      {open ? <div className="inner-collapsible-body">{children}</div> : null}
    </div>
  );
};
