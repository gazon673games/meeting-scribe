import { ChevronDown } from 'lucide-react';

export const SelectShell = ({ children, displayLabel = '', iconSize = 15 }) => (
  <div className={`select-shell ${displayLabel ? 'has-display-label' : ''}`}>
    {displayLabel ? (
      <span className="select-shell-label" title={displayLabel}>
        {displayLabel}
      </span>
    ) : null}
    {children}
    <ChevronDown size={iconSize} />
  </div>
);
