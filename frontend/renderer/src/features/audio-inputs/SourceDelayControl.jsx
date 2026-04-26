import React from "react";

import { formatDelay } from "../../shared/lib/format";

export function SourceDelayControl({ disabled, source, onDelay }) {
  const [value, setValue] = React.useState(formatDelay(source.delayMs));
  React.useEffect(() => {
    setValue(formatDelay(source.delayMs));
  }, [source.delayMs]);

  const commit = React.useCallback(() => {
    const parsed = Number(String(value || "0").replace(",", "."));
    const delayMs = Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
    setValue(formatDelay(delayMs));
    onDelay(source, delayMs);
  }, [onDelay, source, value]);

  return (
    <label className="delay-control">
      <span>delay</span>
      <input
        disabled={disabled}
        min="0"
        onBlur={commit}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
        }}
        step="10"
        type="number"
        value={value}
      />
    </label>
  );
}
