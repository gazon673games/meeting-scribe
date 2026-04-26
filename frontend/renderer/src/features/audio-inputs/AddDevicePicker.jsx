import React from "react";
import { Plus } from "lucide-react";

import { SelectShell } from "../../shared/ui/forms/SelectShell";
import { displayDeviceLabel } from "./deviceLabels";

export function AddDevicePicker({ groups, disabled, onAdd }) {
  const [selectedId, setSelectedId] = React.useState("");
  const normalizedGroups = React.useMemo(
    () =>
      (groups || [])
        .map((group) => ({
          label: group.label,
          devices: (group.devices || []).filter(Boolean)
        }))
        .filter((group) => group.devices.length),
    [groups]
  );
  const devices = React.useMemo(() => normalizedGroups.flatMap((group) => group.devices), [normalizedGroups]);

  React.useEffect(() => {
    setSelectedId((current) => (devices.some((device) => device.id === current) ? current : devices[0]?.id || ""));
  }, [devices]);

  if (!devices.length) {
    return (
      <div className="add-source empty">
        <Plus size={16} />
        <span>Add Source</span>
      </div>
    );
  }

  const selected = devices.find((device) => device.id === selectedId) || devices[0];
  return (
    <details className={`add-source ${disabled ? "disabled" : ""}`}>
      <summary
        onClick={(event) => {
          if (disabled) {
            event.preventDefault();
          }
        }}
      >
        <Plus size={15} />
        <span>Add Source</span>
      </summary>
      <div className="add-source-picker">
        <SelectShell iconSize={14}>
          <select disabled={disabled} value={selected.id} onChange={(event) => setSelectedId(event.target.value)}>
            {normalizedGroups.map((group) => (
              <optgroup key={group.label} label={group.label}>
                {group.devices.map((device) => (
                  <option key={device.id} value={device.id}>
                    {displayDeviceLabel(device.label)}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </SelectShell>
        <button disabled={disabled} onClick={() => onAdd(selected)}>
          Add
        </button>
      </div>
    </details>
  );
}
