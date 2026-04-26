import React from "react";
import { Plus } from "lucide-react";

import { SelectShell } from "../../shared/ui/forms/SelectShell";

export function AddDevicePicker({ devices, disabled, label, onAdd }) {
  const [selectedId, setSelectedId] = React.useState("");

  React.useEffect(() => {
    setSelectedId((current) => current || devices?.[0]?.id || "");
  }, [devices]);

  if (!devices?.length) {
    return (
      <div className="add-source empty">
        <Plus size={16} />
        <span>{label}</span>
      </div>
    );
  }

  const selected = devices.find((device) => device.id === selectedId) || devices[0];
  return (
    <div className="add-source">
      <SelectShell iconSize={14}>
        <select disabled={disabled} value={selected.id} onChange={(event) => setSelectedId(event.target.value)}>
          {devices.map((device) => (
            <option key={device.id} value={device.id}>
              {device.label}
            </option>
          ))}
        </select>
      </SelectShell>
      <button disabled={disabled} onClick={() => onAdd(selected)}>
        <Plus size={15} />
        {label}
      </button>
    </div>
  );
}
