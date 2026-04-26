import React from "react";

import { SelectShell } from "../../shared/ui/forms/SelectShell";
import { SwitchButton } from "../../shared/ui/SwitchButton";
import { displayDeviceLabel } from "./deviceLabels";

export function SourceCard({ devices = [], disabled, icon, source, title, onAdd, onRemove, onToggle }) {
  const [selectedId, setSelectedId] = React.useState("");
  const sourceDevice = React.useMemo(() => devices.find((device) => source && device.label === source.label), [devices, source]);
  const currentOption = React.useMemo(
    () => (source && !sourceDevice ? { id: `current:${source.name}`, label: source.label || source.name, current: true } : null),
    [source, sourceDevice]
  );
  const selectableDevices = React.useMemo(() => (currentOption ? [currentOption, ...devices] : devices), [currentOption, devices]);
  const preferredId = sourceDevice?.id || currentOption?.id || devices[0]?.id || "";
  const selectedIdOrDefault = selectableDevices.some((device) => device.id === selectedId) ? selectedId : preferredId;
  const selected = selectableDevices.find((device) => device.id === selectedIdOrDefault) || selectableDevices[0];
  const level = Math.max(0, Math.min(100, Number(source?.level || 0)));
  const meterClass = level > 80 ? "hot" : level > 45 ? "warm" : "";
  const canToggle = !disabled && (source || (selected && !selected.current));

  React.useEffect(() => {
    setSelectedId((current) => (selectableDevices.some((device) => device.id === current) ? current : preferredId));
  }, [preferredId, selectableDevices]);

  const handleSelect = async (event) => {
    const nextId = event.target.value;
    const next = selectableDevices.find((device) => device.id === nextId);
    setSelectedId(nextId);
    if (!next || next.current || !source || !onAdd || !onRemove || next.label === source.label) {
      return;
    }
    await onRemove(source);
    await onAdd(next);
  };

  const handleToggle = () => {
    if (source) {
      onToggle(source);
      return;
    }
    if (selected && !selected.current) {
      onAdd(selected);
    }
  };

  return (
    <article className="source-card">
      <div className="source-head">
        <div className="source-title">
          {icon}
          <strong>{title}</strong>
        </div>
        <SwitchButton checked={Boolean(source?.enabled)} disabled={!canToggle} onClick={handleToggle} />
      </div>

      <div className="audio-meter">
        <div className={meterClass} style={{ width: `${source?.enabled ? level : 0}%` }} />
      </div>

      <SelectShell iconSize={14}>
        <select disabled={disabled || !selectableDevices.length} value={selected?.id || ""} onChange={handleSelect}>
          {selectableDevices.length ? (
            selectableDevices.map((device) => (
              <option key={device.id} value={device.id}>
                {displayDeviceLabel(device.label)}
              </option>
            ))
          ) : (
            <option value="">No devices found</option>
          )}
        </select>
      </SelectShell>
    </article>
  );
}
