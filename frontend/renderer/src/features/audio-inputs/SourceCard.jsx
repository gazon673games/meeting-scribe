import React from "react";
import { Trash2 } from "lucide-react";

import { SelectShell } from "../../shared/ui/forms/SelectShell";
import { SwitchButton } from "../../shared/ui/SwitchButton";
import { displayDeviceLabel } from "./deviceLabels";

export function SourceCard({
  devices = [],
  disabled,
  icon,
  picker: Picker,
  pickerProps = {},
  removable = false,
  source,
  title,
  onAdd,
  onRemove,
  onToggle
}) {
  const [selectedId, setSelectedId] = React.useState("");
  const [selectedOverride, setSelectedOverride] = React.useState(null);
  const sourceDevice = React.useMemo(() => devices.find((device) => source && device.label === source.label), [devices, source]);
  const currentOption = React.useMemo(
    () => (source && !sourceDevice ? { id: `current:${source.name}`, label: source.label || source.name, current: true } : null),
    [source, sourceDevice]
  );
  const selectableDevices = React.useMemo(() => (currentOption ? [currentOption, ...devices] : devices), [currentOption, devices]);
  const preferredId = sourceDevice?.id || currentOption?.id || devices[0]?.id || "";
  const selectedIdOrDefault = selectableDevices.some((device) => device.id === selectedId) ? selectedId : preferredId;
  const selectedFromDevices = selectableDevices.find((device) => device.id === selectedIdOrDefault);
  const selected = selectedFromDevices || (selectedOverride?.id === selectedId ? selectedOverride : null) || selectableDevices[0];
  const selectedLabel = selected ? displayDeviceLabel(selected.fullLabel || selected.label || selected.name) : "No devices found";
  const staticSelected = !Picker && selected?.current && selectableDevices.length === 1;
  const level = Math.max(0, Math.min(100, Number(source?.level || 0)));
  const meterClass = level > 80 ? "hot" : level > 45 ? "warm" : "";
  const canToggle = !disabled && (Boolean(source) || Boolean(selected && !selected.current));

  React.useEffect(() => {
    setSelectedId((current) => (selectableDevices.some((device) => device.id === current) ? current : preferredId));
  }, [preferredId, selectableDevices]);

  const handleSelectDevice = async (next) => {
    if (!next) {
      return;
    }
    setSelectedId(next.id);
    setSelectedOverride(selectableDevices.some((device) => device.id === next.id) ? null : next);
    if (next.current || !onAdd) {
      return;
    }
    if (!source) {
      await onAdd(next);
      return;
    }
    if (!onRemove || sourceMatchesDevice(source, next)) {
      return;
    }
    await onRemove(source);
    await onAdd(next);
  };

  const handleSelect = (event) => {
    const next = selectableDevices.find((device) => device.id === event.target.value);
    handleSelectDevice(next);
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

  const handleRemove = () => {
    if (!source || disabled || !onRemove) {
      return;
    }
    onRemove(source);
  };

  return (
    <article className="source-card">
      <div className="source-head">
        <div className="source-title">
          {icon}
          <strong title={title}>{title}</strong>
        </div>
        <div className="source-actions">
          {removable && source ? (
            <button
              aria-label={`Remove ${title}`}
              className="source-remove"
              disabled={disabled || !onRemove}
              title="Remove source"
              type="button"
              onClick={handleRemove}
            >
              <Trash2 size={14} />
            </button>
          ) : null}
          <SwitchButton checked={Boolean(source?.enabled)} disabled={!canToggle} onClick={handleToggle} />
        </div>
      </div>

      <div className="audio-meter">
        <div className={meterClass} style={{ width: `${source?.enabled ? level : 0}%` }} />
      </div>

      {staticSelected ? (
        <div className="source-static-value" title={selectedLabel}>
          {selectedLabel}
        </div>
      ) : Picker ? (
        <Picker
          devices={devices}
          disabled={disabled}
          {...pickerProps}
          selectableDevices={selectableDevices}
          selected={selected}
          selectedId={selected?.id || ""}
          onSelect={handleSelectDevice}
        />
      ) : (
        <SelectShell displayLabel={selectedLabel} iconSize={14}>
          <select
            disabled={disabled || !selectableDevices.length}
            title={selectedLabel}
            value={selected?.id || ""}
            onChange={handleSelect}
          >
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
      )}
    </article>
  );
}

function sourceMatchesDevice(source, device) {
  const sourceLabel = String(source?.label || source?.name || "");
  const deviceLabel = String(device?.fullLabel || device?.label || "");
  return sourceLabel === deviceLabel || sourceLabel === String(device?.label || "");
}
