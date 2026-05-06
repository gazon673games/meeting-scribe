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
  const selection = React.useMemo(
    () => resolveSelectionState({ devices, source, selectedId, selectedOverride, pickerAvailable: Boolean(Picker) }),
    [devices, source, selectedId, selectedOverride, Picker]
  );
  const { selectableDevices, preferredId, selected, staticSelected } = selection;
  const selectedLabel = selectedLabelText(selected);
  const level = Math.max(0, Math.min(100, Number(source?.level || 0)));
  const meterClass = level > 80 ? "hot" : level > 45 ? "warm" : "";
  const canToggle = sourceToggleEnabled(disabled, source, selected);

  React.useEffect(() => {
    setSelectedId((current) => (selectableDevices.some((device) => device.id === current) ? current : preferredId));
  }, [preferredId, selectableDevices]);

  const handleSelectDevice = async (next) => {
    if (!next) {
      return;
    }
    setSelectedId(next.id);
    setSelectedOverride(selectableDevices.some((device) => device.id === next.id) ? null : next);
    await syncSelectedDevice({ next, source, onAdd, onRemove });
  };

  const handleSelect = (event) => {
    const next = selectableDevices.find((device) => device.id === event.target.value);
    handleSelectDevice(next);
  };

  const handleToggle = () => {
    toggleSource({ source, selected, onAdd, onToggle });
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

      <SourceCardDevicePicker
        Picker={Picker}
        devices={devices}
        disabled={disabled}
        handleSelect={handleSelect}
        handleSelectDevice={handleSelectDevice}
        pickerProps={pickerProps}
        selectableDevices={selectableDevices}
        selected={selected}
        selectedLabel={selectedLabel}
        staticSelected={staticSelected}
      />
    </article>
  );
}

function SourceCardDevicePicker({
  Picker,
  devices,
  disabled,
  handleSelect,
  handleSelectDevice,
  pickerProps,
  selectableDevices,
  selected,
  selectedLabel,
  staticSelected,
}) {
  if (staticSelected) {
    return (
      <div className="source-static-value" title={selectedLabel}>
        {selectedLabel}
      </div>
    );
  }
  if (Picker) {
    return (
      <Picker
        devices={devices}
        disabled={disabled}
        {...pickerProps}
        selectableDevices={selectableDevices}
        selected={selected}
        selectedId={selected?.id || ""}
        onSelect={handleSelectDevice}
      />
    );
  }
  return (
    <DefaultSourcePicker
      disabled={disabled}
      selectableDevices={selectableDevices}
      selected={selected}
      selectedLabel={selectedLabel}
      onSelect={handleSelect}
    />
  );
}

function DefaultSourcePicker({ disabled, selectableDevices, selected, selectedLabel, onSelect }) {
  return (
    <SelectShell displayLabel={selectedLabel} iconSize={14}>
      <select
        disabled={disabled || !selectableDevices.length}
        title={selectedLabel}
        value={selected?.id || ""}
        onChange={onSelect}
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
  );
}

function resolveSelectionState({ devices, source, selectedId, selectedOverride, pickerAvailable }) {
  const sourceDevice = findSourceDevice(devices, source);
  const currentOption = buildCurrentOption(source, sourceDevice);
  const selectableDevices = withCurrentOption(currentOption, devices);
  const preferredId = resolvePreferredId(sourceDevice, currentOption, devices);
  const selectedIdOrDefault = resolveSelectedId(selectableDevices, selectedId, preferredId);
  const selected = resolveSelectedDevice(selectableDevices, selectedIdOrDefault, selectedOverride, selectedId);
  return {
    selectableDevices,
    preferredId,
    selected,
    staticSelected: computeStaticSelected(pickerAvailable, selected, selectableDevices),
  };
}

function findSourceDevice(devices, source) {
  if (!source) {
    return null;
  }
  return devices.find((device) => device.label === source.label) || null;
}

function buildCurrentOption(source, sourceDevice) {
  if (!source || sourceDevice) {
    return null;
  }
  return {
    id: `current:${source.name}`,
    label: sourceLabelValue(source),
    current: true,
  };
}

function withCurrentOption(currentOption, devices) {
  if (!currentOption) {
    return devices;
  }
  return [currentOption, ...devices];
}

function resolvePreferredId(sourceDevice, currentOption, devices) {
  if (sourceDevice?.id) {
    return sourceDevice.id;
  }
  if (currentOption?.id) {
    return currentOption.id;
  }
  return devices[0]?.id || "";
}

function resolveSelectedId(selectableDevices, selectedId, preferredId) {
  if (selectableDevices.some((device) => device.id === selectedId)) {
    return selectedId;
  }
  return preferredId;
}

function resolveSelectedDevice(selectableDevices, selectedIdOrDefault, selectedOverride, selectedId) {
  const fromList = selectableDevices.find((device) => device.id === selectedIdOrDefault);
  if (fromList) {
    return fromList;
  }
  if (selectedOverride?.id === selectedId) {
    return selectedOverride;
  }
  return selectableDevices[0];
}

function computeStaticSelected(pickerAvailable, selected, selectableDevices) {
  if (pickerAvailable || !selected) {
    return false;
  }
  return Boolean(selected.current) && selectableDevices.length === 1;
}

function selectedLabelText(selected) {
  if (!selected) {
    return "No devices found";
  }
  return displayDeviceLabel(selected.fullLabel || selected.label || selected.name);
}

function sourceToggleEnabled(disabled, source, selected) {
  if (disabled) {
    return false;
  }
  if (source) {
    return true;
  }
  return Boolean(selected) && !selected.current;
}

function toggleSource({ source, selected, onAdd, onToggle }) {
  if (source) {
    onToggle(source);
    return;
  }
  if (selected && !selected.current) {
    onAdd(selected);
  }
}

function sourceLabelValue(source) {
  if (source.label) {
    return source.label;
  }
  return source.name || "";
}

async function syncSelectedDevice({ next, source, onAdd, onRemove }) {
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
}

function sourceMatchesDevice(source, device) {
  const sourceLabel = String(source?.label || source?.name || "");
  const deviceLabel = String(device?.fullLabel || device?.label || "");
  return sourceLabel === deviceLabel || sourceLabel === String(device?.label || "");
}
