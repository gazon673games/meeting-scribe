import { classNames } from '../../shared/lib/classNames';
import React from 'react';
import { Trash2 } from 'lucide-react';

import { SelectShell } from '../../shared/ui/forms/SelectShell';
import { SwitchButton } from '../../shared/ui/SwitchButton';
import { displayDeviceLabel } from './deviceLabels';

const DefaultSourcePicker = ({
  disabled,
  selectableDevices,
  selected,
  selectedLabel,
  onSelect,
}) => (
  <SelectShell displayLabel={selectedLabel} iconSize={14}>
    <select
      disabled={disabled || !selectableDevices.length}
      title={selectedLabel}
      value={selected?.id || ''}
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

const SourceCardDevicePicker = ({
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
}) => {
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
        selectedId={selected?.id || ''}
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
};

const findSourceDevice = (devices, source) => {
  if (!source) {
    return null;
  }
  if (source.deviceId) {
    return devices.find((device) => device.id === source.deviceId) || null;
  }
  const matches = devices.filter((device) => device.label === source.label);
  return matches.length === 1 ? matches[0] : null;
};

const withCurrentOption = (currentOption, devices) => {
  if (!currentOption) {
    return devices;
  }
  return [currentOption, ...devices];
};

const resolvePreferredId = (sourceDevice, currentOption, devices) => {
  if (sourceDevice?.id) {
    return sourceDevice.id;
  }
  if (currentOption?.id) {
    return currentOption.id;
  }
  return devices[0]?.id || '';
};

const resolveSelectedId = (selectableDevices, selectedId, preferredId) => {
  if (selectableDevices.some((device) => device.id === selectedId)) {
    return selectedId;
  }
  return preferredId;
};

const resolveSelectedDevice = (
  selectableDevices,
  selectedIdOrDefault,
  selectedOverride,
  selectedId,
) => {
  const fromList = selectableDevices.find(
    (device) => device.id === selectedIdOrDefault,
  );
  if (fromList) {
    return fromList;
  }
  if (selectedOverride?.id === selectedId) {
    return selectedOverride;
  }
  return selectableDevices[0];
};

const computeStaticSelected = (
  pickerAvailable,
  selected,
  selectableDevices,
) => {
  if (pickerAvailable || !selected) {
    return false;
  }
  return Boolean(selected.current) && selectableDevices.length === 1;
};

const selectedLabelText = (selected) => {
  if (!selected) {
    return 'No devices found';
  }
  return displayDeviceLabel(
    selected.fullLabel || selected.label || selected.name,
  );
};

const sourceToggleEnabled = (disabled, source, selected) => {
  if (disabled) {
    return false;
  }
  if (source) {
    return true;
  }
  return Boolean(selected) && !selected.current;
};

const toggleSource = ({ source, selected, onAdd, onToggle }) => {
  if (source) {
    onToggle(source);
    return;
  }
  if (selected && !selected.current) {
    onAdd(selected);
  }
};

const sourceLabelValue = (source) => {
  if (source.label) {
    return source.label;
  }
  return source.name || '';
};

const buildCurrentOption = (source, sourceDevice) => {
  if (!source || sourceDevice) {
    return null;
  }
  return {
    id: `current:${source.name}`,
    label: sourceLabelValue(source),
    current: true,
  };
};

const resolveSelectionState = ({
  devices,
  source,
  selectedId,
  selectedOverride,
  pickerAvailable,
}) => {
  const sourceDevice = findSourceDevice(devices, source);
  const currentOption = buildCurrentOption(source, sourceDevice);
  const selectableDevices = withCurrentOption(currentOption, devices);
  const preferredId = resolvePreferredId(sourceDevice, currentOption, devices);
  const selectedIdOrDefault = resolveSelectedId(
    selectableDevices,
    selectedId,
    preferredId,
  );
  const selected = resolveSelectedDevice(
    selectableDevices,
    selectedIdOrDefault,
    selectedOverride,
    selectedId,
  );
  return {
    selectableDevices,
    preferredId,
    selected,
    staticSelected: computeStaticSelected(
      pickerAvailable,
      selected,
      selectableDevices,
    ),
  };
};

const sourceMatchesDevice = (source, device) => {
  if (source?.deviceId) return source.deviceId === device?.id;
  const sourceLabel = String(source?.label || source?.name || '');
  const deviceLabel = String(device?.fullLabel || device?.label || '');
  return (
    sourceLabel === deviceLabel || sourceLabel === String(device?.label || '')
  );
};

const syncSelectedDevice = async ({ next, source, onAdd, onRemove }) => {
  if (next.current || !onAdd) {
    return true;
  }
  if (!source) {
    return (await onAdd(next)) !== null;
  }
  if (!onRemove || sourceMatchesDevice(source, next)) {
    return true;
  }
  const removed = await onRemove(source);
  if (removed === null) return false;
  return (await onAdd(next)) !== null;
};

const audioMeterClass = (level) => {
  if (level > 80) return 'hot';
  if (level > 45) return 'warm';
  return '';
};

export const SourceCard = ({
  devices = [],
  disabled,
  icon,
  picker: Picker,
  pickerProps = {},
  removable = false,
  selectionDisabled = false,
  source,
  title,
  onAdd,
  onRemove,
  onToggle,
}) => {
  const [selectedId, setSelectedId] = React.useState('');
  const [selectedOverride, setSelectedOverride] = React.useState(null);
  const [replacing, setReplacing] = React.useState(false);
  const [selectionError, setSelectionError] = React.useState('');
  const operation = React.useRef({ active: true, pending: false });
  React.useEffect(() => {
    const owner = operation.current;
    owner.active = true;
    return () => {
      owner.active = false;
    };
  }, []);
  const selection = React.useMemo(
    () =>
      resolveSelectionState({
        devices,
        source,
        selectedId,
        selectedOverride,
        pickerAvailable: Boolean(Picker),
      }),
    [devices, source, selectedId, selectedOverride, Picker],
  );
  const { selectableDevices, preferredId, selected, staticSelected } =
    selection;
  const selectedLabel = selectedLabelText(selected);
  const level = Math.max(0, Math.min(100, Number(source?.level || 0)));
  const meterClass = audioMeterClass(level);
  const devicePickerDisabled = disabled || selectionDisabled || replacing;
  const canToggle = sourceToggleEnabled(
    disabled || replacing,
    source,
    selected,
  );

  React.useEffect(() => {
    setSelectedId((current) =>
      selectableDevices.some((device) => device.id === current)
        ? current
        : preferredId,
    );
  }, [preferredId, selectableDevices]);

  const handleSelectDevice = async (next) => {
    const owner = operation.current;
    if (!next || devicePickerDisabled || owner.pending || !owner.active) {
      return;
    }
    owner.pending = true;
    setReplacing(true);
    setSelectionError('');
    setSelectedId(next.id);
    setSelectedOverride(
      selectableDevices.some((device) => device.id === next.id) ? null : next,
    );
    try {
      const changed = await syncSelectedDevice({
        next,
        source,
        onAdd,
        onRemove,
      });
      if (changed === false && owner.active) setSelectedId(preferredId);
    } catch (error) {
      if (owner.active) {
        setSelectedId(preferredId);
        setSelectionError(String(error?.message || error));
      }
    } finally {
      owner.pending = false;
      if (owner.active) setReplacing(false);
    }
  };

  const handleSelect = (event) => {
    const next = selectableDevices.find(
      (device) => device.id === event.target.value,
    );
    handleSelectDevice(next);
  };

  const handleToggle = () => {
    toggleSource({ source, selected, onAdd, onToggle });
  };

  const handleRemove = () => {
    if (!source || devicePickerDisabled || !onRemove) {
      return;
    }
    onRemove(source);
  };

  return (
    <article
      className={classNames(
        'source-card',
        source?.enabled ? 'source-active' : 'source-inactive',
      )}
    >
      {selectionError ? (
        <div role="alert" className="panel-note">
          {selectionError}
        </div>
      ) : null}
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
              disabled={devicePickerDisabled || !onRemove}
              title="Remove source"
              type="button"
              onClick={handleRemove}
            >
              <Trash2 size={14} />
            </button>
          ) : null}
          <SwitchButton
            label={`Enable ${title}`}
            checked={Boolean(source?.enabled)}
            disabled={!canToggle}
            onClick={handleToggle}
          />
        </div>
      </div>

      {source?.enabled ? (
        <div
          aria-label={`${title} audio level`}
          aria-valuemax={100}
          aria-valuemin={0}
          aria-valuenow={level}
          className="audio-meter"
          role="meter"
        >
          <div className={meterClass} style={{ width: `${level}%` }} />
        </div>
      ) : null}

      <SourceCardDevicePicker
        Picker={Picker}
        devices={devices}
        disabled={devicePickerDisabled}
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
};
