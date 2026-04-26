import React from "react";
import { Activity, ChevronDown, Mic, Monitor, Plus, Trash2 } from "lucide-react";

import { formatDelay } from "../../shared/lib/format";
import { PipelineColumn } from "../../shared/ui/PipelineColumn";
import { SwitchButton } from "../../shared/ui/SwitchButton";

export function AudioInputs({ devices, disabled, sources, onAdd, onDelay, onRemove, onToggle }) {
  const active = sources.some((source) => source.enabled);
  const inputSources = sources.filter((source) => source.kind === "input");
  const loopbackSources = sources.filter((source) => source.kind === "loopback");
  const otherSources = sources.filter((source) => !["input", "loopback"].includes(source.kind));

  return (
    <PipelineColumn title="Audio Inputs" active={active} className="inputs-column">
      <div className="source-stack">
        {inputSources.map((source) => (
          <SourceCard
            key={source.name}
            disabled={disabled}
            icon={<Mic size={16} />}
            source={source}
            title="Microphone"
            onDelay={onDelay}
            onRemove={onRemove}
            onToggle={onToggle}
          />
        ))}
        {loopbackSources.map((source) => (
          <SourceCard
            key={source.name}
            disabled={disabled}
            icon={<Monitor size={16} />}
            source={source}
            title="System Audio"
            onDelay={onDelay}
            onRemove={onRemove}
            onToggle={onToggle}
          />
        ))}
        {otherSources.map((source) => (
          <SourceCard
            key={source.name}
            disabled={disabled}
            icon={<Activity size={16} />}
            source={source}
            title={source.name}
            onDelay={onDelay}
            onRemove={onRemove}
            onToggle={onToggle}
          />
        ))}
      </div>

      <AddDevicePicker disabled={disabled} devices={devices.input} label="Add Microphone" onAdd={onAdd} />
      <AddDevicePicker disabled={disabled} devices={devices.loopback} label="Add System Audio" onAdd={onAdd} />

      {devices.errors?.length ? <div className="panel-note">{devices.errors.join(" | ")}</div> : null}
    </PipelineColumn>
  );
}

function SourceCard({ disabled, icon, source, title, onDelay, onRemove, onToggle }) {
  const level = Math.max(0, Math.min(100, Number(source.level || 0)));
  return (
    <article className="source-card">
      <div className="source-head">
        <div className="source-title">
          {icon}
          <strong>{title}</strong>
        </div>
        <SwitchButton checked={source.enabled} disabled={disabled} onClick={() => onToggle(source)} />
      </div>

      <div className="audio-meter">
        <div className={level > 80 ? "hot" : level > 55 ? "warm" : ""} style={{ width: `${source.enabled ? level : 0}%` }} />
      </div>

      <div className="select-shell">
        <select value={source.label || source.name} disabled>
          <option>{source.label || source.name}</option>
        </select>
        <ChevronDown size={14} />
      </div>

      <div className="source-actions">
        <SourceDelayControl disabled={disabled} source={source} onDelay={onDelay} />
        <span className="source-status">{source.status}</span>
        <button className="icon-action" disabled={disabled} onClick={() => onRemove(source)} title="Remove source">
          <Trash2 size={14} />
        </button>
      </div>
    </article>
  );
}

function SourceDelayControl({ disabled, source, onDelay }) {
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

function AddDevicePicker({ devices, disabled, label, onAdd }) {
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
      <div className="select-shell">
        <select disabled={disabled} value={selected.id} onChange={(event) => setSelectedId(event.target.value)}>
          {devices.map((device) => (
            <option key={device.id} value={device.id}>
              {device.label}
            </option>
          ))}
        </select>
        <ChevronDown size={14} />
      </div>
      <button disabled={disabled} onClick={() => onAdd(selected)}>
        <Plus size={15} />
        {label}
      </button>
    </div>
  );
}
