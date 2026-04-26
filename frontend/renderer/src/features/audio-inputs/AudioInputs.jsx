import { Activity, Mic, Monitor } from "lucide-react";

import { PipelinePanel } from "../../shared/ui/pipeline/PipelinePanel";
import { AddDevicePicker } from "./AddDevicePicker";
import { SourceCard } from "./SourceCard";

export function AudioInputs({ devices, disabled, sources, onAdd, onDelay, onRemove, onToggle }) {
  const active = sources.some((source) => source.enabled);
  const inputSources = sources.filter((source) => source.kind === "input");
  const loopbackSources = sources.filter((source) => source.kind === "loopback");
  const otherSources = sources.filter((source) => !["input", "loopback"].includes(source.kind));

  return (
    <PipelinePanel title="Audio Inputs" active={active} className="inputs-column">
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
    </PipelinePanel>
  );
}
