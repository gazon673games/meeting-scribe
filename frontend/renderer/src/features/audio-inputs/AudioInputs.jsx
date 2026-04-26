import { Activity, Mic, Monitor } from "lucide-react";

import { PipelinePanel } from "../../shared/ui/pipeline/PipelinePanel";
import { AddDevicePicker } from "./AddDevicePicker";
import { SourceCard } from "./SourceCard";

export function AudioInputs({ devices, disabled, headerProps, layoutControls, sources, onAdd, onRemove, onToggle }) {
  const active = sources.some((source) => source.enabled);
  const inputSources = sources.filter((source) => source.kind === "input");
  const loopbackSources = sources.filter((source) => source.kind === "loopback");
  const extraSources = [
    ...inputSources.slice(1),
    ...loopbackSources.slice(1),
    ...sources.filter((source) => !["input", "loopback"].includes(source.kind))
  ];

  return (
    <PipelinePanel title="Audio Inputs" active={active} className="inputs-column" headerControls={layoutControls} headerProps={headerProps}>
      <div className="source-stack">
        <SourceCard
          devices={devices.input}
          disabled={disabled}
          icon={<Mic size={16} />}
          source={inputSources[0]}
          title="Microphone"
          onAdd={onAdd}
          onRemove={onRemove}
          onToggle={onToggle}
        />
        <SourceCard
          devices={devices.loopback}
          disabled={disabled}
          icon={<Monitor size={16} />}
          source={loopbackSources[0]}
          title="System Audio"
          onAdd={onAdd}
          onRemove={onRemove}
          onToggle={onToggle}
        />
        {extraSources.map((source) => (
          <SourceCard
            key={source.name}
            devices={[]}
            disabled={disabled}
            icon={<Activity size={16} />}
            source={source}
            title={source.name}
            onAdd={onAdd}
            onRemove={onRemove}
            onToggle={onToggle}
          />
        ))}
      </div>

      <AddDevicePicker
        disabled={disabled}
        groups={[
          { label: "Microphones", devices: devices.input },
          { label: "System Audio", devices: devices.loopback }
        ]}
        onAdd={onAdd}
      />

      {devices.errors?.length ? <div className="panel-note">{devices.errors.join(" | ")}</div> : null}
    </PipelinePanel>
  );
}
