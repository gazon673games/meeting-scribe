import React from "react";
import { Activity, Mic, Monitor } from "lucide-react";

import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { PipelinePanel } from "../../shared/ui/pipeline/PipelinePanel";
import { AddDevicePicker } from "./AddDevicePicker";
import { SourceCard } from "./SourceCard";
import { SystemAudioPicker, normalizeProcessGroups } from "./SystemAudioPicker";

export function AudioInputs({
  capabilities,
  devices,
  disabled,
  headerProps,
  layoutControls,
  perProcessAudio,
  sourceSelectionLocked = false,
  sources,
  onAdd,
  onRemove,
  onToggle
}) {
  const active = sources.some((source) => source.enabled);
  const inputSources = sources.filter((source) => source.kind === "input");
  const loopbackSources = sources.filter((source) => source.kind === "loopback");
  const processSources = sources.filter((source) => source.kind === "process");
  const systemSources = [...loopbackSources, ...processSources];
  const extraSources = [
    ...inputSources.slice(1),
    ...systemSources.slice(1),
    ...sources.filter((source) => !["input", "loopback", "process"].includes(source.kind))
  ];

  const useProcessDropdown = Boolean(capabilities?.perProcessAudio && perProcessAudio);
  const [processCatalog, setProcessCatalog] = React.useState(null);
  const [processLoading, setProcessLoading] = React.useState(false);
  const [processError, setProcessError] = React.useState("");
  const processCatalogCacheRef = React.useRef({ ts: 0, value: null });
  const processCatalogRequestRef = React.useRef(null);
  const processGroups = React.useMemo(() => normalizeProcessGroups(processCatalog), [processCatalog]);

  const loadProcessCatalog = React.useCallback((options = {}) => {
    if (!useProcessDropdown) {
      return Promise.resolve(null);
    }
    const force = Boolean(options?.force);
    const cached = processCatalogCacheRef.current;
    if (!force && cached.value && Date.now() - cached.ts < 3000) {
      setProcessCatalog(cached.value);
      return Promise.resolve(cached.value);
    }
    if (processCatalogRequestRef.current) {
      return processCatalogRequestRef.current;
    }
    setProcessLoading(true);
    setProcessError("");
    const request = meetingScribeClient
      .request("list_process_sessions")
      .then((result) => {
        const next = result || { groups: [], sessions: [] };
        processCatalogCacheRef.current = { ts: Date.now(), value: next };
        setProcessCatalog(next);
        return next;
      })
      .catch((error) => {
        setProcessError(String(error?.message || error));
        return null;
      })
      .finally(() => {
        processCatalogRequestRef.current = null;
        setProcessLoading(false);
      });
    processCatalogRequestRef.current = request;
    return request;
  }, [useProcessDropdown]);

  React.useEffect(() => {
    if (!useProcessDropdown) {
      setProcessCatalog(null);
      setProcessLoading(false);
      setProcessError("");
      processCatalogCacheRef.current = { ts: 0, value: null };
      processCatalogRequestRef.current = null;
    }
  }, [useProcessDropdown]);

  return (
    <PipelinePanel title="Audio Inputs" active={active} className="inputs-column" headerControls={layoutControls} headerProps={headerProps}>
      <div className="source-stack">
        <SourceCard
          devices={devices.input}
          disabled={disabled || sourceSelectionLocked}
          icon={<Mic size={16} />}
          source={inputSources[0]}
          title="Microphone"
          onAdd={onAdd}
          onRemove={onRemove}
          onToggle={onToggle}
        />
        <SourceCard
          devices={devices.loopback}
          disabled={disabled || sourceSelectionLocked}
          icon={<Monitor size={16} />}
          picker={useProcessDropdown ? SystemAudioPicker : undefined}
          pickerProps={{
            catalog: processCatalog,
            error: processError,
            loading: processLoading,
            onRefresh: loadProcessCatalog
          }}
          source={systemSources[0]}
          title="System Audio"
          onAdd={onAdd}
          onRemove={onRemove}
          onToggle={onToggle}
        />
        {extraSources.map((source) => (
          <SourceCard
            key={source.name}
            devices={[]}
            disabled={disabled || sourceSelectionLocked}
            icon={<Activity size={16} />}
            removable
            source={source}
            title={source.label || source.name}
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
          { label: "System Audio", devices: devices.loopback },
          ...processGroups.map((group) => ({ label: `Applications / ${group.label}`, devices: group.sessions }))
        ]}
        onAdd={onAdd}
        onOpen={loadProcessCatalog}
      />

      {devices.errors?.length ? <div className="panel-note">{devices.errors.join(" | ")}</div> : null}
    </PipelinePanel>
  );
}
