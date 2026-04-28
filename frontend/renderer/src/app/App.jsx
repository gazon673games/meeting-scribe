import { X } from "lucide-react";

import { AssistantColumn } from "../features/assistant/AssistantColumn";
import { AudioInputs } from "../features/audio-inputs/AudioInputs";
import { ProcessingColumn } from "../features/processing/ProcessingColumn";
import { TopBar } from "../features/top-bar/TopBar";
import { TranscriptColumn } from "../features/transcript/TranscriptColumn";
import { ResizablePipeline } from "../shared/ui/pipeline/ResizablePipeline";
import { usePipelineLayout } from "../shared/ui/pipeline/usePipelineLayout";
import { useMeetingScribeApp } from "./useMeetingScribeApp";
import "./styles.css";

export function App() {
  const app = useMeetingScribeApp();
  const pipelineColumns = [
    {
      id: "audio-inputs",
      label: "Audio Inputs",
      minWidth: 160,
      defaultWidth: 260,
      element: (
        <AudioInputs
          capabilities={app.capabilities}
          devices={app.devices}
          disabled={!app.capabilities.sourceControl}
          sourceSelectionLocked={app.session.running}
          perProcessAudio={app.settingsDraft.perProcessAudio}
          sources={app.sources}
          onAdd={(device) => app.runBackendAction("add_source", { deviceId: device.id, label: device.fullLabel || device.label })}
          onRemove={(source) => app.runBackendAction("remove_source", { name: source.name })}
          onToggle={(source) => app.runBackendAction("set_source_enabled", { name: source.name, enabled: !source.enabled })}
        />
      )
    },
    {
      id: "processing",
      label: "Processing",
      minWidth: 160,
      defaultWidth: 260,
      element: (
        <ProcessingColumn
          asrMetrics={app.asrMetrics}
          locked={app.session.running}
          offlinePass={app.offlinePass}
          options={app.options}
          resourceUsage={app.resourceUsage}
          session={app.session}
          summary={app.summary}
          draft={app.settingsDraft}
          onChange={app.updateSettings}
          onProfileChange={app.applyProfile}
        />
      )
    },
    {
      id: "transcript",
      label: "Live Transcript",
      minWidth: 180,
      defaultWidth: 320,
      flex: true,
      element: <TranscriptColumn lines={app.transcript} session={app.session} status={app.status} onClear={() => app.runBackendAction("clear_transcript")} />
    },
    {
      id: "assistant",
      label: "AI Assistant",
      minWidth: 180,
      defaultWidth: 360,
      element: (
        <AssistantColumn
          assistant={app.assistant}
          contextReady={app.assistantContextReady}
          disabled={!app.capabilities.assistant || !app.assistant.enabled || app.assistant.busy}
          profiles={app.codexProfiles}
          onInvoke={(params) => app.runBackendAction("invoke_assistant", params)}
        />
      )
    }
  ];
  const pipelineLayout = usePipelineLayout(pipelineColumns);

  return (
    <main className="app-shell" data-theme={app.settingsDraft.theme || "dark"}>
      <TopBar
        canStart={app.canStart}
        canStop={app.canStop}
        loading={app.loading}
        pipelineLayout={pipelineLayout}
        settingsPanel={{
          capabilities: app.capabilities,
          dirty: app.settingsDirty,
          draft: app.settingsDraft,
          hardware: app.hardware,
          locked: app.session.running,
          options: app.options,
          saving: app.savingSettings,
          onAsrChange: app.updateAsrSetting,
          onChange: app.updateSettings,
          onReloadApp: app.reloadApp,
          onSave: app.saveSettings
        }}
        status={app.status}
        asrMetrics={app.asrMetrics}
        onRefresh={app.refresh}
        onStartStop={app.startOrStop}
      />

      {app.error ? (
        <div className="error-strip" role="alert">
          <span>{app.error}</span>
          <button aria-label="Close error" onClick={app.clearError} type="button">
            <X size={16} />
          </button>
        </div>
      ) : null}

      <ResizablePipeline
        columns={pipelineLayout.visibleColumns}
        onHideColumn={pipelineLayout.hideColumn}
        onReorderColumn={pipelineLayout.moveColumnTo}
        resetSignal={pipelineLayout.resetRevision}
      />
    </main>
  );
}
