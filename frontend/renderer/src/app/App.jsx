import { AssistantColumn } from "../features/assistant/AssistantColumn";
import { AudioInputs } from "../features/audio-inputs/AudioInputs";
import { ProcessingColumn } from "../features/processing/ProcessingColumn";
import { TopBar } from "../features/top-bar/TopBar";
import { TranscriptColumn } from "../features/transcript/TranscriptColumn";
import { useMeetingScribeApp } from "./useMeetingScribeApp";
import "./styles.css";

export function App() {
  const app = useMeetingScribeApp();

  return (
    <main className="app-shell">
      <TopBar
        canStart={app.canStart}
        canStop={app.canStop}
        loading={app.loading}
        mode={app.mode}
        status={app.status}
        session={app.session}
        backendStatus={app.backendStatus}
        asrMetrics={app.asrMetrics}
        onModeChange={app.setMode}
        onRefresh={app.refresh}
        onStartStop={app.startOrStop}
      />

      {app.error ? <div className="error-strip">{app.error}</div> : null}

      <section className="pipeline">
        <AudioInputs
          devices={app.devices}
          disabled={app.session.running}
          sources={app.sources}
          onAdd={(device) => app.runBackendAction("add_source", { deviceId: device.id })}
          onDelay={(source, delayMs) => app.runBackendAction("set_source_delay", { name: source.name, delayMs })}
          onRemove={(source) => app.runBackendAction("remove_source", { name: source.name })}
          onToggle={(source) => app.runBackendAction("set_source_enabled", { name: source.name, enabled: !source.enabled })}
        />

        <ProcessingColumn
          asrMetrics={app.asrMetrics}
          dirty={app.settingsDirty}
          events={app.events}
          locked={app.session.running}
          offlinePass={app.offlinePass}
          options={app.options}
          saving={app.savingSettings}
          session={app.session}
          summary={app.summary}
          draft={app.settingsDraft}
          onAsrChange={app.updateAsrSetting}
          onChange={app.updateSettings}
          onProfileChange={app.applyProfile}
          onSave={app.saveSettings}
        />

        <TranscriptColumn
          lines={app.transcript}
          session={app.session}
          status={app.status}
          onClear={() => app.runBackendAction("clear_transcript")}
        />

        <AssistantColumn
          assistant={app.assistant}
          disabled={!app.capabilities.assistant || !app.assistant.enabled || app.assistant.busy}
          profiles={app.codexProfiles}
          onInvoke={(params) => app.runBackendAction("invoke_assistant", params)}
        />
      </section>
    </main>
  );
}
