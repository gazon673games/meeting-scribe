import { Check, Save, Settings } from "lucide-react";

import { FALLBACK_OPTIONS, languageLabel, uniqueOptions } from "../../entities/settings/model";
import { SelectShell } from "../../shared/ui/forms/SelectShell";
import { PipelinePanel } from "../../shared/ui/pipeline/PipelinePanel";
import { SectionLabel } from "../../shared/ui/SectionLabel";
import { AdvancedAsrSettings } from "./AdvancedAsrSettings";
import { ProcessingFeatureGrid } from "./ProcessingFeatureGrid";
import { ProcessingStats } from "./ProcessingStats";
import { QualityProfileSelector } from "./QualityProfileSelector";

export function ProcessingColumn({ asrMetrics, dirty, draft, events, locked, offlinePass, options, saving, session, summary, onAsrChange, onChange, onProfileChange, onSave }) {
  const languageOptions = uniqueOptions(options.languages?.length ? options.languages : FALLBACK_OPTIONS.languages, draft.language);
  const modelOptions = uniqueOptions(options.asrModels?.length ? options.asrModels : FALLBACK_OPTIONS.asrModels, draft.model);
  const computeOptions = uniqueOptions(options.computeTypes?.length ? options.computeTypes : FALLBACK_OPTIONS.computeTypes, draft.computeType);
  const overloadOptions = uniqueOptions(options.overloadStrategies?.length ? options.overloadStrategies : FALLBACK_OPTIONS.overloadStrategies, draft.overloadStrategy);

  return (
    <PipelinePanel title="Processing" active={session.running || offlinePass.running} activeTone={offlinePass.running ? "warn" : "muted"}>
      <SectionLabel>Speed vs Quality</SectionLabel>
      <QualityProfileSelector disabled={locked} selectedProfile={draft.profile} onProfileChange={onProfileChange} />

      <SectionLabel>Language</SectionLabel>
      <SelectShell>
        <select disabled={locked} value={draft.language} onChange={(event) => onChange({ language: event.target.value })}>
          {languageOptions.map((language) => (
            <option key={language} value={language}>
              {languageLabel(language)}
            </option>
          ))}
        </select>
      </SelectShell>

      <SectionLabel>Features</SectionLabel>
      <ProcessingFeatureGrid draft={draft} locked={locked} onChange={onChange} />

      <ProcessingStats asrMetrics={asrMetrics} draft={draft} session={session} summary={summary} />

      <AdvancedAsrSettings
        computeOptions={computeOptions}
        draft={draft}
        locked={locked}
        modelOptions={modelOptions}
        overloadOptions={overloadOptions}
        onAsrChange={onAsrChange}
        onChange={onChange}
      />

      <button className="save-button" disabled={!dirty || saving} onClick={onSave}>
        {dirty ? <Save size={15} /> : <Check size={15} />}
        {saving ? "Saving" : dirty ? "Save Settings" : "Saved"}
      </button>

      <div className="state-card">
        <div>
          <Settings size={15} />
          <span>Current State</span>
        </div>
        <strong>{session.running ? "Processing audio stream..." : offlinePass.running ? "Finalizing transcription..." : "Ready to record"}</strong>
        {events.length ? <small>{events[0].type}</small> : null}
      </div>
    </PipelinePanel>
  );
}
