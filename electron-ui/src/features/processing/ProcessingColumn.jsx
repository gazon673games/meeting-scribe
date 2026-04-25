import { Check, ChevronDown, Save, Settings, Zap } from "lucide-react";

import { ASR_FIELDS, FALLBACK_OPTIONS, languageLabel, uniqueOptions } from "../../entities/settings/model";
import { formatNumber } from "../../shared/lib/format";
import { FeatureToggle } from "../../shared/ui/FeatureToggle";
import { Field } from "../../shared/ui/Field";
import { PipelineColumn } from "../../shared/ui/PipelineColumn";
import { SectionLabel } from "../../shared/ui/SectionLabel";
import { StatLine } from "../../shared/ui/StatLine";

const QUALITY_PROFILES = [
  { profile: "Realtime", label: "Fast", icon: Zap },
  { profile: "Balanced", label: "Balanced" },
  { profile: "Quality", label: "High Quality" }
];

export function ProcessingColumn({ asrMetrics, dirty, draft, events, locked, offlinePass, options, saving, session, summary, onAsrChange, onChange, onProfileChange, onSave }) {
  const languageOptions = uniqueOptions(options.languages?.length ? options.languages : FALLBACK_OPTIONS.languages, draft.language);
  const modelOptions = uniqueOptions(options.asrModels?.length ? options.asrModels : FALLBACK_OPTIONS.asrModels, draft.model);
  const computeOptions = uniqueOptions(options.computeTypes?.length ? options.computeTypes : FALLBACK_OPTIONS.computeTypes, draft.computeType);
  const overloadOptions = uniqueOptions(options.overloadStrategies?.length ? options.overloadStrategies : FALLBACK_OPTIONS.overloadStrategies, draft.overloadStrategy);

  return (
    <PipelineColumn title="Processing" active={session.running || offlinePass.running} activeTone={offlinePass.running ? "warn" : "muted"}>
      <SectionLabel>Speed vs Quality</SectionLabel>
      <div className="quality-stack">
        {QUALITY_PROFILES.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.profile}
              className={draft.profile === item.profile ? "selected" : ""}
              disabled={locked}
              onClick={() => onProfileChange(item.profile)}
            >
              <span>{item.label}</span>
              {Icon ? <Icon size={15} /> : null}
            </button>
          );
        })}
      </div>

      <SectionLabel>Language</SectionLabel>
      <div className="select-shell">
        <select disabled={locked} value={draft.language} onChange={(event) => onChange({ language: event.target.value })}>
          {languageOptions.map((language) => (
            <option key={language} value={language}>
              {languageLabel(language)}
            </option>
          ))}
        </select>
        <ChevronDown size={15} />
      </div>

      <SectionLabel>Features</SectionLabel>
      <div className="feature-grid">
        <FeatureToggle checked={draft.asrEnabled} disabled={locked} label="Live ASR" onClick={() => onChange({ asrEnabled: !draft.asrEnabled })} />
        <FeatureToggle checked={draft.asrMode === "split"} disabled={locked} label="Speaker Separation" onClick={() => onChange({ asrMode: draft.asrMode === "split" ? "mix" : "split" })} />
        <FeatureToggle checked={draft.wavEnabled} disabled={locked} label="Write WAV" onClick={() => onChange({ wavEnabled: !draft.wavEnabled })} />
        <FeatureToggle checked={draft.offlineOnStop} disabled={locked} label="Offline Pass" onClick={() => onChange({ offlineOnStop: !draft.offlineOnStop })} />
      </div>

      <div className="settings-strip">
        <StatLine label="Active Model" value={summary.model || draft.model || "-"} />
        <StatLine label="ASR Latency" value={`${formatNumber(asrMetrics.avgLatencyS)}s`} accent="green" />
        <StatLine label="ASR Drops" value={`${asrMetrics.segDroppedTotal ?? 0}/${asrMetrics.segSkippedTotal ?? 0}`} accent="blue" />
        <StatLine label="Audio Drops" value={`${session.drops?.droppedOutBlocks ?? 0}/${session.drops?.droppedTapBlocks ?? 0}`} />
      </div>

      <details className="advanced-settings">
        <summary>Advanced ASR</summary>
        <div className="advanced-grid">
          <Field label="Model">
            <select disabled={locked} value={draft.model} onChange={(event) => onChange({ model: event.target.value })}>
              {modelOptions.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Compute">
            <select disabled={locked} value={draft.computeType} onChange={(event) => onChange({ computeType: event.target.value })}>
              {computeOptions.map((computeType) => (
                <option key={computeType} value={computeType}>
                  {computeType}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Overload">
            <select disabled={locked} value={draft.overloadStrategy} onChange={(event) => onChange({ overloadStrategy: event.target.value })}>
              {overloadOptions.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Output">
            <input disabled={locked} type="text" value={draft.outputFile} onChange={(event) => onChange({ outputFile: event.target.value })} />
          </Field>
          {ASR_FIELDS.map((field) => (
            <Field key={field.key} label={field.label}>
              <input
                disabled={locked}
                max={field.max}
                min={field.min}
                step={field.step}
                type="number"
                value={draft.asr[field.key]}
                onChange={(event) => onAsrChange(field.key, event.target.value)}
              />
            </Field>
          ))}
        </div>
      </details>

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
    </PipelineColumn>
  );
}
