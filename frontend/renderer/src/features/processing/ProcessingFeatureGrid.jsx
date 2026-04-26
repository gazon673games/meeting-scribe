import { FeatureToggle } from "../../shared/ui/FeatureToggle";

export function ProcessingFeatureGrid({ draft, locked, onChange }) {
  return (
    <div className="feature-grid">
      <FeatureToggle checked={draft.asrMode === "split"} disabled={locked} label="Speaker Separation" onClick={() => onChange({ asrMode: draft.asrMode === "split" ? "mix" : "split" })} />
      <FeatureToggle checked={draft.wavEnabled} disabled={locked} label="Record to File" onClick={() => onChange({ wavEnabled: !draft.wavEnabled })} />
      <FeatureToggle checked={draft.offlineOnStop} disabled={locked} label="Offline Pass" onClick={() => onChange({ offlineOnStop: !draft.offlineOnStop })} />
    </div>
  );
}
