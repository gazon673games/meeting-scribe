import { SelectShell } from "../../shared/ui/forms/SelectShell";

const DEFAULT_QUALITY_PROFILES = ["Ultra Fast", "Realtime", "Quality"];
const PROFILE_META = {
  "Ultra Fast": { label: "Ultra Fast", description: "Prioritize low latency for live captions." },
  Realtime: { label: "Fast", description: "Live transcription with a focus on responsiveness." },
  Balanced: { label: "Balanced", description: "Balance transcription quality and responsiveness." },
  Quality: { label: "High Quality", description: "Prioritize recognition quality over response speed." }
};

export function QualityProfileSelector({ disabled, profiles, selectedProfile, onProfileChange }) {
  const qualityProfiles = (profiles?.length ? profiles : DEFAULT_QUALITY_PROFILES)
    .filter((profile) => profile !== "Custom")
    .map((profile) => ({ profile, ...(PROFILE_META[profile] || { label: profile }) }));

  return (
    <div className="quality-selector">
      <SelectShell>
        <select aria-label="Processing mode" disabled={disabled} value={selectedProfile} onChange={(event) => onProfileChange(event.target.value)}>
          {selectedProfile && !qualityProfiles.some((item) => item.profile === selectedProfile) ? <option value={selectedProfile}>{selectedProfile}</option> : null}
          {qualityProfiles.map((item) => <option key={item.profile} value={item.profile}>{item.label}</option>)}
        </select>
      </SelectShell>
      <p className="control-hint">{disabled ? "Stop recording to change the processing mode." : PROFILE_META[selectedProfile]?.description || "Custom transcription settings."}</p>
    </div>
  );
}
