import { SelectShell } from "../../shared/ui/forms/SelectShell";
import { SectionLabel } from "../../shared/ui/SectionLabel";

export function AssistantProfileSelect({ disabled, profileId, profiles, onProfileChange }) {
  return (
    <>
      <SectionLabel>Profile</SectionLabel>
      <SelectShell>
        <select disabled={disabled || !profiles.length} value={profileId} onChange={(event) => onProfileChange(event.target.value)}>
          {profiles.length ? (
            profiles.map((profile) => (
              <option key={profile.id || profile.label} value={profile.id}>
                {profile.label || profile.id}
              </option>
            ))
          ) : (
            <option value="">No profiles</option>
          )}
        </select>
      </SelectShell>
    </>
  );
}
