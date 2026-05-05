import { SelectShell } from "../../shared/ui/forms/SelectShell";
import { SectionLabel } from "../../shared/ui/SectionLabel";

function modelBasename(model) {
  const s = String(model || "");
  const slash = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  const name = slash >= 0 ? s.slice(slash + 1) : s;
  return name.replace(/\.gguf$/i, "");
}

export function AssistantProfileSelect({ disabled, profileId, profiles, onProfileChange }) {
  return (
    <>
      <SectionLabel>Profile</SectionLabel>
      <SelectShell>
        <select disabled={disabled} value={profileId} onChange={(event) => onProfileChange(event.target.value)}>
          <option value="">— no profile —</option>
          {profiles.map((profile) => (
            <option key={profile.id || profile.label} value={profile.id}>
              {profile.model ? `${profile.label || profile.id} - ${modelBasename(profile.model)}` : profile.label || profile.id}
            </option>
          ))}
        </select>
      </SelectShell>
    </>
  );
}
