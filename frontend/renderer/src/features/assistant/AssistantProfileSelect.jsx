import { modelBasename } from '../../shared/lib/format';
import { SelectShell } from '../../shared/ui/forms/SelectShell';
import { SectionLabel } from '../../shared/ui/SectionLabel';

const profileLabel = (profile) => {
  const label = profile.label || profile.id;
  if (!profile.model) return label;
  return `${label} - ${modelBasename(profile.model)}`;
};

export const AssistantProfileSelect = ({
  disabled,
  profileId,
  profiles,
  onProfileChange,
}) => (
  <>
    <SectionLabel>Profile</SectionLabel>
    <SelectShell>
      <select
        disabled={disabled}
        value={profileId}
        onChange={(event) => onProfileChange(event.target.value)}
      >
        <option value="">— no profile —</option>
        {profiles.map((profile) => (
          <option key={profile.id || profile.label} value={profile.id}>
            {profileLabel(profile)}
          </option>
        ))}
      </select>
    </SelectShell>
  </>
);
