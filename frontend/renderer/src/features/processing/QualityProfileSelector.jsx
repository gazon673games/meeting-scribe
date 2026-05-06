import { Gem, Rocket, Zap } from "lucide-react";

const DEFAULT_QUALITY_PROFILES = ["Ultra Fast", "Realtime", "Quality"];
const PROFILE_META = {
  "Ultra Fast": { label: "Ultra Fast", icon: Rocket },
  Realtime: { label: "Fast", icon: Zap },
  Balanced: { label: "Balanced" },
  Quality: { label: "High Quality", icon: Gem }
};

export function QualityProfileSelector({ disabled, profiles, selectedProfile, onProfileChange }) {
  const qualityProfiles = (profiles?.length ? profiles : DEFAULT_QUALITY_PROFILES)
    .filter((profile) => profile !== "Custom")
    .map((profile) => ({ profile, ...(PROFILE_META[profile] || { label: profile }) }));

  return (
    <div className="quality-stack">
      {qualityProfiles.map((item) => {
        const Icon = item.icon;
        return (
          <button
            key={item.profile}
            className={selectedProfile === item.profile ? "selected" : ""}
            disabled={disabled}
            onClick={() => onProfileChange(item.profile)}
          >
            <span>{item.label}</span>
            {Icon ? <Icon size={15} /> : null}
          </button>
        );
      })}
    </div>
  );
}
