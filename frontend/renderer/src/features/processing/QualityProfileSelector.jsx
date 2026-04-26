import { Zap } from "lucide-react";

const QUALITY_PROFILES = [
  { profile: "Realtime", label: "Fast", icon: Zap },
  { profile: "Balanced", label: "Balanced" },
  { profile: "Quality", label: "High Quality" }
];

export function QualityProfileSelector({ disabled, selectedProfile, onProfileChange }) {
  return (
    <div className="quality-stack">
      {QUALITY_PROFILES.map((item) => {
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
