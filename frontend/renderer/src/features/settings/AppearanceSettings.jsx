import { Moon, Palette, Sun } from "lucide-react";

import { THEME_OPTIONS } from "../../entities/settings/model";
import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";

const THEME_ICONS = { dark: Moon, light: Sun, grimdark: Palette };

export function AppearanceSettings({ theme = "dark", onChange }) {
  return (
    <CollapsibleSection title="Appearance" defaultOpen={true}>
      <div className="appearance-stack">
        <div aria-label="Color theme" className="theme-segmented" role="group">
          {THEME_OPTIONS.map(({ id, label }) => {
            const Icon = THEME_ICONS[id];
            return (
              <button
                aria-pressed={theme === id}
                className={theme === id ? "selected" : ""}
                key={id}
                type="button"
                onClick={() => onChange({ theme: id })}
              >
                <Icon size={15} />
                {label}
              </button>
            );
          })}
        </div>
      </div>
    </CollapsibleSection>
  );
}
