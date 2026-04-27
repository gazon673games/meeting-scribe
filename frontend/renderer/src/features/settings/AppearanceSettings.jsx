import { MonitorOff, Moon, RefreshCw, Sun } from "lucide-react";

const THEMES = [
  { id: "dark", label: "Dark", icon: Moon },
  { id: "light", label: "Light", icon: Sun }
];

export function AppearanceSettings({ dirty, reloading, screenCaptureProtection, theme, onChange, onReloadApp }) {
  const selected = theme === "light" ? "light" : "dark";

  return (
    <section className="settings-section">
      <div className="settings-section-head">
        <h3>Appearance</h3>
      </div>
      <div className="appearance-stack">
        <div className="theme-segmented" role="group" aria-label="Theme">
          {THEMES.map((item) => {
            const Icon = item.icon;
            return (
              <button
                aria-pressed={selected === item.id}
                className={selected === item.id ? "selected" : ""}
                key={item.id}
                type="button"
                onClick={() => onChange({ theme: item.id })}
              >
                <Icon size={15} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>

        <button
          aria-pressed={Boolean(screenCaptureProtection)}
          className={`feature-toggle privacy-toggle ${screenCaptureProtection ? "selected" : ""}`}
          type="button"
          onClick={() => onChange({ screenCaptureProtection: !screenCaptureProtection })}
        >
          <MonitorOff size={15} />
          <span>Screen Share Protection</span>
          <b />
        </button>

        <button className="refresh-app-button" disabled={reloading} type="button" onClick={onReloadApp}>
          <RefreshCw size={15} />
          <span>{dirty ? "Save & Refresh" : "Refresh App"}</span>
        </button>
      </div>
    </section>
  );
}
