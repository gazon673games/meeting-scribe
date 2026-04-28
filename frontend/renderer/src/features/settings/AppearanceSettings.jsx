import { Activity, MonitorOff, Moon, RefreshCw, Sun } from "lucide-react";

const THEMES = [
  { id: "dark", label: "Dark", icon: Moon },
  { id: "light", label: "Light", icon: Sun }
];

export function AppearanceSettings({ capabilities, dirty, perProcessAudio, reloading, screenCaptureProtection, theme, onChange, onReloadApp }) {
  const selected = theme === "light" ? "light" : "dark";
  const showProcessAudio = Boolean(capabilities?.perProcessAudio);

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

        {showProcessAudio ? (
          <button
            aria-pressed={Boolean(perProcessAudio)}
            className={`feature-toggle ${perProcessAudio ? "selected" : ""}`}
            title="Capture audio from specific applications instead of all system audio"
            type="button"
            onClick={() => onChange({ perProcessAudio: !perProcessAudio })}
          >
            <Activity size={15} />
            <span>Per-App Audio Capture</span>
            <b />
          </button>
        ) : null}

        <button className="refresh-app-button" disabled={reloading} type="button" onClick={onReloadApp}>
          <RefreshCw size={15} />
          <span>{dirty ? "Save & Refresh" : "Refresh App"}</span>
        </button>
      </div>
    </section>
  );
}
