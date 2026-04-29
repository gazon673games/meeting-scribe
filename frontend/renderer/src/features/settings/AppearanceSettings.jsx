import { Activity, MonitorOff } from "lucide-react";

import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";

export function AppearanceSettings({ capabilities, perProcessAudio, screenCaptureProtection, onChange }) {
  const showProcessAudio = Boolean(capabilities?.perProcessAudio);

  return (
    <CollapsibleSection title="Appearance" defaultOpen={true}>
      <div className="appearance-stack">
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
      </div>
    </CollapsibleSection>
  );
}
