import { Activity, MonitorOff } from "lucide-react";

import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";

export function CaptureSettings({ capabilities, perProcessAudio, screenCaptureProtection, onChange }) {
  return (
    <CollapsibleSection title="Audio & Privacy">
      <div className="appearance-stack">
        <button
          aria-pressed={Boolean(screenCaptureProtection)}
          className={`feature-toggle privacy-toggle ${screenCaptureProtection ? "selected" : ""}`}
          title="Hide this app window from screen captures and screen sharing."
          type="button"
          onClick={() => onChange({ screenCaptureProtection: !screenCaptureProtection })}
        >
          <MonitorOff size={15} />
          <span>Screen Share Protection</span>
          <b />
        </button>
        {capabilities?.perProcessAudio ? (
          <button
            aria-pressed={Boolean(perProcessAudio)}
            className={`feature-toggle ${perProcessAudio ? "selected" : ""}`}
            title="Capture audio from specific applications instead of all system audio."
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
