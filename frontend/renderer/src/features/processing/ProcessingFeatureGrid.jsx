import { asrProfileRequiresStreaming } from "../../entities/settings/model";
import { FeatureToggle } from "../../shared/ui/FeatureToggle";

function diarWarning(draft) {
  if (!draft.diarizationEnabled) return null;
  const backend = draft.diarizationBackend || "online";
  if (backend === "online") return null;
  if (backend === "sherpa_onnx" && !draft.diarSherpaEmbeddingModelPath) {
    return "Speaker ID model path not set — configure in Settings";
  }
  return null;
}

export function ProcessingFeatureGrid({ draft, locked, options, onChange }) {
  const diarMsg = diarWarning(draft);
  const diarLocked = locked || Boolean(diarMsg);
  const streamingLocked = asrProfileRequiresStreaming(draft.profile, options);
  return (
    <div className="feature-grid">
      <FeatureToggle checked={draft.asrMode === "split"} disabled={locked} label="Speaker Separation" onClick={() => onChange({ asrMode: draft.asrMode === "split" ? "mix" : "split" })} />
      <FeatureToggle checked={draft.diarizationEnabled} disabled={diarLocked} label="Speaker ID" onClick={() => onChange({ diarizationEnabled: !draft.diarizationEnabled })} />
      {diarMsg ? <p className="feature-grid-warning">{diarMsg}</p> : null}
      <FeatureToggle checked={draft.wavEnabled} disabled={locked} label="Record to File" onClick={() => onChange({ wavEnabled: !draft.wavEnabled })} />
      <FeatureToggle
        checked={streamingLocked || draft.streamingEnabled}
        disabled={locked || streamingLocked}
        label="Word-by-Word"
        onClick={() => onChange({ streamingEnabled: !draft.streamingEnabled })}
      />
    </div>
  );
}
