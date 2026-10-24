import { asrProfileRequiresStreaming } from '../../entities/settings/model';
import { FeatureToggle } from '../../shared/ui/FeatureToggle';

const diarWarning = (draft) => {
  if (!draft.diarizationEnabled) return null;
  const backend = draft.diarizationBackend || 'online';
  if (backend === 'online') return null;
  if (backend === 'sherpa_onnx' && !draft.diarSherpaEmbeddingModelPath) {
    return 'Speaker ID model path not set — configure in Settings';
  }
  return null;
};

const offlinePassDescription = (draft, locked) => {
  if (!draft.wavEnabled) return 'Enable Record to File first.';
  return locked ? 'Available after recording stops.' : undefined;
};

export const ProcessingFeatureGrid = ({ draft, locked, options, onChange }) => {
  const diarizationWarning = diarWarning(draft);
  const streamingLocked = asrProfileRequiresStreaming(draft.profile, options);
  const offlineDescription = offlinePassDescription(draft, locked);
  return (
    <div className="feature-grid">
      <FeatureToggle
        checked={draft.asrMode === 'split'}
        disabled={locked}
        label="Speaker Separation"
        onClick={() =>
          onChange({ asrMode: draft.asrMode === 'split' ? 'mix' : 'split' })
        }
      />
      <FeatureToggle
        checked={draft.diarizationEnabled}
        disabled={locked}
        label="Speaker ID"
        onClick={() =>
          onChange({ diarizationEnabled: !draft.diarizationEnabled })
        }
      />
      {diarizationWarning ? (
        <p className="feature-grid-warning">{diarizationWarning}</p>
      ) : null}
      <FeatureToggle
        checked={draft.wavEnabled}
        disabled={locked}
        label="Record to File"
        onClick={() => onChange({ wavEnabled: !draft.wavEnabled })}
      />
      <FeatureToggle
        checked={Boolean(draft.offlineOnStop)}
        disabled={locked || !draft.wavEnabled}
        label="Offline Quality Pass"
        description={offlineDescription}
        onClick={() => onChange({ offlineOnStop: !draft.offlineOnStop })}
      />
      <FeatureToggle
        checked={streamingLocked || draft.streamingEnabled}
        disabled={locked || streamingLocked}
        label="Word-by-Word"
        description={
          streamingLocked
            ? 'Required by the selected processing mode.'
            : undefined
        }
        onClick={() => onChange({ streamingEnabled: !draft.streamingEnabled })}
      />
      {locked ? (
        <p className="control-hint">Stop recording to change these features.</p>
      ) : null}
    </div>
  );
};
