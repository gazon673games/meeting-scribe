import { DIARIZATION_BACKENDS, DIARIZATION_PROVIDERS, FALLBACK_OPTIONS, uniqueOptions } from "../../entities/settings/model";
import { Field } from "../../shared/ui/Field";
import { FeatureToggle } from "../../shared/ui/FeatureToggle";

export function DiarizationSettings({ draft, locked, options, onChange }) {
  const enabled = Boolean(draft.diarizationEnabled);
  const availableOptions = options || {};
  const backendOptions = uniqueOptions(
    availableOptions.diarizationBackends?.length
      ? availableOptions.diarizationBackends
      : FALLBACK_OPTIONS.diarizationBackends || DIARIZATION_BACKENDS,
    draft.diarizationBackend
  );
  const providerOptions = uniqueOptions(
    availableOptions.diarizationProviders?.length
      ? availableOptions.diarizationProviders
      : FALLBACK_OPTIONS.diarizationProviders || DIARIZATION_PROVIDERS,
    draft.diarSherpaProvider
  );
  const isSherpa = draft.diarizationBackend === "sherpa_onnx";

  return (
    <section className="settings-section">
      <div className="settings-section-head">
        <h3>Speaker ID</h3>
      </div>
      <div className="feature-grid settings-feature-grid">
        <FeatureToggle
          checked={enabled}
          disabled={locked}
          label="Identify Speakers"
          onClick={() => onChange({ diarizationEnabled: !enabled })}
        />
        <FeatureToggle
          checked={Boolean(draft.diarizationSidecarEnabled)}
          disabled={locked || !enabled}
          label="Sidecar"
          onClick={() => onChange({ diarizationSidecarEnabled: !draft.diarizationSidecarEnabled })}
        />
      </div>
      <div className="settings-grid">
        <Field label="Backend">
          <select
            disabled={locked || !enabled}
            value={draft.diarizationBackend}
            onChange={(event) => onChange({ diarizationBackend: event.target.value })}
          >
            {backendOptions.map((backend) => (
              <option key={backend} value={backend}>
                {backend}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Queue">
          <input
            disabled={locked || !enabled}
            max={500}
            min={1}
            step={1}
            type="number"
            value={draft.diarizationQueueSize}
            onChange={(event) => onChange({ diarizationQueueSize: event.target.value })}
          />
        </Field>
        {isSherpa ? (
          <>
            <Field label="Sherpa Model">
              <input
                disabled={locked || !enabled}
                spellCheck={false}
                value={draft.diarSherpaEmbeddingModelPath}
                onChange={(event) => onChange({ diarSherpaEmbeddingModelPath: event.target.value })}
              />
            </Field>
            <Field label="Sherpa Provider">
              <select
                disabled={locked || !enabled}
                value={draft.diarSherpaProvider}
                onChange={(event) => onChange({ diarSherpaProvider: event.target.value })}
              >
                {providerOptions.map((provider) => (
                  <option key={provider} value={provider}>
                    {provider}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Sherpa Threads">
              <input
                disabled={locked || !enabled}
                max={32}
                min={1}
                step={1}
                type="number"
                value={draft.diarSherpaNumThreads}
                onChange={(event) => onChange({ diarSherpaNumThreads: event.target.value })}
              />
            </Field>
          </>
        ) : null}
      </div>
    </section>
  );
}
