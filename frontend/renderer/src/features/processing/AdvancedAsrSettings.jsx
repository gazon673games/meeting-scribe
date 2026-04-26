import { ASR_FIELDS } from "../../entities/settings/model";
import { Field } from "../../shared/ui/Field";

export function AdvancedAsrSettings({ computeOptions, draft, locked, modelOptions, overloadOptions, onAsrChange, onChange }) {
  return (
    <details className="advanced-settings">
      <summary>Advanced ASR</summary>
      <div className="advanced-grid">
        <Field label="Model">
          <select disabled={locked} value={draft.model} onChange={(event) => onChange({ model: event.target.value })}>
            {modelOptions.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Compute">
          <select disabled={locked} value={draft.computeType} onChange={(event) => onChange({ computeType: event.target.value })}>
            {computeOptions.map((computeType) => (
              <option key={computeType} value={computeType}>
                {computeType}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Overload">
          <select disabled={locked} value={draft.overloadStrategy} onChange={(event) => onChange({ overloadStrategy: event.target.value })}>
            {overloadOptions.map((strategy) => (
              <option key={strategy} value={strategy}>
                {strategy}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Output">
          <input disabled={locked} type="text" value={draft.outputFile} onChange={(event) => onChange({ outputFile: event.target.value })} />
        </Field>
        {ASR_FIELDS.map((field) => (
          <Field key={field.key} label={field.label}>
            <input
              disabled={locked}
              max={field.max}
              min={field.min}
              step={field.step}
              type="number"
              value={draft.asr[field.key]}
              onChange={(event) => onAsrChange(field.key, event.target.value)}
            />
          </Field>
        ))}
      </div>
    </details>
  );
}
