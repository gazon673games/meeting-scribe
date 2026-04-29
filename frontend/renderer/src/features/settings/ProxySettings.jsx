import { Network } from "lucide-react";

import { buildProxyUrl } from "../../entities/settings/model";
import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";
import { Field } from "../../shared/ui/Field";

const SCHEMES = ["http", "https", "socks5"];

export function ProxySettings({ draft, onChange }) {
  const enabled = Boolean(draft.assistantProxyEnabled);

  return (
    <CollapsibleSection title="Proxy" defaultOpen={false}>
      <div className="proxy-settings">
        <button
          aria-pressed={enabled}
          className={`feature-toggle proxy-toggle ${enabled ? "selected" : ""}`}
          type="button"
          onClick={() => onChange({ assistantProxyEnabled: !enabled })}
        >
          <Network size={15} />
          <span>Enable Proxy</span>
          <b />
        </button>

        <div className="settings-grid proxy-grid">
          <Field label="Scheme">
            <select
              disabled={!enabled}
              value={draft.assistantProxyScheme || "http"}
              onChange={(event) => onChange({ assistantProxyScheme: event.target.value })}
            >
              {SCHEMES.map((scheme) => (
                <option key={scheme} value={scheme}>
                  {scheme}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Host">
            <input
              disabled={!enabled}
              spellCheck={false}
              value={draft.assistantProxyHost || ""}
              onChange={(event) => onChange({ assistantProxyHost: event.target.value })}
            />
          </Field>
          <Field label="Port">
            <input
              disabled={!enabled}
              inputMode="numeric"
              pattern="[0-9]*"
              value={draft.assistantProxyPort || ""}
              onChange={(event) => onChange({ assistantProxyPort: event.target.value.replace(/\D/g, "") })}
            />
          </Field>
          <Field label="Username">
            <input
              autoComplete="off"
              disabled={!enabled}
              spellCheck={false}
              value={draft.assistantProxyUsername || ""}
              onChange={(event) => onChange({ assistantProxyUsername: event.target.value })}
            />
          </Field>
          <Field label="Password">
            <input
              autoComplete="new-password"
              disabled={!enabled}
              type="password"
              value={draft.assistantProxyPassword || ""}
              onChange={(event) => onChange({ assistantProxyPassword: event.target.value })}
            />
          </Field>
        </div>

        <div className="proxy-note">
          <Network size={13} />
          <span>Used through HTTP_PROXY, HTTPS_PROXY and ALL_PROXY.</span>
        </div>
      </div>
    </CollapsibleSection>
  );
}
