import { Bot, Network, Plus, Trash2 } from "lucide-react";

import { ASSISTANT_REASONING_OPTIONS, normalizeAssistantProfiles } from "../../entities/settings/model";
import { Field } from "../../shared/ui/Field";

const SCHEMES = ["http", "https", "socks5"];

export function AssistantProxySettings({ draft, onChange }) {
  const enabled = Boolean(draft.assistantProxyEnabled);
  const assistantEnabled = Boolean(draft.assistantEnabled);
  const profiles = normalizeAssistantProfiles(draft.assistantProfiles);

  const updateProfile = (index, patch) => {
    const next = profiles.map((profile, i) => (i === index ? { ...profile, ...patch } : profile));
    onChange({ assistantProfiles: next });
  };

  const addProfile = () => {
    const id = `codex_${Date.now().toString(36)}`;
    onChange({
      assistantProfiles: [
        ...profiles,
        {
          id,
          label: "New Codex Profile",
          provider: "codex",
          prompt: "",
          model: "",
          reasoning_effort: "low",
          codex_profile: "",
          answer_prompt: "",
          extra_args: []
        }
      ],
      assistantSelectedProfileId: id
    });
  };

  const removeProfile = (index) => {
    if (profiles.length <= 1) {
      return;
    }
    const removed = profiles[index];
    const next = profiles.filter((_, i) => i !== index);
    onChange({
      assistantProfiles: next,
      assistantSelectedProfileId: draft.assistantSelectedProfileId === removed.id ? next[0]?.id || "" : draft.assistantSelectedProfileId
    });
  };

  return (
    <section className="settings-section">
      <div className="settings-section-head">
        <h3>Assistant</h3>
      </div>
      <div className="proxy-settings">
        <button
          aria-pressed={assistantEnabled}
          className={`feature-toggle proxy-toggle ${assistantEnabled ? "selected" : ""}`}
          type="button"
          onClick={() => onChange({ assistantEnabled: !assistantEnabled })}
        >
          <Bot size={15} />
          <span>Enable Assistant</span>
          <b />
        </button>

        <Field label="Default Profile">
          <select
            disabled={!assistantEnabled || profiles.length === 0}
            value={draft.assistantSelectedProfileId || profiles[0]?.id || ""}
            onChange={(event) => onChange({ assistantSelectedProfileId: event.target.value })}
          >
            {profiles.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.label || profile.id}
              </option>
            ))}
          </select>
        </Field>

        <div className="assistant-profiles-list">
          {profiles.map((profile, index) => (
            <div className="assistant-profile-card" key={profile.id || index}>
              <div className="assistant-profile-head">
                <span>{profile.provider === "codex" ? "Codex CLI" : profile.provider}</span>
                <button
                  className="model-row-btn danger"
                  disabled={!assistantEnabled || profiles.length <= 1}
                  title="Delete profile"
                  type="button"
                  onClick={() => removeProfile(index)}
                >
                  <Trash2 size={12} />
                </button>
              </div>
              <div className="settings-grid assistant-profile-grid">
                <Field label="Label">
                  <input
                    disabled={!assistantEnabled}
                    value={profile.label}
                    onChange={(event) => updateProfile(index, { label: event.target.value })}
                  />
                </Field>
                <Field label="Model">
                  <input
                    disabled={!assistantEnabled}
                    placeholder="gpt-5.3-codex"
                    spellCheck={false}
                    value={profile.model}
                    onChange={(event) => updateProfile(index, { model: event.target.value })}
                  />
                </Field>
                <Field label="Reasoning">
                  <select
                    disabled={!assistantEnabled}
                    value={profile.reasoning_effort || ""}
                    onChange={(event) => updateProfile(index, { reasoning_effort: event.target.value })}
                  >
                    <option value="">default</option>
                    {ASSISTANT_REASONING_OPTIONS.map((effort) => (
                      <option key={effort} value={effort}>
                        {effort}
                      </option>
                    ))}
                  </select>
                </Field>
                <Field label="Codex Profile">
                  <input
                    disabled={!assistantEnabled}
                    spellCheck={false}
                    value={profile.codex_profile || ""}
                    onChange={(event) => updateProfile(index, { codex_profile: event.target.value })}
                  />
                </Field>
              </div>
              <Field label="Instructions">
                <textarea
                  className="assistant-profile-prompt"
                  disabled={!assistantEnabled}
                  spellCheck={false}
                  value={profile.prompt || ""}
                  onChange={(event) => updateProfile(index, { prompt: event.target.value })}
                />
              </Field>
            </div>
          ))}
        </div>

        <button className="model-download-button assistant-add-profile" disabled={!assistantEnabled} type="button" onClick={addProfile}>
          <Plus size={13} />
          Add Profile
        </button>

        <button
          aria-pressed={enabled}
          className={`feature-toggle proxy-toggle ${enabled ? "selected" : ""}`}
          type="button"
          onClick={() => onChange({ assistantProxyEnabled: !enabled })}
        >
          <Network size={15} />
          <span>Use Proxy</span>
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
          <Bot size={13} />
          <span>Used through HTTP_PROXY, HTTPS_PROXY and ALL_PROXY.</span>
        </div>
      </div>
    </section>
  );
}
