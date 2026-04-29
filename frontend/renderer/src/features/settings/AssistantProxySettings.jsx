import React from "react";
import { Bot, Plus, Radio, Trash2 } from "lucide-react";

import { ASSISTANT_REASONING_OPTIONS, normalizeAssistantProfiles } from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";
import { Field } from "../../shared/ui/Field";

export function AssistantProxySettings({ draft, onChange }) {
  const assistantEnabled = Boolean(draft.assistantEnabled);
  const profiles = normalizeAssistantProfiles(draft.assistantProfiles);
  const [pingStates, setPingStates] = React.useState({});

  const updateProfile = (index, patch) => {
    const next = profiles.map((profile, i) => (i === index ? { ...profile, ...patch } : profile));
    onChange({ assistantProfiles: next });
  };

  const addProfile = () => {
    const id = `codex_${Date.now().toString(36)}`;
    onChange({
      assistantProfiles: [
        ...profiles,
        { id, label: "New Codex Profile", provider: "codex", prompt: "", model: "", reasoning_effort: "low", codex_profile: "", answer_prompt: "", extra_args: [], offline: false }
      ],
      assistantSelectedProfileId: id
    });
  };

  const removeProfile = (index) => {
    if (profiles.length <= 1) return;
    const removed = profiles[index];
    const next = profiles.filter((_, i) => i !== index);
    onChange({
      assistantProfiles: next,
      assistantSelectedProfileId: draft.assistantSelectedProfileId === removed.id ? next[0]?.id || "" : draft.assistantSelectedProfileId
    });
  };

  const pingProfile = React.useCallback(async (profile) => {
    const key = profile.id;
    setPingStates((s) => ({ ...s, [key]: { busy: true } }));
    try {
      const result = await meetingScribeClient.request("ping_assistant_provider", {
        providerId: profile.provider || "codex",
        profileId: profile.id,
        codexProfile: profile.codex_profile || ""
      });
      setPingStates((s) => ({ ...s, [key]: { busy: false, ...result } }));
    } catch (e) {
      setPingStates((s) => ({
        ...s,
        [key]: { busy: false, ok: false, errorCode: "ping_failed", message: String(e?.message || e) }
      }));
    }
  }, []);

  return (
    <CollapsibleSection title="Assistant" defaultOpen={false}>
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
            onChange={(e) => onChange({ assistantSelectedProfileId: e.target.value })}
          >
            {profiles.map((profile) => (
              <option key={profile.id} value={profile.id}>{profile.label || profile.id}</option>
            ))}
          </select>
        </Field>

        <div className="assistant-profiles-list">
          {profiles.map((profile, index) => {
            const ping = pingStates[profile.id];
            return (
              <div className="assistant-profile-card" key={profile.id || index}>
                <div className="assistant-profile-head">
                  <span>{profile.provider === "codex" ? "Codex CLI" : (profile.provider || "codex")}</span>
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
                    <input disabled={!assistantEnabled} value={profile.label} onChange={(e) => updateProfile(index, { label: e.target.value })} />
                  </Field>
                  <Field label="Model">
                    <input disabled={!assistantEnabled} placeholder="gpt-5.3-codex" spellCheck={false} value={profile.model} onChange={(e) => updateProfile(index, { model: e.target.value })} />
                  </Field>
                  <Field label="Reasoning">
                    <select disabled={!assistantEnabled} value={profile.reasoning_effort || ""} onChange={(e) => updateProfile(index, { reasoning_effort: e.target.value })}>
                      <option value="">default</option>
                      {ASSISTANT_REASONING_OPTIONS.map((effort) => (
                        <option key={effort} value={effort}>{effort}</option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Codex Profile">
                    <input disabled={!assistantEnabled} spellCheck={false} value={profile.codex_profile || ""} onChange={(e) => updateProfile(index, { codex_profile: e.target.value })} />
                  </Field>
                  <Field label="Mode">
                    <button
                      aria-pressed={Boolean(profile.offline)}
                      className={`feature-toggle ${profile.offline ? "selected" : ""}`}
                      disabled={!assistantEnabled}
                      type="button"
                      onClick={() => updateProfile(index, { offline: !profile.offline })}
                    >
                      <span>{profile.offline ? "Offline" : "Online"}</span>
                      <b />
                    </button>
                  </Field>
                </div>

                <Field label="Instructions">
                  <textarea
                    className="assistant-profile-prompt"
                    disabled={!assistantEnabled}
                    rows={3}
                    spellCheck={false}
                    value={profile.prompt || ""}
                    onChange={(e) => updateProfile(index, { prompt: e.target.value })}
                  />
                </Field>

                <div className="assistant-profile-ping">
                  <button
                    className="model-download-button"
                    disabled={!assistantEnabled || ping?.busy}
                    type="button"
                    onClick={() => pingProfile(profile)}
                  >
                    <Radio size={13} />
                    Ping
                  </button>
                  <PingStatus ping={ping} />
                </div>
              </div>
            );
          })}
        </div>

        <button className="model-download-button assistant-add-profile" disabled={!assistantEnabled} type="button" onClick={addProfile}>
          <Plus size={13} />
          Add Profile
        </button>
      </div>
    </CollapsibleSection>
  );
}

function PingStatus({ ping }) {
  if (!ping) return null;
  if (ping.busy) return <span className="ping-status ping-busy">pinging…</span>;

  const isAuthError = /auth|unauthorized|not_logged|login/i.test(ping.errorCode || "");
  if (ping.ok) return <span className="ping-status ping-ok">ok</span>;
  if (isAuthError) return <span className="ping-status ping-err">not authorized</span>;
  return <span className="ping-status ping-err">error</span>;
}
