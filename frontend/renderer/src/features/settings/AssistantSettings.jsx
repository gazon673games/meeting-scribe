import React from "react";
import { Cpu, Plus, Radio, Server, Trash2 } from "lucide-react";

import {
  ASSISTANT_PROVIDER_OPTIONS,
  ASSISTANT_REASONING_OPTIONS,
  normalizeAssistantProfiles
} from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";
import { Field } from "../../shared/ui/Field";

export function AssistantSettings({ draft, onChange }) {
  const assistantEnabled = Boolean(draft.assistantEnabled);
  const profiles = normalizeAssistantProfiles(draft.assistantProfiles);
  const [pingStates, setPingStates] = React.useState({});

  const updateProfile = (index, patch) => {
    const next = profiles.map((profile, i) => (i === index ? { ...profile, ...patch } : profile));
    onChange({ assistantProfiles: next });
  };

  const addProfile = (provider = "codex") => {
    const id = uniqueProfileId(profiles, provider);
    onChange({
      assistantProfiles: [...profiles, newAssistantProfile(id, provider)],
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
        profileId: profile.id
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
          <Cpu size={15} />
          <span>Enable Assistant</span>
          <b />
        </button>

        <Field label="Selected Profile">
          <select
            disabled={!assistantEnabled || profiles.length === 0}
            value={draft.assistantSelectedProfileId || profiles[0]?.id || ""}
            onChange={(e) => onChange({ assistantSelectedProfileId: e.target.value })}
          >
            {profiles.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.label || profile.id}
              </option>
            ))}
          </select>
        </Field>

        <div className="assistant-profiles-list">
          {profiles.map((profile, index) => {
            const provider = normalizeProvider(profile.provider);
            const localProvider = provider !== "codex";
            const ping = pingStates[profile.id];
            return (
              <div className="assistant-profile-card" key={profile.id || index}>
                <div className="assistant-profile-head">
                  <span>{providerLabel(provider)}</span>
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
                  <Field label="Provider">
                    <select
                      disabled={!assistantEnabled}
                      value={provider}
                      onChange={(e) => updateProfile(index, providerPatch(profile, e.target.value))}
                    >
                      {ASSISTANT_PROVIDER_OPTIONS.map((option) => (
                        <option key={option.id} value={option.id}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Model">
                    <input
                      disabled={!assistantEnabled}
                      placeholder={modelPlaceholder(provider)}
                      spellCheck={false}
                      value={profile.model}
                      onChange={(e) => updateProfile(index, { model: e.target.value })}
                    />
                  </Field>
                  {provider === "codex" ? (
                    <>
                      <Field label="Reasoning">
                        <select disabled={!assistantEnabled} value={profile.reasoning_effort || ""} onChange={(e) => updateProfile(index, { reasoning_effort: e.target.value })}>
                          <option value="">default</option>
                          {ASSISTANT_REASONING_OPTIONS.map((effort) => (
                            <option key={effort} value={effort}>
                              {effort}
                            </option>
                          ))}
                        </select>
                      </Field>
                      <Field label="Codex Profile">
                        <input disabled={!assistantEnabled} spellCheck={false} value={profile.codex_profile || ""} onChange={(e) => updateProfile(index, { codex_profile: e.target.value })} />
                      </Field>
                    </>
                  ) : (
                    <>
                      <Field label="Base URL">
                        <input
                          disabled={!assistantEnabled}
                          spellCheck={false}
                          value={profile.base_url || ""}
                          placeholder={defaultBaseUrl(provider)}
                          onChange={(e) => updateProfile(index, { base_url: e.target.value })}
                        />
                      </Field>
                      {provider === "openai_local" ? (
                        <Field label="API Key">
                          <input
                            autoComplete="off"
                            disabled={!assistantEnabled}
                            spellCheck={false}
                            type="password"
                            value={profile.api_key || ""}
                            onChange={(e) => updateProfile(index, { api_key: e.target.value })}
                          />
                        </Field>
                      ) : null}
                      <Field label="Temperature">
                        <input
                          disabled={!assistantEnabled}
                          inputMode="decimal"
                          value={profile.temperature ?? ""}
                          onChange={(e) => updateProfile(index, { temperature: e.target.value })}
                        />
                      </Field>
                      <Field label="Max Tokens">
                        <input
                          disabled={!assistantEnabled}
                          inputMode="numeric"
                          pattern="[0-9]*"
                          value={profile.max_tokens || ""}
                          onChange={(e) => updateProfile(index, { max_tokens: e.target.value.replace(/\D/g, "") })}
                        />
                      </Field>
                    </>
                  )}
                  <Field label="Mode">
                    <span className={`profile-mode-badge ${localProvider ? "offline" : "online"}`}>
                      {localProvider ? "local" : "online"}
                    </span>
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

        <div className="assistant-add-row">
          <button className="model-download-button assistant-add-profile" disabled={!assistantEnabled} type="button" onClick={() => addProfile("codex")}>
            <Plus size={13} />
            Codex
          </button>
          <button className="model-download-button assistant-add-profile" disabled={!assistantEnabled} type="button" onClick={() => addProfile("ollama")}>
            <Server size={13} />
            Ollama
          </button>
          <button className="model-download-button assistant-add-profile" disabled={!assistantEnabled} type="button" onClick={() => addProfile("openai_local")}>
            <Server size={13} />
            Local API
          </button>
        </div>
      </div>
    </CollapsibleSection>
  );
}

function PingStatus({ ping }) {
  if (!ping) return null;
  if (ping.busy) return <span className="ping-status ping-busy">pinging...</span>;

  const isAuthError = /auth|unauthorized|not_logged|login/i.test(ping.errorCode || "");
  if (ping.ok) return <span className="ping-status ping-ok">ok</span>;
  if (isAuthError) return <span className="ping-status ping-err">not authorized</span>;
  return <span className="ping-status ping-err">error</span>;
}

function newAssistantProfile(id, provider) {
  const normalized = normalizeProvider(provider);
  return {
    id,
    label: providerLabel(normalized),
    provider: normalized,
    prompt: "Give concise, practical interview support based on session context.",
    model: normalized === "codex" ? "" : modelPlaceholder(normalized),
    reasoning_effort: normalized === "codex" ? "low" : "",
    codex_profile: "",
    base_url: defaultBaseUrl(normalized),
    api_key: "",
    temperature: "",
    max_tokens: 0,
    answer_prompt: "Command ANSWER: provide a short candidate response, key points, and one clarification question.",
    extra_args: [],
    offline: normalized !== "codex"
  };
}

function providerPatch(profile, provider) {
  const normalized = normalizeProvider(provider);
  return {
    provider: normalized,
    base_url: normalized === "codex" ? "" : profile.base_url || defaultBaseUrl(normalized),
    reasoning_effort: normalized === "codex" ? profile.reasoning_effort || "low" : "",
    codex_profile: normalized === "codex" ? profile.codex_profile || "" : "",
    offline: normalized !== "codex"
  };
}

function uniqueProfileId(profiles, provider) {
  const base = normalizeProvider(provider).replace(/[^a-z0-9_]+/g, "_") || "assistant";
  const ids = new Set(profiles.map((profile) => profile.id));
  let index = profiles.length + 1;
  let id = `${base}_${index}`;
  while (ids.has(id)) {
    index += 1;
    id = `${base}_${index}`;
  }
  return id;
}

function normalizeProvider(provider) {
  const value = String(provider || "codex").trim().toLowerCase();
  return ASSISTANT_PROVIDER_OPTIONS.some((option) => option.id === value) ? value : "codex";
}

function providerLabel(provider) {
  return ASSISTANT_PROVIDER_OPTIONS.find((option) => option.id === normalizeProvider(provider))?.label || "Codex CLI";
}

function defaultBaseUrl(provider) {
  return ASSISTANT_PROVIDER_OPTIONS.find((option) => option.id === normalizeProvider(provider))?.defaultBaseUrl || "";
}

function modelPlaceholder(provider) {
  if (provider === "ollama") return "llama3.2";
  if (provider === "openai_local") return "local-model";
  return "gpt-5.3-codex";
}
