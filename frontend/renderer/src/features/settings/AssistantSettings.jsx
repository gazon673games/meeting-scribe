import React from "react";
import { Cpu, Plus, Radio, Trash2 } from "lucide-react";

import {
  ASSISTANT_REASONING_OPTIONS,
  ASSISTANT_PROVIDER_OPTIONS,
  normalizeAssistantProfiles
} from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";
import { Field } from "../../shared/ui/Field";

export function AssistantSettings({ draft, onChange }) {
  const assistantEnabled = Boolean(draft.assistantEnabled);
  const profiles = normalizeAssistantProfiles(draft.assistantProfiles);
  const [pingStates, setPingStates] = React.useState({});
  const [llmModels, setLlmModels] = React.useState([]);
  const [llmLoaded, setLlmLoaded] = React.useState(false);
  const selectedId = draft.assistantSelectedProfileId || profiles[0]?.id || "";
  const selectedIndex = Math.max(0, profiles.findIndex((p) => p.id === selectedId));
  const selectedProfile = profiles[selectedIndex] || profiles[0];

  const hasLocalProfile = profiles.some((p) => (p.provider || "codex") !== "codex");

  React.useEffect(() => {
    if (!hasLocalProfile || llmLoaded) return;
    setLlmLoaded(true);
    meetingScribeClient.request("list_llm_models", {})
      .then((r) => setLlmModels(r?.models || []))
      .catch(() => {});
  }, [hasLocalProfile, llmLoaded]);

  const updateProfile = (index, patch) => {
    const next = profiles.map((p, i) => (i === index ? { ...p, ...patch } : p));
    onChange({ assistantProfiles: next });
  };

  const updateSelected = (patch) => {
    if (!selectedProfile) return;
    updateProfile(selectedIndex, patch);
  };

  const addProfile = (provider = "codex") => {
    const id = uniqueProfileId(profiles, provider);
    onChange({
      assistantProfiles: [...profiles, newProfile(id, provider)],
      assistantSelectedProfileId: id
    });
  };

  const removeProfile = (index) => {
    if (profiles.length <= 1) return;
    const removed = profiles[index];
    const next = profiles.filter((_, i) => i !== index);
    const fallback = next[Math.max(0, Math.min(index, next.length - 1))]?.id || "";
    onChange({
      assistantProfiles: next,
      assistantSelectedProfileId:
        draft.assistantSelectedProfileId === removed.id ? fallback : draft.assistantSelectedProfileId
    });
  };

  const pingProfile = React.useCallback(async (profile) => {
    setPingStates((s) => ({ ...s, [profile.id]: { busy: true } }));
    try {
      const result = await meetingScribeClient.request("ping_assistant_provider", {
        providerId: profile.provider || "codex",
        profileId: profile.id
      });
      setPingStates((s) => ({ ...s, [profile.id]: { busy: false, ...result } }));
    } catch (e) {
      setPingStates((s) => ({
        ...s,
        [profile.id]: { busy: false, ok: false, errorCode: "ping_failed", message: String(e?.message || e) }
      }));
    }
  }, []);

  const cachedLocalModels = llmModels.filter((m) => m.cached).map((m) => String(m.modelAlias || m.name || "")).filter(Boolean);

  return (
    <CollapsibleSection title="Assistant" defaultOpen={false}>
      <div className="assistant-settings-shell">
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

        <div className="assistant-profile-workbench">
          <div className="assistant-profile-sidebar">
            <div className="assistant-profile-sidebar-head">
              <span>Profiles</span>
              <span>{profiles.length}</span>
            </div>
            <div className="assistant-profile-list">
              {profiles.map((profile, index) => {
                const isCodex = normalizeProvider(profile.provider) === "codex";
                const selected = profile.id === selectedProfile?.id;
                return (
                  <div
                    className={`assistant-profile-list-item ${selected ? "selected" : ""}`}
                    key={profile.id || index}
                  >
                    <button
                      className="assistant-profile-list-select"
                      type="button"
                      onClick={() => onChange({ assistantSelectedProfileId: profile.id })}
                    >
                      <span className="assistant-profile-list-main">
                        <b>{profile.label || profile.id}</b>
                        <small>{isCodex ? "Codex CLI" : runtimeLabel(profile.provider)}</small>
                      </span>
                      <span className={`profile-mode-badge ${isCodex ? "online" : "offline"}`}>
                        {isCodex ? "online" : "local"}
                      </span>
                    </button>
                    <button
                      className="model-row-btn danger"
                      disabled={profiles.length <= 1}
                      title="Delete"
                      type="button"
                      onClick={() => removeProfile(index)}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                );
              })}
            </div>
            <div className="assistant-add-row">
              <button className="model-download-button" type="button" onClick={() => addProfile("codex")}>
                <Plus size={13} />
                Codex
              </button>
              <button className="model-download-button" type="button" onClick={() => addProfile("openai_local")}>
                <Plus size={13} />
                Local
              </button>
              <button className="model-download-button" type="button" onClick={() => addProfile("ollama")}>
                <Plus size={13} />
                Ollama
              </button>
            </div>
          </div>

          {selectedProfile ? (
            <ProfileDetails
              assistantEnabled={assistantEnabled}
              cachedLocalModels={cachedLocalModels}
              ping={pingStates[selectedProfile.id]}
              profile={selectedProfile}
              profilesCount={profiles.length}
              onDelete={() => removeProfile(selectedIndex)}
              onPing={() => pingProfile(selectedProfile)}
              onUpdate={updateSelected}
            />
          ) : null}
        </div>
      </div>
    </CollapsibleSection>
  );
}

function ProfileDetails({ assistantEnabled, cachedLocalModels, ping, profile, profilesCount, onDelete, onPing, onUpdate }) {
  const provider = normalizeProvider(profile.provider);
  const isCodex = provider === "codex";
  const [customModel, setCustomModel] = React.useState(false);

  const modelValue = profile.model || "";
  const showModelDropdown = !isCodex && cachedLocalModels.length > 0;
  const modelInList = showModelDropdown && cachedLocalModels.includes(modelValue);
  const useCustomInput = customModel || (!modelInList && modelValue !== "");

  return (
    <div className="assistant-profile-detail">
      <div className="assistant-profile-detail-head">
        <div>
          <span>{isCodex ? "Codex CLI" : runtimeLabel(provider)}</span>
          <h4>{profile.label || profile.id}</h4>
        </div>
        <div className="assistant-profile-actions">
          {isCodex ? (
            <button
              className="model-download-button"
              disabled={!assistantEnabled || ping?.busy}
              type="button"
              onClick={onPing}
            >
              <Radio size={13} />
              {ping?.busy ? "Pinging…" : "Ping"}
            </button>
          ) : null}
          <button
            className="model-row-btn danger"
            disabled={profilesCount <= 1}
            title="Delete profile"
            type="button"
            onClick={onDelete}
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {isCodex && ping && !ping.busy ? (
        <div className="assistant-ping-result-row">
          <PingStatus ping={ping} />
        </div>
      ) : null}

      <div className="settings-grid assistant-detail-grid">
        <Field label="Label">
          <input value={profile.label} onChange={(e) => onUpdate({ label: e.target.value })} />
        </Field>

        {isCodex ? (
          <>
            <Field label="Model">
              <input
                placeholder="gpt-5.3-codex"
                spellCheck={false}
                value={modelValue}
                onChange={(e) => onUpdate({ model: e.target.value })}
              />
            </Field>
            <Field label="Reasoning">
              <select
                value={profile.reasoning_effort || ""}
                onChange={(e) => onUpdate({ reasoning_effort: e.target.value })}
              >
                <option value="">default</option>
                {ASSISTANT_REASONING_OPTIONS.map((r) => (
                  <option key={r} value={r}>{r}</option>
                ))}
              </select>
            </Field>
            <Field label="Codex Profile">
              <input
                spellCheck={false}
                value={profile.codex_profile || ""}
                onChange={(e) => onUpdate({ codex_profile: e.target.value })}
              />
            </Field>
          </>
        ) : (
          <>
            <Field label="Model">
              {showModelDropdown && !useCustomInput ? (
                <select
                  value={modelInList ? modelValue : ""}
                  onChange={(e) => {
                    if (e.target.value === "__custom__") {
                      setCustomModel(true);
                    } else {
                      onUpdate({ model: e.target.value });
                    }
                  }}
                >
                  <option value="">— select —</option>
                  {cachedLocalModels.map((alias) => (
                    <option key={alias} value={alias}>{alias}</option>
                  ))}
                  <option value="__custom__">Custom…</option>
                </select>
              ) : (
                <div className="model-custom-input-row">
                  <input
                    spellCheck={false}
                    value={modelValue}
                    onChange={(e) => onUpdate({ model: e.target.value })}
                  />
                  {showModelDropdown ? (
                    <button
                      className="model-row-btn"
                      title="Pick from list"
                      type="button"
                      onClick={() => { setCustomModel(false); onUpdate({ model: "" }); }}
                    >
                      ↩
                    </button>
                  ) : null}
                </div>
              )}
            </Field>
            <Field label="Runtime">
              <select
                value={provider}
                onChange={(e) => onUpdate(runtimePatch(profile, e.target.value))}
              >
                <option value="openai_local">llama.cpp / LM Studio</option>
                <option value="ollama">Ollama</option>
              </select>
            </Field>
            <Field label="Temperature">
              <input
                inputMode="decimal"
                placeholder="0.7"
                value={profile.temperature ?? ""}
                onChange={(e) => onUpdate({ temperature: e.target.value })}
              />
            </Field>
            <Field label="Max Tokens">
              <input
                inputMode="numeric"
                pattern="[0-9]*"
                placeholder="2048"
                value={profile.max_tokens || ""}
                onChange={(e) => onUpdate({ max_tokens: e.target.value.replace(/\D/g, "") })}
              />
            </Field>
            <Field label="Base URL">
              <input
                spellCheck={false}
                placeholder={defaultBaseUrl(provider)}
                value={profile.base_url || ""}
                onChange={(e) => onUpdate({ base_url: e.target.value })}
              />
            </Field>
          </>
        )}
      </div>

      <Field label="Instructions">
        <textarea
          className="assistant-profile-prompt"
          rows={5}
          spellCheck={false}
          value={profile.prompt || ""}
          onChange={(e) => onUpdate({ prompt: e.target.value })}
        />
      </Field>
    </div>
  );
}

function PingStatus({ ping }) {
  if (!ping || ping.busy) return null;
  const isAuthError = /auth|unauthorized|not_logged|login/i.test(ping.errorCode || "");
  if (ping.ok) return <span className="ping-status ping-ok">ok</span>;
  if (isAuthError) return <span className="ping-status ping-err">not authorized</span>;
  return <span className="ping-status ping-err">{ping.message || "error"}</span>;
}

function newProfile(id, provider) {
  const p = normalizeProvider(provider);
  return {
    id,
    label: p === "codex" ? "Codex" : p === "ollama" ? "Ollama" : "Local Model",
    provider: p,
    prompt: "Give concise, practical support based on the session context.",
    model: "",
    reasoning_effort: p === "codex" ? "low" : "",
    codex_profile: "",
    base_url: defaultBaseUrl(p),
    api_key: "",
    temperature: "",
    max_tokens: 0,
    answer_prompt: "",
    extra_args: [],
    offline: p !== "codex"
  };
}

function runtimePatch(profile, provider) {
  const p = normalizeProvider(provider);
  const prev = normalizeProvider(profile.provider);
  const currentUrl = String(profile.base_url || "");
  const baseUrl = currentUrl && currentUrl !== defaultBaseUrl(prev) ? currentUrl : defaultBaseUrl(p);
  return { provider: p, base_url: baseUrl, offline: p !== "codex" };
}

function uniqueProfileId(profiles, provider) {
  const base = normalizeProvider(provider).replace(/[^a-z0-9_]+/g, "_") || "assistant";
  const ids = new Set(profiles.map((p) => p.id));
  let i = profiles.length + 1;
  let id = `${base}_${i}`;
  while (ids.has(id)) { i += 1; id = `${base}_${i}`; }
  return id;
}

function normalizeProvider(provider) {
  const v = String(provider || "codex").trim().toLowerCase();
  return ASSISTANT_PROVIDER_OPTIONS.some((o) => o.id === v) ? v : "codex";
}

function runtimeLabel(provider) {
  if (provider === "ollama") return "Ollama";
  if (provider === "openai_local") return "Local Model";
  return "Local Model";
}

function defaultBaseUrl(provider) {
  return ASSISTANT_PROVIDER_OPTIONS.find((o) => o.id === provider)?.defaultBaseUrl || "";
}
