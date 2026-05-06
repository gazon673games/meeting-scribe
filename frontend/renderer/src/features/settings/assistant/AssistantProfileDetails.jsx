import React from "react";
import { Radio, Trash2 } from "lucide-react";

import { ASSISTANT_REASONING_OPTIONS } from "../../../entities/settings/model";
import { AssistantPingStatus } from "../../assistant/AssistantPingStatus";
import { Field } from "../../../shared/ui/Field";
import { defaultBaseUrl, normalizeProvider, runtimeLabel, runtimePatch } from "./profileUtils";

const CUSTOM_MODEL_VALUE = "__custom__";

export function AssistantProfileDetails({
  assistantEnabled,
  cachedGgufModels,
  cachedLocalModels,
  ping,
  profile,
  onDelete,
  onPing,
  onUpdate
}) {
  const provider = normalizeProvider(profile.provider);
  const isCodex = provider === "codex";
  const [customModel, setCustomModel] = React.useState(false);

  const modelState = buildModelState({
    provider,
    modelValue: profile.model || "",
    customModel,
    cachedLocalModels,
    cachedGgufModels
  });

  return (
    <div className="assistant-profile-detail">
      <AssistantProfileHead
        assistantEnabled={assistantEnabled}
        isCodex={isCodex}
        ping={ping}
        profile={profile}
        provider={provider}
        onDelete={onDelete}
        onPing={onPing}
      />

      <PingResult isCodex={isCodex} ping={ping} />

      <div className="settings-grid assistant-detail-grid">
        <Field label="Label">
          <input value={profile.label} onChange={(event) => onUpdate({ label: event.target.value })} />
        </Field>

        {isCodex ? (
          <CodexFields profile={profile} modelValue={modelState.modelValue} onUpdate={onUpdate} />
        ) : (
          <LocalProviderFields
            profile={profile}
            provider={provider}
            modelState={modelState}
            cachedGgufModels={cachedGgufModels}
            cachedLocalModels={cachedLocalModels}
            onUpdate={onUpdate}
            onToggleCustomModel={setCustomModel}
          />
        )}
      </div>

      <Field label="Instructions">
        <textarea
          className="assistant-profile-prompt"
          rows={5}
          spellCheck={false}
          value={profile.prompt || ""}
          onChange={(event) => onUpdate({ prompt: event.target.value })}
        />
      </Field>
    </div>
  );
}

function buildModelState({ provider, modelValue, customModel, cachedLocalModels, cachedGgufModels }) {
  const isLocalGguf = provider === "local";
  const showOllamaDropdown = provider === "ollama" && cachedLocalModels.length > 0;
  const showGgufDropdown = isLocalGguf && cachedGgufModels.length > 0;
  const showModelDropdown = showOllamaDropdown;
  const modelInList = showModelDropdown && cachedLocalModels.includes(modelValue);
  const ggufInList = showGgufDropdown && cachedGgufModels.some((model) => model.path === modelValue);
  const useCustomInput = customModel || (!modelInList && !ggufInList && modelValue !== "");

  return {
    isLocalGguf,
    modelValue,
    modelInList,
    ggufInList,
    showGgufDropdown,
    showModelDropdown,
    useCustomInput
  };
}

function PingResult({ isCodex, ping }) {
  if (!isCodex || !ping || ping.busy) return null;
  return (
    <div className="assistant-ping-result-row">
      <AssistantPingStatus ping={ping} showBusy={false} showMessage />
    </div>
  );
}

function AssistantProfileHead({ assistantEnabled, isCodex, ping, profile, provider, onDelete, onPing }) {
  return (
    <div className="assistant-profile-detail-head">
      <div>
        <span>{isCodex ? "Codex CLI" : runtimeLabel(provider)}</span>
        <h4>{profile.label || profile.id}</h4>
      </div>
      <div className="assistant-profile-actions">
        {isCodex ? (
          <button className="model-download-button" disabled={!assistantEnabled || ping?.busy} type="button" onClick={onPing}>
            <Radio size={13} />
            {ping?.busy ? "Pinging..." : "Ping"}
          </button>
        ) : null}
        <button className="model-row-btn danger" title="Delete profile" type="button" onClick={onDelete}>
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}

function CodexFields({ profile, modelValue, onUpdate }) {
  return (
    <>
      <Field label="Model">
        <input
          placeholder="gpt-5.3-codex"
          spellCheck={false}
          value={modelValue}
          onChange={(event) => onUpdate({ model: event.target.value })}
        />
      </Field>
      <Field label="Reasoning">
        <select value={profile.reasoning_effort || ""} onChange={(event) => onUpdate({ reasoning_effort: event.target.value })}>
          <option value="">default</option>
          {ASSISTANT_REASONING_OPTIONS.map((reasoning) => (
            <option key={reasoning} value={reasoning}>{reasoning}</option>
          ))}
        </select>
      </Field>
      <Field label="Codex Profile">
        <input
          spellCheck={false}
          value={profile.codex_profile || ""}
          onChange={(event) => onUpdate({ codex_profile: event.target.value })}
        />
      </Field>
    </>
  );
}

function LocalProviderFields({
  profile,
  provider,
  modelState,
  cachedGgufModels,
  cachedLocalModels,
  onUpdate,
  onToggleCustomModel
}) {
  return (
    <>
      <Field label="Model">
        <LocalModelField
          modelState={modelState}
          cachedGgufModels={cachedGgufModels}
          cachedLocalModels={cachedLocalModels}
          onUpdate={onUpdate}
          onToggleCustomModel={onToggleCustomModel}
        />
      </Field>

      <Field label="Runtime">
        <select value={provider} onChange={(event) => onUpdate(runtimePatch(profile, event.target.value))}>
          <option value="local">Local GGUF (llama.cpp)</option>
          <option value="openai_local">OpenAI-compatible server</option>
          <option value="ollama">Ollama</option>
        </select>
      </Field>

      {modelState.isLocalGguf ? (
        <LocalGgufRuntimeFields profile={profile} onUpdate={onUpdate} />
      ) : (
        <RemoteRuntimeFields provider={provider} profile={profile} onUpdate={onUpdate} />
      )}
    </>
  );
}

function LocalModelField({ modelState, cachedGgufModels, cachedLocalModels, onUpdate, onToggleCustomModel }) {
  const mode = resolveModelInputMode(modelState);
  if (mode === "gguf") {
    return (
      <GgufModelSelect
        cachedGgufModels={cachedGgufModels}
        value={modelState.ggufInList ? modelState.modelValue : ""}
        onUpdate={onUpdate}
        onToggleCustomModel={onToggleCustomModel}
      />
    );
  }

  if (mode === "alias") {
    return (
      <AliasModelSelect
        cachedLocalModels={cachedLocalModels}
        value={modelState.modelInList ? modelState.modelValue : ""}
        onUpdate={onUpdate}
        onToggleCustomModel={onToggleCustomModel}
      />
    );
  }

  return (
    <CustomModelInput
      isLocalGguf={modelState.isLocalGguf}
      modelValue={modelState.modelValue}
      canPickFromList={modelState.showGgufDropdown || modelState.showModelDropdown}
      onUpdate={onUpdate}
      onToggleCustomModel={onToggleCustomModel}
    />
  );
}

function resolveModelInputMode(modelState) {
  if (modelState.showGgufDropdown && !modelState.useCustomInput) return "gguf";
  if (modelState.showModelDropdown && !modelState.useCustomInput) return "alias";
  return "custom";
}

function GgufModelSelect({ cachedGgufModels, value, onUpdate, onToggleCustomModel }) {
  return (
    <select
      value={value}
      onChange={(event) => applySelectModel(event.target.value, onUpdate, onToggleCustomModel)}
    >
      <option value="">-- select --</option>
      {cachedGgufModels.map((model) => (
        <option key={model.path} value={model.path}>{model.label}</option>
      ))}
      <option value={CUSTOM_MODEL_VALUE}>Custom path...</option>
    </select>
  );
}

function AliasModelSelect({ cachedLocalModels, value, onUpdate, onToggleCustomModel }) {
  return (
    <select
      value={value}
      onChange={(event) => applySelectModel(event.target.value, onUpdate, onToggleCustomModel)}
    >
      <option value="">-- select --</option>
      {cachedLocalModels.map((alias) => (
        <option key={alias} value={alias}>{alias}</option>
      ))}
      <option value={CUSTOM_MODEL_VALUE}>Custom...</option>
    </select>
  );
}

function applySelectModel(value, onUpdate, onToggleCustomModel) {
  if (value === CUSTOM_MODEL_VALUE) {
    onToggleCustomModel(true);
    return;
  }
  onUpdate({ model: value });
}

function CustomModelInput({ isLocalGguf, modelValue, canPickFromList, onUpdate, onToggleCustomModel }) {
  return (
    <div className="model-custom-input-row">
      <input
        spellCheck={false}
        placeholder={isLocalGguf ? "path/to/model.gguf" : ""}
        value={modelValue}
        onChange={(event) => onUpdate({ model: event.target.value })}
      />
      {canPickFromList ? (
        <button
          className="model-row-btn"
          title="Pick from list"
          type="button"
          onClick={() => {
            onToggleCustomModel(false);
            onUpdate({ model: "" });
          }}
        >
          {"<-"}
        </button>
      ) : null}
    </div>
  );
}

function LocalGgufRuntimeFields({ profile, onUpdate }) {
  return (
    <>
      <Field label="GPU Layers">
        <input
          inputMode="numeric"
          pattern="[0-9]*"
          placeholder="0 = CPU, 28+ = GPU"
          value={profile.gpu_layers ?? ""}
          onChange={(event) => onUpdate({ gpu_layers: event.target.value.replace(/\D/g, "") })}
        />
      </Field>
      <Field label="Context Size">
        <input
          inputMode="numeric"
          pattern="[0-9]*"
          placeholder="4096"
          value={profile.context_size || ""}
          onChange={(event) => onUpdate({ context_size: event.target.value.replace(/\D/g, "") })}
        />
      </Field>
    </>
  );
}

function RemoteRuntimeFields({ provider, profile, onUpdate }) {
  return (
    <>
      <Field label="Temperature">
        <input
          inputMode="decimal"
          placeholder="0.7"
          value={profile.temperature ?? ""}
          onChange={(event) => onUpdate({ temperature: event.target.value })}
        />
      </Field>
      <Field label="Max Tokens">
        <input
          inputMode="numeric"
          pattern="[0-9]*"
          placeholder="2048"
          value={profile.max_tokens || ""}
          onChange={(event) => onUpdate({ max_tokens: event.target.value.replace(/\D/g, "") })}
        />
      </Field>
      <Field label="Base URL">
        <input
          spellCheck={false}
          placeholder={defaultBaseUrl(provider)}
          value={profile.base_url || ""}
          onChange={(event) => onUpdate({ base_url: event.target.value })}
        />
      </Field>
    </>
  );
}
