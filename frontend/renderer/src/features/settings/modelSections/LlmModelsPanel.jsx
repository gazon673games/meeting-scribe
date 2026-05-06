import React from "react";
import { Download } from "lucide-react";

import { normalizeAssistantProfiles } from "../../../entities/settings/model";
import { meetingScribeClient } from "../../../shared/api/meetingScribeClient";
import { Field } from "../../../shared/ui/Field";
import { InnerCollapsible } from "../../../shared/ui/InnerCollapsible";
import { LlmModelRow } from "../LlmModelRow";
import { formatError, runWithPending, useModelList } from "./useModelList";

function isLocalAssistantProfile(profile) {
  return profile && profile.provider === "openai_local";
}

function createLocalProfile(id, alias) {
  return {
    id,
    label: "Local OpenAI API",
    provider: "openai_local",
    prompt: "Give concise, practical interview support based on session context.",
    model: alias,
    reasoning_effort: "",
    codex_profile: "",
    base_url: "http://127.0.0.1:1234/v1",
    api_key: "",
    temperature: "",
    max_tokens: 0,
    answer_prompt: "Command ANSWER: provide a short candidate response, key points, and one clarification question.",
    extra_args: [],
    offline: true
  };
}

export function LlmModelsPanel({ draft, onChange, open, llmDir, useProxy, downloadProxy, refreshToken }) {
  const [downloading, setDownloading] = React.useState(new Set());
  const [customUrl, setCustomUrl] = React.useState("");

  const llmList = useModelList({
    open,
    refreshToken,
    loadFn: React.useCallback(() => meetingScribeClient.request("list_llm_models", { modelsDir: llmDir }), [llmDir])
  });

  const linkedAliases = React.useMemo(() => {
    const profiles = normalizeAssistantProfiles(draft.assistantProfiles);
    return new Set(
      profiles
        .filter(isLocalAssistantProfile)
        .map((profile) => String(profile.model || ""))
    );
  }, [draft.assistantProfiles]);

  const handleUse = React.useCallback((model) => {
    const alias = String(model.modelAlias || model.name || "").trim();
    if (!alias) return;

    const profiles = normalizeAssistantProfiles(draft.assistantProfiles);
    const selected = profiles.find((profile) => profile.id === draft.assistantSelectedProfileId);
    if (isLocalAssistantProfile(selected)) {
      onChange({
        assistantSelectedProfileId: selected.id,
        assistantProfiles: profiles.map((profile) => (profile.id === selected.id ? { ...profile, model: alias } : profile))
      });
      return;
    }

    const id = `local_${Date.now().toString(36)}`;
    onChange({
      assistantSelectedProfileId: id,
      assistantProfiles: [...profiles, createLocalProfile(id, alias)]
    });
  }, [draft.assistantProfiles, draft.assistantSelectedProfileId, onChange]);

  const handleDownload = React.useCallback(async (name) => {
    llmList.setError("");
    try {
      await runWithPending(setDownloading, name, async () => {
        await meetingScribeClient.request("download_llm_model", {
          name,
          modelsDir: llmDir,
          useProxy,
          proxy: downloadProxy
        });
        llmList.load();
      });
    } catch (requestError) {
      llmList.setError(formatError(requestError));
    }
  }, [downloadProxy, llmDir, llmList, useProxy]);

  const handleDelete = React.useCallback(async (model) => {
    llmList.setError("");
    try {
      await meetingScribeClient.request("delete_llm_model", { path: model.path, modelsDir: llmDir });
      llmList.load();
    } catch (requestError) {
      llmList.setError(formatError(requestError));
    }
  }, [llmDir, llmList]);

  return (
    <InnerCollapsible title="Language Models">
      <div className="inner-block-controls">
        <Field label="Subfolder">
          <input
            spellCheck={false}
            value={draft.llmModelsSubdir || ""}
            placeholder="Default: llm"
            onChange={(event) => onChange({ llmModelsSubdir: event.target.value })}
          />
        </Field>
        <div className="custom-model-row">
          <Field label="Hugging Face Repo/File or URL">
            <input
              spellCheck={false}
              value={customUrl}
              placeholder="https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
              onChange={(event) => setCustomUrl(event.target.value)}
            />
          </Field>
          <button
            className="model-download-button"
            disabled={!customUrl.trim()}
            type="button"
            onClick={() => {
              handleDownload(customUrl.trim());
              setCustomUrl("");
            }}
          >
            <Download size={13} />
            Download
          </button>
        </div>
      </div>

      {llmList.error ? <div className="models-error">{llmList.error}</div> : null}
      {llmList.models === null ? (
        <div className="models-loading">Loading...</div>
      ) : llmList.models.length === 0 ? (
        <div className="inner-collapsible-empty">No GGUF language models found</div>
      ) : (
        <div className="models-list">
          {llmList.models.map((model) => {
            const alias = String(model.modelAlias || model.name || "");
            return (
              <LlmModelRow
                key={model.path || model.name}
                model={model}
                alias={alias}
                linked={linkedAliases.has(alias)}
                isDownloading={model.downloading || downloading.has(model.name)}
                onUse={handleUse}
                onDelete={handleDelete}
              />
            );
          })}
        </div>
      )}
    </InnerCollapsible>
  );
}
