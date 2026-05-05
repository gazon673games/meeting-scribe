import React from "react";
import { Download, Network, RefreshCw, Save, Settings, X } from "lucide-react";

import { FALLBACK_OPTIONS, buildProxyUrl, normalizeAssistantProfiles, uniqueOptions } from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";
import { Field } from "../../shared/ui/Field";
import { InnerCollapsible } from "../../shared/ui/InnerCollapsible";
import { AdvancedAsrSettings } from "./AdvancedAsrSettings";
import { AppearanceSettings } from "./AppearanceSettings";
import { AsrModelRow } from "./AsrModelRow";
import { AssistantSettings } from "./AssistantSettings";
import { DiarModelRow } from "./DiarModelRow";
import { DiarizationSettings } from "./DiarizationSettings";
import { HardwareSummary } from "./HardwareSummary";
import { LlmModelRow } from "./LlmModelRow";
import { ProxySettings } from "./ProxySettings";

// ─── helpers ────────────────────────────────────────────────────────────────

function joinDir(base, sub) {
  const b = String(base || "").trim();
  const s = String(sub || "").trim();
  if (!b && !s) return "";
  if (!b) return s;
  if (!s) return b;
  return `${b}/${s}`;
}

function isLocalAssistantProfile(profile) {
  return profile && profile.provider === "openai_local";
}

// ─── combined models section ─────────────────────────────────────────────────

function ModelsSection({ draft, onChange, open }) {
  const baseDir = String(draft.modelsDirectory || "").trim();

  // ASR state
  const [asrModels, setAsrModels] = React.useState(null);
  const [asrError, setAsrError] = React.useState("");
  const [asrDownloading, setAsrDownloading] = React.useState(new Set());
  const [asrCustomUrl, setAsrCustomUrl] = React.useState("");
  const [asrMetadata, setAsrMetadata] = React.useState({});
  const asrDir = joinDir(baseDir, draft.asrModelsSubdir);

  // Diar state
  const [diarModels, setDiarModels] = React.useState(null);
  const [diarError, setDiarError] = React.useState("");
  const [diarDownloading, setDiarDownloading] = React.useState(new Set());
  const [diarCustomUrl, setDiarCustomUrl] = React.useState("");
  const [diarExpanded, setDiarExpanded] = React.useState(new Set());
  const diarDir = joinDir(baseDir, draft.diarModelsSubdir);

  // LLM
  const [llmModels, setLlmModels] = React.useState(null);
  const [llmError, setLlmError] = React.useState("");
  const [llmDownloading, setLlmDownloading] = React.useState(new Set());
  const [llmCustomUrl, setLlmCustomUrl] = React.useState("");
  const llmDir = joinDir(baseDir, draft.llmModelsSubdir);

  // ── loaders ──────────────────────────────────────────────────────────────
  const loadAsrModels = React.useCallback(() => {
    meetingScribeClient
      .request("list_models", { modelsDir: asrDir })
      .then((r) => setAsrModels(r?.models || []))
      .catch((e) => setAsrError(String(e?.message || e)));
  }, [asrDir]);

  const loadDiarModels = React.useCallback(() => {
    meetingScribeClient
      .request("list_diarization_models", { modelsDir: diarDir })
      .then((r) => setDiarModels(r?.models || []))
      .catch((e) => setDiarError(String(e?.message || e)));
  }, [diarDir]);

  const loadLlmModels = React.useCallback(() => {
    meetingScribeClient
      .request("list_llm_models", { modelsDir: llmDir })
      .then((r) => setLlmModels(r?.models || []))
      .catch((e) => setLlmError(String(e?.message || e)));
  }, [llmDir]);

  React.useEffect(() => {
    if (!open) return;
    loadAsrModels();
    loadDiarModels();
    loadLlmModels();
  }, [open, loadAsrModels, loadDiarModels, loadLlmModels]);

  // polling while downloading
  React.useEffect(() => {
    if (!asrModels?.some((m) => m.downloading)) return undefined;
    const h = window.setInterval(loadAsrModels, 1000);
    return () => window.clearInterval(h);
  }, [asrModels, loadAsrModels]);

  React.useEffect(() => {
    if (!diarModels?.some((m) => m.downloading)) return undefined;
    const h = window.setInterval(loadDiarModels, 1000);
    return () => window.clearInterval(h);
  }, [diarModels, loadDiarModels]);

  React.useEffect(() => {
    if (!llmModels?.some((m) => m.downloading)) return undefined;
    const h = window.setInterval(loadLlmModels, 1000);
    return () => window.clearInterval(h);
  }, [llmModels, loadLlmModels]);

  // auto-select first ready diar model
  React.useEffect(() => {
    if (!open || !draft.diarizationEnabled || draft.diarSherpaEmbeddingModelPath || draft.diarizationBackend !== "online") return;
    const ready = diarModels?.find((m) => m.cached && m.compatible && (m.backend || "sherpa_onnx") === "sherpa_onnx");
    if (!ready?.path) return;
    onChange({
      diarizationBackend: ready.backend || "sherpa_onnx",
      diarizationSidecarEnabled: true,
      diarSherpaEmbeddingModelPath: ready.path,
      diarSherpaProvider: ready.provider || "cpu",
    });
  }, [draft.diarizationBackend, draft.diarizationEnabled, draft.diarSherpaEmbeddingModelPath, diarModels, onChange, open]);

  // ── ASR handlers ─────────────────────────────────────────────────────────
  const handleAsrDownload = React.useCallback(async (name) => {
    setAsrDownloading((p) => new Set([...p, name]));
    setAsrError("");
    try {
      await meetingScribeClient.request("download_model", {
        name,
        modelsDir: asrDir,
        useProxy: Boolean(draft.modelsUseProxy),
        proxy: buildProxyUrl(draft, { forceEnabled: Boolean(draft.modelsUseProxy) }),
      });
      loadAsrModels();
    } catch (e) {
      setAsrError(String(e?.message || e));
    } finally {
      setAsrDownloading((p) => { const n = new Set(p); n.delete(name); return n; });
    }
  }, [asrDir, draft, loadAsrModels]);

  const handleAsrDelete = React.useCallback(async (name) => {
    setAsrError("");
    try {
      await meetingScribeClient.request("delete_model", { name, modelsDir: asrDir });
      loadAsrModels();
    } catch (e) {
      setAsrError(String(e?.message || e));
    }
  }, [asrDir, loadAsrModels]);

  const handleAsrRemoveEntry = React.useCallback(async (name) => {
    setAsrError("");
    try {
      await meetingScribeClient.request("remove_model_entry", { name, modelsDir: asrDir });
      loadAsrModels();
    } catch (e) {
      setAsrError(String(e?.message || e));
    }
  }, [asrDir, loadAsrModels]);

  const handleAsrToggleMeta = React.useCallback(async (model) => {
    const key = model.name;
    if (asrMetadata[key]) {
      setAsrMetadata((c) => { const n = { ...c }; delete n[key]; return n; });
      return;
    }
    setAsrError("");
    setAsrMetadata((c) => ({ ...c, [key]: { loading: true, metadata: null, error: "" } }));
    try {
      const metadata = await meetingScribeClient.request("model_metadata", { name: model.name, modelsDir: asrDir });
      setAsrMetadata((c) => c[key] ? { ...c, [key]: { loading: false, metadata, error: "" } } : c);
    } catch (e) {
      setAsrMetadata((c) => c[key] ? { ...c, [key]: { loading: false, metadata: null, error: String(e?.message || e) } } : c);
    }
  }, [asrDir, asrMetadata]);

  // ── Diar handlers ─────────────────────────────────────────────────────────
  const handleDiarUse = React.useCallback((model) => {
    onChange({
      diarizationEnabled: true,
      diarizationBackend: model.backend || "sherpa_onnx",
      diarizationSidecarEnabled: true,
      diarSherpaEmbeddingModelPath: model.path,
      diarSherpaProvider: model.provider || "cpu",
    });
  }, [onChange]);

  const handleDiarDownload = React.useCallback(async (model) => {
    setDiarDownloading((p) => new Set([...p, model.name]));
    setDiarError("");
    try {
      await meetingScribeClient.request("download_diarization_model", {
        name: model.name,
        modelsDir: diarDir,
        useProxy: Boolean(draft.modelsUseProxy),
        proxy: buildProxyUrl(draft, { forceEnabled: Boolean(draft.modelsUseProxy) }),
      });
      loadDiarModels();
    } catch (e) {
      setDiarError(String(e?.message || e));
    } finally {
      setDiarDownloading((p) => { const n = new Set(p); n.delete(model.name); return n; });
    }
  }, [diarDir, draft, loadDiarModels]);

  const handleDiarDelete = React.useCallback(async (model) => {
    setDiarError("");
    try {
      await meetingScribeClient.request("delete_diarization_model", { path: model.path, modelsDir: diarDir });
      loadDiarModels();
    } catch (e) {
      setDiarError(String(e?.message || e));
    }
  }, [diarDir, loadDiarModels]);

  const handleDiarToggleExpand = React.useCallback((name) => {
    setDiarExpanded((p) => {
      const n = new Set(p);
      n.has(name) ? n.delete(name) : n.add(name);
      return n;
    });
  }, []);

  const handleLlmUse = React.useCallback((model) => {
    const alias = String(model.modelAlias || model.name || "").trim();
    if (!alias) return;
    const profiles = normalizeAssistantProfiles(draft.assistantProfiles);
    const selected = profiles.find((profile) => profile.id === draft.assistantSelectedProfileId);
    const target = isLocalAssistantProfile(selected) ? selected : null;
    if (target) {
      onChange({
        assistantSelectedProfileId: target.id,
        assistantProfiles: profiles.map((profile) => profile.id === target.id ? { ...profile, model: alias } : profile)
      });
      return;
    }
    const id = `local_${Date.now().toString(36)}`;
    onChange({
      assistantSelectedProfileId: id,
      assistantProfiles: [
        ...profiles,
        {
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
        }
      ]
    });
  }, [draft.assistantProfiles, draft.assistantSelectedProfileId, onChange]);

  const handleLlmDownload = React.useCallback(async (name) => {
    setLlmDownloading((p) => new Set([...p, name]));
    setLlmError("");
    try {
      await meetingScribeClient.request("download_llm_model", {
        name,
        modelsDir: llmDir,
        useProxy: Boolean(draft.modelsUseProxy),
        proxy: buildProxyUrl(draft, { forceEnabled: Boolean(draft.modelsUseProxy) }),
      });
      loadLlmModels();
    } catch (e) {
      setLlmError(String(e?.message || e));
    } finally {
      setLlmDownloading((p) => { const n = new Set(p); n.delete(name); return n; });
    }
  }, [draft, llmDir, loadLlmModels]);

  const handleLlmDelete = React.useCallback(async (model) => {
    setLlmError("");
    try {
      await meetingScribeClient.request("delete_llm_model", { path: model.path, modelsDir: llmDir });
      loadLlmModels();
    } catch (e) {
      setLlmError(String(e?.message || e));
    }
  }, [llmDir, loadLlmModels]);

  if (!open) return null;

  const assistantProfiles = normalizeAssistantProfiles(draft.assistantProfiles);
  const linkedLlmAliases = new Set(assistantProfiles.filter(isLocalAssistantProfile).map((profile) => String(profile.model || "")));

  return (
    <CollapsibleSection
      title="Models"
      defaultOpen={false}
      action={
        <button className="icon-button" title="Refresh all models" type="button" onClick={() => { loadAsrModels(); loadDiarModels(); loadLlmModels(); }}>
          <RefreshCw size={13} />
        </button>
      }
    >
      {/* shared controls */}
      <div className="models-controls">
        <Field label="Models Folder">
          <input
            spellCheck={false}
            value={draft.modelsDirectory || ""}
            placeholder="Default: ./models"
            onChange={(e) => onChange({ modelsDirectory: e.target.value })}
          />
        </Field>
        <button
          aria-pressed={Boolean(draft.modelsUseProxy)}
          className={`feature-toggle model-proxy-toggle ${draft.modelsUseProxy ? "selected" : ""}`}
          type="button"
          onClick={() => onChange({ modelsUseProxy: !draft.modelsUseProxy })}
        >
          <Network size={15} />
          <span>Use Proxy for Downloads</span>
          <b />
        </button>
      </div>

      {/* ── Transcription (ASR) ── */}
      <InnerCollapsible title="Transcription">
        <div className="inner-block-controls">
          <Field label="Subfolder">
            <input
              spellCheck={false}
              value={draft.asrModelsSubdir || ""}
              placeholder="e.g. asr"
              onChange={(e) => onChange({ asrModelsSubdir: e.target.value })}
            />
          </Field>
          <div className="custom-model-row">
            <Field label="Hugging Face Repo or URL">
              <input
                spellCheck={false}
                value={asrCustomUrl}
                placeholder="org/model or https://…"
                onChange={(e) => setAsrCustomUrl(e.target.value)}
              />
            </Field>
            <button
              className="model-download-button"
              disabled={!asrCustomUrl.trim()}
              type="button"
              onClick={() => { handleAsrDownload(asrCustomUrl.trim()); setAsrCustomUrl(""); }}
            >
              <Download size={13} />
              Download
            </button>
          </div>
        </div>
        {asrError ? <div className="models-error">{asrError}</div> : null}
        {asrModels === null ? (
          <div className="models-loading">Loading…</div>
        ) : (
          <div className="models-list">
            {asrModels.map((model) => (
              <AsrModelRow
                key={model.name}
                model={model}
                isDownloading={model.downloading || asrDownloading.has(model.name)}
                isSelected={draft.model === model.name}
                isExternal={!model.builtin && model.source !== "recommended"}
                metaEntry={asrMetadata[model.name] || null}
                onDownload={handleAsrDownload}
                onDelete={handleAsrDelete}
                onRemoveEntry={handleAsrRemoveEntry}
                onToggleMeta={handleAsrToggleMeta}
                onChange={onChange}
              />
            ))}
          </div>
        )}
      </InnerCollapsible>

      {/* ── Speaker ID (Diar) ── */}
      <InnerCollapsible title="Speaker ID">
        <div className="inner-block-controls">
          <Field label="Subfolder">
            <input
              spellCheck={false}
              value={draft.diarModelsSubdir || ""}
              placeholder="e.g. diar"
              onChange={(e) => onChange({ diarModelsSubdir: e.target.value })}
            />
          </Field>
          <div className="custom-model-row">
            <Field label="Hugging Face Repo or URL">
              <input
                spellCheck={false}
                value={diarCustomUrl}
                placeholder="org/model or https://…"
                onChange={(e) => setDiarCustomUrl(e.target.value)}
              />
            </Field>
            <button
              className="model-download-button"
              disabled={!diarCustomUrl.trim()}
              type="button"
              onClick={() => { handleDiarDownload({ name: diarCustomUrl.trim() }); setDiarCustomUrl(""); }}
            >
              <Download size={13} />
              Download
            </button>
          </div>
        </div>
        {diarError ? <div className="models-error">{diarError}</div> : null}
        {diarModels === null ? (
          <div className="models-loading">Loading…</div>
        ) : (
          <div className="models-list">
            {diarModels.map((model) => {
              const selected = String(draft.diarSherpaEmbeddingModelPath || "") === String(model.path || "");
              return (
                <DiarModelRow
                  key={model.name}
                  model={model}
                  selected={selected}
                  isDownloading={model.downloading || diarDownloading.has(model.name)}
                  expanded={diarExpanded.has(model.name)}
                  deletable={Boolean(model.deletable || model.cached)}
                  onUse={handleDiarUse}
                  onDownload={handleDiarDownload}
                  onDelete={handleDiarDelete}
                  onToggleExpand={handleDiarToggleExpand}
                />
              );
            })}
          </div>
        )}
      </InnerCollapsible>

      {/* ── Language Models (LLM) ── */}
      <InnerCollapsible title="Language Models">
        <div className="inner-block-controls">
          <Field label="Subfolder">
            <input
              spellCheck={false}
              value={draft.llmModelsSubdir || ""}
              placeholder="Default: llm"
              onChange={(e) => onChange({ llmModelsSubdir: e.target.value })}
            />
          </Field>
          <div className="custom-model-row">
            <Field label="Hugging Face Repo/File or URL">
              <input
                spellCheck={false}
                value={llmCustomUrl}
                placeholder="https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF"
                onChange={(e) => setLlmCustomUrl(e.target.value)}
              />
            </Field>
            <button
              className="model-download-button"
              disabled={!llmCustomUrl.trim()}
              type="button"
              onClick={() => { handleLlmDownload(llmCustomUrl.trim()); setLlmCustomUrl(""); }}
            >
              <Download size={13} />
              Download
            </button>
          </div>
        </div>
        {llmError ? <div className="models-error">{llmError}</div> : null}
        {llmModels === null ? (
          <div className="models-loading">Loading...</div>
        ) : llmModels.length === 0 ? (
          <div className="inner-collapsible-empty">No GGUF language models found</div>
        ) : (
          <div className="models-list">
            {llmModels.map((model) => {
              const alias = String(model.modelAlias || model.name || "");
              return (
                <LlmModelRow
                  key={model.path || model.name}
                  model={model}
                  alias={alias}
                  linked={linkedLlmAliases.has(alias)}
                  isDownloading={model.downloading || llmDownloading.has(model.name)}
                  onUse={handleLlmUse}
                  onDelete={handleLlmDelete}
                />
              );
            })}
          </div>
        )}
      </InnerCollapsible>
    </CollapsibleSection>
  );
}

// ─── settings dialog ─────────────────────────────────────────────────────────

export function SettingsDialogButton({
  capabilities,
  dirty,
  draft,
  hardware,
  locked,
  options,
  saving,
  onAsrChange,
  onChange,
  onReloadApp,
  onSave
}) {
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    if (!open) return undefined;
    const closeOnEscape = (e) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open]);

  const deviceOptions = uniqueOptions(options.asrDevices?.length ? options.asrDevices : FALLBACK_OPTIONS.asrDevices, draft.device);
  const modelOptions = uniqueOptions(options.asrModels?.length ? options.asrModels : FALLBACK_OPTIONS.asrModels, draft.model);
  const computeOptions = uniqueOptions(options.computeTypes?.length ? options.computeTypes : FALLBACK_OPTIONS.computeTypes, draft.computeType);
  const overloadOptions = uniqueOptions(options.overloadStrategies?.length ? options.overloadStrategies : FALLBACK_OPTIONS.overloadStrategies, draft.overloadStrategy);

  const handleReloadApp = React.useCallback(async () => {
    if (dirty) {
      const saved = await onSave();
      if (!saved) return;
    }
    await onReloadApp();
  }, [dirty, onReloadApp, onSave]);

  return (
    <>
      <button className={`icon-button settings-trigger ${dirty ? "dirty" : ""}`} title="Settings" type="button" onClick={() => setOpen(true)}>
        <Settings size={16} />
        {dirty ? <span /> : null}
      </button>

      {open ? (
        <div className="settings-modal-backdrop" role="presentation" onMouseDown={() => setOpen(false)}>
          <section aria-modal="true" className="settings-modal" role="dialog" onMouseDown={(e) => e.stopPropagation()}>
            <header className="settings-modal-head">
              <div>
                <span>Settings</span>
                <h2>Runtime & ASR</h2>
              </div>
              <button aria-label="Close settings" className="icon-button" type="button" onClick={() => setOpen(false)}>
                <X size={16} />
              </button>
            </header>

            <div className="settings-modal-body">
              <AppearanceSettings
                capabilities={capabilities}
                perProcessAudio={draft.perProcessAudio}
                screenCaptureProtection={draft.screenCaptureProtection}
                onChange={onChange}
              />
              <HardwareSummary hardware={hardware} />
              <ProxySettings draft={draft} onChange={onChange} />
              <AssistantSettings draft={draft} onChange={onChange} />
              <AdvancedAsrSettings
                computeOptions={computeOptions}
                deviceOptions={deviceOptions}
                draft={draft}
                locked={locked}
                modelOptions={modelOptions}
                overloadOptions={overloadOptions}
                onAsrChange={onAsrChange}
                onChange={onChange}
              />
              <DiarizationSettings draft={draft} locked={locked} options={options} onChange={onChange} />
              <ModelsSection draft={draft} open={open} onChange={onChange} />
            </div>

            <footer className="settings-modal-foot">
              <button className="save-button secondary" disabled={!dirty || saving} onClick={onSave} type="button">
                {dirty ? <Save size={15} /> : <Check size={15} />}
                {saving ? "Saving" : dirty ? "Save Settings" : "Saved"}
              </button>
              <button className="save-button" type="button" onClick={handleReloadApp}>
                <RefreshCw size={15} />
                Save & Refresh
              </button>
            </footer>
          </section>
        </div>
      ) : null}
    </>
  );
}
