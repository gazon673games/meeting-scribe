import React from "react";
import { Check, ChevronDown, ChevronUp, Download, Network, RefreshCw, Save, Settings, Trash2, X } from "lucide-react";

import { FALLBACK_OPTIONS, buildProxyUrl, uniqueOptions } from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { formatBytes } from "../../shared/lib/format";
import { Field } from "../../shared/ui/Field";
import { AdvancedAsrSettings } from "./AdvancedAsrSettings";
import { AppearanceSettings } from "./AppearanceSettings";
import { AssistantProxySettings } from "./AssistantProxySettings";
import { HardwareSummary } from "./HardwareSummary";

function ModelsSection({ draft, onChange, open }) {
  const [models, setModels] = React.useState(null);
  const [modelError, setModelError] = React.useState("");
  const [customModel, setCustomModel] = React.useState("");
  const [downloadingNames, setDownloadingNames] = React.useState(new Set());
  const [metadataByModel, setMetadataByModel] = React.useState({});
  const requestModelsDir = String(draft.modelsDirectory || "").trim();

  const loadModels = React.useCallback(() => {
    meetingScribeClient
      .request("list_models", { modelsDir: requestModelsDir })
      .then((result) => setModels(result?.models || []))
      .catch((err) => setModelError(String(err?.message || err)));
  }, [requestModelsDir]);

  React.useEffect(() => {
    if (!open) {
      return;
    }
    loadModels();
  }, [open, loadModels]);

  React.useEffect(() => {
    if (!models) {
      return undefined;
    }
    const hasActive = models.some((m) => m.downloading);
    if (!hasActive) {
      return undefined;
    }
    const handle = window.setInterval(loadModels, 1000);
    return () => window.clearInterval(handle);
  }, [models, loadModels]);

  const handleDownload = React.useCallback(
    async (name) => {
      setDownloadingNames((prev) => new Set([...prev, name]));
      setModelError("");
      try {
        await meetingScribeClient.request("download_model", {
          name,
          modelsDir: requestModelsDir,
          useProxy: Boolean(draft.modelsUseProxy),
          proxy: buildProxyUrl(draft, { forceEnabled: Boolean(draft.modelsUseProxy) })
        });
        loadModels();
      } catch (err) {
        setModelError(String(err?.message || err));
      } finally {
        setDownloadingNames((prev) => {
          const next = new Set(prev);
          next.delete(name);
          return next;
        });
      }
    },
    [draft, loadModels, requestModelsDir]
  );

  const handleCustomDownload = React.useCallback(() => {
    const name = customModel.trim();
    if (!name) {
      return;
    }
    handleDownload(name);
  }, [customModel, handleDownload]);

  const handleDelete = React.useCallback(
    async (name) => {
      setModelError("");
      try {
        await meetingScribeClient.request("delete_model", { name, modelsDir: requestModelsDir });
        loadModels();
      } catch (err) {
        setModelError(String(err?.message || err));
      }
    },
    [loadModels, requestModelsDir]
  );

  const handleToggleMetadata = React.useCallback(async (model) => {
    const key = model.name;
    if (metadataByModel[key]) {
      setMetadataByModel((current) => {
        const next = { ...current };
        delete next[key];
        return next;
      });
      return;
    }
    setModelError("");
    setMetadataByModel((current) => ({
      ...current,
      [key]: { model, metadata: null, error: "", loading: true }
    }));
    try {
      const metadata = await meetingScribeClient.request("model_metadata", { name: model.name, modelsDir: requestModelsDir });
      setMetadataByModel((current) => {
        if (!current[key]) {
          return current;
        }
        return { ...current, [key]: { model, metadata, error: "", loading: false } };
      });
    } catch (err) {
      setMetadataByModel((current) => {
        if (!current[key]) {
          return current;
        }
        return { ...current, [key]: { model, metadata: null, error: String(err?.message || err), loading: false } };
      });
    }
  }, [metadataByModel, requestModelsDir]);

  if (!open) {
    return null;
  }

  return (
    <section className="settings-section">
      <div className="settings-section-head">
        <h3>Models</h3>
        <button className="icon-button" title="Refresh models" type="button" onClick={loadModels}>
          <RefreshCw size={13} />
        </button>
      </div>
      <div className="models-controls">
        <Field label="Models Folder">
          <input
            spellCheck={false}
            value={draft.modelsDirectory || ""}
            placeholder="Default: ./models"
            onChange={(event) => onChange({ modelsDirectory: event.target.value })}
          />
        </Field>
        <div className="custom-model-row">
          <Field label="Hugging Face Repo or URL">
            <input
              spellCheck={false}
              value={customModel}
              placeholder="org/model or https://huggingface.co/org/model"
              onChange={(event) => setCustomModel(event.target.value)}
            />
          </Field>
          <button className="model-download-button" disabled={!customModel.trim()} type="button" onClick={handleCustomDownload}>
            <Download size={13} />
            Download
          </button>
        </div>
        <button
          aria-pressed={Boolean(draft.modelsUseProxy)}
          className={`feature-toggle model-proxy-toggle ${draft.modelsUseProxy ? "selected" : ""}`}
          type="button"
          onClick={() => onChange({ modelsUseProxy: !draft.modelsUseProxy })}
        >
          <Network size={15} />
          <span>Use Proxy for Hugging Face</span>
          <b />
        </button>
      </div>
      {modelError ? <div className="models-error">{modelError}</div> : null}
      {models === null ? (
        <div className="models-loading">Loading...</div>
      ) : (
        <div className="models-list">
          {models.map((model) => {
            const isStarting = downloadingNames.has(model.name);
            const isDownloading = model.downloading || isStarting;
            const compatible = Boolean(model.compatible || model.cached);
            const statusLabel = model.cached
              ? model.compatible === false
                ? "incompatible"
                : "ready"
              : isDownloading
                ? downloadStatusLabel(model)
                : model.downloadError
                  ? "error"
                  : model.status === "inaccessible"
                    ? "inaccessible"
                    : model.status === "unsupported_transformers_format"
                      ? "unsupported format"
                      : model.status === "missing_files"
                    ? `missing ${model.missing?.join(", ") || "files"}`
                    : model.source === "recommended"
                      ? "downloadable"
                      : "not ready";
            const statusClass = model.cached
              ? compatible
                ? "model-status-cached"
                : "model-status-error"
              : isDownloading
                ? "model-status-downloading"
                : model.downloadError
                  ? "model-status-error"
                  : model.status === "missing_files" || model.status === "unsupported_transformers_format" || model.status === "inaccessible"
                    ? "model-status-error"
                    : "model-status-missing";
            const metadataEntry = metadataByModel[model.name];
            const expanded = Boolean(metadataEntry);
            return (
              <div key={model.name} className={`model-row-shell ${expanded ? "expanded" : ""}`}>
                <div className="model-row">
                  <span className="model-row-name" title={model.path || model.name}>{model.label || model.name}</span>
                  <span className={`model-row-status ${statusClass}`} title={model.downloadError || model.downloadMessage || model.warnings?.join(", ") || statusLabel}>
                    {statusLabel}
                  </span>
                  <button
                    className="model-row-btn"
                    title={expanded ? `Hide metadata for ${model.name}` : `Show metadata for ${model.name}`}
                    type="button"
                    onClick={() => handleToggleMetadata(model)}
                  >
                    {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                  </button>
                  {compatible ? (
                    <button
                      className="model-row-btn"
                      disabled={draft.model === model.name}
                      title={`Use ${model.name}`}
                      type="button"
                      onClick={() => onChange({ model: model.name })}
                    >
                      <Check size={12} />
                    </button>
                  ) : null}
                  {model.downloadable !== false && !model.cached && !isDownloading ? (
                    <button
                      className="model-row-btn"
                      title={`Download ${model.name}`}
                      type="button"
                      onClick={() => handleDownload(model.name)}
                    >
                      <Download size={12} />
                    </button>
                  ) : null}
                  {model.deletable ? (
                    <button
                      className="model-row-btn danger"
                      disabled={draft.model === model.name || isDownloading}
                      title={draft.model === model.name ? "Selected model cannot be deleted" : `Delete ${model.name}`}
                      type="button"
                      onClick={() => handleDelete(model.name)}
                    >
                      <Trash2 size={12} />
                    </button>
                  ) : null}
                </div>
                {expanded ? (
                  <ModelMetadataPanel
                    loading={metadataEntry.loading}
                    metadata={metadataEntry.metadata}
                    error={metadataEntry.error}
                  />
                ) : null}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function downloadStatusLabel(model) {
  const downloadedBytes = Number(model.downloadedBytes || 0);
  const speedBps = Number(model.speedBps || 0);
  const hasDownloaded = Number.isFinite(downloadedBytes) && downloadedBytes > 0;
  const hasSpeed = Number.isFinite(speedBps) && speedBps > 0;
  if (hasDownloaded || hasSpeed) {
    const downloaded = hasDownloaded ? formatBytes(downloadedBytes) : "0 B";
    if (hasSpeed) {
      return `${downloaded} - ${formatBytes(speedBps)}/s`;
    }
    return downloaded;
  }
  return model.downloadMessage || "downloading...";
}

function ModelMetadataPanel({ error, loading, metadata }) {
  const rows = metadata ? metadataRows(metadata) : [];
  return (
    <div className="model-metadata-panel">
      {loading ? <div className="models-loading">Loading metadata...</div> : null}
      {error ? <div className="models-error">{error}</div> : null}
      {!loading && metadata ? (
        <>
          <dl className="model-metadata-grid">
            {rows.map(([label, value]) => (
              <React.Fragment key={label}>
                <dt>{label}</dt>
                <dd title={String(value)}>{String(value)}</dd>
              </React.Fragment>
            ))}
          </dl>
          <div className="model-metadata-blocks">
            <MetadataObject title="Config" value={metadata.config} />
            <MetadataObject title="Preprocessor" value={metadata.preprocessor} />
            <MetadataObject title="Tokenizer" value={metadata.tokenizer} />
            <MetadataObject title="Model Card" value={metadata.readme} />
          </div>
        </>
      ) : null}
    </div>
  );
}

function metadataRows(metadata) {
  return [
    ["Name", metadata.name || "-"],
    ["Normalized", metadata.normalizedName || "-"],
    ["Source", metadata.source || "-"],
    ["Status", metadata.status || "-"],
    ["Format", metadata.format || "-"],
    ["Compatible", metadata.compatible ? "yes" : "no"],
    ["Builtin", metadata.builtin ? "yes" : "no"],
    ["Size", formatBytes(metadata.totalBytes)],
    ["Repo", metadata.repoId || "-"],
    ["Missing", metadata.missing?.length ? metadata.missing.join(", ") : "-"],
    ["Files", metadata.presentFiles?.length ? metadata.presentFiles.join(", ") : "-"],
    ["Weights", weightFilesLabel(metadata.weightFiles)],
    ["Warnings", metadata.warnings?.length ? metadata.warnings.join(", ") : "-"],
    ["Cache", metadata.cachePath || "-"],
    ["Resolved", metadata.resolvedPath || "-"]
  ];
}

function weightFilesLabel(files) {
  if (!Array.isArray(files) || files.length === 0) {
    return "-";
  }
  return files.map((file) => `${file.name} (${formatBytes(file.bytes)})`).join(", ");
}

function MetadataObject({ title, value }) {
  if (!value || Object.keys(value).length === 0) {
    return null;
  }
  return (
    <section className="model-metadata-object">
      <h4>{title}</h4>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}

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
    if (!open) {
      return undefined;
    }
    const closeOnEscape = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };
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
      if (!saved) {
        return;
      }
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
          <section aria-modal="true" className="settings-modal" role="dialog" onMouseDown={(event) => event.stopPropagation()}>
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
                dirty={dirty}
                perProcessAudio={draft.perProcessAudio}
                reloading={saving}
                screenCaptureProtection={draft.screenCaptureProtection}
                theme={draft.theme}
                onChange={onChange}
                onReloadApp={handleReloadApp}
              />
              <AssistantProxySettings draft={draft} onChange={onChange} />
              <HardwareSummary hardware={hardware} />
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
              <ModelsSection draft={draft} open={open} onChange={onChange} />
            </div>

            <footer className="settings-modal-foot">
              <button className="save-button" disabled={!dirty || saving} onClick={onSave} type="button">
                {dirty ? <Save size={15} /> : <Check size={15} />}
                {saving ? "Saving" : dirty ? "Save Settings" : "Saved"}
              </button>
            </footer>
          </section>
        </div>
      ) : null}
    </>
  );
}
