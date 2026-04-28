import React from "react";
import { Check, Download, RefreshCw, Save, Settings, X } from "lucide-react";

import { FALLBACK_OPTIONS, uniqueOptions } from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { AdvancedAsrSettings } from "./AdvancedAsrSettings";
import { AppearanceSettings } from "./AppearanceSettings";
import { AssistantProxySettings } from "./AssistantProxySettings";
import { HardwareSummary } from "./HardwareSummary";

function ModelsSection({ open }) {
  const [models, setModels] = React.useState(null);
  const [modelError, setModelError] = React.useState("");
  const [downloadingNames, setDownloadingNames] = React.useState(new Set());

  const loadModels = React.useCallback(() => {
    meetingScribeClient
      .request("list_models")
      .then((result) => setModels(result?.models || []))
      .catch((err) => setModelError(String(err?.message || err)));
  }, []);

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
    const handle = window.setInterval(loadModels, 2000);
    return () => window.clearInterval(handle);
  }, [models, loadModels]);

  const handleDownload = React.useCallback(
    async (name) => {
      setDownloadingNames((prev) => new Set([...prev, name]));
      setModelError("");
      try {
        await meetingScribeClient.request("download_model", { name });
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
    [loadModels]
  );

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
      {modelError ? <div className="models-error">{modelError}</div> : null}
      {models === null ? (
        <div className="models-loading">Loading...</div>
      ) : (
        <div className="models-list">
          {models.map((model) => {
            const isStarting = downloadingNames.has(model.name);
            const isDownloading = model.downloading || isStarting;
            const statusLabel = model.cached
              ? "cached"
              : isDownloading
                ? model.downloadMessage || "downloading..."
                : model.downloadError
                  ? "error"
                  : model.builtin
                    ? "auto on first use"
                    : "not cached";
            const statusClass = model.cached
              ? "model-status-cached"
              : isDownloading
                ? "model-status-downloading"
                : model.downloadError
                  ? "model-status-error"
                  : "model-status-missing";
            return (
              <div key={model.name} className="model-row">
                <span className="model-row-name">{model.name}</span>
                <span className={`model-row-status ${statusClass}`}>{statusLabel}</span>
                {!model.cached && !isDownloading ? (
                  <button
                    className="model-row-btn"
                    title={`Download ${model.name}`}
                    type="button"
                    onClick={() => handleDownload(model.name)}
                  >
                    <Download size={12} />
                  </button>
                ) : null}
              </div>
            );
          })}
        </div>
      )}
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
              <ModelsSection open={open} />
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
