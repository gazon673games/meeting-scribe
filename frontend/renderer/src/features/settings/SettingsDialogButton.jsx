import React from "react";
import { Check, RefreshCw, Save, Settings, X } from "lucide-react";

import { FALLBACK_OPTIONS, uniqueOptions } from "../../entities/settings/model";
import { AdvancedAsrSettings } from "./AdvancedAsrSettings";
import { AppearanceSettings } from "./AppearanceSettings";
import { AssistantSettings } from "./AssistantSettings";
import { DiarizationSettings } from "./DiarizationSettings";
import { HardwareSummary } from "./HardwareSummary";
import { ModelsSection } from "./ModelsSection";
import { ProxySettings } from "./ProxySettings";

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
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [open]);

  const deviceOptions = uniqueOptions(
    options.asrDevices?.length ? options.asrDevices : FALLBACK_OPTIONS.asrDevices,
    draft.device
  );
  const modelOptions = uniqueOptions(
    options.asrModels?.length ? options.asrModels : FALLBACK_OPTIONS.asrModels,
    draft.model
  );
  const computeOptions = uniqueOptions(
    options.computeTypes?.length ? options.computeTypes : FALLBACK_OPTIONS.computeTypes,
    draft.computeType
  );
  const overloadOptions = uniqueOptions(
    options.overloadStrategies?.length ? options.overloadStrategies : FALLBACK_OPTIONS.overloadStrategies,
    draft.overloadStrategy
  );

  const handleReloadApp = React.useCallback(async () => {
    if (dirty) {
      const saved = await onSave();
      if (!saved) return;
    }
    await onReloadApp();
  }, [dirty, onReloadApp, onSave]);

  return (
    <>
      <button
        className={`icon-button settings-trigger ${dirty ? "dirty" : ""}`}
        title="Settings"
        type="button"
        onClick={() => setOpen(true)}
      >
        <Settings size={16} />
        {dirty ? <span /> : null}
      </button>

      {open ? (
        <div className="settings-modal-backdrop" role="presentation" onMouseDown={() => setOpen(false)}>
          <section
            aria-modal="true"
            className="settings-modal"
            role="dialog"
            onMouseDown={(event) => event.stopPropagation()}
          >
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
