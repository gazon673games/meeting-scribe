import React from 'react';
import { Check, RefreshCw, Save, Settings, X } from 'lucide-react';

import { FALLBACK_OPTIONS, uniqueOptions } from '../../entities/settings/model';
import { AdvancedAsrSettings } from './AdvancedAsrSettings';
import { AppearanceSettings } from './AppearanceSettings';
import { AssistantSettings } from './AssistantSettings';
import { CaptureSettings } from './CaptureSettings';
import { DiarizationSettings } from './DiarizationSettings';
import { HardwareSummary } from './HardwareSummary';
import { ModelsSection } from './ModelsSection';
import { ProxySettings } from './ProxySettings';

const settingsSaveStatus = (saving, dirty) => {
  if (saving) return 'Saving...';
  if (dirty) return 'Unsaved changes';
  return (
    <>
      <Check size={14} /> Saved
    </>
  );
};

export const SettingsDialogButton = ({
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
  onSave,
}) => {
  const [open, setOpen] = React.useState(false);
  const titleId = React.useId();
  const saveStatus = settingsSaveStatus(saving, dirty);

  React.useEffect(() => {
    if (!open) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [open]);

  const deviceOptions = uniqueOptions(
    options.asrDevices?.length
      ? options.asrDevices
      : FALLBACK_OPTIONS.asrDevices,
    draft.device,
  );
  const modelOptions = uniqueOptions(
    options.asrModels?.length ? options.asrModels : FALLBACK_OPTIONS.asrModels,
    draft.model,
  );
  const computeOptions = uniqueOptions(
    options.computeTypes?.length
      ? options.computeTypes
      : FALLBACK_OPTIONS.computeTypes,
    draft.computeType,
  );
  const offlineModelOptions = uniqueOptions(
    options.offlineAsrModels?.length
      ? options.offlineAsrModels
      : FALLBACK_OPTIONS.offlineAsrModels,
    draft.offlineModel,
  );
  const overloadOptions = uniqueOptions(
    options.overloadStrategies?.length
      ? options.overloadStrategies
      : FALLBACK_OPTIONS.overloadStrategies,
    draft.overloadStrategy,
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
        className={`icon-button settings-trigger ${dirty ? 'dirty' : ''}`}
        title="Settings"
        type="button"
        onClick={() => setOpen(true)}
      >
        <Settings size={16} />
        {dirty ? <span /> : null}
      </button>

      {open ? (
        <div
          className="settings-modal-backdrop"
          role="presentation"
          onMouseDown={() => setOpen(false)}
        >
          <section
            aria-modal="true"
            aria-labelledby={titleId}
            className="settings-modal"
            role="dialog"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header className="settings-modal-head">
              <div>
                <h2 id={titleId}>Settings</h2>
              </div>
              <button
                aria-label="Close settings"
                className="icon-button"
                type="button"
                onClick={() => setOpen(false)}
              >
                <X size={16} />
              </button>
            </header>

            <div className="settings-modal-body">
              <AppearanceSettings theme={draft.theme} onChange={onChange} />
              <CaptureSettings
                capabilities={capabilities}
                perProcessAudio={draft.perProcessAudio}
                screenCaptureProtection={draft.screenCaptureProtection}
                onChange={onChange}
              />
              <HardwareSummary hardware={hardware} />
              <ProxySettings draft={draft} onChange={onChange} />
              <AssistantSettings draft={draft} onChange={onChange} />
              <div className="settings-group">
                <h3 className="settings-group-title">Runtime & ASR</h3>
                <AdvancedAsrSettings
                  computeOptions={computeOptions}
                  deviceOptions={deviceOptions}
                  draft={draft}
                  locked={locked}
                  modelOptions={modelOptions}
                  offlineModelOptions={offlineModelOptions}
                  overloadOptions={overloadOptions}
                  onAsrChange={onAsrChange}
                  onChange={onChange}
                />
                <DiarizationSettings
                  draft={draft}
                  locked={locked}
                  options={options}
                  onChange={onChange}
                />
              </div>
              <ModelsSection
                draft={draft}
                locked={locked}
                open={open}
                onChange={onChange}
              />
            </div>

            <footer className="settings-modal-foot">
              <span className="settings-save-status" role="status">
                {saveStatus}
              </span>
              {dirty ? (
                <button
                  className="save-button secondary"
                  disabled={saving}
                  onClick={onSave}
                  type="button"
                >
                  <Save size={15} />
                  Save Settings
                </button>
              ) : null}
              <button
                className="save-button"
                disabled={saving}
                title="Save settings and reload the app window."
                type="button"
                onClick={handleReloadApp}
              >
                <RefreshCw size={15} />
                Save & Refresh Window
              </button>
            </footer>
          </section>
        </div>
      ) : null}
    </>
  );
};
