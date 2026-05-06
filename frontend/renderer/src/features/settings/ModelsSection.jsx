import React from "react";
import { Network, RefreshCw } from "lucide-react";

import { buildProxyUrl } from "../../entities/settings/model";
import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";
import { Field } from "../../shared/ui/Field";
import { AsrModelsPanel } from "./modelSections/AsrModelsPanel";
import { DiarModelsPanel } from "./modelSections/DiarModelsPanel";
import { LlmModelsPanel } from "./modelSections/LlmModelsPanel";
import { joinDir } from "./modelSections/useModelList";

export function ModelsSection({ draft, onChange, open }) {
  const [refreshToken, setRefreshToken] = React.useState(0);

  const baseDir = String(draft.modelsDirectory || "").trim();
  const asrDir = joinDir(baseDir, draft.asrModelsSubdir);
  const diarDir = joinDir(baseDir, draft.diarModelsSubdir);
  const llmDir = joinDir(baseDir, draft.llmModelsSubdir);
  const useProxy = Boolean(draft.modelsUseProxy);
  const downloadProxy = buildProxyUrl(draft, { forceEnabled: useProxy });

  const refreshAll = React.useCallback(() => {
    setRefreshToken((current) => current + 1);
  }, []);

  if (!open) return null;

  return (
    <CollapsibleSection
      title="Models"
      defaultOpen={false}
      action={(
        <button className="icon-button" title="Refresh all models" type="button" onClick={refreshAll}>
          <RefreshCw size={13} />
        </button>
      )}
    >
      <div className="models-controls">
        <Field label="Models Folder">
          <input
            spellCheck={false}
            value={draft.modelsDirectory || ""}
            placeholder="Default: ./models"
            onChange={(event) => onChange({ modelsDirectory: event.target.value })}
          />
        </Field>
        <button
          aria-pressed={useProxy}
          className={`feature-toggle model-proxy-toggle ${useProxy ? "selected" : ""}`}
          type="button"
          onClick={() => onChange({ modelsUseProxy: !useProxy })}
        >
          <Network size={15} />
          <span>Use Proxy for Downloads</span>
          <b />
        </button>
      </div>

      <AsrModelsPanel
        draft={draft}
        onChange={onChange}
        open={open}
        asrDir={asrDir}
        useProxy={useProxy}
        downloadProxy={downloadProxy}
        refreshToken={refreshToken}
      />
      <DiarModelsPanel
        draft={draft}
        onChange={onChange}
        open={open}
        diarDir={diarDir}
        useProxy={useProxy}
        downloadProxy={downloadProxy}
        refreshToken={refreshToken}
      />
      <LlmModelsPanel
        draft={draft}
        onChange={onChange}
        open={open}
        llmDir={llmDir}
        useProxy={useProxy}
        downloadProxy={downloadProxy}
        refreshToken={refreshToken}
      />
    </CollapsibleSection>
  );
}
