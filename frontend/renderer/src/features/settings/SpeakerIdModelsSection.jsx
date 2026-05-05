import React from "react";
import { RefreshCw } from "lucide-react";

import { buildProxyUrl } from "../../entities/settings/model";
import { meetingScribeClient } from "../../shared/api/meetingScribeClient";
import { CollapsibleSection } from "../../shared/ui/CollapsibleSection";
import { InnerCollapsible } from "../../shared/ui/InnerCollapsible";
import { DiarModelRow } from "./DiarModelRow";

export function SpeakerIdModelsSection({ draft, onChange, open }) {
  const [models, setModels] = React.useState(null);
  const [modelError, setModelError] = React.useState("");
  const [downloadingNames, setDownloadingNames] = React.useState(new Set());
  const [expandedNames, setExpandedNames] = React.useState(new Set());
  const requestModelsDir = String(draft.modelsDirectory || "").trim();

  const loadModels = React.useCallback(() => {
    meetingScribeClient
      .request("list_diarization_models", { modelsDir: requestModelsDir })
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
    if (!models?.some((model) => model.downloading)) {
      return undefined;
    }
    const handle = window.setInterval(loadModels, 1000);
    return () => window.clearInterval(handle);
  }, [models, loadModels]);

  React.useEffect(() => {
    if (!open || !draft.diarizationEnabled || draft.diarSherpaEmbeddingModelPath || draft.diarizationBackend !== "online") {
      return;
    }
    const readyModel = models?.find((model) => model.cached && model.compatible && (model.backend || "sherpa_onnx") === "sherpa_onnx");
    if (!readyModel?.path) {
      return;
    }
    onChange({
      diarizationBackend: readyModel.backend || "sherpa_onnx",
      diarizationSidecarEnabled: true,
      diarSherpaEmbeddingModelPath: readyModel.path,
      diarSherpaProvider: readyModel.provider || "cpu"
    });
  }, [draft.diarizationBackend, draft.diarizationEnabled, draft.diarSherpaEmbeddingModelPath, models, onChange, open]);

  const handleUse = React.useCallback(
    (model) => {
      onChange({
        diarizationEnabled: true,
        diarizationBackend: model.backend || "sherpa_onnx",
        diarizationSidecarEnabled: true,
        diarSherpaEmbeddingModelPath: model.path,
        diarSherpaProvider: model.provider || "cpu"
      });
    },
    [onChange]
  );

  const handleDownload = React.useCallback(
    async (model) => {
      setModelError("");
      setDownloadingNames((current) => new Set([...current, model.name]));
      try {
        await meetingScribeClient.request("download_diarization_model", {
          name: model.name,
          modelsDir: requestModelsDir,
          useProxy: Boolean(draft.modelsUseProxy),
          proxy: buildProxyUrl(draft, { forceEnabled: Boolean(draft.modelsUseProxy) })
        });
        loadModels();
      } catch (err) {
        setModelError(String(err?.message || err));
      } finally {
        setDownloadingNames((current) => {
          const next = new Set(current);
          next.delete(model.name);
          return next;
        });
      }
    },
    [draft, loadModels, requestModelsDir]
  );

  const handleDelete = React.useCallback(
    async (model) => {
      setModelError("");
      try {
        await meetingScribeClient.request("delete_diarization_model", {
          path: model.path,
          modelsDir: requestModelsDir
        });
        loadModels();
      } catch (err) {
        setModelError(String(err?.message || err));
      }
    },
    [loadModels, requestModelsDir]
  );

  const handleToggleExpand = React.useCallback((name) => {
    setExpandedNames((current) => {
      const next = new Set(current);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  }, []);

  if (!open) {
    return null;
  }

  return (
    <CollapsibleSection
      title="Speaker ID Models"
      defaultOpen={false}
      action={
        <button className="icon-button" title="Refresh Speaker ID models" type="button" onClick={loadModels}>
          <RefreshCw size={13} />
        </button>
      }
    >
      {modelError ? <div className="models-error">{modelError}</div> : null}
      {models === null ? (
        <div className="models-loading">Loading...</div>
      ) : (
        <div className="models-list">
          {models.map((model) => {
            const selected = String(draft.diarSherpaEmbeddingModelPath || "") === String(model.path || "");
            return (
              <DiarModelRow
                key={model.name}
                model={model}
                selected={selected}
                isDownloading={model.downloading || downloadingNames.has(model.name)}
                expanded={expandedNames.has(model.name)}
                deletable={Boolean(model.deletable)}
                showStatus
                onUse={handleUse}
                onDownload={handleDownload}
                onDelete={handleDelete}
                onToggleExpand={handleToggleExpand}
              />
            );
          })}
        </div>
      )}
      <InnerCollapsible title="Local Models">
        <div className="inner-collapsible-empty">No local models configured</div>
      </InnerCollapsible>
    </CollapsibleSection>
  );
}

function downloadStatusLabel(model) {
  const downloadedBytes = Number(model.downloadedBytes || 0);
  const totalBytes = Number(model.totalBytes || 0);
  const speedBps = Number(model.speedBps || 0);
  const downloaded = downloadedBytes > 0 ? formatBytes(downloadedBytes) : "0 B";
  const total = totalBytes > 0 ? ` / ${formatBytes(totalBytes)}` : "";
  const speed = speedBps > 0 ? ` - ${formatBytes(speedBps)}/s` : "";
  return `${downloaded}${total}${speed}`;
}
