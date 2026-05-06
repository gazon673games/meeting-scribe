import React from "react";
import { Download } from "lucide-react";

import { meetingScribeClient } from "../../../shared/api/meetingScribeClient";
import { Field } from "../../../shared/ui/Field";
import { InnerCollapsible } from "../../../shared/ui/InnerCollapsible";
import { DiarModelRow } from "../DiarModelRow";
import { formatError, runWithPending, useModelList } from "./useModelList";

export function DiarModelsPanel({ draft, onChange, open, diarDir, useProxy, downloadProxy, refreshToken }) {
  const [downloading, setDownloading] = React.useState(new Set());
  const [customUrl, setCustomUrl] = React.useState("");
  const [expanded, setExpanded] = React.useState(new Set());

  const diarList = useModelList({
    open,
    refreshToken,
    loadFn: React.useCallback(
      () => meetingScribeClient.request("list_diarization_models", { modelsDir: diarDir }),
      [diarDir]
    )
  });

  React.useEffect(() => {
    if (!open || !draft.diarizationEnabled || draft.diarSherpaEmbeddingModelPath || draft.diarizationBackend !== "online") return;
    const ready = diarList.models?.find(
      (model) => model.cached && model.compatible && (model.backend || "sherpa_onnx") === "sherpa_onnx"
    );
    if (!ready?.path) return;
    onChange({
      diarizationBackend: ready.backend || "sherpa_onnx",
      diarizationSidecarEnabled: true,
      diarSherpaEmbeddingModelPath: ready.path,
      diarSherpaProvider: ready.provider || "cpu"
    });
  }, [
    diarList.models,
    draft.diarSherpaEmbeddingModelPath,
    draft.diarizationBackend,
    draft.diarizationEnabled,
    onChange,
    open
  ]);

  const handleUse = React.useCallback((model) => {
    onChange({
      diarizationEnabled: true,
      diarizationBackend: model.backend || "sherpa_onnx",
      diarizationSidecarEnabled: true,
      diarSherpaEmbeddingModelPath: model.path,
      diarSherpaProvider: model.provider || "cpu"
    });
  }, [onChange]);

  const handleDownload = React.useCallback(async (model) => {
    diarList.setError("");
    try {
      await runWithPending(setDownloading, model.name, async () => {
        await meetingScribeClient.request("download_diarization_model", {
          name: model.name,
          modelsDir: diarDir,
          useProxy,
          proxy: downloadProxy
        });
        diarList.load();
      });
    } catch (requestError) {
      diarList.setError(formatError(requestError));
    }
  }, [diarDir, diarList, downloadProxy, useProxy]);

  const handleDelete = React.useCallback(async (model) => {
    diarList.setError("");
    try {
      await meetingScribeClient.request("delete_diarization_model", { path: model.path, modelsDir: diarDir });
      diarList.load();
    } catch (requestError) {
      diarList.setError(formatError(requestError));
    }
  }, [diarDir, diarList]);

  const handleToggleExpand = React.useCallback((name) => {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });
  }, []);

  return (
    <InnerCollapsible title="Speaker ID">
      <div className="inner-block-controls">
        <Field label="Subfolder">
          <input
            spellCheck={false}
            value={draft.diarModelsSubdir || ""}
            placeholder="e.g. diar"
            onChange={(event) => onChange({ diarModelsSubdir: event.target.value })}
          />
        </Field>
        <div className="custom-model-row">
          <Field label="Hugging Face Repo or URL">
            <input
              spellCheck={false}
              value={customUrl}
              placeholder="org/model or https://..."
              onChange={(event) => setCustomUrl(event.target.value)}
            />
          </Field>
          <button
            className="model-download-button"
            disabled={!customUrl.trim()}
            type="button"
            onClick={() => {
              handleDownload({ name: customUrl.trim() });
              setCustomUrl("");
            }}
          >
            <Download size={13} />
            Download
          </button>
        </div>
      </div>

      {diarList.error ? <div className="models-error">{diarList.error}</div> : null}
      {diarList.models === null ? (
        <div className="models-loading">Loading...</div>
      ) : (
        <div className="models-list">
          {diarList.models.map((model) => {
            const selected = String(draft.diarSherpaEmbeddingModelPath || "") === String(model.path || "");
            return (
              <DiarModelRow
                key={model.name}
                model={model}
                selected={selected}
                isDownloading={model.downloading || downloading.has(model.name)}
                expanded={expanded.has(model.name)}
                deletable={Boolean(model.deletable || model.cached)}
                onUse={handleUse}
                onDownload={handleDownload}
                onDelete={handleDelete}
                onToggleExpand={handleToggleExpand}
              />
            );
          })}
        </div>
      )}
    </InnerCollapsible>
  );
}
