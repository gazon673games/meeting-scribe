import { Check, Trash2 } from "lucide-react";

import { formatBytes } from "../../shared/lib/format";
import { formatDownloadProgress } from "./modelDownloadProgress";

function llmStatusLabel(model, linked, isDownloading) {
  if (isDownloading) return formatDownloadProgress(model);
  if (model.downloadError) return "error";
  if (linked) return "linked";
  return model.bytes ? formatBytes(model.bytes) : "ready";
}

function llmStatusClass(model, isDownloading) {
  if (model.cached) return "model-status-cached";
  if (isDownloading) return "model-status-downloading";
  if (model.downloadError) return "model-status-error";
  return "model-status-missing";
}

export function LlmModelRow({ model, alias, linked, isDownloading, onUse, onDelete }) {
  const row = buildLlmRowState(model, alias, linked, isDownloading);

  return (
    <div className="model-row-shell">
      <div className="model-row">
        <span className="model-row-name" title={row.title}>{row.label}</span>
        <span className={`model-row-status ${row.statusClass}`} title={row.statusTitle}>
          {row.statusLabel}
        </span>
        <button className="model-row-btn" disabled={row.useDisabled} title={`Use ${alias} in selected assistant profile`} type="button" onClick={() => onUse(model)}>
          <Check size={12} />
        </button>
        <button className="model-row-btn danger" disabled={row.deleteDisabled} title={row.deleteTitle} type="button" onClick={() => onDelete(model)}>
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}

function buildLlmRowState(model, alias, linked, isDownloading) {
  const label = model.label || model.name;
  const statusLabel = llmStatusLabel(model, linked, isDownloading);
  return {
    label,
    title: model.path || model.name,
    statusLabel,
    statusClass: llmStatusClass(model, isDownloading),
    statusTitle: model.downloadError || model.downloadMessage || statusLabel,
    useDisabled: isDownloading || !alias,
    deleteDisabled: linked || isDownloading || !model.path,
    deleteTitle: linked ? "Model is used by an assistant profile" : `Delete ${label}`,
  };
}
