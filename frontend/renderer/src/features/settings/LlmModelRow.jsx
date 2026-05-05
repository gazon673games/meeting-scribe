import { Check, Trash2 } from "lucide-react";
import { formatBytes } from "../../shared/lib/format";

function llmStatusLabel(model, linked, isDownloading) {
  if (isDownloading) return llmDownloadLabel(model);
  if (model.downloadError) return "error";
  if (linked) return "linked";
  return model.bytes ? formatBytes(model.bytes) : "ready";
}

function llmDownloadLabel(model) {
  const dl = Number(model.downloadedBytes || 0);
  const tot = Number(model.totalBytes || 0);
  const sp = Number(model.speedBps || 0);
  const dlStr = dl > 0 ? formatBytes(dl) : "0 B";
  const totStr = tot > 0 ? ` / ${formatBytes(tot)}` : "";
  const spStr = sp > 0 ? ` – ${formatBytes(sp)}/s` : "";
  return `${dlStr}${totStr}${spStr}`;
}

function llmStatusClass(model, isDownloading) {
  if (model.cached) return "model-status-cached";
  if (isDownloading) return "model-status-downloading";
  if (model.downloadError) return "model-status-error";
  return "model-status-missing";
}

export function LlmModelRow({ model, alias, linked, isDownloading, onUse, onDelete }) {
  const statusLabel = llmStatusLabel(model, linked, isDownloading);
  const statusClass = llmStatusClass(model, isDownloading);

  return (
    <div className="model-row-shell">
      <div className="model-row">
        <span className="model-row-name" title={model.path || model.name}>{model.label || model.name}</span>
        <span className={`model-row-status ${statusClass}`} title={model.downloadError || model.downloadMessage || statusLabel}>
          {statusLabel}
        </span>
        <button className="model-row-btn" disabled={isDownloading || !alias} title={`Use ${alias} in selected assistant profile`} type="button" onClick={() => onUse(model)}>
          <Check size={12} />
        </button>
        <button className="model-row-btn danger" disabled={linked || isDownloading || !model.path} title={linked ? "Model is used by an assistant profile" : `Delete ${model.label || model.name}`} type="button" onClick={() => onDelete(model)}>
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}
