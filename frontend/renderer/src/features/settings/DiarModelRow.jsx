import React from "react";
import { Check, ChevronDown, ChevronUp, Download, Trash2 } from "lucide-react";
import { formatBytes } from "../../shared/lib/format";

function diarDownloadLabel(model) {
  const dl = Number(model.downloadedBytes || 0);
  const tot = Number(model.totalBytes || 0);
  const sp = Number(model.speedBps || 0);
  const dlStr = dl > 0 ? formatBytes(dl) : "0 B";
  const totStr = tot > 0 ? ` / ${formatBytes(tot)}` : "";
  const spStr = sp > 0 ? ` – ${formatBytes(sp)}/s` : "";
  return `${dlStr}${totStr}${spStr}`;
}

function diarStatusLabel(model, selected, ready, isDownloading) {
  if (selected) return "selected";
  if (ready) return model.bytes ? formatBytes(model.bytes) : "ready";
  if (isDownloading) return diarDownloadLabel(model);
  if (model.downloadError) return "error";
  return "downloadable";
}

function diarStatusClass(model, selected, ready, isDownloading) {
  if (selected || ready) return "model-status-cached";
  if (isDownloading) return "model-status-downloading";
  if (model.downloadError) return "model-status-error";
  return "model-status-missing";
}

function diarDetailFields(model, statusLabel, showStatus) {
  const fields = [
    ["Name", model.name],
    ["Label", model.label || "-"],
    ["Backend", model.backend || "-"],
    ["Provider", model.provider || "-"],
    ["Size", model.bytes ? formatBytes(model.bytes) : "-"],
    ["Path", model.path || "-"],
  ];
  if (showStatus) fields.push(["Status", statusLabel]);
  return fields;
}

export function DiarModelRow({ model, selected, isDownloading, expanded, deletable, showStatus, onUse, onDownload, onDelete, onToggleExpand }) {
  const ready = Boolean(model.cached || model.compatible);
  const statusLabel = diarStatusLabel(model, selected, ready, isDownloading);
  const statusClass = diarStatusClass(model, selected, ready, isDownloading);

  return (
    <div className={`model-row-shell${expanded ? " expanded" : ""}`}>
      <div className="model-row">
        <span className="model-row-name" title={model.path || model.url || model.name}>
          {model.label || model.name}
        </span>
        <span className={`model-row-status ${statusClass}`} title={model.downloadError || model.downloadMessage || statusLabel}>
          {statusLabel}
        </span>

        <button className="model-row-btn" title={expanded ? "Hide info" : "Show info"} type="button" onClick={() => onToggleExpand(model.name)}>
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        {ready ? (
          <button className="model-row-btn" disabled={selected} title={selected ? "Already selected" : `Use ${model.label || model.name}`} type="button" onClick={() => onUse(model)}>
            <Check size={12} />
          </button>
        ) : (
          <button className="model-row-btn" disabled={isDownloading || model.downloadable === false} title={isDownloading ? "Downloading…" : `Download ${model.label || model.name}`} type="button" onClick={() => !isDownloading && model.downloadable !== false && onDownload(model)}>
            <Download size={12} />
          </button>
        )}

        {deletable ? (
          <button className="model-row-btn danger" disabled={selected || isDownloading} title={selected ? "Selected model cannot be deleted" : `Delete ${model.label || model.name}`} type="button" onClick={() => onDelete(model)}>
            <Trash2 size={12} />
          </button>
        ) : (
          <div className="model-row-btn-gap" />
        )}
      </div>

      {expanded ? (
        <div className="model-metadata-panel">
          <dl className="model-metadata-grid">
            {diarDetailFields(model, statusLabel, showStatus).map(([l, v]) => (
              <React.Fragment key={l}>
                <dt>{l}</dt>
                <dd title={String(v)}>{String(v)}</dd>
              </React.Fragment>
            ))}
          </dl>
        </div>
      ) : null}
    </div>
  );
}
