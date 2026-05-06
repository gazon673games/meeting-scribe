import React from "react";
import { Check, ChevronDown, ChevronUp, Download, Trash2 } from "lucide-react";

import { formatBytes } from "../../shared/lib/format";

function diarDownloadLabel(model) {
  const dl = Number(model.downloadedBytes || 0);
  const tot = Number(model.totalBytes || 0);
  const sp = Number(model.speedBps || 0);
  const dlStr = dl > 0 ? formatBytes(dl) : "0 B";
  const totStr = tot > 0 ? ` / ${formatBytes(tot)}` : "";
  const spStr = sp > 0 ? ` - ${formatBytes(sp)}/s` : "";
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

export function DiarModelRow({
  model,
  selected,
  isDownloading,
  expanded,
  deletable,
  showStatus,
  onUse,
  onDownload,
  onDelete,
  onToggleExpand,
}) {
  const ready = Boolean(model.cached || model.compatible);
  const modelLabel = model.label || model.name;
  const statusLabel = diarStatusLabel(model, selected, ready, isDownloading);
  const statusClass = diarStatusClass(model, selected, ready, isDownloading);
  const canDownload = !isDownloading && model.downloadable !== false;

  return (
    <div className={`model-row-shell${expanded ? " expanded" : ""}`}>
      <div className="model-row">
        <span className="model-row-name" title={model.path || model.url || model.name}>
          {modelLabel}
        </span>
        <span className={`model-row-status ${statusClass}`} title={model.downloadError || model.downloadMessage || statusLabel}>
          {statusLabel}
        </span>

        <button className="model-row-btn" title={expanded ? "Hide info" : "Show info"} type="button" onClick={() => onToggleExpand(model.name)}>
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        <ModelActionButton
          canDownload={canDownload}
          isDownloading={isDownloading}
          model={model}
          modelLabel={modelLabel}
          ready={ready}
          selected={selected}
          onDownload={onDownload}
          onUse={onUse}
        />

        {deletable ? (
          <button
            className="model-row-btn danger"
            disabled={selected || isDownloading}
            title={selected ? "Selected model cannot be deleted" : `Delete ${modelLabel}`}
            type="button"
            onClick={() => onDelete(model)}
          >
            <Trash2 size={12} />
          </button>
        ) : (
          <div className="model-row-btn-gap" />
        )}
      </div>

      <ModelDetailsPanel expanded={expanded} model={model} showStatus={showStatus} statusLabel={statusLabel} />
    </div>
  );
}

function ModelActionButton({
  canDownload,
  isDownloading,
  model,
  modelLabel,
  ready,
  selected,
  onDownload,
  onUse,
}) {
  if (ready) {
    return (
      <button className="model-row-btn" disabled={selected} title={selected ? "Already selected" : `Use ${modelLabel}`} type="button" onClick={() => onUse(model)}>
        <Check size={12} />
      </button>
    );
  }
  return (
    <button
      className="model-row-btn"
      disabled={!canDownload}
      title={isDownloading ? "Downloading..." : `Download ${modelLabel}`}
      type="button"
      onClick={() => canDownload && onDownload(model)}
    >
      <Download size={12} />
    </button>
  );
}

function ModelDetailsPanel({ expanded, model, showStatus, statusLabel }) {
  if (!expanded) {
    return null;
  }
  return (
    <div className="model-metadata-panel">
      <dl className="model-metadata-grid">
        {diarDetailFields(model, statusLabel, showStatus).map(([label, value]) => (
          <React.Fragment key={label}>
            <dt>{label}</dt>
            <dd title={String(value)}>{String(value)}</dd>
          </React.Fragment>
        ))}
      </dl>
    </div>
  );
}
