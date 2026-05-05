import React from "react";
import { Check, ChevronDown, ChevronUp, Download, Trash2, X } from "lucide-react";
import { formatBytes } from "../../shared/lib/format";

function downloadStatusLabel(model) {
  const dl = Number(model.downloadedBytes || 0);
  const sp = Number(model.speedBps || 0);
  if (dl > 0 || sp > 0) {
    const dlStr = dl > 0 ? formatBytes(dl) : "0 B";
    return sp > 0 ? `${dlStr} – ${formatBytes(sp)}/s` : dlStr;
  }
  return model.downloadMessage || "downloading...";
}

export function asrStatusLabel(model, isDownloading) {
  if (model.cached) return model.compatible === false ? "incompatible" : "ready";
  if (isDownloading) return downloadStatusLabel(model);
  if (model.downloadError) return "error";
  if (model.status === "inaccessible") return "inaccessible";
  if (model.status === "unsupported_transformers_format") return "unsupported format";
  if (model.status === "missing_files") return `missing ${model.missing?.join(", ") || "files"}`;
  return model.source === "recommended" ? "downloadable" : "not ready";
}

export function asrStatusClass(model, compatible, isDownloading) {
  if (model.cached) return compatible ? "model-status-cached" : "model-status-error";
  if (isDownloading) return "model-status-downloading";
  if (model.downloadError) return "model-status-error";
  if (model.status === "missing_files" || model.status === "unsupported_transformers_format" || model.status === "inaccessible") return "model-status-error";
  return "model-status-missing";
}

function MetadataObject({ title, value }) {
  if (!value || Object.keys(value).length === 0) return null;
  return (
    <section className="model-metadata-object">
      <h4>{title}</h4>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}

export function ModelMetadataPanel({ error, loading, metadata }) {
  const rows = metadata ? metadataRows(metadata) : [];
  return (
    <div className="model-metadata-panel">
      {loading ? <div className="models-loading">Loading metadata...</div> : null}
      {error ? <div className="models-error">{error}</div> : null}
      {!loading && metadata ? (
        <>
          <dl className="model-metadata-grid">
            {rows.map(([label, value]) => (
              <React.Fragment key={label}>
                <dt>{label}</dt>
                <dd title={String(value)}>{String(value)}</dd>
              </React.Fragment>
            ))}
          </dl>
          <div className="model-metadata-blocks">
            <MetadataObject title="Config" value={metadata.config} />
            <MetadataObject title="Preprocessor" value={metadata.preprocessor} />
            <MetadataObject title="Tokenizer" value={metadata.tokenizer} />
            <MetadataObject title="Model Card" value={metadata.readme} />
          </div>
        </>
      ) : null}
    </div>
  );
}

function metadataRows(m) {
  return [
    ["Name", m.name || "-"],
    ["Normalized", m.normalizedName || "-"],
    ["Source", m.source || "-"],
    ["Status", m.status || "-"],
    ["Format", m.format || "-"],
    ["Compatible", m.compatible ? "yes" : "no"],
    ["Builtin", m.builtin ? "yes" : "no"],
    ["Size", formatBytes(m.totalBytes)],
    ["Repo", m.repoId || "-"],
    ["Missing", m.missing?.length ? m.missing.join(", ") : "-"],
    ["Files", m.presentFiles?.length ? m.presentFiles.join(", ") : "-"],
    ["Weights", weightFilesLabel(m.weightFiles)],
    ["Warnings", m.warnings?.length ? m.warnings.join(", ") : "-"],
    ["Cache", m.cachePath || "-"],
    ["Resolved", m.resolvedPath || "-"],
  ];
}

function weightFilesLabel(files) {
  if (!Array.isArray(files) || files.length === 0) return "-";
  return files.map((f) => `${f.name} (${formatBytes(f.bytes)})`).join(", ");
}

export function AsrModelRow({ model, isDownloading, isSelected, isExternal, metaEntry, onDownload, onDelete, onRemoveEntry, onToggleMeta, onChange }) {
  const compatible = Boolean(model.compatible || model.cached);
  const statusLabel = asrStatusLabel(model, isDownloading);
  const statusClass = asrStatusClass(model, compatible, isDownloading);
  const expanded = Boolean(metaEntry);

  return (
    <div className={`model-row-shell${expanded ? " expanded" : ""}`}>
      <div className="model-row">
        <span className="model-row-name" title={model.path || model.name}>{model.label || model.name}</span>
        <span className={`model-row-status ${statusClass}`} title={model.downloadError || model.downloadMessage || model.warnings?.join(", ") || statusLabel}>
          {statusLabel}
        </span>

        <button className="model-row-btn" title={expanded ? "Hide metadata" : "Show metadata"} type="button" onClick={() => onToggleMeta(model)}>
          {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        {compatible ? (
          <button className="model-row-btn" disabled={isSelected} title={isSelected ? "Already selected" : `Use ${model.name}`} type="button" onClick={() => onChange({ model: model.name })}>
            <Check size={12} />
          </button>
        ) : (
          <button className="model-row-btn" disabled={isDownloading || model.downloadable === false} title={isDownloading ? "Downloading…" : `Download ${model.name}`} type="button" onClick={() => !isDownloading && model.downloadable !== false && onDownload(model.name)}>
            <Download size={12} />
          </button>
        )}

        {(model.deletable || model.cached) ? (
          <button className="model-row-btn danger" disabled={isSelected || isDownloading} title={isSelected ? "Selected model cannot be deleted" : `Delete ${model.name}`} type="button" onClick={() => onDelete(model.name)}>
            <Trash2 size={12} />
          </button>
        ) : (
          <div className="model-row-btn-gap" />
        )}

        {isExternal ? (
          <button className="model-row-btn danger" disabled={isSelected} title="Remove from list" type="button" onClick={() => onRemoveEntry(model.name)}>
            <X size={12} />
          </button>
        ) : null}
      </div>
      {expanded ? <ModelMetadataPanel loading={metaEntry.loading} metadata={metaEntry.metadata} error={metaEntry.error} /> : null}
    </div>
  );
}
