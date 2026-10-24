import {
  Check,
  ChevronDown,
  ChevronUp,
  Download,
  Trash2,
  X,
} from 'lucide-react';

import { formatBytes } from '../../shared/lib/format';
import { ModelMetadataGrid } from './ModelMetadataGrid';
import { formatDownloadProgress } from './modelDownloadProgress';

const downloadStatusLabel = (model) =>
  formatDownloadProgress(model, {
    fallback: model.downloadMessage || 'downloading...',
    includeTotal: false,
  });

export const asrStatusLabel = (model, isDownloading) => {
  if (model.cached) {
    return model.compatible === false ? 'incompatible' : 'ready';
  }
  if (isDownloading) return downloadStatusLabel(model);
  if (model.downloadError) return 'error';
  if (model.status === 'inaccessible') return 'inaccessible';
  if (model.status === 'unsupported_transformers_format') {
    return 'unsupported format';
  }
  if (model.status === 'missing_files') {
    return `missing ${model.missing?.join(', ') || 'files'}`;
  }
  return model.source === 'recommended' ? 'downloadable' : 'not ready';
};

export const asrStatusClass = (model, compatible, isDownloading) => {
  if (model.cached) {
    return compatible ? 'model-status-cached' : 'model-status-error';
  }
  if (isDownloading) return 'model-status-downloading';
  if (model.downloadError) return 'model-status-error';
  if (
    model.status === 'missing_files' ||
    model.status === 'unsupported_transformers_format' ||
    model.status === 'inaccessible'
  ) {
    return 'model-status-error';
  }
  return 'model-status-missing';
};

const MetadataObject = ({ title, value }) => {
  if (!value || Object.keys(value).length === 0) return null;
  return (
    <section className="model-metadata-object">
      <h4>{title}</h4>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
};

const weightFilesLabel = (files) => {
  if (!Array.isArray(files) || files.length === 0) return '-';
  return files
    .map((file) => `${file.name} (${formatBytes(file.bytes)})`)
    .join(', ');
};

const metadataRows = (metadata) => [
  ['Name', metadata.name || '-'],
  ['Normalized', metadata.normalizedName || '-'],
  ['Source', metadata.source || '-'],
  ['Status', metadata.status || '-'],
  ['Format', metadata.format || '-'],
  ['Compatible', metadata.compatible ? 'yes' : 'no'],
  ['Builtin', metadata.builtin ? 'yes' : 'no'],
  ['Size', formatBytes(metadata.totalBytes)],
  ['Repo', metadata.repoId || '-'],
  ['Missing', metadata.missing?.length ? metadata.missing.join(', ') : '-'],
  [
    'Files',
    metadata.presentFiles?.length ? metadata.presentFiles.join(', ') : '-',
  ],
  ['Weights', weightFilesLabel(metadata.weightFiles)],
  ['Warnings', metadata.warnings?.length ? metadata.warnings.join(', ') : '-'],
  ['Cache', metadata.cachePath || '-'],
  ['Resolved', metadata.resolvedPath || '-'],
];

export const ModelMetadataPanel = ({ error, loading, metadata }) => {
  const rows = metadata ? metadataRows(metadata) : [];
  return (
    <div className="model-metadata-panel">
      {loading ? (
        <div className="models-loading">Loading metadata...</div>
      ) : null}
      {error ? <div className="models-error">{error}</div> : null}
      {!loading && metadata ? (
        <>
          <ModelMetadataGrid rows={rows} />
          <div className="model-metadata-blocks">
            <MetadataObject title="Config" value={metadata.config} />
            <MetadataObject
              title="Preprocessor"
              value={metadata.preprocessor}
            />
            <MetadataObject title="Tokenizer" value={metadata.tokenizer} />
            <MetadataObject title="Model Card" value={metadata.readme} />
          </div>
        </>
      ) : null}
    </div>
  );
};

const asrStatusTitle = (model, statusLabel) => {
  if (model.downloadError) {
    return model.downloadError;
  }
  if (model.downloadMessage) {
    return model.downloadMessage;
  }
  const warning = Array.isArray(model.warnings)
    ? model.warnings.join(', ')
    : '';
  return warning || statusLabel;
};

const buildAsrRowState = (model, isDownloading, isSelected, metaEntry) => {
  const compatible = Boolean(model.compatible ?? model.cached);
  const statusLabel = asrStatusLabel(model, isDownloading);
  return {
    compatible,
    expanded: Boolean(metaEntry),
    label: model.label || model.name,
    nameTitle: model.path || model.name,
    statusClass: asrStatusClass(model, compatible, isDownloading),
    statusLabel,
    statusTitle: asrStatusTitle(model, statusLabel),
    canDownload: !isDownloading && model.downloadable !== false,
    canDelete: Boolean(model.deletable || model.cached),
    isSelected: Boolean(isSelected),
  };
};

const DeleteModelButton = ({
  canDelete,
  isDownloading,
  isSelected,
  model,
  onDelete,
}) => {
  if (!canDelete) {
    return <div className="model-row-btn-gap" />;
  }
  return (
    <button
      className="model-row-btn danger"
      disabled={isSelected || isDownloading}
      title={
        isSelected ? 'Selected model cannot be deleted' : `Delete ${model.name}`
      }
      type="button"
      onClick={() => onDelete(model.name)}
    >
      <Trash2 size={12} />
    </button>
  );
};

const AsrActionButton = ({
  selectionDisabled,
  canDownload,
  compatible,
  isDownloading,
  isSelected,
  model,
  onChange,
  onDownload,
}) => {
  if (compatible) {
    return (
      <button
        className="model-row-btn"
        disabled={isSelected || selectionDisabled}
        title={isSelected ? 'Already selected' : `Use ${model.name}`}
        type="button"
        onClick={() => onChange({ model: model.name })}
      >
        <Check size={12} />
      </button>
    );
  }
  return (
    <button
      className="model-row-btn"
      disabled={!canDownload}
      title={isDownloading ? 'Downloading...' : `Download ${model.name}`}
      type="button"
      onClick={() => canDownload && onDownload(model.name)}
    >
      <Download size={12} />
    </button>
  );
};

export const AsrModelRow = ({
  selectionDisabled = false,
  model,
  isDownloading,
  isSelected,
  isExternal,
  metaEntry,
  onDownload,
  onDelete,
  onRemoveEntry,
  onToggleMeta,
  onChange,
}) => {
  const row = buildAsrRowState(model, isDownloading, isSelected, metaEntry);

  return (
    <div className={`model-row-shell${row.expanded ? ' expanded' : ''}`}>
      <div className="model-row">
        <span className="model-row-name" title={row.nameTitle}>
          {row.label}
        </span>
        <span
          className={`model-row-status ${row.statusClass}`}
          title={row.statusTitle}
        >
          {row.statusLabel}
        </span>

        <button
          className="model-row-btn"
          title={row.expanded ? 'Hide metadata' : 'Show metadata'}
          type="button"
          onClick={() => onToggleMeta(model)}
        >
          {row.expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        <AsrActionButton
          selectionDisabled={selectionDisabled}
          canDownload={row.canDownload}
          compatible={row.compatible}
          isDownloading={isDownloading}
          isSelected={row.isSelected}
          model={model}
          onChange={onChange}
          onDownload={onDownload}
        />

        <DeleteModelButton
          canDelete={row.canDelete}
          isDownloading={isDownloading}
          isSelected={row.isSelected}
          model={model}
          onDelete={onDelete}
        />

        {isExternal ? (
          <button
            className="model-row-btn danger"
            disabled={isSelected}
            title="Remove from list"
            type="button"
            onClick={() => onRemoveEntry(model.name)}
          >
            <X size={12} />
          </button>
        ) : null}
      </div>
      {row.expanded ? (
        <ModelMetadataPanel
          loading={metaEntry.loading}
          metadata={metaEntry.metadata}
          error={metaEntry.error}
        />
      ) : null}
    </div>
  );
};
