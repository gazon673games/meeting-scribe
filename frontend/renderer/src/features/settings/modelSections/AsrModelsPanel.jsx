import React from 'react';
import { Download } from 'lucide-react';

import { meetingScribeClient } from '../../../shared/api/meetingScribeClient';
import { Field } from '../../../shared/ui/Field';
import { InnerCollapsible } from '../../../shared/ui/InnerCollapsible';
import { AsrModelRow } from '../AsrModelRow';
import { formatError, runWithPending, useModelList } from './useModelList';

export const AsrModelsPanel = ({
  draft,
  locked = false,
  onChange,
  open,
  asrDir,
  useProxy,
  downloadProxy,
  refreshToken,
}) => {
  const [downloading, setDownloading] = React.useState(new Set());
  const [customUrl, setCustomUrl] = React.useState('');
  const [metadata, setMetadata] = React.useState(() => new Map());
  const metadataOwner = React.useRef({
    active: false,
    directory: null,
    requests: new Map(),
  });
  React.useEffect(() => {
    const owner = { active: open, directory: asrDir, requests: new Map() };
    metadataOwner.current = owner;
    setMetadata(new Map());
    return () => {
      owner.active = false;
    };
  }, [asrDir, open]);

  const asrList = useModelList({
    open,
    refreshToken,
    loadFn: React.useCallback(
      () => meetingScribeClient.request('list_models', { ...asrDir }),
      [asrDir],
    ),
  });

  const handleDownload = React.useCallback(
    async (name) => {
      asrList.setError('');
      try {
        await runWithPending(setDownloading, name, async () => {
          await meetingScribeClient.request('download_model', {
            name,
            ...asrDir,
            useProxy,
            proxy: downloadProxy,
          });
          asrList.load();
        });
      } catch (requestError) {
        asrList.setError(formatError(requestError));
      }
    },
    [asrDir, asrList, downloadProxy, useProxy],
  );

  const handleDelete = React.useCallback(
    async (name) => {
      asrList.setError('');
      try {
        await meetingScribeClient.request('delete_model', {
          name,
          ...asrDir,
        });
        asrList.load();
      } catch (requestError) {
        asrList.setError(formatError(requestError));
      }
    },
    [asrDir, asrList],
  );

  const handleRemoveEntry = React.useCallback(
    async (name) => {
      asrList.setError('');
      try {
        await meetingScribeClient.request('remove_model_entry', {
          name,
          ...asrDir,
        });
        asrList.load();
      } catch (requestError) {
        asrList.setError(formatError(requestError));
      }
    },
    [asrDir, asrList],
  );

  const handleToggleMeta = React.useCallback(
    async (model) => {
      const key = model.name;
      const owner = metadataOwner.current;
      if (!owner.active || owner.directory !== asrDir) return;
      if (owner.requests.has(key)) {
        owner.requests.delete(key);
        setMetadata((current) => {
          const next = new Map(current);
          next.delete(key);
          return next;
        });
        return;
      }
      const request = {};
      owner.requests.set(key, request);

      asrList.setError('');
      setMetadata((current) =>
        new Map(current).set(key, {
          loading: true,
          metadata: null,
          error: '',
        }),
      );

      try {
        const details = await meetingScribeClient.request('model_metadata', {
          name: model.name,
          ...asrDir,
        });
        if (!owner.active || owner.requests.get(key) !== request) return;
        setMetadata((current) => {
          if (!current.has(key)) return current;
          return new Map(current).set(key, {
            loading: false,
            metadata: details,
            error: '',
          });
        });
      } catch (requestError) {
        if (!owner.active || owner.requests.get(key) !== request) return;
        setMetadata((current) => {
          if (!current.has(key)) return current;
          return new Map(current).set(key, {
            loading: false,
            metadata: null,
            error: formatError(requestError),
          });
        });
      }
    },
    [asrDir, asrList],
  );

  return (
    <InnerCollapsible title="Transcription">
      <div className="inner-block-controls">
        <Field label="Subfolder">
          <input
            spellCheck={false}
            value={draft.asrModelsSubdir || ''}
            placeholder="e.g. asr"
            onChange={(event) =>
              onChange({ asrModelsSubdir: event.target.value })
            }
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
              handleDownload(customUrl.trim());
              setCustomUrl('');
            }}
          >
            <Download size={13} />
            Download
          </button>
        </div>
      </div>

      {asrList.error ? (
        <div className="models-error">{asrList.error}</div>
      ) : null}
      {asrList.models === null ? (
        <div className="models-loading">Loading...</div>
      ) : (
        <div className="models-list">
          {asrList.models.map((model) => (
            <AsrModelRow
              selectionDisabled={locked}
              key={model.name}
              model={model}
              isDownloading={model.downloading || downloading.has(model.name)}
              isSelected={draft.model === model.name}
              isExternal={!model.builtin && model.source !== 'recommended'}
              metaEntry={metadata.get(model.name) || null}
              onDownload={handleDownload}
              onDelete={handleDelete}
              onRemoveEntry={handleRemoveEntry}
              onToggleMeta={handleToggleMeta}
              onChange={onChange}
            />
          ))}
        </div>
      )}
    </InnerCollapsible>
  );
};
