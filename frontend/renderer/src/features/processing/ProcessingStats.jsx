import { formatBytes, formatNumber } from '../../shared/lib/format';
import { StatLine } from '../../shared/ui/StatLine';

const formatSpeed = (bps) => {
  if (!bps) return '';
  const mbps = bps / (1024 * 1024);
  return ` · ${mbps.toFixed(1)} MB/s`;
};

export const ProcessingStats = ({ asrMetrics, draft, session, summary }) => {
  const downloading = session.state === 'downloading_model';
  const dl = session.modelDownload || {};
  const hasAsrDrops =
    Number(asrMetrics.segDroppedTotal || 0) > 0 ||
    Number(asrMetrics.segSkippedTotal || 0) > 0;
  const hasAudioDrops =
    Number(session.drops?.droppedOutBlocks || 0) > 0 ||
    Number(session.drops?.droppedTapBlocks || 0) > 0;
  const asrDropSummary = [
    asrMetrics.segDroppedTotal ?? 0,
    asrMetrics.segSkippedTotal ?? 0,
  ].join('/');
  const audioDropSummary = [
    session.drops?.droppedOutBlocks ?? 0,
    session.drops?.droppedTapBlocks ?? 0,
  ].join('/');
  const progressClass =
    'model-download-bar-fill model-download-bar-indeterminate';
  return (
    <div className="settings-strip">
      <StatLine
        label="Active Model"
        value={summary.model || draft.model || '-'}
      />
      {downloading ? (
        <div className="model-download-progress">
          <span className="model-download-label">Downloading model</span>
          <span className="model-download-bytes">
            {formatBytes(dl.downloadedBytes)}
            {formatSpeed(dl.speedBps)}
          </span>
          <div className="model-download-bar">
            <div className={progressClass} />
          </div>
        </div>
      ) : (
        <>
          <StatLine
            label="ASR Latency"
            value={`${formatNumber(asrMetrics.avgLatencyS)}s`}
            accent="green"
          />
          <StatLine
            label="ASR Drops"
            value={asrDropSummary}
            accent={hasAsrDrops ? 'warn' : ''}
          />
          <StatLine label="ASR Device" value={asrMetrics.device || '-'} />
          <StatLine
            label="Queue Wait"
            value={`${formatNumber(asrMetrics.queueWaitS)}s`}
          />
          <StatLine label="RTF" value={formatNumber(asrMetrics.rtf)} />
          <StatLine
            label="Audio Drops"
            value={audioDropSummary}
            accent={hasAudioDrops ? 'warn' : ''}
          />
        </>
      )}
    </div>
  );
};
