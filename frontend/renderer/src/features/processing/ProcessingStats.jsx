import { formatNumber } from "../../shared/lib/format";
import { StatLine } from "../../shared/ui/StatLine";

function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  const mb = bytes / (1024 * 1024);
  return mb >= 1000 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

function formatSpeed(bps) {
  if (!bps) return "";
  const mbps = bps / (1024 * 1024);
  return ` · ${mbps.toFixed(1)} MB/s`;
}

export function ProcessingStats({ asrMetrics, draft, session, summary }) {
  const downloading = session.state === "downloading_model";
  const dl = session.modelDownload || {};
  return (
    <div className="settings-strip">
      <StatLine label="Active Model" value={summary.model || draft.model || "-"} />
      {downloading ? (
        <div className="model-download-progress">
          <span className="model-download-label">Downloading model</span>
          <span className="model-download-bytes">
            {formatBytes(dl.downloadedBytes)}{formatSpeed(dl.speedBps)}
          </span>
          <div className="model-download-bar">
            <div className="model-download-bar-fill model-download-bar-indeterminate" />
          </div>
        </div>
      ) : (
        <>
          <StatLine label="ASR Latency" value={`${formatNumber(asrMetrics.avgLatencyS)}s`} accent="green" />
          <StatLine label="ASR Drops" value={`${asrMetrics.segDroppedTotal ?? 0}/${asrMetrics.segSkippedTotal ?? 0}`} accent="blue" />
          <StatLine label="Audio Drops" value={`${session.drops?.droppedOutBlocks ?? 0}/${session.drops?.droppedTapBlocks ?? 0}`} />
        </>
      )}
    </div>
  );
}
