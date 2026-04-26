import { formatNumber } from "../../shared/lib/format";
import { StatLine } from "../../shared/ui/StatLine";

export function ProcessingStats({ asrMetrics, draft, session, summary }) {
  return (
    <div className="settings-strip">
      <StatLine label="Active Model" value={summary.model || draft.model || "-"} />
      <StatLine label="ASR Latency" value={`${formatNumber(asrMetrics.avgLatencyS)}s`} accent="green" />
      <StatLine label="ASR Drops" value={`${asrMetrics.segDroppedTotal ?? 0}/${asrMetrics.segSkippedTotal ?? 0}`} accent="blue" />
      <StatLine label="Audio Drops" value={`${session.drops?.droppedOutBlocks ?? 0}/${session.drops?.droppedTapBlocks ?? 0}`} />
    </div>
  );
}
