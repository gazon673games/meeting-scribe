import { PipelinePanel } from "../../shared/ui/pipeline/PipelinePanel";
import { LatencyMeta } from "./LatencyMeta";
import { TranscriptList } from "./TranscriptList";
import { TranscriptToolbar } from "./TranscriptToolbar";

export function TranscriptColumn({ headerProps, layoutControls, lines, session, status, onClear }) {
  const running = status === "recording";
  return (
    <PipelinePanel
      title="Live Transcript"
      active={running}
      className="transcript-column"
      headerControls={layoutControls}
      headerProps={headerProps}
      rightMeta={<LatencyMeta session={session} />}
    >
      <TranscriptToolbar lines={lines} session={session} onClear={onClear} />
      <TranscriptList lines={lines} running={running} />
    </PipelinePanel>
  );
}
