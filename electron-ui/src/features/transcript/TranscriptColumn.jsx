import { Activity } from "lucide-react";

import { formatTime } from "../../shared/lib/format";
import { PipelineColumn } from "../../shared/ui/PipelineColumn";

export function TranscriptColumn({ lines, session, status, onClear }) {
  const running = status === "recording";
  return (
    <PipelineColumn title="Live Transcript" active={running} className="transcript-column" rightMeta={<LatencyMeta session={session} />}>
      <div className="transcript-tools">
        <span>{session.realtimeTranscriptPath || session.humanLogPath || "memory"}</span>
        <button disabled={!lines.length} onClick={onClear}>
          Clear
        </button>
      </div>

      <div className="transcript-list">
        {lines.length ? (
          lines.map((line, index) => <TranscriptLine key={`${line.ts}-${index}`} line={line} index={index} />)
        ) : (
          <div className="transcript-empty">Transcript stream</div>
        )}
        {running ? (
          <div className="typing-row">
            <div>
              <time>live</time>
              <span>ASR</span>
            </div>
            <p>
              <i />
              <i />
              <i />
            </p>
          </div>
        ) : null}
      </div>
    </PipelineColumn>
  );
}

function LatencyMeta({ session }) {
  const text = session.asrRunning ? "live" : "standby";
  return (
    <span className="column-meta">
      <Activity size={13} />
      {text}
    </span>
  );
}

function TranscriptLine({ line, index }) {
  const speaker = line.stream || "mix";
  return (
    <div className="transcript-line">
      <div className="speaker-block">
        <time>{formatTime(line.ts)}</time>
        <span className={index % 2 ? "speaker-two" : ""}>{speaker}</span>
      </div>
      <p>{line.text}</p>
    </div>
  );
}
