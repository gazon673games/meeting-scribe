import { formatTime } from "../../shared/lib/format";

export function TranscriptLine({ line, speaker, tone = 0 }) {
  return (
    <div className="transcript-line">
      <div className="speaker-block">
        <time>{formatTime(line.ts)}</time>
        <span className={`speaker-tone-${tone}`}>{speaker}</span>
      </div>
      <p>{line.text}</p>
    </div>
  );
}
