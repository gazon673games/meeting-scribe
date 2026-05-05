import { formatTime } from "../../shared/lib/format";

export function TranscriptLine({ line, speaker, tone = 0 }) {
  const body = line.tentative ? (
    <>
      {line.confirmedText || ""}
      {line.tentativeText ? <span className="transcript-tentative"> {line.tentativeText}</span> : null}
    </>
  ) : (
    line.text
  );
  return (
    <div className="transcript-line">
      <div className="speaker-block">
        <time>{formatTime(line.ts)}</time>
        <span className={`speaker-tone-${tone}`}>{speaker}</span>
      </div>
      <p>{body}</p>
    </div>
  );
}
