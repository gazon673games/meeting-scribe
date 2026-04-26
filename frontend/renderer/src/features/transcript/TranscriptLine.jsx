import { formatTime } from "../../shared/lib/format";

export function TranscriptLine({ line, index }) {
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
