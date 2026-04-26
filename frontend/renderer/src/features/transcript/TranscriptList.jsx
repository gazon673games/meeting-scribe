import { TranscriptLine } from "./TranscriptLine";
import { TypingRow } from "./TypingRow";

export function TranscriptList({ lines, running }) {
  return (
    <div className="transcript-list">
      {lines.length ? (
        lines.map((line, index) => <TranscriptLine key={`${line.ts}-${index}`} line={line} index={index} />)
      ) : (
        <div className="transcript-empty">Transcript stream</div>
      )}
      {running ? <TypingRow /> : null}
    </div>
  );
}
