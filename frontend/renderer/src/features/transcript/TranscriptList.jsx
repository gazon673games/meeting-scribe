import { TranscriptLine } from "./TranscriptLine";
import { TypingRow } from "./TypingRow";

const SPEAKER_TONE_COUNT = 10;

function speakerLabel(line) {
  const speaker = String(line.speaker ?? "").trim();
  const stream = String(line.stream ?? "").trim();
  return speaker || stream || "mix";
}

function speakerKey(line) {
  return speakerLabel(line).toLocaleLowerCase();
}

function resolveSpeakerTone(line, tones) {
  const key = speakerKey(line);
  if (!tones.has(key)) {
    tones.set(key, tones.size % SPEAKER_TONE_COUNT);
  }
  return tones.get(key);
}

export function TranscriptList({ lines, running }) {
  const speakerTones = new Map();

  return (
    <div className="transcript-list">
      {lines.map((line, index) => (
        <TranscriptLine
          key={`${line.ts}-${index}`}
          line={line}
          speaker={speakerLabel(line)}
          tone={resolveSpeakerTone(line, speakerTones)}
        />
      ))}
      {!lines.length && !running ? (
        <div className="transcript-empty">Transcript stream</div>
      ) : null}
      {running ? <TypingRow /> : null}
    </div>
  );
}
