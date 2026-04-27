import React from "react";

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
  const listRef = React.useRef(null);
  const shouldStickToBottomRef = React.useRef(true);
  const speakerTones = new Map();

  const updateStickiness = React.useCallback(() => {
    const element = listRef.current;
    if (!element) {
      return;
    }
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    shouldStickToBottomRef.current = distanceFromBottom <= 24;
  }, []);

  React.useLayoutEffect(() => {
    const element = listRef.current;
    if (!element || !shouldStickToBottomRef.current) {
      return;
    }
    element.scrollTop = element.scrollHeight;
  }, [lines.length, running]);

  return (
    <div className="transcript-list" onScroll={updateStickiness} ref={listRef}>
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
