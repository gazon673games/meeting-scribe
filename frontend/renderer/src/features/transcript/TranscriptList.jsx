import React from "react";

import { TranscriptLine } from "./TranscriptLine";
import { TypingRow } from "./TypingRow";

const SPEAKER_TONE_COUNT = 10;

function speakerLabel(line) {
  const speaker = String(line.speaker ?? "").trim();
  const stream = String(line.stream ?? "").trim();
  return speaker && speaker !== "S?" ? speaker : stream || "mix";
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
  const isProgrammaticScrollRef = React.useRef(false);
  const speakerTones = new Map();

  const updateStickiness = React.useCallback(() => {
    if (isProgrammaticScrollRef.current) {
      isProgrammaticScrollRef.current = false;
      return;
    }
    const element = listRef.current;
    if (!element) {
      return;
    }
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
    shouldStickToBottomRef.current = distanceFromBottom <= 80;
  }, []);

  const lastLineTs = lines.length > 0 ? lines[lines.length - 1]?.ts : 0;

  React.useLayoutEffect(() => {
    const element = listRef.current;
    if (!element || !shouldStickToBottomRef.current) {
      return;
    }
    isProgrammaticScrollRef.current = true;
    element.scrollTop = element.scrollHeight;
  }, [lastLineTs, running]);

  return (
    <div className="transcript-list" onScroll={updateStickiness} ref={listRef}>
      {lines.map((line, index) => (
        <TranscriptLine
          key={line.id || `${line.ts}-${index}`}
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
