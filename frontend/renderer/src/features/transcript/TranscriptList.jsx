import React from 'react';

import { TranscriptLine } from './TranscriptLine';
import { TypingRow } from './TypingRow';

const SPEAKER_TONE_COUNT = 10;

const speakerLabel = (line) => {
  const speaker = String(line.speaker ?? '').trim();
  const stream = String(line.stream ?? '').trim();
  return speaker && speaker !== 'S?' ? speaker : stream || 'mix';
};

const speakerKey = (line) => speakerLabel(line).toLocaleLowerCase();

const resolveSpeakerTone = (line) => {
  const key = speakerKey(line);
  let hash = 0;
  for (const character of key) {
    hash = (Math.imul(hash, 31) + character.codePointAt(0)) >>> 0;
  }
  return hash % SPEAKER_TONE_COUNT;
};

export const TranscriptList = ({ lines, running }) => {
  const listRef = React.useRef(null);
  const shouldStickToBottomRef = React.useRef(true);
  const isProgrammaticScrollRef = React.useRef(false);

  const updateStickiness = React.useCallback(() => {
    if (isProgrammaticScrollRef.current) {
      isProgrammaticScrollRef.current = false;
      return;
    }
    const element = listRef.current;
    if (!element) {
      return;
    }
    const distanceFromBottom =
      element.scrollHeight - element.scrollTop - element.clientHeight;
    shouldStickToBottomRef.current = distanceFromBottom <= 80;
  }, []);

  const lastLine = lines.at(-1);
  const lastLineTs = lastLine?.ts;
  const lastLineText = lastLine?.text;

  React.useLayoutEffect(() => {
    const element = listRef.current;
    if (!element || !shouldStickToBottomRef.current) {
      return;
    }
    isProgrammaticScrollRef.current = true;
    element.scrollTop = element.scrollHeight;
  }, [lastLineTs, lastLineText, lines.length, running]);

  return (
    <div className="transcript-list" onScroll={updateStickiness} ref={listRef}>
      {lines.map((line, index) => (
        <TranscriptLine
          key={line.id || `${line.ts}-${index}`}
          line={line}
          speaker={speakerLabel(line)}
          tone={resolveSpeakerTone(line)}
        />
      ))}
      {!lines.length && !running ? (
        <div className="transcript-empty">
          <span>Transcript stream</span>
          <strong>Waiting for speech</strong>
          <p>
            Enable an audio source and press Start. Recognized speech will
            appear here.
          </p>
        </div>
      ) : null}
      {running ? <TypingRow /> : null}
    </div>
  );
};
