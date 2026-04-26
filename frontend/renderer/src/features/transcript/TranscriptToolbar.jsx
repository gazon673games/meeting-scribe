export function TranscriptToolbar({ lines, session, onClear }) {
  return (
    <div className="transcript-tools">
      <span>{session.realtimeTranscriptPath || session.humanLogPath || "memory"}</span>
      <button disabled={!lines.length} onClick={onClear}>
        Clear
      </button>
    </div>
  );
}
