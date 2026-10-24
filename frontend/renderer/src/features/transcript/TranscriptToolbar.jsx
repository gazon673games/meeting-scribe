import React from 'react';

export const TranscriptToolbar = ({ lines, session, onClear }) => {
  const [confirming, setConfirming] = React.useState(false);
  const path = session.realtimeTranscriptPath || session.humanLogPath;
  const savedLabel = path ? 'Saved transcript' : 'Session transcript';
  const label = confirming ? 'Clear this transcript?' : savedLabel;
  React.useEffect(() => {
    if (!lines.length) setConfirming(false);
  }, [lines.length]);
  return (
    <div className="transcript-tools">
      <span title={path || 'Transcript is held in this session.'}>{label}</span>
      <div className="transcript-tool-actions">
        {confirming ? (
          <button onClick={() => setConfirming(false)} type="button">
            Cancel
          </button>
        ) : null}
        <button
          disabled={!lines.length}
          onClick={() => {
            if (!confirming) {
              setConfirming(true);
              return;
            }
            setConfirming(false);
            onClear();
          }}
          type="button"
        >
          {confirming ? 'Confirm clear' : 'Clear transcript'}
        </button>
      </div>
    </div>
  );
};
