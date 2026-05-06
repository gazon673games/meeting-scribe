const MAX_TRANSCRIPT_LINES = 200;

export function appendTranscriptLine(state, event) {
  if (!state?.session || !String(event?.text || "").trim()) {
    return state;
  }
  const line = {
    id: String(event.id || transcriptLineId(event)),
    ts: Number(event.ts || Date.now() / 1000),
    stream: String(event.stream || "mix"),
    speaker: String(event.speaker || ""),
    t_start: optionalNumber(event.t_start),
    t_end: optionalNumber(event.t_end),
    text: String(event.text || ""),
    overload: Boolean(event.overload)
  };
  const transcript = state.session.transcript || [];
  const existingIndex = transcript.findIndex((item) => transcriptLineKey(item) === transcriptLineKey(line));
  if (existingIndex >= 0) {
    const existing = transcript[existingIndex];
    if (existing.ts === line.ts && existing.stream === line.stream && existing.speaker === line.speaker && existing.text === line.text) {
      return state;
    }
    return {
      ...state,
      session: {
        ...state.session,
        transcript: transcript.map((item, index) => (index === existingIndex ? { ...item, ...line } : item))
      }
    };
  }
  return {
    ...state,
    session: {
      ...state.session,
      transcript: [...transcript, line].slice(-MAX_TRANSCRIPT_LINES)
    }
  };
}

export function updateTranscriptLine(state, event) {
  if (!state?.session) {
    return state;
  }
  const id = String(event?.id || "");
  const transcript = state.session.transcript || [];
  let changed = false;
  const updated = transcript.map((line) => {
    if (!sameTranscriptLine(line, event, id)) {
      return line;
    }
    changed = true;
    return {
      ...line,
      speaker: String(event.speaker || line.speaker || ""),
      speakerSource: String(event.speakerSource || line.speakerSource || ""),
      speakerConfidence: event.speakerConfidence ?? line.speakerConfidence
    };
  });
  if (!changed) {
    return state;
  }
  return {
    ...state,
    session: {
      ...state.session,
      transcript: updated
    }
  };
}

export function applyStreamingWords(state, event) {
  if (!state?.session) {
    return state;
  }
  const stream = String(event.stream || "mix");
  const newConfirmed = (event.confirmed || []).map((word) => word.text).join(" ").trim();
  const tentativeText = (event.tentative || []).map((word) => word.text).join(" ").trim();
  if (!newConfirmed && !tentativeText) {
    return state;
  }

  const id = `streaming-${stream}`;
  const transcript = state.session.transcript || [];
  const existingIndex = transcript.findIndex((item) => item.id === id);
  let line;
  if (existingIndex >= 0) {
    const previousLine = transcript[existingIndex];
    const confirmedText = [String(previousLine.confirmedText || ""), newConfirmed].filter(Boolean).join(" ");
    line = {
      ...previousLine,
      confirmedText,
      tentativeText,
      t_end: optionalNumber(event.t_end),
      ts: Number(event.ts || Date.now() / 1000)
    };
  } else {
    line = {
      id,
      ts: Number(event.ts || Date.now() / 1000),
      stream,
      speaker: "",
      t_start: optionalNumber(event.t_start),
      t_end: optionalNumber(event.t_end),
      confirmedText: newConfirmed,
      tentativeText,
      text: "",
      tentative: true,
      overload: false
    };
  }

  const updatedTranscript =
    existingIndex >= 0
      ? transcript.map((item, index) => (index === existingIndex ? line : item))
      : [...transcript, line].slice(-MAX_TRANSCRIPT_LINES);
  return { ...state, session: { ...state.session, transcript: updatedTranscript } };
}

export function applyStreamingFinal(state, event) {
  if (!state?.session) {
    return state;
  }
  const stream = String(event.stream || "mix");
  const text = (event.words || [])
    .map((word) => word.text)
    .join(" ")
    .trim();
  const streamingId = `streaming-${stream}`;
  const transcript = (state.session.transcript || []).filter((item) => item.id !== streamingId);
  if (!text) {
    return { ...state, session: { ...state.session, transcript } };
  }
  const finalLine = {
    id: transcriptLineId(event),
    ts: Number(event.ts || Date.now() / 1000),
    stream,
    speaker: "",
    t_start: optionalNumber(event.t_start),
    t_end: optionalNumber(event.t_end),
    text,
    tentative: false,
    overload: false
  };
  return {
    ...state,
    session: { ...state.session, transcript: [...transcript, finalLine].slice(-MAX_TRANSCRIPT_LINES) }
  };
}

export function mergeBackendStateSnapshot(current, next) {
  if (!next) {
    return current;
  }
  if (!current) {
    return next;
  }
  const merged = { ...current, ...next };
  if (!current?.session || !next?.session) {
    return merged;
  }
  if (!current.session.running || !next.session.running) {
    return merged;
  }
  return {
    ...merged,
    session: {
      ...next.session,
      transcript: mergeTranscriptLines(current.session.transcript || [], next.session.transcript || [])
    }
  };
}

function mergeTranscriptLines(currentLines, nextLines) {
  const merged = [];
  const indexByKey = new Map();

  const appendOrUpdate = (line) => {
    if (!line || (!String(line.text || "").trim() && !line.tentative)) {
      return;
    }
    const key = transcriptLineKey(line);
    const existingIndex = indexByKey.get(key);
    if (existingIndex === undefined) {
      indexByKey.set(key, merged.length);
      merged.push(line);
      return;
    }
    merged[existingIndex] = { ...merged[existingIndex], ...line };
  };

  currentLines.forEach(appendOrUpdate);
  nextLines.forEach(appendOrUpdate);

  return merged.sort(compareTranscriptLines).slice(-MAX_TRANSCRIPT_LINES);
}

function sameTranscriptLine(line, event, id) {
  if (id && String(line.id || "") === id) {
    return true;
  }
  return (
    String(line.stream || "") === String(event?.stream || "") &&
    Number(line.t_start ?? -1) === Number(event?.t_start ?? -2) &&
    Number(line.t_end ?? -1) === Number(event?.t_end ?? -2)
  );
}

function transcriptLineId(event) {
  const stream = String(event?.stream || "mix").replace(/[^a-zA-Z0-9_-]+/g, "_") || "mix";
  const ts = Math.round(Number(event?.ts || Date.now() / 1000) * 1000);
  const start = optionalNumber(event?.t_start);
  const end = optionalNumber(event?.t_end);
  return `${stream}:${Math.round((start ?? event?.ts ?? 0) * 1000) || ts}:${Math.round((end ?? event?.ts ?? 0) * 1000) || ts}`;
}

function transcriptLineKey(line) {
  if (line?.id) {
    return String(line.id);
  }
  return [
    line?.stream || "mix",
    optionalNumber(line?.t_start) ?? "",
    optionalNumber(line?.t_end) ?? "",
    optionalNumber(line?.ts) ?? "",
    String(line?.text || "")
  ].join(":");
}

function compareTranscriptLines(left, right) {
  return (optionalNumber(left?.ts) ?? 0) - (optionalNumber(right?.ts) ?? 0);
}

function optionalNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
