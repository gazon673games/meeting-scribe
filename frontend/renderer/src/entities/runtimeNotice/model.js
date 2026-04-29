const MAX_MESSAGE_LENGTH = 360;

export function noticeFromBackendEvent(event) {
  if (!event || typeof event !== "object") {
    return null;
  }

  if (event.type === "error") {
    return noticeFromAsrError(event);
  }
  if (event.type === "source_error") {
    return makeNotice({
      key: noticeKey(event.type, event.source, event.error || event.message),
      title: "Audio Source",
      message: joinParts([event.source, event.error || event.message]),
      severity: "error"
    });
  }
  if (event.type === "session_error" || event.type === "asr_start_attempt_failed" || event.type === "asr_stop_error") {
    return makeNotice({
      key: noticeKey(event.type, event.message),
      title: "Runtime",
      message: event.message,
      severity: "error"
    });
  }
  if (event.type === "offline_pass_error") {
    return makeNotice({
      key: noticeKey(event.type, event.error),
      title: "Offline Pass",
      message: event.error,
      severity: "error"
    });
  }
  if (event.type === "diarization_model_download_updated" && String(event.state || "") === "error") {
    return makeNotice({
      key: noticeKey(event.type, event.model, event.error),
      title: "Speaker ID Models",
      message: event.error,
      detail: event.model ? `Model: ${event.model}` : "",
      severity: "error"
    });
  }
  if (event.type === "backend_stderr") {
    return noticeFromBackendStderr(event);
  }
  if (event.type === "backend_exit") {
    return makeNotice({
      key: noticeKey(event.type, event.code, event.signal),
      title: "Backend",
      message: `Python backend stopped (${event.code ?? "null"}, ${event.signal ?? "null"})`,
      severity: "error"
    });
  }
  return null;
}

export function upsertRuntimeNotice(current, notice, maxCount = 4) {
  if (!notice) {
    return current;
  }
  const existing = current.find((item) => item.key === notice.key);
  if (!existing) {
    return [notice, ...current].slice(0, maxCount);
  }
  return [
    { ...existing, ...notice, count: Number(existing.count || 1) + 1, ts: Date.now() },
    ...current.filter((item) => item.key !== notice.key)
  ].slice(0, maxCount);
}

function noticeFromAsrError(event) {
  const where = String(event.where || "").trim();
  const component = String(event.component || "asr").trim();
  const message = String(event.error || "").trim();
  if (!message) {
    return null;
  }
  const speakerIdError = where.startsWith("diar") || component.includes("diar") || message.toLowerCase().includes("diarization");
  return makeNotice({
    key: noticeKey(event.type, component, where, message),
    title: speakerIdError ? "Speaker ID" : "ASR",
    message,
    detail: where ? `Where: ${where}` : "",
    severity: "error"
  });
}

function noticeFromBackendStderr(event) {
  const text = String(event.text || "").trim();
  if (!text || !looksActionable(text)) {
    return null;
  }
  return makeNotice({
    key: noticeKey(event.type, text),
    title: "Backend",
    message: text,
    severity: "error"
  });
}

function makeNotice({ key, title, message, detail = "", severity = "warning" }) {
  const cleanMessage = compactText(message);
  if (!cleanMessage) {
    return null;
  }
  return {
    key,
    title: String(title || "Runtime"),
    message: truncate(cleanMessage),
    detail: truncate(compactText(detail), 160),
    severity,
    count: 1,
    ts: Date.now()
  };
}

function looksActionable(text) {
  return /error|exception|traceback|failed|requires|missing|not found|cannot/i.test(text);
}

function noticeKey(...parts) {
  return parts.map((part) => compactText(part).slice(0, 180)).filter(Boolean).join(":") || "runtime";
}

function joinParts(parts) {
  return parts.map(compactText).filter(Boolean).join(": ");
}

function compactText(value) {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function truncate(value, limit = MAX_MESSAGE_LENGTH) {
  const text = String(value || "");
  return text.length > limit ? `${text.slice(0, limit - 3)}...` : text;
}
