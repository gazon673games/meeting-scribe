import { noticeFromBackendEvent, upsertRuntimeNotice } from "../../entities/runtimeNotice/model";
import { REFRESHING_EVENTS } from "./constants";
import { applyAsrMetrics, removeSessionSource, upsertSessionSource } from "./sessionState";
import {
  appendTranscriptLine,
  applyStreamingFinal,
  applyStreamingWords,
  updateTranscriptLine
} from "./transcriptState";

const MAX_VISIBLE_EVENTS = 8;

export function handleBackendEvent(event, handlers) {
  handlers.setEvents((current) => [event, ...current].slice(0, MAX_VISIBLE_EVENTS));

  const runtimeNotice = noticeFromBackendEvent(event);
  if (runtimeNotice) {
    handlers.setRuntimeNotices((current) => upsertRuntimeNotice(current, runtimeNotice));
  }

  applyBackendEventState(event, handlers);

  if (REFRESHING_EVENTS.includes(event.type)) {
    handlers.refresh({ includeDevices: false, showLoading: false });
  }
}

function applyBackendEventState(event, handlers) {
  switch (event.type) {
    case "transcript_line":
      handlers.setState((current) => appendTranscriptLine(current, event));
      break;
    case "transcript_line_update":
      handlers.setState((current) => updateTranscriptLine(current, event));
      break;
    case "asr_metrics":
      handlers.setState((current) => applyAsrMetrics(current, event));
      break;
    case "streaming_words":
      handlers.setState((current) => applyStreamingWords(current, event));
      break;
    case "streaming_final":
      handlers.setState((current) => applyStreamingFinal(current, event));
      break;
    case "assistant_ping_result":
      handlers.setAssistantPing((current) => ({ ...current, busy: false, ...event, ts: Date.now() }));
      break;
    case "local_llm_status":
      handlers.setLocalLlmStatus((current) => ({
        ...current,
        [event.profileId]: { state: event.state, message: event.message || "" }
      }));
      break;
    case "backend_ready":
      handlers.setBackendStatus((current) => ({ ...current, ready: true, running: true }));
      break;
    case "backend_exit":
      handlers.setBackendStatus((current) => ({
        ...current,
        ready: false,
        running: false,
        lastError: "Python backend stopped"
      }));
      handlers.setError(`Python backend stopped (${event.code ?? "null"}, ${event.signal ?? "null"})`);
      break;
    case "source_added":
      if (event.source) {
        handlers.setState((current) => upsertSessionSource(current, event.source));
      }
      break;
    case "source_removed":
      if (event.source) {
        handlers.setState((current) => removeSessionSource(current, event.source));
      }
      break;
    case "source_updated":
      if (event.source) {
        handlers.setState((current) => upsertSessionSource(current, event.source));
      }
      break;
    default:
      break;
  }
}
