import {
  noticeFromBackendEvent,
  upsertRuntimeNotice,
} from '../../entities/runtimeNotice/model';
import { REFRESHING_EVENTS } from './constants';
import {
  applyAsrMetrics,
  removeSessionSource,
  upsertSessionSource,
} from './sessionState';
import {
  appendTranscriptLine,
  applyStreamingFinal,
  applyStreamingWords,
  updateTranscriptLine,
} from './transcriptState';

const MAX_VISIBLE_EVENTS = 8;

const STATE_REDUCERS = Object.freeze({
  transcript_line: appendTranscriptLine,
  transcript_line_update: updateTranscriptLine,
  asr_metrics: applyAsrMetrics,
  streaming_words: applyStreamingWords,
  streaming_final: applyStreamingFinal,
});

const handleSourceUpsert = (event, handlers) => {
  if (!event.source) return;
  handlers.setState((current) => upsertSessionSource(current, event.source));
};

const EVENT_HANDLERS = Object.freeze({
  assistant_ping_result: (event, handlers) => {
    handlers.setAssistantPing((current) => ({
      ...current,
      busy: false,
      ...event,
      ts: Date.now(),
    }));
  },
  local_llm_status: (event, handlers) => {
    handlers.setLocalLlmStatus((current) => ({
      ...current,
      [event.profileId]: {
        state: event.state,
        message: event.message || '',
      },
    }));
  },
  backend_ready: (event, handlers) => {
    handlers.setBackendStatus((current) => ({
      ...current,
      ready: true,
      running: true,
    }));
  },
  backend_exit: (event, handlers) => {
    handlers.setBackendStatus((current) => ({
      ...current,
      ready: false,
      running: false,
      lastError: 'Python backend stopped',
    }));
    const code = event.code ?? 'null';
    const signal = event.signal ?? 'null';
    handlers.setError(`Python backend stopped (${code}, ${signal})`);
  },
  source_added: handleSourceUpsert,
  source_updated: handleSourceUpsert,
  source_removed: (event, handlers) => {
    if (!event.source) return;
    handlers.setState((current) => removeSessionSource(current, event.source));
  },
});

export const handleBackendEvent = (event, handlers) => {
  handlers.setEvents((current) =>
    [event, ...current].slice(0, MAX_VISIBLE_EVENTS),
  );

  const runtimeNotice = noticeFromBackendEvent(event);
  if (runtimeNotice) {
    handlers.setRuntimeNotices((current) =>
      upsertRuntimeNotice(current, runtimeNotice),
    );
  }

  if (Object.hasOwn(STATE_REDUCERS, event.type)) {
    const reduce = STATE_REDUCERS[event.type];
    handlers.setState((current) => reduce(current, event));
  } else if (Object.hasOwn(EVENT_HANDLERS, event.type)) {
    EVENT_HANDLERS[event.type](event, handlers);
  }

  if (REFRESHING_EVENTS.includes(event.type)) {
    handlers.refresh({ includeDevices: false, showLoading: false });
  }
};
