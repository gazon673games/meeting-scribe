import React from 'react';
import {
  Activity,
  AlertTriangle,
  Cpu,
  Database,
  Pause,
  Play,
  Trash2,
} from 'lucide-react';
import { useColorTheme } from '../entities/settings/useColorTheme';

import { meetingScribeClient } from '../shared/api/meetingScribeClient';
import { formatBytes } from '../shared/lib/format';
import './styles.css';

const MAX_EVENTS = 600;
const COMPACT_EVENT_KEYS = Object.freeze([
  'type',
  'where',
  'stream',
  'speaker',
  'model',
  'message',
  'error',
  'text',
]);

const foldStatusEvents = (events, initialStatus, reducer) => {
  let status = initialStatus;
  for (let index = events.length - 1; index >= 0; index--) {
    status = reducer(status, events[index]);
  }
  return status;
};

const statusRecord = (status, detail, tone) => ({ status, detail, tone });

const transcriptSummary = (event) => {
  const speaker = event.speaker || '';
  const stream = event.stream || '';
  return `${speaker} ${stream}: ${event.text || ''}`.trim();
};

const errorSummary = (event) =>
  `${event.where || 'error'}: ${event.error || event.message || ''}`;

const isBlankValue = (value) =>
  value === undefined || value === null || value === '';

const compactEventParts = (event) => {
  const parts = [];
  for (const key of COMPACT_EVENT_KEYS) {
    const value = event[key];
    if (isBlankValue(value)) {
      continue;
    }
    parts.push(`${key}=${String(value).replace(/\s+/g, ' ').slice(0, 120)}`);
  }
  return parts;
};

const compactEventText = (event, maxLength = 180) => {
  if (!event || typeof event !== 'object') {
    return '';
  }
  const parts = compactEventParts(event);
  const text = parts.join(' ');
  if (text) {
    return text.length > maxLength
      ? `${text.slice(0, maxLength - 1)}...`
      : text;
  }
  return String(event.type || 'event');
};

const eventTitle = (event) => compactEventText(event, 300);

const firstDefined = (values, fallback) => {
  for (const value of values) {
    if (value !== undefined && value !== null && value !== '') {
      return value;
    }
  }
  return fallback;
};

const valueOrDefault = (value, fallback) =>
  value === undefined || value === null ? fallback : value;

const eventKey = (event, fallbackIndex) => {
  const ts = Number(valueOrDefault(event?.ts, 0)).toFixed(6);
  const type = String(valueOrDefault(event?.type, ''));
  const id = String(firstDefined([event?.id, event?.line?.id], ''));
  const stream = String(firstDefined([event?.stream, event?.line?.stream], ''));
  const start = firstDefined([event?.t_start, event?.line?.t_start], '');
  const end = firstDefined([event?.t_end, event?.line?.t_end], '');
  const span = `${valueOrDefault(start, '')}:${valueOrDefault(end, '')}`;
  const text = String(
    firstDefined(
      [event?.text, event?.line?.text, event?.message, event?.error],
      '',
    ),
  ).slice(0, 80);
  return `${ts}:${type}:${id}:${stream}:${span}:${text || fallbackIndex}`;
};

const mergeEventLists = (primary, secondary) => {
  const merged = [];
  const seen = new Set();
  for (const event of [...(primary || []), ...(secondary || [])]) {
    const key = eventKey(event, merged.length);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(event);
  }
  merged.sort((left, right) => Number(right?.ts || 0) - Number(left?.ts || 0));
  return merged.slice(0, MAX_EVENTS);
};

const eventTone = (event) => {
  if (
    event.type === 'error' ||
    event.type === 'backend_stderr' ||
    event.type === 'backend_exit'
  ) {
    return 'error';
  }
  if (String(event.type || '').includes('processing')) {
    return 'warn';
  }
  if (
    event.type === 'transcript_speaker_update' ||
    event.type === 'diar_debug'
  ) {
    return 'good';
  }
  return '';
};

const formatTime = (ts) => {
  const value = Number(ts ?? Date.now() / 1000);
  return new Date(value * 1000).toLocaleTimeString();
};

const seconds = (value) => {
  const numberValue = Number(value || 0);
  return Number.isFinite(numberValue) ? `${numberValue.toFixed(2)}s` : '-';
};

const number = (value) => {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue.toFixed(2) : '-';
};

const diarDebugDetail = (event) =>
  [
    event.stream || '-',
    'speaker',
    event.speaker || '-',
    'sim',
    number(event.best_sim),
    'speakers',
    event.n_speakers_window ?? '-',
  ].join(' ');

const segmentDetail = (event) => {
  const span = `${number(event.t_start)}-${number(event.t_end)}s`;
  return `${event.stream || '-'} ${span}`;
};

const ASR_EVENT_HANDLERS = Object.freeze({
  asr_init_start: (event) =>
    statusRecord(
      'starting',
      `${event.model || 'model'} on ${event.device || '-'}`,
      'warn',
    ),
  asr_init_ok: (event) =>
    statusRecord('ready', event.model || 'ASR initialized', 'good'),
  asr_started: (event) =>
    statusRecord('ready', event.model || 'ASR initialized', 'good'),
  asr_segment_processing: (event) =>
    statusRecord('processing', segmentDetail(event), 'warn'),
  asr_segment_done: (event) =>
    statusRecord(
      'idle',
      `${segmentDetail(event)} latency ${seconds(event.latency_s)}`,
      'good',
    ),
  segment: (event) =>
    statusRecord(
      'idle',
      `${segmentDetail(event)} latency ${seconds(event.latency_s)}`,
      'good',
    ),
  asr_stopped: () => statusRecord('stopped', 'Session stopped', 'idle'),
  session_stopped: () => statusRecord('stopped', 'Session stopped', 'idle'),
});
const DIAR_EVENT_HANDLERS = Object.freeze({
  diar_init_ok: (event) =>
    statusRecord('ready', `backend ${event.backend || '-'}`, 'good'),
  diar_sidecar_started: () => statusRecord('ready', 'sidecar started', 'good'),
  diar_segment_processing: (event) =>
    statusRecord('processing', segmentDetail(event), 'warn'),
  diar_debug: (event) => statusRecord('idle', diarDebugDetail(event), 'good'),
  diar_segment_done: (event) =>
    statusRecord(
      'idle',
      `${event.speaker || 'no speaker'} in ${seconds(event.latency_s)}`,
      event.speaker ? 'good' : 'idle',
    ),
  transcript_speaker_update: (event) =>
    statusRecord(
      'updated',
      `${event.stream || '-'} -> ${event.speaker || '-'}`,
      'good',
    ),
});
const PROCESSING_TYPES = Object.freeze([
  'asr_segment_processing',
  'diar_segment_processing',
]);
const DONE_TYPES = Object.freeze(['asr_segment_done', 'diar_segment_done']);
const SUMMARY_HANDLERS = Object.freeze({
  utterance: transcriptSummary,
  transcript_line: transcriptSummary,
  transcript_speaker_update: (event) =>
    [
      event.stream || '-',
      `${event.t_start ?? ''}-${event.t_end ?? ''}`,
      '->',
      event.speaker || '-',
    ].join(' '),
  diar_debug: diarDebugDetail,
  backend_stderr: (event) => event.text || '',
});

const eventSummary = (event) => {
  const type = String(event?.type || '');
  if (type === 'error') return errorSummary(event);
  if (Object.hasOwn(SUMMARY_HANDLERS, type)) {
    return SUMMARY_HANDLERS[type](event);
  }
  if (PROCESSING_TYPES.includes(type)) {
    return `processing ${segmentDetail(event)}`;
  }
  if (DONE_TYPES.includes(type)) {
    return [
      'done',
      segmentDetail(event),
      'in',
      seconds(event.latency_s),
      event.speaker || '',
    ]
      .join(' ')
      .trim();
  }
  return compactEventText(event);
};

const LogRow = ({ event }) => {
  const tone = eventTone(event);
  return (
    <div className={`debug-log-row ${tone}`}>
      <time>{formatTime(event.ts)}</time>
      <strong>{event.type || 'event'}</strong>
      <span title={eventTitle(event)}>{eventSummary(event)}</span>
    </div>
  );
};

const applyAsrEvent = (status, event) => {
  if (event.type === 'error') {
    return String(event.where || '')
      .toLowerCase()
      .startsWith('asr')
      ? statusRecord(
          'error',
          event.error || event.message || 'ASR error',
          'error',
        )
      : status;
  }
  if (!Object.hasOwn(ASR_EVENT_HANDLERS, event.type)) return status;
  return ASR_EVENT_HANDLERS[event.type](event, status);
};

const buildAsrStatus = (events) => {
  const initial = statusRecord('idle', 'Waiting for ASR events', 'idle');
  return foldStatusEvents(events, initial, applyAsrEvent);
};

const applyDiarEvent = (status, event) => {
  if (event.type === 'error') {
    return String(event.where || '').startsWith('diar')
      ? statusRecord('error', event.error || 'Speaker ID error', 'error')
      : status;
  }
  if (!Object.hasOwn(DIAR_EVENT_HANDLERS, event.type)) return status;
  return DIAR_EVENT_HANDLERS[event.type](event, status);
};

const buildDiarStatus = (events) => {
  const initial = statusRecord('idle', 'Waiting for Speaker ID events', 'idle');
  return foldStatusEvents(events, initial, applyDiarEvent);
};

const backendProcessView = (backendStatus) => {
  if (!backendStatus.running) return { status: 'stopped', tone: 'error' };
  return backendStatus.ready
    ? { status: 'ready', tone: 'good' }
    : { status: 'starting', tone: 'warn' };
};

const gpuUsageDetail = (gpu, memory, resourceUsage) => {
  const appMemory = formatBytes(resourceUsage.app?.memoryBytes || 0);
  return `${gpu.name || 'GPU'} memory ${memory}, app ${appMemory}`;
};

const buildGpuStatus = (resourceUsage) => {
  const gpu = resourceUsage.gpus?.[0];
  if (!gpu) {
    return { status: 'unknown', detail: 'No GPU metrics', tone: 'idle' };
  }
  const util = Number(gpu.gpuUtilizationPct || 0);
  const memory = `${gpu.memoryUsedMiB || 0}/${gpu.memoryTotalMiB || 0} MiB`;
  return {
    status: `${Math.round(util)}%`,
    detail: gpuUsageDetail(gpu, memory, resourceUsage),
    tone: util >= 90 ? 'warn' : 'good',
  };
};

const buildProcessRows = ({ backendStatus, events, resourceUsage }) => {
  const backendView = backendProcessView(backendStatus);
  const asr = buildAsrStatus(events);
  const diar = buildDiarStatus(events);
  const gpu = buildGpuStatus(resourceUsage);
  return [
    {
      id: 'backend',
      label: 'Backend',
      status: backendView.status,
      detail:
        backendStatus.lastError ||
        `PID ${resourceUsage.app?.backendPid || '-'}`,
      tone: backendView.tone,
      icon: <Database size={18} />,
    },
    { id: 'asr', label: 'ASR', icon: <Activity size={18} />, ...asr },
    {
      id: 'diar',
      label: 'Speaker ID',
      icon: <AlertTriangle size={18} />,
      ...diar,
    },
    { id: 'gpu', label: 'GPU', icon: <Cpu size={18} />, ...gpu },
  ];
};

export const DebugConsoleApp = () => {
  const [events, setEvents] = React.useState([]);
  const [runtimeEvents, setRuntimeEvents] = React.useState([]);
  const [pausedEvents, setPausedEvents] = React.useState(null);
  const [search, setSearch] = React.useState('');
  const [filter, setFilter] = React.useState('all');
  const theme = useColorTheme();
  const [backendStatus, setBackendStatus] = React.useState({
    ready: false,
    running: false,
    lastError: '',
  });
  const [resourceUsage, setResourceUsage] = React.useState({
    app: {},
    system: {},
    gpus: [],
  });

  React.useEffect(() => {
    let cancelled = false;
    meetingScribeClient
      .recentBackendEvents()
      .then((result) => {
        if (!cancelled) {
          setEvents((current) =>
            mergeEventLists(
              (result || []).slice(-MAX_EVENTS).reverse(),
              current,
            ),
          );
          setRuntimeEvents((current) =>
            mergeEventLists(
              (result || []).slice(-MAX_EVENTS).reverse(),
              current,
            ),
          );
        }
      })
      .catch((error) => {
        if (!cancelled) {
          console.warn('[diagnostics] event history failed', error);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  React.useEffect(
    () =>
      meetingScribeClient.onBackendEvent((event) => {
        setEvents((current) => mergeEventLists([event], current));
        setRuntimeEvents((current) => mergeEventLists([event], current));
      }),
    [],
  );

  React.useEffect(() => {
    let cancelled = false;
    let isPending = false;
    const refreshRuntime = async () => {
      if (isPending) return;
      isPending = true;
      try {
        const results = await Promise.all([
          meetingScribeClient.status(),
          meetingScribeClient.resourceUsage(),
        ]);
        if (cancelled) return;
        setBackendStatus(
          results[0] || { ready: false, running: false, lastError: '' },
        );
        setResourceUsage(results[1] || { app: {}, system: {}, gpus: [] });
      } catch (error) {
        if (!cancelled) {
          console.warn('[diagnostics] runtime polling failed', error);
        }
      } finally {
        isPending = false;
      }
    };
    refreshRuntime();
    const handle = window.setInterval(refreshRuntime, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, []);

  const processes = React.useMemo(
    () =>
      buildProcessRows({ backendStatus, events: runtimeEvents, resourceUsage }),
    [backendStatus, runtimeEvents, resourceUsage],
  );
  const displayedEvents = pausedEvents || events;
  const query = search.trim().toLocaleLowerCase();
  const filteredEvents = displayedEvents.filter(
    (event) =>
      (filter === 'all' || eventTone(event) === filter) &&
      (!query ||
        `${event.type || ''} ${eventSummary(event)} ${eventTitle(event)}`
          .toLocaleLowerCase()
          .includes(query)),
  );

  return (
    <main className="debug-console-shell" data-theme={theme}>
      <header className="debug-console-head">
        <div>
          <span>Diagnostics</span>
          <h1>ASR / Speaker ID Console</h1>
        </div>
      </header>

      <section className="debug-process-grid">
        {processes.map((process) => (
          <article
            key={process.id}
            className={`debug-process-card ${process.tone}`}
          >
            <div className="debug-process-icon">{process.icon}</div>
            <div className="debug-process-copy">
              <div>
                <strong>{process.label}</strong>
                <span>
                  <i className={`debug-status-dot ${process.tone}`} />
                  {process.status}
                </span>
              </div>
              <p title={process.detail}>{process.detail || '-'}</p>
            </div>
          </article>
        ))}
      </section>

      <section className="debug-log-panel">
        <div className="debug-log-head">
          <strong>Live Logs</strong>
          <span>
            {filteredEvents.length === displayedEvents.length
              ? `${displayedEvents.length} events`
              : `${filteredEvents.length} of ${displayedEvents.length} events`}
          </span>
          <div className="debug-log-actions">
            <button
              aria-pressed={pausedEvents !== null}
              type="button"
              onClick={() =>
                setPausedEvents((current) =>
                  current === null ? [...events] : null,
                )
              }
            >
              {pausedEvents === null ? <Pause size={13} /> : <Play size={13} />}
              {pausedEvents === null ? 'Pause view' : 'Resume'}
            </button>
            <button
              title="Clear visible logs"
              type="button"
              onClick={() => {
                setEvents([]);
                if (pausedEvents !== null) setPausedEvents([]);
              }}
            >
              <Trash2 size={13} />
              Clear logs
            </button>
          </div>
        </div>
        <div className="debug-log-filters">
          <input
            aria-label="Search logs"
            placeholder="Search events or details..."
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
          <select
            aria-label="Filter log events"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          >
            <option value="all">All events</option>
            <option value="error">Errors</option>
            <option value="warn">Processing</option>
            <option value="good">Speaker updates</option>
          </select>
        </div>
        <div className="debug-log-list">
          <div className="debug-log-row debug-log-columns">
            <span>Time</span>
            <span>Event</span>
            <span>Details</span>
          </div>
          {filteredEvents.length ? (
            filteredEvents.map((event, index) => (
              <LogRow
                event={event}
                key={`${event.ts || index}:${event.type}:${index}`}
              />
            ))
          ) : (
            <div className="debug-log-empty">
              {displayedEvents.length
                ? 'No events match your filters.'
                : 'No backend events yet.'}
            </div>
          )}
        </div>
      </section>
    </main>
  );
};
