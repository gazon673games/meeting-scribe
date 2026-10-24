from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

import numpy as np

from asr.application.metrics import ASRMetrics
from asr.application.ports import (
    AsrBackendFactoryPort,
    AsrBackendPort,
    StatefulTranscribePort,
    StopSignalPort,
)
from asr.application.streaming_worker_config import StreamingWorkerConfig
from asr.domain.streaming import (
    ConfirmedPrefixTracker,
    StreamingChunk,
    StreamingWord,
)

LogEvent = Callable[[dict], None]


_logger = logging.getLogger(__name__)


class StreamingWhisperWorker:
    """
    Consumes StreamingChunk objects and emits streaming transcript events.

    For each chunk it re-transcribes the full audio buffer (growing since
    speech start) and uses ConfirmedPrefixTracker to identify stable words.

    The ASR backend is created lazily inside run() (same thread that uses it),
    so CUDA resources are owned and released in the worker thread.

    Events emitted via log_event:
      streaming_words — intermediate chunk: newly confirmed + tentative words
      streaming_final — final chunk: complete confirmed utterance
    """

    def __init__(
        self,
        *,
        config: StreamingWorkerConfig,
        chunk_queue: queue.Queue[StreamingChunk],
        stop_event: StopSignalPort,
        log_event: LogEvent,
        asr_backend_factory: AsrBackendFactoryPort,
        metrics: ASRMetrics | None = None,
    ) -> None:
        self._cfg = config
        self._chunk_q = chunk_queue
        self._stop = stop_event
        self._log = log_event
        self._factory = asr_backend_factory
        self._metrics = (
            metrics
            if metrics is not None
            else ASRMetrics(latency_window=200, emit_interval_s=1.0)
        )
        self._asr: AsrBackendPort | None = None
        self._stateful_transcribe: StatefulTranscribePort | None = None
        self._trackers: dict[str, ConfirmedPrefixTracker] = {}
        self._last_inference_s = 0.0
        self._init_done = threading.Event()
        self.asr_init_error: str | None = None

    def reset_runtime(self) -> None:
        self._trackers.clear()
        self._metrics.reset()
        self.asr_init_error = None
        self._init_done.clear()

    def wait_initialized(self, timeout_s: float) -> None:
        if not self._init_done.wait(max(0.0, float(timeout_s))):
            raise TimeoutError(
                f'Streaming ASR initialization timed out after {float(timeout_s):.0f}s'
            )
        if self.asr_init_error:
            raise RuntimeError(self.asr_init_error)

    def run_safe(self) -> None:
        try:
            self.run()
        except Exception as error:
            if not self._init_done.is_set():
                self.asr_init_error = str(error)
                self._init_done.set()
            self._log(
                {
                    'type': 'error',
                    'where': 'streaming_worker',
                    'error': str(error),
                    'ts': time.time(),
                }
            )
        finally:
            # Release CUDA resources in the worker thread that owns them
            if self._asr is not None:
                try:
                    self._asr.close()
                except Exception:
                    _logger.debug(
                        'Best-effort operation failed in run_safe',
                        exc_info=True,
                    )
            self._asr = None
            self._stateful_transcribe = None

    def run(self) -> None:
        self._log(
            {
                'type': 'asr_init_start',
                'model': self._cfg.model_name,
                'device': self._cfg.device,
                'ts': time.time(),
            }
        )
        try:
            backend = self._factory(
                model_name=self._cfg.model_name,
                language=self._cfg.language,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
                cpu_threads=self._cfg.cpu_threads,
                num_workers=self._cfg.num_workers,
                beam_size=1,
                initial_prompt=self._cfg.initial_prompt,
                hotwords=self._cfg.hotwords,
            )
            self._bind_backend(backend)
            self._emit_metrics(force=True)
            self._log(
                {
                    'type': 'asr_init_ok',
                    'model': self._cfg.model_name,
                    'ts': time.time(),
                }
            )
            self._init_done.set()
        except Exception as error:
            self.asr_init_error = str(error)
            self._log(
                {
                    'type': 'error',
                    'where': 'asr_init',
                    'error': str(error),
                    'ts': time.time(),
                }
            )
            self._init_done.set()
            return

        while not self._stop.is_set():
            try:
                chunk = self._chunk_q.get(timeout=self._cfg.queue_timeout_s)
            except queue.Empty:
                continue
            self._process(chunk)

    # ── processing ────────────────────────────────────────────────────

    def _bind_backend(self, backend: AsrBackendPort) -> None:
        self._asr = backend
        transcribe = getattr(backend, 'transcribe_stream_chunk', None)
        self._stateful_transcribe = transcribe if callable(transcribe) else None
        self._metrics.device = str(getattr(backend, 'device', self._cfg.device))

    def _process(self, chunk: StreamingChunk) -> None:
        started_at = time.monotonic()
        queue_wait_s = chunk.queue_wait_s(started_at)
        try:
            succeeded = self._process_chunk(chunk)
        finally:
            if chunk.is_final:
                self._trackers.pop(chunk.stream, None)
        if not succeeded:
            return
        latency_s = time.monotonic() - started_at
        audio = chunk.incremental_audio if self._stateful_transcribe else None
        duration_s = (audio if audio is not None else chunk.audio).duration_s
        self._metrics.record_latency(
            asr_latency_s=self._last_inference_s,
            total_lag_s=queue_wait_s + latency_s,
            queue_wait_s=queue_wait_s,
            audio_duration_s=duration_s,
        )
        self._emit_metrics(force=chunk.is_final)

    def _emit_metrics(self, *, force: bool) -> None:
        event = self._metrics.build_event(
            force=force,
            seg_qsize=self._chunk_q.qsize(),
            overload_active=False,
            overload_strategy='coalesce',
            hard_overload=False,
        )
        if event is not None:
            self._log(event)

    def _process_chunk(self, chunk: StreamingChunk) -> bool:
        if self._stateful_transcribe is not None:
            return self._process_stateful(chunk, self._stateful_transcribe)

        stream = chunk.stream
        tracker = self._trackers.setdefault(
            stream, ConfirmedPrefixTracker(lookahead=self._cfg.lookahead)
        )

        try:
            inference_started_at = time.monotonic()
            raw = self._asr.transcribe_words(
                np.asarray(chunk.audio.samples, dtype=np.float32)
            )  # type: ignore[union-attr]
            self._last_inference_s = time.monotonic() - inference_started_at
        except Exception as error:
            self._log(
                {
                    'type': 'error',
                    'where': 'streaming_transcribe',
                    'stream': stream,
                    'error': str(error),
                    'ts': time.time(),
                }
            )
            if chunk.is_final:
                self._emit_final(
                    chunk,
                    [_word_to_dict(word) for word in tracker.confirmed_words],
                )
            return False

        words = [
            StreamingWord(
                text=w['text'], start_s=float(w['start']), end_s=float(w['end'])
            )
            for w in raw
        ]

        if chunk.is_final:
            tracker.update(words)
            tracker.flush()
            self._emit_final(
                chunk, [_word_to_dict(w) for w in tracker.confirmed_words]
            )
            tracker.reset()
        else:
            update = tracker.update(words)
            if update.newly_confirmed or update.tentative:
                self._log(
                    {
                        'type': 'streaming_words',
                        'stream': stream,
                        'confirmed': [
                            _word_to_dict(w) for w in update.newly_confirmed
                        ],
                        'tentative': [
                            _word_to_dict(w) for w in update.tentative
                        ],
                        't_start': chunk.t_start,
                        't_end': chunk.t_end,
                        'ts': time.time(),
                    }
                )

        return True

    def _process_stateful(
        self, chunk: StreamingChunk, transcribe: StatefulTranscribePort
    ) -> bool:
        audio = chunk.incremental_audio or chunk.audio
        try:
            inference_started_at = time.monotonic()
            result = transcribe(
                chunk.stream,
                np.asarray(audio.samples, dtype=np.float32),
                is_final=bool(chunk.is_final),
            )
            self._last_inference_s = time.monotonic() - inference_started_at
        except Exception as error:
            self._log(
                {
                    'type': 'error',
                    'where': 'streaming_transcribe',
                    'stream': chunk.stream,
                    'error': str(error),
                    'ts': time.time(),
                }
            )
            if chunk.is_final:
                self._emit_final(chunk, [])
            return False

        words = _plain_text_words(
            str(result.get('text') or ''),
            float(chunk.t_start),
            float(chunk.t_end),
        )
        if chunk.is_final:
            self._emit_final(chunk, words)
        elif words:
            self._log(
                {
                    'type': 'streaming_words',
                    'stream': chunk.stream,
                    'confirmed': [],
                    'tentative': words,
                    't_start': chunk.t_start,
                    't_end': chunk.t_end,
                    'ts': time.time(),
                }
            )

        return True

    def _emit_final(self, chunk: StreamingChunk, words: list[dict]) -> None:
        self._log(
            {
                'type': 'streaming_final',
                'stream': chunk.stream,
                'words': words,
                't_start': chunk.t_start,
                't_end': chunk.t_end,
                'ts': time.time(),
            }
        )


def _word_to_dict(w: StreamingWord) -> dict:
    return {'text': w.text, 'start': w.start_s, 'end': w.end_s}


def _plain_text_words(text: str, t_start: float, t_end: float) -> list[dict]:
    parts = [part for part in str(text or '').strip().split() if part]
    duration = max(0.0, float(t_end) - float(t_start))
    step = duration / max(1, len(parts))
    return [
        {'text': word, 'start': index * step, 'end': (index + 1) * step}
        for index, word in enumerate(parts)
    ]
