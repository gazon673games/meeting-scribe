from __future__ import annotations

import queue
import time
from typing import Callable, Dict, Optional

import numpy as np

from asr.application.ports import AsrBackendFactoryPort, AsrBackendPort, StopSignalPort
from asr.application.streaming_worker_config import StreamingWorkerConfig
from asr.domain.streaming import ConfirmedPrefixTracker, StreamingChunk, StreamingWord

LogEvent = Callable[[dict], None]


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
        chunk_queue: "queue.Queue[StreamingChunk]",
        stop_event: StopSignalPort,
        log_event: LogEvent,
        asr_backend_factory: AsrBackendFactoryPort,
    ) -> None:
        self._cfg = config
        self._chunk_q = chunk_queue
        self._stop = stop_event
        self._log = log_event
        self._factory = asr_backend_factory
        self._asr: Optional[AsrBackendPort] = None
        self._trackers: Dict[str, ConfirmedPrefixTracker] = {}

    def reset_runtime(self) -> None:
        self._trackers.clear()

    def run_safe(self) -> None:
        try:
            self.run()
        except Exception as e:
            self._log({"type": "error", "where": "streaming_worker", "error": str(e), "ts": time.time()})
        finally:
            # Release CUDA resources in the worker thread that owns them
            if self._asr is not None:
                try:
                    self._asr.close()
                except Exception:
                    pass
            self._asr = None

    def run(self) -> None:
        self._log({"type": "asr_init_start", "model": self._cfg.model_name, "device": self._cfg.device, "ts": time.time()})
        try:
            self._asr = self._factory(
                model_name=self._cfg.model_name,
                language=self._cfg.language,
                device=self._cfg.device,
                compute_type=self._cfg.compute_type,
                cpu_threads=self._cfg.cpu_threads,
                num_workers=self._cfg.num_workers,
                beam_size=1,
                initial_prompt=self._cfg.initial_prompt,
            )
            self._log({"type": "asr_init_ok", "model": self._cfg.model_name, "ts": time.time()})
        except Exception as e:
            self._log({"type": "error", "where": "asr_init", "error": str(e), "ts": time.time()})
            return

        while not self._stop.is_set():
            try:
                chunk = self._chunk_q.get(timeout=self._cfg.queue_timeout_s)
            except queue.Empty:
                continue
            self._process(chunk)

    # ── processing ────────────────────────────────────────────────────

    def _process(self, chunk: StreamingChunk) -> None:
        stream = chunk.stream
        tracker = self._trackers.setdefault(
            stream, ConfirmedPrefixTracker(lookahead=self._cfg.lookahead)
        )

        try:
            raw = self._asr.transcribe_words(np.asarray(chunk.audio.samples, dtype=np.float32))  # type: ignore[union-attr]
        except Exception as e:
            self._log({
                "type": "error", "where": "streaming_transcribe",
                "stream": stream, "error": str(e), "ts": time.time(),
            })
            return

        words = [
            StreamingWord(text=w["text"], start_s=float(w["start"]), end_s=float(w["end"]))
            for w in raw
        ]

        if chunk.is_final:
            tracker.update(words)
            tracker.flush()
            self._log({
                "type": "streaming_final",
                "stream": stream,
                "words": [_word_to_dict(w) for w in tracker.confirmed_words],
                "t_start": chunk.t_start,
                "t_end": chunk.t_end,
                "ts": time.time(),
            })
            tracker.reset()
        else:
            update = tracker.update(words)
            if update.newly_confirmed or update.tentative:
                self._log({
                    "type": "streaming_words",
                    "stream": stream,
                    "confirmed": [_word_to_dict(w) for w in update.newly_confirmed],
                    "tentative": [_word_to_dict(w) for w in update.tentative],
                    "t_start": chunk.t_start,
                    "t_end": chunk.t_end,
                    "ts": time.time(),
                })


def _word_to_dict(w: StreamingWord) -> dict:
    return {"text": w.text, "start": w.start_s, "end": w.end_s}
