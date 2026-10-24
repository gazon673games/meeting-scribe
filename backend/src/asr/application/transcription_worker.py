from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from asr.application.metrics import ASRMetrics
from asr.application.overload import OverloadController
from asr.application.policies import AdaptiveBeam
from asr.application.ports import (
    AsrBackendFactoryPort,
    AsrBackendPort,
    StopSignalPort,
)
from asr.application.utterances import UtteranceAggregator
from asr.application.worker_config import TranscriptionWorkerConfig
from asr.domain.dedup import StreamDedupFilter
from asr.domain.segments import Segment
from asr.domain.text import normalize_text
from diarization.application.diarization import DiarizationPort
from diarization.domain.speaker_labels import source_speaker_label

LogEvent = Callable[[dict], None]


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentRecognition:
    text: str
    latency_s: float
    metadata: dict[str, Any]


class TranscriptionWorkerRuntime:
    def __init__(
        self,
        *,
        config: TranscriptionWorkerConfig,
        segment_queue: queue.Queue[Segment],
        stop_event: StopSignalPort,
        log_event: LogEvent,
        metrics: ASRMetrics,
        overload: OverloadController,
        beam_controller: AdaptiveBeam,
        diarization: DiarizationPort,
        utterances: UtteranceAggregator,
        asr_backend_factory: AsrBackendFactoryPort,
    ) -> None:
        self._segment_queue = segment_queue
        self._stop = stop_event
        self._log_event = log_event
        self._metrics = metrics
        self._overload = overload
        self._beam_controller = beam_controller
        self._diarization = diarization
        self._utterances = utterances
        self._asr_backend_factory = asr_backend_factory

        self._model_name = str(config.model_name)
        self._language = config.language
        self._device = str(config.device)
        self._compute_type = str(config.compute_type)
        self._cpu_threads = int(config.cpu_threads)
        self._num_workers = int(config.num_workers)
        self._beam_size = int(config.beam_size)
        self._initial_prompt = config.initial_prompt
        self._hotwords = config.hotwords
        self._adaptive_beam_enabled = bool(config.adaptive_beam_enabled)
        self._log_speaker_labels = bool(config.log_speaker_labels)
        self._init_diarization = bool(config.init_diarization)
        self._diarization_blocking_lookup = bool(
            config.diarization_blocking_lookup
        )
        self._source_speaker_labels = dict(config.source_speaker_labels or {})

        self._dedup = StreamDedupFilter(
            enabled=bool(config.text_dedup_enabled),
            window=int(config.text_dedup_window),
        )
        self._asr: AsrBackendPort | None = None
        self.asr_init_error: str | None = None
        self._init_done = threading.Event()

    def reset_runtime(self) -> None:
        self._dedup.reset()
        self.asr_init_error = None
        self._init_done.clear()

    def wait_initialized(self, timeout_s: float) -> None:
        if not self._init_done.wait(max(0.0, float(timeout_s))):
            raise TimeoutError(
                f'ASR model initialization timed out after {float(timeout_s):.0f}s'
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
            self._log_event(
                {
                    'type': 'error',
                    'where': 'worker',
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

    def run(self) -> None:
        self._log_event(
            {
                'type': 'asr_init_start',
                'model': self._model_name,
                'device': self._device,
                'ts': time.time(),
            }
        )

        try:
            self._asr = self._asr_backend_factory(
                model_name=self._model_name,
                language=self._language,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
                num_workers=self._num_workers,
                beam_size=self._beam_size,
                initial_prompt=self._initial_prompt,
                hotwords=self._hotwords,
            )
            self._metrics.device = str(
                getattr(self._asr, 'device', self._device)
            )
            self.emit_metrics(force=True)
            self._log_event(
                {
                    'type': 'asr_init_ok',
                    'model': self._model_name,
                    'ts': time.time(),
                }
            )
            self._init_done.set()
        except Exception as error:
            self.asr_init_error = str(error)
            self._log_event(
                {
                    'type': 'error',
                    'where': 'asr_init',
                    'error': str(error),
                    'ts': time.time(),
                }
            )
            self._init_done.set()
            return

        if self._init_diarization:
            self._diarization.init_backend(self._log_event)

        while not self._stop.is_set():
            self._update_overload_state()
            self._drain_old_segments_if_hard_overload()

            try:
                seg = self._segment_queue.get(timeout=0.2)
            except queue.Empty:
                if self._utterances.enabled:
                    try:
                        self.flush_utterances(force=False)
                    except Exception:
                        _logger.debug(
                            'Best-effort operation failed in run', exc_info=True
                        )
                self.emit_metrics(force=False)
                continue

            self._update_overload_state()
            self._transcribe_segment(seg)
            self.emit_metrics(force=False)

    def emit_metrics(self, *, force: bool = False) -> None:
        event = self._metrics.build_event(
            force=force,
            seg_qsize=int(self._segment_queue.qsize()),
            overload_active=bool(self._overload.active),
            overload_strategy=str(self._overload.strategy),
            hard_overload=bool(self._overload.hard_active),
        )
        if event is not None:
            self._log_event(event)

    def flush_utterances(self, *, force: bool) -> None:
        for event in self._utterances.flush_all(
            now=time.monotonic(),
            force=force,
            overload=bool(self._overload.active),
        ):
            self._log_event(event)

    def _update_overload_state(self) -> None:
        now = time.monotonic()
        for event in self._overload.update(
            seg_qsize=int(self._segment_queue.qsize()),
            beam_cur=int(self._beam_controller.cur_beam),
            lag_s=float(self._metrics.last_lag_s),
            now=now,
        ):
            self._log_event(event)

    def _drain_old_segments_if_hard_overload(self) -> None:
        queue_size = int(self._segment_queue.qsize())
        drop_count = self._overload.drop_old_count(queue_size)
        if drop_count <= 0:
            return

        dropped = 0
        for _ in range(drop_count):
            try:
                self._segment_queue.get_nowait()
                dropped += 1
            except queue.Empty:
                break

        if dropped <= 0:
            return

        self._metrics.record_segments_skipped(dropped)
        self._log_event(
            {
                'type': 'segment_skipped_overload',
                'count': int(dropped),
                'seg_qsize': int(self._segment_queue.qsize()),
                'ts': time.time(),
            }
        )

    def _transcribe_segment(self, seg: Segment) -> None:
        speaker = self._speaker_for_segment(seg)
        seg_dur_s = max(1e-6, float(seg.duration_s))
        beam_to_use = self._overload.limit_beam(
            int(self._beam_controller.cur_beam)
        )
        queue_wait_s = seg.queue_wait_s(
            time.time(), monotonic_now=time.monotonic()
        )

        self._log_event(
            {
                'type': 'asr_segment_processing',
                'stream': str(seg.stream),
                't_start': float(seg.t_start),
                't_end': float(seg.t_end),
                'seg_dur_s': float(seg_dur_s),
                'seg_qsize': int(self._segment_queue.qsize()),
                'ts': time.time(),
            }
        )
        asr_result = self._run_asr(seg, speaker, beam_to_use)
        if asr_result is None:
            return
        text = asr_result.text
        asr_latency_s = asr_result.latency_s
        asr_metadata = asr_result.metadata

        text, removed = self._dedup.filter(seg.stream, text)
        total_lag_s = queue_wait_s + asr_latency_s
        self._metrics.record_latency(
            asr_latency_s=asr_latency_s,
            total_lag_s=total_lag_s,
            queue_wait_s=queue_wait_s,
            audio_duration_s=seg_dur_s,
        )
        self._maybe_update_beam(seg_dur_s, asr_latency_s)
        self._log_segment_event(
            seg,
            speaker,
            text,
            asr_latency_s,
            seg_dur_s,
            beam_to_use,
            removed,
            asr_metadata,
        )
        self._log_event(
            {
                'type': 'asr_segment_done',
                'stream': str(seg.stream),
                'speaker': str(speaker),
                't_start': float(seg.t_start),
                't_end': float(seg.t_end),
                'latency_s': float(asr_latency_s),
                'text_chars': len(text),
                'seg_qsize': int(self._segment_queue.qsize()),
                'ts': time.time(),
            }
        )
        if self._utterances.enabled and text.strip():
            self._update_utterances(seg, speaker, text)

    def _speaker_for_segment(self, seg: Segment) -> str:
        if self._diarization_blocking_lookup:
            return self._diarization.speaker_for_segment(seg, self._log_event)
        return source_speaker_label(self._source_speaker_labels, seg.stream)

    def _run_asr(
        self, seg: Segment, speaker: str, beam_to_use: int
    ) -> SegmentRecognition | None:
        started_at = time.monotonic()
        if self._asr is None:
            raise RuntimeError('ASR backend is not initialized')
        try:
            result = self._asr.transcribe(
                seg.audio.samples, beam_size=beam_to_use
            )
        except (RuntimeError, OSError) as error:
            self._log_event(
                {
                    'type': 'segment',
                    'stream': seg.stream,
                    'speaker': speaker if self._log_speaker_labels else 'S?',
                    't_start': seg.t_start,
                    't_end': seg.t_end,
                    'text': '',
                    'error': str(error),
                    'overload': bool(self._overload.active),
                    'ts': time.time(),
                }
            )
            return None

        text = (result.get('text') or '').strip()
        metadata = {
            key: result.get(key)
            for key in (
                'language',
                'language_probability',
                'avg_logprob',
                'no_speech_probability',
                'compression_ratio',
            )
            if result.get(key) is not None
        }
        return SegmentRecognition(
            text=text,
            latency_s=time.monotonic() - started_at,
            metadata=metadata,
        )

    def _maybe_update_beam(
        self, seg_dur_s: float, asr_latency_s: float
    ) -> None:
        if not self._adaptive_beam_enabled:
            return
        now = time.monotonic()
        new_beam, reason = self._beam_controller.maybe_update(
            seg_qsize=int(self._segment_queue.qsize()),
            last_latency_s=float(asr_latency_s),
            last_dur_s=float(seg_dur_s),
            now=now,
        )
        if reason:
            self._log_event(
                {
                    'type': 'asr_beam_update',
                    'beam': int(new_beam),
                    'reason': reason,
                    'ts': time.time(),
                }
            )

    def _log_segment_event(
        self,
        seg: Segment,
        speaker: str,
        text: str,
        asr_latency_s: float,
        seg_dur_s: float,
        beam_to_use: int,
        removed: int,
        asr_metadata: dict,
    ) -> None:
        self._log_event(
            {
                'type': 'segment',
                'stream': seg.stream,
                'speaker': speaker if self._log_speaker_labels else 'S?',
                't_start': seg.t_start,
                't_end': seg.t_end,
                'text': normalize_text(text),
                'latency_s': float(asr_latency_s),
                'seg_dur_s': float(seg_dur_s),
                'beam_used': int(beam_to_use),
                'dedup_removed_chars': int(removed),
                'seg_qsize': int(self._segment_queue.qsize()),
                'overload': bool(self._overload.active),
                'overload_strategy': self._overload.strategy,
                'hard_overload': bool(self._overload.hard_active),
                'lag_s': float(self._metrics.last_lag_s),
                'ts': time.time(),
                **asr_metadata,
            }
        )

    def _update_utterances(self, seg: Segment, speaker: str, text: str) -> None:
        for event in self._utterances.update(
            stream=str(seg.stream),
            speaker=str(speaker),
            t_start=float(seg.t_start),
            t_end=float(seg.t_end),
            text=str(text),
            now=time.monotonic(),
            overload=bool(self._overload.active),
        ):
            self._log_event(event)
