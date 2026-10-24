from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from asr.domain.segments import Segment
from diarization.application.diarization import (
    DiarizationConfig,
    DiarizationPort,
)
from diarization.application.ports import (
    OnlineDiarizerFactoryPort,
    OnlineDiarizerPort,
    PyannoteDiarizerFactoryPort,
    PyannoteDiarizerPort,
)
from diarization.domain.segments import DiarSegment, pick_speaker
from diarization.domain.speaker_labels import source_speaker_label
from diarization.domain.types import DiarBackend

LogEvent = Callable[[dict], None]
_logger = logging.getLogger(__name__)


@dataclass
class _PyannoteStream:
    chunks: deque[np.ndarray] = field(default_factory=deque)
    samples: int = 0
    start_s: float = 0.0
    last_run_monotonic: float | None = None
    timeline: list[DiarSegment] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def append_audio(
        self, samples: np.ndarray, *, end_s: float, max_samples: int
    ) -> None:
        with self.lock:
            self.chunks.append(samples.copy())
            self.samples += int(samples.size)
            while self.samples > max_samples and self.chunks:
                cut = self.samples - max_samples
                head = self.chunks[0]
                if cut >= head.size:
                    self.samples -= int(head.size)
                    self.chunks.popleft()
                    continue
                self.chunks[0] = head[int(cut) :]
                self.samples -= int(cut)
                break
            self.start_s = float(end_s) - self.samples / 16000.0

    def snapshot_audio(self) -> tuple[np.ndarray, float]:
        with self.lock:
            if not self.chunks:
                return np.zeros((0,), dtype=np.float32), self.start_s
            if len(self.chunks) == 1:
                return self.chunks[0].copy(), self.start_s
            return np.concatenate(tuple(self.chunks)), self.start_s


class DiarizationRuntime:
    def __init__(
        self,
        *,
        config: DiarizationConfig,
        online_diarizer_factory: OnlineDiarizerFactoryPort,
        pyannote_diarizer_factory: PyannoteDiarizerFactoryPort,
    ) -> None:
        self.enabled = bool(config.enabled)
        self.backend: DiarBackend = config.backend
        self.sim_threshold = float(config.sim_threshold)
        self.min_segment_s = float(config.min_segment_s)
        self.window_s = float(config.window_s)
        self.chunk_s = float(config.chunk_s)
        self.step_s = float(config.step_s)
        self.device = str(config.device)
        self.temp_dir = (
            Path(config.temp_dir) if config.temp_dir is not None else None
        )
        self.source_speaker_labels = dict(config.source_speaker_labels or {})
        self.sherpa_embedding_model_path = str(
            config.sherpa_embedding_model_path or ''
        )
        self.sherpa_provider = str(config.sherpa_provider or 'cpu')
        self.sherpa_num_threads = max(1, int(config.sherpa_num_threads))
        self._online_diarizer_factory = online_diarizer_factory
        self._pyannote_diarizer_factory = pyannote_diarizer_factory

        self._diarizers: dict[str, OnlineDiarizerPort] = {}
        self._pyannote: PyannoteDiarizerPort | None = None
        self._pyannote_streams: dict[str, _PyannoteStream] = {}

    def ensure_stream(self, name: str) -> None:
        if not self.enabled:
            return

        if (
            self.backend in ('online', 'nemo', 'sherpa_onnx')
            and name not in self._diarizers
        ):
            self._diarizers[name] = self._online_diarizer_factory(
                similarity_threshold=self.sim_threshold,
                min_segment_s=self.min_segment_s,
                window_s=self.window_s,
                backend=_embedding_backend_name(self.backend),
                device=self.device,
                temp_dir=self.temp_dir,
                sherpa_model_path=self.sherpa_embedding_model_path,
                sherpa_provider=self.sherpa_provider,
                sherpa_num_threads=self.sherpa_num_threads,
            )

        if self.backend == 'pyannote' and name not in self._pyannote_streams:
            self._pyannote_streams[name] = _PyannoteStream()

    def update_ring(self, stream: str, t1: float, x16: np.ndarray) -> None:
        if not self.enabled or self.backend != 'pyannote':
            return

        self.ensure_stream(stream)

        x16 = np.asarray(x16, dtype=np.float32).reshape(-1)
        if x16.size == 0:
            return

        max_len = int(max(1.0, self.chunk_s) * 16000)
        state = self._pyannote_streams[stream]
        state.append_audio(x16, end_s=t1, max_samples=max_len)

    def init_backend(self, log_event: LogEvent) -> None:
        if not self.enabled:
            return

        if self.backend == 'pyannote':
            try:
                self._pyannote = self._pyannote_diarizer_factory(
                    device=self.device
                )
                log_event(
                    {
                        'type': 'diar_init_ok',
                        'backend': 'pyannote',
                        'ts': time.time(),
                    }
                )
                return
            except Exception as error:
                log_event(
                    {
                        'type': 'error',
                        'where': 'diar_init',
                        'error': str(error),
                        'ts': time.time(),
                    }
                )
                self.backend = 'nemo'
                self._pyannote = None
                log_event(
                    {
                        'type': 'diar_fallback',
                        'backend': 'nemo',
                        'ts': time.time(),
                    }
                )

        if self.backend == 'online':
            log_event(
                {'type': 'diar_init_ok', 'backend': 'online', 'ts': time.time()}
            )
            return

        if self.backend == 'nemo':
            log_event(
                {'type': 'diar_init_ok', 'backend': 'nemo', 'ts': time.time()}
            )
            return

        if self.backend == 'sherpa_onnx':
            log_event(
                {
                    'type': 'diar_init_ok',
                    'backend': 'sherpa_onnx',
                    'ts': time.time(),
                }
            )
            return

    def speaker_for_segment(self, seg: Segment, log_event: LogEvent) -> str:
        if not self.enabled:
            return self._fallback_speaker(seg.stream)

        self.ensure_stream(seg.stream)

        if self.backend == 'pyannote':
            self._maybe_update_pyannote_timeline(seg.stream, log_event)
            timeline = self._pyannote_streams[seg.stream].timeline
            return self._with_fallback(
                pick_speaker(timeline, seg.t_start, seg.t_end), seg.stream
            )

        try:
            diar = self._diarizers.get(seg.stream)
            if diar is None:
                return self._fallback_speaker(seg.stream)

            audio = np.asarray(seg.audio.samples, dtype=np.float32)
            speaker, nsp, best_sim, created = diar.assign_with_debug(
                audio, ts=time.time()
            )
            if str(speaker).startswith('S_ERR'):
                derr = None
                try:
                    derr = diar.last_error()
                except Exception:
                    _logger.exception(
                        'Cannot read diarizer error for %s', seg.stream
                    )
                    derr = None
                log_event(
                    {
                        'type': 'error',
                        'where': 'diar_embed',
                        'stream': seg.stream,
                        'error': derr or 'unknown',
                        'ts': time.time(),
                    }
                )
            log_event(
                {
                    'type': 'diar_debug',
                    'stream': seg.stream,
                    'speaker': speaker,
                    'best_sim': best_sim,
                    'created_new': bool(created),
                    'n_speakers_window': nsp,
                    'seg_dur_s': float(seg.duration_s),
                    'ts': time.time(),
                }
            )
            return self._with_fallback(str(speaker), seg.stream)
        except Exception as error:
            log_event(
                {
                    'type': 'error',
                    'where': 'diar_assign',
                    'error': str(error),
                    'ts': time.time(),
                }
            )
            return self._fallback_speaker(seg.stream)

    def identity_snapshot(self) -> dict[str, dict[str, dict]]:
        snapshots: dict[str, dict[str, dict]] = {}
        model_name = _embedding_backend_name(self.backend)
        for stream, diar in self._diarizers.items():
            items: dict[str, dict] = {}
            try:
                clusters = diar.clusters_snapshot()
            except Exception:
                _logger.exception(
                    'Cannot snapshot diarizer clusters for %s', stream
                )
                clusters = []
            for cluster in clusters:
                label = str(cluster.get('label') or '').strip()
                centroid = cluster.get('centroid')
                if not label or centroid is None:
                    continue
                vector = (
                    np.asarray(centroid, dtype=np.float32).reshape(-1).copy()
                )
                items[label] = {
                    'embedding': vector,
                    'embedding_model': model_name,
                    'embedding_dim': int(vector.shape[0]),
                    'count': int(cluster.get('count') or 0),
                    'last_ts': float(cluster.get('last_ts') or 0.0),
                }
            if items:
                snapshots[str(stream)] = items
        return snapshots

    def _fallback_speaker(self, stream: str) -> str:
        return source_speaker_label(self.source_speaker_labels, stream)

    def _with_fallback(self, speaker: str, stream: str) -> str:
        text = str(speaker or '').strip()
        if not text or text == 'S?' or text.startswith('S_ERR'):
            return self._fallback_speaker(stream)
        return text

    def _maybe_update_pyannote_timeline(
        self, stream: str, log_event: LogEvent
    ) -> None:
        if (
            not self.enabled
            or self.backend != 'pyannote'
            or self._pyannote is None
        ):
            return

        state = self._pyannote_streams[stream]
        now = time.monotonic()
        last = state.last_run_monotonic
        if last is not None and (now - last) < max(0.5, self.step_s):
            return

        ring, t0 = state.snapshot_audio()
        if ring.size < int(6.0 * 16000):
            return

        try:
            state.timeline = self._pyannote.diarize(ring, t_offset=t0)
            state.last_run_monotonic = now
        except Exception as error:
            log_event(
                {
                    'type': 'error',
                    'where': 'diar_run',
                    'error': str(error),
                    'ts': time.time(),
                }
            )
            self.enabled = False

    def _ring_array(self, stream: str) -> np.ndarray:
        state = self._pyannote_streams.get(stream)
        if state is None:
            return np.zeros((0,), dtype=np.float32)
        return state.snapshot_audio()[0]


class DefaultDiarizationRuntimeFactory:
    def __init__(
        self,
        *,
        online_diarizer_factory: OnlineDiarizerFactoryPort,
        pyannote_diarizer_factory: PyannoteDiarizerFactoryPort,
    ) -> None:
        self._online_diarizer_factory = online_diarizer_factory
        self._pyannote_diarizer_factory = pyannote_diarizer_factory

    def __call__(self, *, config: DiarizationConfig) -> DiarizationPort:
        return DiarizationRuntime(
            config=config,
            online_diarizer_factory=self._online_diarizer_factory,
            pyannote_diarizer_factory=self._pyannote_diarizer_factory,
        )


def _embedding_backend_name(backend: DiarBackend) -> str:
    if backend == 'nemo':
        return 'nemo'
    if backend == 'sherpa_onnx':
        return 'sherpa_onnx'
    return 'resemblyzer'
