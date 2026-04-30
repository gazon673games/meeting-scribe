from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

from asr.domain.segments import Segment
from diarization.application.diarization import DiarizationConfig, DiarizationPort
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
        self.temp_dir = Path(config.temp_dir) if config.temp_dir is not None else None
        self.source_speaker_labels = dict(config.source_speaker_labels or {})
        self.sherpa_embedding_model_path = str(config.sherpa_embedding_model_path or "")
        self.sherpa_provider = str(config.sherpa_provider or "cpu")
        self.sherpa_num_threads = max(1, int(config.sherpa_num_threads))
        self._online_diarizer_factory = online_diarizer_factory
        self._pyannote_diarizer_factory = pyannote_diarizer_factory

        self._diarizers: Dict[str, OnlineDiarizerPort] = {}
        self._pyannote: Optional[PyannoteDiarizerPort] = None
        self._ring16: Dict[str, Deque[np.ndarray]] = {}
        self._ring_samples: Dict[str, int] = {}
        self._ring_t0: Dict[str, float] = {}
        self._last_run_ts: Dict[str, float] = {}
        self._timeline: Dict[str, List[DiarSegment]] = {}

    def ensure_stream(self, name: str) -> None:
        if not self.enabled:
            return

        if self.backend in ("online", "nemo", "sherpa_onnx") and name not in self._diarizers:
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

        if self.backend == "pyannote" and name not in self._ring16:
            self._ring16[name] = deque()
            self._ring_samples[name] = 0
            self._ring_t0[name] = 0.0
            self._last_run_ts[name] = 0.0
            self._timeline[name] = []

    def update_ring(self, stream: str, t1: float, x16: np.ndarray) -> None:
        if not self.enabled or self.backend != "pyannote":
            return

        self.ensure_stream(stream)

        x16 = np.asarray(x16, dtype=np.float32).reshape(-1)
        if x16.size == 0:
            return

        max_len = int(max(1.0, self.chunk_s) * 16000)
        chunks = self._ring16.get(stream)
        if chunks is None:
            chunks = deque()
            self._ring16[stream] = chunks
            self._ring_samples[stream] = 0

        chunks.append(x16)
        samples = int(self._ring_samples.get(stream, 0)) + int(x16.size)
        while samples > max_len and chunks:
            cut = samples - max_len
            head = chunks[0]
            if cut >= head.size:
                samples -= int(head.size)
                chunks.popleft()
                continue
            chunks[0] = head[int(cut):]
            samples -= int(cut)
            break

        self._ring_samples[stream] = samples
        self._ring_t0[stream] = float(t1) - (samples / 16000.0)

    def init_backend(self, log_event: LogEvent) -> None:
        if not self.enabled:
            return

        if self.backend == "pyannote":
            try:
                self._pyannote = self._pyannote_diarizer_factory(device=self.device)
                log_event({"type": "diar_init_ok", "backend": "pyannote", "ts": time.time()})
                return
            except Exception as e:
                log_event({"type": "error", "where": "diar_init", "error": str(e), "ts": time.time()})
                self.backend = "nemo"
                self._pyannote = None
                log_event({"type": "diar_fallback", "backend": "nemo", "ts": time.time()})

        if self.backend == "online":
            log_event({"type": "diar_init_ok", "backend": "online", "ts": time.time()})
            return

        if self.backend == "nemo":
            log_event({"type": "diar_init_ok", "backend": "nemo", "ts": time.time()})
            return

        if self.backend == "sherpa_onnx":
            log_event({"type": "diar_init_ok", "backend": "sherpa_onnx", "ts": time.time()})
            return

    def speaker_for_segment(self, seg: Segment, log_event: LogEvent) -> str:
        if not self.enabled:
            return self._fallback_speaker(seg.stream)

        self.ensure_stream(seg.stream)

        if self.backend == "pyannote":
            self._maybe_update_pyannote_timeline(seg.stream, log_event)
            timeline = self._timeline.get(seg.stream, [])
            return self._with_fallback(pick_speaker(timeline, seg.t_start, seg.t_end), seg.stream)

        try:
            diar = self._diarizers.get(seg.stream)
            if diar is None:
                return self._fallback_speaker(seg.stream)

            audio = np.asarray(seg.audio.samples, dtype=np.float32)
            speaker, nsp, best_sim, created = diar.assign_with_debug(audio, ts=time.time())
            if str(speaker).startswith("S_ERR"):
                derr = None
                try:
                    derr = diar.last_error()
                except Exception:
                    derr = None
                log_event(
                    {
                        "type": "error",
                        "where": "diar_embed",
                        "stream": seg.stream,
                        "error": derr or "unknown",
                        "ts": time.time(),
                    }
                )
            log_event(
                {
                    "type": "diar_debug",
                    "stream": seg.stream,
                    "speaker": speaker,
                    "best_sim": best_sim,
                    "created_new": bool(created),
                    "n_speakers_window": nsp,
                    "seg_dur_s": float(seg.duration_s),
                    "ts": time.time(),
                }
            )
            return self._with_fallback(str(speaker), seg.stream)
        except Exception as e:
            log_event({"type": "error", "where": "diar_assign", "error": str(e), "ts": time.time()})
            return self._fallback_speaker(seg.stream)

    def _fallback_speaker(self, stream: str) -> str:
        return source_speaker_label(self.source_speaker_labels, stream)

    def _with_fallback(self, speaker: str, stream: str) -> str:
        text = str(speaker or "").strip()
        if not text or text == "S?" or text.startswith("S_ERR"):
            return self._fallback_speaker(stream)
        return text

    def _maybe_update_pyannote_timeline(self, stream: str, log_event: LogEvent) -> None:
        if not self.enabled or self.backend != "pyannote" or self._pyannote is None:
            return

        now = time.time()
        last = float(self._last_run_ts.get(stream, 0.0))
        if (now - last) < max(0.5, self.step_s):
            return

        if int(self._ring_samples.get(stream, 0)) < int(6.0 * 16000):
            return

        ring = self._ring_array(stream)
        if ring.size < int(6.0 * 16000):
            return

        t0 = float(self._ring_t0.get(stream, 0.0))
        try:
            self._timeline[stream] = self._pyannote.diarize(ring, t_offset=t0)
            self._last_run_ts[stream] = now
        except Exception as e:
            log_event({"type": "error", "where": "diar_run", "error": str(e), "ts": time.time()})
            self.enabled = False

    def _ring_array(self, stream: str) -> np.ndarray:
        chunks = self._ring16.get(stream)
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0].astype(np.float32, copy=False)
        return np.concatenate(list(chunks)).astype(np.float32, copy=False)


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
    if backend == "nemo":
        return "nemo"
    if backend == "sherpa_onnx":
        return "sherpa_onnx"
    return "resemblyzer"
