from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np

from asr.diar_backend_pyannote import PyannoteDiarizer
from asr.diarizer import OnlineDiarizer
from asr.domain import DiarBackend, DiarSegment, Segment, pick_speaker

LogEvent = Callable[[dict], None]


class DiarizationRuntime:
    def __init__(
        self,
        *,
        enabled: bool,
        backend: DiarBackend,
        sim_threshold: float,
        min_segment_s: float,
        window_s: float,
        chunk_s: float,
        step_s: float,
        device: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.backend: DiarBackend = backend
        self.sim_threshold = float(sim_threshold)
        self.min_segment_s = float(min_segment_s)
        self.window_s = float(window_s)
        self.chunk_s = float(chunk_s)
        self.step_s = float(step_s)
        self.device = str(device)

        self._diarizers: Dict[str, OnlineDiarizer] = {}
        self._pyannote: Optional[PyannoteDiarizer] = None
        self._ring16: Dict[str, np.ndarray] = {}
        self._ring_t0: Dict[str, float] = {}
        self._last_run_ts: Dict[str, float] = {}
        self._timeline: Dict[str, List[DiarSegment]] = {}

    def ensure_stream(self, name: str) -> None:
        if not self.enabled:
            return

        if self.backend in ("online", "nemo") and name not in self._diarizers:
            self._diarizers[name] = OnlineDiarizer(
                similarity_threshold=self.sim_threshold,
                min_segment_s=self.min_segment_s,
                window_s=self.window_s,
                backend=("nemo" if self.backend == "nemo" else "resemblyzer"),
                device=self.device,
            )

        if self.backend == "pyannote" and name not in self._ring16:
            self._ring16[name] = np.zeros((0,), dtype=np.float32)
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

        ring = self._ring16.get(stream, np.zeros((0,), dtype=np.float32))
        ring = np.concatenate([ring, x16]) if ring.size else x16

        max_len = int(max(1.0, self.chunk_s) * 16000)
        if ring.size > max_len:
            cut = ring.size - max_len
            ring = ring[cut:]

        self._ring16[stream] = ring
        self._ring_t0[stream] = float(t1) - (ring.size / 16000.0)

    def init_backend(self, log_event: LogEvent) -> None:
        if not self.enabled:
            return

        if self.backend == "pyannote":
            try:
                self._pyannote = PyannoteDiarizer(device=self.device)
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

    def speaker_for_segment(self, seg: Segment, log_event: LogEvent) -> str:
        if not self.enabled:
            return "S?"

        self.ensure_stream(seg.stream)

        if self.backend == "pyannote":
            self._maybe_update_pyannote_timeline(seg.stream, log_event)
            timeline = self._timeline.get(seg.stream, [])
            return pick_speaker(timeline, seg.t_start, seg.t_end)

        try:
            diar = self._diarizers.get(seg.stream)
            if diar is None:
                return "S?"

            speaker, nsp, best_sim, created = diar.assign_with_debug(seg.audio_16k, ts=time.time())
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
                    "seg_dur_s": float(seg.audio_16k.shape[0]) / 16000.0,
                    "ts": time.time(),
                }
            )
            return str(speaker)
        except Exception as e:
            log_event({"type": "error", "where": "diar_assign", "error": str(e), "ts": time.time()})
            return "S?"

    def _maybe_update_pyannote_timeline(self, stream: str, log_event: LogEvent) -> None:
        if not self.enabled or self.backend != "pyannote" or self._pyannote is None:
            return

        now = time.time()
        last = float(self._last_run_ts.get(stream, 0.0))
        if (now - last) < max(0.5, self.step_s):
            return

        ring = self._ring16.get(stream)
        if ring is None or ring.size < int(6.0 * 16000):
            return

        t0 = float(self._ring_t0.get(stream, 0.0))
        try:
            self._timeline[stream] = self._pyannote.diarize(ring, t_offset=t0)
            self._last_run_ts[stream] = now
        except Exception as e:
            log_event({"type": "error", "where": "diar_run", "error": str(e), "ts": time.time()})
            self.enabled = False
