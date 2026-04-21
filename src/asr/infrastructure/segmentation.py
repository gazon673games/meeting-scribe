from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from asr.application.metrics import ASRMetrics
from asr.application.segmentation import SegmenterConfig
from asr.domain.segments import Segment
from asr.infrastructure.audio_data import MonoAudio16kBuffer
from asr.infrastructure.audio_utils import resample_linear, stereo_to_mono
from asr.infrastructure.gain import PreGainAGC
from asr.infrastructure.vad import EnergyVAD

LogEvent = Callable[[dict], None]
SegmentationParams = Callable[[], Tuple[float, float, float]]


@dataclass
class _StreamState:
    vad: EnergyVAD
    buffer: List[np.ndarray]
    speech_start_ts: Optional[float]
    residual: np.ndarray
    agc: Optional[PreGainAGC]


class AudioSegmenter:
    def __init__(
        self,
        *,
        config: SegmenterConfig,
        segment_queue: "queue.Queue[Segment]",
        diarization,
        metrics: ASRMetrics,
        log_event: LogEvent,
        segmentation_params: SegmentationParams,
    ) -> None:
        self._cfg = config
        self._seg_q = segment_queue
        self._diar = diarization
        self._metrics = metrics
        self._log_event = log_event
        self._segmentation_params = segmentation_params
        self._streams: Dict[str, _StreamState] = {}
        self.pkt_count = 0
        self._last_heartbeat = 0.0

    def reset_runtime(self) -> None:
        self.pkt_count = 0
        self._last_heartbeat = time.time()
        self._streams.clear()

    def feed_packet(self, *, mode: str, pkt: dict) -> None:
        self.pkt_count += 1
        t0 = float(pkt.get("t_start", 0.0))
        t1 = float(pkt.get("t_end", 0.0))
        if mode == "mix":
            block = pkt.get("mix")
            if isinstance(block, np.ndarray):
                self.feed_stream("mix", t0, t1, block)
            return
        for name, block in (pkt.get("sources") or {}).items():
            if isinstance(block, np.ndarray):
                self.feed_stream(str(name), t0, t1, block)

    def feed_stream(self, stream: str, t0: float, t1: float, block_48k: np.ndarray, sample_rate: int = 48000) -> None:
        state = self._ensure_stream(stream)
        x16 = self._preprocess(block_48k, sample_rate, state.agc)
        self._diar.update_ring(stream, t1, x16)
        self._run_vad_loop(stream, state, t0, t1, x16)
        self._heartbeat(stream, state)

    # ── stream lifecycle ───────────────────────────────────────────────

    def _ensure_stream(self, name: str) -> _StreamState:
        if name not in self._streams:
            vad = EnergyVAD(
                sample_rate=16000, frame_ms=20,
                energy_threshold=self._cfg.vad_energy_threshold,
                hangover_ms=self._cfg.vad_hangover_ms,
                min_speech_ms=self._cfg.vad_min_speech_ms,
                adaptive=True, noise_mult=3.0, noise_alpha=0.05,
                band_ratio_min=self._cfg.vad_band_ratio_min,
                voiced_min=self._cfg.vad_voiced_min,
                pre_speech_ms=self._cfg.vad_pre_speech_ms,
                min_end_silence_ms=self._cfg.vad_min_end_silence_ms,
            )
            agc = PreGainAGC(
                target_rms=self._cfg.agc_target_rms,
                max_gain=self._cfg.agc_max_gain,
                alpha=self._cfg.agc_alpha,
            ) if self._cfg.agc_enabled else None
            self._streams[name] = _StreamState(
                vad=vad, buffer=[], speech_start_ts=None,
                residual=np.zeros((0,), dtype=np.float32), agc=agc,
            )
            self._diar.ensure_stream(name)
        return self._streams[name]

    # ── audio processing ───────────────────────────────────────────────

    @staticmethod
    def _preprocess(block: np.ndarray, sample_rate: int, agc: Optional[PreGainAGC]) -> np.ndarray:
        mono = stereo_to_mono(block)
        x16  = resample_linear(mono, sample_rate, 16000)
        return agc.process(x16) if agc is not None else x16

    def _run_vad_loop(self, stream: str, state: _StreamState, t0: float, t1: float, x16: np.ndarray) -> None:
        endpoint_silence_ms, max_segment_s, _ = self._segmentation_params()
        vad = state.vad
        frame_len     = vad.frame_len
        silence_limit = int(endpoint_silence_ms / vad.frame_ms)
        max_frames    = int(max_segment_s * 1000 / vad.frame_ms)

        merged = np.concatenate([state.residual, x16]) if state.residual.size else x16
        total_frames  = merged.size // frame_len
        silence_frames = 0

        for i in range(total_frames):
            frame  = merged[i * frame_len : (i + 1) * frame_len]
            speech = vad.is_speech_frame(frame)
            if speech:
                silence_frames = 0
                if state.speech_start_ts is None:
                    frac = i / max(1, total_frames)
                    state.speech_start_ts = t0 + (t1 - t0) * frac
                    self._prepend_preroll(state)
                state.buffer.append(frame)
            else:
                silence_frames += 1
                if state.speech_start_ts is not None:
                    state.buffer.append(frame)

            if state.speech_start_ts is not None and (
                silence_frames >= silence_limit or len(state.buffer) >= max_frames
            ):
                self._finalize_segment(stream, state, t1)
                vad.reset()
                silence_frames = 0

        state.residual = merged[total_frames * frame_len :]

    @staticmethod
    def _prepend_preroll(state: _StreamState) -> None:
        """Seed the buffer with pre-speech frames captured before VAD triggered."""
        preroll, _ = state.vad.pop_preroll()
        if preroll.size > 0:
            frame_len = state.vad.frame_len
            n = preroll.size // frame_len
            if n > 0:
                aligned = preroll[-(n * frame_len):]
                state.buffer.extend(aligned[j * frame_len : (j + 1) * frame_len] for j in range(n))

    # ── segment finalization ───────────────────────────────────────────

    def _finalize_segment(self, stream: str, state: _StreamState, t_end: float) -> None:
        t_start = state.speech_start_ts
        state.speech_start_ts = None
        frames = state.buffer
        state.buffer = []

        if not frames or t_start is None:
            return
        audio = np.concatenate(frames).astype(np.float32, copy=False)
        if not self._is_segment_valid(state, audio):
            return

        self._log_event({
            "type": "segment_ready", "stream": stream,
            "t_start": t_start, "t_end": t_end,
            "samples": int(audio.shape[0]),
            "dur_s": float(audio.shape[0]) / 16000.0,
            "seg_qsize": int(self._seg_q.qsize()), "ts": time.time(),
        })
        self._enqueue_segment(stream, t_start, t_end, audio)
        self._reseed_overlap(stream, state, audio, t_end)

    def _is_segment_valid(self, state: _StreamState, audio: np.ndarray) -> bool:
        seg_ms = int(round(audio.shape[0] / 16000.0 * 1000.0))
        return seg_ms >= self._cfg.min_segment_ms and state.vad.speech_long_enough()

    def _enqueue_segment(self, stream: str, t_start: float, t_end: float, audio: np.ndarray) -> None:
        try:
            self._seg_q.put_nowait(Segment(
                stream=stream, t_start=float(t_start), t_end=float(t_end),
                audio=MonoAudio16kBuffer.from_array(audio), enqueue_ts=time.time(),
            ))
        except queue.Full:
            self._metrics.record_segment_dropped()
            self._log_event({
                "type": "segment_dropped", "stream": stream,
                "reason": "seg_queue_full",
                "seg_qsize": int(self._seg_q.qsize()), "ts": time.time(),
            })

    def _reseed_overlap(self, stream: str, state: _StreamState, audio: np.ndarray, t_end: float) -> None:
        """Re-seed next segment buffer with tail overlap to preserve cross-boundary context."""
        _, _, overlap_ms = self._segmentation_params()
        overlap_samples = max(0, min(int(round(overlap_ms / 1000.0 * 16000.0)), int(audio.shape[0])))
        if overlap_samples <= 0:
            return
        tail = audio[-overlap_samples:]
        frame_len = self._streams[stream].vad.frame_len
        n = tail.size // frame_len
        if n > 0:
            aligned = tail[-(n * frame_len):]
            state.buffer = [aligned[i * frame_len : (i + 1) * frame_len] for i in range(n)]
            state.speech_start_ts = float(t_end) - (aligned.size / 16000.0)

    # ── heartbeat ──────────────────────────────────────────────────────

    def _heartbeat(self, stream: str, state: _StreamState) -> None:
        now = time.time()
        if now - self._last_heartbeat < 2.0:
            return
        self._last_heartbeat = now
        vad = state.vad
        self._log_event({
            "type": "audio_seen", "stream": stream, "pkts": self.pkt_count,
            "vad": {
                "last_rms": vad.last_rms(), "thr": vad.last_threshold(),
                "noise_rms": vad.noise_rms(), "band_ratio": vad.last_band_ratio(),
                "voiced": vad.last_voiced(),
            },
            "agc_gain":   float(state.agc.gain)        if state.agc is not None else None,
            "agc_in_rms": float(state.agc.last_in_rms) if state.agc is not None else None,
            "ts": now,
        })
