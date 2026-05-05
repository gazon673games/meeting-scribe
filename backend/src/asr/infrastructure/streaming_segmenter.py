from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from asr.application.segmentation import StreamingSegmenterConfig
from asr.domain.streaming import StreamingChunk
from asr.infrastructure.audio_data import MonoAudio16kBuffer
from asr.infrastructure.audio_utils import resample_linear, stereo_to_mono
from asr.infrastructure.gain import PreGainAGC
from asr.infrastructure.vad import EnergyVAD

LogEvent = Callable[[dict], None]


@dataclass
class _StreamState:
    vad: EnergyVAD
    buffer: List[np.ndarray]
    speech_start_ts: Optional[float]
    last_chunk_ts: float
    residual: np.ndarray
    agc: Optional[PreGainAGC]


class StreamingAudioSegmenter:
    """
    Feeds audio into VAD and emits StreamingChunk objects:
      - Intermediate chunks every `chunk_interval_s` during active speech (is_final=False)
      - A final chunk after `endpoint_silence_ms` of silence (is_final=True)

    Each chunk carries the full audio buffer since speech start so the
    transcription worker always has complete context for Whisper.
    """

    def __init__(
        self,
        *,
        config: StreamingSegmenterConfig,
        chunk_queue: "queue.Queue[StreamingChunk]",
        log_event: LogEvent,
    ) -> None:
        self._cfg = config
        self._chunk_q = chunk_queue
        self._log_event = log_event
        self._streams: Dict[str, _StreamState] = {}
        self.pkt_count = 0

    def reset_runtime(self) -> None:
        self.pkt_count = 0
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
        mono = stereo_to_mono(block_48k)
        x16 = resample_linear(mono, sample_rate, 16000)
        if state.agc is not None:
            x16 = state.agc.process(x16)
        self._run_vad_loop(stream, state, t0, t1, x16)

    # ── stream lifecycle ───────────────────────────────────────────────

    def _ensure_stream(self, name: str) -> _StreamState:
        if name not in self._streams:
            cfg = self._cfg
            vad = EnergyVAD(
                sample_rate=16000, frame_ms=20,
                energy_threshold=cfg.vad_energy_threshold,
                hangover_ms=cfg.vad_hangover_ms,
                min_speech_ms=cfg.vad_min_speech_ms,
                adaptive=True, noise_mult=3.0, noise_alpha=0.05,
                band_ratio_min=cfg.vad_band_ratio_min,
                voiced_min=cfg.vad_voiced_min,
                pre_speech_ms=cfg.vad_pre_speech_ms,
                min_end_silence_ms=cfg.vad_min_end_silence_ms,
            )
            agc = PreGainAGC(
                target_rms=cfg.agc_target_rms,
                max_gain=cfg.agc_max_gain,
                alpha=cfg.agc_alpha,
            ) if cfg.agc_enabled else None
            self._streams[name] = _StreamState(
                vad=vad, buffer=[], speech_start_ts=None,
                last_chunk_ts=0.0,
                residual=np.zeros((0,), dtype=np.float32), agc=agc,
            )
        return self._streams[name]

    # ── VAD loop ───────────────────────────────────────────────────────

    def _run_vad_loop(self, stream: str, state: _StreamState, t0: float, t1: float, x16: np.ndarray) -> None:
        vad = state.vad
        frame_len = vad.frame_len
        silence_limit = int(self._cfg.endpoint_silence_ms / vad.frame_ms)
        max_frames = int(self._cfg.max_segment_s * 1000 / vad.frame_ms)

        merged = np.concatenate([state.residual, x16]) if state.residual.size else x16
        total_frames = merged.size // frame_len
        silence_frames = 0

        for i in range(total_frames):
            frame = merged[i * frame_len : (i + 1) * frame_len]
            speech = vad.is_speech_frame(frame)

            if speech:
                silence_frames = 0
                if state.speech_start_ts is None:
                    frac = i / max(1, total_frames)
                    state.speech_start_ts = t0 + (t1 - t0) * frac
                    state.last_chunk_ts = time.time()
                    _prepend_preroll(state)
                state.buffer.append(frame)
            else:
                silence_frames += 1
                if state.speech_start_ts is not None:
                    state.buffer.append(frame)

            if state.speech_start_ts is not None and (
                silence_frames >= silence_limit or len(state.buffer) >= max_frames
            ):
                self._emit_chunk(stream, state, t1, is_final=True)
                vad.reset()
                silence_frames = 0

        # Emit intermediate chunk if the speech window exceeds chunk_interval_s
        if state.speech_start_ts is not None and state.buffer:
            elapsed = time.time() - state.last_chunk_ts
            if elapsed >= self._cfg.chunk_interval_s:
                self._emit_chunk(stream, state, t1, is_final=False)

        state.residual = merged[total_frames * frame_len:]

    # ── chunk emission ─────────────────────────────────────────────────

    def _emit_chunk(self, stream: str, state: _StreamState, t_end: float, is_final: bool) -> None:
        if state.speech_start_ts is None or not state.buffer:
            return

        audio = np.concatenate(state.buffer).astype(np.float32, copy=False)
        chunk = StreamingChunk(
            stream=stream,
            t_start=float(state.speech_start_ts),
            t_end=float(t_end),
            audio=MonoAudio16kBuffer.from_array(audio),
            is_final=is_final,
            enqueue_ts=time.time(),
        )
        try:
            self._chunk_q.put_nowait(chunk)
        except queue.Full:
            self._log_event({
                "type": "streaming_chunk_dropped",
                "stream": stream,
                "reason": "queue_full",
                "ts": time.time(),
            })

        state.last_chunk_ts = time.time()

        if is_final:
            state.speech_start_ts = None
            state.buffer = []
            state.last_chunk_ts = 0.0


def _prepend_preroll(state: _StreamState) -> None:
    preroll, _ = state.vad.pop_preroll()
    if preroll.size == 0:
        return
    frame_len = state.vad.frame_len
    n = preroll.size // frame_len
    if n > 0:
        aligned = preroll[-(n * frame_len):]
        state.buffer.extend(aligned[j * frame_len : (j + 1) * frame_len] for j in range(n))
