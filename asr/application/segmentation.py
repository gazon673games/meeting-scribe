from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from asr.application.audio_data import MonoAudio16kBuffer
from asr.domain import Segment
from asr.application.metrics import ASRMetrics
from asr.application.policies import PreGainAGC
from asr.application.utils_audio import resample_linear, stereo_to_mono
from asr.domain.vad import EnergyVAD

LogEvent = Callable[[dict], None]
SegmentationParams = Callable[[], Tuple[float, float, float]]


@dataclass(frozen=True)
class SegmenterConfig:
    vad_energy_threshold: float
    vad_hangover_ms: int
    vad_min_speech_ms: int
    vad_band_ratio_min: float
    vad_voiced_min: float
    vad_pre_speech_ms: int
    vad_min_end_silence_ms: int
    min_segment_ms: int
    agc_enabled: bool
    agc_target_rms: float
    agc_max_gain: float
    agc_alpha: float


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

        self._vads: Dict[str, EnergyVAD] = {}
        self._buffers: Dict[str, List[np.ndarray]] = {}
        self._buf_t0: Dict[str, Optional[float]] = {}
        self._residual_16k: Dict[str, np.ndarray] = {}
        self._agc: Dict[str, PreGainAGC] = {}

        self.pkt_count = 0
        self._last_heartbeat = 0.0

    def reset_runtime(self) -> None:
        self.pkt_count = 0
        self._last_heartbeat = time.time()
        self._vads.clear()
        self._buffers.clear()
        self._buf_t0.clear()
        self._residual_16k.clear()
        self._agc.clear()

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
        self._ensure_stream(stream)
        vad = self._vads[stream]

        mono = stereo_to_mono(block_48k)
        x16 = resample_linear(mono, sample_rate, 16000)

        if self._cfg.agc_enabled:
            agc = self._agc.get(stream)
            if agc is not None:
                x16 = agc.process(x16)

        self._diar.update_ring(stream, t1, x16)

        frame_len = vad.frame_len
        residual = self._residual_16k[stream]
        merged = np.concatenate([residual, x16]) if residual.size else x16

        endpoint_silence_ms, max_segment_s, _overlap_ms = self._segmentation_params()

        total_frames = merged.size // frame_len
        silence_frames = 0
        silence_limit = int(endpoint_silence_ms / vad.frame_ms)
        max_frames = int(max_segment_s * 1000 / vad.frame_ms)

        for i in range(total_frames):
            frame = merged[i * frame_len : (i + 1) * frame_len]
            speech = vad.is_speech_frame(frame)

            if speech:
                silence_frames = 0
                if self._buf_t0[stream] is None:
                    frac = i / max(1, total_frames)
                    self._buf_t0[stream] = t0 + (t1 - t0) * frac

                    preroll, _n = vad.pop_preroll()
                    if preroll.size > 0:
                        n_frames = preroll.size // frame_len
                        if n_frames > 0:
                            prer = preroll[-(n_frames * frame_len) :]
                            pre_frames = [prer[j * frame_len : (j + 1) * frame_len] for j in range(n_frames)]
                            self._buffers[stream].extend(pre_frames)

                self._buffers[stream].append(frame)
            else:
                silence_frames += 1
                if self._buf_t0[stream] is not None:
                    self._buffers[stream].append(frame)

            should_end = (self._buf_t0[stream] is not None) and (
                silence_frames >= silence_limit or len(self._buffers[stream]) >= max_frames
            )
            if should_end:
                self._finalize_segment(stream, t1)
                vad.reset()
                silence_frames = 0

        used = total_frames * frame_len
        self._residual_16k[stream] = merged[used:]
        self._heartbeat(stream, vad)

    def _ensure_stream(self, name: str) -> None:
        if name not in self._vads:
            self._vads[name] = EnergyVAD(
                sample_rate=16000,
                frame_ms=20,
                energy_threshold=self._cfg.vad_energy_threshold,
                hangover_ms=self._cfg.vad_hangover_ms,
                min_speech_ms=self._cfg.vad_min_speech_ms,
                adaptive=True,
                noise_mult=3.0,
                noise_alpha=0.05,
                band_ratio_min=self._cfg.vad_band_ratio_min,
                voiced_min=self._cfg.vad_voiced_min,
                pre_speech_ms=self._cfg.vad_pre_speech_ms,
                min_end_silence_ms=self._cfg.vad_min_end_silence_ms,
            )
            self._buffers[name] = []
            self._buf_t0[name] = None
            self._residual_16k[name] = np.zeros((0,), dtype=np.float32)

        if self._cfg.agc_enabled and name not in self._agc:
            self._agc[name] = PreGainAGC(
                target_rms=self._cfg.agc_target_rms,
                max_gain=self._cfg.agc_max_gain,
                alpha=self._cfg.agc_alpha,
            )

        self._diar.ensure_stream(name)

    def _heartbeat(self, stream: str, vad: EnergyVAD) -> None:
        now = time.time()
        if now - self._last_heartbeat < 2.0:
            return
        self._last_heartbeat = now

        agc = self._agc.get(stream)
        self._log_event(
            {
                "type": "audio_seen",
                "stream": stream,
                "pkts": self.pkt_count,
                "vad": {
                    "last_rms": vad.last_rms(),
                    "thr": vad.last_threshold(),
                    "noise_rms": vad.noise_rms(),
                    "band_ratio": vad.last_band_ratio(),
                    "voiced": vad.last_voiced(),
                },
                "agc_gain": float(agc.gain) if agc is not None else None,
                "agc_in_rms": float(agc.last_in_rms) if agc is not None else None,
                "ts": now,
            }
        )

    def _finalize_segment(self, stream: str, t_end: float) -> None:
        t_start = self._buf_t0[stream]
        frames = self._buffers[stream]

        if t_start is None or not frames:
            self._buf_t0[stream] = None
            self._buffers[stream] = []
            return

        audio = np.concatenate(frames).astype(np.float32, copy=False)

        seg_ms = int(round((audio.shape[0] / 16000.0) * 1000.0))
        if seg_ms < self._cfg.min_segment_ms:
            self._buf_t0[stream] = None
            self._buffers[stream] = []
            return

        if not self._vads[stream].speech_long_enough():
            self._buf_t0[stream] = None
            self._buffers[stream] = []
            return

        self._log_event(
            {
                "type": "segment_ready",
                "stream": stream,
                "t_start": t_start,
                "t_end": t_end,
                "samples": int(audio.shape[0]),
                "dur_s": float(audio.shape[0]) / 16000.0,
                "seg_qsize": int(self._seg_q.qsize()),
                "ts": time.time(),
            }
        )

        try:
            self._seg_q.put_nowait(
                Segment(
                    stream=stream,
                    t_start=float(t_start),
                    t_end=float(t_end),
                    audio=MonoAudio16kBuffer.from_array(audio),
                    enqueue_ts=time.time(),
                )
            )
        except queue.Full:
            self._metrics.record_segment_dropped()
            self._log_event(
                {
                    "type": "segment_dropped",
                    "stream": stream,
                    "reason": "seg_queue_full",
                    "seg_qsize": int(self._seg_q.qsize()),
                    "ts": time.time(),
                }
            )

        _, _max_seg_s, overlap_ms = self._segmentation_params()
        overlap_samples = int(round((overlap_ms / 1000.0) * 16000.0))
        overlap_samples = max(0, min(overlap_samples, int(audio.shape[0])))

        if overlap_samples > 0:
            tail = audio[-overlap_samples:]
            frame_len = self._vads[stream].frame_len
            n_frames = tail.size // frame_len
            if n_frames > 0:
                tail_used = tail[-(n_frames * frame_len) :]
                tail_frames = [tail_used[i * frame_len : (i + 1) * frame_len] for i in range(n_frames)]
                self._buffers[stream] = tail_frames
                self._buf_t0[stream] = float(t_end) - (tail_used.size / 16000.0)
                return

        self._buffers[stream] = []
        self._buf_t0[stream] = None
