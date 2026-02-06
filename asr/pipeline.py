# --- File: D:\work\own\voice2textTest\asr\pipeline.py ---
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Literal, List, Any

import numpy as np

from asr.logger import ASRLogger
from asr.utils_audio import stereo_to_mono, resample_linear
from asr.vad import EnergyVAD
from asr.diarizer import OnlineDiarizer

Mode = Literal["mix", "split"]


@dataclass
class Segment:
    stream: str
    t_start: float
    t_end: float
    audio_16k: np.ndarray  # mono float32


class ASRPipeline:
    """
    Consumes AudioEngine tap_queue packets and produces ASR events (+ optional diarization).

    - VAD segments speech
    - Faster-Whisper transcribes each segment
    - Online diarizer assigns speaker labels per segment (per stream)
    """

    def __init__(
        self,
        *,
        tap_queue: "queue.Queue[dict]",
        project_root,
        language: str = "ru",
        mode: Mode = "mix",
        source_names: Optional[List[str]] = None,
        asr_model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 5,
        ui_queue: Optional["queue.Queue[dict]"] = None,
        # ---- ASR/VAD quality tuning knobs ----
        endpoint_silence_ms: float = 800.0,
        max_segment_s: float = 12.0,
        overlap_ms: float = 300.0,
        vad_energy_threshold: float = 0.006,
        vad_hangover_ms: int = 400,
        vad_min_speech_ms: int = 350,
        # ---- diarization ----
        diarization_enabled: bool = True,
        diar_sim_threshold: float = 0.74,
        diar_min_segment_s: float = 1.0,
        diar_window_s: float = 120.0,
    ):
        self.tap_q = tap_queue
        self.project_root = project_root
        self.language = language
        self.mode = mode
        self.source_names = source_names

        self.asr_model_name = asr_model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = int(beam_size)

        self.ui_q = ui_queue

        self._stop = threading.Event()
        self._ingest_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._seg_q: "queue.Queue[Segment]" = queue.Queue(maxsize=50)

        self._vads: Dict[str, EnergyVAD] = {}
        self._buffers: Dict[str, List[np.ndarray]] = {}
        self._buf_t0: Dict[str, Optional[float]] = {}
        self._residual_16k: Dict[str, np.ndarray] = {}

        # quality tuning
        self._endpoint_silence_ms = float(endpoint_silence_ms)
        self._max_segment_s = float(max_segment_s)
        self._overlap_ms = float(overlap_ms)

        self._vad_energy_threshold = float(vad_energy_threshold)
        self._vad_hangover_ms = int(vad_hangover_ms)
        self._vad_min_speech_ms = int(vad_min_speech_ms)

        # diarization per stream
        self._diar_enabled = bool(diarization_enabled)
        self._diar_sim_threshold = float(diar_sim_threshold)
        self._diar_min_segment_s = float(diar_min_segment_s)
        self._diar_window_s = float(diar_window_s)
        self._diarizers: Dict[str, OnlineDiarizer] = {}
        self._last_speaker_estimate_ts: float = 0.0

        self.session_id = f"sess_{int(time.time())}"
        self.logger = ASRLogger(
            root=self.project_root,
            session_id=self.session_id,
            language=language,
        )

        self._asr = None
        self._asr_init_error: Optional[str] = None

        self._pkt_count = 0
        self._last_heartbeat = 0.0

    # ================= lifecycle =================

    def start(self) -> None:
        self._stop.clear()
        self._pkt_count = 0
        self._last_heartbeat = time.time()

        self._log_event({
            "type": "asr_started",
            "session_id": self.session_id,
            "language": self.language,
            "mode": self.mode,
            "model": self.asr_model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "endpoint_silence_ms": self._endpoint_silence_ms,
            "max_segment_s": self._max_segment_s,
            "overlap_ms": self._overlap_ms,
            "vad_energy_threshold": self._vad_energy_threshold,
            "vad_hangover_ms": self._vad_hangover_ms,
            "vad_min_speech_ms": self._vad_min_speech_ms,
            "diarization_enabled": self._diar_enabled,
            "diar_sim_threshold": self._diar_sim_threshold,
            "diar_min_segment_s": self._diar_min_segment_s,
            "diar_window_s": self._diar_window_s,
            "ts": time.time(),
        })

        self._ingest_thread = threading.Thread(
            target=self._ingest_loop_safe,
            name="asr-ingest",
            daemon=True,
        )
        self._worker_thread = threading.Thread(
            target=self._worker_loop_safe,
            name="asr-worker",
            daemon=True,
        )
        self._ingest_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ingest_thread:
            self._ingest_thread.join(timeout=2.0)
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

        self._log_event({"type": "asr_stopped", "ts": time.time()})
        self.logger.close()

    # ================= helpers =================

    def _push_ui(self, rec: Dict[str, Any]) -> None:
        if not self.ui_q:
            return
        try:
            self.ui_q.put_nowait(rec)
        except queue.Full:
            pass

    def _log_event(self, rec: Dict[str, Any]) -> None:
        try:
            self.logger.write(rec)
        except Exception:
            pass
        self._push_ui(rec)

    def _ensure_stream(self, name: str) -> None:
        if name not in self._vads:
            self._vads[name] = EnergyVAD(
                sample_rate=16000,
                energy_threshold=self._vad_energy_threshold,
                hangover_ms=self._vad_hangover_ms,
                min_speech_ms=self._vad_min_speech_ms,
            )
            self._buffers[name] = []
            self._buf_t0[name] = None
            self._residual_16k[name] = np.zeros((0,), dtype=np.float32)

        if self._diar_enabled and name not in self._diarizers:
            self._diarizers[name] = OnlineDiarizer(
                similarity_threshold=self._diar_sim_threshold,
                min_segment_s=self._diar_min_segment_s,
                window_s=self._diar_window_s,
            )

    def _heartbeat(self, stream: str, vad: EnergyVAD) -> None:
        now = time.time()
        if now - self._last_heartbeat < 2.0:
            return
        self._last_heartbeat = now
        self._log_event({
            "type": "audio_seen",
            "stream": stream,
            "pkts": self._pkt_count,
            "last_rms": vad.last_rms(),
            "thr": vad.last_threshold(),
            "ts": now,
        })

    # ================= ingest =================

    def _feed_stream(
        self,
        stream: str,
        t0: float,
        t1: float,
        block_48k: np.ndarray,
        sample_rate: int = 48000,
    ) -> None:
        self._ensure_stream(stream)
        vad = self._vads[stream]

        mono = stereo_to_mono(block_48k)
        x16 = resample_linear(mono, sample_rate, 16000)

        frame_len = vad.frame_len
        res = self._residual_16k[stream]
        merged = np.concatenate([res, x16]) if res.size else x16

        total_frames = merged.size // frame_len
        silence_frames = 0
        silence_limit = int(self._endpoint_silence_ms / vad.frame_ms)
        max_frames = int(self._max_segment_s * 1000 / vad.frame_ms)

        for i in range(total_frames):
            fr = merged[i * frame_len:(i + 1) * frame_len]
            speech = vad.is_speech_frame(fr)

            if speech:
                silence_frames = 0
                if self._buf_t0[stream] is None:
                    frac = i / max(1, total_frames)
                    self._buf_t0[stream] = t0 + (t1 - t0) * frac
                self._buffers[stream].append(fr)
            else:
                silence_frames += 1
                if self._buf_t0[stream] is not None:
                    self._buffers[stream].append(fr)

            should_end = (
                (self._buf_t0[stream] is not None) and (
                    silence_frames >= silence_limit or
                    len(self._buffers[stream]) >= max_frames
                )
            )
            if should_end:
                self._finalize_segment(stream, t1)
                vad.reset()
                silence_frames = 0

        used = total_frames * frame_len
        self._residual_16k[stream] = merged[used:]
        self._heartbeat(stream, vad)

    def _finalize_segment(self, stream: str, t_end: float) -> None:
        t_start = self._buf_t0[stream]
        frames = self._buffers[stream]

        if not t_start or not frames:
            self._buf_t0[stream] = None
            self._buffers[stream] = []
            return

        audio = np.concatenate(frames).astype(np.float32, copy=False)

        if not self._vads[stream].speech_long_enough():
            self._buf_t0[stream] = None
            self._buffers[stream] = []
            return

        self._log_event({
            "type": "segment_ready",
            "stream": stream,
            "t_start": t_start,
            "t_end": t_end,
            "samples": int(audio.shape[0]),
            "dur_s": float(audio.shape[0]) / 16000.0,
            "ts": time.time(),
        })

        try:
            self._seg_q.put_nowait(Segment(stream, t_start, t_end, audio))
        except queue.Full:
            self._log_event({
                "type": "segment_dropped",
                "stream": stream,
                "ts": time.time(),
            })

        # overlap: keep tail
        overlap_samples = int(round((self._overlap_ms / 1000.0) * 16000.0))
        overlap_samples = max(0, min(overlap_samples, int(audio.shape[0])))

        if overlap_samples > 0:
            tail = audio[-overlap_samples:]
            vad = self._vads[stream]
            frame_len = vad.frame_len
            n_frames = tail.size // frame_len
            if n_frames > 0:
                tail_used = tail[-(n_frames * frame_len):]
                tail_frames = [
                    tail_used[i * frame_len:(i + 1) * frame_len]
                    for i in range(n_frames)
                ]
                self._buffers[stream] = tail_frames
                self._buf_t0[stream] = float(t_end) - (tail_used.size / 16000.0)
            else:
                self._buffers[stream] = []
                self._buf_t0[stream] = None
        else:
            self._buffers[stream] = []
            self._buf_t0[stream] = None

    def _ingest_loop_safe(self) -> None:
        try:
            self._ingest_loop()
        except Exception as e:
            self._log_event({"type": "error", "where": "ingest", "error": str(e), "ts": time.time()})

    def _ingest_loop(self) -> None:
        while not self._stop.is_set():
            try:
                pkt = self.tap_q.get(timeout=0.2)
            except queue.Empty:
                continue

            self._pkt_count += 1
            t0 = float(pkt.get("t_start", 0.0))
            t1 = float(pkt.get("t_end", 0.0))

            if self.mode == "mix":
                blk = pkt.get("mix")
                if isinstance(blk, np.ndarray):
                    self._feed_stream("mix", t0, t1, blk)
            else:
                for n, blk in (pkt.get("sources") or {}).items():
                    if isinstance(blk, np.ndarray):
                        self._feed_stream(str(n), t0, t1, blk)

    # ================= ASR worker =================

    def _worker_loop_safe(self) -> None:
        try:
            self._worker_loop()
        except Exception as e:
            self._log_event({"type": "error", "where": "worker", "error": str(e), "ts": time.time()})

    def _maybe_emit_speaker_estimate(self, stream: str, n_est: Optional[int]) -> None:
        now = time.time()
        if now - self._last_speaker_estimate_ts < 2.0:
            return
        self._last_speaker_estimate_ts = now
        if n_est is None:
            return
        self._log_event({
            "type": "speaker_estimate",
            "stream": stream,
            "n_speakers": int(n_est),
            "window_s": float(self._diar_window_s),
            "ts": now,
        })

    def _worker_loop(self) -> None:
        self._log_event({
            "type": "asr_init_start",
            "model": self.asr_model_name,
            "device": self.device,
            "ts": time.time(),
        })

        from asr.worker_faster_whisper import FasterWhisperASR

        try:
            self._asr = FasterWhisperASR(
                model_name=self.asr_model_name,
                language=self.language,
                device=self.device,
                compute_type=self.compute_type,
                beam_size=self.beam_size,
            )
            self._log_event({"type": "asr_init_ok", "model": self.asr_model_name, "ts": time.time()})
        except Exception as e:
            self._asr_init_error = str(e)
            self._log_event({"type": "error", "where": "asr_init", "error": str(e), "ts": time.time()})
            return

        # ---- init diarization ONCE (not per segment) ----
        if self._diar_enabled:
            try:
                # constructs encoder; will fail fast if torch/resemblyzer broken
                from asr.diarizer import _ResemblyzerBackend  # noqa
                _ResemblyzerBackend()
                self._log_event({"type": "diar_init_ok", "ts": time.time()})
            except Exception as e:
                self._log_event({"type": "error", "where": "diar_init", "error": str(e), "ts": time.time()})
                self._diar_enabled = False

        while not self._stop.is_set():
            try:
                seg = self._seg_q.get(timeout=0.2)
            except queue.Empty:
                continue

            # ---- diarization per segment ----
            speaker = "S?"
            n_est: Optional[int] = None
            if self._diar_enabled:
                try:
                    diar = self._diarizers.get(seg.stream)
                    if diar is not None:
                        speaker, n_est = diar.assign(seg.audio_16k, ts=time.time())
                        self._maybe_emit_speaker_estimate(seg.stream, n_est)
                except Exception as e:
                    self._log_event({"type": "error", "where": "diar_assign", "error": str(e), "ts": time.time()})
                    speaker = "S?"
                    n_est = None

            t0 = time.time()
            try:
                res = self._asr.transcribe(seg.audio_16k)
                text = (res.get("text") or "").strip()
            except Exception as e:
                self._log_event({
                    "type": "segment",
                    "stream": seg.stream,
                    "speaker": speaker,
                    "t_start": seg.t_start,
                    "t_end": seg.t_end,
                    "text": "",
                    "error": str(e),
                    "ts": time.time(),
                })
                continue

            self._log_event({
                "type": "segment",
                "stream": seg.stream,
                "speaker": speaker,
                "t_start": seg.t_start,
                "t_end": seg.t_end,
                "text": text,
                "latency_s": time.time() - t0,
                "ts": time.time(),
            })
