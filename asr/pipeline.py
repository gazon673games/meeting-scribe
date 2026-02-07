# --- File: D:\work\own\voice2textTest\asr\pipeline.py ---
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Literal, List, Any, Tuple

import numpy as np

from asr.logger import ASRLogger
from asr.utils_audio import stereo_to_mono, resample_linear
from asr.vad import EnergyVAD
from asr.diarizer import OnlineDiarizer

Mode = Literal["mix", "split"]
DiarBackend = Literal["pyannote", "online", "nemo"]


@dataclass
class Segment:
    stream: str
    t_start: float
    t_end: float
    audio_16k: np.ndarray


@dataclass
class DiarSegment:
    t0: float
    t1: float
    speaker: str


def _pick_speaker(timeline: List[DiarSegment], t0: float, t1: float) -> str:
    best_label = "S?"
    best_ov = 0.0
    for s in timeline:
        a = max(float(t0), float(s.t0))
        b = min(float(t1), float(s.t1))
        ov = b - a
        if ov > best_ov:
            best_ov = ov
            best_label = str(s.speaker)
    return best_label if best_ov > 0 else "S?"


class _PyannoteBackend:
    def __init__(self, device: str = "cuda") -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "pyannote.audio is not installed or failed to import.\n"
                "Install example:\n"
                "  pip install pyannote.audio\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "pyannote.audio requires torch.\n"
                "Install torch compatible with your CUDA.\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        self._torch = torch
        self._device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        try:
            self._pipeline.to(self._device)
        except Exception:
            pass

    def diarize(self, audio_16k: np.ndarray, *, t_offset: float = 0.0) -> List[DiarSegment]:
        x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return []
        wav = self._torch.from_numpy(x).unsqueeze(0)
        diar = self._pipeline({"waveform": wav, "sample_rate": 16000})

        out: List[DiarSegment] = []
        for turn, _, label in diar.itertracks(yield_label=True):
            out.append(DiarSegment(t0=float(t_offset) + float(turn.start), t1=float(t_offset) + float(turn.end), speaker=str(label)))
        return out


class _PreGainAGC:
    def __init__(self, target_rms: float = 0.06, max_gain: float = 6.0, alpha: float = 0.02):
        self.target_rms = float(target_rms)
        self.max_gain = float(max_gain)
        self.alpha = float(alpha)
        self.gain = 1.0
        self.last_in_rms = 0.0

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        xf = x.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(xf * xf)))

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        r = self._rms(x)
        self.last_in_rms = r

        if r > 1e-7:
            desired = self.target_rms / r
            desired = max(1.0 / self.max_gain, min(self.max_gain, desired))
            self.gain = (1.0 - self.alpha) * self.gain + self.alpha * desired

        y = x * float(self.gain)
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    # normalize whitespace; keep punctuation as-is
    s = " ".join(s.split())
    return s


def _trim_overlap(prev_text: str, cur_text: str, *, max_window: int = 80, min_match: int = 8) -> Tuple[str, int]:
    """
    Removes duplicated prefix from cur_text if it matches a suffix of prev_text.
    Returns: (trimmed_text, removed_chars_count)
    """
    p = _norm_text(prev_text)
    c = _norm_text(cur_text)

    if not p or not c:
        return (c, 0)

    # compare on normalized, but we will trim on normalized too (fine for UI/logs)
    ps = p[-max_window:]
    best = 0
    # find longest k such that ps[-k:] == c[:k]
    max_k = min(len(ps), len(c))
    for k in range(min(max_k, max_window), min_match - 1, -1):
        if ps[-k:] == c[:k]:
            best = k
            break

    if best > 0:
        trimmed = c[best:].lstrip()
        return (trimmed, best)

    return (c, 0)


@dataclass
class _AdaptiveBeam:
    """
    Simple global adaptive beam controller.
      - If backlog or latency ratio high -> decrease beam
      - If stable/idle -> slowly increase beam
    """
    min_beam: int = 1
    max_beam: int = 5
    cur_beam: int = 5

    backlog_hi: int = 12          # seg queue size to trigger downshift
    backlog_lo: int = 2           # seg queue size to allow upshift
    latency_ratio_hi: float = 1.1 # latency > dur * ratio -> downshift
    latency_ratio_lo: float = 0.7 # latency < dur * ratio -> upshift

    cool_down_s: float = 2.0
    last_change_ts: float = 0.0

    def maybe_update(self, *, seg_qsize: int, last_latency_s: float, last_dur_s: float, now: float) -> Tuple[int, Optional[str]]:
        if (now - float(self.last_change_ts)) < float(self.cool_down_s):
            return (int(self.cur_beam), None)

        dur = max(1e-6, float(last_dur_s))
        ratio = float(last_latency_s) / dur

        reason = None

        if seg_qsize >= int(self.backlog_hi) or ratio >= float(self.latency_ratio_hi):
            if self.cur_beam > self.min_beam:
                self.cur_beam -= 1
                self.last_change_ts = now
                reason = f"downshift (q={seg_qsize}, lat_ratio={ratio:.2f})"
        elif seg_qsize <= int(self.backlog_lo) and ratio <= float(self.latency_ratio_lo):
            if self.cur_beam < self.max_beam:
                self.cur_beam += 1
                self.last_change_ts = now
                reason = f"upshift (q={seg_qsize}, lat_ratio={ratio:.2f})"

        return (int(self.cur_beam), reason)


class ASRPipeline:
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
        endpoint_silence_ms: float = 800.0,
        max_segment_s: float = 12.0,
        overlap_ms: float = 300.0,
        vad_energy_threshold: float = 0.006,
        vad_hangover_ms: int = 400,
        vad_min_speech_ms: int = 350,
        diarization_enabled: bool = True,
        diar_backend: DiarBackend = "pyannote",
        diar_sim_threshold: float = 0.74,
        diar_min_segment_s: float = 1.0,
        diar_window_s: float = 120.0,
        diar_chunk_s: float = 30.0,
        diar_step_s: float = 10.0,
        agc_enabled: bool = True,
        agc_target_rms: float = 0.06,
        agc_max_gain: float = 6.0,
        agc_alpha: float = 0.02,
        # NEW (Step 3): overlap text dedup + adaptive beam
        text_dedup_enabled: bool = True,
        text_dedup_window: int = 80,
        adaptive_beam_enabled: bool = True,
        adaptive_beam_min: int = 1,
        adaptive_beam_max: Optional[int] = None,  # default -> beam_size
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

        self._endpoint_silence_ms = float(endpoint_silence_ms)
        self._max_segment_s = float(max_segment_s)
        self._overlap_ms = float(overlap_ms)

        self._vad_energy_threshold = float(vad_energy_threshold)
        self._vad_hangover_ms = int(vad_hangover_ms)
        self._vad_min_speech_ms = int(vad_min_speech_ms)

        self._diar_enabled = bool(diarization_enabled)
        self._diar_backend: DiarBackend = diar_backend

        self._diar_sim_threshold = float(diar_sim_threshold)
        self._diar_min_segment_s = float(diar_min_segment_s)
        self._diar_window_s = float(diar_window_s)

        self._diar_chunk_s = float(diar_chunk_s)
        self._diar_step_s = float(diar_step_s)

        self._diarizers: Dict[str, OnlineDiarizer] = {}
        self._pyannote: Optional[_PyannoteBackend] = None

        self._ring16: Dict[str, np.ndarray] = {}
        self._ring_t0: Dict[str, float] = {}
        self._diar_last_run_ts: Dict[str, float] = {}
        self._timeline: Dict[str, List[DiarSegment]] = {}

        self._agc_enabled = bool(agc_enabled)
        self._agc_target_rms = float(agc_target_rms)
        self._agc_max_gain = float(agc_max_gain)
        self._agc_alpha = float(agc_alpha)
        self._agc: Dict[str, _PreGainAGC] = {}

        # Step 3: overlap text dedup state (per stream)
        self._text_dedup_enabled = bool(text_dedup_enabled)
        self._text_dedup_window = int(text_dedup_window)
        self._last_text: Dict[str, str] = {}

        # Step 3: adaptive beam controller
        self._adaptive_beam_enabled = bool(adaptive_beam_enabled)
        maxb = int(adaptive_beam_max) if adaptive_beam_max is not None else int(self.beam_size)
        maxb = max(1, maxb)
        self._beam_ctl = _AdaptiveBeam(
            min_beam=max(1, int(adaptive_beam_min)),
            max_beam=maxb,
            cur_beam=max(1, min(int(self.beam_size), maxb)),
        )

        self.session_id = f"sess_{int(time.time())}"
        self.logger = ASRLogger(root=self.project_root, session_id=self.session_id, language=language)

        self._asr = None
        self._asr_init_error: Optional[str] = None

        self._pkt_count = 0
        self._last_heartbeat = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._pkt_count = 0
        self._last_heartbeat = time.time()

        self._log_event(
            {
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
                "vad_adaptive": True,
                "diarization_enabled": self._diar_enabled,
                "diar_backend": self._diar_backend,
                "agc_enabled": self._agc_enabled,
                "text_dedup_enabled": self._text_dedup_enabled,
                "adaptive_beam_enabled": self._adaptive_beam_enabled,
                "adaptive_beam": {
                    "min": self._beam_ctl.min_beam,
                    "max": self._beam_ctl.max_beam,
                    "cur": self._beam_ctl.cur_beam,
                },
                "ts": time.time(),
            }
        )

        self._ingest_thread = threading.Thread(target=self._ingest_loop_safe, name="asr-ingest", daemon=True)
        self._worker_thread = threading.Thread(target=self._worker_loop_safe, name="asr-worker", daemon=True)
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
                adaptive=True,
                noise_mult=3.0,
                noise_alpha=0.05,
            )
            self._buffers[name] = []
            self._buf_t0[name] = None
            self._residual_16k[name] = np.zeros((0,), dtype=np.float32)

        if self._agc_enabled and name not in self._agc:
            self._agc[name] = _PreGainAGC(
                target_rms=self._agc_target_rms,
                max_gain=self._agc_max_gain,
                alpha=self._agc_alpha,
            )

        if self._diar_enabled and self._diar_backend in ("online", "nemo") and name not in self._diarizers:
            self._diarizers[name] = OnlineDiarizer(
                similarity_threshold=self._diar_sim_threshold,
                min_segment_s=self._diar_min_segment_s,
                window_s=self._diar_window_s,
                backend=("nemo" if self._diar_backend == "nemo" else "resemblyzer"),
                device=self.device,
            )

        if self._diar_enabled and self._diar_backend == "pyannote":
            if name not in self._ring16:
                self._ring16[name] = np.zeros((0,), dtype=np.float32)
                self._ring_t0[name] = 0.0
                self._diar_last_run_ts[name] = 0.0
                self._timeline[name] = []

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
                "pkts": self._pkt_count,
                "last_rms": vad.last_rms(),
                "thr": vad.last_threshold(),
                "noise_rms": vad.noise_rms(),
                "agc_gain": float(agc.gain) if agc is not None else None,
                "agc_in_rms": float(agc.last_in_rms) if agc is not None else None,
                "beam_cur": int(self._beam_ctl.cur_beam),
                "seg_qsize": int(self._seg_q.qsize()),
                "ts": now,
            }
        )

    def _update_ring(self, stream: str, t0: float, t1: float, x16: np.ndarray) -> None:
        if not self._diar_enabled or self._diar_backend != "pyannote":
            return

        self._ensure_stream(stream)

        x16 = np.asarray(x16, dtype=np.float32).reshape(-1)
        if x16.size == 0:
            return

        ring = self._ring16.get(stream, np.zeros((0,), dtype=np.float32))
        ring = np.concatenate([ring, x16]) if ring.size else x16

        max_len = int(max(1.0, self._diar_chunk_s) * 16000)
        if ring.size > max_len:
            cut = ring.size - max_len
            ring = ring[cut:]

        ring_t0 = float(t1) - (ring.size / 16000.0)

        self._ring16[stream] = ring
        self._ring_t0[stream] = ring_t0

    def _feed_stream(self, stream: str, t0: float, t1: float, block_48k: np.ndarray, sample_rate: int = 48000) -> None:
        self._ensure_stream(stream)
        vad = self._vads[stream]

        mono = stereo_to_mono(block_48k)
        x16 = resample_linear(mono, sample_rate, 16000)

        if self._agc_enabled:
            agc = self._agc.get(stream)
            if agc is not None:
                x16 = agc.process(x16)

        self._update_ring(stream, t0, t1, x16)

        frame_len = vad.frame_len
        res = self._residual_16k[stream]
        merged = np.concatenate([res, x16]) if res.size else x16

        total_frames = merged.size // frame_len
        silence_frames = 0
        silence_limit = int(self._endpoint_silence_ms / vad.frame_ms)
        max_frames = int(self._max_segment_s * 1000 / vad.frame_ms)

        for i in range(total_frames):
            fr = merged[i * frame_len : (i + 1) * frame_len]
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

    def _finalize_segment(self, stream: str, t_end: float) -> None:
        t_start = self._buf_t0[stream]
        frames = self._buffers[stream]

        if t_start is None or not frames:
            self._buf_t0[stream] = None
            self._buffers[stream] = []
            return

        audio = np.concatenate(frames).astype(np.float32, copy=False)

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
                "ts": time.time(),
            }
        )

        try:
            self._seg_q.put_nowait(Segment(stream, float(t_start), float(t_end), audio))
        except queue.Full:
            self._log_event({"type": "segment_dropped", "stream": stream, "ts": time.time()})

        overlap_samples = int(round((self._overlap_ms / 1000.0) * 16000.0))
        overlap_samples = max(0, min(overlap_samples, int(audio.shape[0])))

        if overlap_samples > 0:
            tail = audio[-overlap_samples:]
            vad = self._vads[stream]
            frame_len = vad.frame_len
            n_frames = tail.size // frame_len
            if n_frames > 0:
                tail_used = tail[-(n_frames * frame_len) :]
                tail_frames = [tail_used[i * frame_len : (i + 1) * frame_len] for i in range(n_frames)]
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

    def _init_diarization_backend(self) -> None:
        if not self._diar_enabled:
            return

        if self._diar_backend == "pyannote":
            try:
                self._pyannote = _PyannoteBackend(device=self.device)
                self._log_event({"type": "diar_init_ok", "backend": "pyannote", "ts": time.time()})
                return
            except Exception as e:
                self._log_event({"type": "error", "where": "diar_init", "error": str(e), "ts": time.time()})
                self._diar_backend = "nemo"
                self._pyannote = None
                self._log_event({"type": "diar_fallback", "backend": "nemo", "ts": time.time()})

        if self._diar_backend == "online":
            self._log_event({"type": "diar_init_ok", "backend": "online", "ts": time.time()})
            return

        if self._diar_backend == "nemo":
            self._log_event({"type": "diar_init_ok", "backend": "nemo", "ts": time.time()})
            return

    def _maybe_update_pyannote_timeline(self, stream: str) -> None:
        if not self._diar_enabled or self._diar_backend != "pyannote" or self._pyannote is None:
            return

        now = time.time()
        last = float(self._diar_last_run_ts.get(stream, 0.0))
        if (now - last) < max(0.5, float(self._diar_step_s)):
            return

        ring = self._ring16.get(stream)
        if ring is None or ring.size < int(6.0 * 16000):
            return

        t0 = float(self._ring_t0.get(stream, 0.0))
        try:
            tl = self._pyannote.diarize(ring, t_offset=t0)
            self._timeline[stream] = tl
            self._diar_last_run_ts[stream] = now
        except Exception as e:
            self._log_event({"type": "error", "where": "diar_run", "error": str(e), "ts": time.time()})
            self._diar_enabled = False

    def _worker_loop_safe(self) -> None:
        try:
            self._worker_loop()
        except Exception as e:
            self._log_event({"type": "error", "where": "worker", "error": str(e), "ts": time.time()})

    def _worker_loop(self) -> None:
        self._log_event({"type": "asr_init_start", "model": self.asr_model_name, "device": self.device, "ts": time.time()})

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

        self._init_diarization_backend()

        while not self._stop.is_set():
            try:
                seg = self._seg_q.get(timeout=0.2)
            except queue.Empty:
                continue

            speaker = "S?"

            if self._diar_enabled:
                if self._diar_backend == "pyannote":
                    self._maybe_update_pyannote_timeline(seg.stream)
                    tl = self._timeline.get(seg.stream, [])
                    speaker = _pick_speaker(tl, seg.t_start, seg.t_end)
                else:
                    try:
                        diar = self._diarizers.get(seg.stream)
                        if diar is not None:
                            speaker, nsp, best_sim, created = diar.assign_with_debug(seg.audio_16k, ts=time.time())
                            if str(speaker).startswith("S_ERR"):
                                derr = None
                                try:
                                    derr = diar.last_error()
                                except Exception:
                                    derr = None
                                self._log_event({"type": "error", "where": "diar_embed", "stream": seg.stream, "error": derr or "unknown", "ts": time.time()})
                            self._log_event({"type": "diar_debug", "stream": seg.stream, "speaker": speaker, "best_sim": best_sim, "created_new": bool(created), "n_speakers_window": nsp, "seg_dur_s": float(seg.audio_16k.shape[0]) / 16000.0, "ts": time.time()})
                    except Exception as e:
                        self._log_event({"type": "error", "where": "diar_assign", "error": str(e), "ts": time.time()})
                        speaker = "S?"

            seg_dur_s = max(1e-6, float(seg.audio_16k.shape[0]) / 16000.0)

            # Adaptive beam selection based on backlog + recent latency ratio
            beam_to_use = int(self._beam_ctl.cur_beam)

            t0 = time.time()
            try:
                res = self._asr.transcribe(seg.audio_16k, beam_size=beam_to_use)
                text = (res.get("text") or "").strip()
            except Exception as e:
                self._log_event({"type": "segment", "stream": seg.stream, "speaker": speaker, "t_start": seg.t_start, "t_end": seg.t_end, "text": "", "error": str(e), "ts": time.time()})
                continue

            latency_s = time.time() - t0

            # Step 3: overlap text dedup
            removed = 0
            if self._text_dedup_enabled:
                prev = self._last_text.get(seg.stream, "")
                trimmed, removed = _trim_overlap(prev, text, max_window=self._text_dedup_window, min_match=8)
                text = trimmed
                if text:
                    self._last_text[seg.stream] = _norm_text(prev + " " + text).strip()
                else:
                    # if we removed everything, keep prev as-is
                    pass
            else:
                if text:
                    self._last_text[seg.stream] = _norm_text(text)

            # Step 3: adaptive beam update AFTER we know latency
            if self._adaptive_beam_enabled:
                now = time.time()
                qsz = int(self._seg_q.qsize())
                new_beam, reason = self._beam_ctl.maybe_update(
                    seg_qsize=qsz,
                    last_latency_s=float(latency_s),
                    last_dur_s=float(seg_dur_s),
                    now=now,
                )
                if reason:
                    self._log_event({"type": "asr_beam_update", "beam": int(new_beam), "reason": reason, "ts": now})

            self._log_event(
                {
                    "type": "segment",
                    "stream": seg.stream,
                    "speaker": speaker,
                    "t_start": seg.t_start,
                    "t_end": seg.t_end,
                    "text": text,
                    "latency_s": float(latency_s),
                    "seg_dur_s": float(seg_dur_s),
                    "beam_used": int(beam_to_use),
                    "dedup_removed_chars": int(removed),
                    "seg_qsize": int(self._seg_q.qsize()),
                    "ts": time.time(),
                }
            )
