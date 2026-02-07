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

Mode = Literal["mix", "split"]


@dataclass
class Segment:
    stream: str
    t_start: float
    t_end: float
    audio_16k: np.ndarray


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())
    return s


def _trim_overlap(prev_text: str, cur_text: str, *, max_window: int = 80, min_match: int = 8) -> Tuple[str, int]:
    p = _norm_text(prev_text)
    c = _norm_text(cur_text)
    if not p or not c:
        return (c, 0)

    ps = p[-max_window:]
    best = 0
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
    min_beam: int = 1
    max_beam: int = 5
    cur_beam: int = 5

    backlog_hi: int = 12
    backlog_lo: int = 2
    latency_ratio_hi: float = 1.1
    latency_ratio_lo: float = 0.7

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


@dataclass
class _UtteranceState:
    stream: str
    t_start: float
    t_end: float
    text: str
    last_emit_ts: float


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


class ASRPipeline:
    """
    Consumes AudioEngine tap_queue packets and produces ASR events.

    Step 1 (this revision):
      - diarization removed/disabled entirely (no speaker labeling)
      - utterance aggregation keyed only by stream
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
        endpoint_silence_ms: float = 800.0,
        max_segment_s: float = 12.0,
        overlap_ms: float = 300.0,
        vad_energy_threshold: float = 0.006,
        vad_hangover_ms: int = 400,
        vad_min_speech_ms: int = 350,
        agc_enabled: bool = True,
        agc_target_rms: float = 0.06,
        agc_max_gain: float = 6.0,
        agc_alpha: float = 0.02,
        text_dedup_enabled: bool = True,
        text_dedup_window: int = 80,
        adaptive_beam_enabled: bool = True,
        adaptive_beam_min: int = 1,
        adaptive_beam_max: Optional[int] = None,
        # utterance aggregation
        utterance_enabled: bool = True,
        utterance_gap_s: float = 0.85,
        utterance_max_s: float = 18.0,
        utterance_flush_s: float = 2.5,
        # log rotation
        log_max_bytes: int = 25 * 1024 * 1024,
        log_backup_count: int = 5,
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

        self._agc_enabled = bool(agc_enabled)
        self._agc_target_rms = float(agc_target_rms)
        self._agc_max_gain = float(agc_max_gain)
        self._agc_alpha = float(agc_alpha)
        self._agc: Dict[str, _PreGainAGC] = {}

        self._text_dedup_enabled = bool(text_dedup_enabled)
        self._text_dedup_window = int(text_dedup_window)
        self._last_text: Dict[str, str] = {}

        self._adaptive_beam_enabled = bool(adaptive_beam_enabled)
        maxb = int(adaptive_beam_max) if adaptive_beam_max is not None else int(self.beam_size)
        maxb = max(1, maxb)
        self._beam_ctl = _AdaptiveBeam(
            min_beam=max(1, int(adaptive_beam_min)),
            max_beam=maxb,
            cur_beam=max(1, min(int(self.beam_size), maxb)),
        )

        self._utt_enabled = bool(utterance_enabled)
        self._utt_gap_s = float(utterance_gap_s)
        self._utt_max_s = float(utterance_max_s)
        self._utt_flush_s = float(utterance_flush_s)
        self._utt_state: Dict[str, _UtteranceState] = {}  # key=stream only
        self._last_utt_flush_check = 0.0

        self.session_id = f"sess_{int(time.time())}"
        self.logger = ASRLogger(
            root=self.project_root,
            session_id=self.session_id,
            language=language,
            max_bytes=int(log_max_bytes),
            backup_count=int(log_backup_count),
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
                "agc_enabled": self._agc_enabled,
                "text_dedup_enabled": self._text_dedup_enabled,
                "adaptive_beam_enabled": self._adaptive_beam_enabled,
                "utterance_enabled": self._utt_enabled,
                "utterance_gap_s": self._utt_gap_s,
                "utterance_max_s": self._utt_max_s,
                "utterance_flush_s": self._utt_flush_s,
                "log_rotation": {"max_bytes": self.logger.max_bytes, "backup_count": self.logger.backup_count},
                "diarization_enabled": False,
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

        try:
            self._flush_all_utterances(force=True)
        except Exception:
            pass

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

    # ================= ingest =================

    def _feed_stream(self, stream: str, t0: float, t1: float, block_48k: np.ndarray, sample_rate: int = 48000) -> None:
        self._ensure_stream(stream)
        vad = self._vads[stream]

        mono = stereo_to_mono(block_48k)
        x16 = resample_linear(mono, sample_rate, 16000)

        if self._agc_enabled:
            agc = self._agc.get(stream)
            if agc is not None:
                x16 = agc.process(x16)

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

    # ================= utterance aggregation =================

    def _flush_all_utterances(self, *, force: bool = False) -> None:
        now = time.time()
        for stream in list(self._utt_state.keys()):
            self._flush_utterance(stream, now=now, force=force)

    def _flush_utterance(self, stream: str, *, now: float, force: bool) -> None:
        st = self._utt_state.get(stream)
        if st is None:
            return
        if not st.text.strip():
            self._utt_state.pop(stream, None)
            return

        if force or (now - float(st.last_emit_ts)) >= float(self._utt_flush_s):
            self._log_event(
                {
                    "type": "utterance",
                    "stream": st.stream,
                    "t_start": float(st.t_start),
                    "t_end": float(st.t_end),
                    "text": _norm_text(st.text),
                    "ts": now,
                }
            )
            self._utt_state.pop(stream, None)

    def _update_utterance(self, *, stream: str, t_start: float, t_end: float, text: str) -> None:
        if not self._utt_enabled:
            return
        txt = _norm_text(text)
        if not txt:
            return

        now = time.time()
        key = str(stream)
        st = self._utt_state.get(key)

        if (now - self._last_utt_flush_check) > 0.5:
            self._last_utt_flush_check = now
            for k in list(self._utt_state.keys()):
                self._flush_utterance(k, now=now, force=False)

        if st is None:
            self._utt_state[key] = _UtteranceState(
                stream=str(stream),
                t_start=float(t_start),
                t_end=float(t_end),
                text=txt,
                last_emit_ts=now,
            )
            return

        gap = float(t_start) - float(st.t_end)
        new_len = float(t_end) - float(st.t_start)

        if gap <= float(self._utt_gap_s) and new_len <= float(self._utt_max_s):
            st.t_end = float(t_end)
            st.text = (st.text + " " + txt).strip()
            st.last_emit_ts = now
            self._utt_state[key] = st
            return

        self._flush_utterance(key, now=now, force=True)
        self._utt_state[key] = _UtteranceState(
            stream=str(stream),
            t_start=float(t_start),
            t_end=float(t_end),
            text=txt,
            last_emit_ts=now,
        )

    # ================= ASR worker =================

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

        while not self._stop.is_set():
            try:
                seg = self._seg_q.get(timeout=0.2)
            except queue.Empty:
                if self._utt_enabled:
                    try:
                        self._flush_all_utterances(force=False)
                    except Exception:
                        pass
                continue

            seg_dur_s = max(1e-6, float(seg.audio_16k.shape[0]) / 16000.0)
            beam_to_use = int(self._beam_ctl.cur_beam)

            t0 = time.time()
            try:
                res = self._asr.transcribe(seg.audio_16k, beam_size=beam_to_use)
                text = (res.get("text") or "").strip()
            except Exception as e:
                self._log_event({"type": "segment", "stream": seg.stream, "t_start": seg.t_start, "t_end": seg.t_end, "text": "", "error": str(e), "ts": time.time()})
                continue

            latency_s = time.time() - t0

            removed = 0
            if self._text_dedup_enabled:
                prev = self._last_text.get(seg.stream, "")
                trimmed, removed = _trim_overlap(prev, text, max_window=self._text_dedup_window, min_match=8)
                text = trimmed
                if text:
                    self._last_text[seg.stream] = _norm_text(prev + " " + text).strip()
            else:
                if text:
                    self._last_text[seg.stream] = _norm_text(text)

            if self._adaptive_beam_enabled:
                now = time.time()
                qsz = int(self._seg_q.qsize())
                new_beam, reason = self._beam_ctl.maybe_update(seg_qsize=qsz, last_latency_s=float(latency_s), last_dur_s=float(seg_dur_s), now=now)
                if reason:
                    self._log_event({"type": "asr_beam_update", "beam": int(new_beam), "reason": reason, "ts": now})

            self._log_event(
                {
                    "type": "segment",
                    "stream": seg.stream,
                    "t_start": seg.t_start,
                    "t_end": seg.t_end,
                    "text": _norm_text(text),
                    "latency_s": float(latency_s),
                    "seg_dur_s": float(seg_dur_s),
                    "beam_used": int(beam_to_use),
                    "dedup_removed_chars": int(removed),
                    "seg_qsize": int(self._seg_q.qsize()),
                    "ts": time.time(),
                }
            )

            if self._utt_enabled and text.strip():
                self._update_utterance(stream=str(seg.stream), t_start=float(seg.t_start), t_end=float(seg.t_end), text=str(text))
