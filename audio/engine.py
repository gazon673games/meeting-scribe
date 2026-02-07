# --- File: D:\work\own\voice2textTest\audio\engine.py ---
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int
    channels: int
    dtype: str = "float32"
    blocksize: int = 1024


class AudioFilter(Protocol):
    def process(self, x: np.ndarray, fmt: AudioFormat) -> np.ndarray:
        ...


class AudioSource(Protocol):
    name: str

    def start(self, on_audio: Callable[[str, np.ndarray], None]) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_format(self) -> AudioFormat:
        ...

    def get_filters(self) -> List[AudioFilter]:
        ...


@dataclass
class _SourceState:
    src: AudioSource
    enabled: bool = True

    buf: Deque[np.ndarray] = None  # type: ignore[assignment]
    buf_lock: threading.Lock = None  # type: ignore[assignment]

    delay_frames: int = 0
    delay_buf: Deque[np.ndarray] = None  # type: ignore[assignment]

    rms: float = 0.0
    last_ts: float = 0.0

    dropped_in_frames: int = 0
    missing_out_frames: int = 0
    buffer_frames: int = 0
    src_rate: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "buf", deque())
        object.__setattr__(self, "buf_lock", threading.Lock())
        object.__setattr__(self, "delay_buf", deque())


class AudioEngine:
    """
    Real-time safe mixer with per-source ring buffers and master clock.

    Adds:
      - tap_queue: optional queue of per-tick packets for downstream consumers (WAV/ASR/etc.)
        Packet:
          {
            "t_start": float,
            "t_end": float,
            "mix": np.ndarray (blocksize, channels),
            "sources": {name: np.ndarray (blocksize, channels)}
          }
    """

    def __init__(
        self,
        format: AudioFormat,
        output_queue: "queue.Queue[np.ndarray]",
        *,
        max_source_buffer_blocks: int = 50,
        tap_queue: Optional["queue.Queue[dict]"] = None,
        tap_queue_max: int = 200,
    ):
        self._fmt = format
        self._out_q = output_queue
        self._max_buf_blocks = int(max_source_buffer_blocks)

        self._sources: List[AudioSource] = []
        self._state: Dict[str, _SourceState] = {}
        self._master_filters: List[AudioFilter] = []

        self._lock = threading.RLock()
        self._running = False

        self._mix_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._master_rms: float = 0.0
        self._master_last_ts: float = 0.0

        self._dropped_out_blocks: int = 0

        self._t0_mono: float = 0.0
        self._tick_index: int = 0

        self._tap_q: Optional["queue.Queue[dict]"] = tap_queue
        self._tap_q_max = int(tap_queue_max)
        self._dropped_tap_blocks: int = 0

        self._autosync_enabled: bool = False
        self._autosync_ref: Optional[str] = None
        self._autosync_target: Optional[str] = None
        self._autosync_last_offset_ms: float = 0.0

    @property
    def format(self) -> AudioFormat:
        return self._fmt

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def set_tap_queue(self, tap_queue: Optional["queue.Queue[dict]"]) -> None:
        with self._lock:
            self._tap_q = tap_queue
            self._dropped_tap_blocks = 0

    def add_source(self, src: AudioSource) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("Cannot add sources while engine is running (MVP).")
            if src.name in self._state:
                raise ValueError(f"Source '{src.name}' already exists")
            self._sources.append(src)
            st = _SourceState(src=src)
            st.src_rate = int(src.get_format().sample_rate)
            self._state[src.name] = st

    def add_master_filter(self, flt: AudioFilter) -> None:
        with self._lock:
            self._master_filters.append(flt)

    def set_source_enabled(self, name: str, enabled: bool) -> None:
        with self._lock:
            st = self._state.get(name)
            if st is None:
                return
            st.enabled = bool(enabled)

    def set_source_delay_ms(self, name: str, delay_ms: float) -> None:
        if delay_ms < 0:
            delay_ms = 0.0
        frames = int(round((delay_ms / 1000.0) * self._fmt.sample_rate))
        with self._lock:
            st = self._state.get(name)
            if st is None:
                return
            st.delay_frames = frames
            with st.buf_lock:
                st.delay_buf.clear()

    def enable_auto_sync(self, reference_source: str, target_source: str) -> None:
        with self._lock:
            self._autosync_enabled = True
            self._autosync_ref = reference_source
            self._autosync_target = target_source
            self._autosync_last_offset_ms = 0.0

    def disable_auto_sync(self) -> None:
        with self._lock:
            self._autosync_enabled = False
            self._autosync_ref = None
            self._autosync_target = None
            self._autosync_last_offset_ms = 0.0

    def get_meters(self) -> dict:
        with self._lock:
            return {
                "sources": {
                    n: {
                        "enabled": bool(st.enabled),
                        "rms": float(st.rms),
                        "last_ts": float(st.last_ts),
                        "dropped_in_frames": int(st.dropped_in_frames),
                        "missing_out_frames": int(st.missing_out_frames),
                        "buffer_frames": int(st.buffer_frames),
                        "delay_ms": float(st.delay_frames) * 1000.0 / float(self._fmt.sample_rate),
                        "src_rate": int(st.src_rate),
                    }
                    for n, st in self._state.items()
                },
                "master": {
                    "rms": float(self._master_rms),
                    "last_ts": float(self._master_last_ts),
                },
                "drops": {
                    "dropped_out_blocks": int(self._dropped_out_blocks),
                    "dropped_tap_blocks": int(self._dropped_tap_blocks),
                },
                "autosync": {
                    "enabled": bool(self._autosync_enabled),
                    "ref": self._autosync_ref,
                    "target": self._autosync_target,
                    "last_offset_ms": float(self._autosync_last_offset_ms),
                },
            }

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            if not self._sources:
                raise RuntimeError("No sources added")
            self._running = True
            self._stop_evt.clear()
            self._t0_mono = time.monotonic()
            self._tick_index = 0
            self._dropped_out_blocks = 0
            self._dropped_tap_blocks = 0

        for src in self._sources:
            src.start(self._on_audio_from_source)

        self._mix_thread = threading.Thread(
            target=self._mix_loop,
            name="audio-mixer",
            daemon=True,
        )
        self._mix_thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_evt.set()

        for src in self._sources:
            try:
                src.stop()
            except Exception:
                pass

        if self._mix_thread is not None:
            self._mix_thread.join(timeout=2.0)
            self._mix_thread = None

        with self._lock:
            for st in self._state.values():
                with st.buf_lock:
                    st.buf.clear()
                    st.delay_buf.clear()
                st.rms = 0.0
                st.last_ts = 0.0
                st.dropped_in_frames = 0
                st.missing_out_frames = 0
                st.buffer_frames = 0
            self._master_rms = 0.0
            self._master_last_ts = 0.0
            self._dropped_out_blocks = 0
            self._dropped_tap_blocks = 0
            self._autosync_last_offset_ms = 0.0

    # ---------------- internals ----------------

    @staticmethod
    def _apply_filters(x: np.ndarray, fmt: AudioFormat, filters: List[AudioFilter]) -> np.ndarray:
        y = x
        for f in filters:
            y = f.process(y, fmt)
        return y

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        xf = x.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(xf * xf)))

    @staticmethod
    def _pad_or_crop_n(x: np.ndarray, n: int) -> np.ndarray:
        """Ensure x has exactly n frames (rows)."""
        if x.shape[0] == n:
            return x.astype(np.float32, copy=False)
        if x.shape[0] > n:
            return x[:n].astype(np.float32, copy=False)
        pad = np.zeros((n - x.shape[0], x.shape[1]), dtype=np.float32)
        return np.vstack([x.astype(np.float32, copy=False), pad])

    def _on_audio_from_source(self, source_name: str, frame: np.ndarray) -> None:
        with self._lock:
            if not self._running:
                return
            st = self._state.get(source_name)
            if st is None:
                return

        x = np.asarray(frame)
        if x.ndim == 1:
            x = x[:, None]
        x = np.array(x, dtype=np.float32, copy=True)

        with st.buf_lock:
            st.buf.append(x)
            while len(st.buf) > self._max_buf_blocks:
                dropped = st.buf.popleft()
                st.dropped_in_frames += int(len(dropped))

            st.buffer_frames = int(sum(len(b) for b in st.buf))

    def _mix_loop(self) -> None:
        fmt = self._fmt
        period_s = fmt.blocksize / float(fmt.sample_rate)
        next_t = time.monotonic()

        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue
            next_t += period_s

            with self._lock:
                if not self._running:
                    break
                items: List[Tuple[str, _SourceState]] = list(self._state.items())
                master_filters = list(self._master_filters)
                tap_q = self._tap_q

            t_start = float(self._tick_index) * period_s
            t_end = t_start + period_s
            self._tick_index += 1

            mix = np.zeros((fmt.blocksize, fmt.channels), dtype=np.float32)
            sources_out: Dict[str, np.ndarray] = {}
            active_sources = 0

            ts_mono = time.monotonic()

            for name, st in items:
                if not st.enabled:
                    continue

                raw = None
                with st.buf_lock:
                    if st.buf:
                        raw = st.buf.popleft()
                        st.buffer_frames = int(sum(len(b) for b in st.buf))

                src_fmt = st.src.get_format()
                st.src_rate = int(src_fmt.sample_rate)

                if raw is None:
                    block_src = np.zeros((fmt.blocksize, max(1, src_fmt.channels)), dtype=np.float32)
                    st.missing_out_frames += fmt.blocksize
                else:
                    block_src = raw
                    if block_src.ndim == 1:
                        block_src = block_src[:, None]

                # Normalize input block in SOURCE rate to engine blocksize for predictable resample duration
                if block_src.shape[0] < fmt.blocksize:
                    pad = np.zeros((fmt.blocksize - block_src.shape[0], block_src.shape[1]), dtype=np.float32)
                    block_src = np.vstack([block_src, pad])
                    if raw is not None:
                        st.missing_out_frames += (fmt.blocksize - raw.shape[0])
                elif block_src.shape[0] > fmt.blocksize:
                    block_src = block_src[: fmt.blocksize]

                # Per-source filters (source format)
                try:
                    block_src = self._apply_filters(block_src, src_fmt, st.src.get_filters())
                except Exception:
                    block_src = np.zeros((fmt.blocksize, max(1, src_fmt.channels)), dtype=np.float32)
                    st.missing_out_frames += fmt.blocksize

                # Resample to engine rate
                block_src = self._resample_to_engine_rate(block_src, int(src_fmt.sample_rate), int(fmt.sample_rate))

                # FIX: after resample, enforce exact engine blocksize (dst_n may differ from fmt.blocksize)
                if block_src.ndim == 1:
                    block_src = block_src[:, None]
                block_src = self._pad_or_crop_n(block_src, fmt.blocksize)

                # Channel map to engine format
                block_eng = self._channel_map_to_engine(block_src, int(src_fmt.channels), fmt.channels)

                # Delay compensation
                if st.delay_frames > 0:
                    block_eng = self._apply_delay_block(st, block_eng, fmt.blocksize, fmt.channels)

                st.rms = self._rms(block_eng)
                st.last_ts = ts_mono

                sources_out[name] = block_eng
                mix += block_eng
                active_sources += 1

            if active_sources > 1:
                mix *= 1.0 / float(active_sources)

            try:
                mixed = self._apply_filters(mix, fmt, master_filters)
            except Exception:
                mixed = mix

            mixed = np.clip(mixed, -1.0, 1.0)

            with self._lock:
                self._master_rms = self._rms(mixed)
                self._master_last_ts = ts_mono

            try:
                self._out_q.put_nowait(mixed)
            except queue.Full:
                with self._lock:
                    self._dropped_out_blocks += 1

            if tap_q is not None:
                pkt = {
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "mix": np.array(mixed, copy=True),
                    "sources": {k: np.array(v, copy=True) for k, v in sources_out.items()},
                }
                try:
                    tap_q.put_nowait(pkt)
                except queue.Full:
                    with self._lock:
                        self._dropped_tap_blocks += 1

    @staticmethod
    def _channel_map_to_engine(x: np.ndarray, src_ch: int, eng_ch: int) -> np.ndarray:
        if src_ch == eng_ch:
            return x.astype(np.float32, copy=False)

        if src_ch == 1 and eng_ch == 2:
            return np.repeat(x, 2, axis=1).astype(np.float32, copy=False)

        if src_ch == 2 and eng_ch == 1:
            return x.mean(axis=1, keepdims=True).astype(np.float32, copy=False)

        n = x.shape[0]
        if x.shape[1] > eng_ch:
            return x[:, :eng_ch].astype(np.float32, copy=False)

        out = np.zeros((n, eng_ch), dtype=np.float32)
        out[:, : x.shape[1]] = x.astype(np.float32, copy=False)
        return out

    @staticmethod
    def _resample_to_engine_rate(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return x.astype(np.float32, copy=False)

        n = x.shape[0]
        if n == 0:
            return x.astype(np.float32, copy=False)

        duration = n / float(src_rate)
        dst_n = int(round(duration * dst_rate))
        if dst_n <= 0:
            return np.zeros((0, x.shape[1]), dtype=np.float32)

        src_t = np.linspace(0.0, duration, num=n, endpoint=False, dtype=np.float64)
        dst_t = np.linspace(0.0, duration, num=dst_n, endpoint=False, dtype=np.float64)

        out = np.zeros((dst_n, x.shape[1]), dtype=np.float32)
        for ch in range(x.shape[1]):
            out[:, ch] = np.interp(dst_t, src_t, x[:, ch].astype(np.float64, copy=False)).astype(np.float32, copy=False)
        return out

    @staticmethod
    def _apply_delay_block(st: _SourceState, block: np.ndarray, blocksize: int, channels: int) -> np.ndarray:
        if st.delay_frames <= 0:
            return block

        blocks_delay = int(np.ceil(st.delay_frames / float(blocksize)))
        while len(st.delay_buf) < blocks_delay:
            st.delay_buf.append(np.zeros((blocksize, channels), dtype=np.float32))

        st.delay_buf.append(block)
        out = st.delay_buf.popleft()
        return out
