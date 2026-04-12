# --- File: D:\work\own\voice2textTest\audio\engine.py ---
from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from audio.dsp import (
    apply_delay_block,
    apply_filters,
    channel_map_to_engine,
    pad_or_crop_n,
    resample_to_engine_rate,
    rms,
)
from audio.source_state import SourceState as _SourceState
from audio.tap import build_tap_packet, tap_should_send
from audio.types import AudioFilter, AudioFormat, AudioSource, TapMode


class AudioEngine:
    """
    Real-time mixer with per-source ring buffers + master clock.

    output_queue: master mix blocks (blocksize, channels) float32 in [-1..1]
    tap_queue packet (if enabled):
      {
        "t_start": float,
        "t_end": float,
        "mix": np.ndarray (blocksize, channels)      # if tap_mode includes mix
        "sources": {name: np.ndarray (...)}          # if tap_mode includes sources
      }

    Important behavior:
      - "enabled=False" means "mute": source still ticks and exports silence so ASR SPLIT stays stable.
      - tap has early-drop to avoid expensive copies when queue is near full.
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

        # tap config
        self._tap_mode: TapMode = "both"
        self._tap_sources_filter: Optional[Set[str]] = None
        self._tap_drop_threshold: float = 0.85  # if tap_q fill ratio >= this, skip packet build/copies

        # autosync (not implemented here, just state)
        self._autosync_enabled: bool = False
        self._autosync_ref: Optional[str] = None
        self._autosync_target: Optional[str] = None
        self._autosync_last_offset_ms: float = 0.0

        # mix normalization behavior
        self._mix_active_rms_eps: float = 1e-4

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

    def set_tap_config(
        self,
        *,
        mode: TapMode = "both",
        sources: Optional[List[str]] = None,
        drop_threshold: float = 0.85,
    ) -> None:
        """
        mode:
          - "mix": put only mix into tap packets
          - "sources": put only sources into tap packets
          - "both": put mix + sources into tap packets

        sources:
          - None: no filtering (all sources)
          - list[str]: only include these source names in tap packets (when mode includes "sources")

        drop_threshold:
          - if tap queue fill ratio >= threshold, we skip building/copying tap packets
        """
        m = str(mode).strip().lower()
        if m not in ("mix", "sources", "both"):
            m = "both"

        thr = float(drop_threshold)
        if thr < 0.1:
            thr = 0.1
        if thr > 0.99:
            thr = 0.99

        with self._lock:
            self._tap_mode = m  # type: ignore[assignment]
            self._tap_sources_filter = set(sources) if sources else None
            self._tap_drop_threshold = thr

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
            new_val = bool(enabled)
            if st.enabled == new_val:
                return

            st.enabled = new_val

            # Сбрасываем буферы при любом переключении, чтобы не тащить старое аудио
            with st.buf_lock:
                st.buf.clear()
                st.delay_buf.clear()
                st.buffer_frames = 0

            if not new_val:
                # Мгновенно обнуляем метры, чтобы UI не показывал старое
                st.rms = 0.0
                st.last_ts = time.monotonic()

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
                "tap": {
                    "enabled": bool(self._tap_q is not None),
                    "mode": str(self._tap_mode),
                    "filter": sorted(list(self._tap_sources_filter)) if self._tap_sources_filter else None,
                    "drop_threshold": float(self._tap_drop_threshold),
                    "tap_queue_max": int(self._tap_q_max),
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

        started_sources: List[AudioSource] = []
        try:
            for src in self._sources:
                try:
                    src.start(self._on_audio_from_source)
                except Exception:
                    started_sources.append(src)
                    raise
                started_sources.append(src)

            self._mix_thread = threading.Thread(
                target=self._mix_loop,
                name="audio-mixer",
                daemon=True,
            )
            self._mix_thread.start()
        except Exception:
            with self._lock:
                self._running = False
                self._stop_evt.set()
                self._mix_thread = None

            for src in reversed(started_sources):
                try:
                    src.stop()
                except Exception:
                    pass

            with self._lock:
                self._reset_runtime_state_locked()
            raise

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
            self._reset_runtime_state_locked()

    # ---------------- internals ----------------

    def _reset_runtime_state_locked(self) -> None:
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

    def _on_audio_from_source(self, source_name: str, frame: np.ndarray) -> None:
        with self._lock:
            if not self._running:
                return
            st = self._state.get(source_name)
            if st is None:
                return
            # ВАЖНО: если источник muted, не копим аудио в буфере (иначе после unmute будет огромный лаг)
            if not st.enabled:
                return

        x = np.asarray(frame)
        if x.ndim == 1:
            x = x[:, None]
        x = np.array(x, dtype=np.float32, copy=True)

        with st.buf_lock:
            st.buf.append(x)
            st.buffer_frames += int(len(x))
            while len(st.buf) > self._max_buf_blocks:
                dropped = st.buf.popleft()
                st.dropped_in_frames += int(len(dropped))
                st.buffer_frames = max(0, int(st.buffer_frames) - int(len(dropped)))

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
                tap_mode = self._tap_mode
                tap_filter = set(self._tap_sources_filter) if self._tap_sources_filter else None

            t_start = float(self._tick_index) * period_s
            t_end = t_start + period_s
            self._tick_index += 1

            mix = np.zeros((fmt.blocksize, fmt.channels), dtype=np.float32)

            want_sources = (tap_q is not None) and (tap_mode in ("sources", "both"))
            sources_out: Optional[Dict[str, np.ndarray]] = {} if want_sources else None

            active_sources = 0
            ts_mono = time.monotonic()

            for name, st in items:
                # Always read source format so meters stay current even when muted
                src_fmt = st.src.get_format()
                st.src_rate = int(src_fmt.sample_rate)

                # MUTE behavior: keep stream alive (tap stability), but output silence and don't add to mix.
                if not st.enabled:
                    block_eng = np.zeros((fmt.blocksize, fmt.channels), dtype=np.float32)
                    st.rms = 0.0
                    st.last_ts = ts_mono

                    if sources_out is not None:
                        if tap_filter is None or name in tap_filter:
                            sources_out[name] = block_eng
                    continue

                raw = None
                with st.buf_lock:
                    if st.buf:
                        raw = st.buf.popleft()
                        st.buffer_frames = max(0, int(st.buffer_frames) - int(len(raw)))

                if raw is None:
                    block_src = np.zeros((fmt.blocksize, max(1, src_fmt.channels)), dtype=np.float32)
                    st.missing_out_frames += fmt.blocksize
                else:
                    block_src = raw
                    if block_src.ndim == 1:
                        block_src = block_src[:, None]

                # Normalize input block in SOURCE rate
                if block_src.shape[0] < fmt.blocksize:
                    pad = np.zeros((fmt.blocksize - block_src.shape[0], block_src.shape[1]), dtype=np.float32)
                    block_src = np.vstack([block_src, pad])
                    if raw is not None:
                        st.missing_out_frames += (fmt.blocksize - raw.shape[0])
                elif block_src.shape[0] > fmt.blocksize:
                    block_src = block_src[: fmt.blocksize]

                # Per-source filters (source format)
                try:
                    block_src = apply_filters(block_src, src_fmt, st.src.get_filters())
                except Exception:
                    block_src = np.zeros((fmt.blocksize, max(1, src_fmt.channels)), dtype=np.float32)
                    st.missing_out_frames += fmt.blocksize

                # Resample to engine rate
                block_src = resample_to_engine_rate(block_src, int(src_fmt.sample_rate), int(fmt.sample_rate))

                # Enforce exact engine blocksize after resample
                if block_src.ndim == 1:
                    block_src = block_src[:, None]
                block_src = pad_or_crop_n(block_src, fmt.blocksize)

                # Channel map to engine format
                block_eng = channel_map_to_engine(block_src, int(src_fmt.channels), fmt.channels)

                # Delay compensation
                if st.delay_frames > 0:
                    block_eng = apply_delay_block(st, block_eng, fmt.blocksize, fmt.channels)

                st.rms = rms(block_eng)
                st.last_ts = ts_mono

                if float(st.rms) > float(self._mix_active_rms_eps):
                    active_sources += 1

                if sources_out is not None:
                    if tap_filter is None or name in tap_filter:
                        sources_out[name] = block_eng

                mix += block_eng

            # Normalize only by actually "active" sources
            if active_sources > 1:
                mix *= 1.0 / float(active_sources)

            try:
                mixed = apply_filters(mix, fmt, master_filters)
            except Exception:
                mixed = mix

            mixed = np.clip(mixed, -1.0, 1.0)

            with self._lock:
                self._master_rms = rms(mixed)
                self._master_last_ts = ts_mono

            try:
                self._out_q.put_nowait(mixed)
            except queue.Full:
                with self._lock:
                    self._dropped_out_blocks += 1

            # Tap packet: early drop + minimal copies
            if tap_q is not None:
                if not tap_should_send(
                    tap_q,
                    tap_queue_max=self._tap_q_max,
                    drop_threshold=self._tap_drop_threshold,
                ):
                    with self._lock:
                        self._dropped_tap_blocks += 1
                else:
                    pkt = build_tap_packet(
                        t_start=t_start,
                        t_end=t_end,
                        mixed=mixed,
                        sources_out=sources_out,
                        mode=tap_mode,
                    )
                    try:
                        tap_q.put_nowait(pkt)  # type: ignore[arg-type]
                    except queue.Full:
                        with self._lock:
                            self._dropped_tap_blocks += 1
