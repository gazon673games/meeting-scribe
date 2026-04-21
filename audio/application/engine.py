from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from audio.application.engine_meters import build_meter_snapshot  # noqa: F401 (kept for callers)
from audio.application.engine_runtime import EngineRuntimeState
from audio.application.mix_worker import AudioMixWorker, MixLoopSnapshot
from audio.application.mixer import enqueue_source_frame, normalize_source_frame
from audio.application.source_registry import SourceRegistry
from audio.application.tap_config import normalize_tap_config
from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioFilter, AudioSource
from audio.domain.types import TapMode


class AudioEngine:
    """Real-time mixer with per-source ring buffers + master clock.

    output_queue receives master mix blocks (blocksize, channels) float32 in [-1..1].
    tap packets: {"t_start", "t_end", "mix"?, "sources"?} depending on tap_mode.
    "enabled=False" mutes a source but keeps it ticking so ASR SPLIT stays stable.
    tap has early-drop to avoid expensive copies when queue is near full.
    """

    _MIX_ACTIVE_RMS_EPS: float = 1e-4

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
        self._registry = SourceRegistry(format)
        self._lock = threading.RLock()
        self._running = False
        self._mix_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._runtime = EngineRuntimeState(
            tap_q=tap_queue,
            tap_queue_max=int(tap_queue_max),
        )
        self._mix_worker = AudioMixWorker(
            format=self._fmt,
            output_queue=self._out_q,
            stop_event=self._stop_evt,
            tap_queue_max=self._runtime.tap_queue_max,
            active_rms_eps=self._MIX_ACTIVE_RMS_EPS,
            snapshot_provider=self._build_mix_snapshot,
            record_master_metrics=self._record_master_metrics,
            record_output_drop=self._record_output_drop,
            record_tap_drop=self._record_tap_drop,
        )

    @property
    def format(self) -> AudioFormat:
        return self._fmt

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running)

    def set_tap_queue(self, tap_queue: Optional["queue.Queue[dict]"]) -> None:
        with self._lock:
            self._runtime.set_tap_queue(tap_queue)

    def set_tap_config(
        self,
        *,
        mode: TapMode = "both",
        sources: Optional[List[str]] = None,
        drop_threshold: float = 0.85,
    ) -> None:
        """mode: "mix" | "sources" | "both"; sources: None = all; drop_threshold: skip tap when queue fill >= threshold."""
        with self._lock:
            self._runtime.apply_tap_config(normalize_tap_config(mode=mode, sources=sources, drop_threshold=drop_threshold))

    def add_source(self, src: AudioSource) -> None:
        with self._lock:
            self._registry.add_source(src, running=self._running)

    def add_master_filter(self, flt: AudioFilter) -> None:
        with self._lock:
            self._registry.add_master_filter(flt)

    def set_source_enabled(self, name: str, enabled: bool) -> None:
        with self._lock:
            self._registry.set_source_enabled(name, enabled)

    def set_source_delay_ms(self, name: str, delay_ms: float) -> None:
        with self._lock:
            self._registry.set_source_delay_ms(name, delay_ms)

    def enable_auto_sync(self, reference_source: str, target_source: str) -> None:
        with self._lock:
            self._runtime.enable_auto_sync(reference_source, target_source)

    def disable_auto_sync(self) -> None:
        with self._lock:
            self._runtime.disable_auto_sync()

    def get_meters(self) -> dict:
        with self._lock:
            return self._runtime.meter_snapshot(self._fmt, self._registry.state)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            if not self._registry.has_sources():
                raise RuntimeError("No sources added")
            self._running = True
            self._stop_evt.clear()
            self._runtime.reset_for_start()

        started: List[AudioSource] = []
        try:
            for src in self._registry.sources:
                src.start(self._on_audio_from_source)
                started.append(src)  # only append after successful start

            self._mix_thread = threading.Thread(target=self._mix_worker.run, name="audio-mixer", daemon=True)
            self._mix_thread.start()
        except Exception:
            with self._lock:
                self._running = False
                self._stop_evt.set()
                self._mix_thread = None
            for src in reversed(started):
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

        for src in self._registry.sources:
            try:
                src.stop()
            except Exception:
                pass

        if self._mix_thread is not None:
            self._mix_thread.join(timeout=2.0)
            self._mix_thread = None

        with self._lock:
            self._reset_runtime_state_locked()

    def _reset_runtime_state_locked(self) -> None:
        self._registry.reset_runtime_state()
        self._runtime.reset_after_stop()

    def _on_audio_from_source(self, source_name: str, frame: np.ndarray) -> None:
        with self._lock:
            if not self._running:
                return
            st = self._registry.get_state(source_name)
            if st is None or not st.enabled:
                return
        # Intentionally lock-free: ring buffer append is thread-safe by design.
        enqueue_source_frame(st, normalize_source_frame(frame), max_buffer_blocks=self._max_buf_blocks)

    def _build_mix_snapshot(self, period_s: float) -> MixLoopSnapshot:
        with self._lock:
            if not self._running:
                return MixLoopSnapshot(running=False)
            t_start, t_end = self._runtime.next_mix_window(period_s)
            return MixLoopSnapshot(
                running=True,
                t_start=t_start,
                t_end=t_end,
                items=self._registry.source_items(),
                master_filters=self._registry.master_filters(),
                tap_q=self._runtime.tap_q,
                tap_mode=self._runtime.tap_mode,
                tap_sources_filter=set(self._runtime.tap_sources_filter) if self._runtime.tap_sources_filter else None,
                tap_drop_threshold=self._runtime.tap_drop_threshold,
            )

    def _record_master_metrics(self, master_rms: float, ts_mono: float) -> None:
        with self._lock:
            self._runtime.record_master_metrics(master_rms, ts_mono)

    def _record_output_drop(self) -> None:
        with self._lock:
            self._runtime.record_output_drop()

    def _record_tap_drop(self) -> None:
        with self._lock:
            self._runtime.record_tap_drop()
