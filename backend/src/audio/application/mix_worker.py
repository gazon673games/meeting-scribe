from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from threading import Event
from typing import Callable, Optional, Sequence, Set, Tuple

import numpy as np

from audio.application.dsp import apply_filters, rms
from audio.application.mixer import render_source_block
from audio.application.source_state import SourceState
from audio.application.tap import try_emit_tap_packet
from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioFilter
from audio.domain.types import TapMode


@dataclass(frozen=True)
class MixLoopSnapshot:
    running: bool
    t_start: float = 0.0
    t_end: float = 0.0
    items: Sequence[Tuple[str, SourceState]] = ()
    master_filters: Sequence[AudioFilter] = ()
    tap_q: Optional["queue.Queue[dict]"] = None
    tap_mode: TapMode = "both"
    tap_sources_filter: Optional[Set[str]] = None
    tap_drop_threshold: float = 0.85
    output_enabled: bool = False


class AudioMixWorker:
    def __init__(
        self,
        *,
        format: AudioFormat,
        output_queue: "queue.Queue[np.ndarray]",
        stop_event: Event,
        tap_queue_max: int,
        active_rms_eps: float,
        snapshot_provider: Callable[[float], MixLoopSnapshot],
        record_master_metrics: Callable[[float, float], None],
        record_output_drop: Callable[[], None],
        record_tap_drop: Callable[[], None],
    ) -> None:
        self._fmt = format
        self._out_q = output_queue
        self._stop_evt = stop_event
        self._tap_q_max = int(tap_queue_max)
        self._active_rms_eps = float(active_rms_eps)
        self._snapshot_provider = snapshot_provider
        self._record_master_metrics = record_master_metrics
        self._record_output_drop = record_output_drop
        self._record_tap_drop = record_tap_drop

    def run(self) -> None:
        period_s = self._fmt.blocksize / float(self._fmt.sample_rate)
        next_t = time.monotonic()

        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue
            next_t += period_s

            snapshot = self._snapshot_provider(period_s)
            if not snapshot.running:
                break

            mixed, sources_out, ts_mono = self._render_mix(snapshot)
            self._record_master_metrics(rms(mixed), ts_mono)

            if snapshot.output_enabled:
                try:
                    self._out_q.put_nowait(mixed)
                except queue.Full:
                    self._record_output_drop()

            if snapshot.tap_q is not None:
                sent = try_emit_tap_packet(
                    tap_q=snapshot.tap_q,
                    tap_queue_max=self._tap_q_max,
                    drop_threshold=snapshot.tap_drop_threshold,
                    t_start=snapshot.t_start,
                    t_end=snapshot.t_end,
                    mixed=mixed,
                    sources_out=sources_out,
                    mode=snapshot.tap_mode,
                )
                if not sent:
                    self._record_tap_drop()

    def _render_mix(
        self,
        snapshot: MixLoopSnapshot,
    ) -> Tuple[np.ndarray, Optional[dict[str, np.ndarray]], float]:
        mix = np.zeros((self._fmt.blocksize, self._fmt.channels), dtype=np.float32)
        want_sources = (snapshot.tap_q is not None) and (snapshot.tap_mode in ("sources", "both"))
        sources_out: Optional[dict[str, np.ndarray]] = {} if want_sources else None

        active_sources = 0
        ts_mono = time.monotonic()

        for name, state in snapshot.items:
            rendered = render_source_block(
                state=state,
                engine_format=self._fmt,
                ts_mono=ts_mono,
                active_rms_eps=self._active_rms_eps,
            )
            block_eng = rendered.block

            if rendered.active:
                active_sources += 1

            if sources_out is not None:
                if snapshot.tap_sources_filter is None or name in snapshot.tap_sources_filter:
                    sources_out[name] = block_eng

            mix += block_eng

        if active_sources > 1:
            mix *= 1.0 / float(active_sources)

        try:
            mixed = apply_filters(mix, self._fmt, snapshot.master_filters)
        except Exception:
            mixed = mix

        return np.clip(mixed, -1.0, 1.0), sources_out, ts_mono
