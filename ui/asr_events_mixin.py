from __future__ import annotations

import queue
import time

from application.event_types import (
    AsrErrorEvent,
    AsrInitOkEvent,
    AsrInitStartEvent,
    AsrMetricsEvent,
    AsrOverloadEvent,
    AsrStartedEvent,
    AsrStoppedEvent,
    EventType,
    SegmentDroppedEvent,
    SegmentSkippedOverloadEvent,
    SourceErrorEvent,
    UtteranceEvent,
    event_from_record,
)


class AsrEventsMixin:
    def _drain_asr_ui_events(self, limit: int = 140) -> None:
        n = 0
        while n < limit:
            try:
                raw = self.asr_ui_q.get_nowait()
            except queue.Empty:
                break
            n += 1

            ev = event_from_record(raw)
            typ = ev.event_type
            ts = float(ev.ts or time.time())
            tss = self._fmt_ts(ts)

            lite = bool(self._long_run_mode)

            if isinstance(ev, UtteranceEvent):
                text = ev.text.strip()
                if not text:
                    continue
                if ev.overload:
                    self._asr_overload_active = True
                self._append_transcript_line(f"[{tss}] {ev.stream}: {text}")

            elif isinstance(ev, SourceErrorEvent):
                self._append_transcript_line(f"[{tss}] SOURCE ERROR {ev.source}: {ev.error}")
                self._set_status(f"Audio source failed: {ev.source}")

            elif isinstance(ev, AsrOverloadEvent):
                if bool(ev.active):
                    self._asr_overload_active = True
                    self._append_transcript_line(
                        f"[{tss}] OVERLOAD: {ev.reason} q={ev.seg_qsize} beam={ev.beam_cur} lag={ev.lag_s}"
                    )
                else:
                    self._asr_overload_active = False
                    self._append_transcript_line(f"[{tss}] OVERLOAD OFF: {ev.reason} q={ev.seg_qsize}")

            elif isinstance(ev, SegmentDroppedEvent):
                self._warn_throttle(f"ASR dropped segment ({ev.stream}) reason={ev.reason} q={ev.seg_qsize}")

            elif isinstance(ev, SegmentSkippedOverloadEvent):
                self._warn_throttle(f"ASR skipped {ev.count} old segments due to overload (q={ev.seg_qsize})")

            elif isinstance(ev, AsrMetricsEvent):
                self._seg_dropped_total = int(ev.seg_dropped_total)
                self._seg_skipped_total = int(ev.seg_skipped_total)
                self._avg_latency_s = float(ev.avg_latency_s)
                self._p95_latency_s = float(ev.p95_latency_s)
                self._lag_s = float(ev.lag_s)

            elif typ.value in ("segment", "segment_ready", "diar_debug", "asr_beam_update"):
                if lite:
                    continue

            elif isinstance(ev, AsrInitStartEvent):
                if lite:
                    continue
                self._append_transcript_line(f"[{tss}] ASR init start ({ev.model}, {ev.device})")

            elif isinstance(ev, AsrStartedEvent):
                self._append_transcript_line(
                    f"[{tss}] ASR started "
                    f"(lang={ev.language}, mode={ev.mode}, model={ev.model}, overload={ev.overload_strategy})"
                )

            elif isinstance(ev, AsrInitOkEvent):
                if lite:
                    continue
                self._append_transcript_line(f"[{tss}] ASR init OK ({ev.model})")

            elif isinstance(ev, AsrErrorEvent):
                self._append_transcript_line(f"[{tss}] ERROR {ev.where}: {ev.error}")

            elif isinstance(ev, AsrStoppedEvent):
                self._append_transcript_line(f"[{tss}] ASR stopped")

            else:
                continue
