from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Callable, Dict

from asr.domain.segments import Segment
from diarization.application.diarization import DiarizationPort
from diarization.domain.speaker_labels import clean_speaker_label, source_speaker_label

LogEvent = Callable[[dict], None]


@dataclass(frozen=True)
class DiarizationUpdateConfig:
    enabled: bool
    source_speaker_labels: Dict[str, str] = field(default_factory=dict)
    event_source: str = "diarization_sidecar"


class DiarizationUpdateRuntime:
    def __init__(
        self,
        *,
        config: DiarizationUpdateConfig,
        segment_queue: "queue.Queue[Segment]",
        stop_event,
        diarization: DiarizationPort,
        log_event: LogEvent,
    ) -> None:
        self.enabled = bool(config.enabled)
        self._source_speaker_labels = dict(config.source_speaker_labels or {})
        self._event_source = str(config.event_source or "diarization_sidecar")
        self._seg_q = segment_queue
        self._stop = stop_event
        self._diar = diarization
        self._log_event = log_event
        self._started = False

    def reset_runtime(self) -> None:
        self._started = False

    def run_safe(self) -> None:
        try:
            self.run()
        except Exception as exc:
            self._log_event({"type": "error", "where": "diar_sidecar", "error": str(exc), "ts": time.time()})

    def run(self) -> None:
        if not self.enabled:
            return
        self._start_backend()
        while not self._stop.is_set():
            self.run_once(timeout_s=0.2)

    def run_once(self, *, timeout_s: float = 0.0) -> bool:
        if not self.enabled:
            return False
        if not self._started:
            self._start_backend()
        try:
            segment = self._seg_q.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            return False
        self._process_segment(segment)
        return True

    def _start_backend(self) -> None:
        if self._started:
            return
        self._diar.init_backend(self._log_event)
        self._started = True
        self._log_event({"type": "diar_sidecar_started", "ts": time.time()})

    def _process_segment(self, segment: Segment) -> None:
        speaker = clean_speaker_label(self._diar.speaker_for_segment(segment, self._log_event))
        if not speaker:
            return
        if speaker == source_speaker_label(self._source_speaker_labels, segment.stream):
            return
        self._log_event(
            {
                "type": "transcript_speaker_update",
                "stream": str(segment.stream),
                "speaker": speaker,
                "t_start": float(segment.t_start),
                "t_end": float(segment.t_end),
                "source": self._event_source,
                "ts": time.time(),
            }
        )
