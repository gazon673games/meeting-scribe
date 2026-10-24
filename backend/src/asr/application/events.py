from __future__ import annotations

import logging
import queue
from collections.abc import Iterable

from application.event_types import event_from_record, event_to_record
from asr.application.event_queue import is_retained_event
from asr.application.ports import AsrLoggerPort

PUBLIC_EVENT_TYPES = {
    'utterance',
    'transcript_speaker_update',
    'asr_overload',
    'segment_dropped',
    'segment_skipped_overload',
    'segment',
    'segment_ready',
    'asr_metrics',
    'asr_init_start',
    'asr_started',
    'asr_init_ok',
    'asr_segment_processing',
    'asr_segment_done',
    'error',
    'asr_stopped',
    'diar_init_ok',
    'diar_sidecar_started',
    'diar_debug',
    'diar_segment_processing',
    'diar_segment_done',
    'diar_segment_dropped',
    'streaming_words',
    'streaming_final',
}


_logger = logging.getLogger(__name__)


class ASREventPublisher:
    def __init__(
        self,
        *,
        logger: AsrLoggerPort,
        event_queue: queue.Queue[object] | None,
        public_types: Iterable[str] | None = None,
    ):
        self._logger = logger
        self._event_q = event_queue
        self._public_types = set(
            PUBLIC_EVENT_TYPES if public_types is None else public_types
        )

    def log(self, rec: dict) -> None:
        record = event_to_record(rec)
        try:
            self._logger.write(record)
        except Exception:
            _logger.warning(
                'Cannot persist ASR event %s', record.get('type'), exc_info=True
            )
        self.publish(record)

    def publish(self, rec: dict) -> None:
        if not self._event_q:
            return
        typ = str(rec.get('type', ''))
        if typ not in self._public_types:
            return
        try:
            self._event_q.put_nowait(event_from_record(rec))
        except queue.Full:
            if is_retained_event(rec):
                raise
