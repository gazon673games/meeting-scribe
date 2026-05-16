from __future__ import annotations

import unittest

from application.events.asr import (
    AsrErrorEvent,
    AsrInitOkEvent,
    AsrInitStartEvent,
    AsrOverloadEvent,
    AsrStoppedEvent,
    SegmentDroppedEvent,
    SegmentSkippedOverloadEvent,
)
from application.events.base import EventType, optional_int
from application.events.parsing import event_from_record
from application.events.session import (
    AsrStopDoneEvent,
    OfflinePassDoneEvent,
    OfflinePassErrorEvent,
    OfflinePassStartedEvent,
)


class EventConstructorTests(unittest.TestCase):
    def test_asr_events_emit_expected_record_types_and_fields(self) -> None:
        events = [
            AsrInitStartEvent(model="tiny", device="cpu", ts=1),
            AsrInitOkEvent(model="tiny", ts=2),
            AsrOverloadEvent(active=True, reason="hard", seg_qsize="3", beam_cur="2", lag_s="1.5", ts=3),
            SegmentDroppedEvent(stream="mic", reason="queue", seg_qsize="4", ts=4),
            SegmentSkippedOverloadEvent(count="5", seg_qsize="6", ts=5),
            AsrErrorEvent(where="worker", error="boom", component="", ts=6),
            AsrStoppedEvent(ts=7),
        ]

        records = [event.as_record() for event in events]

        self.assertEqual(records[0]["type"], EventType.ASR_INIT_START.value)
        self.assertEqual(records[2]["seg_qsize"], "3")
        self.assertEqual(records[4]["count"], 5)
        self.assertEqual(records[5]["component"], "asr")
        self.assertEqual(records[6]["type"], EventType.ASR_STOPPED.value)

    def test_session_events_and_identity_fields_round_trip_from_records(self) -> None:
        events = [
            AsrStopDoneEvent(
                wav_path="session.wav",
                run_offline_pass=True,
                offline_model_name="large-v3",
                offline_language=None,
                stop_error=None,
                ts=1,
            ),
            OfflinePassStartedEvent(ts=2),
            OfflinePassDoneEvent(out_txt="out.txt", ts=3),
            OfflinePassErrorEvent(error="failed", ts=4),
        ]

        self.assertEqual([event.as_record()["type"] for event in events], [
            EventType.ASR_STOP_DONE.value,
            EventType.OFFLINE_PASS_STARTED.value,
            EventType.OFFLINE_PASS_DONE.value,
            EventType.OFFLINE_PASS_ERROR.value,
        ])
        parsed = event_from_record(
            {
                "type": EventType.ASR_STOP_DONE.value,
                "wav_path": "x.wav",
                "run_offline_pass": True,
                "offline_model_name": "tiny",
                "offline_language": {"raw": "kept"},
                "stop_error": None,
            }
        )
        self.assertEqual(parsed.offline_language, {"raw": "kept"})
        self.assertIsNone(optional_int(None))
        self.assertEqual(optional_int("9"), 9)


if __name__ == "__main__":
    unittest.main()
