from __future__ import annotations

import unittest

from transcription.domain.transcript_lines import (
    best_line_for_speaker_update,
    build_transcript_line_id,
    update_line_speaker,
)


class TranscriptLineTests(unittest.TestCase):
    def test_builds_stable_line_id_from_stream_and_timing(self) -> None:
        line_id = build_transcript_line_id(stream="desktop audio", t_start=1.234, t_end=2.345, ts=9.0)

        self.assertEqual(line_id, "desktop_audio:1234:2345")

    def test_finds_line_by_time_overlap_and_updates_speaker(self) -> None:
        line = {"stream": "desktop", "t_start": 1.0, "t_end": 3.0, "speaker": "Remote"}
        found = best_line_for_speaker_update(
            [line],
            stream="desktop",
            t_start=2.0,
            t_end=4.0,
        )

        self.assertIs(found, line)
        self.assertTrue(update_line_speaker(line, speaker="Remote S1", speaker_source="test", confidence=0.8))
        self.assertEqual(line["speaker"], "Remote S1")
        self.assertEqual(line["speakerSource"], "test")
        self.assertEqual(line["speakerConfidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
