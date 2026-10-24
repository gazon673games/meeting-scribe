from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from transcription.infrastructure.file_transcript_store import (
    FileTranscriptStore,
    _srt_timestamp,
)


class FileTranscriptStoreRealtimeTests(unittest.TestCase):
    def test_realtime_srt_entries_are_deferred_and_flushed_on_close(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            store = FileTranscriptStore(root)

            store.write_realtime_srt_entry(10.0, 'mic', 'ignored')
            self.assertIsNone(store.realtime_transcript_path)

            store.set_realtime_enabled(True)
            with (
                patch(
                    'transcription.infrastructure.file_transcript_store.time.strftime',
                    return_value='20260101_120000',
                ),
                patch(
                    'transcription.infrastructure.file_transcript_store.time.monotonic',
                    return_value=100.0,
                ),
            ):
                store.write_realtime_srt_entry(
                    10.0, 'mic', 'hello', speaker='Alice'
                )
                store.write_realtime_srt_entry(12.0, 'speaker', 'world')
                path = store.realtime_transcript_path
                store.close_realtime_transcript()

            self.assertIsNotNone(path)
            assert path is not None
            text = path.read_text(encoding='utf-8')
            self.assertIn('Alice: hello', text)
            self.assertIn('speaker: world', text)
            self.assertIn('00:00:00,000 --> 00:00:01,950', text)
            self.assertIn('00:00:02,000 --> 00:00:06,000', text)
            self.assertIsNone(store.realtime_transcript_path)

    def test_human_log_flush_waits_for_interval_before_flushing_dirty_lines(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            store = FileTranscriptStore(root)
            path = store.open_human_log()
            assert path is not None
            store.write_human_line('pending')
            self.assertTrue(store._human_dirty)

            with patch(
                'transcription.infrastructure.file_transcript_store.time.monotonic',
                return_value=store._human_last_flush,
            ):
                store._flush_human_if_due()
            self.assertTrue(store._human_dirty)

            with patch(
                'transcription.infrastructure.file_transcript_store.time.monotonic',
                return_value=store._human_last_flush + 2.0,
            ):
                store._flush_human_if_due()
            self.assertFalse(store._human_dirty)
            store.close_human_log()

    def test_open_failures_and_timestamp_format_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            store = FileTranscriptStore(root)

            with patch(
                'transcription.infrastructure.file_transcript_store.project_human_logs_dir',
                side_effect=OSError('nope'),
            ):
                self.assertIsNone(store.open_human_log())

            store.set_realtime_enabled(True)
            with patch(
                'transcription.infrastructure.file_transcript_store.project_logs_dir',
                side_effect=OSError('nope'),
            ):
                store.write_realtime_srt_entry(1.0, 'mic', 'hidden')
            self.assertIsNone(store.realtime_transcript_path)

        self.assertEqual(_srt_timestamp(-1.0), '00:00:00,000')
        self.assertEqual(_srt_timestamp(3661.234), '01:01:01,234')

    def test_close_releases_both_files_after_a_flush_failure(self) -> None:
        cases = (
            ('_human_log_fh', 'close_human_log'),
            ('_realtime_transcript_fh', 'close_realtime_transcript'),
        )
        for field, method in cases:
            with self.subTest(field=field):
                store = FileTranscriptStore(Path('.'))
                handle = Mock()
                handle.flush.side_effect = OSError('disk full')
                setattr(store, field, handle)
                with self.assertLogs(level='WARNING'):
                    getattr(store, method)()
                handle.close.assert_called_once_with()
                self.assertIsNone(getattr(store, field))


if __name__ == '__main__':
    unittest.main()
