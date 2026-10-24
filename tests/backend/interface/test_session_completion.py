from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from application.session_tasks import OfflinePassResult
from interface.session_controller import HeadlessSessionController
from tests.helpers.electron_interface_fakes import (
    _FakeAudioRuntimeFactory,
    _FakeAudioSourceFactory,
    _FakeWavRecorderFactory,
)


class SessionCompletionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = directory.name
        self.offline = Mock()
        self.controller = HeadlessSessionController(
            project_root=Path(root),
            audio_runtime_factory=_FakeAudioRuntimeFactory(),
            audio_source_factory=_FakeAudioSourceFactory(),
            wav_recorder_factory=_FakeWavRecorderFactory(),
            offline_pass_use_case=self.offline,
        )
        self.addCleanup(self.controller.shutdown)

    def test_failed_asr_stop_retains_runtime_for_retry_and_blocks_restart(self):
        runtime = Mock()
        runtime.stop.side_effect = [OSError('native stop failed'), None]
        self.controller._asr_runtime = runtime
        self.controller._asr_running = True
        with patch.object(
            self.controller, '_write_session_outputs_locked'
        ) as write:
            failed = self.controller.stop_session()
            self.assertIs(failed['asrRunning'], True)
            self.assertIs(self.controller._asr_runtime, runtime)
            write.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'retry stopping'):
                self.controller.start_session({})
            stopped = self.controller.stop_session()
            self.assertIs(stopped['asrRunning'], False)
            self.assertIsNone(self.controller._asr_runtime)
            self.assertEqual(runtime.stop.call_count, 2)
            write.assert_called_once()

    def test_failed_shutdown_can_be_retried(self):
        with patch.object(
            self.controller.writer,
            'stop',
            side_effect=[OSError('writer'), None],
        ):
            with (
                self.assertLogs(
                    'interface.session_controller_parts.runtime_session_mixin',
                    level='ERROR',
                ),
                self.assertRaisesRegex(OSError, 'writer'),
            ):
                self.controller.shutdown()
            self.assertIs(self.controller._shutdown_complete, False)
            self.controller.shutdown()
        self.assertIs(self.controller._shutdown_complete, True)

    def test_offline_completion_publishes_idle_state_before_notification(self):
        finished = threading.Event()
        snapshots = []
        self.offline.execute.return_value = OfflinePassResult(
            out_txt=Path('transcript.txt')
        )

        def emit(kind, payload):
            if kind == 'offline_pass_done':
                snapshots.append(self.controller.snapshot())
                finished.set()

        self.controller.event_sink = emit
        self.controller._start_offline_pass(Path('recording.wav'), {})
        self.assertTrue(finished.wait(3))
        self.assertEqual(snapshots[0]['state'], 'idle')
        self.assertIs(snapshots[0]['offlinePass']['running'], False)

    def test_failed_thread_start_restores_idle_state(self):
        with patch.object(
            threading.Thread, 'start', side_effect=RuntimeError('start failed')
        ):
            with self.assertRaisesRegex(RuntimeError, 'start failed'):
                self.controller._start_offline_pass(Path('recording.wav'), {})
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot['state'], 'idle')
        self.assertIs(snapshot['offlinePass']['running'], False)

    def test_session_archive_keeps_lines_beyond_ui_history_and_copies_snapshot(
        self,
    ):
        for index in range(620):
            self.controller._append_transcript_line_locked(
                {'id': str(index), 'text': str(index), 'words': [{'text': 'x'}]}
            )
        self.assertEqual(len(self.controller._transcript_lines), 500)
        self.assertEqual(len(self.controller._session_transcript_lines), 620)
        snapshot = self.controller.snapshot()
        snapshot['transcript'][-1]['words'][0]['text'] = 'changed'
        self.assertEqual(
            self.controller._session_transcript_lines[-1]['words'][0]['text'],
            'x',
        )
