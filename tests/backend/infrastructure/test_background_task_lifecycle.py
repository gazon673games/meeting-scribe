from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from infrastructure.background_tasks import ThreadBackgroundTaskRunner


class BackgroundTaskLifecycleTests(unittest.TestCase):
    def test_shutdown_retains_live_worker_on_timeout_and_allows_cleanup_retry(
        self,
    ):
        runner = ThreadBackgroundTaskRunner()
        entered = threading.Event()
        release = threading.Event()

        def work():
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release task')

        worker = runner.start(name='blocked-download', target=work)
        try:
            self.assertTrue(entered.wait(3))
            with self.assertRaisesRegex(TimeoutError, 'blocked-download'):
                runner.shutdown(timeout_s=0)
            self.assertIn(worker, runner._workers)
            with self.assertRaisesRegex(RuntimeError, 'shutting down'):
                runner.start(name='new', target=lambda: None)
        finally:
            release.set()
            runner.shutdown(timeout_s=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(runner._workers, set())

    def test_failed_start_does_not_leave_an_unjoinable_worker(self):
        runner = ThreadBackgroundTaskRunner()
        with patch.object(
            threading.Thread, 'start', side_effect=RuntimeError('cannot start')
        ):
            with self.assertRaisesRegex(RuntimeError, 'cannot start'):
                runner.start(name='failed', target=lambda: None)
        self.assertEqual(runner._workers, set())
        runner.shutdown()
