from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from application.codex_config import CodexProfile
from assistant.application.provider import (
    AssistantExecutionSettings,
    AssistantProviderRequest,
)
from infrastructure import llama_cpp_runner as runtime


class LlamaCppLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = CodexProfile(
            'lifecycle', 'Local', '', model='model.gguf'
        )
        self.addCleanup(runtime.shutdown)

    def test_duplicate_load_is_rejected_and_cancelled_handle_is_closed(self):
        entered = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        model = Mock()
        events: list[dict] = []

        def load(profile, root):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release model creation')
            return runtime._LoadedModel(model, None)

        def emit(event):
            events.append(event)
            if event['state'] in ('stopped', 'error', 'running'):
                completed.set()

        with patch.object(runtime, '_load_model', side_effect=load) as factory:
            try:
                first = runtime.load_model_async(self.profile, Path('.'), emit)
                self.assertIs(first['started'], True)
                self.assertTrue(entered.wait(3))
                duplicate = runtime.load_model_async(
                    self.profile, Path('.'), emit
                )
                self.assertIs(duplicate['started'], False)
                runtime.unload_model(self.profile)
                release.set()
                self.assertTrue(completed.wait(3))
                runtime.shutdown()
                self.assertEqual(factory.call_count, 1)
                model.close.assert_called_once()
                self.assertNotIn('running', [e['state'] for e in events])
                self.assertNotIn(self.profile.id, runtime._LOADED)
            finally:
                release.set()
                runtime.shutdown()

    def test_unload_waits_for_active_inference_and_preserves_failed_close(self):
        entered = threading.Event()
        release = threading.Event()
        results = []
        model = Mock()

        def infer(**kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release inference')
            return {'choices': [{'message': {'content': 'answer'}}]}

        model.create_chat_completion.side_effect = infer
        runtime._LOADED[self.profile.id] = runtime._LoadedModel(model, None)
        settings = AssistantExecutionSettings([], [], '', 10)
        request = AssistantProviderRequest(
            'question', self.profile, 'ASK', Path('.'), settings
        )
        worker = threading.Thread(
            target=lambda: results.append(runtime.LlamaCppRunner().run(request))
        )
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            # Holding the inference lease also prevents direct close.
            guard = runtime._model_guard(self.profile.id)
            self.assertIs(guard.acquire(blocking=False), False)
            model.close.assert_not_called()
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0].text, 'answer')
        model.close.side_effect = OSError('close failed')
        with self.assertRaisesRegex(OSError, 'close failed'):
            runtime.unload_model(self.profile)
        self.assertIs(runtime._LOADED[self.profile.id].engine, model)
        model.close.side_effect = None
        runtime.unload_model(self.profile)
        self.assertNotIn(self.profile.id, runtime._LOADED)

    def test_failed_thread_start_releases_load_reservation(self):
        with patch.object(
            threading.Thread, 'start', side_effect=RuntimeError('cannot start')
        ):
            with self.assertRaisesRegex(RuntimeError, 'cannot start'):
                runtime.load_model_async(self.profile, Path('.'), Mock())
        self.assertNotIn(self.profile.id, runtime._LOADS)
