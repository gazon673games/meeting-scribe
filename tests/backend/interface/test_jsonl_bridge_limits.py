from __future__ import annotations

import io
import json
import threading
import unittest

from interface.jsonl_bridge import JsonLineBridge


class _CountingInput(io.StringIO):
    def __init__(self, count: int) -> None:
        super().__init__(
            ''.join(
                json.dumps({'id': index, 'method': 'ping'}) + '\n'
                for index in range(count)
            )
        )
        self.read_count = 0
        self.third_read = threading.Event()

    def __next__(self) -> str:
        line = super().__next__()
        self.read_count += 1
        if self.read_count == 3:
            self.third_read.set()
        return line


class JsonLineBridgeLimitTests(unittest.TestCase):
    def test_disconnect_signals_running_lifecycle_work_before_waiting_for_it(
        self,
    ):
        disconnected = threading.Event()
        observed = []

        def handle(method, params):
            observed.append(disconnected.wait(2))
            return {'cancelled': disconnected.is_set()}

        output = io.StringIO()
        bridge = JsonLineBridge(
            handle,
            stdin=io.StringIO('{"id":1,"method":"start_session"}\n'),
            stdout=output,
            stderr=io.StringIO(),
            on_disconnect=disconnected.set,
        )
        bridge.serve_forever()
        self.assertEqual(observed, [True])
        reply = json.loads(output.getvalue().splitlines()[-1])
        self.assertIs(reply['result']['cancelled'], True)

    def test_request_capacity_pauses_input_until_work_completes(self) -> None:
        source = _CountingInput(8)
        release = threading.Event()
        output = io.StringIO()

        def handler(method, params):
            if not release.wait(2):
                raise TimeoutError('Test handler was not released')
            return {'pong': True}

        bridge = JsonLineBridge(
            handler,
            stdin=source,
            stdout=output,
            stderr=io.StringIO(),
            max_pending_requests=2,
        )
        worker = threading.Thread(target=bridge.serve_forever, daemon=True)
        worker.start()
        try:
            self.assertTrue(source.third_read.wait(1))
            self.assertEqual(source.read_count, 3)
        finally:
            release.set()
            worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(replies), 9)
        self.assertTrue(all(reply['ok'] for reply in replies[1:]))

    def test_request_capacity_must_be_positive(self) -> None:
        for capacity in [0, -1, 1.5, True]:
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                JsonLineBridge(
                    lambda method, params: None, max_pending_requests=capacity
                )
