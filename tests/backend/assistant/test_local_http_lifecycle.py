from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from infrastructure.local_llm_parts.errors import LocalLlmError
from infrastructure.local_llm_parts.http_client import request_json


class LocalHttpLifecycleTests(unittest.TestCase):
    def test_body_timeout_is_translated_and_response_is_closed(self):
        response = Mock()
        timeout = TimeoutError('body stalled')
        response.read.side_effect = timeout
        with patch(
            'infrastructure.local_llm_parts.http_client.http_open',
            return_value=response,
        ):
            with self.assertRaises(LocalLlmError) as raised:
                request_json('GET', 'http://localhost/models', timeout_s=1)
        self.assertEqual(raised.exception.code, 'local_llm_unavailable')
        self.assertIs(raised.exception.__cause__, timeout)
        response.close.assert_called_once()

    def test_close_error_does_not_replace_primary_body_failure(self):
        response = Mock()
        response.read.side_effect = TimeoutError('primary')
        response.close.side_effect = OSError('secondary')
        with (
            patch(
                'infrastructure.local_llm_parts.http_client.http_open',
                return_value=response,
            ),
            self.assertLogs(
                'infrastructure.local_llm_parts.http_client', level='WARNING'
            ),
            self.assertRaises(LocalLlmError) as raised,
        ):
            request_json('GET', 'http://localhost/models', timeout_s=1)
        self.assertIn('primary', raised.exception.message)
