from __future__ import annotations

import unittest
from unittest.mock import Mock

from asr.infrastructure.worker_sherpa_nemotron import (
    _result_dict,
    _text_to_words,
)


class NemotronResultTests(unittest.TestCase):
    def test_normalized_result_preserves_sdk_timestamps_and_extra_fields(self):
        recognizer = Mock()
        recognizer.get_result_as_json_string.return_value = (
            '{"text": "hello world", "timestamps": [0.1, 0.8], '
            '"tokens": ["hello", "world"]}'
        )
        stream = object()

        result = _result_dict(recognizer, stream)

        self.assertEqual(result['text'], 'hello world')
        self.assertEqual(result['timestamps'], [0.1, 0.8])
        self.assertEqual(result['tokens'], ['hello', 'world'])
        self.assertEqual(
            _text_to_words(result['text'], result['timestamps']),
            [
                {'text': 'hello', 'start': 0.0, 'end': 0.4},
                {'text': 'world', 'start': 0.4, 'end': 0.8},
            ],
        )
        recognizer.get_result.assert_not_called()

    def test_missing_text_is_an_empty_recognition_result(self):
        recognizer = Mock()
        recognizer.get_result_as_json_string.return_value = '{"text": null}'

        self.assertEqual(_result_dict(recognizer, object()), {'text': ''})

    def test_invalid_sdk_json_falls_back_to_plain_text(self):
        recognizer = Mock()
        recognizer.get_result_as_json_string.return_value = 'invalid json'
        recognizer.get_result.return_value = 'fallback'
        stream = object()

        self.assertEqual(_result_dict(recognizer, stream), {'text': 'fallback'})
        recognizer.get_result.assert_called_once_with(stream)


if __name__ == '__main__':
    unittest.main()
