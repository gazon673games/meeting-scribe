from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from tools.batch_scribe_parts.output import (
    append_srt_segment,
    srt_ts,
    write_output,
)


class BatchOutputTests(unittest.TestCase):
    def test_srt_rounding_carries_across_time_boundaries(self) -> None:
        cases = (
            (-0.1, '00:00:00,000'),
            (0.9996, '00:00:01,000'),
            (59.9996, '00:01:00,000'),
            (3599.9996, '01:00:00,000'),
            (3661.234, '01:01:01,234'),
        )
        for seconds, expected in cases:
            with self.subTest(seconds=seconds):
                self.assertEqual(srt_ts(seconds), expected)

    def test_batch_and_incremental_srt_use_the_same_valid_timing(self) -> None:
        segment = {'t0': 59.9996, 't1': 60.9996, 'text': 'hello'}
        expected = '1\n00:01:00,000 --> 00:01:01,000\nhello\n'
        with tempfile.TemporaryDirectory() as raw_root:
            output = Path(raw_root) / 'transcript.srt'
            write_output([segment], output, 'srt')
            self.assertEqual(output.read_text(encoding='utf-8'), expected)
        output_stream = io.StringIO()
        append_srt_segment(segment, 1, output_stream)
        self.assertEqual(output_stream.getvalue(), expected)
