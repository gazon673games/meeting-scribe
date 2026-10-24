from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import numpy as np
from tools.asr import bench_latency

from application.native_asr_models import PARAKEET_MODEL_ID


class NativeAsrBenchmarkTests(unittest.TestCase):
    def test_native_benchmark_uses_onnx_and_reports_normalized_wer(self):
        backend = Mock()
        backend.transcribe.return_value = {'text': 'Hello, world!'}
        cfg = bench_latency.BenchConfig(
            'Parakeet', PARAKEET_MODEL_ID, 'int8', 1, 'cpu'
        )
        with (
            patch(
                'asr.infrastructure.worker_sherpa_offline.SherpaOfflineASR',
                return_value=backend,
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            results = bench_latency._bench_config(
                cfg,
                [np.zeros(16000, np.float32)],
                [1],
                0,
                1,
                reference='hello world',
            )
        self.assertEqual(results[0].wer, 0)
        self.assertIn('WER=0.00%', output.getvalue())
        backend.close.assert_called_once()

    def test_wer_counts_insertions_deletions_and_substitutions(self):
        self.assertEqual(
            bench_latency.word_error_rate('one two', 'one two three'), 0.5
        )
        self.assertEqual(bench_latency.word_error_rate('one two', 'one'), 0.5)
        self.assertEqual(
            bench_latency.word_error_rate('one two', 'one three'), 0.5
        )
        with self.assertRaisesRegex(ValueError, 'empty'):
            bench_latency.word_error_rate('', 'unexpected')


if __name__ == '__main__':
    unittest.main()
