from __future__ import annotations

import io
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from tools.asr import bench_latency, eval_vad


class AsrBenchmarkContractsTests(unittest.TestCase):
    def test_structured_inference_error_does_not_become_a_latency_sample(self):
        model = Mock()
        model.transcribe.return_value = {'error': 'decode failed'}
        with self.assertRaisesRegex(RuntimeError, 'decode failed'):
            bench_latency._transcribe_checked(
                model, np.zeros(16, np.float32), beam_size=1
            )

    def test_nonfinite_vad_threshold_is_rejected_before_audio_execution(self):
        with (
            patch('sys.argv', ['eval_vad', '--min-f1', 'nan']),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            eval_vad.main()
        self.assertEqual(raised.exception.code, 2)

    def test_p95_uses_upper_nearest_rank_for_small_samples(self):
        result = bench_latency.RunResult('test', 1.0, [1.0, 2.0])
        self.assertEqual(result.p95, 2.0)

    def test_invalid_wav_width_is_rejected_instead_of_decoded_as_pcm16(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pcm8.wav'
            with wave.open(str(path), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(1)
                output.setframerate(16000)
                output.writeframes(b'\x80' * 16)
            with redirect_stdout(io.StringIO()):
                self.assertIsNone(bench_latency._load_wav(path))

    def test_inference_failure_releases_benchmark_model(self):
        model = Mock()
        model.transcribe.side_effect = ValueError('decode failed')
        config = bench_latency.BenchConfig('test', 'model', 'float32', 1, 'cpu')
        with (
            patch(
                'asr.infrastructure.worker_faster_whisper.FasterWhisperASR',
                return_value=model,
            ),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, 'decode failed'),
        ):
            bench_latency._bench_config(
                config, [np.zeros(16, np.float32)], [0.001], 0, 1
            )
        model.close.assert_called_once()

    def test_vad_summary_aggregates_samples_instead_of_scenario_averages(self):
        results = []
        for labels, predictions in (
            ([True, True, False], [False, True, False]),
            ([True, True, False] * 3, [True, True, False] * 3),
        ):
            confusion = eval_vad._confusion(labels, predictions)
            results.append(
                eval_vad.ScenarioResult(
                    'test',
                    1.0,
                    confusion,
                    confusion,
                    eval_vad._segment_metrics(labels, predictions, frame_ms=10),
                    0,
                    0,
                )
            )
        summary = eval_vad._aggregate_segments(results)
        self.assertEqual(summary.onset_delay_ms_avg, 2.5)
        self.assertAlmostEqual(summary.onset_delay_ms_p95, 8.5)
        self.assertNotIn('onset_delays_ms', eval_vad._segment_record(summary))
