from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from application.native_asr_models import (
    PARAKEET_MODEL_ID,
    QWEN_ASR_MODEL_ID,
    native_asr_required_files,
)
from asr.infrastructure.worker_sherpa_offline import SherpaOfflineASR


class SherpaOfflineBackendTests(unittest.TestCase):
    def test_each_model_uses_its_sdk_and_preserves_recognition_and_cleanup(
        self,
    ):
        for model_name, method in (
            (PARAKEET_MODEL_ID, 'from_transducer'),
            (QWEN_ASR_MODEL_ID, 'from_qwen3_asr'),
        ):
            with (
                self.subTest(model=model_name),
                tempfile.TemporaryDirectory() as raw,
            ):
                root = Path(raw)
                for name in native_asr_required_files(model_name):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'fixture')
                sdk = ModuleType('sherpa_onnx')
                sdk.OfflineRecognizer = Mock()
                recognizer = getattr(sdk.OfflineRecognizer, method).return_value
                stream = recognizer.create_stream.return_value
                stream.result = SimpleNamespace(
                    text=' hello world ', timestamps=[0.1, 0.4]
                )
                with (
                    patch.dict('sys.modules', {'sherpa_onnx': sdk}),
                    self.assertLogs(
                        'asr.infrastructure.sherpa_runtime', level='WARNING'
                    ),
                ):
                    backend = SherpaOfflineASR(
                        model_dir=root,
                        model_name=model_name,
                        language='ru',
                        device='cuda',
                        compute_type='int8',
                        cpu_threads=3,
                        num_workers=2,
                        beam_size=1,
                        initial_prompt=None,
                        hotwords='OpenAI',
                    )
                samples = np.zeros(16000, np.float32)
                self.assertEqual(backend.device, 'cpu')
                result = backend.transcribe(samples)
                self.assertEqual(result['text'], 'hello world')
                self.assertEqual(result['timestamps'], [0.1, 0.4])
                recognizer.decode_stream.assert_called_once_with(stream)
                options = getattr(
                    sdk.OfflineRecognizer, method
                ).call_args.kwargs
                self.assertEqual(options['provider'], 'cpu')
                self.assertEqual(options['num_threads'], 3)
                if model_name == QWEN_ASR_MODEL_ID:
                    self.assertEqual(options['hotwords'], 'OpenAI')
                    self.assertEqual(
                        options['tokenizer'], str(root / 'tokenizer')
                    )
                self.assertEqual(
                    [
                        word['text']
                        for word in backend.transcribe_words(samples)
                    ],
                    ['hello', 'world'],
                )
                backend.close()
                backend.close()
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    backend.transcribe(samples)

    def test_incomplete_model_is_rejected_before_creating_native_recognizer(
        self,
    ):
        sdk = ModuleType('sherpa_onnx')
        sdk.OfflineRecognizer = Mock()
        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict('sys.modules', {'sherpa_onnx': sdk}),
        ):
            with self.assertRaisesRegex(FileNotFoundError, 'incomplete'):
                SherpaOfflineASR(
                    model_dir=root,
                    model_name=QWEN_ASR_MODEL_ID,
                    language=None,
                    device='cpu',
                    compute_type='int8',
                    cpu_threads=1,
                    num_workers=1,
                    beam_size=1,
                    initial_prompt=None,
                    hotwords=None,
                )
        sdk.OfflineRecognizer.from_qwen3_asr.assert_not_called()


if __name__ == '__main__':
    unittest.main()
