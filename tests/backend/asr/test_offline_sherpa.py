from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from application.native_asr_models import PARAKEET_MODEL_ID
from asr.infrastructure.offline_runner import OfflineProfile, OfflineRunner


class OfflineSherpaTests(unittest.TestCase):
    def test_offline_pass_resamples_stereo_in_bounded_chunks_and_closes_backend(
        self,
    ):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wav = root / 'meeting.wav'
            sf.write(wav, np.zeros((31 * 8000, 2), np.float32), 8000)
            backend = Mock()
            backend.transcribe.side_effect = [
                {'text': 'first'},
                {'text': 'second'},
            ]
            with patch(
                'asr.infrastructure.offline_sherpa.SherpaOfflineASR',
                return_value=backend,
            ):
                output = OfflineRunner(project_root=root).run(
                    wav_path=wav,
                    out_txt=root / 'out.txt',
                    out_jsonl=root / 'out.jsonl',
                    profile=OfflineProfile(
                        model_name=PARAKEET_MODEL_ID, device='cpu'
                    ),
                )
            self.assertEqual(
                output.read_text(encoding='utf-8'), 'first\nsecond\n'
            )
            self.assertEqual(
                [
                    len(call.args[0])
                    for call in backend.transcribe.call_args_list
                ],
                [30 * 16000, 16000],
            )
            backend.close.assert_called_once()
            self.assertEqual(
                len(
                    (root / 'out.jsonl')
                    .read_text(encoding='utf-8')
                    .splitlines()
                ),
                2,
            )

    def test_inference_failure_closes_backend_and_preserves_existing_text(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wav = root / 'meeting.wav'
            sf.write(wav, np.zeros(8000, np.float32), 8000)
            output = root / 'out.txt'
            output.write_text('keep', encoding='utf-8')
            backend = Mock()
            backend.transcribe.side_effect = RuntimeError('decode failed')
            with patch(
                'asr.infrastructure.offline_sherpa.SherpaOfflineASR',
                return_value=backend,
            ):
                with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                    OfflineRunner(project_root=root).run(
                        wav_path=wav,
                        out_txt=output,
                        profile=OfflineProfile(
                            model_name=PARAKEET_MODEL_ID, device='cpu'
                        ),
                    )
            backend.close.assert_called_once()
            self.assertEqual(output.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
