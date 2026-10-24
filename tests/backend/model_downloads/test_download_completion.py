from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from application import diarization_model_download as diarization
from application import llm_model_download as llm
from application.model_download_parts import download as asr


class DownloadCompletionTests(unittest.TestCase):
    def test_failed_progress_monitor_start_still_reports_completion_once(self):
        completed = Mock()

        def run_inline(*, name, target):
            target()

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                asr.model_download_tasks, 'start', side_effect=run_inline
            ),
            patch.object(
                asr.threading.Thread,
                'start',
                side_effect=RuntimeError('monitor start failed'),
            ),
            patch.object(asr, 'download_snapshot') as download,
        ):
            asr.download_model_async(
                'owner/repo', Mock(), completed, models_dir=Path(directory)
            )
        completed.assert_called_once()
        self.assertIn('monitor start failed', completed.call_args.args[0])
        download.assert_not_called()

    def test_completion_callback_errors_are_not_retried(self) -> None:
        for kind in ('asr', 'diarization', 'llm'):
            with (
                self.subTest(kind=kind),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                completion = Mock(side_effect=TypeError('callback bug'))
                target = root / 'models' / 'llm' / 'model' / 'model.gguf'
                target.parent.mkdir(parents=True)
                target.write_bytes(b'gguf')
                spec = diarization.RECOMMENDED_DIARIZATION_MODELS[0]
                staging = (
                    root / 'models' / 'diarization' / (spec.file_name + '.part')
                )
                staging.parent.mkdir(parents=True)
                staging.write_bytes(b'onnx')

                with (
                    patch.object(asr.threading, 'Thread') as thread,
                    patch.object(asr, 'download_snapshot'),
                    patch.object(
                        asr,
                        'inspect_model_path',
                        return_value={'compatible': True},
                    ),
                    patch.object(diarization, '_download_file'),
                    patch.object(llm, '_download_url'),
                ):
                    callbacks = {
                        'on_progress': lambda update: None,
                        'on_done': completion,
                    }
                    if kind == 'asr':
                        asr.download_model_async(
                            'org/repo', models_dir=root, **callbacks
                        )
                    elif kind == 'diarization':
                        diarization.download_diarization_model_async(
                            name=spec.name, project_root=root, **callbacks
                        )
                    else:
                        llm.download_llm_model_async(
                            name='https://example.test/model.gguf',
                            project_root=root,
                            **callbacks,
                        )
                    worker = thread.call_args.kwargs['target']
                    with self.assertRaisesRegex(TypeError, 'callback bug'):
                        worker()

                completion.assert_called_once_with(None)
                if kind == 'asr':
                    thread.return_value.join.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
