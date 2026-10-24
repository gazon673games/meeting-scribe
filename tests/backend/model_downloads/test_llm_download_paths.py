from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from application import llm_model_download as models
from infrastructure import background_tasks


class LlmDownloadPathTests(unittest.TestCase):
    def test_download_stays_in_its_model_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, valid in (
                ('../neighbor.gguf', False),
                (str(root / 'outside.gguf'), False),
                ('nested/model_\u0430.gguf', True),
            ):
                with self.subTest(filename=filename):
                    source = models.LlmSource(
                        filename=filename, folder='repo', repo_id='org/repo'
                    )
                    completed = Mock()

                    def transfer(source, target: Path, **options) -> None:  # noqa: ANN001
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(b'gguf')

                    with (
                        patch.object(
                            background_tasks.threading, 'Thread'
                        ) as thread,
                        patch.object(
                            models, '_resolve_repo_source', return_value=source
                        ),
                        patch.object(
                            models, '_download_hf_file', side_effect=transfer
                        ) as download,
                    ):
                        models.download_llm_model_async(
                            name='org/repo',
                            project_root=root,
                            on_progress=lambda update: None,
                            on_done=completed,
                        )
                        thread.call_args.kwargs['target']()

                    if valid:
                        completed.assert_called_once_with(None)
                        download.assert_called_once()
                    else:
                        completed.assert_called_once()
                        self.assertIn(
                            'inside its model directory',
                            completed.call_args.args[0],
                        )
                        download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
