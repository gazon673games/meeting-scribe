from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from application.model_download import download_model_async, model_metadata
from application.native_asr_models import (
    PARAKEET_MODEL_ID,
    QWEN_ASR_MODEL_ID,
    is_native_asr_model_cached,
    native_asr_model_dir,
    native_asr_required_files,
)
from infrastructure.native_asr_install import install_native_asr_model


def _write_archive(
    path: Path, model_name: str, *, invalid: bool = False
) -> None:
    directory = native_asr_model_dir(model_name).name
    with tarfile.open(path, 'w:bz2') as archive:
        for index, name in enumerate(native_asr_required_files(model_name)):
            entry = tarfile.TarInfo(f'{directory}/{name}')
            if invalid and index == 1:
                entry.type = tarfile.SYMTYPE
                entry.linkname = '../../escaped'
                archive.addfile(entry)
            else:
                entry.size = 7
                archive.addfile(entry, io.BytesIO(b'fixture'))
        unrelated = tarfile.TarInfo('../escaped')
        unrelated.size = 3
        archive.addfile(unrelated, io.BytesIO(b'bad'))


class NativeAsrInstallTests(unittest.TestCase):
    def test_async_installer_completes_once_on_success_and_failure(self):
        for failure in (None, RuntimeError('download failed')):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as raw,
            ):
                done = Mock()

                def start(*, target, name):
                    target()

                with (
                    patch(
                        'application.model_download_parts.download.model_download_tasks.start',
                        side_effect=start,
                    ),
                    patch(
                        'infrastructure.native_asr_install.install_native_asr_model',
                        side_effect=failure,
                    ),
                ):
                    download_model_async(
                        PARAKEET_MODEL_ID, Mock(), done, models_dir=raw
                    )
                done.assert_called_once()
                if failure is None:
                    self.assertIsNone(done.call_args.args[0])
                else:
                    self.assertIn('download failed', done.call_args.args[0])

    def test_both_bundles_install_with_metadata_and_reuse_complete_cache(self):
        for model_name in (PARAKEET_MODEL_ID, QWEN_ASR_MODEL_ID):
            with (
                self.subTest(model=model_name),
                tempfile.TemporaryDirectory() as raw,
            ):
                root = Path(raw) / 'models'

                def download(url, destination, proxy, on_progress):
                    self.assertTrue(
                        url.startswith(
                            'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/'
                        )
                    )
                    self.assertEqual(proxy, 'http://proxy:8080')
                    _write_archive(destination, model_name)

                with patch(
                    'infrastructure.native_asr_install._download_archive',
                    side_effect=download,
                ) as fetch:
                    target = install_native_asr_model(
                        model_name, models_dir=root, proxy='http://proxy:8080'
                    )
                    self.assertEqual(
                        install_native_asr_model(model_name, models_dir=root),
                        target,
                    )
                    fetch.assert_called_once()
                self.assertTrue(
                    is_native_asr_model_cached(model_name, models_dir=root)
                )
                self.assertFalse((root / 'escaped').exists())
                self.assertEqual(list(root.glob('.native-asr-*')), [])
                metadata = model_metadata(model_name, models_dir=root)
                self.assertTrue(metadata['compatible'])
                self.assertEqual(metadata['format'], 'onnx')
                self.assertEqual(metadata['missing'], [])
                self.assertEqual(
                    set(metadata['presentFiles']),
                    set(native_asr_required_files(model_name)),
                )

    def test_invalid_archive_never_becomes_an_installed_model(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / 'models'

            def download(url, destination, proxy, on_progress):
                _write_archive(destination, QWEN_ASR_MODEL_ID, invalid=True)

            with patch(
                'infrastructure.native_asr_install._download_archive',
                side_effect=download,
            ):
                with self.assertRaisesRegex(
                    ValueError, 'Invalid model archive'
                ):
                    install_native_asr_model(QWEN_ASR_MODEL_ID, models_dir=root)
            self.assertFalse(
                native_asr_model_dir(
                    QWEN_ASR_MODEL_ID, models_dir=root
                ).exists()
            )
            self.assertEqual(list(root.iterdir()), [])

    def test_existing_incomplete_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = native_asr_model_dir(PARAKEET_MODEL_ID, models_dir=root)
            target.mkdir()
            marker = target / 'in-progress'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                install_native_asr_model(PARAKEET_MODEL_ID, models_dir=root)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')


if __name__ == '__main__':
    unittest.main()
