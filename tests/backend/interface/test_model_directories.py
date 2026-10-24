from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from application.model_directories import model_directory
from interface.backend import ElectronBackend
from settings.infrastructure.json_config_repository import JsonConfigRepository
from tests.helpers.electron_interface_fakes import _DeviceCatalog


class ModelDirectoryTests(unittest.TestCase):
    def test_defaults_and_custom_base_preserve_existing_layout(self):
        root = Path(tempfile.gettempdir()).resolve() / 'model-directories-test'
        for kind, suffix in [
            ('asr', ''),
            ('diarization', 'diarization'),
            ('llm', 'llm'),
        ]:
            with self.subTest(kind=kind):
                self.assertEqual(
                    model_directory(root, {}, kind), root / 'models' / suffix
                )
        config = {'models': {'cache_dir': str(root / 'cache')}}
        self.assertEqual(model_directory(root, config, 'llm'), root / 'cache')

    def test_configured_subdirectory_and_request_override_share_one_policy(
        self,
    ):
        root = Path(tempfile.gettempdir()).resolve() / 'model-directories-test'
        config = {'models': {'asr_subdir': 'speech'}}
        self.assertEqual(
            model_directory(root, config, 'asr'), root / 'models' / 'speech'
        )
        self.assertEqual(
            model_directory(root, config, 'asr', {'modelsSubdir': 'other'}),
            root / 'models' / 'other',
        )
        for subdir in ['../outside', str(root / 'absolute')]:
            with self.subTest(subdir=subdir), self.assertRaises(ValueError):
                model_directory(root, config, 'asr', {'modelsSubdir': subdir})

    def test_backend_catalogs_and_startup_use_saved_subdirectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = JsonConfigRepository(root / 'config.json')
            repository.write(
                {
                    'models': {
                        'asr_subdir': 'speech',
                        'diarization_subdir': 'speakers',
                        'llm_subdir': 'chat',
                    }
                }
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog())
            self.assertEqual(
                backend._models_dir_from_params(), root / 'models' / 'speech'
            )
            self.assertEqual(
                Path(backend.list_diarization_models()['modelsDir']),
                root / 'models' / 'speakers',
            )
            self.assertEqual(
                Path(backend.list_llm_models()['modelsDir']),
                root / 'models' / 'chat',
            )

    def test_invalid_subdirectory_does_not_replace_saved_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = JsonConfigRepository(root / 'config.json')
            repository.write({'models': {'asr_subdir': 'speech'}})
            backend = ElectronBackend(root, repository, _DeviceCatalog())
            with self.assertRaises(ValueError):
                backend.save_config(
                    {'config': {'models': {'asr_subdir': '../outside'}}}
                )
            self.assertEqual(
                repository.read()['models']['asr_subdir'], 'speech'
            )
