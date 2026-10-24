from __future__ import annotations

import io
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from tools import batch_scribe
from tools.batch_scribe_parts import cli


class BatchBootstrapTests(unittest.TestCase):
    def test_import_does_not_configure_cache_or_modify_environment(self):
        environment = dict(os.environ)
        with (
            patch.object(sys, 'path', list(sys.path)),
            patch(
                'application.local_paths.configure_project_local_io'
            ) as configure,
        ):
            runpy.run_path(batch_scribe.__file__)
        configure.assert_not_called()
        self.assertEqual(dict(os.environ), environment)

    def test_bootstrap_uses_explicit_project_and_model_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / 'backend' / 'src'
            source_root.mkdir(parents=True)
            models = root / 'cache'
            with (
                patch.object(sys, 'path', list(sys.path)),
                patch(
                    'application.local_paths.configure_project_local_io'
                ) as configure,
            ):
                batch_scribe._bootstrap(root, models)
                self.assertIn(str(root.resolve()), sys.path)
                self.assertIn(str(source_root.resolve()), sys.path)
            configure.assert_called_once_with(root.resolve(), models_dir=models)

    def test_bootstrap_propagates_io_and_programming_failures(self):
        for error in (PermissionError('cache denied'), RuntimeError('bug')):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(sys, 'path', list(sys.path)),
                patch(
                    'application.local_paths.configure_project_local_io',
                    side_effect=error,
                ),
                self.assertRaises(type(error)) as raised,
            ):
                batch_scribe._bootstrap()
            self.assertIs(raised.exception, error)

    def test_default_cli_prepares_cache_and_writes_output(self):
        bootstrap = Mock()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'meeting.wav'
            source.write_bytes(b'audio fixture')
            with (
                patch.object(cli, 'Scribe') as factory,
                redirect_stdout(io.StringIO()),
            ):
                factory.return_value.__enter__.return_value.process.return_value = []
                cli.main([str(source)], bootstrap=bootstrap)
            bootstrap.assert_called_once_with(None, None)
            factory.assert_called_once()
            self.assertTrue(source.with_suffix('.srt').is_file())

    def test_failed_cache_setup_does_not_acquire_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'meeting.wav'
            source.write_bytes(b'audio fixture')
            with (
                patch.object(cli, 'Scribe') as factory,
                self.assertRaisesRegex(PermissionError, 'cache denied'),
            ):
                cli.main(
                    [str(source)],
                    bootstrap=Mock(side_effect=PermissionError('cache denied')),
                )
            factory.assert_not_called()

    def test_help_does_not_prepare_runtime_directories(self):
        output = io.StringIO()
        with (
            patch(
                'application.local_paths.configure_project_local_io'
            ) as configure,
            redirect_stdout(output),
            self.assertRaises(SystemExit) as raised,
        ):
            batch_scribe.main(['--help'])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn('batch_scribe', output.getvalue())
        configure.assert_not_called()
