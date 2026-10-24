from __future__ import annotations

import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from settings.infrastructure.json_config_repository import JsonConfigRepository


class ConfigRepositoryTests(unittest.TestCase):
    def test_non_object_config_is_logged_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('[]', encoding='utf-8')
            repo = JsonConfigRepository(path)
            with self.assertLogs(
                'settings.infrastructure.json_config_repository',
                level='WARNING',
            ):
                self.assertEqual(repo.read(), {})

    def test_concurrent_writes_use_independent_staging_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = JsonConfigRepository(root / 'config.json')
            other_repo = JsonConfigRepository(repo.path)
            ready = threading.Barrier(2)
            write_text = Path.write_text
            staged = []
            staged_lock = threading.Lock()

            def synchronized_stage(path, contents, **kwargs):
                written = write_text(path, contents, **kwargs)
                with staged_lock:
                    staged.append(path)
                ready.wait(timeout=5.0)
                return written

            with (
                patch.object(Path, 'write_text', synchronized_stage),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                results = [
                    executor.submit(owner.write, {'version': version})
                    for version, owner in ((1, repo), (2, other_repo))
                ]
                for result in results:
                    result.result(timeout=10.0)

            self.assertIn(repo.read()['version'], (1, 2))
            self.assertEqual(len(set(staged)), 2)
            self.assertIs(repo._lock, other_repo._lock)
            self.assertEqual(list(root.iterdir()), [repo.path])

    def test_failed_write_preserves_config_and_cleans_staging_files(
        self,
    ) -> None:
        for operation in ('write_text', 'replace'):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                repo = JsonConfigRepository(root / 'config.json')
                repo.write({'version': 1})
                with (
                    patch.object(Path, operation, side_effect=OSError('disk')),
                    self.assertRaisesRegex(OSError, 'disk'),
                ):
                    repo.write({'version': 2})
                self.assertEqual(repo.read(), {'version': 1})
                self.assertEqual(list(root.iterdir()), [repo.path])


if __name__ == '__main__':
    unittest.main()
