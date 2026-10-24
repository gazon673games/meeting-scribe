from __future__ import annotations

import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from settings.infrastructure import runtime_config


class _Repo:
    def __init__(
        self, *, exists: bool, data: dict | object | None = None
    ) -> None:
        self._exists = exists
        self.data = data if data is not None else {}
        self.writes: list[dict] = []

    def exists(self) -> bool:
        return self._exists

    def read(self):  # noqa: ANN201
        return self.data

    def write(self, data: dict) -> None:
        self.writes.append(dict(data))
        self.data = dict(data)
        self._exists = True


class RuntimeConfigTests(unittest.TestCase):
    def test_non_object_bundle_is_logged_and_never_written(self) -> None:
        for contents in (b'[]', b'null', b'"config"'):
            for exists in (False, True):
                with (
                    self.subTest(contents=contents, exists=exists),
                    self._frozen_bundle(contents) as root,
                    self.assertLogs(runtime_config.__name__, level='WARNING'),
                ):
                    repo = _Repo(exists=exists)
                    runtime_config.ensure_runtime_config(root, repo)
                    self.assertEqual(repo.writes, [])

    def test_merging_does_not_mutate_read_snapshot(self) -> None:
        current = {'ui': {'model': 'tiny'}, 'codex': {}}
        repo = _Repo(exists=True, data=current)
        with self._frozen_bundle(
            b'{"codex":{"profiles":[{"id":"bundled"}]}}'
        ) as root:
            runtime_config.ensure_runtime_config(root, repo)
        self.assertEqual(current, {'ui': {'model': 'tiny'}, 'codex': {}})
        self.assertEqual(
            repo.writes[0]['codex']['profiles'], [{'id': 'bundled'}]
        )

    @contextmanager
    def _frozen_bundle(self, contents: bytes) -> Iterator[Path]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_bytes(contents)
            with (
                patch.object(runtime_config.sys, 'frozen', True, create=True),
                patch.object(
                    runtime_config.sys, '_MEIPASS', str(root), create=True
                ),
            ):
                yield root

    def test_operational_failures_are_logged(self) -> None:
        contents = b'{"codex":{"profiles":[{"id":"bundled"}]}}'
        for phase in ('read', 'seed', 'merge'):
            with (
                self.subTest(phase=phase),
                self._frozen_bundle(contents) as root,
            ):
                repo = _Repo(exists=phase != 'seed')
                operation = 'read' if phase == 'read' else 'write'
                with (
                    patch.object(repo, operation, side_effect=OSError('disk')),
                    self.assertLogs(runtime_config.__name__, level='WARNING'),
                ):
                    runtime_config.ensure_runtime_config(root, repo)
                self.assertEqual(repo.writes, [])

    def test_programmer_failures_are_propagated(self) -> None:
        contents = b'{"codex":{"profiles":[{"id":"bundled"}]}}'
        for phase in ('read', 'seed', 'merge'):
            with (
                self.subTest(phase=phase),
                self._frozen_bundle(contents) as root,
            ):
                repo = _Repo(exists=phase != 'seed')
                operation = 'read' if phase == 'read' else 'write'
                with (
                    patch.object(repo, operation, side_effect=TypeError('bug')),
                    self.assertRaisesRegex(TypeError, 'bug'),
                ):
                    runtime_config.ensure_runtime_config(root, repo)

    def test_malformed_bundled_config_is_logged(self) -> None:
        for contents in (b'not-json', b'\xff'):
            for exists in (False, True):
                with (
                    self.subTest(contents=contents, exists=exists),
                    self._frozen_bundle(contents) as root,
                    self.assertLogs(runtime_config.__name__, level='WARNING'),
                ):
                    repo = _Repo(exists=exists)
                    runtime_config.ensure_runtime_config(root, repo)
                    self.assertEqual(repo.writes, [])

    def test_non_frozen_runtime_does_not_write_config(self) -> None:
        repo = _Repo(exists=False)

        with patch.object(runtime_config.sys, 'frozen', False, create=True):
            runtime_config.ensure_runtime_config(Path('.'), repo)

        self.assertEqual(repo.writes, [])

    def test_frozen_runtime_seeds_missing_config_from_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            bundle = Path(raw_root) / 'bundle'
            bundle.mkdir()
            (bundle / 'config.json').write_text(
                '{"codex":{"profiles":[{"id":"local"}]}}', encoding='utf-8'
            )
            repo = _Repo(exists=False)

            with (
                patch.object(runtime_config.sys, 'frozen', True, create=True),
                patch.object(
                    runtime_config.sys, '_MEIPASS', str(bundle), create=True
                ),
            ):
                runtime_config.ensure_runtime_config(Path(raw_root), repo)

        self.assertEqual(
            repo.writes, [{'codex': {'profiles': [{'id': 'local'}]}}]
        )

    def test_frozen_runtime_merges_bundled_codex_when_profiles_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            bundle = Path(raw_root) / 'bundle'
            bundle.mkdir()
            (bundle / 'config.json').write_text(
                '{"codex":{"profiles":[{"id":"bundled"}],"proxy":"http://proxy"}}',
                encoding='utf-8',
            )
            repo = _Repo(
                exists=True, data={'ui': {'model': 'tiny'}, 'codex': {}}
            )

            with (
                patch.object(runtime_config.sys, 'frozen', True, create=True),
                patch.object(
                    runtime_config.sys, '_MEIPASS', str(bundle), create=True
                ),
            ):
                runtime_config.ensure_runtime_config(Path(raw_root), repo)

        self.assertEqual(repo.writes[0]['ui'], {'model': 'tiny'})
        self.assertEqual(
            repo.writes[0]['codex']['profiles'][0]['id'], 'bundled'
        )

    def test_frozen_runtime_keeps_existing_codex_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            bundle = Path(raw_root) / 'bundle'
            bundle.mkdir()
            (bundle / 'config.json').write_text(
                '{"codex":{"profiles":[{"id":"bundled"}]}}', encoding='utf-8'
            )
            repo = _Repo(
                exists=True, data={'codex': {'profiles': [{'id': 'user'}]}}
            )

            with (
                patch.object(runtime_config.sys, 'frozen', True, create=True),
                patch.object(
                    runtime_config.sys, '_MEIPASS', str(bundle), create=True
                ),
            ):
                runtime_config.ensure_runtime_config(Path(raw_root), repo)

        self.assertEqual(repo.writes, [])


if __name__ == '__main__':
    unittest.main()
