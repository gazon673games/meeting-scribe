from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from infrastructure.huggingface_transfer import download_snapshot


class HuggingFaceTransferTests(unittest.TestCase):
    def test_per_request_proxy_preserves_process_environment(self) -> None:
        calls = []

        def download(*, repo_id, cache_dir, local_files_only, proxies):
            calls.append((repo_id, cache_dir, local_files_only, proxies))
            self.assertEqual(os.environ['HTTP_PROXY'], 'ambient')

        with (
            patch.dict(os.environ, {'HTTP_PROXY': 'ambient'}),
            patch.dict(
                'sys.modules',
                {
                    'huggingface_hub': SimpleNamespace(
                        snapshot_download=download
                    ),
                },
            ),
        ):
            download_snapshot(
                repo_id='owner/repo',
                cache_dir='cache',
                proxy='http://proxy',
            )
            self.assertEqual(os.environ['HTTP_PROXY'], 'ambient')
        self.assertEqual(
            calls,
            [
                (
                    'owner/repo',
                    'cache',
                    False,
                    {'http': 'http://proxy', 'https': 'http://proxy'},
                )
            ],
        )

    def test_legacy_sdk_restores_environment_when_download_fails(self) -> None:
        def download(*, repo_id, cache_dir, local_files_only):
            self.assertEqual(os.environ['HTTP_PROXY'], 'http://proxy')
            raise OSError('connection closed')

        with (
            patch.dict(os.environ, {'HTTP_PROXY': 'ambient'}),
            patch.dict(
                'sys.modules',
                {
                    'huggingface_hub': SimpleNamespace(
                        snapshot_download=download
                    ),
                },
            ),
        ):
            with self.assertRaisesRegex(OSError, 'connection closed'):
                download_snapshot(
                    repo_id='owner/repo',
                    cache_dir='cache',
                    proxy='http://proxy',
                )
            self.assertEqual(os.environ['HTTP_PROXY'], 'ambient')
