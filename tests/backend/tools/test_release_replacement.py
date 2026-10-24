from __future__ import annotations

import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tools.release.replace_release_assets import ReleaseAsset, replace_asset


class _Release:
    def __init__(self, *, upload_fails=False, promotion_fails=False):
        self.assets = [ReleaseAsset(1, 'app.zip', 3, 'uploaded')]
        self.operations = []
        self.upload_fails = upload_fails
        self.promotion_fails = promotion_fails

    def list_assets(self):
        return list(self.assets)

    def upload(self, path):
        self.operations.append(('upload', path.name))
        if self.upload_fails:
            raise subprocess.CalledProcessError(1, ['gh', 'upload'])
        self.assets.append(
            ReleaseAsset(2, path.name, path.stat().st_size, 'uploaded')
        )

    def rename(self, asset, name):
        self.operations.append(('rename', asset.id, name))
        if asset.id == 2 and self.promotion_fails:
            raise subprocess.CalledProcessError(1, ['gh', 'rename'])
        self.assets = [
            replace(item, name=name) if item.id == asset.id else item
            for item in self.assets
        ]

    def delete(self, asset):
        self.operations.append(('delete', asset.id))
        self.assets = [item for item in self.assets if item.id != asset.id]


class ReleaseReplacementTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'app.zip'
        self.path.write_bytes(b'new archive')

    def test_upload_failure_keeps_previous_asset_untouched(self):
        release = _Release(upload_fails=True)
        with self.assertRaises(subprocess.CalledProcessError):
            replace_asset(release, self.path)
        self.assertEqual(
            release.assets, [ReleaseAsset(1, 'app.zip', 3, 'uploaded')]
        )
        self.assertEqual(len(release.operations), 1)

    def test_promotion_failure_restores_name_and_keeps_both_byte_streams(self):
        release = _Release(promotion_fails=True)
        with self.assertRaises(subprocess.CalledProcessError):
            replace_asset(release, self.path)
        self.assertEqual(release.assets[0].name, 'app.zip')
        self.assertEqual(len(release.assets), 2)
        self.assertFalse(
            any(operation[0] == 'delete' for operation in release.operations)
        )

    def test_success_deletes_previous_only_after_verified_upload_and_promotion(
        self,
    ):
        release = _Release()
        replace_asset(release, self.path)
        self.assertEqual(
            release.assets, [ReleaseAsset(2, 'app.zip', 11, 'uploaded')]
        )
        self.assertEqual(release.operations[0][0], 'upload')
        self.assertEqual(
            release.operations[-2:], [('rename', 2, 'app.zip'), ('delete', 1)]
        )
