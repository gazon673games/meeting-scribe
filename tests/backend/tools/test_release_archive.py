from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools.release.build_release import _zip_dir_contents


class ReleaseArchiveTests(unittest.TestCase):
    def test_failed_archive_build_preserves_existing_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'app.txt').write_text('new', encoding='ascii')
            archive = root / 'release.zip'
            archive.write_bytes(b'previous release')
            with (
                patch.object(
                    zipfile.ZipFile, 'write', side_effect=OSError('disk full')
                ),
                self.assertRaisesRegex(OSError, 'disk full'),
            ):
                _zip_dir_contents(source, archive)
            self.assertEqual(archive.read_bytes(), b'previous release')
            self.assertEqual(list(root.glob('*.tmp')), [])
            _zip_dir_contents(source, archive)
            with zipfile.ZipFile(archive) as result:
                self.assertEqual(result.read('app.txt'), b'new')
