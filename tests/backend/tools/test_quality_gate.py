from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec('radon'), 'radon is not installed'
)
class QualityGateTests(unittest.TestCase):
    def test_invalid_source_fails_instead_of_disappearing_from_report(self):
        from tools.quality.complexity_gate import collect

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'invalid.py').write_text('def broken(:\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'invalid.py'):
                collect(root)
