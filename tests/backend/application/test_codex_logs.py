from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from application import codex_logs


class CodexLogsTests(unittest.TestCase):
    def test_exports_human_log_tail_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "human.log"
            log_path.write_text("first\n" + ("x" * 2100), encoding="utf-8")

            tail = codex_logs.read_human_log_tail(
                project_root=Path(tmp),
                human_log_path=log_path,
                human_log_fh=None,
                max_chars=20,
            )

        self.assertTrue(tail.startswith("[log tail]"))
        self.assertIn("trim_text_tail", codex_logs.__all__)


if __name__ == "__main__":
    unittest.main()
