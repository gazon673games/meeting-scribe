from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from application.local_paths import project_human_logs_dir
from settings.infrastructure.json_config_repository import JsonConfigRepository
from transcription.infrastructure.file_transcript_store import FileTranscriptStore


class RepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_parent = Path(__file__).resolve().parents[1] / "tmp_tests"
        cls._tmp_parent.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp_parent, ignore_errors=True)

    def _project_root(self, name: str) -> Path:
        root = self._tmp_parent / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        return root

    def test_json_config_repository_round_trips_config(self) -> None:
        root = self._project_root("json_config")
        repo = JsonConfigRepository(root / "config.json")

        repo.write({"version": 2, "ui": {"profile": "Realtime"}})

        self.assertTrue(repo.exists())
        self.assertEqual(repo.read()["ui"]["profile"], "Realtime")

    def test_file_transcript_store_writes_project_local_logs(self) -> None:
        root = self._project_root("transcript_store")
        store = FileTranscriptStore(root)

        path = store.open_human_log()
        store.write_human_line("hello")
        store.close_human_log()

        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path.parent, project_human_logs_dir(root))
        self.assertIn("hello", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
