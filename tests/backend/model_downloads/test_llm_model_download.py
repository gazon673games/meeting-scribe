from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests import PROJECT_ROOT
from application.llm_model_download import LlmSource, _choose_gguf_file, _resolve_repo_source, list_llm_models, parse_llm_source


class LlmModelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_parent = PROJECT_ROOT / "tmp_tests"
        cls._tmp_parent.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp_parent, ignore_errors=True)

    def _root(self, name: str) -> Path:
        root = self._tmp_parent / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        return root

    def test_lists_local_gguf_models_recursively(self) -> None:
        root = self._root("llm_scan")
        model_dir = root / "models" / "llm" / "llama3"
        model_dir.mkdir(parents=True)
        model_path = model_dir / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        model_path.write_bytes(b"gguf")

        result = list_llm_models(project_root=root)

        self.assertEqual(result["modelsDir"], str(root / "models" / "llm"))
        self.assertEqual(result["models"][0]["label"], model_path.name)
        self.assertEqual(result["models"][0]["modelAlias"], "Meta-Llama-3-8B-Instruct-Q4_K_M")
        self.assertTrue(result["models"][0]["cached"])

    def test_parses_huggingface_repo_file_reference(self) -> None:
        source = parse_llm_source(
            "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
        )

        self.assertEqual(source.repo_id, "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF")
        self.assertEqual(source.filename, "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")

    def test_parses_huggingface_repo_url_without_file(self) -> None:
        source = parse_llm_source("https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF")

        self.assertEqual(source.repo_id, "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF")
        self.assertEqual(source.filename, "")
        self.assertEqual(source.folder, "bartowski_Qwen2.5-Coder-7B-Instruct-GGUF")

    def test_parses_plain_huggingface_repo_without_file(self) -> None:
        source = parse_llm_source("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF")

        self.assertEqual(source.repo_id, "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF")
        self.assertEqual(source.filename, "")

    def test_prefers_q4_k_m_when_repo_has_many_ggufs(self) -> None:
        filename = _choose_gguf_file(
            [
                "Qwen2.5-Coder-7B-Instruct-Q2_K.gguf",
                "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
                "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            ]
        )

        self.assertEqual(filename, "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")

    def test_resolves_repo_source_by_querying_huggingface_siblings(self) -> None:
        siblings = [
            SimpleNamespace(rfilename="model-Q8_0.gguf"),
            {"rfilename": "model-Q4_K_M.gguf"},
        ]
        fake_hf = SimpleNamespace(model_info=lambda repo_id: SimpleNamespace(siblings=siblings))
        source = LlmSource(filename="", folder="repo", repo_id="org/repo")

        with patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
            resolved = _resolve_repo_source(source, proxy="http://proxy")

        ready = LlmSource(filename="ready.gguf", folder="repo")
        self.assertEqual(resolved.filename, "model-Q4_K_M.gguf")
        self.assertIs(_resolve_repo_source(ready, proxy=""), ready)


if __name__ == "__main__":
    unittest.main()
