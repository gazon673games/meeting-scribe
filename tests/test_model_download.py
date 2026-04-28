from __future__ import annotations

import shutil
import unittest
import os
from pathlib import Path

from application.model_download import (
    delete_local_model,
    inspect_model_path,
    is_model_cached,
    is_valid_huggingface_repo_id,
    model_metadata,
    normalize_model_reference,
    scan_local_models,
    _temporary_proxy_env,
)


class ModelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_parent = Path(__file__).resolve().parents[1] / "tmp_tests"
        cls._tmp_parent.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp_parent, ignore_errors=True)

    def _models_root(self, name: str) -> Path:
        root = self._tmp_parent / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        return root

    def test_huggingface_cache_folder_is_discovered_and_deletable(self) -> None:
        root = self._models_root("hf_cache")
        snapshot = root / "models--paulpengtw--faster-whisper-Breeze-ASR-26" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.bin").write_bytes(b"0" * 1_000_001)

        models = scan_local_models(root)

        self.assertEqual(models[0]["name"], "paulpengtw/faster-whisper-Breeze-ASR-26")
        self.assertTrue(models[0]["compatible"])
        self.assertTrue(is_model_cached("paulpengtw/faster-whisper-Breeze-ASR-26", models_dir=root))

        delete_local_model("paulpengtw/faster-whisper-Breeze-ASR-26", models_dir=root)

        self.assertFalse((root / "models--paulpengtw--faster-whisper-Breeze-ASR-26").exists())

    def test_huggingface_hub_subfolder_is_discovered_and_deletable(self) -> None:
        root = self._models_root("hf_home")
        snapshot = root / "hub" / "models--owner--repo" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.bin").write_bytes(b"0" * 1_000_001)

        models = scan_local_models(root)

        self.assertEqual(models[0]["name"], "owner/repo")
        self.assertTrue(is_model_cached("owner/repo", models_dir=root))

        delete_local_model("owner/repo", models_dir=root)

        self.assertFalse((root / "hub" / "models--owner--repo").exists())

    def test_cache_folder_with_double_dash_in_repo_name_round_trips(self) -> None:
        root = self._models_root("hf_double_dash")
        snapshot = root / "models--owner--repo--name" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.bin").write_bytes(b"0" * 1_000_001)

        models = scan_local_models(root)

        self.assertEqual(models[0]["name"], "owner/repo--name")
        self.assertTrue(is_model_cached("owner/repo--name", models_dir=root))

    def test_huggingface_url_is_normalized_to_repo_id(self) -> None:
        self.assertEqual(
            normalize_model_reference("https://huggingface.co/paulpengtw/faster-whisper-Breeze-ASR-26/tree/main"),
            "paulpengtw/faster-whisper-Breeze-ASR-26",
        )

    def test_model_metadata_reads_local_config_summary(self) -> None:
        root = self._models_root("metadata")
        snapshot = root / "models--owner--demo" / "snapshots" / "abc"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            '{"model_type":"whisper","d_model":384,"encoder_layers":4}',
            encoding="utf-8",
        )
        (snapshot / "preprocessor_config.json").write_text('{"sampling_rate":16000}', encoding="utf-8")
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.bin").write_bytes(b"0" * 1_000_001)

        metadata = model_metadata("owner/demo", models_dir=root)

        self.assertTrue(metadata["compatible"])
        self.assertEqual(metadata["config"]["model_type"], "whisper")
        self.assertEqual(metadata["preprocessor"]["sampling_rate"], 16000)
        self.assertIn("model.bin", metadata["presentFiles"])

    def test_transformers_safetensors_are_reported_as_unsupported_for_faster_whisper(self) -> None:
        root = self._models_root("transformers")
        model = root / "models--owner--transformers" / "snapshots" / "abc"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}", encoding="utf-8")
        (model / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model / "model-00001-of-00002.safetensors").write_bytes(b"0" * 1_000_001)

        info = inspect_model_path(root / "models--owner--transformers")

        self.assertFalse(info["compatible"])
        self.assertEqual(info["status"], "unsupported_transformers_format")
        self.assertIn("model.bin", info["missing"])

    def test_invalid_repo_id_rejects_windows_like_missing_path(self) -> None:
        self.assertFalse(is_valid_huggingface_repo_id("C:/missing/model"))
        self.assertFalse(is_valid_huggingface_repo_id("/missing/model"))
        self.assertTrue(is_valid_huggingface_repo_id("owner/model"))

    def test_non_folder_models_path_raises_clear_error(self) -> None:
        root = self._models_root("not_folder")
        file_path = root / "models.txt"
        file_path.write_text("not a folder", encoding="utf-8")

        with self.assertRaises(ValueError):
            scan_local_models(file_path)

    def test_download_proxy_env_is_scoped(self) -> None:
        old_http = os.environ.get("HTTP_PROXY")
        try:
            os.environ["HTTP_PROXY"] = "http://old.proxy:1"
            with _temporary_proxy_env("socks5://127.0.0.1:10808"):
                self.assertEqual(os.environ["HTTP_PROXY"], "socks5://127.0.0.1:10808")
                self.assertEqual(os.environ["HTTPS_PROXY"], "socks5://127.0.0.1:10808")
                self.assertEqual(os.environ["ALL_PROXY"], "socks5://127.0.0.1:10808")
            self.assertEqual(os.environ.get("HTTP_PROXY"), "http://old.proxy:1")
        finally:
            if old_http is None:
                os.environ.pop("HTTP_PROXY", None)
            else:
                os.environ["HTTP_PROXY"] = old_http


if __name__ == "__main__":
    unittest.main()
