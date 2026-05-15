from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from application import diarization_model_download as diar_models
from application import llm_model_download as llm_models
from application.model_download_parts import download as asr_download


class _Response:
    def __init__(self, chunks: list[bytes], total: int = 0) -> None:
        self.chunks = list(chunks)
        self.headers = {"Content-Length": str(total)}

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def read(self, size: int) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""


class _SyncThread:
    def __init__(self, *, target, args=(), kwargs=None, name="", **thread_kwargs) -> None:  # noqa: ANN001
        self.target = target
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.name = str(name)

    def start(self) -> None:
        if self.name == "model-download-progress":
            return
        self.target(*self.args, **self.kwargs)

    def join(self, timeout=None) -> None:  # noqa: ANN001
        return None


class ModelDownloadWorkflowTests(unittest.TestCase):
    def test_llm_download_helpers_parse_urls_track_downloads_delete_files_and_scope_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            models_root = root / "models" / "llm"
            model_dir = models_root / "demo"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "demo.gguf"
            model_path.write_bytes(b"gguf")

            direct = llm_models.parse_llm_source("https://example.test/models/demo.Q4.gguf")
            self.assertEqual(direct.filename, "demo.Q4.gguf")
            self.assertEqual(direct.folder, "demo.Q4")
            with self.assertRaises(ValueError):
                llm_models.parse_llm_source("https://example.test/models/demo.bin")
            with self.assertRaises(ValueError):
                llm_models.parse_llm_source("owner/repo/readme.txt")
            with self.assertRaises(ValueError):
                llm_models._choose_gguf_file(["README.md"])

            listed = llm_models.list_llm_models(
                project_root=root,
                downloads={"remote.gguf": {"state": "downloading", "downloadedBytes": 3, "path": "remote.gguf"}},
            )
            self.assertEqual([item["name"] for item in listed["models"]], ["demo", "remote.gguf"])

            target = root / "downloaded" / "model.gguf"
            updates: list[dict] = []
            with patch("application.llm_model_download.urlopen", return_value=_Response([b"ab", b"cd"], total=4)):
                llm_models._download_url("https://example.test/model.gguf", target, proxy="", on_progress=updates.append)
            self.assertEqual(target.read_bytes(), b"abcd")
            self.assertEqual(updates[-1]["downloadedBytes"], 4)

            old_http = os.environ.get("HTTP_PROXY")
            try:
                with llm_models._temporary_proxy_env("http://proxy"):
                    self.assertEqual(os.environ["HTTP_PROXY"], "http://proxy")
            finally:
                if old_http is None:
                    os.environ.pop("HTTP_PROXY", None)
                else:
                    os.environ["HTTP_PROXY"] = old_http

            llm_models.delete_llm_model(project_root=root, path=str(model_path))
            self.assertFalse(model_path.exists())
            self.assertFalse(model_dir.exists())
            with self.assertRaises(ValueError):
                llm_models.delete_llm_model(project_root=root, path=str(root / "outside.gguf"))

    def test_llm_async_download_runs_successfully_with_sync_thread_and_fake_network(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            progress: list[dict] = []
            done: list[str | None] = []

            with (
                patch("application.llm_model_download.threading.Thread", _SyncThread),
                patch("application.llm_model_download._download_url") as download_url,
            ):
                download_url.side_effect = lambda url, target, **kwargs: target.write_bytes(b"gguf")
                llm_models.download_llm_model_async(
                    name="https://example.test/models/local.gguf",
                    project_root=root,
                    on_progress=progress.append,
                    on_done=done.append,
                )

            self.assertEqual(done, [None])
            self.assertTrue((root / "models" / "llm" / "local" / "local.gguf").exists())
            self.assertEqual(progress[-1]["message"], "Downloaded")

    def test_diarization_download_lists_cached_model_downloads_file_and_scopes_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            spec = diar_models.RECOMMENDED_DIARIZATION_MODELS[0]
            model_path = root / "models" / "diarization" / spec.file_name

            self.assertIsNone(diar_models.default_cached_diarization_model(project_root=root))
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"onnx")
            cached = diar_models.default_cached_diarization_model(project_root=root)
            assert cached is not None
            self.assertTrue(cached["cached"])
            self.assertEqual(diar_models._model_spec(spec.file_name), spec)
            self.assertEqual(diar_models._model_spec(spec.url), spec)
            with self.assertRaises(ValueError):
                diar_models._model_spec("unknown")

            target = root / "download" / "speaker.onnx"
            updates: list[dict] = []
            with patch("application.diarization_model_download.urlopen", return_value=_Response([b"on", b"nx"], total=4)):
                diar_models._download_file(url=spec.url, target=target, proxy="", on_progress=updates.append)
            self.assertEqual(target.read_bytes(), b"onnx")
            self.assertEqual(updates[-1]["downloadedBytes"], 4)

            progress: list[dict] = []
            done: list[str | None] = []

            def fake_download_file(**kwargs):  # noqa: ANN001
                kwargs["target"].write_bytes(b"onnx")
                kwargs["on_progress"]({"message": "Downloading...", "downloadedBytes": 4})

            with (
                patch("application.diarization_model_download.threading.Thread", _SyncThread),
                patch("application.diarization_model_download._download_file", side_effect=fake_download_file),
            ):
                diar_models.download_diarization_model_async(
                    name=spec.name,
                    project_root=root,
                    on_progress=progress.append,
                    on_done=done.append,
                )

            self.assertEqual(done, [None])
            self.assertEqual(progress[-1]["message"], "Downloaded")

            old_http = os.environ.get("HTTP_PROXY")
            try:
                with diar_models._temporary_proxy_env("http://proxy"):
                    self.assertEqual(os.environ["HTTP_PROXY"], "http://proxy")
            finally:
                if old_http is None:
                    os.environ.pop("HTTP_PROXY", None)
                else:
                    os.environ["HTTP_PROXY"] = old_http

    def test_asr_model_download_reports_friendly_errors_progress_and_success(self) -> None:
        self.assertIn("Private", asr_download.friendly_download_error(RuntimeError("401 gated")))
        self.assertIn("ValueError", asr_download.friendly_download_error(ValueError("bad repo_id")))
        self.assertIn("not found", asr_download.friendly_download_error(RuntimeError("Repository not found")))
        self.assertIn("RuntimeError", asr_download.friendly_download_error(RuntimeError("boom")))

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repo_dir = root / "models--owner--repo"
            repo_dir.mkdir()
            (repo_dir / "file.bin").write_bytes(b"1234")
            progress: list[dict] = []
            stop = SimpleNamespace(wait=lambda timeout: len(progress) >= 1)

            asr_download.monitor_download_progress(repo_dir, stop, progress.append, "owner/repo")
            self.assertEqual(progress[0]["downloadedBytes"], 4)

            done: list[str | None] = []
            async_progress: list[dict] = []

            def fake_snapshot_download(repo_id, cache_dir, local_files_only):  # noqa: ANN001
                model_root = Path(cache_dir) / "models--owner--repo"
                model_root.mkdir(parents=True, exist_ok=True)
                (model_root / "config.json").write_text("{}", encoding="utf-8")

            with (
                patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(snapshot_download=fake_snapshot_download)}),
                patch("application.model_download_parts.download.threading.Thread", _SyncThread),
                patch("application.model_download_parts.download.inspect_model_path", return_value={"compatible": True}),
            ):
                asr_download.download_model_async(
                    "owner/repo",
                    async_progress.append,
                    done.append,
                    models_dir=root / "cache",
                )

            self.assertEqual(done, [None])
            self.assertEqual(async_progress[-1]["message"], "Downloaded")

            errors: list[str | None] = []
            with patch("application.model_download_parts.download.threading.Thread", _SyncThread):
                asr_download.download_model_async("", async_progress.append, errors.append, models_dir=root / "cache")
            self.assertIn("Model name is empty", errors[0])


if __name__ == "__main__":
    unittest.main()
