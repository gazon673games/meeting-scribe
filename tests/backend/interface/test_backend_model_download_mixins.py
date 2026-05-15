from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from interface.backend_parts.model_state_mixin import BackendModelStateMixin
from interface.backend_parts.models_asr_mixin import BackendAsrModelsMixin
from interface.backend_parts.models_extra_mixin import BackendExtraModelsMixin


class _Backend(BackendModelStateMixin, BackendAsrModelsMixin, BackendExtraModelsMixin):
    def __init__(self, root: Path) -> None:
        self.project_root = root
        self._downloads = {}
        self._diarization_downloads = {}
        self._llm_downloads = {}
        self._download_lock = threading.RLock()
        self._diarization_download_lock = threading.RLock()
        self._llm_download_lock = threading.RLock()
        self._catalog_cache = {}
        self._catalog_cache_lock = threading.RLock()
        self.config = {"ui": {"model": "custom/current"}, "models": {}}
        self.session = {}
        self.emitted: list[tuple[str, dict]] = []

    def _read_config(self) -> dict:
        return dict(self.config)

    def _session_snapshot(self) -> dict:
        return dict(self.session)

    def _emit(self, event_type: str, payload: dict) -> None:
        self.emitted.append((event_type, dict(payload)))

    def _models_dir(self, config: dict | None = None) -> Path:
        return self.project_root / "models"

    def _models_dir_from_params(self, params: dict | None = None, config: dict | None = None) -> Path:
        if params and params.get("modelsDir"):
            return Path(params["modelsDir"])
        return self.project_root / "models"

    def _llm_models_dir_from_params(self, params: dict | None = None) -> Path:
        if params and params.get("modelsDir"):
            return Path(params["modelsDir"])
        return self.project_root / "models" / "llm"


class BackendModelDownloadMixinsTests(unittest.TestCase):
    def test_list_models_uses_cache_and_adds_local_current_and_download_records(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            backend = _Backend(Path(raw_root))
            backend._downloads["download/remote"] = {"state": "downloading", "message": "Loading"}

            with (
                patch("interface.backend_parts.models_asr_mixin.ASR_MODEL_NAMES", ["owner/recommended"]),
                patch("application.model_download.is_model_cached", side_effect=lambda name, models_dir=None: name == "owner/recommended"),
                patch("application.model_download.is_builtin_model", return_value=False),
                patch("application.model_download.scan_local_models", return_value=[{"name": "owner/local", "label": "Local"}]),
            ):
                active = backend.list_models()

            self.assertEqual(active["activeDownloads"], 1)
            self.assertIn("owner/recommended", [item["name"] for item in active["models"]])
            self.assertIn("owner/local", [item["name"] for item in active["models"]])
            self.assertIn("custom/current", [item["name"] for item in active["models"]])
            self.assertIn("download/remote", [item["name"] for item in active["models"]])

            backend._downloads.clear()
            with (
                patch("interface.backend_parts.models_asr_mixin.ASR_MODEL_NAMES", ["owner/recommended"]),
                patch("application.model_download.is_model_cached", return_value=True),
                patch("application.model_download.is_builtin_model", return_value=False),
                patch("application.model_download.scan_local_models", return_value=[]),
            ):
                first = backend.list_models()
                second = backend.list_models()

            self.assertEqual(first, second)
            self.assertTrue(backend._catalog_cache)

    def test_asr_model_download_lifecycle_updates_state_and_delegates_delete_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            backend = _Backend(Path(raw_root))
            progress_callbacks = []
            done_callbacks = []

            def fake_download(name, on_progress, on_done, **kwargs):  # noqa: ANN001
                progress_callbacks.append(on_progress)
                done_callbacks.append(on_done)
                on_progress({"message": "Half", "downloadedBytes": "25", "speedBps": "3.5"})
                on_done(None)

            with patch("application.model_download.download_model_async", side_effect=fake_download) as started:
                result = backend.download_model({"name": "owner/model", "useProxy": True, "proxy": "http://proxy"})

            self.assertTrue(result["started"])
            self.assertTrue(result["proxy"])
            started.assert_called_once()
            self.assertEqual(backend._download_record("owner/model")["state"], "done")
            self.assertEqual(backend._download_record("owner/model")["downloadedBytes"], 25)

            backend._downloads["owner/busy"] = {"state": "downloading"}
            self.assertFalse(backend.download_model({"name": "owner/busy"})["started"])
            with self.assertRaises(ValueError):
                backend.download_model({"name": ""})

            backend.config = {"ui": {"model": "owner/selected"}}
            with self.assertRaises(RuntimeError):
                backend.delete_model({"name": "owner/selected"})

            backend.config = {"ui": {"model": ""}}
            with patch("application.model_download.delete_local_model") as delete_local:
                self.assertEqual(backend.delete_model({"name": "owner/old"})["deleted"], True)
            delete_local.assert_called_once()

            with self.assertRaises(ValueError):
                backend.model_metadata({"name": ""})
            with patch("application.model_download.model_metadata", return_value={"compatible": True}):
                self.assertEqual(backend.model_metadata({"name": "owner/model"}), {"compatible": True})

    def test_diarization_and_llm_download_lifecycles_update_state_and_delegate_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            backend = _Backend(Path(raw_root))

            with patch("application.diarization_model_download.list_diarization_models", return_value={"models": []}) as list_diar:
                self.assertEqual(backend.list_diarization_models(), {"models": []})
                self.assertEqual(backend.list_diarization_models(), {"models": []})
            list_diar.assert_called_once()

            def fake_diar_download(**kwargs):  # noqa: ANN001
                kwargs["on_progress"]({"message": "Loading", "downloadedBytes": "5", "totalBytes": "10", "path": "speaker.onnx"})
                kwargs["on_done"](None)

            with patch("application.diarization_model_download.download_diarization_model_async", side_effect=fake_diar_download):
                result = backend.download_diarization_model({"name": "speaker", "use_proxy": True, "proxy": "http://proxy"})
            self.assertTrue(result["started"])
            self.assertEqual(backend._diarization_download_record("speaker")["state"], "done")

            backend._diarization_downloads["busy"] = {"state": "downloading"}
            self.assertFalse(backend.download_diarization_model({"name": "busy"})["started"])
            with self.assertRaises(ValueError):
                backend.download_diarization_model({"name": ""})

            with patch("application.diarization_model_download.delete_diarization_model") as delete_diar:
                self.assertTrue(backend.delete_diarization_model({"path": str(Path(raw_root) / "speaker.onnx")})["deleted"])
            delete_diar.assert_called_once()

            with patch("application.llm_model_download.list_llm_models", return_value={"models": []}) as list_llm:
                self.assertEqual(backend.list_llm_models(), {"models": []})
                self.assertEqual(backend.list_llm_models(), {"models": []})
            list_llm.assert_called_once()

            def fake_llm_download(**kwargs):  # noqa: ANN001
                kwargs["on_progress"]({"message": "Loading", "downloadedBytes": "7", "totalBytes": "9", "path": "model.gguf"})
                kwargs["on_done"]("failed")

            with patch("application.llm_model_download.download_llm_model_async", side_effect=fake_llm_download):
                result = backend.download_llm_model({"name": "owner/repo/model.gguf"})
            self.assertTrue(result["started"])
            self.assertEqual(backend._llm_download_record("model.gguf")["state"], "error")
            self.assertEqual(backend._llm_download_record("model.gguf")["error"], "failed")

            backend._llm_downloads["busy.gguf"] = {"state": "downloading"}
            self.assertFalse(backend.download_llm_model({"name": "owner/repo/busy.gguf"})["started"])
            with self.assertRaises(ValueError):
                backend.download_llm_model({"name": ""})
            with self.assertRaises(ValueError):
                backend.delete_llm_model({"path": ""})

            with patch("application.llm_model_download.delete_llm_model") as delete_llm:
                self.assertTrue(backend.delete_llm_model({"path": str(Path(raw_root) / "model.gguf")})["deleted"])
            delete_llm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
