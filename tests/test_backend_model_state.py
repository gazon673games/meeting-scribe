from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from interface.backend_parts.model_state_mixin import BackendModelStateMixin


class _StateBackend(BackendModelStateMixin):
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
        self.config = {}
        self.session = {}
        self.emitted: list[tuple[str, dict]] = []

    def _read_config(self) -> dict:
        return dict(self.config)

    def _session_snapshot(self) -> dict:
        return dict(self.session)

    def _emit(self, event_type: str, payload: dict) -> None:
        self.emitted.append((event_type, dict(payload)))


class BackendModelStateTests(unittest.TestCase):
    def test_download_snapshot_is_copied_and_fields_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            backend = _StateBackend(Path(raw_root))
            backend._downloads["owner/model"] = {
                "state": "downloading",
                "downloadedBytes": "12",
                "speedBps": "2.5",
                "proxy": True,
            }

            snapshot = backend._download_snapshot()
            snapshot["owner/model"]["state"] = "done"

            self.assertEqual(backend._download_record("owner/model")["state"], "downloading")
            self.assertTrue(backend._download_fields("owner/model", backend._downloads)["downloading"])
            self.assertEqual(backend._download_fields("owner/model", backend._downloads)["downloadedBytes"], 12)
            self.assertEqual(backend._active_download_count(), 1)

    def test_catalog_cache_returns_deep_copies_and_expires(self) -> None:
        backend = _StateBackend(Path("."))

        with patch("interface.backend_parts.model_state_mixin.time.monotonic", return_value=10.0):
            backend._catalog_cache_put(("models",), {"items": [{"name": "a"}]})

        with patch("interface.backend_parts.model_state_mixin.time.monotonic", return_value=11.0):
            cached = backend._catalog_cache_get(("models",), max_age_s=10.0)
        assert cached is not None
        cached["items"][0]["name"] = "changed"
        with patch("interface.backend_parts.model_state_mixin.time.monotonic", return_value=11.0):
            self.assertEqual(backend._catalog_cache_get(("models",))["items"][0]["name"], "a")  # type: ignore[index]

        with patch("interface.backend_parts.model_state_mixin.time.monotonic", return_value=25.0):
            self.assertIsNone(backend._catalog_cache_get(("models",), max_age_s=10.0))

    def test_download_state_updates_emit_events_and_invalidate_catalog_cache(self) -> None:
        backend = _StateBackend(Path("."))
        backend._catalog_cache_put(("old",), {"value": 1})

        backend._set_download_state("asr", {"state": "downloading", "downloadedBytes": 5})
        backend._set_diarization_download_state("diar", {"state": "done"})
        backend._set_llm_download_state("llm", {"state": "error", "error": "bad"})

        self.assertEqual(backend._catalog_cache, {})
        self.assertEqual([event for event, _ in backend.emitted], [
            "model_download_updated",
            "diarization_model_download_updated",
            "llm_model_download_updated",
        ])
        self.assertEqual(backend.emitted[0][1]["activeDownloads"], 1)
        self.assertEqual(backend.emitted[2][1]["error"], "bad")

    def test_proxy_and_selected_model_checks_use_config_and_session(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            diar_model = root / "models" / "diarization" / "speaker.onnx"
            diar_model.parent.mkdir(parents=True)
            diar_model.write_bytes(b"onnx")
            backend = _StateBackend(root)

            backend.config = {
                "models": {"use_proxy": True, "proxy": ""},
                "codex": {"proxy": "http://codex.proxy"},
                "ui": {"model": "owner/model"},
                "asr": {"diar_sherpa_embedding_model_path": str(diar_model)},
            }
            backend.session = {"running": True}

            self.assertEqual(backend._model_download_proxy(), "http://codex.proxy")
            self.assertTrue(backend._model_is_selected_or_running("owner/model"))
            self.assertFalse(backend._model_is_selected_or_running("owner/other"))
            self.assertTrue(backend._diarization_model_is_selected_or_running(str(diar_model)))
            self.assertFalse(backend._diarization_model_is_selected_or_running(str(root / "other.onnx")))


if __name__ == "__main__":
    unittest.main()
