from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from assistant.application.provider import AssistantExecutionSettings, AssistantProviderRequest
from infrastructure import llama_cpp_runner
from infrastructure.llama_cpp_runner import LlamaCppRunner


class _SyncThread:
    def __init__(self, *, target, args=(), kwargs=None, **thread_kwargs) -> None:  # noqa: ANN001
        self.target = target
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})

    def start(self) -> None:
        self.target(*self.args, **self.kwargs)


class _FakeLlama:
    def __init__(self, text: str = "answer") -> None:
        self.text = text
        self.closed = False
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append(kwargs)
        if self.text == "raise":
            raise RuntimeError("boom")
        return {"choices": [{"message": {"content": self.text}}]}

    def close(self) -> None:
        self.closed = True


class LlamaCppRunnerTests(unittest.TestCase):
    def tearDown(self) -> None:
        llama_cpp_runner._LOADED.clear()
        llama_cpp_runner._CANCELLED.clear()

    def _profile(self, **kwargs):  # noqa: ANN003, ANN201
        values = {
            "id": "local",
            "label": "Local",
            "model": "model.gguf",
            "temperature": "0.2",
            "max_tokens": 64,
            "gpu_layers": 2,
            "context_size": 2048,
        }
        values.update(kwargs)
        return SimpleNamespace(**values)

    def _settings(self, profile) -> AssistantExecutionSettings:  # noqa: ANN001
        return AssistantExecutionSettings(command_tokens=[], path_hints=[], proxy="", timeout_s=10, profile=profile)

    def test_runner_reports_unloaded_state_and_runs_loaded_model(self) -> None:
        runner = LlamaCppRunner()
        profile = self._profile()

        self.assertFalse(runner.status(self._settings(profile)).available)
        self.assertEqual(runner.ping(self._settings(profile)).error_code, "model_not_loaded")

        request = AssistantProviderRequest(
            prompt="hello",
            profile=profile,
            original_cmd="ASK",
            project_root=Path("."),
            settings=self._settings(profile),
        )
        self.assertEqual(runner.run(request).error_code, "model_not_loaded")

        fake = _FakeLlama(" answer ")
        llama_cpp_runner._LOADED["local"] = fake
        self.assertTrue(runner.status(self._settings(profile)).available)
        self.assertTrue(runner.ping(self._settings(profile)).ok)

        result = runner.run(request)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "answer")
        self.assertEqual(fake.calls[0]["temperature"], 0.2)
        self.assertEqual(fake.calls[0]["max_tokens"], 64)

        llama_cpp_runner._LOADED["local"] = _FakeLlama("raise")
        self.assertEqual(runner.run(request).error_code, "inference_error")

    def test_model_path_resolution_and_optional_float_parsing_handle_edges(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            direct = root / "direct.gguf"
            direct.write_bytes(b"gguf")
            nested = root / "models" / "llm" / "repo" / "named.gguf"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"gguf")

            self.assertEqual(llama_cpp_runner._resolve_model_path(str(direct), root), direct.resolve())
            self.assertEqual(llama_cpp_runner._resolve_model_path("named", root), nested.resolve())
            self.assertEqual(llama_cpp_runner._search_models_dir(root / "models" / "llm", "named.gguf"), nested.resolve())
            with self.assertRaises(FileNotFoundError):
                llama_cpp_runner._resolve_model_path("missing", root)

        self.assertEqual(llama_cpp_runner._optional_float(" 0.5 "), 0.5)
        self.assertIsNone(llama_cpp_runner._optional_float(""))
        self.assertIsNone(llama_cpp_runner._optional_float("bad"))

    def test_load_model_builds_llama_and_reports_import_or_profile_errors(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            model = root / "model.gguf"
            model.write_bytes(b"gguf")
            created: list[dict] = []

            class FakeLlamaCtor:
                def __init__(self, **kwargs) -> None:  # noqa: ANN003
                    created.append(kwargs)

            with patch.dict("sys.modules", {"llama_cpp": SimpleNamespace(Llama=FakeLlamaCtor)}):
                loaded = llama_cpp_runner._load_model(self._profile(model=str(model)), root)

            self.assertIsInstance(loaded, FakeLlamaCtor)
            self.assertEqual(created[0]["model_path"], str(model.resolve()))
            self.assertEqual(created[0]["n_gpu_layers"], 2)
            self.assertEqual(created[0]["n_ctx"], 2048)

            with patch.dict("sys.modules", {"llama_cpp": None}):
                with self.assertRaises(RuntimeError):
                    llama_cpp_runner._load_model(self._profile(model=str(model)), root)

            with patch.dict("sys.modules", {"llama_cpp": SimpleNamespace(Llama=FakeLlamaCtor)}):
                with self.assertRaises(ValueError):
                    llama_cpp_runner._load_model(self._profile(model=""), root)

    def test_async_load_unload_reports_ready_existing_cancelled_and_failed_states(self) -> None:
        profile = self._profile()
        events: list[dict] = []

        with (
            patch("infrastructure.llama_cpp_runner.threading.Thread", _SyncThread),
            patch("infrastructure.llama_cpp_runner._load_model", return_value=_FakeLlama("ok")),
        ):
            result = llama_cpp_runner.load_model_async(profile, Path("."), events.append)

        self.assertTrue(result["started"])
        self.assertEqual(events[-1]["state"], "running")
        self.assertIn("local", llama_cpp_runner._LOADED)

        events.clear()
        with patch("infrastructure.llama_cpp_runner.threading.Thread", _SyncThread):
            llama_cpp_runner.load_model_async(profile, Path("."), events.append)
        self.assertEqual(events[-1]["state"], "running")
        self.assertIn("already loaded", events[-1]["message"])

        loaded = llama_cpp_runner._LOADED["local"]
        stopped = llama_cpp_runner.unload_model(profile)
        self.assertTrue(stopped["stopped"])
        self.assertTrue(loaded.closed)
        self.assertIn("local", llama_cpp_runner._CANCELLED)

        events.clear()
        cancelled_model = _FakeLlama("ok")

        def fake_load(profile_arg, root_arg):  # noqa: ANN001
            llama_cpp_runner._CANCELLED.add("local")
            return cancelled_model

        with (
            patch("infrastructure.llama_cpp_runner.threading.Thread", _SyncThread),
            patch("infrastructure.llama_cpp_runner._load_model", side_effect=fake_load),
        ):
            llama_cpp_runner.load_model_async(profile, Path("."), events.append)
        self.assertTrue(cancelled_model.closed)
        self.assertEqual(events[-1]["state"], "stopped")

        events.clear()
        with (
            patch("infrastructure.llama_cpp_runner.threading.Thread", _SyncThread),
            patch("infrastructure.llama_cpp_runner._load_model", side_effect=RuntimeError("nope")),
        ):
            llama_cpp_runner.load_model_async(profile, Path("."), events.append)
        self.assertEqual(events[-1]["state"], "error")


if __name__ == "__main__":
    unittest.main()
