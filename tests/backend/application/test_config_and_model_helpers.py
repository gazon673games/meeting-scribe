from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from application import diarization_model_download, llm_model_download
from application.codex_config import (
    CodexProfile,
    CodexSettings,
    codex_command_config_value,
    codex_profile_to_dict,
    codex_settings_to_dict,
)
from application.local_paths import (
    application_root,
    configure_project_local_io,
    project_identity_dir,
    project_sessions_dir,
)
from application.model_download_parts.catalog import inaccessible_model_record
from application.model_download_parts.inspection import parse_readme_frontmatter
from application.model_download_parts.references import is_builtin_model
from application.model_policy import ModelOrchestrator


class ConfigAndModelHelperTests(unittest.TestCase):
    def test_codex_profiles_settings_and_command_values_serialize_for_config(self) -> None:
        profile = CodexProfile(
            id="deep",
            label="Deep",
            prompt="answer",
            provider_id="ollama",
            model="llama",
            extra_args=["--flag", ""],
            temperature=None,
        )
        settings = CodexSettings(enabled=True, command_tokens=["codex", "exec"], profiles=[profile])

        self.assertEqual(codex_command_config_value([]), "codex")
        self.assertEqual(codex_command_config_value(["codex", "exec"]), ["codex", "exec"])
        self.assertEqual(codex_profile_to_dict(profile)["provider"], "ollama")
        serialized = codex_settings_to_dict(settings)
        self.assertEqual(serialized["command"], ["codex", "exec"])
        self.assertEqual(serialized["profiles"][0]["model"], "llama")

    def test_project_local_paths_create_local_dirs_and_cache_environment(self) -> None:
        keys = ("HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "NEMO_CACHE_DIR", "XDG_CACHE_HOME", "TMP", "TEMP", "TMPDIR")
        old_env = {key: os.environ.get(key) for key in keys}
        old_tempdir = tempfile.tempdir
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                models_dir = root / "custom_models"

                configure_project_local_io(root, models_dir=models_dir)

                self.assertTrue(project_sessions_dir(root, create=True).is_dir())
                self.assertTrue(project_identity_dir(root, create=True).is_dir())
                self.assertTrue(Path(os.environ["HF_HOME"]).is_dir())
                self.assertEqual(tempfile.tempdir, str(root / ".local" / "tmp"))
                self.assertTrue(application_root().exists())
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            tempfile.tempdir = old_tempdir

    def test_model_records_frontmatter_builtin_policy_and_private_download_helpers(self) -> None:
        record = inaccessible_model_record(Path("broken-model"), OSError("denied"))
        self.assertEqual(record["status"], "inaccessible")
        self.assertIn("OSError", record["warnings"][0])

        metadata = parse_readme_frontmatter(["---", "license: mit", "ignored: value", "---"])
        self.assertEqual(metadata["license"], "mit")
        self.assertTrue(is_builtin_model("large-v3"))

        with tempfile.TemporaryDirectory() as tmp:
            local_path = Path(tmp) / "speaker.onnx"
            local_path.write_bytes(b"data")
            diar_record = diarization_model_download._local_record(local_path)
        self.assertEqual(diar_record["source"], "local-file")

        with patch.dict(os.environ, {"HF_TOKEN": "secret"}, clear=False):
            self.assertEqual(llm_model_download._hf_headers()["Authorization"], "Bearer secret")

        fake_hf = SimpleNamespace(hf_hub_url=lambda repo_id, filename: f"https://hf/{repo_id}/{filename}")
        source = llm_model_download.LlmSource(filename="model.gguf", folder="repo", repo_id="org/repo")
        with (
            patch.dict(sys.modules, {"huggingface_hub": fake_hf}),
            patch.object(llm_model_download, "_download_url") as download_url,
        ):
            llm_model_download._download_hf_file(source, Path("model.gguf"), proxy="http://proxy", on_progress=lambda _: None)
        download_url.assert_called_once()

    def test_model_orchestrator_recommend_returns_combined_asr_and_codex_decision(self) -> None:
        profiles = [
            SimpleNamespace(id="fast", label="Fast", reasoning_effort="low"),
            SimpleNamespace(id="deep", label="Deep", reasoning_effort="high"),
        ]

        decision = ModelOrchestrator().recommend(
            asr_profile="quality",
            language="ru",
            current_asr_model="small",
            available_asr_models=["large-v3", "bzikst/faster-whisper-large-v3-russian"],
            codex_profiles=profiles,
            current_codex_profile_id="fast",
        )

        self.assertEqual(decision.asr_model, "bzikst/faster-whisper-large-v3-russian")
        self.assertEqual(decision.codex_profile_id, "deep")


if __name__ == "__main__":
    unittest.main()
