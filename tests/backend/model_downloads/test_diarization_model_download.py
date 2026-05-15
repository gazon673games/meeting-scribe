from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from application.diarization_model_download import (
    RECOMMENDED_DIARIZATION_MODELS,
    delete_diarization_model,
    diarization_models_dir,
    list_diarization_models,
)


class DiarizationModelDownloadTests(unittest.TestCase):
    def test_recommended_model_catalog_uses_dedicated_diarization_dir(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            project_root = Path(raw_root) / "project"
            models_root = Path(raw_root) / "models"

            result = list_diarization_models(project_root=project_root, models_dir=models_root)
            model = result["models"][0]

            self.assertEqual(Path(result["modelsDir"]), models_root.resolve() / "diarization")
            self.assertEqual(model["backend"], "sherpa_onnx")
            self.assertFalse(model["cached"])
            self.assertTrue(model["downloadable"])

    def test_cached_recommended_model_is_ready_and_deletable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            project_root = Path(raw_root) / "project"
            models_root = Path(raw_root) / "models"
            spec = RECOMMENDED_DIARIZATION_MODELS[0]
            models_dir = diarization_models_dir(project_root, models_root)
            models_dir.mkdir(parents=True)
            model_path = models_dir / spec.file_name
            model_path.write_bytes(b"onnx")

            result = list_diarization_models(project_root=project_root, models_dir=models_root)
            model = next(item for item in result["models"] if item["name"] == spec.name)

            self.assertTrue(model["cached"])
            self.assertTrue(model["compatible"])
            self.assertTrue(model["deletable"])
            self.assertEqual(model["bytes"], 4)

            delete_diarization_model(project_root=project_root, models_dir=models_root, path=str(model_path))

            self.assertFalse(model_path.exists())

    def test_delete_rejects_path_outside_diarization_models_dir(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            project_root = Path(raw_root) / "project"
            models_root = Path(raw_root) / "models"
            outside_model = Path(raw_root) / "outside.onnx"
            outside_model.write_bytes(b"onnx")

            with self.assertRaises(ValueError):
                delete_diarization_model(project_root=project_root, models_dir=models_root, path=str(outside_model))


if __name__ == "__main__":
    unittest.main()
