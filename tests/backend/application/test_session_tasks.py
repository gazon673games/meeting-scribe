from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from application.session_tasks import OfflinePassRequest, OfflinePassUseCase


class _OfflineRunner:
    def __init__(self) -> None:
        self.requests = []

    def available(self) -> bool:
        return True

    def run(self, request):  # noqa: ANN001
        self.requests.append(request)
        return request.out_txt


class SessionTaskTests(unittest.TestCase):
    def test_offline_pass_use_case_creates_log_output_and_delegates_to_runner(self) -> None:
        runner = _OfflineRunner()

        with tempfile.TemporaryDirectory() as tmp:
            result = OfflinePassUseCase(runner).execute(
                OfflinePassRequest(
                    project_root=Path(tmp),
                    wav_path=Path(tmp) / "session.wav",
                    model_name="",
                    language=None,
                )
            )

        self.assertTrue(OfflinePassUseCase(runner).available())
        self.assertTrue(result.out_txt.name.startswith("offline_transcript_"))
        self.assertEqual(runner.requests[0].model_name, "large-v3")


if __name__ == "__main__":
    unittest.main()
