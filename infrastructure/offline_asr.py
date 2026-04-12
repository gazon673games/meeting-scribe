from __future__ import annotations

from pathlib import Path

from application.offline_pass import OfflineAsrRequest, OfflineAsrRunnerPort

try:
    from asr.infrastructure.offline_runner import OfflineProfile, OfflineRunner  # type: ignore
except Exception:
    OfflineProfile = None  # type: ignore
    OfflineRunner = None  # type: ignore


class FasterWhisperOfflineAsrRunner(OfflineAsrRunnerPort):
    def available(self) -> bool:
        return OfflineRunner is not None and OfflineProfile is not None

    def run(self, request: OfflineAsrRequest) -> Path:
        if not self.available():
            raise RuntimeError("Offline ASR runner is unavailable.")

        runner = OfflineRunner(project_root=Path(request.project_root))
        profile = OfflineProfile(
            model_name=str(request.model_name or "large-v3"),
            language=request.language,
        )
        return Path(
            runner.run(
                wav_path=Path(request.wav_path),
                out_txt=Path(request.out_txt),
                profile=profile,
            )
        )
