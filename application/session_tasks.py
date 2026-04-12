from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from application.asr_session import ASRRuntime
from application.offline_pass import OfflineAsrRequest, OfflineAsrRunnerPort


@dataclass(frozen=True)
class StopAsrRequest:
    asr: ASRRuntime
    wav_path: Path
    run_offline_pass: bool
    offline_model_name: str
    offline_language: Optional[str]


@dataclass(frozen=True)
class StopAsrResult:
    wav_path: Path
    run_offline_pass: bool
    offline_model_name: str
    offline_language: Optional[str]
    stop_error: Optional[str]


class StopAsrSessionUseCase:
    def execute(self, request: StopAsrRequest) -> StopAsrResult:
        stop_error: Optional[str] = None
        try:
            request.asr.stop()
        except Exception as e:
            stop_error = f"{type(e).__name__}: {e}"

        return StopAsrResult(
            wav_path=Path(request.wav_path),
            run_offline_pass=bool(request.run_offline_pass),
            offline_model_name=str(request.offline_model_name),
            offline_language=request.offline_language,
            stop_error=stop_error,
        )


@dataclass(frozen=True)
class OfflinePassRequest:
    project_root: Path
    wav_path: Path
    model_name: str
    language: Optional[str]


@dataclass(frozen=True)
class OfflinePassResult:
    out_txt: Path


class OfflinePassUseCase:
    def __init__(self, offline_runner: OfflineAsrRunnerPort) -> None:
        self._offline_runner = offline_runner

    def available(self) -> bool:
        return bool(self._offline_runner.available())

    def execute(self, request: OfflinePassRequest) -> OfflinePassResult:
        logs_dir = Path(request.project_root) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_txt = logs_dir / f"offline_transcript_{ts}.txt"

        result_path = self._offline_runner.run(
            OfflineAsrRequest(
                project_root=Path(request.project_root),
                wav_path=Path(request.wav_path),
                out_txt=out_txt,
                model_name=str(request.model_name or "large-v3"),
                language=request.language,
            )
        )
        return OfflinePassResult(out_txt=Path(result_path))
