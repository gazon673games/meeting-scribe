from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from asr.offline_runner import OfflineProfile, OfflineRunner as OfflineASRRunner  # type: ignore
except Exception:
    OfflineProfile = None  # type: ignore
    OfflineASRRunner = None  # type: ignore


def offline_asr_available() -> bool:
    return OfflineASRRunner is not None and OfflineProfile is not None


def run_offline_asr_pass(
    *,
    project_root: Path,
    wav_path: Path,
    out_txt: Path,
    model_name: str,
    language: Optional[str],
) -> Path:
    if not offline_asr_available():
        raise RuntimeError("Offline ASR runner is unavailable.")

    runner = OfflineASRRunner(project_root=Path(project_root))
    profile = OfflineProfile(
        model_name=str(model_name or "large-v3"),
        language=language,
    )
    return Path(
        runner.run(
            wav_path=Path(wav_path),
            out_txt=Path(out_txt),
            profile=profile,
        )
    )
