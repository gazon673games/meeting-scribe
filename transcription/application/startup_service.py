from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from application.asr_session import ASRRuntime, ASRRuntimeFactory, ASRSessionSettings
from application.asr_supervisor import ASRStartupSupervisor, ASRStartupAttempt
from application.supervision import SupervisionReport


@dataclass(frozen=True)
class TranscriptionStartupResult:
    asr: Optional[ASRRuntime]
    attempt: Optional[ASRStartupAttempt]
    supervision_report: SupervisionReport
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.asr is not None and self.attempt is not None

    @property
    def degraded(self) -> bool:
        return bool(self.attempt and self.attempt.degraded)


class TranscriptionStartupService:
    def __init__(self, *, supervisor: Optional[ASRStartupSupervisor] = None) -> None:
        self._supervisor = supervisor or ASRStartupSupervisor()

    def start(
        self,
        settings: ASRSessionSettings,
        *,
        runtime_factory: ASRRuntimeFactory,
        tap_queue: Any,
        project_root: Path,
        event_queue: Any = None,
    ) -> TranscriptionStartupResult:
        attempts = self._supervisor.build_attempts(settings)
        errors: List[str] = []

        for attempt in attempts:
            asr = None
            try:
                asr = runtime_factory.build(
                    attempt.settings,
                    tap_queue=tap_queue,
                    project_root=Path(project_root),
                    event_queue=event_queue,
                )
                asr.start()
                return TranscriptionStartupResult(
                    asr=asr,
                    attempt=attempt,
                    supervision_report=self._supervisor.success_report(attempt, errors),
                    errors=list(errors),
                )
            except Exception as exc:
                errors.append(f"{attempt.label}: {type(exc).__name__}: {exc}")
                try:
                    if asr is not None:
                        asr.stop()
                except Exception:
                    pass

        return TranscriptionStartupResult(
            asr=None,
            attempt=None,
            supervision_report=self._supervisor.failure_report(errors),
            errors=list(errors),
        )
