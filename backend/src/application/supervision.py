from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class SupervisionStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class SupervisionReport:
    component: str
    status: SupervisionStatus
    active_attempt: str
    errors: tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        return self.status == SupervisionStatus.DEGRADED

    @property
    def failed(self) -> bool:
        return self.status == SupervisionStatus.FAILED


def supervision_report(
    *,
    component: str,
    active_attempt: str,
    fallback_used: bool,
    errors: Sequence[str] = (),
    failed: bool = False,
) -> SupervisionReport:
    if failed:
        status = SupervisionStatus.FAILED
    elif fallback_used:
        status = SupervisionStatus.DEGRADED
    else:
        status = SupervisionStatus.HEALTHY
    return SupervisionReport(
        component=str(component),
        status=status,
        active_attempt=str(active_attempt),
        errors=tuple(str(error) for error in errors if str(error).strip()),
    )
