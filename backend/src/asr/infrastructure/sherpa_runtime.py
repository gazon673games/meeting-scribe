from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)


def sherpa_cpu_threads(requested: int) -> int:
    return int(requested) if requested > 0 else min(4, os.cpu_count() or 1)


def report_cpu_execution(model_name: str, requested_device: str) -> None:
    if requested_device != 'cpu':
        _logger.warning(
            '%s uses the CPU INT8 bundle; requested device %s is unavailable '
            'for this execution profile',
            model_name,
            requested_device,
        )
