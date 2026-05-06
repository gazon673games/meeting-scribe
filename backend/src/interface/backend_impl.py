from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from application.device_catalog import DeviceCatalog
from interface.assistant_controller import AssistantController
from interface.backend_parts import (
    BackendAsrModelsMixin,
    BackendBaseMixin,
    BackendExtraModelsMixin,
    BackendModelStateMixin,
    BackendSessionMixin,
    BackendStateMixin,
)
from interface.backend_parts.system_utils import (
    int_or_default as _int_or_default,
    int_or_zero as _int_or_zero,
    module_available as _module_available,
    nvidia_gpu_snapshot as _nvidia_gpu_snapshot,
    per_process_audio_supported as _per_process_audio_supported,
    safe_token_preview as _safe_token_preview,
    scan_compatible_asr_models as _scan_compatible_asr_models,
)
from interface.session_controller import HeadlessSessionController
from settings.application.config_repository import ConfigRepository


DeviceToken = Tuple[str, object, str]


@dataclass
class ElectronBackend(
    BackendSessionMixin,
    BackendExtraModelsMixin,
    BackendAsrModelsMixin,
    BackendModelStateMixin,
    BackendStateMixin,
    BackendBaseMixin,
):
    project_root: Path
    config_repository: ConfigRepository
    device_catalog: DeviceCatalog
    session_controller: HeadlessSessionController | None = None
    assistant_controller: AssistantController | None = None
    _device_tokens: Dict[str, DeviceToken] = field(default_factory=dict)
    _hardware_cache: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _hardware_cache_ts: float = field(default=0.0, init=False, repr=False)
    _gpu_cache: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _gpu_cache_ts: float = field(default=0.0, init=False, repr=False)
    _resource_cpu_time_s: float = field(default=0.0, init=False, repr=False)
    _resource_wall_time_s: float = field(default=0.0, init=False, repr=False)
    _resource_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _catalog_cache: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _catalog_cache_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _device_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _downloads: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _download_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _diarization_downloads: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _diarization_download_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _llm_downloads: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _llm_download_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _event_sink: Any = field(default=None, init=False, repr=False)

    protocol_version: int = 1

    def _gpu_snapshot(self, *, max_age_s: float) -> List[Dict[str, Any]]:
        with self._resource_lock:
            import time

            now = time.monotonic()
            if self._gpu_cache_ts > 0.0 and now - self._gpu_cache_ts < max(0.0, float(max_age_s)):
                return [dict(gpu) for gpu in self._gpu_cache]
            gpus = _nvidia_gpu_snapshot()
            self._gpu_cache = [dict(gpu) for gpu in gpus]
            self._gpu_cache_ts = now
            return [dict(gpu) for gpu in self._gpu_cache]
