from __future__ import annotations

# ruff: noqa: F401
# The public backend facade also exposes these legacy helper aliases.
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from application.device_catalog import DeviceCatalog
from interface.assistant_controller import AssistantController
from interface.backend_parts import base_mixin as _base_ops
from interface.backend_parts import model_state_mixin as _model_state_ops
from interface.backend_parts import models_asr_mixin as _models_asr_ops
from interface.backend_parts import models_extra_mixin as _models_extra_ops
from interface.backend_parts import session_mixin as _session_ops
from interface.backend_parts import state_mixin as _state_ops
from interface.backend_parts.records import DownloadState
from interface.backend_parts.system_utils import (
    int_or_default as _int_or_default,
)
from interface.backend_parts.system_utils import (
    int_or_zero as _int_or_zero,
)
from interface.backend_parts.system_utils import (
    module_available as _module_available,
)
from interface.backend_parts.system_utils import (
    nvidia_gpu_snapshot as _nvidia_gpu_snapshot,
)
from interface.backend_parts.system_utils import (
    per_process_audio_supported as _per_process_audio_supported,
)
from interface.backend_parts.system_utils import (
    safe_token_preview as _safe_token_preview,
)
from interface.backend_parts.system_utils import (
    scan_compatible_asr_models as _scan_compatible_asr_models,
)
from interface.session_controller import HeadlessSessionController
from interface.session_controller_parts.types import EventSink
from settings.application.config_repository import ConfigRepository

DeviceToken = tuple[str, object, str]


@dataclass
class ElectronBackend:
    project_root: Path
    config_repository: ConfigRepository
    device_catalog: DeviceCatalog
    session_controller: HeadlessSessionController | None = None
    assistant_controller: AssistantController | None = None
    _device_tokens: dict[str, DeviceToken] = field(default_factory=dict)
    _hardware_cache: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _hardware_cache_ts: float = field(default=0.0, init=False, repr=False)
    _gpu_cache: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _gpu_cache_ts: float = field(default=0.0, init=False, repr=False)
    _resource_cpu_time_s: float = field(default=0.0, init=False, repr=False)
    _resource_wall_time_s: float = field(default=0.0, init=False, repr=False)
    _resource_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _catalog_cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _catalog_cache_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _device_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _downloads: dict[str, DownloadState] = field(
        default_factory=dict, init=False, repr=False
    )
    _download_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _diarization_downloads: dict[str, DownloadState] = field(
        default_factory=dict, init=False, repr=False
    )
    _diarization_download_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _llm_downloads: dict[str, DownloadState] = field(
        default_factory=dict, init=False, repr=False
    )
    _llm_download_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _event_sink: EventSink | None = field(default=None, init=False, repr=False)
    _shutdown: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )

    def request_shutdown(self) -> None:
        self._shutdown.set()

    protocol_version: int = 1

    _read_config = _base_ops._read_config
    _emit = _base_ops._emit
    _models_dir = _base_ops._models_dir
    _models_dir_from_params = _base_ops._models_dir_from_params
    _llm_models_dir_from_params = _base_ops._llm_models_dir_from_params
    _require_session_controller = _base_ops._require_session_controller
    _download_snapshot = _model_state_ops._download_snapshot
    _diarization_download_snapshot = (
        _model_state_ops._diarization_download_snapshot
    )
    _llm_download_snapshot = _model_state_ops._llm_download_snapshot
    _download_record = _model_state_ops._download_record
    _diarization_download_record = _model_state_ops._diarization_download_record
    _llm_download_record = _model_state_ops._llm_download_record
    _download_fields = _model_state_ops._download_fields
    _catalog_cache_get = _model_state_ops._catalog_cache_get
    _catalog_cache_put = _model_state_ops._catalog_cache_put
    _catalog_cache_clear = _model_state_ops._catalog_cache_clear
    _set_download_state = _model_state_ops._set_download_state
    _set_diarization_download_state = (
        _model_state_ops._set_diarization_download_state
    )
    _set_llm_download_state = _model_state_ops._set_llm_download_state
    _active_download_count = _model_state_ops._active_download_count
    _model_download_proxy = _model_state_ops._model_download_proxy
    _model_is_selected_or_running = (
        _model_state_ops._model_is_selected_or_running
    )
    _diarization_model_is_selected_or_running = (
        _model_state_ops._diarization_model_is_selected_or_running
    )
    list_models = _models_asr_ops.list_models
    _build_model_list = _models_asr_ops._build_model_list
    _local_model_records = _models_asr_ops._local_model_records
    _add_unlisted_models = _models_asr_ops._add_unlisted_models
    _add_if_missing = _models_asr_ops._add_if_missing
    _recommended_model_record = _models_asr_ops._recommended_model_record
    _custom_model_record = _models_asr_ops._custom_model_record
    download_model = _models_asr_ops.download_model
    delete_model = _models_asr_ops.delete_model
    model_metadata = _models_asr_ops.model_metadata
    _asr_model_options = _models_asr_ops._asr_model_options
    list_diarization_models = _models_extra_ops.list_diarization_models
    download_diarization_model = _models_extra_ops.download_diarization_model
    delete_diarization_model = _models_extra_ops.delete_diarization_model
    list_llm_models = _models_extra_ops.list_llm_models
    download_llm_model = _models_extra_ops.download_llm_model
    delete_llm_model = _models_extra_ops.delete_llm_model
    list_process_sessions = _session_ops.list_process_sessions
    save_config = _session_ops.save_config
    list_devices = _session_ops.list_devices
    add_source = _session_ops.add_source
    remove_source = _session_ops.remove_source
    set_source_enabled = _session_ops.set_source_enabled
    set_source_delay = _session_ops.set_source_delay
    start_session = _session_ops.start_session
    _download_then_start = _session_ops._download_then_start
    stop_session = _session_ops.stop_session
    clear_transcript = _session_ops.clear_transcript
    invoke_assistant = _session_ops.invoke_assistant
    start_assistant_login = _session_ops.start_assistant_login
    ping_assistant_provider = _session_ops.ping_assistant_provider
    start_local_llm = _session_ops.start_local_llm
    stop_local_llm = _session_ops.stop_local_llm
    handle = _session_ops.handle
    _start_params_from_config = _session_ops._start_params_from_config
    _resolve_diarization_start_params = (
        _session_ops._resolve_diarization_start_params
    )
    _read_device_group = _session_ops._read_device_group
    _clear_device_tokens_for_kinds = _session_ops._clear_device_tokens_for_kinds
    _NO_PARAMS = _session_ops.NO_PARAMS_METHODS
    _METHODS = _session_ops.SESSION_METHODS
    set_event_sink = _state_ops.set_event_sink
    ping = _state_ops.ping
    get_state = _state_ops.get_state
    get_runtime_state = _state_ops.get_runtime_state
    get_config = _state_ops.get_config
    get_resource_usage = _state_ops.get_resource_usage
    _hardware_snapshot = _state_ops._hardware_snapshot
    _session_snapshot = _state_ops._session_snapshot
    _assistant_snapshot = _state_ops._assistant_snapshot

    def _gpu_snapshot(self, *, max_age_s: float) -> list[dict[str, Any]]:
        with self._resource_lock:
            import time

            now = time.monotonic()
            if self._gpu_cache_ts > 0.0 and now - self._gpu_cache_ts < max(
                0.0, float(max_age_s)
            ):
                return [dict(gpu) for gpu in self._gpu_cache]
            gpus = _nvidia_gpu_snapshot()
            self._gpu_cache = [dict(gpu) for gpu in gpus]
            self._gpu_cache_ts = now
            return [dict(gpu) for gpu in self._gpu_cache]
