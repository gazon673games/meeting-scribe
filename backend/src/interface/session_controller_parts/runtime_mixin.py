from __future__ import annotations

from interface.session_controller_parts.runtime_asr_mixin import RuntimeAsrMixin
from interface.session_controller_parts.runtime_download_mixin import RuntimeDownloadMixin
from interface.session_controller_parts.runtime_offline_pass_mixin import RuntimeOfflinePassMixin
from interface.session_controller_parts.runtime_session_mixin import RuntimeSessionMixin


class RuntimeLifecycleMixin(
    RuntimeDownloadMixin,
    RuntimeSessionMixin,
    RuntimeAsrMixin,
    RuntimeOfflinePassMixin,
):
    """Composed runtime lifecycle facade.

    HeadlessSessionController inherits this class to keep method names stable while
    the implementation stays split by bounded responsibilities.
    """

