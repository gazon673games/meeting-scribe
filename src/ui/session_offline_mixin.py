from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from application.event_types import OfflinePassDoneEvent, OfflinePassErrorEvent, OfflinePassStartedEvent
from application.session_tasks import OfflinePassRequest


class SessionOfflineMixin:
    def _start_offline_pass(self, wav_path: Path, *, model_name: str, language: Optional[str]) -> None:
        if self._session_state.is_offline_pass:
            self._set_status("Offline pass is already running.")
            return
        if not self._offline_asr_available():
            self._set_status("Offline pass is unavailable.")
            return
        if not self._session_state.can_start:
            self._set_status(f"Cannot start offline pass while state is {self._session_state.state.value}.")
            return

        self._session_state.begin_offline_pass(str(wav_path))
        self.btn_start.setEnabled(False)
        self._offline_thread = self.background_task_runner.start(
            name="offline-asr-pass",
            target=self._run_offline_pass_worker,
            args=(Path(wav_path), str(model_name), language),
        )

    def _run_offline_pass_worker(self, wav_path: Path, model_name: str, language: Optional[str]) -> None:
        try:
            self.background_event.emit(OfflinePassStartedEvent())
            result = self.offline_pass_use_case.execute(
                OfflinePassRequest(
                    project_root=self.project_root,
                    wav_path=Path(wav_path),
                    model_name=str(model_name or "large-v3"),
                    language=language,
                )
            )
            self.background_event.emit(OfflinePassDoneEvent(out_txt=str(result.out_txt)))
        except Exception as e:
            self.background_event.emit(OfflinePassErrorEvent(error=f"{type(e).__name__}: {e}"))

    def _handle_offline_pass_started(self) -> None:
        self._append_transcript_line(f"[{self._fmt_ts(time.time())}] offline pass: starting...")
        self._set_status("offline pass: running")

    def _handle_offline_pass_done_event(self, event: OfflinePassDoneEvent) -> None:
        if self._session_state.is_offline_pass:
            self._session_state.finish_offline_pass(out_txt=str(event.out_txt))
        self._offline_thread = None
        self.btn_start.setEnabled(True)
        out_txt = str(event.out_txt).strip()
        self._append_transcript_line(f"[{self._fmt_ts(time.time())}] offline pass: done -> {out_txt}")
        self._set_status("offline pass: done")

    def _handle_offline_pass_error_event(self, event: OfflinePassErrorEvent) -> None:
        if self._session_state.is_offline_pass:
            self._session_state.finish_offline_pass(error=str(event.error or "unknown error"))
        self._offline_thread = None
        self.btn_start.setEnabled(True)
        err = str(event.error or "unknown error")
        self._append_transcript_line(f"[{self._fmt_ts(time.time())}] offline pass: ERROR {err}")
        self._set_status("offline pass: failed")
