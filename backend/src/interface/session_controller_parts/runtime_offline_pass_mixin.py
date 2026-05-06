from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict

from application.asr_language import runtime_asr_language
from application.session_tasks import OfflinePassRequest


class RuntimeOfflinePassMixin:
    def _should_run_offline_pass(self, params: Dict[str, Any], wav_path: Path) -> bool:
        return bool(
            params.get("runOfflinePass", False)
            and self.offline_pass_use_case is not None
            and self.offline_pass_use_case.available()
            and Path(wav_path).exists()
        )

    def _start_offline_pass(self, wav_path: Path, params: Dict[str, Any]) -> None:
        if self.offline_pass_use_case is None:
            return
        if self._offline_pass_running:
            return
        self._offline_pass_running = True
        self._offline_pass_result = {"wavPath": str(wav_path), "status": "running", "ts": time.time()}
        try:
            self._session_state.begin_offline_pass(str(wav_path))
        except Exception:
            pass
        self._emit("offline_pass_started", dict(self._offline_pass_result))
        self._offline_pass_thread = threading.Thread(
            target=self._run_offline_pass,
            args=(Path(wav_path), dict(params)),
            name="electron-offline-pass",
            daemon=True,
        )
        self._offline_pass_thread.start()

    def _run_offline_pass(self, wav_path: Path, params: Dict[str, Any]) -> None:
        assert self.offline_pass_use_case is not None
        error = ""
        out_txt = ""
        try:
            result = self.offline_pass_use_case.execute(
                OfflinePassRequest(
                    project_root=self.project_root,
                    wav_path=Path(wav_path),
                    model_name=str(params.get("offlineModelName", params.get("model", "large-v3")) or "large-v3"),
                    language=runtime_asr_language(str(params.get("language", "ru"))),
                )
            )
            out_txt = str(result.out_txt)
            payload = {"status": "done", "wavPath": str(wav_path), "outTxt": out_txt, "ts": time.time()}
            self._emit("offline_pass_done", payload)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            payload = {"status": "error", "wavPath": str(wav_path), "error": error, "ts": time.time()}
            self._emit("offline_pass_error", payload)
        with self._lock:
            self._offline_pass_running = False
            self._offline_pass_result = {
                "wavPath": str(wav_path),
                "status": "error" if error else "done",
                "error": error,
                "outTxt": out_txt,
                "ts": time.time(),
            }
            try:
                self._session_state.finish_offline_pass(error)
            except Exception:
                pass
            if error:
                self._last_error = error
