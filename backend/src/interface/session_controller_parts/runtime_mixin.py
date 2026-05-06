from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict

from application.asr_language import runtime_asr_language
from application.local_paths import project_recordings_dir
from application.session_tasks import OfflinePassRequest, StopAsrRequest
from interface.session_controller_parts.artifacts import generated_session_id, write_session_outputs
from interface.session_controller_parts.helpers import drops_record, line_speaker_or_stream, master_record, wav_requested
from interface.session_controller_parts.settings import asr_settings_from_params


class RuntimeLifecycleMixin:
    def begin_model_download(self, model_name: str) -> None:
        with self._lock:
            self._session_state.begin_model_download(model_name)
            self._model_download_info = {
                "model": model_name,
                "downloadedBytes": 0,
                "speedBps": 0.0,
                "message": "Starting download...",
            }
            self._emit("session_state_changed", {"state": "downloading_model", "model": model_name})

    def update_model_download_progress(self, info: Dict[str, Any]) -> None:
        with self._lock:
            self._model_download_info = {
                "model": self._model_download_info.get("model", ""),
                "downloadedBytes": int(info.get("downloadedBytes", 0)),
                "speedBps": float(info.get("speedBps", 0.0)),
                "message": str(info.get("message", "")),
            }

    def finish_model_download(self, error: str = "") -> None:
        with self._lock:
            model_name = self._model_download_info.get("model", "")
            self._session_state.finish_model_download(model_name=model_name, error=error)
            self._model_download_info = {}
            if error:
                self._last_error = error
                self._emit("session_error", {"message": error})
            self._emit("session_state_changed", {"state": "idle", "error": error})

    def start_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if not self._session_state.can_start:
                raise RuntimeError(f"Session is busy: {self._session_state.state.value}")
            if not self._sources:
                raise RuntimeError("Add at least one source before starting a session")

            asr_enabled = bool(params.get("asrEnabled", params.get("asr_enabled", False)))
            self._asr_requested = bool(asr_enabled)
            self._last_warning = ""
            self._last_error = ""
            self._session_state.begin_start(
                source_count=len(self._sources),
                asr_enabled=asr_enabled,
                profile=str(params.get("profile", "")),
                language=str(params.get("language", "")),
            )

            for source in self._sources.values():
                self.engine.set_source_enabled(source.name, source.enabled)
                self.engine.set_source_delay_ms(source.name, source.delay_ms)
            if asr_enabled:
                self._drain_tap_queue()
            self._apply_tap_config()
            requested_wav = wav_requested(params)
            self._set_engine_output_enabled(requested_wav)

            try:
                self.engine.start()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._session_state.fail_start(self._last_error)
                self._status = "start failed"
                self._emit("session_error", {"message": self._last_error})
                raise

            wav_error = self._start_wav_if_requested(params)
            if wav_error:
                self._set_engine_output_enabled(False)
            asr_error = ""
            if asr_enabled:
                asr_error = self._start_asr_locked(params)
            self._configure_transcript_files_locked(params)

            self._session_state.finish_start(asr_running=self._asr_running, wav_recording=self.writer.is_recording())
            self._status = "running"
            payload = self.snapshot()
            warnings = [warning for warning in (wav_error, asr_error, self._last_warning) if warning]
            if warnings:
                payload["warning"] = " ".join(warnings)
            self._emit("session_started", payload)
            return payload

    def stop_session(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with self._lock:
            stop_params = dict(params or {})
            if self._session_state.is_stopping:
                raise RuntimeError("Session is already stopping")
            if not self._session_state.can_stop:
                if not self.engine.is_running() and not self.writer.is_recording():
                    return self.snapshot()
                raise RuntimeError(f"Cannot stop while session is {self._session_state.state.value}")

            wav_path = self.writer.target_path() or self._current_output_path(str(stop_params.get("outputFile", "")))
            stop_params.setdefault("wavPath", str(wav_path))

            self._session_state.begin_stop(run_offline_pass=bool(stop_params.get("runOfflinePass", False)))
            if self.writer.is_recording():
                self.writer.stop_recording()
            self._set_engine_output_enabled(False)
            self._stop_asr_locked(stop_params)
            try:
                if self.engine.is_running():
                    self.engine.stop()
            finally:
                try:
                    self.engine.set_tap_queue(None)
                except Exception:
                    pass
                self._drain_tap_queue()

            self._asr_requested = False
            self._asr_running = False
            self._close_transcript_files_locked()
            self._session_state.finish_stop("")
            self._status = "idle"
            if self._should_run_offline_pass(stop_params, Path(wav_path)):
                self._start_offline_pass(Path(wav_path), stop_params)
            payload = self.snapshot()
            self._emit("session_stopped", payload)
            return payload

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            meters = self._safe_meters()
            return {
                "status": self._status,
                "running": bool(self._session_state.is_running),
                "state": self._session_state.state.value,
                "asrRunning": bool(self._asr_running),
                "asrRequested": bool(self._asr_requested),
                "wavRecording": bool(self.writer.is_recording()),
                "wavPath": str(self.writer.target_path() or ""),
                "lastError": self._last_error,
                "lastWarning": self._last_warning,
                "master": master_record(meters),
                "drops": drops_record(meters, self.writer),
                "asrMetrics": dict(self._asr_metrics),
                "offlinePass": {
                    "running": bool(self._offline_pass_running),
                    "result": dict(self._offline_pass_result),
                },
                "humanLogPath": str(self._human_log_path or ""),
                "realtimeTranscriptPath": str(self._realtime_transcript_path or ""),
                "transcript": list(self._transcript_lines[-200:]),
                "sources": [self._source_record(name, meters=meters) for name in self._sources],
                "modelDownload": dict(self._model_download_info),
            }

    def transcript_text(self, *, limit: int = 500) -> str:
        with self._lock:
            lines = list(self._transcript_lines[-int(limit):])
        return "\n".join(
            f"[{time.strftime('%H:%M:%S', time.localtime(float(line.get('ts', time.time()))))}] "
            f"{line_speaker_or_stream(line)}: {line.get('text', '')}"
            for line in lines
        )

    def shutdown(self) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
        try:
            if self._session_state.can_stop:
                self.stop_session({"runOfflinePass": False})
        finally:
            try:
                self.writer.stop()
            except Exception:
                pass
            with self._lock:
                self._close_transcript_files_locked()

    def _start_wav_if_requested(self, params: Dict[str, Any]) -> str:
        if not wav_requested(params):
            return ""
        if not self.wav_recorder_factory.available():
            return "WAV recording is unavailable because soundfile is not installed."
        output_file = str(params.get("outputFile", params.get("output_file", "")) or "").strip()
        output_path = self._current_output_path(output_file)
        try:
            self.writer.start_recording(output_path, self.format)
        except Exception as exc:
            return f"WAV recording could not start: {type(exc).__name__}: {exc}"
        return ""

    def _set_engine_output_enabled(self, enabled: bool) -> None:
        set_output_enabled = getattr(self.engine, "set_output_enabled", None)
        if callable(set_output_enabled):
            try:
                set_output_enabled(bool(enabled))
            except Exception:
                pass

    def _write_session_outputs_locked(
        self,
        *,
        asr: Any,
        settings,
        session_id: str,
    ) -> None:
        write_session_outputs(
            asr=asr,
            settings=settings,
            session_id=session_id,
            project_root=self.project_root,
            transcript_lines=self._transcript_lines,
        )

    def _start_asr_locked(self, params: Dict[str, Any]) -> str:
        if self.asr_runtime_factory is None or self.transcription_startup_service is None:
            self._last_warning = "Live ASR is not configured for the Electron backend yet."
            return self._last_warning

        self._drain_asr_events()
        settings = asr_settings_from_params(params, source_speaker_labels=self._source_speaker_labels_locked())
        self._active_asr_settings = settings
        self._transcription_state.begin_start(
            model_name=settings.model_name,
            mode=str(settings.mode),
            language=settings.language,
        )
        result = self.transcription_startup_service.start(
            settings,
            runtime_factory=self.asr_runtime_factory,
            tap_queue=self.tap_queue,
            project_root=self.project_root,
            event_queue=self.asr_event_queue,
        )

        for error in result.errors:
            self._emit("asr_start_attempt_failed", {"message": error})

        if not result.ok or result.asr is None or result.attempt is None:
            message = "; ".join(result.errors) or "unknown ASR startup error"
            self._transcription_state.fail_start(message)
            self._asr_running = False
            self._asr_runtime = None
            self._active_asr_session_id = ""
            self._last_warning = f"Live ASR could not start: {message}"
            return self._last_warning

        self._asr_runtime = result.asr
        self._asr_runtime.start()
        self._asr_running = True
        self._active_asr_session_id = str(getattr(result.asr, "session_id", "") or generated_session_id())
        self._transcription_state.finish_start(degraded=result.attempt.degraded, attempt_label=result.attempt.label)
        self._start_asr_event_pump()
        if result.degraded:
            self._last_warning = f"Live ASR started in fallback mode: {result.attempt.label}."
            return self._last_warning
        self._last_warning = ""
        return ""

    def _resolve_stop_wav_path(self, params: Dict[str, Any]) -> Path:
        return Path(str(
            params.get("wavPath")
            or self.writer.target_path()
            or self._current_output_path(str(params.get("outputFile", "")))
        ))

    def _build_stop_asr_request(self, asr: Any, wav_path: Path, params: Dict[str, Any]) -> StopAsrRequest:
        p = params or {}
        return StopAsrRequest(
            asr=asr,
            wav_path=wav_path,
            run_offline_pass=bool(p.get("runOfflinePass", False)),
            offline_model_name=str(p.get("offlineModelName", p.get("model", "large-v3"))),
            offline_language=runtime_asr_language(str(p.get("language", "ru"))),
        )

    def _stop_asr_locked(self, params: Dict[str, Any]) -> None:
        asr = self._asr_runtime
        self._asr_runtime = None
        self._asr_running = False
        self._asr_event_stop.set()
        self._transcript_lines = [l for l in self._transcript_lines if not str(l.get("id", "")).startswith("streaming-")]
        if asr is None:
            self._transcription_state.reset()
            self._active_asr_settings = None
            self._active_asr_session_id = ""
            return

        if self._transcription_state.can_stop:
            self._transcription_state.begin_stop()
        wav_path = self._resolve_stop_wav_path(params)
        result = self.stop_asr_use_case.execute(self._build_stop_asr_request(asr, wav_path, params))
        self._drain_asr_events()
        self._write_session_outputs_locked(
            asr=asr,
            settings=self._active_asr_settings,
            session_id=self._active_asr_session_id,
        )
        stop_error = str(result.stop_error or "")
        if self._transcription_state.is_stopping:
            self._transcription_state.finish_stop(stop_error)
        else:
            self._transcription_state.reset()
        self._active_asr_settings = None
        self._active_asr_session_id = ""
        if stop_error:
            self._last_error = stop_error
            self._emit("asr_stop_error", {"message": stop_error})

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

    def _start_asr_event_pump(self) -> None:
        self._asr_event_stop = threading.Event()
        self._asr_event_thread = threading.Thread(target=self._run_asr_event_pump, name="electron-asr-events", daemon=True)
        self._asr_event_thread.start()

    def _run_asr_event_pump(self) -> None:
        while not self._asr_event_stop.is_set():
            try:
                item = self.asr_event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_asr_event(item)

    def _drain_asr_events(self, *, limit: int = 500) -> int:
        count = 0
        while count < limit:
            try:
                item = self.asr_event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_asr_event(item)
            count += 1
        return count

    def _drain_tap_queue(self) -> int:
        count = 0
        while True:
            try:
                self.tap_queue.get_nowait()
                count += 1
            except queue.Empty:
                return count

    def _current_output_path(self, raw_name: str) -> Path:
        name = Path(raw_name or "capture_mix.wav").name
        if not name.lower().endswith(".wav"):
            name += ".wav"
        return project_recordings_dir(self.project_root, create=True) / name

    def _apply_tap_config(self) -> None:
        if not self._asr_requested:
            self.engine.set_tap_queue(None)
            return
        self.engine.set_tap_queue(self.tap_queue)
        self.engine.set_tap_config(
            mode="both",
            sources=[source.name for source in self._sources.values() if source.enabled],
            drop_threshold=0.85,
        )
