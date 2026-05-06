from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from application.local_paths import project_recordings_dir
from interface.session_controller_parts.helpers import drops_record, line_speaker_or_stream, master_record, wav_requested


class RuntimeSessionMixin:
    def start_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            asr_enabled = self._prepare_start_locked(params)
            wav_error, asr_error = self._start_runtime_features_locked(params, asr_enabled)
            self._configure_transcript_files_locked(params)
            return self._finalize_start_locked(wav_error=wav_error, asr_error=asr_error)

    def stop_session(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with self._lock:
            stop_params = self._prepare_stop_params_locked(params)
            wav_path = Path(str(stop_params.get("wavPath", "")))
            self._stop_runtime_locked(stop_params)
            return self._finalize_stop_locked(stop_params, wav_path)

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

    def _prepare_start_locked(self, params: Dict[str, Any]) -> bool:
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
        self._set_engine_output_enabled(wav_requested(params))

        try:
            self.engine.start()
            return asr_enabled
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._session_state.fail_start(self._last_error)
            self._status = "start failed"
            self._emit("session_error", {"message": self._last_error})
            raise

    def _start_runtime_features_locked(self, params: Dict[str, Any], asr_enabled: bool) -> tuple[str, str]:
        wav_error = self._start_wav_if_requested(params)
        if wav_error:
            self._set_engine_output_enabled(False)
        asr_error = self._start_asr_locked(params) if asr_enabled else ""
        return wav_error, asr_error

    def _finalize_start_locked(self, *, wav_error: str, asr_error: str) -> Dict[str, Any]:
        self._session_state.finish_start(asr_running=self._asr_running, wav_recording=self.writer.is_recording())
        self._status = "running"
        payload = self.snapshot()
        warnings = [warning for warning in (wav_error, asr_error, self._last_warning) if warning]
        if warnings:
            payload["warning"] = " ".join(warnings)
        self._emit("session_started", payload)
        return payload

    def _prepare_stop_params_locked(self, params: Dict[str, Any] | None) -> Dict[str, Any]:
        stop_params = dict(params or {})
        if self._session_state.is_stopping:
            raise RuntimeError("Session is already stopping")
        if not self._session_state.can_stop:
            if not self.engine.is_running() and not self.writer.is_recording():
                return {"_snapshotOnly": True}
            raise RuntimeError(f"Cannot stop while session is {self._session_state.state.value}")

        wav_path = self.writer.target_path() or self._current_output_path(str(stop_params.get("outputFile", "")))
        stop_params.setdefault("wavPath", str(wav_path))
        self._session_state.begin_stop(run_offline_pass=bool(stop_params.get("runOfflinePass", False)))
        return stop_params

    def _stop_runtime_locked(self, stop_params: Dict[str, Any]) -> None:
        if stop_params.get("_snapshotOnly"):
            return
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

    def _finalize_stop_locked(self, stop_params: Dict[str, Any], wav_path: Path) -> Dict[str, Any]:
        if stop_params.get("_snapshotOnly"):
            return self.snapshot()

        self._asr_requested = False
        self._asr_running = False
        self._close_transcript_files_locked()
        self._session_state.finish_stop("")
        self._status = "idle"
        if self._should_run_offline_pass(stop_params, wav_path):
            self._start_offline_pass(wav_path, stop_params)
        payload = self.snapshot()
        self._emit("session_stopped", payload)
        return payload

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
