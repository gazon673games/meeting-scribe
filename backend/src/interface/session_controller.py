from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from application.asr_language import initial_prompt_for_language, normalize_asr_language, runtime_asr_language
from application.asr_session import ASRRuntime, ASRRuntimeFactory, ASRSessionSettings
from application.audio_runtime import AudioRuntimeFactory, AudioRuntimePort
from application.audio_sources import AudioSourceFactory
from application.events import (
    AsrMetricsEvent,
    SourceErrorEvent,
    TranscriptSpeakerUpdateEvent,
    UtteranceEvent,
    event_from_record,
    event_to_record,
)
from application.local_paths import project_recordings_dir
from application.recording import WavRecorder, WavRecorderFactory
from application.session_tasks import OfflinePassRequest, OfflinePassUseCase, StopAsrRequest, StopAsrSessionUseCase
from audio.domain.formats import AudioFormat
from session.domain.aggregate import SessionAggregate
from session.domain.speaker_labels import default_speaker_label_for_source_kind
from transcription.application.startup_service import TranscriptionStartupService
from transcription.application.transcript_store import TranscriptStore
from transcription.domain.transcript_lines import (
    best_line_for_speaker_update,
    build_transcript_line_id,
    update_line_speaker,
)
from transcription.domain.aggregate import TranscriptionJobAggregate


EventSink = Callable[[str, Dict[str, Any]], None]
_MAX_PENDING_SPEAKER_UPDATES = 200


@dataclass
class HeadlessSource:
    name: str
    kind: str
    label: str
    enabled: bool = True
    delay_ms: float = 0.0


class HeadlessSessionController:
    def __init__(
        self,
        *,
        project_root: Path,
        audio_runtime_factory: AudioRuntimeFactory,
        audio_source_factory: AudioSourceFactory,
        wav_recorder_factory: WavRecorderFactory,
        asr_runtime_factory: Optional[ASRRuntimeFactory] = None,
        transcription_startup_service: Optional[TranscriptionStartupService] = None,
        stop_asr_use_case: Optional[StopAsrSessionUseCase] = None,
        offline_pass_use_case: Optional[OfflinePassUseCase] = None,
        transcript_store: Optional[TranscriptStore] = None,
        event_sink: Optional[EventSink] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.audio_runtime_factory = audio_runtime_factory
        self.audio_source_factory = audio_source_factory
        self.wav_recorder_factory = wav_recorder_factory
        self.asr_runtime_factory = asr_runtime_factory
        self.transcription_startup_service = transcription_startup_service
        self.stop_asr_use_case = stop_asr_use_case or StopAsrSessionUseCase()
        self.offline_pass_use_case = offline_pass_use_case
        self.transcript_store = transcript_store
        self.event_sink = event_sink

        self.format = AudioFormat(sample_rate=48000, channels=2, dtype="float32", blocksize=1024)
        self.output_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
        self.tap_queue: "queue.Queue[dict]" = queue.Queue(maxsize=200)
        self.asr_event_queue: "queue.Queue[object]" = queue.Queue(maxsize=600)
        self.engine: AudioRuntimePort = self.audio_runtime_factory.create(
            format=self.format,
            output_queue=self.output_queue,
            tap_queue=None,
        )
        self.writer: WavRecorder = self.wav_recorder_factory.create(self.output_queue)
        self.writer.start()

        self._lock = threading.RLock()
        self._session_state = SessionAggregate()
        self._transcription_state = TranscriptionJobAggregate()
        self._sources: dict[str, HeadlessSource] = {}
        self._source_objs: dict[str, Any] = {}
        self._status = "idle"
        self._last_error = ""
        self._last_warning = ""
        self._asr_requested = False
        self._asr_running = False
        self._asr_runtime: Optional[ASRRuntime] = None
        self._asr_event_stop = threading.Event()
        self._asr_event_thread: Optional[threading.Thread] = None
        self._transcript_lines: list[Dict[str, Any]] = []
        self._pending_speaker_updates: list[TranscriptSpeakerUpdateEvent] = []
        self._realtime_transcript_enabled = False
        self._human_log_path: Optional[Path] = None
        self._realtime_transcript_path: Optional[Path] = None
        self._asr_metrics: Dict[str, Any] = {
            "segDroppedTotal": 0,
            "segSkippedTotal": 0,
            "avgLatencyS": 0.0,
            "p95LatencyS": 0.0,
            "lagS": 0.0,
        }
        self._offline_pass_thread: Optional[threading.Thread] = None
        self._offline_pass_running = False
        self._offline_pass_result: Dict[str, Any] = {}
        self._closing = False

    def set_event_sink(self, event_sink: Optional[EventSink]) -> None:
        with self._lock:
            self.event_sink = event_sink

    def add_source(self, *, kind: str, token: object, label: str = "", name: str = "") -> Dict[str, Any]:
        with self._lock:
            normalized_kind = _normalize_source_kind(kind)
            source_name = self._make_unique_name(name or _default_source_name(normalized_kind))

            if normalized_kind == "loopback":
                source = self.audio_source_factory.create_loopback_source(
                    name=source_name,
                    engine_format=self.format,
                    device=token,
                    error_callback=self._on_source_error,
                )
            elif normalized_kind == "process":
                source = self.audio_source_factory.create_process_source(
                    name=source_name,
                    token=token,
                    error_callback=self._on_source_error,
                )
            else:
                source = self.audio_source_factory.create_microphone_source(name=source_name, device=token)

            self.engine.add_source(source)
            self._source_objs[source_name] = source
            self._sources[source_name] = HeadlessSource(
                name=source_name,
                kind=normalized_kind,
                label=str(label or source_name),
            )
            self._status = f"added {source_name}"
            if self._session_state.is_running:
                self.engine.set_source_enabled(source_name, True)
                self.engine.set_source_delay_ms(source_name, 0.0)
                if self._asr_requested:
                    self._apply_tap_config()
            self._emit("source_added", {"source": self._source_record(source_name)})
            return self._source_record(source_name)

    def remove_source(self, *, name: str) -> Dict[str, Any]:
        with self._lock:
            self._ensure_not_running("remove a source")
            record = self._source_record(name)
            self.engine.remove_source(record["name"])
            self._source_objs.pop(record["name"], None)
            self._sources.pop(record["name"], None)
            self._status = f"removed {record['name']}"
            self._emit("source_removed", {"source": record})
            return record

    def set_source_enabled(self, *, name: str, enabled: bool) -> Dict[str, Any]:
        with self._lock:
            source = self._require_source(name)
            source.enabled = bool(enabled)
            self.engine.set_source_enabled(source.name, source.enabled)
            if self._session_state.is_running and self._asr_requested:
                self._apply_tap_config()
            self._emit("source_updated", {"source": self._source_record(source.name)})
            return self._source_record(source.name)

    def set_source_delay(self, *, name: str, delay_ms: object) -> Dict[str, Any]:
        with self._lock:
            source = self._require_source(name)
            value = _safe_float(delay_ms, 0.0)
            if value < 0.0:
                value = 0.0
            source.delay_ms = value
            self.engine.set_source_delay_ms(source.name, value)
            self._emit("source_updated", {"source": self._source_record(source.name)})
            return self._source_record(source.name)

    def clear_transcript(self) -> Dict[str, Any]:
        with self._lock:
            self._transcript_lines.clear()
            self._pending_speaker_updates.clear()
            self._emit("transcript_cleared", {})
            return self.snapshot()

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

            try:
                self.engine.start()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._session_state.fail_start(self._last_error)
                self._status = "start failed"
                self._emit("session_error", {"message": self._last_error})
                raise

            wav_error = self._start_wav_if_requested(params)
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
                "master": _master_record(meters),
                "drops": _drops_record(meters, self.writer),
                "asrMetrics": dict(self._asr_metrics),
                "offlinePass": {
                    "running": bool(self._offline_pass_running),
                    "result": dict(self._offline_pass_result),
                },
                "humanLogPath": str(self._human_log_path or ""),
                "realtimeTranscriptPath": str(self._realtime_transcript_path or ""),
                "transcript": list(self._transcript_lines[-200:]),
                "sources": [self._source_record(name, meters=meters) for name in self._sources],
            }

    def transcript_text(self, *, limit: int = 500) -> str:
        with self._lock:
            lines = list(self._transcript_lines[-int(limit):])
        return "\n".join(
            f"[{time.strftime('%H:%M:%S', time.localtime(float(line.get('ts', time.time()))))}] "
            f"{_line_speaker_or_stream(line)}: {line.get('text', '')}"
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
        wav_enabled = bool(params.get("wavEnabled", params.get("wav_enabled", False)))
        if not wav_enabled:
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

    def _configure_transcript_files_locked(self, params: Dict[str, Any]) -> None:
        self._realtime_transcript_enabled = bool(
            params.get(
                "realtimeTranscriptToFile",
                params.get("rtTranscriptToFile", params.get("rt_transcript_to_file", False)),
            )
        )
        store = self.transcript_store
        if store is None:
            return
        try:
            store.set_realtime_enabled(self._realtime_transcript_enabled)
            if self._asr_running:
                self._human_log_path = store.open_human_log()
            else:
                self._human_log_path = None
            self._sync_transcript_store_paths_locked()
        except Exception:
            self._human_log_path = None
            self._realtime_transcript_path = None

    def _close_transcript_files_locked(self) -> None:
        store = self.transcript_store
        if store is None:
            self._human_log_path = None
            self._realtime_transcript_path = None
            return
        try:
            store.close_human_log()
            store.close_realtime_transcript()
        finally:
            self._realtime_transcript_enabled = False
            self._sync_transcript_store_paths_locked()

    def _write_transcript_line_locked(self, line: Dict[str, Any]) -> None:
        store = self.transcript_store
        if store is None:
            return
        text = str(line.get("text", "")).strip()
        if not text:
            return
        ts = float(line.get("ts") or time.time())
        stream = str(line.get("stream") or "mix")
        speaker = _clean_speaker(line.get("speaker", ""))
        label = speaker or stream
        formatted = (
            f"[{time.strftime('%H:%M:%S', time.localtime(ts))}] "
            f"{label}: {text}"
        )
        try:
            store.write_human_line(formatted)
            if self._realtime_transcript_enabled:
                store.write_realtime_srt_entry(ts, stream, text, speaker=speaker)
            self._sync_transcript_store_paths_locked()
        except Exception:
            pass

    def _sync_transcript_store_paths_locked(self) -> None:
        store = self.transcript_store
        if store is None:
            return
        self._human_log_path = store.current_human_log_path
        self._realtime_transcript_path = store.realtime_transcript_path

    def _start_asr_locked(self, params: Dict[str, Any]) -> str:
        if self.asr_runtime_factory is None or self.transcription_startup_service is None:
            self._last_warning = "Live ASR is not configured for the Electron backend yet."
            return self._last_warning

        self._drain_asr_events()
        settings = _asr_settings_from_params(params, source_speaker_labels=self._source_speaker_labels_locked())
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
            self._last_warning = f"Live ASR could not start: {message}"
            return self._last_warning

        if result.degraded:
            self._transcription_state.begin_fallback(
                attempt_label=result.attempt.label,
                model_name=result.attempt.settings.model_name,
                reason=result.errors[-1] if result.errors else "",
            )
        self._asr_runtime = result.asr
        self._asr_running = True
        self._transcription_state.finish_start(degraded=result.attempt.degraded, attempt_label=result.attempt.label)
        self._start_asr_event_pump()
        if result.degraded:
            self._last_warning = f"Live ASR started in fallback mode: {result.attempt.label}."
            return self._last_warning
        self._last_warning = ""
        return ""

    def _stop_asr_locked(self, params: Dict[str, Any]) -> None:
        asr = self._asr_runtime
        self._asr_runtime = None
        self._asr_running = False
        self._asr_event_stop.set()
        if asr is None:
            self._transcription_state.reset()
            return

        if self._transcription_state.can_stop:
            self._transcription_state.begin_stop()
        wav_path = Path(str(params.get("wavPath") or self.writer.target_path() or self._current_output_path(str(params.get("outputFile", "")))))
        result = self.stop_asr_use_case.execute(
            StopAsrRequest(
                asr=asr,
                wav_path=wav_path,
                run_offline_pass=bool((params or {}).get("runOfflinePass", False)),
                offline_model_name=str((params or {}).get("offlineModelName", (params or {}).get("model", "large-v3"))),
                offline_language=runtime_asr_language(str((params or {}).get("language", "ru"))),
            )
        )
        self._drain_asr_events()
        stop_error = str(result.stop_error or "")
        if self._transcription_state.is_stopping:
            self._transcription_state.finish_stop(stop_error)
        else:
            self._transcription_state.reset()
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
                "status": "error" if error else "done",
                "wavPath": str(wav_path),
                "outTxt": out_txt,
                "error": error,
                "ts": time.time(),
            }
            try:
                self._session_state.finish_offline_pass(out_txt=out_txt, error=error)
            except Exception:
                pass

    def _start_asr_event_pump(self) -> None:
        if self._asr_event_thread is not None and self._asr_event_thread.is_alive():
            return
        self._asr_event_stop.clear()
        self._asr_event_thread = threading.Thread(target=self._run_asr_event_pump, name="electron-asr-events", daemon=True)
        self._asr_event_thread.start()

    def _run_asr_event_pump(self) -> None:
        while not self._asr_event_stop.is_set():
            drained = self._drain_asr_events(limit=80)
            if drained <= 0:
                time.sleep(0.08)
        self._drain_asr_events(limit=500)

    def _drain_asr_events(self, *, limit: int = 500) -> int:
        count = 0
        while count < limit:
            try:
                raw = self.asr_event_queue.get_nowait()
            except queue.Empty:
                break
            count += 1
            self._handle_asr_event(raw)
        return count

    def _drain_tap_queue(self) -> int:
        count = 0
        while True:
            try:
                self.tap_queue.get_nowait()
                count += 1
            except queue.Empty:
                return count

    def _handle_asr_event(self, raw: object) -> None:
        event = event_from_record(raw)
        record = event_to_record(event)
        with self._lock:
            if isinstance(event, UtteranceEvent):
                text = event.text.strip()
                if text:
                    stream = str(event.stream)
                    speaker = _clean_speaker(event.speaker) or self._speaker_label_for_stream_locked(stream)
                    t_start = _optional_float(event.t_start)
                    t_end = _optional_float(event.t_end)
                    line = {
                        "id": build_transcript_line_id(
                            stream=stream,
                            t_start=t_start,
                            t_end=t_end,
                            ts=float(event.ts),
                        ),
                        "ts": float(event.ts),
                        "stream": stream,
                        "speaker": speaker,
                        "t_start": t_start,
                        "t_end": t_end,
                        "text": text,
                        "overload": bool(event.overload),
                    }
                    self._apply_pending_speaker_updates_locked(line)
                    self._transcript_lines.append(line)
                    if len(self._transcript_lines) > 500:
                        del self._transcript_lines[:-500]
                    record["line"] = line
                    self._write_transcript_line_locked(line)
            elif isinstance(event, TranscriptSpeakerUpdateEvent):
                line = self._find_line_for_speaker_update_locked(event)
                if line is None:
                    self._remember_pending_speaker_update_locked(event)
                elif self._update_line_speaker_from_event_locked(line, event):
                    record["line"] = dict(line)
            elif isinstance(event, AsrMetricsEvent):
                self._asr_metrics = {
                    "segDroppedTotal": int(event.seg_dropped_total),
                    "segSkippedTotal": int(event.seg_skipped_total),
                    "avgLatencyS": float(event.avg_latency_s),
                    "p95LatencyS": float(event.p95_latency_s),
                    "lagS": float(event.lag_s),
                }
            elif isinstance(event, SourceErrorEvent):
                self._last_error = f"{event.source}: {event.error}"
        self._emit("asr_event", record)
        if isinstance(event, UtteranceEvent) and str(record.get("text", "")).strip():
            self._emit("transcript_line", record.get("line", {}))
        if isinstance(event, TranscriptSpeakerUpdateEvent) and record.get("line"):
            self._emit("transcript_line_update", record.get("line", {}))

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
            mode="sources",
            sources=[source.name for source in self._sources.values() if source.enabled],
            drop_threshold=0.85,
        )

    def _safe_meters(self) -> Dict[str, Any]:
        try:
            meters = self.engine.get_meters()
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return {}
        return meters if isinstance(meters, dict) else {}

    def _source_record(self, name: str, *, meters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        source = self._require_source(name)
        source_meters = ((meters or {}).get("sources", {}) or {}).get(name, {})
        rms = _safe_float(source_meters.get("rms", 0.0), 0.0)
        last_ts = _safe_float(source_meters.get("last_ts", 0.0), 0.0)
        active = bool(time.monotonic() - last_ts < 0.6 and rms > 1e-4)
        return {
            "name": source.name,
            "kind": source.kind,
            "label": source.label,
            "enabled": bool(source.enabled),
            "delayMs": float(source.delay_ms),
            "rms": rms,
            "level": _rms_to_pct(rms),
            "active": active,
            "status": "muted" if not source.enabled else ("active" if active else "silence"),
            "bufferFrames": int(_safe_float(source_meters.get("buffer_frames", 0), 0.0)),
            "droppedInFrames": int(_safe_float(source_meters.get("dropped_in_frames", 0), 0.0)),
            "missingOutFrames": int(_safe_float(source_meters.get("missing_out_frames", 0), 0.0)),
            "sampleRate": int(_safe_float(source_meters.get("src_rate", 0), 0.0)),
        }

    def _source_speaker_labels_locked(self) -> Dict[str, str]:
        return {
            source.name: default_speaker_label_for_source_kind(source.kind)
            for source in self._sources.values()
            if default_speaker_label_for_source_kind(source.kind)
        }

    def _speaker_label_for_stream_locked(self, stream: str) -> str:
        source = self._sources.get(str(stream))
        if source is None:
            return ""
        return default_speaker_label_for_source_kind(source.kind)

    def _update_line_speaker_from_event_locked(
        self,
        line: Dict[str, Any],
        event: TranscriptSpeakerUpdateEvent,
    ) -> bool:
        return update_line_speaker(
            line,
            speaker=str(event.speaker),
            speaker_source=str(event.source or "diarization"),
            confidence=_optional_float(event.confidence),
        )

    def _find_line_for_speaker_update_locked(
        self,
        event: TranscriptSpeakerUpdateEvent,
        *,
        lines: Optional[list[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        return best_line_for_speaker_update(
            lines if lines is not None else self._transcript_lines,
            line_id=str(event.line_id or ""),
            stream=str(event.stream or ""),
            t_start=_optional_float(event.t_start),
            t_end=_optional_float(event.t_end),
        )

    def _remember_pending_speaker_update_locked(self, event: TranscriptSpeakerUpdateEvent) -> None:
        if not _clean_speaker(event.speaker):
            return
        self._pending_speaker_updates.append(event)
        if len(self._pending_speaker_updates) > _MAX_PENDING_SPEAKER_UPDATES:
            del self._pending_speaker_updates[:-_MAX_PENDING_SPEAKER_UPDATES]

    def _apply_pending_speaker_updates_locked(self, line: Dict[str, Any]) -> None:
        pending: list[TranscriptSpeakerUpdateEvent] = []
        for event in self._pending_speaker_updates:
            target = self._find_line_for_speaker_update_locked(event, lines=[line])
            if target is line:
                self._update_line_speaker_from_event_locked(line, event)
            else:
                pending.append(event)
        self._pending_speaker_updates = pending[-_MAX_PENDING_SPEAKER_UPDATES:]

    def _require_source(self, name: str) -> HeadlessSource:
        source = self._sources.get(str(name))
        if source is None:
            raise KeyError(f"Unknown source: {name}")
        return source

    def _make_unique_name(self, base: str) -> str:
        candidate = str(base or "source").strip() or "source"
        if candidate not in self._sources:
            return candidate
        index = 2
        while f"{candidate}_{index}" in self._sources:
            index += 1
        return f"{candidate}_{index}"

    def _ensure_not_running(self, action: str) -> None:
        if self._session_state.is_running or self.engine.is_running():
            raise RuntimeError(f"Stop the current session before you {action}")

    def _on_source_error(self, source: str, error: str) -> None:
        message = f"{source}: {error}"
        with self._lock:
            self._last_error = message
        self._emit("source_error", {"source": str(source), "message": str(error)})

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        sink = self.event_sink
        if sink is None:
            return
        try:
            sink(event_type, {"ts": time.time(), **payload})
        except Exception:
            pass


def _normalize_source_kind(kind: str) -> str:
    normalized = str(kind or "").strip().lower()
    if normalized in {"loopback", "system", "desktop", "desktop_audio"}:
        return "loopback"
    if normalized in {"input", "mic", "microphone"}:
        return "input"
    if normalized in {"process", "app", "application", "per_process"}:
        return "process"
    raise ValueError(f"Unsupported source kind: {kind}")


def _default_source_name(kind: str) -> str:
    if kind == "loopback":
        return "desktop_audio"
    if kind == "process":
        return "app_audio"
    return "mic"


def _safe_float(raw: object, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _optional_float(raw: object) -> Optional[float]:
    try:
        if raw is None:
            return None
        return float(raw)
    except Exception:
        return None


def _rms_to_pct(rms: float) -> int:
    value = max(0.0, min(1.0, float(rms)))
    return max(0, min(100, int((value**0.5) * 100.0)))


def _master_record(meters: Dict[str, Any]) -> Dict[str, Any]:
    master = meters.get("master", {}) if isinstance(meters, dict) else {}
    rms = _safe_float(master.get("rms", 0.0), 0.0)
    last_ts = _safe_float(master.get("last_ts", 0.0), 0.0)
    return {
        "rms": rms,
        "level": _rms_to_pct(rms),
        "active": bool(time.monotonic() - last_ts < 0.6 and rms > 1e-4),
        "lastTs": last_ts,
    }


def _drops_record(meters: Dict[str, Any], writer: WavRecorder) -> Dict[str, Any]:
    drops = meters.get("drops", {}) if isinstance(meters, dict) else {}
    drained = getattr(writer, "drained_blocks", lambda: 0)
    written = getattr(writer, "written_blocks", lambda: 0)
    return {
        "droppedOutBlocks": int(_safe_float(drops.get("dropped_out_blocks", 0), 0.0)),
        "droppedTapBlocks": int(_safe_float(drops.get("dropped_tap_blocks", 0), 0.0)),
        "drainedBlocks": int(drained()),
        "writtenBlocks": int(written()),
    }


def _asr_settings_from_params(
    params: Dict[str, Any],
    *,
    source_speaker_labels: Optional[Dict[str, str]] = None,
) -> ASRSessionSettings:
    language = normalize_asr_language(str(params.get("language", params.get("lang", "ru"))))
    mode_raw = str(params.get("asrMode", params.get("asr_mode", "split"))).strip().lower()
    mode = "split" if mode_raw in {"1", "split", "sources"} else "mix"
    overload_strategy = str(params.get("overload_strategy", params.get("overloadStrategy", "drop_old"))).strip().lower()
    return ASRSessionSettings(
        language=language,
        mode=mode,  # type: ignore[arg-type]
        model_name=str(params.get("model", params.get("model_name", "medium")) or "medium"),
        device=str(params.get("device", "cuda") or "cuda"),
        compute_type=str(params.get("compute_type", params.get("computeType", "float16")) or "float16"),
        cpu_threads=_safe_int(params.get("cpu_threads", params.get("cpuThreads", 0)), 0, 0, 64),
        num_workers=_safe_int(params.get("num_workers", params.get("numWorkers", 1)), 1, 1, 16),
        beam_size=_safe_int(params.get("beam_size", params.get("beamSize", 5)), 5, 1, 20),
        endpoint_silence_ms=_safe_float_clamped(
            params.get("endpoint_silence_ms", params.get("endpointSilenceMs", 650.0)),
            650.0,
            50.0,
            5000.0,
        ),
        max_segment_s=_safe_float_clamped(params.get("max_segment_s", params.get("maxSegmentS", 7.0)), 7.0, 1.0, 60.0),
        overlap_ms=_safe_float_clamped(params.get("overlap_ms", params.get("overlapMs", 200.0)), 200.0, 0.0, 2000.0),
        vad_energy_threshold=_safe_float_clamped(
            params.get("vad_energy_threshold", params.get("vadEnergyThreshold", 0.0055)),
            0.0055,
            1e-5,
            1.0,
        ),
        overload_strategy="keep_all" if overload_strategy == "keep_all" else "drop_old",  # type: ignore[arg-type]
        overload_enter_qsize=_safe_int(params.get("overload_enter_qsize", params.get("overloadEnterQsize", 18)), 18, 1, 999),
        overload_exit_qsize=_safe_int(params.get("overload_exit_qsize", params.get("overloadExitQsize", 6)), 6, 1, 999),
        overload_hard_qsize=_safe_int(params.get("overload_hard_qsize", params.get("overloadHardQsize", 28)), 28, 1, 999),
        overload_beam_cap=_safe_int(params.get("overload_beam_cap", params.get("overloadBeamCap", 2)), 2, 1, 20),
        overload_max_segment_s=_safe_float_clamped(
            params.get("overload_max_segment_s", params.get("overloadMaxSegmentS", 5.0)),
            5.0,
            0.5,
            60.0,
        ),
        overload_overlap_ms=_safe_float_clamped(
            params.get("overload_overlap_ms", params.get("overloadOverlapMs", 120.0)),
            120.0,
            0.0,
            2000.0,
        ),
        asr_language=runtime_asr_language(language),
        asr_initial_prompt=initial_prompt_for_language(language),
        source_speaker_labels=dict(source_speaker_labels or {}),
        diarization_enabled=bool(params.get("diarizationEnabled", params.get("diarization_enabled", False))),
        diar_backend=_normalize_diar_backend(params.get("diarBackend", params.get("diar_backend", "online"))),
        diarization_sidecar_enabled=bool(
            params.get("diarizationSidecarEnabled", params.get("diarization_sidecar_enabled", True))
        ),
        diarization_queue_size=_safe_int(
            params.get("diarization_queue_size", params.get("diarizationQueueSize", 50)),
            50,
            1,
            500,
        ),
        diar_sherpa_embedding_model_path=str(
            _first_param(
                params,
                "diarSherpaEmbeddingModelPath",
                "diar_sherpa_embedding_model_path",
                "diarizationSherpaEmbeddingModelPath",
                default="",
            )
            or ""
        ).strip(),
        diar_sherpa_provider=str(
            _first_param(params, "diarSherpaProvider", "diar_sherpa_provider", default="cpu") or "cpu"
        ).strip()
        or "cpu",
        diar_sherpa_num_threads=_safe_int(
            _first_param(params, "diarSherpaNumThreads", "diar_sherpa_num_threads", default=1),
            1,
            1,
            32,
        ),
    )


def _clean_speaker(raw: object) -> str:
    text = str(raw or "").strip()
    if not text or text == "S?":
        return ""
    return text


def _line_speaker_or_stream(line: Dict[str, Any]) -> str:
    return _clean_speaker(line.get("speaker", "")) or str(line.get("stream") or "mix")


def _normalize_diar_backend(raw: object):
    value = str(raw or "online").strip().lower()
    if value in {"pyannote", "online", "nemo", "sherpa_onnx"}:
        return value
    return "online"


def _first_param(params: Dict[str, Any], *keys: str, default: object = None) -> object:
    for key in keys:
        if key in params:
            return params[key]
    return default


def _safe_int(raw: object, default: int, lo: int, hi: int) -> int:
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    return max(int(lo), min(int(hi), value))


def _safe_float_clamped(raw: object, default: float, lo: float, hi: float) -> float:
    value = _safe_float(raw, default)
    return max(float(lo), min(float(hi), value))
