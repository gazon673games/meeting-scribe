from __future__ import annotations

import time
from typing import Any, Dict, Optional

from interface.session_controller_parts.helpers import default_source_name, normalize_source_kind, rms_to_pct, safe_float
from interface.session_controller_parts.types import EventSink, HeadlessSource
from session.domain.speaker_labels import default_speaker_label_for_source_kind


class SourceControlMixin:
    def set_event_sink(self, event_sink: Optional[EventSink]) -> None:
        with self._lock:
            self.event_sink = event_sink

    def add_source(self, *, kind: str, token: object, label: str = "", name: str = "") -> Dict[str, Any]:
        with self._lock:
            normalized_kind = normalize_source_kind(kind)
            source_name = self._make_unique_name(name or default_source_name(normalized_kind))

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
            value = safe_float(delay_ms, 0.0)
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
        rms = safe_float(source_meters.get("rms", 0.0), 0.0)
        last_ts = safe_float(source_meters.get("last_ts", 0.0), 0.0)
        active = bool(time.monotonic() - last_ts < 0.6 and rms > 1e-4)
        return {
            "name": source.name,
            "kind": source.kind,
            "label": source.label,
            "enabled": bool(source.enabled),
            "delayMs": float(source.delay_ms),
            "rms": rms,
            "level": rms_to_pct(rms),
            "active": active,
            "status": "muted" if not source.enabled else ("active" if active else "silence"),
            "bufferFrames": int(safe_float(source_meters.get("buffer_frames", 0), 0.0)),
            "droppedInFrames": int(safe_float(source_meters.get("dropped_in_frames", 0), 0.0)),
            "missingOutFrames": int(safe_float(source_meters.get("missing_out_frames", 0), 0.0)),
            "sampleRate": int(safe_float(source_meters.get("src_rate", 0), 0.0)),
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
