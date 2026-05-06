from __future__ import annotations

import time
from typing import Any, Dict, Optional

from application.events import (
    AsrMetricsEvent,
    SourceErrorEvent,
    TranscriptSpeakerUpdateEvent,
    UtteranceEvent,
    event_from_record,
    event_to_record,
)
from interface.session_controller_parts.helpers import clean_speaker, join_words, optional_float
from interface.session_controller_parts.types import MAX_PENDING_SPEAKER_UPDATES
from transcription.domain.transcript_lines import (
    best_line_for_speaker_update,
    build_transcript_line_id,
    update_line_speaker,
)


class TranscriptPipelineMixin:
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
        speaker = clean_speaker(line.get("speaker", ""))
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

    def _handle_asr_event(self, raw: object) -> None:
        event = event_from_record(raw)
        record = event_to_record(event)
        with self._lock:
            self._dispatch_asr_event(event, record)
        self._emit_asr_notifications(event, record)

    def _dispatch_asr_event(self, event: object, record: Dict[str, Any]) -> None:
        if isinstance(event, UtteranceEvent):
            self._handle_utterance_locked(event, record)
        elif isinstance(event, TranscriptSpeakerUpdateEvent):
            self._handle_speaker_update_locked(event, record)
        elif isinstance(event, AsrMetricsEvent):
            self._handle_metrics_locked(event)
        elif isinstance(event, SourceErrorEvent):
            self._last_error = f"{event.source}: {event.error}"
        elif record.get("type") == "streaming_words":
            self._handle_streaming_words_locked(record)
        elif record.get("type") == "streaming_final":
            self._handle_streaming_final_locked(record)

    def _handle_utterance_locked(self, event: UtteranceEvent, record: Dict[str, Any]) -> None:
        text = event.text.strip()
        if not text:
            return
        stream = str(event.stream)
        speaker = clean_speaker(event.speaker) or self._speaker_label_for_stream_locked(stream)
        t_start = optional_float(event.t_start)
        t_end = optional_float(event.t_end)
        line = {
            "id": build_transcript_line_id(stream=stream, t_start=t_start, t_end=t_end, ts=float(event.ts)),
            "ts": float(event.ts),
            "stream": stream,
            "speaker": speaker,
            "t_start": t_start,
            "t_end": t_end,
            "text": text,
            "overload": bool(event.overload),
        }
        self._apply_pending_speaker_updates_locked(line)
        self._append_transcript_line_locked(line)
        record["line"] = line
        self._write_transcript_line_locked(line)

    def _handle_speaker_update_locked(self, event: TranscriptSpeakerUpdateEvent, record: Dict[str, Any]) -> None:
        line = self._find_line_for_speaker_update_locked(event)
        if line is None:
            self._remember_pending_speaker_update_locked(event)
        elif self._update_line_speaker_from_event_locked(line, event):
            record["line"] = dict(line)

    def _handle_metrics_locked(self, event: AsrMetricsEvent) -> None:
        self._asr_metrics = {
            "segDroppedTotal": int(event.seg_dropped_total),
            "segSkippedTotal": int(event.seg_skipped_total),
            "avgLatencyS": float(event.avg_latency_s),
            "p95LatencyS": float(event.p95_latency_s),
            "lagS": float(event.lag_s),
        }

    def _handle_streaming_words_locked(self, record: Dict[str, Any]) -> None:
        stream = str(record.get("stream") or "mix")
        new_confirmed = join_words(record.get("confirmed"))
        tentative_text = join_words(record.get("tentative"))
        tentative_id = f"streaming-{stream}"
        existing = next((l for l in self._transcript_lines if l.get("id") == tentative_id), None)
        if existing is not None:
            confirmed = " ".join(filter(None, [str(existing.get("_confirmed", "")), new_confirmed]))
            existing["_confirmed"] = confirmed
            existing["text"] = " ".join(filter(None, [confirmed, tentative_text]))
            existing["t_end"] = optional_float(record.get("t_end"))
        else:
            text = " ".join(filter(None, [new_confirmed, tentative_text]))
            if text:
                self._append_transcript_line_locked({
                    "id": tentative_id,
                    "_confirmed": new_confirmed,
                    "ts": float(record.get("ts") or time.time()),
                    "stream": stream,
                    "speaker": "",
                    "t_start": optional_float(record.get("t_start")),
                    "t_end": optional_float(record.get("t_end")),
                    "text": text,
                    "overload": False,
                })

    def _handle_streaming_final_locked(self, record: Dict[str, Any]) -> None:
        stream = str(record.get("stream") or "mix")
        tentative_id = f"streaming-{stream}"
        self._transcript_lines = [l for l in self._transcript_lines if l.get("id") != tentative_id]
        text = join_words(record.get("words"))
        if not text:
            return
        t_start = optional_float(record.get("t_start"))
        t_end = optional_float(record.get("t_end"))
        ts = float(record.get("ts") or time.time())
        line = {
            "id": build_transcript_line_id(stream=stream, t_start=t_start, t_end=t_end, ts=ts),
            "ts": ts, "stream": stream, "speaker": "",
            "t_start": t_start, "t_end": t_end, "text": text, "overload": False,
        }
        self._append_transcript_line_locked(line)
        self._write_transcript_line_locked(line)

    def _append_transcript_line_locked(self, line: Dict[str, Any]) -> None:
        self._transcript_lines.append(line)
        if len(self._transcript_lines) > 500:
            del self._transcript_lines[:-500]

    def _emit_asr_notifications(self, event: object, record: Dict[str, Any]) -> None:
        self._emit("asr_event", record)
        if isinstance(event, UtteranceEvent) and str(record.get("text", "")).strip():
            self._emit("transcript_line", record.get("line", {}))
        if isinstance(event, TranscriptSpeakerUpdateEvent) and record.get("line"):
            self._emit("transcript_line_update", record.get("line", {}))

    def _update_line_speaker_from_event_locked(
        self,
        line: Dict[str, Any],
        event: TranscriptSpeakerUpdateEvent,
    ) -> bool:
        return update_line_speaker(
            line,
            speaker=str(event.speaker),
            speaker_source=str(event.source or "diarization"),
            confidence=optional_float(event.confidence),
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
            t_start=optional_float(event.t_start),
            t_end=optional_float(event.t_end),
        )

    def _remember_pending_speaker_update_locked(self, event: TranscriptSpeakerUpdateEvent) -> None:
        if not clean_speaker(event.speaker):
            return
        self._pending_speaker_updates.append(event)
        if len(self._pending_speaker_updates) > MAX_PENDING_SPEAKER_UPDATES:
            del self._pending_speaker_updates[:-MAX_PENDING_SPEAKER_UPDATES]

    def _apply_pending_speaker_updates_locked(self, line: Dict[str, Any]) -> None:
        pending: list[TranscriptSpeakerUpdateEvent] = []
        for event in self._pending_speaker_updates:
            target = self._find_line_for_speaker_update_locked(event, lines=[line])
            if target is line:
                self._update_line_speaker_from_event_locked(line, event)
            else:
                pending.append(event)
        self._pending_speaker_updates = pending[-MAX_PENDING_SPEAKER_UPDATES:]

