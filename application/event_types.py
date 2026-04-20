from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class EventType(str, Enum):
    SOURCE_ERROR = "source_error"
    UTTERANCE = "utterance"
    ASR_OVERLOAD = "asr_overload"
    SEGMENT_DROPPED = "segment_dropped"
    SEGMENT_SKIPPED_OVERLOAD = "segment_skipped_overload"
    ASR_METRICS = "asr_metrics"
    ASR_INIT_START = "asr_init_start"
    ASR_STARTED = "asr_started"
    ASR_INIT_OK = "asr_init_ok"
    ERROR = "error"
    ASR_STOPPED = "asr_stopped"
    ASR_STOP_DONE = "asr_stop_done"
    OFFLINE_PASS_STARTED = "offline_pass_started"
    OFFLINE_PASS_DONE = "offline_pass_done"
    OFFLINE_PASS_ERROR = "offline_pass_error"
    CODEX_RESULT = "codex_result"
    CODEX_FALLBACK_STARTED = "codex_fallback_started"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TypedEvent:
    event_type: EventType
    ts: float

    def as_record(self) -> Dict[str, Any]:
        return {"type": self.event_type.value, "ts": float(self.ts)}


@dataclass(frozen=True)
class UnknownEvent(TypedEvent):
    record: Dict[str, Any]

    def __init__(self, record: Dict[str, Any]):
        object.__setattr__(self, "event_type", EventType.UNKNOWN)
        object.__setattr__(self, "ts", _float(record.get("ts"), time.time()))
        object.__setattr__(self, "record", dict(record))

    def as_record(self) -> Dict[str, Any]:
        return dict(self.record)


@dataclass(frozen=True)
class SourceErrorEvent(TypedEvent):
    source: str
    error: str

    def __init__(self, *, source: str, error: str, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.SOURCE_ERROR)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "source", str(source))
        object.__setattr__(self, "error", str(error))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"source": self.source, "error": self.error})
        return rec


@dataclass(frozen=True)
class UtteranceEvent(TypedEvent):
    text: str
    stream: str
    overload: bool = False

    def __init__(self, *, text: str, stream: str = "", overload: bool = False, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.UTTERANCE)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "text", str(text))
        object.__setattr__(self, "stream", str(stream))
        object.__setattr__(self, "overload", bool(overload))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"text": self.text, "stream": self.stream, "overload": self.overload})
        return rec


@dataclass(frozen=True)
class AsrOverloadEvent(TypedEvent):
    active: bool
    reason: str
    seg_qsize: Optional[int]
    beam_cur: Optional[int]
    lag_s: Optional[float]

    def __init__(
        self,
        *,
        active: bool,
        reason: str,
        seg_qsize: Optional[int] = None,
        beam_cur: Optional[int] = None,
        lag_s: Optional[float] = None,
        ts: Optional[float] = None,
    ):
        object.__setattr__(self, "event_type", EventType.ASR_OVERLOAD)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "active", bool(active))
        object.__setattr__(self, "reason", str(reason))
        object.__setattr__(self, "seg_qsize", seg_qsize)
        object.__setattr__(self, "beam_cur", beam_cur)
        object.__setattr__(self, "lag_s", lag_s)

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update(
            {
                "active": self.active,
                "reason": self.reason,
                "seg_qsize": self.seg_qsize,
                "beam_cur": self.beam_cur,
                "lag_s": self.lag_s,
            }
        )
        return rec


@dataclass(frozen=True)
class SegmentDroppedEvent(TypedEvent):
    stream: str
    reason: str
    seg_qsize: Optional[int]

    def __init__(self, *, stream: str = "", reason: str = "", seg_qsize: Optional[int] = None, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.SEGMENT_DROPPED)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "stream", str(stream))
        object.__setattr__(self, "reason", str(reason))
        object.__setattr__(self, "seg_qsize", seg_qsize)

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"stream": self.stream, "reason": self.reason, "seg_qsize": self.seg_qsize})
        return rec


@dataclass(frozen=True)
class SegmentSkippedOverloadEvent(TypedEvent):
    count: int
    seg_qsize: Optional[int]

    def __init__(self, *, count: int, seg_qsize: Optional[int] = None, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.SEGMENT_SKIPPED_OVERLOAD)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "count", int(count))
        object.__setattr__(self, "seg_qsize", seg_qsize)

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"count": self.count, "seg_qsize": self.seg_qsize})
        return rec


@dataclass(frozen=True)
class AsrMetricsEvent(TypedEvent):
    seg_dropped_total: int
    seg_skipped_total: int
    avg_latency_s: float
    p95_latency_s: float
    lag_s: float

    def __init__(
        self,
        *,
        seg_dropped_total: int = 0,
        seg_skipped_total: int = 0,
        avg_latency_s: float = 0.0,
        p95_latency_s: float = 0.0,
        lag_s: float = 0.0,
        ts: Optional[float] = None,
    ):
        object.__setattr__(self, "event_type", EventType.ASR_METRICS)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "seg_dropped_total", int(seg_dropped_total))
        object.__setattr__(self, "seg_skipped_total", int(seg_skipped_total))
        object.__setattr__(self, "avg_latency_s", float(avg_latency_s))
        object.__setattr__(self, "p95_latency_s", float(p95_latency_s))
        object.__setattr__(self, "lag_s", float(lag_s))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update(
            {
                "seg_dropped_total": self.seg_dropped_total,
                "seg_skipped_total": self.seg_skipped_total,
                "avg_latency_s": self.avg_latency_s,
                "p95_latency_s": self.p95_latency_s,
                "lag_s": self.lag_s,
            }
        )
        return rec


@dataclass(frozen=True)
class AsrInitStartEvent(TypedEvent):
    model: str
    device: str

    def __init__(self, *, model: str = "", device: str = "", ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.ASR_INIT_START)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "model", str(model))
        object.__setattr__(self, "device", str(device))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"model": self.model, "device": self.device})
        return rec


@dataclass(frozen=True)
class AsrStartedEvent(TypedEvent):
    model: str
    mode: str
    language: str
    overload_strategy: str

    def __init__(
        self,
        *,
        model: str = "",
        mode: str = "",
        language: str = "",
        overload_strategy: str = "",
        ts: Optional[float] = None,
    ):
        object.__setattr__(self, "event_type", EventType.ASR_STARTED)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "model", str(model))
        object.__setattr__(self, "mode", str(mode))
        object.__setattr__(self, "language", str(language))
        object.__setattr__(self, "overload_strategy", str(overload_strategy))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update(
            {
                "model": self.model,
                "mode": self.mode,
                "language": self.language,
                "overload_strategy": self.overload_strategy,
            }
        )
        return rec


@dataclass(frozen=True)
class AsrInitOkEvent(TypedEvent):
    model: str

    def __init__(self, *, model: str = "", ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.ASR_INIT_OK)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "model", str(model))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"model": self.model})
        return rec


@dataclass(frozen=True)
class AsrErrorEvent(TypedEvent):
    where: str
    error: str

    def __init__(self, *, where: str = "", error: str = "", ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.ERROR)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "where", str(where))
        object.__setattr__(self, "error", str(error))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"where": self.where, "error": self.error})
        return rec


@dataclass(frozen=True)
class AsrStoppedEvent(TypedEvent):
    def __init__(self, *, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.ASR_STOPPED)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))


@dataclass(frozen=True)
class AsrStopDoneEvent(TypedEvent):
    wav_path: str
    run_offline_pass: bool
    offline_model_name: str
    offline_language: Optional[str]
    stop_error: Optional[str]

    def __init__(
        self,
        *,
        wav_path: str,
        run_offline_pass: bool,
        offline_model_name: str,
        offline_language: Optional[str],
        stop_error: Optional[str],
        ts: Optional[float] = None,
    ):
        object.__setattr__(self, "event_type", EventType.ASR_STOP_DONE)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "wav_path", str(wav_path))
        object.__setattr__(self, "run_offline_pass", bool(run_offline_pass))
        object.__setattr__(self, "offline_model_name", str(offline_model_name))
        object.__setattr__(self, "offline_language", offline_language)
        object.__setattr__(self, "stop_error", stop_error)

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update(
            {
                "wav_path": self.wav_path,
                "run_offline_pass": self.run_offline_pass,
                "offline_model_name": self.offline_model_name,
                "offline_language": self.offline_language,
                "stop_error": self.stop_error,
            }
        )
        return rec


@dataclass(frozen=True)
class OfflinePassStartedEvent(TypedEvent):
    def __init__(self, *, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.OFFLINE_PASS_STARTED)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))


@dataclass(frozen=True)
class OfflinePassDoneEvent(TypedEvent):
    out_txt: str

    def __init__(self, *, out_txt: str, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.OFFLINE_PASS_DONE)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "out_txt", str(out_txt))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"out_txt": self.out_txt})
        return rec


@dataclass(frozen=True)
class OfflinePassErrorEvent(TypedEvent):
    error: str

    def __init__(self, *, error: str, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.OFFLINE_PASS_ERROR)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "error", str(error))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"error": self.error})
        return rec


@dataclass(frozen=True)
class CodexResultEvent(TypedEvent):
    ok: bool
    profile: str
    cmd: str
    text: str
    dt_s: float

    def __init__(self, *, ok: bool, profile: str, cmd: str, text: str, dt_s: float, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.CODEX_RESULT)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "ok", bool(ok))
        object.__setattr__(self, "profile", str(profile))
        object.__setattr__(self, "cmd", str(cmd))
        object.__setattr__(self, "text", str(text))
        object.__setattr__(self, "dt_s", float(dt_s))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"ok": self.ok, "profile": self.profile, "cmd": self.cmd, "text": self.text, "dt_s": self.dt_s})
        return rec


@dataclass(frozen=True)
class CodexFallbackStartedEvent(TypedEvent):
    profile: str
    cmd: str
    reason: str

    def __init__(self, *, profile: str, cmd: str, reason: str, ts: Optional[float] = None):
        object.__setattr__(self, "event_type", EventType.CODEX_FALLBACK_STARTED)
        object.__setattr__(self, "ts", time.time() if ts is None else float(ts))
        object.__setattr__(self, "profile", str(profile))
        object.__setattr__(self, "cmd", str(cmd))
        object.__setattr__(self, "reason", str(reason))

    def as_record(self) -> Dict[str, Any]:
        rec = super().as_record()
        rec.update({"profile": self.profile, "cmd": self.cmd, "reason": self.reason})
        return rec


def event_from_record(raw: Any) -> TypedEvent:
    if isinstance(raw, TypedEvent):
        return raw
    if not isinstance(raw, dict):
        return UnknownEvent({"type": "unknown", "value": repr(raw), "ts": time.time()})

    typ = str(raw.get("type", ""))
    ts = _float(raw.get("ts"), time.time())
    if typ == EventType.SOURCE_ERROR.value:
        return SourceErrorEvent(source=str(raw.get("source", "")), error=str(raw.get("error", "")), ts=ts)
    if typ == EventType.UTTERANCE.value:
        return UtteranceEvent(
            text=str(raw.get("text", "")),
            stream=str(raw.get("stream", "")),
            overload=bool(raw.get("overload", False)),
            ts=ts,
        )
    if typ == EventType.ASR_OVERLOAD.value:
        return AsrOverloadEvent(
            active=bool(raw.get("active", False)),
            reason=str(raw.get("reason", "")),
            seg_qsize=_optional_int(raw.get("seg_qsize")),
            beam_cur=_optional_int(raw.get("beam_cur")),
            lag_s=_optional_float(raw.get("lag_s")),
            ts=ts,
        )
    if typ == EventType.SEGMENT_DROPPED.value:
        return SegmentDroppedEvent(
            stream=str(raw.get("stream", "")),
            reason=str(raw.get("reason", "")),
            seg_qsize=_optional_int(raw.get("seg_qsize")),
            ts=ts,
        )
    if typ == EventType.SEGMENT_SKIPPED_OVERLOAD.value:
        return SegmentSkippedOverloadEvent(
            count=_int(raw.get("count"), 0),
            seg_qsize=_optional_int(raw.get("seg_qsize")),
            ts=ts,
        )
    if typ == EventType.ASR_METRICS.value:
        return AsrMetricsEvent(
            seg_dropped_total=_int(raw.get("seg_dropped_total"), 0),
            seg_skipped_total=_int(raw.get("seg_skipped_total"), 0),
            avg_latency_s=_float(raw.get("avg_latency_s"), 0.0),
            p95_latency_s=_float(raw.get("p95_latency_s"), 0.0),
            lag_s=_float(raw.get("lag_s"), 0.0),
            ts=ts,
        )
    if typ == EventType.ASR_INIT_START.value:
        return AsrInitStartEvent(model=str(raw.get("model", "")), device=str(raw.get("device", "")), ts=ts)
    if typ == EventType.ASR_STARTED.value:
        return AsrStartedEvent(
            model=str(raw.get("model", "")),
            mode=str(raw.get("mode", "")),
            language=str(raw.get("language", "")),
            overload_strategy=str(raw.get("overload_strategy", "")),
            ts=ts,
        )
    if typ == EventType.ASR_INIT_OK.value:
        return AsrInitOkEvent(model=str(raw.get("model", "")), ts=ts)
    if typ == EventType.ERROR.value:
        return AsrErrorEvent(where=str(raw.get("where", "")), error=str(raw.get("error", "")), ts=ts)
    if typ == EventType.ASR_STOPPED.value:
        return AsrStoppedEvent(ts=ts)
    if typ == EventType.ASR_STOP_DONE.value:
        return AsrStopDoneEvent(
            wav_path=str(raw.get("wav_path", "")),
            run_offline_pass=bool(raw.get("run_offline_pass", False)),
            offline_model_name=str(raw.get("offline_model_name", "")),
            offline_language=raw.get("offline_language"),
            stop_error=raw.get("stop_error"),
            ts=ts,
        )
    if typ == EventType.OFFLINE_PASS_STARTED.value:
        return OfflinePassStartedEvent(ts=ts)
    if typ == EventType.OFFLINE_PASS_DONE.value:
        return OfflinePassDoneEvent(out_txt=str(raw.get("out_txt", "")), ts=ts)
    if typ == EventType.OFFLINE_PASS_ERROR.value:
        return OfflinePassErrorEvent(error=str(raw.get("error", "")), ts=ts)
    if typ == EventType.CODEX_RESULT.value:
        return CodexResultEvent(
            ok=bool(raw.get("ok", False)),
            profile=str(raw.get("profile", "")),
            cmd=str(raw.get("cmd", "")),
            text=str(raw.get("text", "")),
            dt_s=_float(raw.get("dt_s"), 0.0),
            ts=ts,
        )
    if typ == EventType.CODEX_FALLBACK_STARTED.value:
        return CodexFallbackStartedEvent(
            profile=str(raw.get("profile", "")),
            cmd=str(raw.get("cmd", "")),
            reason=str(raw.get("reason", "")),
            ts=ts,
        )
    return UnknownEvent(raw)


def event_to_record(event: Any) -> Dict[str, Any]:
    if isinstance(event, TypedEvent):
        return event.as_record()
    if isinstance(event, dict):
        return dict(event)
    return {"type": "unknown", "value": repr(event), "ts": time.time()}


def _int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _optional_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    return _int(raw, 0)


def _optional_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    return _float(raw, 0.0)
