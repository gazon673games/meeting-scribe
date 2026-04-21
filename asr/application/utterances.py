from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from asr.application.pipeline_config import ASRPipelineSettings

from asr.domain.utterances import UtteranceState
from asr.domain.text import normalize_text


class UtteranceAggregator:
    @classmethod
    def from_settings(cls, settings: ASRPipelineSettings) -> UtteranceAggregator:
        return cls(
            enabled=bool(settings.utterance_enabled),
            gap_s=float(settings.utterance_gap_s),
            max_s=float(settings.utterance_max_s),
            flush_s=float(settings.utterance_flush_s),
            log_speaker_labels=bool(settings.log_speaker_labels),
        )

    def to_event_dict(self) -> dict:
        return {
            "utterance_enabled": self.enabled,
            "utterance_gap_s": self.gap_s,
            "utterance_max_s": self.max_s,
            "utterance_flush_s": self.flush_s,
        }

    def __init__(
        self,
        *,
        enabled: bool,
        gap_s: float,
        max_s: float,
        flush_s: float,
        log_speaker_labels: bool,
    ) -> None:
        self.enabled = bool(enabled)
        self.gap_s = float(gap_s)
        self.max_s = float(max_s)
        self.flush_s = float(flush_s)
        self.log_speaker_labels = bool(log_speaker_labels)
        self._state: Dict[Tuple[str, str], UtteranceState] = {}
        self._last_flush_check_ts = 0.0

    def flush_all(self, *, now: float, force: bool, overload: bool) -> List[dict]:
        events: List[dict] = []
        for key in list(self._state.keys()):
            ev = self._flush_one(key, now=now, force=force, overload=overload)
            if ev is not None:
                events.append(ev)
        return events

    def update(
        self,
        *,
        stream: str,
        speaker: str,
        t_start: float,
        t_end: float,
        text: str,
        now: float,
        overload: bool,
    ) -> List[dict]:
        if not self.enabled:
            return []

        txt = normalize_text(text)
        if not txt:
            return []

        events: List[dict] = []
        spk = str(speaker if self.log_speaker_labels else "S?")
        key = (str(stream), spk)
        state = self._state.get(key)

        if (now - float(self._last_flush_check_ts)) > 0.5:
            self._last_flush_check_ts = now
            events.extend(self.flush_all(now=now, force=False, overload=overload))
            state = self._state.get(key)

        if state is None:
            self._state[key] = UtteranceState(
                stream=str(stream),
                speaker=spk,
                t_start=float(t_start),
                t_end=float(t_end),
                text=txt,
                last_emit_ts=now,
            )
            return events

        if state.can_extend(t_start=t_start, t_end=t_end, gap_s=self.gap_s, max_s=self.max_s):
            state.extend(t_end=t_end, text=txt, last_emit_ts=now)
            self._state[key] = state
            return events

        ev = self._flush_one(key, now=now, force=True, overload=overload)
        if ev is not None:
            events.append(ev)

        self._state[key] = UtteranceState(
            stream=str(stream),
            speaker=spk,
            t_start=float(t_start),
            t_end=float(t_end),
            text=txt,
            last_emit_ts=now,
        )
        return events

    def _flush_one(self, key: Tuple[str, str], *, now: float, force: bool, overload: bool) -> dict | None:
        state = self._state.get(key)
        if state is None:
            return None
        if not state.text.strip():
            self._state.pop(key, None)
            return None

        if not state.should_flush(now=now, flush_s=self.flush_s, force=force):
            return None

        self._state.pop(key, None)
        return {
            "type": "utterance",
            "stream": state.stream,
            "speaker": state.speaker if self.log_speaker_labels else "S?",
            "t_start": float(state.t_start),
            "t_end": float(state.t_end),
            "text": normalize_text(state.text),
            "overload": bool(overload),
            "ts": now,
        }
