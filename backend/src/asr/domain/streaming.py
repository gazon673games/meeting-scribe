from __future__ import annotations

from dataclasses import dataclass
from typing import List

from asr.domain.audio import MonoAudio16k


@dataclass(frozen=True)
class StreamingWord:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class PrefixUpdate:
    newly_confirmed: List[StreamingWord]
    tentative: List[StreamingWord]


@dataclass(frozen=True)
class StreamingChunk:
    stream: str
    t_start: float
    t_end: float
    audio: MonoAudio16k
    is_final: bool
    enqueue_ts: float


class ConfirmedPrefixTracker:
    """
    Confirms words that appear unchanged in consecutive Whisper runs
    with at least `lookahead` stable words following them.

    A word at position i is confirmed when the prefix length stable across
    two consecutive runs satisfies: stable_len - lookahead > i.
    This ensures the word has `lookahead` stable successors, making
    revision by Whisper unlikely.
    """

    def __init__(self, lookahead: int = 2) -> None:
        self._lookahead = max(1, lookahead)
        self._confirmed: List[StreamingWord] = []
        self._prev: List[StreamingWord] = []

    @property
    def confirmed_words(self) -> List[StreamingWord]:
        return list(self._confirmed)

    def update(self, words: List[StreamingWord]) -> PrefixUpdate:
        """Compare current run against previous, extend confirmed prefix."""
        stable_len = _common_prefix_len(words, self._prev)
        n = len(self._confirmed)
        confirm_up_to = max(n, stable_len - self._lookahead)
        newly_confirmed = list(words[n:confirm_up_to])
        self._confirmed.extend(newly_confirmed)
        self._prev = list(words)
        return PrefixUpdate(
            newly_confirmed=newly_confirmed,
            tentative=list(words[len(self._confirmed):]),
        )

    def flush(self) -> None:
        """Confirm all remaining tentative words (call when utterance ends)."""
        remaining = self._prev[len(self._confirmed):]
        self._confirmed.extend(remaining)

    def reset(self) -> None:
        self._confirmed.clear()
        self._prev.clear()


def _common_prefix_len(a: List[StreamingWord], b: List[StreamingWord]) -> int:
    length = 0
    for i, w in enumerate(a):
        if i >= len(b):
            break
        if w.text == b[i].text:
            length = i + 1
        else:
            break
    return length
