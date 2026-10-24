from __future__ import annotations

from dataclasses import dataclass

from asr.domain.audio import MonoAudio16k


@dataclass(frozen=True)
class StreamingWord:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class PrefixUpdate:
    newly_confirmed: list[StreamingWord]
    tentative: list[StreamingWord]


@dataclass(frozen=True)
class StreamingChunk:
    stream: str
    t_start: float
    t_end: float
    audio: MonoAudio16k
    is_final: bool
    enqueue_ts: float
    incremental_audio: MonoAudio16k | None = None
    enqueue_monotonic: float | None = None

    def queue_wait_s(self, monotonic_now: float) -> float:
        if self.enqueue_monotonic is None:
            return 0.0
        return max(0.0, monotonic_now - self.enqueue_monotonic)


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
        self._confirmed: list[StreamingWord] = []
        self._prev: list[StreamingWord] = []

    @property
    def confirmed_words(self) -> list[StreamingWord]:
        return list(self._confirmed)

    def update(self, words: list[StreamingWord]) -> PrefixUpdate:
        """Compare current run against previous, extend confirmed prefix."""
        stable_prefix_length = _common_prefix_len(words, self._prev)
        confirmed_count = len(self._confirmed)
        confirm_up_to = max(
            confirmed_count, stable_prefix_length - self._lookahead
        )
        newly_confirmed = words[confirmed_count:confirm_up_to]
        self._confirmed.extend(newly_confirmed)
        self._prev = list(words)
        return PrefixUpdate(
            newly_confirmed=newly_confirmed,
            tentative=words[len(self._confirmed) :],
        )

    def flush(self) -> None:
        """Confirm all remaining tentative words (call when utterance ends)."""
        remaining = self._prev[len(self._confirmed) :]
        self._confirmed.extend(remaining)

    def reset(self) -> None:
        self._confirmed.clear()
        self._prev.clear()


def _common_prefix_len(
    words: list[StreamingWord], previous_words: list[StreamingWord]
) -> int:
    length = 0
    for word, previous_word in zip(words, previous_words):
        if word.text != previous_word.text:
            break
        length += 1
    return length
