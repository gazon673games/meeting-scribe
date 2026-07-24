from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Deque, Optional

import numpy as np

from asr.domain.streaming import StreamingChunk
from asr.infrastructure.audio_data import MonoAudio16kBuffer


class CoalescingStreamingQueue:
    """Bound intermediate latency while preserving every final utterance."""

    def __init__(self, maxsize: int = 50) -> None:
        self.maxsize = max(1, int(maxsize))
        self._items: Deque[StreamingChunk] = deque()
        self._condition = threading.Condition()

    def put_nowait(self, chunk: StreamingChunk) -> Optional[str]:
        with self._condition:
            if not chunk.is_final:
                for index in range(len(self._items) - 1, -1, -1):
                    pending = self._items[index]
                    if pending.stream == chunk.stream and not pending.is_final:
                        self._items[index] = _merge_incremental_audio(pending, chunk)
                        self._condition.notify()
                        return "coalesced"
                if len(self._items) >= self.maxsize:
                    raise queue.Full
                self._items.append(chunk)
                self._condition.notify()
                return None

            retained: Deque[StreamingChunk] = deque()
            for pending in self._items:
                if pending.stream == chunk.stream and not pending.is_final:
                    chunk = _merge_incremental_audio(pending, chunk)
                else:
                    retained.append(pending)
            self._items = retained
            self._items.append(chunk)
            self._condition.notify()
            return "final_overflow" if len(self._items) > self.maxsize else None

    def get(self, timeout: Optional[float] = None) -> StreamingChunk:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._items:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise queue.Empty
                self._condition.wait(remaining)
            return self._items.popleft()

    def qsize(self) -> int:
        with self._condition:
            return len(self._items)

    def empty(self) -> bool:
        return self.qsize() == 0


def _merge_incremental_audio(previous: StreamingChunk, current: StreamingChunk) -> StreamingChunk:
    previous_audio = previous.incremental_audio
    current_audio = current.incremental_audio
    if previous_audio is None:
        return current
    if current_audio is None:
        return replace(current, incremental_audio=previous_audio)
    merged = np.concatenate([previous_audio.samples, current_audio.samples]).astype(np.float32, copy=False)
    return replace(current, incremental_audio=MonoAudio16kBuffer.from_array(merged))
