from __future__ import annotations

from typing import TypedDict


class RecognitionMetadata(TypedDict, total=False):
    beam_size: int
    language: str | None
    language_probability: float | None
    avg_logprob: float
    no_speech_probability: float
    compression_ratio: float
    timestamps: list[float]


class RecognitionResult(RecognitionMetadata):
    text: str


class RecognizedWord(TypedDict):
    text: str
    start: float
    end: float
