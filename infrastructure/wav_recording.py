from __future__ import annotations

from typing import Any

from application.recording import WavRecorder, WavRecorderFactory
from audio.writer import WavWriterThread, soundfile_available


class WavWriterFactory(WavRecorderFactory):
    def available(self) -> bool:
        return soundfile_available()

    def create(self, output_queue: Any) -> WavRecorder:
        return WavWriterThread(output_queue)


def wav_recording_available() -> bool:
    return soundfile_available()


def create_wav_writer(output_queue: Any) -> WavRecorder:
    return WavWriterThread(output_queue)
