from __future__ import annotations

from audio.writer import WavWriterThread, soundfile_available


def wav_recording_available() -> bool:
    return soundfile_available()


def create_wav_writer(output_queue) -> WavWriterThread:
    return WavWriterThread(output_queue)
