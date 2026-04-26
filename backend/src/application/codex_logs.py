from transcription.application.transcript_context import trim_text_tail
from transcription.infrastructure.file_transcript_context import FileTranscriptContextReader

_reader = FileTranscriptContextReader()
read_human_log_tail = _reader.read_human_log_tail

__all__ = ["read_human_log_tail", "trim_text_tail"]
