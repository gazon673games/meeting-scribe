from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from application.asr_profiles import PROFILE_ULTRA_FAST
from application.diarization_model_download import RECOMMENDED_DIARIZATION_MODELS, diarization_models_dir
from application.event_types import TranscriptSpeakerUpdateEvent, UtteranceEvent
from interface.backend import ElectronBackend
from interface.session_controller import HeadlessSessionController
from settings.infrastructure.json_config_repository import JsonConfigRepository
from tests.electron_interface_fakes import (
    _DeviceCatalog,
    _FakeAsrRuntimeFactory,
    _FakeAudioRuntimeFactory,
    _FakeAudioSourceFactory,
    _FakeWavRecorderFactory,
)
from transcription.application.startup_service import TranscriptionStartupService


class ElectronInterfaceSessionTests(unittest.TestCase):
    def test_headless_session_streams_asr_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runtime_factory = _FakeAudioRuntimeFactory()
            asr_factory = _FakeAsrRuntimeFactory()
            events: list[tuple[str, dict]] = []
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=runtime_factory,
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                asr_runtime_factory=asr_factory,
                transcription_startup_service=TranscriptionStartupService(),
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )

            controller.add_source(kind="input", token=1, label="Mic")
            started = controller.start_session({"asrEnabled": True, "model": "medium", "language": "en"})
            for _ in range(20):
                if any(kind == "transcript_line" for kind, _ in events):
                    break
                time.sleep(0.02)
            snapshot = controller.snapshot()
            controller.stop_session({})

            self.assertTrue(started["asrRunning"])
            self.assertTrue(snapshot["transcript"])
            self.assertEqual(snapshot["transcript"][0]["text"], "hello from asr")
            self.assertEqual(snapshot["transcript"][0]["speaker"], "Me")
            self.assertTrue(asr_factory.runtime.stopped)

    def test_headless_session_writes_speakers_json_and_transcript_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runtime_factory = _FakeAudioRuntimeFactory()
            asr_factory = _FakeAsrRuntimeFactory()
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=runtime_factory,
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                asr_runtime_factory=asr_factory,
                transcription_startup_service=TranscriptionStartupService(),
            )

            controller.add_source(kind="input", token=1, label="Mic")
            controller.start_session({"asrEnabled": True, "model": "medium", "language": "en"})
            for _ in range(20):
                if controller.snapshot()["transcript"]:
                    break
                time.sleep(0.02)
            controller.stop_session({})

            sessions_root = root / ".local" / "sessions"
            session_dirs = [path for path in sessions_root.iterdir() if path.is_dir()]
            self.assertEqual(len(session_dirs), 1)
            session_dir = session_dirs[0]

            speakers_payload = json.loads((session_dir / "speakers.json").read_text(encoding="utf-8"))
            self.assertIn("session_id", speakers_payload)
            self.assertIn("diarization_model", speakers_payload)
            self.assertIn("embedding_model", speakers_payload)
            self.assertIn("speakers", speakers_payload)
            self.assertIsInstance(speakers_payload["speakers"], list)
            self.assertTrue(speakers_payload["speakers"])
            speaker = speakers_payload["speakers"][0]
            self.assertIn("session_speaker_id", speaker)
            self.assertIn("label", speaker)
            self.assertIn("total_speech_ms", speaker)
            self.assertIn("embedding_id", speaker)
            self.assertIn("matched_person_id", speaker)
            self.assertIn("match_similarity", speaker)
            self.assertIn("match_status", speaker)

            transcript_lines = [
                json.loads(line)
                for line in (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(transcript_lines)
            first_line = transcript_lines[0]
            self.assertIn("session_speaker_id", first_line)
            self.assertIn("person_id", first_line)
            self.assertIn("identity_confidence", first_line)

            srt_text = (session_dir / "transcript.srt").read_text(encoding="utf-8")
            self.assertNotIn("person_id", srt_text)
            self.assertNotIn("identity_confidence", srt_text)

    def test_disabled_speaker_identity_does_not_write_persistent_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runtime_factory = _FakeAudioRuntimeFactory()
            asr_factory = _FakeAsrRuntimeFactory()
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=runtime_factory,
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                asr_runtime_factory=asr_factory,
                transcription_startup_service=TranscriptionStartupService(),
            )

            controller.add_source(kind="input", token=1, label="Mic")
            controller.start_session(
                {
                    "asrEnabled": True,
                    "model": "medium",
                    "language": "en",
                    "speakerIdentity": {
                        "enabled": False,
                        "persistent_profiles_enabled": True,
                        "backend": "file",
                    },
                }
            )
            for _ in range(20):
                if controller.snapshot()["transcript"]:
                    break
                time.sleep(0.02)
            controller.stop_session({})

            identity_root = root / ".local" / "identity"
            self.assertFalse((identity_root / "profiles.jsonl").exists())
            self.assertFalse((identity_root / "links.jsonl").exists())
            self.assertFalse((identity_root / "embeddings").exists())

    def test_backend_passes_diarization_config_to_asr_session(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "ui": {"asr_enabled": True, "model": "medium", "lang": "en", "asr_mode": 1},
                    "asr": {
                        "diarization_enabled": True,
                        "diar_backend": "sherpa_onnx",
                        "diarization_sidecar_enabled": True,
                        "diarization_queue_size": 12,
                        "diar_sherpa_embedding_model_path": "models/speaker.onnx",
                        "diar_sherpa_provider": "cpu",
                        "diar_sherpa_num_threads": 2,
                    },
                }
            )
            asr_factory = _FakeAsrRuntimeFactory()
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                asr_runtime_factory=asr_factory,
                transcription_startup_service=TranscriptionStartupService(),
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog(), controller)

            backend.handle("list_devices")
            backend.handle("add_source", {"deviceId": "input:0"})
            state = backend.handle("get_state")
            with patch("application.model_download.is_model_cached", return_value=True):
                backend.handle("start_session", {})
            backend.handle("stop_session", {"runOfflinePass": False})

            self.assertTrue(state["configSummary"]["diarizationEnabled"])
            self.assertIn("sherpa_onnx", state["options"]["diarizationBackends"])
            self.assertTrue(asr_factory.settings.diarization_enabled)
            self.assertEqual(asr_factory.settings.diar_backend, "sherpa_onnx")
            self.assertEqual(asr_factory.settings.diarization_queue_size, 12)
            self.assertEqual(asr_factory.settings.diar_sherpa_embedding_model_path, "models/speaker.onnx")
            self.assertEqual(asr_factory.settings.diar_sherpa_num_threads, 2)

    def test_backend_forces_streaming_for_ultra_fast_profile(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "ui": {
                        "asr_enabled": True,
                        "model": "medium",
                        "lang": "en",
                        "profile": PROFILE_ULTRA_FAST,
                    },
                    "asr": {"streaming_enabled": False},
                }
            )
            asr_factory = _FakeAsrRuntimeFactory()
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                asr_runtime_factory=asr_factory,
                transcription_startup_service=TranscriptionStartupService(),
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog(), controller)

            backend.handle("list_devices")
            backend.handle("add_source", {"deviceId": "input:0"})
            with patch("application.model_download.is_model_cached", return_value=True):
                backend.handle("start_session", {"streamingEnabled": False})
            backend.handle("stop_session", {"runOfflinePass": False})

            self.assertTrue(asr_factory.settings.streaming_enabled)
            self.assertEqual(asr_factory.settings.streaming_chunk_interval_s, 0.45)
            self.assertEqual(asr_factory.settings.streaming_endpoint_silence_ms, 180.0)

    def test_backend_uses_cached_sherpa_model_when_online_backend_dependency_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            spec = RECOMMENDED_DIARIZATION_MODELS[0]
            model_dir = diarization_models_dir(root, root / "models")
            model_dir.mkdir(parents=True)
            model_path = model_dir / spec.file_name
            model_path.write_bytes(b"onnx")
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "ui": {"asr_enabled": True, "model": "medium", "lang": "en"},
                    "asr": {
                        "diarization_enabled": True,
                        "diar_backend": "online",
                        "diar_sherpa_embedding_model_path": "",
                    },
                    "models": {"cache_dir": str(root / "models")},
                }
            )
            asr_factory = _FakeAsrRuntimeFactory()
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                asr_runtime_factory=asr_factory,
                transcription_startup_service=TranscriptionStartupService(),
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog(), controller)

            backend.handle("list_devices")
            backend.handle("add_source", {"deviceId": "input:0"})
            with (
                patch("interface.backend._module_available", return_value=False),
                patch("application.model_download.is_model_cached", return_value=True),
            ):
                backend.handle("start_session", {})
            backend.handle("stop_session", {"runOfflinePass": False})

            self.assertEqual(asr_factory.settings.diar_backend, "sherpa_onnx")
            self.assertEqual(asr_factory.settings.diar_sherpa_embedding_model_path, str(model_path))

    def test_headless_session_applies_post_fact_speaker_update(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            events: list[tuple[str, dict]] = []
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )
            controller.add_source(kind="loopback", token=1, label="Desktop")

            controller._handle_asr_event(  # noqa: SLF001
                UtteranceEvent(
                    text="question",
                    stream="desktop_audio",
                    speaker="Remote",
                    t_start=1.0,
                    t_end=2.0,
                    ts=2.2,
                )
            )
            line_id = controller.snapshot()["transcript"][0]["id"]
            controller._handle_asr_event(  # noqa: SLF001
                TranscriptSpeakerUpdateEvent(
                    line_id=line_id,
                    speaker="Remote S1",
                    confidence=0.9,
                    source="test",
                    ts=2.5,
                )
            )

            line = controller.snapshot()["transcript"][0]
            self.assertEqual(line["speaker"], "Remote S1")
            self.assertEqual(line["speakerSource"], "test")
            self.assertTrue(any(kind == "transcript_line_update" for kind, _ in events))

    def test_headless_session_applies_pending_speaker_update_to_new_line(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            events: list[tuple[str, dict]] = []
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )
            controller.add_source(kind="loopback", token=1, label="Desktop")

            controller._handle_asr_event(  # noqa: SLF001
                TranscriptSpeakerUpdateEvent(
                    stream="desktop_audio",
                    speaker="Remote S2",
                    t_start=1.0,
                    t_end=2.0,
                    source="test",
                    ts=1.5,
                )
            )
            controller._handle_asr_event(  # noqa: SLF001
                UtteranceEvent(
                    text="answer",
                    stream="desktop_audio",
                    speaker="Remote",
                    t_start=1.0,
                    t_end=2.0,
                    ts=2.2,
                )
            )

            line = controller.snapshot()["transcript"][0]
            self.assertEqual(line["speaker"], "Remote S2")
            self.assertEqual(line["speakerSource"], "test")
            self.assertTrue(any(kind == "transcript_line" for kind, _ in events))

