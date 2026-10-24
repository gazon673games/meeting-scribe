from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application.asr_session import ASRSessionSettings
from application.native_asr_models import (
    NEMOTRON_MODEL_ID,
    PARAKEET_MODEL_ID,
    native_asr_model_dir,
)
from infrastructure.asr_pipeline_factory import ASRPipelineFactory
from interface.session_controller_parts.settings import asr_settings_from_params


class _FakePipeline:
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.kwargs = kwargs


class _FakeWorkerRunner:
    pass


class _FakeDiarizationFactory:
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.kwargs = kwargs


class ASRPipelineFactoryTests(unittest.TestCase):
    def test_nemotron_default_cadence_is_shorter_and_explicit_values_are_respected(
        self,
    ):
        settings = asr_settings_from_params({'model': NEMOTRON_MODEL_ID})
        self.assertTrue(settings.streaming_enabled)
        self.assertEqual(settings.streaming_chunk_interval_s, 0.56)
        settings = asr_settings_from_params(
            {'model': NEMOTRON_MODEL_ID, 'streamingChunkIntervalS': 2.0}
        )
        self.assertEqual(settings.streaming_chunk_interval_s, 2.0)

    def test_native_backend_uses_the_selected_models_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            selected = root / 'custom-asr'
            settings = asr_settings_from_params(
                {
                    'model': PARAKEET_MODEL_ID,
                    'asrModelsDir': str(selected),
                }
            )
            with (
                patch(
                    'infrastructure.asr_pipeline_factory.ASRPipeline',
                    _FakePipeline,
                ),
                patch(
                    'infrastructure.asr_pipeline_factory.SherpaOfflineASR'
                ) as factory,
            ):
                runtime = ASRPipelineFactory().build(
                    settings, tap_queue=queue.Queue(), project_root=root
                )
                runtime.kwargs['dependencies'].asr_backend_factory(
                    model_name=PARAKEET_MODEL_ID,
                    language='ru',
                    device='cpu',
                    compute_type='int8',
                    cpu_threads=1,
                    num_workers=1,
                    beam_size=1,
                    initial_prompt=None,
                    hotwords=None,
                )
            self.assertEqual(
                factory.call_args.kwargs['model_dir'],
                native_asr_model_dir(PARAKEET_MODEL_ID, models_dir=selected),
            )

    def test_build_maps_session_settings_into_pipeline_runtime(self) -> None:
        settings = ASRSessionSettings(
            language='ru',
            mode='split',
            model_name='tiny',
            device='cpu',
            compute_type='int8',
            cpu_threads=2,
            num_workers=1,
            beam_size=3,
            endpoint_silence_ms=600,
            max_segment_s=12.5,
            overlap_ms=250,
            vad_energy_threshold=0.02,
            overload_strategy='drop_old',
            overload_enter_qsize=8,
            overload_exit_qsize=3,
            overload_hard_qsize=16,
            overload_beam_cap=2,
            overload_max_segment_s=4.0,
            overload_overlap_ms=50,
            asr_language='ru',
            asr_initial_prompt='terms',
            source_speaker_labels={'mic': 'Speaker A'},
            diarization_enabled=True,
            diar_backend='pyannote',
            diarization_sidecar_enabled=False,
            diarization_queue_size=7,
            diar_sherpa_embedding_model_path='embedding.onnx',
            diar_sherpa_provider='cpu',
            diar_sherpa_num_threads=4,
            streaming_enabled=True,
            streaming_chunk_interval_s=0.25,
            streaming_endpoint_silence_ms=200,
        )
        tap_queue: queue.Queue = queue.Queue()
        event_queue: queue.Queue = queue.Queue()

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                'infrastructure.asr_pipeline_factory.ASRPipeline', _FakePipeline
            ),
            patch(
                'infrastructure.asr_pipeline_factory.ThreadRealtimeWorkerRunner',
                _FakeWorkerRunner,
            ),
            patch(
                'infrastructure.asr_pipeline_factory.DefaultDiarizationRuntimeFactory',
                _FakeDiarizationFactory,
            ),
        ):
            runtime = ASRPipelineFactory().build(
                settings,
                tap_queue=tap_queue,
                project_root=Path(tmp),
                event_queue=event_queue,
            )

        pipeline_settings = runtime.kwargs['settings']
        dependencies = runtime.kwargs['dependencies']
        self.assertIs(runtime.kwargs['tap_queue'], tap_queue)
        self.assertIs(runtime.kwargs['event_queue'], event_queue)
        self.assertEqual(pipeline_settings.language, 'ru')
        self.assertEqual(pipeline_settings.mode, 'split')
        self.assertEqual(
            pipeline_settings.source_speaker_labels, {'mic': 'Speaker A'}
        )
        self.assertEqual(pipeline_settings.asr_model_name, 'tiny')
        self.assertEqual(pipeline_settings.diar_backend, 'pyannote')
        self.assertFalse(pipeline_settings.diarization_sidecar_enabled)
        self.assertTrue(pipeline_settings.streaming_enabled)
        self.assertEqual(pipeline_settings.streaming_chunk_interval_s, 0.25)
        self.assertIsInstance(dependencies.worker_runner, _FakeWorkerRunner)
        self.assertIsInstance(
            dependencies.diarization_factory, _FakeDiarizationFactory
        )


if __name__ == '__main__':
    unittest.main()
