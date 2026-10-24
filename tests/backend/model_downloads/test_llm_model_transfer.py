from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from application import diarization_model_download
from infrastructure import http_transport
from infrastructure import llm_model_transfer as transfer


class _Response(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data))}


class LlmModelTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        headers = patch(
            'huggingface_hub.utils.build_hf_headers',
            return_value={'authorization': 'Bearer test-token'},
        )
        offline = patch('huggingface_hub.constants.HF_HUB_OFFLINE', False)
        headers.start()
        offline.start()
        self.addCleanup(headers.stop)
        self.addCleanup(offline.stop)

    def test_concurrent_downloads_keep_proxy_configuration_local(self) -> None:
        barrier = threading.Barrier(2)

        def opener(handler):
            proxy = handler.proxies['https']

            def open_request(request, *, timeout):
                barrier.wait(timeout=2)
                self.assertEqual(os.environ['HTTP_PROXY'], 'ambient')
                return _Response(proxy.encode('ascii'))

            return SimpleNamespace(open=open_request)

        with (
            tempfile.TemporaryDirectory() as root,
            patch.dict(os.environ, {'HTTP_PROXY': 'ambient'}),
            patch.object(http_transport, 'build_opener', side_effect=opener),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            targets = [Path(root) / f'{index}.gguf' for index in range(2)]
            futures = [
                executor.submit(
                    transfer._download_url,
                    'https://example.test/model.gguf',
                    targets[0],
                    proxy='http://proxy-0',
                    on_progress=lambda update: None,
                ),
                executor.submit(
                    diarization_model_download._download_file,
                    url='https://example.test/model.onnx',
                    target=targets[1],
                    proxy='http://proxy-1',
                    on_progress=lambda update: None,
                ),
            ]
            for future in futures:
                future.result(timeout=3)
            self.assertEqual(
                [target.read_bytes() for target in targets],
                [b'http://proxy-0', b'http://proxy-1'],
            )
            self.assertEqual(os.environ['HTTP_PROXY'], 'ambient')

    def test_repository_metadata_uses_requested_proxy_and_authentication(
        self,
    ) -> None:
        payload = {'siblings': [{'rfilename': 'model.gguf'}, {}, None]}
        with (
            patch.dict(
                os.environ,
                {'HF_TOKEN': 'test-token', 'HF_ENDPOINT': 'https://hf.test'},
            ),
            patch.object(
                transfer,
                '_open_model_url',
                return_value=_Response(json.dumps(payload).encode()),
            ) as open_url,
        ):
            files = transfer.repository_files(
                'owner/repo', proxy='http://proxy'
            )
        self.assertEqual(files, ['model.gguf'])
        request = open_url.call_args.args[0]
        self.assertEqual(
            request.full_url, 'https://hf.test/api/models/owner/repo'
        )
        self.assertEqual(
            request.get_header('Authorization'), 'Bearer test-token'
        )
        self.assertEqual(
            open_url.call_args.kwargs, {'proxy': 'http://proxy', 'timeout': 30}
        )

    def test_invalid_repository_metadata_is_reported(self) -> None:
        for payload in [None, [], {}, {'siblings': {}}]:
            with (
                self.subTest(payload=payload),
                patch.object(
                    transfer,
                    '_open_model_url',
                    return_value=_Response(json.dumps(payload).encode()),
                ),
            ):
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    transfer.repository_files('owner/repo', proxy='')

    def test_repository_metadata_preserves_sdk_authentication_headers(
        self,
    ) -> None:
        with (
            patch(
                'huggingface_hub.utils.build_hf_headers',
                return_value={'authorization': 'Bearer cached-token'},
            ),
            patch.object(
                transfer,
                '_open_model_url',
                return_value=_Response(b'{"siblings":[]}'),
            ) as open_url,
        ):
            self.assertEqual(
                transfer.repository_files('owner/repo', proxy=''), []
            )
        request = open_url.call_args.args[0]
        self.assertEqual(
            request.get_header('Authorization'), 'Bearer cached-token'
        )

    def test_offline_metadata_lookup_does_not_open_a_connection(self) -> None:
        from huggingface_hub.errors import OfflineModeIsEnabled

        with (
            patch('huggingface_hub.constants.HF_HUB_OFFLINE', True),
            patch.object(transfer, '_open_model_url') as open_url,
        ):
            with self.assertRaises(OfflineModeIsEnabled):
                transfer.repository_files('owner/repo', proxy='http://proxy')
        open_url.assert_not_called()
