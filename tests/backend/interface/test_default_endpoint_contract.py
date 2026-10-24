from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from urllib.parse import urlparse

from application.codex_config import DEFAULT_CODEX_PROXY
from infrastructure.local_llm import (
    OLLAMA_DEFAULT_URL,
    OPENAI_LOCAL_DEFAULT_URL,
)
from tests import PROJECT_ROOT


@unittest.skipUnless(shutil.which('node'), 'Node.js is required')
class DefaultEndpointContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        script = """
import { ASSISTANT_PROVIDER_OPTIONS } from
  './frontend/renderer/src/entities/settings/modelParts/constants.js';
import { parseProxy } from
  './frontend/renderer/src/entities/settings/modelParts/helpers.js';
console.log(JSON.stringify({
  providers: Object.fromEntries(ASSISTANT_PROVIDER_OPTIONS.map(
    (option) => [option.id, option.defaultBaseUrl],
  )),
  proxy: parseProxy(''),
}));
"""
        output = subprocess.check_output(
            ['node', '--input-type=module', '-e', script],
            cwd=PROJECT_ROOT,
            text=True,
            encoding='utf-8',
            timeout=10,
        )
        cls.frontend_defaults = json.loads(output)

    def test_frontend_local_llm_defaults_match_backend_defaults(self) -> None:
        providers = self.frontend_defaults['providers']
        self.assertEqual(providers['ollama'], OLLAMA_DEFAULT_URL)
        self.assertEqual(providers['openai_local'], OPENAI_LOCAL_DEFAULT_URL)

    def test_frontend_proxy_draft_matches_backend_default_proxy_endpoint(
        self,
    ) -> None:
        parsed = urlparse(DEFAULT_CODEX_PROXY)
        proxy = self.frontend_defaults['proxy']
        self.assertEqual(proxy['host'], parsed.hostname)
        self.assertEqual(proxy['port'], str(parsed.port))


if __name__ == '__main__':
    unittest.main()
