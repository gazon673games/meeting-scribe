from __future__ import annotations

import re
import unittest
from pathlib import Path
from urllib.parse import urlparse

from application.codex_config import DEFAULT_CODEX_PROXY
from infrastructure.local_llm import OLLAMA_DEFAULT_URL, OPENAI_LOCAL_DEFAULT_URL


ROOT = Path(__file__).resolve().parents[1]


def _frontend_provider_default(provider: str) -> str:
    text = (ROOT / "frontend/renderer/src/entities/settings/modelParts/constants.js").read_text(encoding="utf-8")
    match = re.search(rf'id:\s*"{re.escape(provider)}".*?defaultBaseUrl:\s*"([^"]*)"', text, re.S)
    if not match:
        raise AssertionError(f"Missing frontend defaultBaseUrl for {provider}")
    return match.group(1)


class DefaultEndpointContractTests(unittest.TestCase):
    def test_frontend_local_llm_defaults_match_backend_defaults(self) -> None:
        self.assertEqual(_frontend_provider_default("ollama"), OLLAMA_DEFAULT_URL)
        self.assertEqual(_frontend_provider_default("openai_local"), OPENAI_LOCAL_DEFAULT_URL)

    def test_frontend_proxy_draft_matches_backend_default_proxy_endpoint(self) -> None:
        parsed = urlparse(DEFAULT_CODEX_PROXY)
        text = (ROOT / "frontend/renderer/src/entities/settings/modelParts/helpers.js").read_text(encoding="utf-8")

        self.assertIn(f'host: "{parsed.hostname}"', text)
        self.assertIn(f'port: "{parsed.port}"', text)


if __name__ == "__main__":
    unittest.main()
