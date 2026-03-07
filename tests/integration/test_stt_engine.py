"""Tests for STT engine local/API routing."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from liagent.engine.stt import STTEngine


class STTEngineApiTests(unittest.TestCase):
    def test_api_backend_transcribe_numpy(self):
        engine = STTEngine(
            model="",
            language="auto",
            backend="api",
            api_base_url="http://test",
            api_key="key",
            api_model="gpt-4o-mini-transcribe",
        )

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=SimpleNamespace(text="hello world")
        )

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            out = asyncio.run(engine.transcribe(np.zeros(1600, dtype=np.float32)))

        self.assertEqual(out, "hello world")
        call = mock_client.audio.transcriptions.create.await_args
        self.assertEqual(call.kwargs.get("model"), "gpt-4o-mini-transcribe")
        self.assertIsNone(call.kwargs.get("language"))
        self.assertIn("file", call.kwargs)

    def test_api_backend_requires_model(self):
        engine = STTEngine(
            model="",
            language="auto",
            backend="api",
            api_base_url="http://test",
            api_key="key",
            api_model="",
        )
        with self.assertRaises(ValueError):
            asyncio.run(engine.transcribe(np.zeros(16, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
