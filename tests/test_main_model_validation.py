"""Tests for startup model-path validation behavior."""

import unittest
from types import SimpleNamespace

from liagent.main import (
    _validate_local_model_paths,
    _bootstrap_missing_local_models_to_api,
)


class MainModelValidationTests(unittest.TestCase):
    def test_stt_api_mode_skips_local_path_check(self):
        cfg = SimpleNamespace(
            llm=SimpleNamespace(backend="api", local_model_path="/missing/llm"),
            tts_enabled=False,
            tts=SimpleNamespace(backend="api", local_model_path="/missing/tts"),
            stt=SimpleNamespace(backend="api", model="/missing/stt"),
        )
        missing = _validate_local_model_paths(cfg)
        self.assertEqual(missing, [])

    def test_stt_local_mode_requires_local_path(self):
        cfg = SimpleNamespace(
            llm=SimpleNamespace(backend="api", local_model_path="/missing/llm"),
            tts_enabled=False,
            tts=SimpleNamespace(backend="api", local_model_path="/missing/tts"),
            stt=SimpleNamespace(backend="local", model="/missing/stt"),
        )
        missing = _validate_local_model_paths(cfg)
        self.assertTrue(any("STT local model not found" in item for item in missing))

    def test_bootstrap_switches_missing_local_models_to_api(self):
        class _Cfg(SimpleNamespace):
            saved = False

            def save(self):
                self.saved = True

        cfg = _Cfg(
            llm=SimpleNamespace(
                backend="local",
                api_base_url="",
                api_key="",
                api_model="",
            ),
            tts_enabled=True,
            tts=SimpleNamespace(
                backend="local",
                api_base_url="",
                api_key="",
                api_model="",
                api_voice="",
            ),
            stt=SimpleNamespace(
                backend="local",
                api_base_url="",
                api_key="",
                api_model="",
            ),
        )
        missing = [
            "LLM local model not found: /missing/llm",
            "TTS local model not found: /missing/tts",
            "STT local model not found: /missing/stt",
        ]

        notes = _bootstrap_missing_local_models_to_api(cfg, missing)

        self.assertEqual(cfg.llm.backend, "api")
        self.assertEqual(cfg.llm.api_base_url, "https://api.openai.com/v1")
        self.assertEqual(cfg.llm.api_model, "gpt-4o")
        self.assertEqual(cfg.tts.backend, "api")
        self.assertEqual(cfg.tts.api_model, "tts-1")
        self.assertEqual(cfg.tts.api_voice, "alloy")
        self.assertEqual(cfg.stt.backend, "api")
        self.assertEqual(cfg.stt.api_model, "gpt-4o-mini-transcribe")
        self.assertTrue(cfg.saved)
        self.assertEqual(len(notes), 3)

    def test_bootstrap_noop_when_no_matching_missing_models(self):
        class _Cfg(SimpleNamespace):
            saved = False

            def save(self):
                self.saved = True

        cfg = _Cfg(
            llm=SimpleNamespace(
                backend="api",
                api_base_url="",
                api_key="",
                api_model="",
            ),
            tts_enabled=False,
            tts=SimpleNamespace(
                backend="api",
                api_base_url="",
                api_key="",
                api_model="",
                api_voice="",
            ),
            stt=SimpleNamespace(
                backend="api",
                api_base_url="",
                api_key="",
                api_model="",
            ),
        )

        notes = _bootstrap_missing_local_models_to_api(cfg, [])
        self.assertEqual(notes, [])
        self.assertFalse(cfg.saved)


if __name__ == "__main__":
    unittest.main()
