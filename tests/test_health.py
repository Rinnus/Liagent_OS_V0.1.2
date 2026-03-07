"""Tests for health check, maybe_gc, and graceful degradation."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class HealthCheckTests(unittest.TestCase):
    def _make_engine(self):
        """Create a minimal EngineManager mock with health_check/maybe_gc."""
        import time
        from liagent.engine.engine_manager import EngineManager

        config = MagicMock()
        config.llm.backend = "api"
        config.llm.api_base_url = "http://test"
        config.llm.api_key = "key"
        config.llm.api_model = "model"
        config.tts_enabled = False
        config.tts = MagicMock()
        config.tts.backend = "api"
        config.stt = MagicMock()
        config.stt.model = "test"
        config.stt.language = "auto"

        with patch("liagent.engine.engine_manager.EngineManager._create_llm") as mock_llm, \
             patch("liagent.engine.engine_manager.STTEngine"):
            mock_llm.return_value = MagicMock()
            engine = EngineManager(config, voice_mode=False)
        return engine

    def test_health_check_returns_dict(self):
        engine = self._make_engine()
        health = engine.health_check()
        self.assertIn("memory_rss_mb", health)
        self.assertIn("llm_loaded", health)
        self.assertIn("reasoning_loaded", health)
        self.assertIn("tts_loaded", health)
        self.assertIn("uptime_seconds", health)
        self.assertIsInstance(health["memory_rss_mb"], float)
        self.assertTrue(health["llm_loaded"])
        self.assertFalse(health["reasoning_loaded"])
        self.assertFalse(health["tts_loaded"])
        self.assertGreaterEqual(health["uptime_seconds"], 0)

    def test_maybe_gc_no_crash(self):
        engine = self._make_engine()
        # Should not crash even with default threshold
        engine.maybe_gc(threshold_mb=999999)  # Very high, no GC triggered
        engine.maybe_gc(threshold_mb=0)  # Very low, GC always triggered


class GracefulDegradationTests(unittest.TestCase):
    def test_generate_reasoning_falls_back_on_file_not_found(self):
        import asyncio

        config = MagicMock()
        config.llm.backend = "api"
        config.llm.api_base_url = "http://test"
        config.llm.api_key = "key"
        config.llm.api_model = "model"
        config.tts_enabled = False
        config.tts = MagicMock()
        config.tts.backend = "api"
        config.stt = MagicMock()
        config.stt.model = "test"
        config.stt.language = "auto"

        with patch("liagent.engine.engine_manager.EngineManager._create_llm") as mock_llm, \
             patch("liagent.engine.engine_manager.STTEngine"):
            mock_llm.return_value = MagicMock()
            from liagent.engine.engine_manager import EngineManager
            engine = EngineManager(config, voice_mode=False)

        engine._ensure_reasoning_llm = MagicMock(
            side_effect=FileNotFoundError("model not found")
        )

        async def fake_generate_llm(messages, **kwargs):
            yield "api_fallback"

        engine._generate_llm_unlocked = fake_generate_llm

        async def run():
            return await engine.generate_reasoning(
                [{"role": "user", "content": "hi"}],
                enable_thinking=False,
            )

        result = asyncio.run(run())
        self.assertEqual(result, "api_fallback")


class StructuredLoggerTests(unittest.TestCase):
    def test_logger_event(self):
        from liagent.logging import get_logger
        log = get_logger("test")
        # Should not crash
        log.event("test_event", key="value")
        log.tool_call("test_tool", {"arg": 1}, "ok", 42.0)
        log.llm_call("model", 100, 50, 500.0)
        log.error("test", ValueError("test error"), context="unit_test")
        log.warning("test warning", extra="data")


if __name__ == "__main__":
    unittest.main()
