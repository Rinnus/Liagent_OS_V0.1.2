"""Tests for config migration logic."""

import unittest

from liagent.config import AppConfig, LLMConfig, TTSConfig, CONFIG_VERSION


class ConfigMigrationTests(unittest.TestCase):
    def test_v1_to_v3_adds_runtime_policy_defaults(self):
        data = {
            "llm": {"backend": "local", "max_tokens": 2048},
            "tts": {"backend": "local"},
            "stt": {"model": "/tmp/stt"},
        }
        migrated = AppConfig._migrate(data)
        self.assertEqual(migrated["config_version"], 3)
        self.assertEqual(migrated["llm"]["model_family"], "glm47")
        self.assertEqual(migrated["llm"]["tool_protocol"], "auto")
        self.assertEqual(migrated["llm"]["api_cache_policy"], "tiered")
        self.assertEqual(migrated["tts"]["tts_engine"], "qwen3")
        self.assertEqual(migrated["stt"]["backend"], "local")
        self.assertEqual(migrated["runtime_mode"], "hybrid_balanced")
        self.assertIn("routing", migrated)
        self.assertIn("sandbox", migrated)
        self.assertIn("budget", migrated)

    def test_v1_api_tts_gets_api_engine(self):
        data = {
            "llm": {},
            "tts": {"backend": "api"},
        }
        migrated = AppConfig._migrate(data)
        self.assertEqual(migrated["tts"]["tts_engine"], "api")

    def test_v2_not_modified(self):
        data = {
            "config_version": 2,
            "llm": {"model_family": "deepseek"},
            "tts": {"tts_engine": "kokoro"},
        }
        migrated = AppConfig._migrate(data)
        self.assertEqual(migrated["llm"]["model_family"], "deepseek")

    def test_config_version_constant(self):
        self.assertEqual(CONFIG_VERSION, 3)

    def test_default_config_has_model_family(self):
        cfg = AppConfig()
        self.assertEqual(cfg.llm.model_family, "glm47")
        self.assertEqual(cfg.llm.tool_protocol, "auto")
        self.assertEqual(cfg.tts.tts_engine, "qwen3")
        self.assertEqual(cfg.stt.backend, "local")
        self.assertEqual(cfg.runtime_mode, "hybrid_balanced")
        self.assertEqual(cfg.repl_mode, "sandboxed")
        self.assertEqual(cfg.config_version, CONFIG_VERSION)


if __name__ == "__main__":
    unittest.main()
