"""Tests for LLM switching normalization and parser sync."""

import asyncio
import os
import unittest
from unittest.mock import patch

from liagent.config import AppConfig, LLMConfig
from liagent.engine.engine_manager import EngineManager
from liagent.engine.tool_format import OpenAIFormat, GLM47Format


class _FakeLLM:
    def __init__(self):
        self.unloaded = False

    async def generate(self, *args, **kwargs):
        if False:
            yield ""

    def unload(self):
        self.unloaded = True

    def is_loaded(self):
        return True


class _FakeSTT:
    def unload(self):
        return None


class _UsageLLM(_FakeLLM):
    def __init__(self, text: str, usage: dict[str, int | str | bool], *, fail: bool = False):
        super().__init__()
        self._text = text
        self.last_usage = dict(usage)
        self._fail = fail

    async def generate(self, *args, **kwargs):
        if self._fail:
            raise RuntimeError("forced failure")
        yield self._text


class EngineManagerLLMSwitchTests(unittest.TestCase):
    def test_init_normalizes_api_family_from_default(self):
        cfg = AppConfig()
        cfg.tts_enabled = False
        cfg.llm = LLMConfig(
            backend="api",
            model_family="glm47",  # default value should be replaced for API mode
            api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key="test-key",
            api_model="gemini-3.0-flash",
        )

        with patch.object(EngineManager, "_create_llm", return_value=_FakeLLM()), \
             patch.object(EngineManager, "_create_stt", return_value=_FakeSTT()), \
             patch.object(AppConfig, "save", return_value=None):
            mgr = EngineManager(cfg)

        self.assertEqual(mgr.config.llm.model_family, "openai")
        self.assertEqual(mgr.config.llm.tool_protocol, "openai_function")
        self.assertEqual(mgr.config.llm.api_cache_mode, "implicit")
        self.assertIsInstance(mgr.tool_parser, OpenAIFormat)

    def test_init_keeps_explicit_api_family(self):
        cfg = AppConfig()
        cfg.tts_enabled = False
        cfg.llm = LLMConfig(
            backend="api",
            model_family="deepseek",
            api_base_url="https://example.com/v1",
            api_key="test-key",
            api_model="deepseek-chat",
        )

        with patch.object(EngineManager, "_create_llm", return_value=_FakeLLM()), \
             patch.object(EngineManager, "_create_stt", return_value=_FakeSTT()), \
             patch.object(AppConfig, "save", return_value=None):
            mgr = EngineManager(cfg)

        self.assertEqual(mgr.config.llm.model_family, "deepseek")
        self.assertEqual(mgr.config.llm.tool_protocol, "openai_function")
        self.assertIsInstance(mgr.tool_parser, OpenAIFormat)

    def test_switch_llm_refreshes_tool_parser(self):
        cfg = AppConfig()
        cfg.tts_enabled = False

        old_llm = _FakeLLM()
        new_llm = _FakeLLM()

        with patch.object(EngineManager, "_create_llm", side_effect=[old_llm, new_llm]), \
             patch.object(EngineManager, "_create_stt", return_value=_FakeSTT()), \
             patch.object(AppConfig, "save", return_value=None):
            mgr = EngineManager(cfg)
            self.assertIsInstance(mgr.tool_parser, GLM47Format)

            mgr.switch_llm(
                LLMConfig(
                    backend="api",
                    model_family="glm47",
                    api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    api_key="test-key",
                    api_model="gemini-3.0-flash",
                )
            )

        self.assertTrue(old_llm.unloaded)
        self.assertEqual(mgr.config.llm.model_family, "openai")
        self.assertEqual(mgr.config.llm.tool_protocol, "openai_function")
        self.assertIsInstance(mgr.tool_parser, OpenAIFormat)

    def test_normalize_llm_cache_fields(self):
        cfg = LLMConfig(
            backend="api",
            model_family="openai",
            api_base_url="https://example.com/v1",
            api_key="k",
            api_model="gpt-4o-mini",
            api_cache_mode="invalid",
            api_cache_ttl_sec=10,
        )
        normalized = EngineManager._normalize_llm_config(cfg)
        self.assertEqual(normalized.api_cache_mode, "implicit")
        self.assertEqual(normalized.api_cache_policy, "tiered")
        self.assertGreaterEqual(normalized.api_cache_ttl_sec, 60)
        self.assertGreaterEqual(normalized.api_cache_ttl_static_sec, normalized.api_cache_ttl_sec)
        self.assertGreaterEqual(normalized.api_cache_ttl_memory_sec, 60)

    def test_local_tool_protocol_infers_native_xml(self):
        cfg = LLMConfig(
            backend="local",
            model_family="glm47",
            tool_protocol="auto",
        )
        normalized = EngineManager._normalize_llm_config(cfg)
        self.assertEqual(normalized.tool_protocol, "native_xml")


class EngineManagerUsageCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, it):
        chunks = []
        async for tok in it:
            chunks.append(tok)
        return "".join(chunks)

    async def test_routed_prefers_api_captures_usage(self):
        cfg = AppConfig()
        cfg.tts_enabled = False
        local = _UsageLLM(
            "local",
            {"provider": "local", "prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        )
        api = _UsageLLM(
            "api",
            {
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
                "cached_prompt_tokens": 6,
            },
        )

        with patch.object(EngineManager, "_create_llm", return_value=local), \
             patch.object(EngineManager, "_create_stt", return_value=_FakeSTT()), \
             patch.object(AppConfig, "save", return_value=None):
            mgr = EngineManager(cfg)

        mgr._llm_fallback = api
        mgr._can_fallback_api = lambda: True
        mgr.reset_llm_usage_counters()

        with patch.dict(os.environ, {"LIAGENT_ACTIVE_MODEL_ROUTING": "true"}):
            out = await self._collect(
                mgr.generate_llm_routed(
                    [{"role": "user", "content": "deep task"}],
                    service_tier="deep_task",
                    planning_enabled=True,
                )
            )
        self.assertEqual(out, "api")
        self.assertEqual(mgr.get_last_llm_usage().get("total_tokens"), 20)
        self.assertEqual(mgr.get_last_llm_usage().get("cached_prompt_tokens"), 6)
        self.assertGreater(float(mgr.get_last_llm_usage().get("estimated_cost_usd", 0.0) or 0.0), 0.0)
        self.assertEqual(mgr.get_cumulative_llm_usage().get("total_tokens"), 20)
        self.assertEqual(mgr.get_cumulative_llm_usage().get("cached_prompt_tokens"), 6)

    async def test_routed_prefers_local_captures_usage(self):
        cfg = AppConfig()
        cfg.tts_enabled = False
        local = _UsageLLM(
            "local",
            {"provider": "local", "prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
        )
        api = _UsageLLM(
            "api",
            {"provider": "openai", "prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

        with patch.object(EngineManager, "_create_llm", return_value=local), \
             patch.object(EngineManager, "_create_stt", return_value=_FakeSTT()), \
             patch.object(AppConfig, "save", return_value=None):
            mgr = EngineManager(cfg)

        mgr._llm_fallback = api
        mgr._can_fallback_api = lambda: True
        mgr.reset_llm_usage_counters()

        with patch.dict(os.environ, {"LIAGENT_ACTIVE_MODEL_ROUTING": "true"}):
            out = await self._collect(
                mgr.generate_llm_routed(
                    [{"role": "user", "content": "quick"}],
                    service_tier="standard_chat",
                    planning_enabled=False,
                )
            )
        self.assertEqual(out, "local")
        self.assertEqual(mgr.get_last_llm_usage().get("provider"), "local")
        self.assertEqual(mgr.get_cumulative_llm_usage().get("total_tokens"), 10)
        self.assertEqual(float(mgr.get_last_llm_usage().get("estimated_cost_usd", 0.0) or 0.0), 0.0)

    async def test_generate_reasoning_api_backend_skips_local_reasoning_model(self):
        cfg = AppConfig()
        cfg.tts_enabled = False
        cfg.llm = LLMConfig(
            backend="api",
            model_family="openai",
            api_base_url="https://api.openai.com/v1",
            api_key="k",
            api_model="gpt-4o-mini",
        )
        api = _UsageLLM(
            "api-reasoning",
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_tokens": 8,
                "completion_tokens": 6,
                "total_tokens": 14,
            },
        )

        with patch.object(EngineManager, "_create_llm", return_value=api), \
             patch.object(EngineManager, "_create_stt", return_value=_FakeSTT()), \
             patch.object(AppConfig, "save", return_value=None):
            mgr = EngineManager(cfg)

        with patch.object(
            mgr,
            "_ensure_reasoning_llm",
            side_effect=AssertionError("local reasoning path should not be used"),
        ):
            out = await mgr.generate_reasoning(
                [{"role": "user", "content": "hello"}],
                max_tokens=128,
                temperature=0.2,
                enable_thinking=False,
            )
        self.assertEqual(out, "api-reasoning")
