"""Tests for API VLM streaming behavior."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from liagent.engine.vlm_api import ApiVLM


class _AsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._idx = 0

    def __aiter__(self):
        self._idx = 0
        return self

    async def __anext__(self):
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        item = self._chunks[self._idx]
        self._idx += 1
        return item


def _chunk(*, content=None, tool_calls=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


def _tool_chunk(*, index=0, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, function=fn)


class ApiVLMTests(unittest.IsolatedAsyncioTestCase):
    def test_build_messages_maps_tool_observation_for_api_compatibility(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            out = llm._build_messages(
                [
                    {"role": "system", "content": "sys"},
                    {"role": "assistant", "content": ""},
                    {"role": "tool", "content": "price=100"},
                    {"role": "user", "content": "summarize"},
                ],
                images=None,
            )
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[1]["role"], "user")
        self.assertIn("[Tool Result]", out[1]["content"])
        self.assertEqual(out[2]["role"], "user")

    def test_build_messages_preserves_assistant_tool_calls(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            out = llm._build_messages(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "web_search", "arguments": {"query": "x"}}],
                    },
                    {"role": "user", "content": "continue"},
                ],
                images=None,
            )
        self.assertEqual(out[0]["role"], "assistant")
        self.assertIn("tool_calls", out[0])
        self.assertTrue(str(out[0].get("content", "")).strip())

    def test_build_messages_pairs_tool_messages_with_tool_call_id(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            out = llm._build_messages(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "web_search", "arguments": {"query": "sf tech events"}}],
                    },
                    {"role": "tool", "content": "result list"},
                    {"role": "user", "content": "summarize"},
                ],
                images=None,
            )
        self.assertEqual(out[0]["role"], "assistant")
        self.assertIn("tool_calls", out[0])
        tcid = out[0]["tool_calls"][0]["id"]
        self.assertEqual(out[1]["role"], "tool")
        self.assertEqual(out[1]["tool_call_id"], tcid)
        self.assertIn("result list", out[1]["content"])
        self.assertEqual(out[2]["role"], "user")

    def test_build_messages_backfills_missing_tool_reply_for_pending_call(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            out = llm._build_messages(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "web_search", "arguments": {"query": "GOOG latest"}}],
                    },
                    {"role": "user", "content": "continue"},
                ],
                images=None,
            )
        self.assertEqual(out[0]["role"], "assistant")
        self.assertEqual(out[1]["role"], "tool")
        self.assertTrue(str(out[1].get("tool_call_id", "")).strip())
        self.assertIn("unavailable", out[1]["content"])
        self.assertEqual(out[2]["role"], "user")

    def test_build_messages_adds_reasoning_content_for_moonshot_tool_history(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://api.moonshot.cn/v1", "k", "kimi-k2.5")
            out = llm._build_messages(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "web_search", "arguments": {"query": "sf forums"}}],
                    },
                    {"role": "tool", "content": "list"},
                ],
                images=None,
            )
        self.assertEqual(out[0]["role"], "assistant")
        self.assertIn("tool_calls", out[0])
        self.assertIn("reasoning_content", out[0])
        self.assertTrue(str(out[0]["reasoning_content"]).strip())

    def test_moonshot_scope_temperature_hint_uses_0_6_for_disable_thinking(self):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://api.moonshot.cn/v1", "k", "kimi-k2.5")
            normal = llm._build_request_kwargs(
                api_messages=[{"role": "user", "content": "hello"}],
                max_tokens=128,
                temperature=0.3,
                tools=None,
                stream=True,
                disable_thinking=False,
            )
            extract = llm._build_request_kwargs(
                api_messages=[{"role": "user", "content": "extract"}],
                max_tokens=128,
                temperature=0.1,
                tools=None,
                stream=True,
                disable_thinking=True,
            )
        self.assertEqual(float(normal["temperature"]), 1.0)
        self.assertEqual(float(extract["temperature"]), 0.6)

    async def test_generate_converts_stream_tool_calls_to_xml_block(self):
        chunks = [
            _chunk(tool_calls=[_tool_chunk(index=0, name="web_search", arguments='{"query":"Gemini')]),
            _chunk(tool_calls=[_tool_chunk(index=0, arguments=' API"}')]),
        ]
        create_mock = AsyncMock(return_value=_AsyncStream(chunks))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )

        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "gemini-3.0-flash")
            tokens = []
            async for tok in llm.generate(
                [{"role": "user", "content": "search"}],
                tools=[{"type": "function", "function": {"name": "web_search"}}],
            ):
                tokens.append(tok)

        merged = "".join(tokens)
        self.assertIn("<tool_call>", merged)
        payload = merged.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
        obj = json.loads(payload)
        self.assertEqual(obj["name"], "web_search")
        self.assertEqual(obj["args"].get("query"), "Gemini API")
        self.assertEqual(create_mock.await_count, 1)
        self.assertIn("tools", create_mock.await_args.kwargs)

    async def test_generate_emits_multiple_tool_blocks(self):
        chunks = [
            _chunk(tool_calls=[
                _tool_chunk(index=0, name="web_search", arguments='{"query":"x"}'),
                _tool_chunk(index=1, name="stock", arguments='{"symbol":"AAPL"}'),
            ]),
        ]
        create_mock = AsyncMock(return_value=_AsyncStream(chunks))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )

        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            tokens = []
            async for tok in llm.generate(
                [{"role": "user", "content": "run tools"}],
                tools=[{"type": "function", "function": {"name": "web_search"}}],
            ):
                tokens.append(tok)

        merged = "".join(tokens)
        self.assertEqual(merged.count("<tool_call>"), 2)
        self.assertIn('"name": "web_search"', merged)
        self.assertIn('"name": "stock"', merged)

    async def test_generate_progressively_downgrades_tool_request_shape(self):
        create_mock = AsyncMock(
            side_effect=[
                RuntimeError("tool_choice parameter unsupported"),
                RuntimeError("tools parameter unsupported"),
                _AsyncStream([_chunk(content="ok")]),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )

        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            tokens = []
            async for tok in llm.generate(
                [{"role": "user", "content": "hello"}],
                    tools=[{"type": "function", "function": {"name": "web_search"}}],
                ):
                    tokens.append(tok)

        self.assertEqual("".join(tokens), "ok")
        self.assertEqual(create_mock.await_count, 3)
        self.assertIn("tools", create_mock.await_args_list[0].kwargs)
        self.assertIn("tool_choice", create_mock.await_args_list[0].kwargs)
        self.assertIn("tools", create_mock.await_args_list[1].kwargs)
        self.assertNotIn("tool_choice", create_mock.await_args_list[1].kwargs)
        self.assertNotIn("tools", create_mock.await_args_list[2].kwargs)

    async def test_tool_runtime_downgrade_persists_across_requests(self):
        create_mock = AsyncMock(
            side_effect=[
                RuntimeError("tool_choice parameter unsupported"),
                _AsyncStream([_chunk(content="ok1")]),
                _AsyncStream([_chunk(content="ok2")]),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            async for _ in llm.generate(
                [{"role": "user", "content": "hello"}],
                tools=[{"type": "function", "function": {"name": "web_search"}}],
            ):
                pass
            async for _ in llm.generate(
                [{"role": "user", "content": "hello again"}],
                tools=[{"type": "function", "function": {"name": "web_search"}}],
            ):
                pass
        self.assertEqual(create_mock.await_count, 3)
        self.assertIn("tool_choice", create_mock.await_args_list[0].kwargs)
        self.assertNotIn("tool_choice", create_mock.await_args_list[1].kwargs)
        # Persisted runtime capability should keep tool_choice disabled on later requests.
        self.assertNotIn("tool_choice", create_mock.await_args_list[2].kwargs)

    async def test_generate_retries_on_retryable_error(self):
        create_mock = AsyncMock(
            side_effect=[RuntimeError("429 rate limit exceeded"), _AsyncStream([_chunk(content="ok")])]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client), \
             patch("liagent.engine.vlm_api.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            tokens = []
            async for tok in llm.generate([{"role": "user", "content": "hello"}]):
                tokens.append(tok)
        self.assertEqual("".join(tokens), "ok")
        self.assertEqual(create_mock.await_count, 2)
        self.assertGreaterEqual(sleep_mock.await_count, 1)

    async def test_generate_falls_back_to_non_stream_when_stream_fails_early(self):
        stream_err = RuntimeError("stream connection reset")
        non_stream_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="fallback ok", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        create_mock = AsyncMock(side_effect=[stream_err, non_stream_resp])
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            tokens = []
            async for tok in llm.generate([{"role": "user", "content": "hello"}]):
                tokens.append(tok)
        self.assertEqual("".join(tokens), "fallback ok")
        self.assertEqual(create_mock.await_count, 2)
        self.assertEqual(llm.last_usage["total_tokens"], 15)
        self.assertFalse(llm.last_usage["estimated"])

    async def test_circuit_breaker_opens_after_repeated_failures(self):
        create_mock = AsyncMock(side_effect=RuntimeError("503 service unavailable"))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch.dict(
            "os.environ",
            {
                "LIAGENT_API_RETRY_ATTEMPTS": "1",
                "LIAGENT_API_CIRCUIT_BREAKER_THRESHOLD": "2",
                "LIAGENT_API_CIRCUIT_COOLDOWN_SEC": "30",
            },
            clear=False,
        ), patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            with self.assertRaises(RuntimeError):
                async for _ in llm.generate([{"role": "user", "content": "a"}]):
                    pass
            with self.assertRaises(RuntimeError):
                async for _ in llm.generate([{"role": "user", "content": "b"}]):
                    pass
            # Circuit is open now: should fail fast without extra provider call.
            with self.assertRaises(RuntimeError):
                async for _ in llm.generate([{"role": "user", "content": "c"}]):
                    pass
            self.assertEqual(create_mock.await_count, 2)

    async def test_circuit_breaker_cooldown_uses_exponential_backoff_with_cap(self):
        create_mock = AsyncMock(side_effect=RuntimeError("503 service unavailable"))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch.dict(
            "os.environ",
            {
                "LIAGENT_API_CIRCUIT_BREAKER_THRESHOLD": "2",
                "LIAGENT_API_CIRCUIT_COOLDOWN_SEC": "12",
                "LIAGENT_API_CIRCUIT_COOLDOWN_MAX_SEC": "48",
            },
            clear=False,
        ), patch("openai.AsyncOpenAI", return_value=fake_client), patch(
            "liagent.engine.vlm_api.time.time", return_value=100.0
        ):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            llm._record_failure()  # streak=1, below threshold
            self.assertEqual(llm._circuit_open_until, 0.0)
            llm._record_failure()  # streak=2 -> 12s
            self.assertEqual(llm._circuit_open_until, 112.0)
            llm._record_failure()  # streak=3 -> 24s
            self.assertEqual(llm._circuit_open_until, 124.0)
            llm._record_failure()  # streak=4 -> 48s
            self.assertEqual(llm._circuit_open_until, 148.0)
            llm._record_failure()  # streak=5 -> 96s but capped at 48s
            self.assertEqual(llm._circuit_open_until, 148.0)

    async def test_gemini_profile_sets_provider_specific_request_shape(self):
        create_mock = AsyncMock(return_value=_AsyncStream([_chunk(content="ok")]))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM(
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "k",
                "gemini-3.0-flash",
            )
            async for _ in llm.generate(
                [{"role": "user", "content": "hello"}],
                tools=[{"type": "function", "function": {"name": "web_search"}}],
            ):
                pass
        kwargs = create_mock.await_args.kwargs
        self.assertEqual(llm.last_usage["provider"], "gemini")
        self.assertIn("max_tokens", kwargs)
        self.assertIn("parallel_tool_calls", kwargs)
        self.assertFalse(kwargs["parallel_tool_calls"])
        self.assertNotIn("tool_choice", kwargs)

    async def test_moonshot_model_is_inferred_with_openai_shape(self):
        create_mock = AsyncMock(return_value=_AsyncStream([_chunk(content="ok")]))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM(
                "https://openrouter.ai/api/v1",
                "k",
                "moonshotai/kimi-k2.5",
            )
            async for _ in llm.generate(
                [{"role": "user", "content": "hello"}],
                tools=[{"type": "function", "function": {"name": "web_search"}}],
            ):
                pass
        kwargs = create_mock.await_args.kwargs
        self.assertEqual(llm.last_usage["provider"], "moonshot")
        self.assertIn("max_tokens", kwargs)
        self.assertIn("tool_choice", kwargs)
        self.assertEqual(float(kwargs.get("temperature")), 1.0)

    async def test_stream_options_runtime_downgrade_persists(self):
        create_mock = AsyncMock(
            side_effect=[
                RuntimeError("stream_options parameter unsupported"),
                _AsyncStream([_chunk(content="ok")]),
                _AsyncStream([_chunk(content="ok2")]),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            async for _ in llm.generate([{"role": "user", "content": "hello"}]):
                pass
            async for _ in llm.generate([{"role": "user", "content": "hello2"}]):
                pass
        self.assertEqual(create_mock.await_count, 3)
        self.assertIn("stream_options", create_mock.await_args_list[0].kwargs)
        self.assertNotIn("stream_options", create_mock.await_args_list[1].kwargs)
        self.assertNotIn("stream_options", create_mock.await_args_list[2].kwargs)

    async def test_runtime_switches_max_tokens_field_when_provider_rejects_default(self):
        create_mock = AsyncMock(
            side_effect=[
                RuntimeError("max_completion_tokens is unsupported"),
                _AsyncStream([_chunk(content="ok")]),
                _AsyncStream([_chunk(content="ok2")]),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://api.anthropic.com/v1", "k", "claude-3-5-sonnet")
            async for _ in llm.generate([{"role": "user", "content": "hello"}]):
                pass
            async for _ in llm.generate([{"role": "user", "content": "hello2"}]):
                pass
        self.assertEqual(create_mock.await_count, 3)
        self.assertIn("max_completion_tokens", create_mock.await_args_list[0].kwargs)
        self.assertIn("max_tokens", create_mock.await_args_list[1].kwargs)
        self.assertIn("max_tokens", create_mock.await_args_list[2].kwargs)
        self.assertNotIn("max_completion_tokens", create_mock.await_args_list[2].kwargs)

    async def test_runtime_clamps_temperature_when_model_requires_fixed_value(self):
        create_mock = AsyncMock(
            side_effect=[
                RuntimeError("invalid temperature: only 1 is allowed for this model"),
                _AsyncStream([_chunk(content="ok")]),
                _AsyncStream([_chunk(content="ok2")]),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            async for _ in llm.generate([{"role": "user", "content": "hello"}]):
                pass
            async for _ in llm.generate([{"role": "user", "content": "hello2"}]):
                pass
        self.assertEqual(create_mock.await_count, 3)
        self.assertEqual(float(create_mock.await_args_list[0].kwargs.get("temperature")), 0.7)
        self.assertEqual(float(create_mock.await_args_list[1].kwargs.get("temperature")), 1.0)
        self.assertEqual(float(create_mock.await_args_list[2].kwargs.get("temperature")), 1.0)

    async def test_runtime_temperature_clamp_is_isolated_by_request_shape(self):
        create_mock = AsyncMock(
            side_effect=[
                _AsyncStream([_chunk(content="chat")]),
                RuntimeError("invalid temperature: only 0.6 is allowed for this model"),
                _AsyncStream([_chunk(content="extract")]),
                _AsyncStream([_chunk(content="chat2")]),
                _AsyncStream([_chunk(content="extract2")]),
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            async for _ in llm.generate([{"role": "user", "content": "hello"}]):
                pass
            async for _ in llm.generate(
                [{"role": "user", "content": "extract"}],
                disable_thinking=True,
            ):
                pass
            async for _ in llm.generate([{"role": "user", "content": "hello again"}]):
                pass
            async for _ in llm.generate(
                [{"role": "user", "content": "extract again"}],
                disable_thinking=True,
            ):
                pass
        self.assertEqual(create_mock.await_count, 5)
        self.assertEqual(float(create_mock.await_args_list[0].kwargs.get("temperature")), 0.7)
        self.assertEqual(float(create_mock.await_args_list[1].kwargs.get("temperature")), 0.7)
        self.assertEqual(float(create_mock.await_args_list[2].kwargs.get("temperature")), 0.6)
        self.assertEqual(float(create_mock.await_args_list[3].kwargs.get("temperature")), 0.7)
        self.assertEqual(float(create_mock.await_args_list[4].kwargs.get("temperature")), 0.6)

    async def test_stream_usage_is_captured(self):
        chunks = [
            _chunk(content="hello"),
            _chunk(content=" world", usage=SimpleNamespace(prompt_tokens=20, completion_tokens=8, total_tokens=28)),
        ]
        create_mock = AsyncMock(return_value=_AsyncStream(chunks))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            out = []
            async for tok in llm.generate([{"role": "user", "content": "usage test"}]):
                out.append(tok)
        self.assertEqual("".join(out), "hello world")
        self.assertEqual(llm.last_usage["total_tokens"], 28)
        self.assertFalse(llm.last_usage["estimated"])

    async def test_stream_usage_captures_cached_prompt_tokens(self):
        chunks = [
            _chunk(
                content="cached",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "prompt_tokens_details": {"cached_tokens": 70},
                },
            ),
        ]
        create_mock = AsyncMock(return_value=_AsyncStream(chunks))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM("https://example.com/v1", "k", "model-x")
            async for _ in llm.generate([{"role": "user", "content": "cache test"}]):
                pass
        self.assertEqual(llm.last_usage["cached_prompt_tokens"], 70)
        self.assertAlmostEqual(float(llm.last_usage["cache_hit_ratio"]), 0.7, places=3)

    async def test_explicit_cache_adds_openrouter_headers_and_body(self):
        create_mock = AsyncMock(return_value=_AsyncStream([_chunk(content="ok")]))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM(
                "https://openrouter.ai/api/v1",
                "k",
                "gemini-3.0-flash",
                cache_mode="explicit",
                cache_ttl_sec=900,
            )
            async for _ in llm.generate([{"role": "user", "content": "hello"}]):
                pass
        kwargs = create_mock.await_args.kwargs
        self.assertIn("extra_headers", kwargs)
        self.assertEqual(
            kwargs["extra_headers"].get("x-openrouter-cache-control"),
            "max-age=900",
        )
        self.assertIn("extra_body", kwargs)
        self.assertIn("cache_control", kwargs["extra_body"])

    async def test_tiered_cache_uses_longer_ttl_for_static_prefix(self):
        create_mock = AsyncMock(return_value=_AsyncStream([_chunk(content="ok")]))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with patch("openai.AsyncOpenAI", return_value=fake_client):
            llm = ApiVLM(
                "https://openrouter.ai/api/v1",
                "k",
                "gemini-3.0-flash",
                cache_mode="explicit",
                cache_policy="tiered",
                cache_ttl_sec=120,
                cache_ttl_static_sec=1800,
                cache_ttl_memory_sec=600,
                cache_min_prefix_chars=100,
            )
            async for _ in llm.generate(
                [
                    {"role": "system", "content": "S" * 260},
                    {"role": "user", "content": "hello"},
                ]
            ):
                pass
        kwargs = create_mock.await_args.kwargs
        self.assertEqual(
            kwargs.get("extra_headers", {}).get("x-openrouter-cache-control"),
            "max-age=1800",
        )
        self.assertEqual(
            int(kwargs.get("extra_body", {}).get("cache_control", {}).get("ttl_seconds", 0)),
            1800,
        )
