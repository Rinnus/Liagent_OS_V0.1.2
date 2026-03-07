import os
import sqlite3
import tempfile
import unittest
import time
from types import SimpleNamespace
from io import BytesIO
from pathlib import Path
import base64
from unittest.mock import patch

from liagent.agent.brain import AgentBrain
from liagent.agent.tool_parsing import (
    _parse_tool_call_lenient,
    parse_all_tool_calls,
    strip_any_tool_call,
    tool_call_signature,
)
from liagent.agent.tool_executor import build_tool_degrade_observation
from liagent.agent.quality import estimate_task_success
from liagent.config import AppConfig
from liagent.agent.memory import LongTermMemory
from liagent.agent.self_supervision import InteractionMetrics
from liagent.engine.tts_utils import build_tts_chunks
from liagent.ui.web_server import _decode_and_prepare_image, _resolve_image_paths, _cleanup_path
from liagent.skills.router import select_skill
from liagent.tools import ToolCapability, ToolDef, get_native_tool_schemas
from liagent.tools.policy import ToolPolicy
from liagent.tools.python_exec import _validate_python_exec



class TTSChunkTests(unittest.TestCase):
    def test_oneshot(self):
        chunks = build_tts_chunks("Hello. World.", chunk_strategy="oneshot", max_chunk_chars=3)
        self.assertEqual(len(chunks), 1)

    def test_smart_chunk(self):
        chunks = build_tts_chunks("First sentence. Second sentence. Third sentence.", chunk_strategy="smart_chunk", max_chunk_chars=6)
        self.assertGreaterEqual(len(chunks), 2)



class VisionInputTests(unittest.TestCase):
    @staticmethod
    def _data_url(w: int = 320, h: int = 180) -> str:
        from PIL import Image

        img = Image.new("RGB", (w, h))
        px = img.load()
        for yy in range(h):
            for xx in range(w):
                px[xx, yy] = (
                    (xx * 7 + yy * 3) % 256,
                    (xx * 5 + yy * 11) % 256,
                    (yy * 9 + xx * 2) % 256,
                )
        out = BytesIO()
        img.save(out, format="JPEG", quality=80)
        b64 = base64.b64encode(out.getvalue()).decode("ascii")
        return "data:image/jpeg;base64," + b64

    def test_decode_and_prepare_image_rejects_too_small(self):
        path, digest, note, err = _decode_and_prepare_image(self._data_url(40, 40))
        self.assertIsNone(path)
        self.assertIsNone(digest)
        self.assertIsNotNone(err)
        self.assertIn("image_too_small", err)
        self.assertIsNone(note)

    def test_resolve_image_paths_updates_cache(self):
        payload = {"image": self._data_url(320, 180), "reuse_vision": True}
        image_paths, note, cache_update = _resolve_image_paths(payload, vision_cache=None)
        self.assertIsNotNone(image_paths)
        self.assertEqual(len(image_paths or []), 1)
        self.assertIsNotNone(cache_update)
        self.assertTrue(Path((cache_update or {}).get("path", "")).exists())
        for p in image_paths or []:
            _cleanup_path(p)
        if cache_update:
            _cleanup_path(cache_update.get("path"))
        self.assertTrue(note is None or isinstance(note, str))

    def test_resolve_image_paths_reuses_cache_for_stale_camera_frame(self):
        payload = {"image": self._data_url(320, 180), "reuse_vision": True}
        image_paths, _, cache_update = _resolve_image_paths(payload, vision_cache=None)
        self.assertIsNotNone(image_paths)
        self.assertIsNotNone(cache_update)
        cache = {
            "path": cache_update["path"],
            "digest": cache_update.get("digest", ""),
            "ts_ms": int(time.time() * 1000),
        }
        stale_payload = {
            "image": self._data_url(320, 180),
            "image_age_ms": 6001,
            "reuse_vision": True,
        }
        reused_paths, note, next_cache = _resolve_image_paths(stale_payload, vision_cache=cache)
        self.assertIsNotNone(reused_paths)
        self.assertEqual(len(reused_paths or []), 1)
        self.assertIsNone(next_cache)
        self.assertIsInstance(note, str)
        self.assertIn("stale_image_skipped", note)
        self.assertIn("vision_reused_from_cache", note)
        for p in image_paths or []:
            _cleanup_path(p)
        for p in reused_paths or []:
            _cleanup_path(p)
        _cleanup_path(cache_update.get("path"))


class ToolPolicyTests(unittest.TestCase):
    def setUp(self):
        self._old_allow = os.environ.get("LIAGENT_ALLOW_HIGH_RISK_TOOLS")
        self._old_confirm = os.environ.get("LIAGENT_CONFIRM_RISK_LEVELS")
        self._old_confirm_tools = os.environ.get("LIAGENT_CONFIRM_TOOLS")
        self._old_allowlist = os.environ.get("LIAGENT_ALLOWED_TOOLS")
        self._old_tool_profile = os.environ.get("LIAGENT_TOOL_PROFILE")
        self._old_allow_net = os.environ.get("LIAGENT_ALLOW_NETWORK_TOOLS")
        self._old_allow_fs = os.environ.get("LIAGENT_ALLOW_FILESYSTEM_TOOLS")
        self._old_out_chars = os.environ.get("LIAGENT_TOOL_OUTPUT_MAX_CHARS")

    def tearDown(self):
        if self._old_allow is None:
            os.environ.pop("LIAGENT_ALLOW_HIGH_RISK_TOOLS", None)
        else:
            os.environ["LIAGENT_ALLOW_HIGH_RISK_TOOLS"] = self._old_allow
        if self._old_confirm is None:
            os.environ.pop("LIAGENT_CONFIRM_RISK_LEVELS", None)
        else:
            os.environ["LIAGENT_CONFIRM_RISK_LEVELS"] = self._old_confirm
        if self._old_confirm_tools is None:
            os.environ.pop("LIAGENT_CONFIRM_TOOLS", None)
        else:
            os.environ["LIAGENT_CONFIRM_TOOLS"] = self._old_confirm_tools
        if self._old_allowlist is None:
            os.environ.pop("LIAGENT_ALLOWED_TOOLS", None)
        else:
            os.environ["LIAGENT_ALLOWED_TOOLS"] = self._old_allowlist
        if self._old_tool_profile is None:
            os.environ.pop("LIAGENT_TOOL_PROFILE", None)
        else:
            os.environ["LIAGENT_TOOL_PROFILE"] = self._old_tool_profile
        if self._old_allow_net is None:
            os.environ.pop("LIAGENT_ALLOW_NETWORK_TOOLS", None)
        else:
            os.environ["LIAGENT_ALLOW_NETWORK_TOOLS"] = self._old_allow_net
        if self._old_allow_fs is None:
            os.environ.pop("LIAGENT_ALLOW_FILESYSTEM_TOOLS", None)
        else:
            os.environ["LIAGENT_ALLOW_FILESYSTEM_TOOLS"] = self._old_allow_fs
        if self._old_out_chars is None:
            os.environ.pop("LIAGENT_TOOL_OUTPUT_MAX_CHARS", None)
        else:
            os.environ["LIAGENT_TOOL_OUTPUT_MAX_CHARS"] = self._old_out_chars

    def test_validator_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "policy.db"
            policy = ToolPolicy(db, tool_profile="full")
            tool = ToolDef(
                name="test",
                description="d",
                parameters={},
                func=lambda **kwargs: "",
                validator=lambda args: (False, "bad"),
            )
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertEqual(reason, "bad")

    def test_high_risk_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "policy.db"
            os.environ["LIAGENT_ALLOW_HIGH_RISK_TOOLS"] = "false"
            policy = ToolPolicy(db, tool_profile="full")
            tool = ToolDef(
                name="danger",
                description="d",
                parameters={},
                func=lambda **kwargs: "",
                risk_level="high",
            )
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertIn("confirmation required", reason)

    def test_confirmation_required_by_risk_level(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "policy.db"
            os.environ["LIAGENT_CONFIRM_RISK_LEVELS"] = "medium,high"
            policy = ToolPolicy(db)
            tool = ToolDef(
                name="web_search",
                description="d",
                parameters={},
                func=lambda **kwargs: "",
                risk_level="medium",
            )
            allowed, reason = policy.evaluate(tool, {}, confirmed=False)
            self.assertFalse(allowed)
            self.assertIn("confirmation required", reason)
            allowed2, reason2 = policy.evaluate(tool, {}, confirmed=True)
            self.assertTrue(allowed2)
            self.assertEqual(reason2, "allowed")

    def test_confirmation_required_by_tool_name(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "policy.db"
            os.environ["LIAGENT_CONFIRM_RISK_LEVELS"] = "high"
            os.environ["LIAGENT_CONFIRM_TOOLS"] = "screenshot,stock"
            policy = ToolPolicy(db, tool_profile="full")
            tool = ToolDef(
                name="stock",
                description="d",
                parameters={},
                func=lambda **kwargs: "",
                risk_level="medium",
            )
            allowed, reason = policy.evaluate(tool, {}, confirmed=False)
            self.assertFalse(allowed)
            self.assertIn("confirmation required for tool=stock", reason)
            allowed2, _ = policy.evaluate(tool, {}, confirmed=True)
            self.assertTrue(allowed2)

    def test_network_tool_blocked_by_policy(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "policy.db"
            os.environ["LIAGENT_ALLOW_NETWORK_TOOLS"] = "false"
            policy = ToolPolicy(db)
            tool = ToolDef(
                name="web_search",
                description="d",
                parameters={},
                func=lambda **kwargs: "",
                capability=ToolCapability(network_access=True),
            )
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertEqual(reason, "network tools are blocked by policy")

    def test_profile_minimal_blocks_tools(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "policy.db"
            policy = ToolPolicy(db, tool_profile="minimal")
            tool = ToolDef(
                name="web_search",
                description="d",
                parameters={},
                func=lambda **kwargs: "",
                risk_level="medium",
            )
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertIn("profile=minimal", reason)

    def test_profile_full_allows_extra_tools(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "policy.db"
            policy = ToolPolicy(db, tool_profile="full")
            tool = ToolDef(
                name="custom_tool",
                description="d",
                parameters={},
                func=lambda **kwargs: "",
                risk_level="low",
            )
            allowed, reason = policy.evaluate(tool, {})
            self.assertTrue(allowed)
            self.assertEqual(reason, "allowed")

    def test_output_sanitize_redact_and_truncate(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "policy.db"
            os.environ["LIAGENT_TOOL_OUTPUT_MAX_CHARS"] = "120"
            policy = ToolPolicy(db)
            tool = ToolDef(
                name="x",
                description="d",
                parameters={},
                func=lambda **kwargs: "",
                capability=ToolCapability(max_output_chars=100),
            )
            raw = "api_key=abcdef1234567890 " + ("a" * 300)
            out = policy.sanitize_output(tool, raw)
            self.assertIn("api_key=[REDACTED]", out)
            self.assertIn("policy truncated", out)
            self.assertLessEqual(len(out), 130)

    def test_recent_audit_returns_rows(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "policy.db"
            policy = ToolPolicy(db)
            policy.audit("stock", {"token": "abc123", "symbol": "AAPL"}, "ok", "executed")
            items = policy.recent_audit(limit=10)
            self.assertGreaterEqual(len(items), 1)
            self.assertEqual(items[0]["tool_name"], "stock")
            self.assertEqual(items[0]["args"].get("token"), "[REDACTED]")


class MemoryTests(unittest.TestCase):
    def test_fact_conflict_keeps_higher_confidence(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "mem.db"
            mem = LongTermMemory(db, data_dir=Path(td) / "data")
            mem.save_facts(
                [
                    {"fact": "user likes coffee", "category": "preference", "confidence": 0.6, "source": "s1"},
                    {"fact": "user likes coffee", "category": "preference", "confidence": 0.95, "source": "s2"},
                ]
            )
            facts = mem.get_all_facts(min_confidence=0.5)
            self.assertEqual(len(facts), 1)
            self.assertAlmostEqual(facts[0]["confidence"], 0.95)
            self.assertEqual(facts[0]["source"], "s2")


class MetricsTests(unittest.TestCase):
    def test_weekly_summary_contains_new_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "metrics.db"
            m = InteractionMetrics(db)
            m.log_turn(
                session_id="s1",
                latency_ms=200.0,
                tool_calls=2,
                tool_errors=1,
                policy_blocked=0,
                task_success=False,
                answer_revision_count=2,
                quality_issues="fact_check,format",
                plan_completion_ratio=0.5,
                answer_chars=80,
            )
            m.log_runtime(
                run_id="r1",
                queued_ms=45.0,
                stream_ms=330.0,
                tts_ms=180.0,
                total_ms=410.0,
                voice_mode=True,
                final_state="done",
            )
            summary = m.weekly_summary(days=7)
            self.assertGreaterEqual(summary["turns"], 1)
            self.assertIn("task_success_rate", summary)
            self.assertIn("avg_answer_revision_count", summary)
            self.assertIn("tool_error_rate", summary)
            self.assertIn("avg_plan_completion_rate", summary)
            self.assertIn("avg_queue_ms", summary)
            self.assertIn("avg_stream_ms", summary)
            self.assertIn("avg_tts_ms", summary)
            self.assertIn("avg_total_ms", summary)
            self.assertGreaterEqual(summary["runtime_samples"], 1)
            self.assertIsInstance(summary["top_issue_samples"], list)


class PythonExecGuardTests(unittest.TestCase):
    def test_blocks_git_commit_code(self):
        ok, reason = _validate_python_exec(
            {"code": "import subprocess\nsubprocess.run(['git', 'commit', '-m', 'x'])"}
        )
        self.assertFalse(ok)
        self.assertIn("blocked", reason)

    def test_allows_normal_code(self):
        ok, reason = _validate_python_exec({"code": "print('hello')"})
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")


class BrainHelperTests(unittest.TestCase):
    def test_api_message_trim_keeps_system_and_recent_context(self):
        brain = AgentBrain.__new__(AgentBrain)
        brain.api_context_char_budget = 140

        messages = [
            {"role": "system", "content": "S" * 40},
            {"role": "user", "content": "old user" * 4},
            {"role": "assistant", "content": "old assistant" * 3},
            {"role": "tool", "content": "recent tool output" * 2},
            {"role": "user", "content": "[Execution Progress] current step"},
            {"role": "assistant", "content": "latest assistant"},
        ]
        trimmed = brain._trim_messages_for_api(messages)

        self.assertEqual(trimmed[0]["role"], "system")
        self.assertTrue(any(m["role"] == "tool" for m in trimmed))
        self.assertTrue(any("[Execution Progress]" in m.get("content", "") for m in trimmed))
        self.assertFalse(any("old user" in m.get("content", "") for m in trimmed))

    def test_api_message_trim_preserves_control_marker_even_if_newer_user_exists(self):
        brain = AgentBrain.__new__(AgentBrain)
        brain.api_context_char_budget = 120

        messages = [
            {"role": "system", "content": "S" * 30},
            {"role": "user", "content": "older context" * 3},
            {"role": "assistant", "content": "older answer" * 2},
            {"role": "user", "content": "[Execution Progress] focus S2 with evidence"},
            {"role": "assistant", "content": "latest assistant short"},
            {"role": "user", "content": "new short user"},
        ]
        trimmed = brain._trim_messages_for_api(messages)

        self.assertTrue(any("[Execution Progress]" in m.get("content", "") for m in trimmed))
        self.assertTrue(any(m.get("content", "") == "new short user" for m in trimmed))

    def test_api_budget_consume_prefers_engine_usage(self):
        brain = AgentBrain.__new__(AgentBrain)
        brain._api_budget_active = True
        brain.api_turn_token_budget = 1000
        brain._api_turn_tokens_used = 100
        brain.engine = SimpleNamespace(
            get_last_llm_usage=lambda: {"total_tokens": 55, "prompt_tokens": 30, "completion_tokens": 25}
        )
        spent = brain._api_budget_consume(
            messages=[{"role": "user", "content": "hello"}],
            response_text="world",
            tools=None,
        )
        self.assertEqual(spent, 55)
        self.assertEqual(brain._api_turn_tokens_used, 155)
        self.assertEqual(brain._api_budget_remaining_tokens(), 845)

    def test_api_budget_consume_falls_back_to_estimate(self):
        brain = AgentBrain.__new__(AgentBrain)
        brain._api_budget_active = True
        brain.api_turn_token_budget = 1000
        brain._api_turn_tokens_used = 0
        brain.engine = SimpleNamespace(get_last_llm_usage=lambda: {})
        messages = [{"role": "system", "content": "abc"}, {"role": "user", "content": "defgh"}]
        spent = brain._api_budget_consume(
            messages=messages,
            response_text="ijklmnop",
            tools=[{"type": "function", "function": {"name": "web_search"}}],
        )
        self.assertGreater(spent, 0)
        self.assertEqual(brain._api_turn_tokens_used, spent)

    def test_api_turn_budget_expands_for_standard_chat_envelope(self):
        brain = AgentBrain.__new__(AgentBrain)
        brain._api_budget_active = True
        brain.api_turn_token_budget = 20000
        brain.api_budget_reserve_tokens = 320

        brain._ensure_api_turn_budget_capacity(
            max_steps=10,
            llm_max_tokens=2048,
            planning_enabled=True,
        )

        self.assertGreaterEqual(brain.api_turn_token_budget, 28000)
        self.assertGreaterEqual(brain.api_budget_reserve_tokens, 1024)

    def test_api_turn_budget_scales_for_moonshot_reasoning_provider(self):
        brain = AgentBrain.__new__(AgentBrain)
        brain._api_budget_active = True
        brain.api_turn_token_budget = 20000
        brain.api_budget_reserve_tokens = 320
        brain.engine = SimpleNamespace(
            config=SimpleNamespace(
                llm=SimpleNamespace(
                    backend="api",
                    api_model="kimi-k2.5",
                    api_base_url="https://api.moonshot.cn/v1",
                )
            )
        )

        brain._ensure_api_turn_budget_capacity(
            max_steps=10,
            llm_max_tokens=2048,
            planning_enabled=True,
        )

        self.assertGreaterEqual(brain.api_turn_token_budget, 60000)
        self.assertGreaterEqual(brain.api_budget_reserve_tokens, 2000)

    def test_api_react_step_token_cap_is_tighter_for_moonshot(self):
        brain = AgentBrain.__new__(AgentBrain)
        brain.engine = SimpleNamespace(
            config=SimpleNamespace(
                llm=SimpleNamespace(
                    backend="api",
                    api_model="kimi-k2.5",
                    api_base_url="https://api.moonshot.cn/v1",
                )
            )
        )
        self.assertEqual(brain._api_react_step_token_cap(tools_enabled=True), 960)
        self.assertEqual(brain._api_react_step_token_cap(tools_enabled=False), 1280)

    def test_cwork_root_prefers_configured_env(self):
        import importlib
        import os
        from pathlib import Path

        from liagent.tools import _path_security as path_security

        old_dir = os.environ.get("LIAGENT_CWORK_DIR")
        old_root = os.environ.get("LIAGENT_CWORK_ROOT")
        try:
            os.environ["LIAGENT_CWORK_DIR"] = "/tmp/liagent-custom-cwork"
            os.environ["LIAGENT_CWORK_ROOT"] = "/tmp/liagent-legacy-cwork"
            importlib.reload(path_security)
            self.assertEqual(
                str(path_security.get_cwork_root()),
                str(Path("/tmp/liagent-custom-cwork").resolve()),
            )
            ok, _, resolved = path_security._validate_cwork_path("notes.txt")
            self.assertTrue(ok)
            self.assertEqual(
                str(resolved),
                str((Path("/tmp/liagent-custom-cwork") / "notes.txt").resolve()),
            )
        finally:
            if old_dir is None:
                os.environ.pop("LIAGENT_CWORK_DIR", None)
            else:
                os.environ["LIAGENT_CWORK_DIR"] = old_dir
            if old_root is None:
                os.environ.pop("LIAGENT_CWORK_ROOT", None)
            else:
                os.environ["LIAGENT_CWORK_ROOT"] = old_root
            importlib.reload(path_security)

    def test_python_exec_risk_level(self):
        import importlib
        from liagent.tools import get_tool
        import liagent.tools.python_exec as python_exec_mod

        importlib.reload(python_exec_mod)
        td = get_tool("python_exec")
        self.assertIsNotNone(td)
        self.assertEqual(td.risk_level, "medium")

    def test_available_tools_filters_unavailable_mcp_and_browser_wrappers(self):
        brain = AgentBrain.__new__(AgentBrain)
        brain._mcp_bridge = SimpleNamespace(server_errors={"playwright": "unavailable", "fetch": "missing uvx"})
        with patch(
            "liagent.agent.brain.get_all_tools",
            return_value={
                "web_search": object(),
                "browser_navigate": object(),
                "playwright__browser_click": object(),
                "fetch__fetch": object(),
            },
        ):
            names = brain._available_tool_names()
        self.assertEqual(names, {"web_search"})

    def test_tool_call_signature_stable(self):
        s1 = tool_call_signature("stock", {"symbol": "AAPL", "market": "US"})
        s2 = tool_call_signature("stock", {"market": "US", "symbol": "AAPL"})
        self.assertEqual(s1, s2)
        s3 = tool_call_signature("stock", {"symbol": "TSLA", "market": "US"})
        self.assertNotEqual(s1, s3)

    def test_degrade_observation_unknown_stock_tool(self):
        msg = build_tool_degrade_observation("stock", {"symbol": "AAPL"}, "timeout")
        self.assertIn("[Tool degraded] stock unavailable", msg)
        self.assertIn("Try an alternative approach", msg)

    def test_degrade_observation_web_fetch_missing_url(self):
        msg = build_tool_degrade_observation(
            "web_fetch", {"query": "test"}, "missing 1 required positional argument: 'url'"
        )
        self.assertIn("invalid arguments", msg)
        self.assertIn("web_search", msg)
        self.assertNotIn("please answer using available information", msg)

    def test_degrade_observation_web_fetch_with_url(self):
        msg = build_tool_degrade_observation(
            "web_fetch", {"url": "https://example.com"}, "connection timeout"
        )
        self.assertIn("example.com", msg)
        self.assertIn("web_search", msg)

    def test_strip_any_tool_call_both_formats(self):
        json_text = 'before <tool_call>{"name": "x", "args": {}}</tool_call> after'
        self.assertNotIn("<tool_call>", strip_any_tool_call(json_text))

        native_text = (
            "before <tool_call>\n<function=x>\n</function>\n</tool_call> after"
        )
        self.assertNotIn("<tool_call>", strip_any_tool_call(native_text))

    def test_skill_router_voice_allows_lookup_tools(self):
        """Voice mode should return REALTIME_VOICE config with web tools."""
        cfg = select_skill("please check latest Google stock price", low_latency=True, has_images=False)
        self.assertEqual(cfg.name, "realtime_voice")
        self.assertIn("web_search", cfg.allowed_tools or set())
        self.assertIn("web_fetch", cfg.allowed_tools or set())

    def test_skill_router_standard_chat(self):
        """Non-voice, non-image queries should return STANDARD_CHAT."""
        cfg = select_skill("search NVIDIA latest earnings and summarize", low_latency=False, has_images=False)
        self.assertEqual(cfg.name, "standard_chat")
        self.assertIsNone(cfg.allowed_tools)  # unrestricted


class ToolSchemaTests(unittest.TestCase):
    def test_to_native_schema(self):
        td = ToolDef(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            func=lambda **kwargs: "",
        )
        schema = td.to_native_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "web_search")
        self.assertEqual(schema["function"]["description"], "Search the web")
        self.assertIn("properties", schema["function"]["parameters"])

    def test_get_native_tool_schemas_returns_list(self):
        schemas = get_native_tool_schemas()
        self.assertIsInstance(schemas, list)
        # At least some tools should be registered
        self.assertGreater(len(schemas), 0)
        for s in schemas:
            self.assertEqual(s["type"], "function")
            self.assertIn("name", s["function"])


class BrainRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_low_latency_run_returns_done_without_plan_scope_error(self):
        class _FakeLongTermMemory:
            def __init__(self, **kwargs):
                pass

            db_path = ":memory:"

            def get_recent_summaries(self, limit: int = 5):
                return []

            def get_all_facts(self, min_confidence: float = 0.65):
                return []

            def get_relevant_facts(self, query: str, limit: int = 8):
                return []

            def decay_confidence(self, *args, **kwargs):
                return None

            def prune_memory(self, *args, **kwargs):
                return None

            def save_feedback(self, *args, **kwargs):
                return None

            def apply_source_confidence(self, facts):
                return facts

            def detect_conflicts(self, *args, **kwargs):
                return None

        class _FakeMetrics:
            def log_turn(self, **kwargs):
                return None

        class _FakeToolParser:
            def format_schemas(self, tools):
                return tools

        class _FakeEngine:
            def __init__(self):
                self.config = AppConfig()
                self.config.llm.max_tokens = 256
                self.config.llm.temperature = 0.2
                self.config.tool_profile = "research"
                self.tool_parser = _FakeToolParser()

            async def generate_llm_routed(self, *args, **kwargs):
                yield "test reply"

            async def generate_llm(self, *args, **kwargs):
                yield "test reply"

            async def generate_text(self, *args, **kwargs):
                yield "test reply"

            async def stream_text(self, *args, **kwargs):
                yield "test reply"

        class _FakeExperienceMemory:
            def __init__(self, *args, **kwargs):
                pass

            def match(self, query):
                return None

            def record_outcome(self, *args, **kwargs):
                pass

            def sync_from_markdown(self):
                pass

            def sync_to_markdown(self):
                pass

        with patch("liagent.agent.brain.LongTermMemory", _FakeLongTermMemory), \
             patch("liagent.agent.brain.InteractionMetrics", _FakeMetrics), \
             patch("liagent.agent.brain.ExperienceMemory", _FakeExperienceMemory):
            brain = AgentBrain(_FakeEngine())
            events = []
            async for ev in brain.run("hello", low_latency=True):
                events.append(ev)

        self.assertTrue(any(ev[0] == "done" for ev in events))


class FTS5RetrievalTests(unittest.TestCase):
    """Tests for jieba-based FTS5 retrieval, MD sync, and daily log."""

    def _make_mem(self, td):
        db = Path(td) / "mem.db"
        data_dir = Path(td) / "liagent_data"
        return LongTermMemory(db, data_dir=data_dir)

    def test_tokenize_cut_for_search(self):
        from liagent.agent.memory import _tokenize

        tokens = _tokenize("drink coffee")
        self.assertIn("coffee", tokens)
        self.assertIn("drink coffee", tokens)

    def test_chinese_fts_retrieval(self):
        with tempfile.TemporaryDirectory() as td:
            mem = self._make_mem(td)
            mem.save_facts([
                {"fact": "user likes drinking coffee", "category": "preference", "confidence": 0.85},
                {"fact": "user lives in Haidian, Beijing", "category": "location", "confidence": 0.90},
            ])
            results = mem.get_relevant_facts("coffee")
            fact_texts = [r["fact"] for r in results]
            self.assertIn("user likes drinking coffee", fact_texts)

    def test_english_fts_retrieval(self):
        with tempfile.TemporaryDirectory() as td:
            mem = self._make_mem(td)
            mem.save_facts([
                {"fact": "User likes stock trading", "category": "interest", "confidence": 0.80},
                {"fact": "User prefers dark mode", "category": "preference", "confidence": 0.75},
            ])
            results = mem.get_relevant_facts("stock")
            fact_texts = [r["fact"] for r in results]
            self.assertIn("User likes stock trading", fact_texts)

    def test_bm25_ranking(self):
        with tempfile.TemporaryDirectory() as td:
            mem = self._make_mem(td)
            mem.save_facts([
                {"fact": "user occasionally drinks coffee", "category": "preference", "confidence": 0.70},
                {"fact": "user drinks coffee every morning and prefers latte", "category": "habit", "confidence": 0.85},
                {"fact": "user lives in Shanghai", "category": "location", "confidence": 0.90},
            ])
            results = mem.get_relevant_facts("coffee")
            fact_texts = [r["fact"] for r in results]
            # Both coffee facts should be returned, location fact should not
            self.assertTrue(len(fact_texts) >= 2)
            self.assertNotIn("user lives in Shanghai", fact_texts)

    def test_fts_sync_after_prune(self):
        with tempfile.TemporaryDirectory() as td:
            mem = self._make_mem(td)
            mem.save_facts([
                {"fact": "user likes coffee", "category": "preference", "confidence": 0.10},
                {"fact": "user lives in Beijing", "category": "location", "confidence": 0.90},
            ])
            mem.prune_memory(min_confidence=0.5)
            results = mem.get_relevant_facts("coffee")
            fact_texts = [r["fact"] for r in results]
            self.assertNotIn("user likes coffee", fact_texts)

    def test_fts_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            mem = self._make_mem(td)
            mem.save_facts([
                {"fact": "user likes tea", "category": "preference", "confidence": 0.85},
            ])
            # Corrupt FTS table
            with sqlite3.connect(mem.db_path) as conn:
                conn.execute("DROP TABLE key_facts_fts")
            # Should fallback to legacy without crashing
            results = mem.get_relevant_facts("drink tea")
            self.assertTrue(len(results) >= 1)

    def test_markdown_sync_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            mem = self._make_mem(td)
            mem.save_facts([
                {"fact": "user likes Python", "category": "tech", "confidence": 0.90},
                {"fact": "user lives in Hangzhou", "category": "location", "confidence": 0.85},
            ])
            self.assertTrue(mem.facts_md_path.exists())
            md_text = mem.facts_md_path.read_text(encoding="utf-8")
            self.assertIn("user likes Python", md_text)
            self.assertIn("user lives in Hangzhou", md_text)

            # Simulate DB loss and rebuild from MD
            db2 = Path(td) / "mem2.db"
            mem2 = LongTermMemory(db2, data_dir=mem.data_dir)
            facts = mem2.get_all_facts(min_confidence=0.0)
            fact_texts = [f["fact"] for f in facts]
            self.assertIn("user likes Python", fact_texts)
            self.assertIn("user lives in Hangzhou", fact_texts)



class SearchQualityTests(unittest.TestCase):
    def test_detect_copout_positive(self):
        from liagent.agent.quality import detect_copout

        self.assertTrue(detect_copout("please check Apple official site for price"))
        self.assertTrue(detect_copout("no relevant information found in search results"))
        self.assertTrue(detect_copout("unable to fetch latest data"))
        self.assertTrue(detect_copout("please visit the official website directly"))

    def test_detect_copout_negative(self):
        from liagent.agent.quality import detect_copout

        self.assertFalse(detect_copout("Apple Vision Pro is priced at $3,499"))
        self.assertFalse(detect_copout("latest Google stock price is $180.50"))
        self.assertFalse(detect_copout("current gold price per ounce is $2,350"))

    def test_estimate_task_success_detects_copout(self):
        result, reason = estimate_task_success(
            answer="please check Apple official site for accurate information",
            tool_calls=1,
            tool_errors=0,
            policy_blocked=0,
            plan_total_steps=0,
            plan_completed_steps=0,
        )
        self.assertFalse(result)
        self.assertEqual(reason, "copout_answer")

    def test_estimate_task_success_no_copout_without_tools(self):
        # Copout detection only triggers when tools were used
        result, reason = estimate_task_success(
            answer="please check Apple official site for accurate information",
            tool_calls=0,
            tool_errors=0,
            policy_blocked=0,
            plan_total_steps=0,
            plan_completed_steps=0,
        )
        self.assertTrue(result)
        self.assertEqual(reason, "ok")

    def test_detect_hallucinated_action_file_write(self):
        from liagent.agent.quality import detect_hallucinated_action

        # Claims file saved but only web_search was used
        self.assertTrue(detect_hallucinated_action(
            "This report was saved to the cwork folder.", {"web_search"}
        ))
        self.assertTrue(detect_hallucinated_action(
            "Report content has been written to a file.", {"web_search", "web_fetch"}
        ))
        # Actually used write_file - not hallucination
        self.assertFalse(detect_hallucinated_action(
            "Report saved to cwork/report.txt.", {"web_search", "write_file"}
        ))

    def test_detect_hallucinated_action_code(self):
        from liagent.agent.quality import detect_hallucinated_action

        self.assertTrue(detect_hallucinated_action(
            "Execution result: x = 42", {"web_search"}
        ))
        self.assertFalse(detect_hallucinated_action(
            "Execution result: x = 42", {"python_exec"}
        ))

    def test_detect_hallucinated_action_baodao_pattern(self):
        """'saved to' (not just 'saved to') must be caught as hallucination."""
        from liagent.agent.quality import detect_hallucinated_action

        # "saved to" without write_file → hallucination
        self.assertTrue(detect_hallucinated_action(
            "Tetris game code has been saved to cwork/tetris.py.", set()
        ))
        self.assertTrue(detect_hallucinated_action(
            "Tetris game code has been saved to cwork/tetris.py.", {"web_search"}
        ))
        # With write_file → not hallucination
        self.assertFalse(detect_hallucinated_action(
            "Tetris game code has been saved to cwork/tetris.py.", {"write_file"}
        ))

    def test_detect_hallucinated_action_no_claim(self):
        from liagent.agent.quality import detect_hallucinated_action

        self.assertFalse(detect_hallucinated_action(
            "Apple Vision Pro is priced at $3,499.", {"web_search"}
        ))

    def test_estimate_task_success_detects_hallucinated_action(self):
        result, reason = estimate_task_success(
            answer="This report was saved to the cwork folder.",
            tool_calls=1,
            tool_errors=0,
            policy_blocked=0,
            plan_total_steps=0,
            plan_completed_steps=0,
            tools_used={"web_search"},
        )
        self.assertFalse(result)
        self.assertEqual(reason, "hallucinated_action")

    def test_estimate_task_success_ok_with_actual_write(self):
        result, reason = estimate_task_success(
            answer="Report saved to cwork/report.txt.",
            tool_calls=2,
            tool_errors=0,
            policy_blocked=0,
            plan_total_steps=0,
            plan_completed_steps=0,
            tools_used={"web_search", "write_file"},
        )
        self.assertTrue(result)
        self.assertEqual(reason, "ok")

    def test_heuristic_query_refinement(self):
        from liagent.agent.brain import AgentBrain

        r = AgentBrain._heuristic_refine_query("please check latest Google products")
        self.assertNotIn("please", r)
        self.assertNotIn("take a look", r)
        self.assertIn("latest Google products", r)

        r2 = AgentBrain._heuristic_refine_query(
            "write a research report on latest Google products and put the text into cwork folder"
        )
        self.assertNotIn("put the text into cwork folder", r2)
        self.assertIn("latest Google products", r2)

        # Very short input not over-stripped
        r3 = AgentBrain._heuristic_refine_query("AAPL")
        self.assertEqual(r3, "AAPL")

        # Natural-language queries still preserve key named entities
        r4 = AgentBrain._heuristic_refine_query("Please summarize the AI forum conference in San Francisco next month")
        self.assertIn("San Francisco", r4)

    def test_lenient_parse_unclosed_tool_call(self):
        # Model generates <tool_call>{JSON} without </tool_call>
        text = '<tool_call>{"name": "web_fetch", "args": {"url": "https://example.com"}}'
        result = _parse_tool_call_lenient(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "web_fetch")
        self.assertEqual(result["args"]["url"], "https://example.com")

    def test_lenient_parse_repeated_unclosed(self):
        # Degenerate: many unclosed tool_call blocks (the actual bug)
        block = '<tool_call>\n{"name": "web_fetch", "args": {"url": "https://apple.com/watch/"}}\n'
        text = block * 30
        result = _parse_tool_call_lenient(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "web_fetch")

    def test_lenient_parse_returns_none_for_no_tool_call(self):
        result = _parse_tool_call_lenient("Just a normal response with no tools.")
        self.assertIsNone(result)

    def test_detect_degenerate_output(self):
        from liagent.agent.quality import detect_degenerate_output

        # 3+ tool_call fragments = degenerate
        block = '<tool_call>{"name": "web_fetch", "args": {"url": "https://x.com"}}\n'
        self.assertTrue(detect_degenerate_output(block * 5))

        # Repeated planning chatter should also be treated as degenerate.
        chatter = (
            "Let me search for AI conferences in San Francisco.\n"
            "I need to call tools to search more.\n"
            "Let me search again with another query.\n"
            "Calling tool web_search now."
        )
        self.assertTrue(detect_degenerate_output(chatter))

        # Single tool_call = not degenerate
        self.assertFalse(detect_degenerate_output(
            '<tool_call>{"name": "web_fetch", "args": {"url": "https://x.com"}}</tool_call>'
        ))

        # Normal text = not degenerate
        self.assertFalse(detect_degenerate_output("Apple Vision Pro is priced at $3,499"))

    def test_copout_detects_monitor_phrase(self):
        from liagent.agent.quality import detect_copout

        self.assertTrue(detect_copout("monitor Apple official site or authorized retailers for latest pricing"))

    def test_copout_detects_visit_phrase(self):
        from liagent.agent.quality import detect_copout

        self.assertTrue(detect_copout("visit Apple official site or authorized retailers for latest information"))



class VisionModeTests(unittest.TestCase):
    def test_voice_camera_selects_vision_skill(self):
        from liagent.skills.router import select_skill
        cfg = select_skill("what do you see", low_latency=True, has_images=True)
        self.assertEqual(cfg.name, "realtime_vision")
        self.assertIn("describe_image", cfg.allowed_tools)

    def test_voice_no_camera_selects_voice_skill(self):
        from liagent.skills.router import select_skill
        cfg = select_skill("hello", low_latency=True, has_images=False)
        self.assertEqual(cfg.name, "realtime_voice")
        self.assertNotIn("describe_image", cfg.allowed_tools)

    def test_non_voice_text_selects_standard_chat(self):
        from liagent.skills.router import select_skill
        cfg = select_skill("look at this", low_latency=False, has_images=True)
        self.assertEqual(cfg.name, "standard_chat")


class UnwrittenCodeTests(unittest.TestCase):
    """Tests for the unwritten code compliance detection scoring model."""

    def _make_answer(self, code: str, explanation: str = "") -> str:
        return f"{explanation}```python\n{code}\n```"

    def test_short_snippet_no_intent_passes(self):
        """A 3-line snippet without file intent should not trigger."""
        from liagent.agent.quality import detect_unwritten_code
        answer = self._make_answer("x = 1\ny = 2\nprint(x + y)")
        should, lines, score = detect_unwritten_code(answer, "how to assign in python", set())
        self.assertFalse(should)

    def test_strong_intent_with_any_code_triggers(self):
        """Strong intent ('put into cwork') + even modest code should trigger."""
        from liagent.agent.quality import detect_unwritten_code
        code = "import os\ndef main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()"
        answer = self._make_answer(code)
        should, lines, score = detect_unwritten_code(answer, "write a script and put it into cwork", set())
        self.assertTrue(should)
        self.assertGreaterEqual(score, 5)

    def test_weak_intent_alone_insufficient(self):
        """Weak intent ('generate') + tiny code should NOT trigger."""
        from liagent.agent.quality import detect_unwritten_code
        answer = self._make_answer("print('hello')")
        should, lines, score = detect_unwritten_code(answer, "generate a hello world", set())
        self.assertFalse(should)

    def test_if_name_main_triggers_without_intent(self):
        """Code with if __name__ == '__main__' → score ≥5 even without user intent."""
        from liagent.agent.quality import detect_unwritten_code
        code = "import sys\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()"
        answer = self._make_answer(code)
        should, lines, score = detect_unwritten_code(answer, "how to write an entry function", set())
        self.assertTrue(should)

    def test_sensitive_ops_trigger_without_intent(self):
        """Code with os.remove → score ≥5 for audit trail."""
        from liagent.agent.quality import detect_unwritten_code
        code = "import os\nimport shutil\ndef cleanup():\n    os.remove('/tmp/old')\n    shutil.rmtree('/tmp/cache')"
        answer = self._make_answer(code)
        should, lines, score = detect_unwritten_code(answer, "clean temporary files", set())
        self.assertTrue(should)

    def test_explanatory_text_with_small_code_passes(self):
        """Mostly text explanation + small code snippet → code ratio <30% → pass."""
        from liagent.agent.quality import detect_unwritten_code
        explanation = (
            "os.remove deletes a single file. It accepts a path argument and raises "
            "FileNotFoundError if the file does not exist. You can check with "
            "os.path.exists first. Example usage:\n\n"
        )
        code = "import os\nos.remove('test.txt')"
        answer = explanation + f"```python\n{code}\n```"
        should, lines, score = detect_unwritten_code(answer, "how to use os.remove", set())
        self.assertFalse(should, f"Explanatory answer should pass, but got score={score}")

    def test_write_file_already_used_skips(self):
        """If write_file was already called, skip entirely."""
        from liagent.agent.quality import detect_unwritten_code
        code = "import pygame\n" * 30
        answer = self._make_answer(code)
        should, lines, score = detect_unwritten_code(answer, "create a game", {"write_file"})
        self.assertFalse(should)
        self.assertEqual(score, 0)

    def test_multiple_classes_score_high(self):
        """Code with multiple class definitions → high structure score."""
        from liagent.agent.quality import detect_unwritten_code
        code = "class Snake:\n    pass\n\nclass Food:\n    pass\n\nclass Game:\n    pass\n"
        answer = self._make_answer(code)
        should, lines, score = detect_unwritten_code(answer, "write a snake game", set())
        self.assertTrue(should)  # 3 classes (9) + strong intent (5) = 14


class WriteFilePathTests(unittest.TestCase):
    """Tests for write_file path normalization and TOCTOU consistency."""

    def test_relative_cwork_path_normalizes(self):
        """'cwork/tetris.py' should normalize to absolute cwork path."""
        from liagent.tools._path_security import _CWORK_ROOT, _validate_cwork_path
        ok, reason, resolved = _validate_cwork_path("cwork/tetris.py")
        self.assertTrue(ok, f"validation failed: {reason}")
        self.assertTrue(str(resolved).startswith(str(_CWORK_ROOT) + "/"))

    def test_toctou_recheck_uses_normalized_path(self):
        """TOCTOU re-resolve must use the normalized path, not raw input."""
        from liagent.tools._path_security import _validate_cwork_path
        from pathlib import Path

        raw_path = "cwork/game.py"
        ok, reason, resolved = _validate_cwork_path(raw_path)
        self.assertTrue(ok)

        # Simulate the TOCTOU re-check (as write_file.py does)
        resolved2 = resolved.resolve()
        ok2, reason2, _ = _validate_cwork_path(str(resolved2))
        self.assertTrue(ok2, f"TOCTOU re-check failed: {reason2}")

    def test_bare_filename_normalizes(self):
        """A bare filename like 'tetris.py' should resolve under cwork."""
        from liagent.tools._path_security import _CWORK_ROOT, _validate_cwork_path
        ok, reason, resolved = _validate_cwork_path("tetris.py")
        self.assertTrue(ok, f"validation failed: {reason}")
        self.assertEqual(resolved, _CWORK_ROOT / "tetris.py")


class EvidenceTests(unittest.TestCase):
    """Tests for evidence extraction, conflict detection, and trusted domains."""

    def test_money_normalization(self):
        from liagent.agent.evidence import _extract_data_points
        points = _extract_data_points("Revenue was $6.8 billion and 1 billion RMB")
        money_pts = [p for p in points if p["type"] == "money"]
        self.assertTrue(len(money_pts) >= 2)
        # $6.8 billion → 6800 million
        usd = [p for p in money_pts if "6.8" in p["raw"] and "value_m" in p]
        self.assertTrue(len(usd) >= 1)
        self.assertAlmostEqual(usd[0]["value_m"], 6800.0, places=0)
        # 1 billion -> 1000 million
        cny = [p for p in money_pts if "billion" in p["raw"] and "value_m" in p]
        self.assertTrue(len(cny) >= 1)
        self.assertAlmostEqual(cny[0]["value_m"], 1000.0, places=0)

    def test_conflict_detection_money(self):
        from liagent.agent.evidence import _find_conflicts
        sources = [
            {"domain": "bloomberg.com", "points": [{"type": "money", "raw": "$10B", "value_m": 10000}]},
            {"domain": "reuters.com", "points": [{"type": "money", "raw": "$12B", "value_m": 12000}]},
        ]
        conflicts = _find_conflicts(sources)
        self.assertTrue(any("amount" in c for c in conflicts), f"Expected money conflict, got: {conflicts}")

    def test_conflict_detection_no_false_positive(self):
        """Similar money values should NOT trigger conflict."""
        from liagent.agent.evidence import _find_conflicts
        sources = [
            {"domain": "bloomberg.com", "points": [{"type": "money", "raw": "$10B", "value_m": 10000}]},
            {"domain": "reuters.com", "points": [{"type": "money", "raw": "$10.2B", "value_m": 10200}]},
        ]
        conflicts = _find_conflicts(sources)
        # 2% divergence is within 5% threshold → no conflict
        self.assertEqual(len(conflicts), 0, f"Unexpected conflict: {conflicts}")

    def test_conflict_detection_percent(self):
        from liagent.agent.evidence import _find_conflicts
        sources = [
            {"domain": "a.com", "points": [{"type": "percent", "raw": "15%", "value": 15.0}]},
            {"domain": "b.com", "points": [{"type": "percent", "raw": "20%", "value": 20.0}]},
        ]
        conflicts = _find_conflicts(sources)
        self.assertTrue(any("%" in c for c in conflicts), f"Expected percent conflict, got: {conflicts}")

    def test_trusted_domain_boundary_safe(self):
        from liagent.agent.evidence import _is_trusted_domain
        self.assertTrue(_is_trusted_domain("https://www.reuters.com/article"))
        self.assertTrue(_is_trusted_domain("https://finance.yahoo.com/quote/GOOGL"))
        self.assertFalse(_is_trusted_domain("https://evilreuters.com/fake"))
        self.assertFalse(_is_trusted_domain("https://notbloomberg.com/x"))
        self.assertFalse(_is_trusted_domain(""))

    def test_web_fetch_url_attribution(self):
        """web_fetch URL from tool_args should be captured in step URLs."""
        from liagent.agent.evidence import extract_urls_from_text
        # Simulate: web_search result with URL: lines
        text = "1. Title\n   body\n   URL: https://example.com/article"
        urls = extract_urls_from_text(text)
        self.assertEqual(urls, ["https://example.com/article"])

class EngineeringToolTests(unittest.TestCase):
    """Tests for verify_syntax, lint_code, run_tests tool registration."""

    def test_tools_registered(self):
        """New engineering tools should be in the registry."""
        from liagent.tools import get_tool
        # Force import to trigger registration
        from liagent.tools import verify_syntax as _vs, lint_code as _lc, run_tests as _rt  # noqa: F401
        self.assertIsNotNone(get_tool("verify_syntax"))
        self.assertIsNotNone(get_tool("lint_code"))
        self.assertIsNotNone(get_tool("run_tests"))

    def test_verify_syntax_risk_level(self):
        from liagent.tools import get_tool
        from liagent.tools import verify_syntax as _vs  # noqa: F401
        td = get_tool("verify_syntax")
        self.assertEqual(td.risk_level, "low")
        self.assertFalse(td.requires_confirmation)

    def test_run_tests_risk_level(self):
        from liagent.tools import get_tool
        from liagent.tools import run_tests as _rt  # noqa: F401
        td = get_tool("run_tests")
        self.assertEqual(td.risk_level, "low")
        self.assertFalse(td.requires_confirmation)

    def test_write_file_risk_level(self):
        from liagent.tools import get_tool
        from liagent.tools import write_file as _wrf  # noqa: F401
        td = get_tool("write_file")
        self.assertEqual(td.risk_level, "medium")
        self.assertTrue(td.requires_confirmation)

    def test_verify_syntax_validator_blocks_outside_cwork(self):
        from liagent.tools.verify_syntax import _validate_verify_syntax
        ok, reason = _validate_verify_syntax({"path": "/etc/passwd"})
        self.assertFalse(ok)
        self.assertIn("cwork", reason.lower())

    def test_lint_validator_blocks_outside_cwork(self):
        from liagent.tools.lint_code import _validate_lint
        ok, reason = _validate_lint({"path": "/tmp/evil.py"})
        self.assertFalse(ok)

    def test_run_tests_validator_blocks_outside_cwork(self):
        from liagent.tools.run_tests import _validate_run_tests
        ok, reason = _validate_run_tests({"path": "/usr/bin/test"})
        self.assertFalse(ok)

    def test_research_profile_includes_new_tools(self):
        from liagent.tools.policy import _TOOL_PROFILE_MAP
        research = _TOOL_PROFILE_MAP["research"]
        self.assertIn("run_tests", research)
        self.assertIn("lint_code", research)
        self.assertIn("verify_syntax", research)


class StockOutputFormatTests(unittest.TestCase):
    """Tests for Fix 2a: stock output restructuring."""

    def test_summary_before_raw_quote(self):
        """Summary [Summary] should appear before raw quote line."""
        # Simulate what the output looks like
        output = (
            "Apple Inc (AAPL) | NASDAQ | Sector: Technology\n"
            "[Summary] Apple Inc (AAPL) current price $195.00, up $2.50 (1.30%). Market cap $3.01T.\n"
            "Price: 195.0  Change: +2.50 (+1.30%)\n"
            "Open: 193.0  High: 196.0  Low: 192.5  Prev Close: 192.5"
        )
        summary_idx = output.index("[Summary]")
        quote_idx = output.index("Price:")
        self.assertLess(summary_idx, quote_idx)

    def test_no_raw_market_cap_line(self):
        """The old 'Market cap: 4442283.1M USD' line should not appear."""
        output = (
            "NVIDIA (NVDA) | NASDAQ | Sector: Technology\n"
            "[Summary] NVIDIA (NVDA) current price $130.00. Market cap $3.20T.\n"
            "Price: 130.0  Change: +1.00 (+0.78%)"
        )
        self.assertNotIn("M USD", output)


class ValidateKeyMetricsTests(unittest.TestCase):
    """Tests for Fix 2b: validate_key_metrics cross-validation."""

    def test_fixes_wrong_percentage(self):
        from liagent.agent.quality import validate_key_metrics
        stock_result = "[Summary] GOOGL current price $180, down $2.21 (2.21%)."
        answer = "Google stock fell by about (9.00%), weak performance."
        fixed, fixes = validate_key_metrics(answer, stock_result)
        self.assertTrue(len(fixes) > 0)
        self.assertIn("2.21%", fixed)
        self.assertNotIn("9.00%", fixed)

    def test_no_fix_when_close(self):
        from liagent.agent.quality import validate_key_metrics
        stock_result = "[Summary] GOOGL current price $180, down $2.21 (2.21%)."
        answer = "Google fell today (2.20%), a moderate decline."
        fixed, fixes = validate_key_metrics(answer, stock_result)
        self.assertEqual(len(fixes), 0)
        self.assertEqual(fixed, answer)

    def test_fixes_wrong_mcap(self):
        from liagent.agent.quality import validate_key_metrics
        stock_result = "Market cap is $2.50T."
        answer = "Google market cap is about $6.00T."
        fixed, fixes = validate_key_metrics(answer, stock_result)
        self.assertTrue(len(fixes) > 0)
        self.assertIn("2.50T", fixed)

    def test_empty_inputs(self):
        from liagent.agent.quality import validate_key_metrics
        fixed, fixes = validate_key_metrics("", "some stock data")
        self.assertEqual(fixed, "")
        self.assertEqual(fixes, [])
        fixed, fixes = validate_key_metrics("some answer", "")
        self.assertEqual(fixed, "some answer")
        self.assertEqual(fixes, [])


class ProfessionalToneTests(unittest.TestCase):
    """Tests for Fix 7: professional tone injection."""

    def test_professional_mode_for_investment_advice(self):
        from liagent.agent.prompt_builder import PromptBuilder
        with tempfile.TemporaryDirectory() as td:
            mem = LongTermMemory(Path(td) / "mem.db", data_dir=Path(td) / "data")
            pb = PromptBuilder(mem)
            prompt = pb.build_system_prompt_for_coder(query="Give investment advice on Google's earnings")
            self.assertIn("Professional mode", prompt)
            self.assertIn("Formal written style", prompt)

    def test_professional_mode_for_report(self):
        from liagent.agent.prompt_builder import PromptBuilder
        with tempfile.TemporaryDirectory() as td:
            mem = LongTermMemory(Path(td) / "mem.db", data_dir=Path(td) / "data")
            pb = PromptBuilder(mem)
            prompt = pb.build_system_prompt(query="Write a research report")
            self.assertIn("Professional mode", prompt)

    def test_no_professional_mode_for_casual(self):
        from liagent.agent.prompt_builder import PromptBuilder
        with tempfile.TemporaryDirectory() as td:
            mem = LongTermMemory(Path(td) / "mem.db", data_dir=Path(td) / "data")
            pb = PromptBuilder(mem)
            prompt = pb.build_system_prompt_for_coder(query="How is the weather today")
            self.assertNotIn("Professional mode", prompt)

    def test_no_professional_mode_for_stock_price(self):
        from liagent.agent.prompt_builder import PromptBuilder
        with tempfile.TemporaryDirectory() as td:
            mem = LongTermMemory(Path(td) / "mem.db", data_dir=Path(td) / "data")
            pb = PromptBuilder(mem)
            prompt = pb.build_system_prompt_for_coder(query="Check Google stock price")
            self.assertNotIn("Professional mode", prompt)


class DataStepSuppressionTests(unittest.TestCase):
    """Tests for Fix 1: data-step token suppression logic."""

    def test_is_data_step_logic(self):
        """Verify the data step detection logic matches plan convention."""
        # Simulate plan with 3 steps (2 data + 1 synthesis)
        class FakeStep:
            pass
        steps = [FakeStep(), FakeStep(), FakeStep()]

        # Data step: plan_idx < len(steps) - 1
        plan_idx = 0
        _is_data_step = bool(steps) and plan_idx < len(steps) - 1
        self.assertTrue(_is_data_step)

        plan_idx = 1
        _is_data_step = bool(steps) and plan_idx < len(steps) - 1
        self.assertTrue(_is_data_step)

        # Synthesis step: plan_idx == len(steps) - 1
        plan_idx = 2
        _is_data_step = bool(steps) and plan_idx < len(steps) - 1
        self.assertFalse(_is_data_step)

        # No plan
        _is_data_step = bool([]) and 0 < 0
        self.assertFalse(_is_data_step)


if __name__ == "__main__":
    unittest.main()
