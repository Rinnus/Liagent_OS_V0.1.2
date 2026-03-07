# tests/test_heartbeat.py
"""Tests for HEARTBEAT.md parsing, pre-check, and ActionGate."""

import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path

from liagent.agent.heartbeat import (
    HeartbeatConfig,
    parse_heartbeat_md,
    ActionGate,
    CandidateAction,
    FileWatchPreCheck,
    CursorStore,
)


class HeartbeatConfigParsingTests(unittest.TestCase):
    def test_parse_with_frontmatter(self):
        md = "---\nactive_hours: \"08:00-22:00\"\ntimezone: \"Asia/Shanghai\"\ncooldown_minutes: 30\nchannels: [\"websocket\"]\nmax_actions_per_run: 3\ndry_run: true\naction_allowlist:\n  - notify_user\n  - web_search\n---\n\nCheck inbox for new files."
        config = parse_heartbeat_md(md)
        self.assertEqual(config.active_hours, "08:00-22:00")
        self.assertEqual(config.timezone, "Asia/Shanghai")
        self.assertTrue(config.dry_run)
        self.assertIn("notify_user", config.action_allowlist)
        self.assertIn("Check inbox", config.instructions)

    def test_parse_without_frontmatter(self):
        config = parse_heartbeat_md("Just check my inbox.")
        self.assertTrue(config.dry_run)
        self.assertEqual(config.instructions, "Just check my inbox.")

    def test_defaults(self):
        config = parse_heartbeat_md("")
        self.assertTrue(config.dry_run)
        self.assertEqual(config.max_actions_per_run, 3)


class ActionGateTests(unittest.TestCase):
    def _config(self, **kw):
        defaults = dict(active_hours="00:00-23:59", timezone="UTC", cooldown_minutes=30,
                        channels=[], max_actions_per_run=3, dry_run=False,
                        action_allowlist=["notify_user", "web_search"], instructions="")
        defaults.update(kw)
        return HeartbeatConfig(**defaults)

    def test_execute_allowed(self):
        gate = ActionGate()
        action = CandidateAction("notify_user", "t:1", "test", "low")
        self.assertEqual(gate.evaluate(action, self._config()), "execute")

    def test_dry_run(self):
        gate = ActionGate()
        action = CandidateAction("notify_user", "t:1", "test", "low")
        self.assertEqual(gate.evaluate(action, self._config(dry_run=True)), "dry_run_log")

    def test_blocked_outside_allowlist(self):
        gate = ActionGate()
        action = CandidateAction("write_file", "t:1", "test", "medium")
        self.assertEqual(gate.evaluate(action, self._config(action_allowlist=["notify_user"])), "blocked")

    def test_dedup(self):
        gate = ActionGate(dedup_window_sec=3600)
        action = CandidateAction("notify_user", "same:key", "test", "low")
        self.assertEqual(gate.evaluate(action, self._config()), "execute")
        self.assertEqual(gate.evaluate(action, self._config()), "blocked")

    def test_dedup_uses_semantic_key_when_llm_keys_differ(self):
        gate = ActionGate(dedup_window_sec=3600)
        a1 = CandidateAction(
            "web_search", "llm:key:1", "search latest news", "low",
            tool_name="web_search", tool_args={"query": "latest news"},
        )
        a2 = CandidateAction(
            "web_search", "llm:key:2", "search latest news", "low",
            tool_name="web_search", tool_args={"query": "latest news"},
        )
        self.assertEqual(gate.evaluate(a1, self._config(action_allowlist=["web_search"])), "execute")
        self.assertEqual(gate.evaluate(a2, self._config(action_allowlist=["web_search"])), "blocked")

    def test_high_risk_confirmation(self):
        gate = ActionGate()
        action = CandidateAction("notify_user", "t:2", "test", "high")
        self.assertEqual(gate.evaluate(action, self._config()), "needs_confirmation")

    def test_high_risk_confirmation_is_case_insensitive(self):
        gate = ActionGate()
        action = CandidateAction("notify_user", "t:2", "test", "HIGH")
        self.assertEqual(gate.evaluate(action, self._config()), "needs_confirmation")

    def test_dedup_during_pending_confirmation(self):
        """Same action_key should be blocked while pending confirmation."""
        gate = ActionGate(dedup_window_sec=3600)
        action = CandidateAction("notify_user", "confirm:key", "test", "high")
        self.assertEqual(gate.evaluate(action, self._config()), "needs_confirmation")
        self.assertEqual(gate.evaluate(action, self._config()), "blocked")


class CursorStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.store = CursorStore(db_path=self.db_path)

    def tearDown(self):
        self.tmp.close()
        self.db_path.unlink(missing_ok=True)

    def test_get_returns_none_initially(self):
        self.assertIsNone(self.store.get("email"))

    def test_set_and_get(self):
        self.store.set("filewatch", '{"hash": "abc123"}')
        self.assertEqual(self.store.get("filewatch"), '{"hash": "abc123"}')

    def test_update_overwrites(self):
        self.store.set("filewatch", "v1")
        self.store.set("filewatch", "v2")
        self.assertEqual(self.store.get("filewatch"), "v2")


class FileWatchPreCheckTests(unittest.TestCase):
    def test_detects_new_file(self):
        with tempfile.TemporaryDirectory() as d:
            checker = FileWatchPreCheck(watch_dirs=[d])
            r1 = asyncio.run(checker.check(cursor=None))
            (Path(d) / "test.txt").write_text("hello")
            r2 = asyncio.run(checker.check(cursor=r1.new_cursor))
            self.assertTrue(r2.has_changes)
            self.assertIn("test.txt", r2.change_summary)


# --- Additional tests for HeartbeatRunner, prompt, memory injection, metrics ---

from liagent.agent.heartbeat import (
    HEARTBEAT_SYSTEM_PROMPT,
    HeartbeatMetrics,
    HeartbeatRunner,
    inject_heartbeat_context,
    format_evidence_for_prompt,
)


class HeartbeatSystemPromptTests(unittest.TestCase):
    def test_prompt_contains_security_rules(self):
        self.assertIn("SECURITY RULES", HEARTBEAT_SYSTEM_PROMPT)
        self.assertIn("{allowlist}", HEARTBEAT_SYSTEM_PROMPT)
        self.assertIn("{max_actions}", HEARTBEAT_SYSTEM_PROMPT)

    def test_prompt_format_substitution(self):
        formatted = HEARTBEAT_SYSTEM_PROMPT.format(
            allowlist="notify_user, web_search", max_actions=3
        )
        self.assertIn("notify_user, web_search", formatted)
        self.assertIn("3", formatted)


class InjectHeartbeatContextTests(unittest.TestCase):
    def test_budget_enforcement(self):
        """inject_heartbeat_context respects max_chars budget."""
        from liagent.agent.memory import EvidenceChunk
        from unittest.mock import MagicMock

        mock_ltm = MagicMock()
        mock_ltm.get_relevant_evidence.return_value = [
            EvidenceChunk("e1", "fact", "ref1", None, None, "a" * 1000, 0.9, "2026-01-01"),
            EvidenceChunk("e2", "fact", "ref2", None, None, "b" * 1000, 0.8, "2026-01-01"),
            EvidenceChunk("e3", "fact", "ref3", None, None, "c" * 1000, 0.7, "2026-01-01"),
        ]
        result = asyncio.run(
            inject_heartbeat_context(mock_ltm, "test query", max_chars=2000)
        )
        self.assertEqual(len(result), 2)  # Third chunk exceeds budget

    def test_empty_evidence(self):
        from unittest.mock import MagicMock
        mock_ltm = MagicMock()
        mock_ltm.get_relevant_evidence.return_value = []
        result = asyncio.run(
            inject_heartbeat_context(mock_ltm, "test")
        )
        self.assertEqual(len(result), 0)


class FormatEvidenceTests(unittest.TestCase):
    def test_structural_markers(self):
        from liagent.agent.memory import EvidenceChunk
        chunks = [
            EvidenceChunk("e1", "fact", "memory:key_facts:test", None, None, "test snippet", 0.85, "2026-01-15T00:00:00"),
        ]
        text = format_evidence_for_prompt(chunks)
        self.assertIn("[RECALLED_EVIDENCE", text)
        self.assertIn("[/RECALLED_EVIDENCE]", text)
        self.assertIn("source=memory:key_facts:test", text)
        self.assertIn("score=0.85", text)

    def test_empty_chunks(self):
        self.assertEqual(format_evidence_for_prompt([]), "")


class HeartbeatMetricsTests(unittest.TestCase):
    def test_defaults(self):
        m = HeartbeatMetrics(run_id="test-1")
        self.assertEqual(m.precheck_sources_checked, 0)
        self.assertFalse(m.llm_invoked)
        self.assertEqual(m.actions_proposed, 0)


class HeartbeatRunnerTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def _make_config(self, **kw):
        defaults = dict(active_hours="00:00-23:59", timezone="UTC", cooldown_minutes=30,
                        channels=[], max_actions_per_run=3, dry_run=True,
                        action_allowlist=["notify_user"], instructions="Check inbox")
        defaults.update(kw)
        return HeartbeatConfig(**defaults)

    def test_dry_run_no_execution(self):
        """With dry_run=True, no actions should be executed."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        store = CursorStore(db_path=db_path)

        runner = HeartbeatRunner(
            config=self._make_config(dry_run=True),
            engine=None,  # No LLM needed for dry-run skip
            long_term_memory=None,
            notification_router=None,
            cursor_store=store,
        )
        metrics = self._run(runner.run())
        self.assertEqual(metrics.actions_queued, 0)
        db_path.unlink(missing_ok=True)

    def test_no_changes_no_instructions_skips_llm(self):
        """If no pre-checks detect changes and no instructions, LLM is not invoked."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        store = CursorStore(db_path=db_path)

        runner = HeartbeatRunner(
            config=self._make_config(instructions=""),
            engine=None,
            long_term_memory=None,
            notification_router=None,
            cursor_store=store,
            pre_checks=[],
        )
        metrics = self._run(runner.run())
        self.assertFalse(metrics.llm_invoked)
        db_path.unlink(missing_ok=True)

    def test_parse_actions_valid_json(self):
        """_parse_actions handles valid JSON array with tool_name/tool_args."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        store = CursorStore(db_path=db_path)
        runner = HeartbeatRunner(
            config=self._make_config(action_allowlist=["notify_user"]),
            engine=None, long_term_memory=None,
            notification_router=None, cursor_store=store,
        )
        actions = runner._parse_actions(
            '[{"action_type": "notify_user", "action_key": "k1", '
            '"description": "test", "risk_level": "low", '
            '"tool_name": "notify_user", "tool_args": {"msg": "hi"}}]'
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "notify_user")
        self.assertEqual(actions[0].tool_name, "notify_user")
        self.assertEqual(actions[0].tool_args, {"msg": "hi"})
        db_path.unlink(missing_ok=True)

    def test_parse_actions_with_tool_payload(self):
        """_parse_actions extracts tool_name and tool_args into CandidateAction."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        store = CursorStore(db_path=db_path)
        runner = HeartbeatRunner(
            config=self._make_config(action_allowlist=["web_search", "notify_user"]),
            engine=None, long_term_memory=None,
            notification_router=None, cursor_store=store,
        )
        actions = runner._parse_actions(
            '[{"action_type": "web_search", "action_key": "ws:1", '
            '"description": "search latest news", "risk_level": "low", '
            '"tool_name": "web_search", "tool_args": {"query": "latest news"}}, '
            '{"action_type": "notify_user", "action_key": "n:1", '
            '"description": "send alert", "risk_level": "low", '
            '"tool_name": "notify_user", "tool_args": {"message": "alert!"}}]'
        )
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].tool_name, "web_search")
        self.assertEqual(actions[0].tool_args, {"query": "latest news"})
        self.assertEqual(actions[1].tool_name, "notify_user")
        self.assertEqual(actions[1].tool_args, {"message": "alert!"})
        db_path.unlink(missing_ok=True)

    def test_parse_actions_rejects_unlisted_tool(self):
        """_parse_actions drops actions whose tool_name is not in the allowlist."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        store = CursorStore(db_path=db_path)
        runner = HeartbeatRunner(
            config=self._make_config(action_allowlist=["notify_user"]),
            engine=None, long_term_memory=None,
            notification_router=None, cursor_store=store,
        )
        actions = runner._parse_actions(
            '[{"action_type": "hack_server", "action_key": "h:1", '
            '"description": "bad action", "risk_level": "low", '
            '"tool_name": "hack_server", "tool_args": {}}]'
        )
        self.assertEqual(len(actions), 0)
        db_path.unlink(missing_ok=True)

    def test_parse_actions_missing_tool_name_dropped(self):
        """_parse_actions drops actions where tool_name is empty or missing."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        store = CursorStore(db_path=db_path)
        runner = HeartbeatRunner(
            config=self._make_config(action_allowlist=["notify_user"]),
            engine=None, long_term_memory=None,
            notification_router=None, cursor_store=store,
        )
        # Missing tool_name entirely
        actions1 = runner._parse_actions(
            '[{"action_type": "notify_user", "action_key": "k1", '
            '"description": "test", "risk_level": "low"}]'
        )
        self.assertEqual(len(actions1), 0)

        # Empty tool_name
        actions2 = runner._parse_actions(
            '[{"action_type": "notify_user", "action_key": "k2", '
            '"description": "test", "risk_level": "low", '
            '"tool_name": "", "tool_args": {}}]'
        )
        self.assertEqual(len(actions2), 0)
        db_path.unlink(missing_ok=True)

    def test_build_execution_prompt(self):
        """_build_execution_prompt builds deterministic prompt from CandidateAction."""
        action = CandidateAction(
            action_type="web_search",
            action_key="ws:1",
            description="search latest news",
            risk_level="low",
            tool_name="web_search",
            tool_args={"query": "latest news"},
        )
        prompt = HeartbeatRunner._build_execution_prompt(action)
        self.assertIn("Tool: web_search", prompt)
        self.assertIn('"query": "latest news"', prompt)
        self.assertIn("Context: search latest news", prompt)
        self.assertIn("Do NOT use any tool other than web_search", prompt)

    def test_parse_actions_invalid_json(self):
        """_parse_actions returns empty list for invalid input."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        store = CursorStore(db_path=db_path)
        runner = HeartbeatRunner(
            config=self._make_config(),
            engine=None, long_term_memory=None,
            notification_router=None, cursor_store=store,
        )
        self.assertEqual(runner._parse_actions("not json"), [])
        self.assertEqual(runner._parse_actions(""), [])
        db_path.unlink(missing_ok=True)

    def test_metrics_persisted(self):
        """Metrics are saved to SQLite after a run."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(tmp.name)
        store = CursorStore(db_path=db_path)

        runner = HeartbeatRunner(
            config=self._make_config(instructions=""),
            engine=None, long_term_memory=None,
            notification_router=None, cursor_store=store,
        )
        self._run(runner.run())
        count = store._conn.execute("SELECT COUNT(*) FROM heartbeat_metrics").fetchone()[0]
        self.assertGreaterEqual(count, 1)
        db_path.unlink(missing_ok=True)


class HeartbeatExecuteCallbackTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def _make_config(self, **kw):
        defaults = dict(active_hours="00:00-23:59", timezone="UTC", cooldown_minutes=30,
                        channels=[], max_actions_per_run=3, dry_run=False,
                        action_allowlist=["web_search", "notify_user"], instructions="Check things")
        defaults.update(kw)
        return HeartbeatConfig(**defaults)

    def test_execute_callback_called_for_low_risk(self):
        """on_execute is called when ActionGate returns 'execute'."""
        from liagent.agent.heartbeat import ExecuteResult
        calls = []
        async def fake_execute(action, prompt, allowed_tools):
            calls.append({"action": action, "prompt": prompt, "tools": allowed_tools})
            return ExecuteResult(run_id="r1", status="queued")

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        store = CursorStore(db_path=Path(tmp.name))
        from unittest.mock import AsyncMock
        engine = AsyncMock()
        engine.generate_reasoning = AsyncMock(return_value=(
            '[{"action_type": "web_search", "action_key": "q1", "description": "search",'
            ' "risk_level": "low", "tool_name": "web_search", "tool_args": {"query": "test"}}]'
        ))
        runner = HeartbeatRunner(
            config=self._make_config(), engine=engine, long_term_memory=None,
            notification_router=None, cursor_store=store, on_execute=fake_execute,
        )
        metrics = self._run(runner.run())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["action"].tool_name, "web_search")
        self.assertIn("web_search", calls[0]["prompt"])
        self.assertEqual(metrics.actions_queued, 1)
        Path(tmp.name).unlink(missing_ok=True)

    def test_needs_confirmation_callback_called_for_high_risk(self):
        """on_needs_confirmation is called for high-risk actions."""
        confirm_calls = []
        async def fake_confirm(action, prompt, allowed_tools, reason):
            confirm_calls.append(action.action_key)
            return "token-123"

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        store = CursorStore(db_path=Path(tmp.name))
        from unittest.mock import AsyncMock
        engine = AsyncMock()
        engine.generate_reasoning = AsyncMock(return_value=(
            '[{"action_type": "web_search", "action_key": "q2", "description": "risky",'
            ' "risk_level": "high", "tool_name": "web_search", "tool_args": {"query": "test"}}]'
        ))
        runner = HeartbeatRunner(
            config=self._make_config(), engine=engine, long_term_memory=None,
            notification_router=None, cursor_store=store, on_needs_confirmation=fake_confirm,
        )
        metrics = self._run(runner.run())
        self.assertEqual(len(confirm_calls), 1)
        self.assertEqual(metrics.actions_pending_confirm, 1)
        Path(tmp.name).unlink(missing_ok=True)

    def test_no_callbacks_falls_back_to_notify(self):
        """Without callbacks, execute branch falls back to notification."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        store = CursorStore(db_path=Path(tmp.name))
        notified = []
        from unittest.mock import AsyncMock
        engine = AsyncMock()
        engine.generate_reasoning = AsyncMock(return_value=(
            '[{"action_type": "notify_user", "action_key": "n1", "description": "hi",'
            ' "risk_level": "low", "tool_name": "notify_user", "tool_args": {"message": "hi"}}]'
        ))
        router = AsyncMock()
        router.dispatch = AsyncMock(side_effect=lambda msg, **kw: notified.append(msg) or True)
        runner = HeartbeatRunner(
            config=self._make_config(), engine=engine, long_term_memory=None,
            notification_router=router, cursor_store=store,
        )
        metrics = self._run(runner.run())
        self.assertGreaterEqual(len(notified), 1)
        Path(tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
