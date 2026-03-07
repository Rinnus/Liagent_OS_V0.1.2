import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from liagent.config import AppConfig


class _FakeEngine:
    def __init__(self):
        self.config = AppConfig()
        self.config.llm.max_tokens = 128
        self.config.llm.temperature = 0.2
        self.config.tool_profile = "research"
        self.config.mcp_discovery.enabled = False
        self.config.mcp_servers = []
        self.tool_parser = SimpleNamespace(
            parse=lambda *_a, **_kw: None,
            format_schemas=lambda tools: tools,
        )

    async def generate_llm_routed(self, *args, **kwargs):
        yield "OK"

    async def generate_llm(self, *args, **kwargs):
        yield "OK"

    async def generate_text(self, *args, **kwargs):
        yield "OK"

    async def stream_text(self, *args, **kwargs):
        yield "OK"

    async def generate_extraction(self, *args, **kwargs):
        return "[]"

    async def generate_reasoning(self, *args, **kwargs):
        return "OK"


class _FakeLongTermMemory:
    def __init__(self, **kwargs):
        self.db_path = kwargs.get("db_path") or Path(tempfile.gettempdir()) / "brain-session-runtime.db"
        self.journal = kwargs.get("journal")
        self.feedback_calls: list[dict] = []

    def set_embedder(self, *args, **kwargs):
        return None

    def get_recent_summaries(self, *args, **kwargs):
        return []

    def get_all_facts(self, *args, **kwargs):
        return []

    def get_relevant_facts(self, *args, **kwargs):
        return []

    def get_relevant_evidence(self, *args, **kwargs):
        return []

    def decay_confidence(self, *args, **kwargs):
        return None

    def prune_memory(self, *args, **kwargs):
        return None

    def prune_old_records(self, *args, **kwargs):
        return None

    def save_feedback(self, **kwargs):
        self.feedback_calls.append(kwargs)

    def apply_source_confidence(self, facts):
        return facts

    def detect_conflicts(self, *args, **kwargs):
        return None

    def close(self):
        return None


class _FakeExperience:
    def __init__(self, *args, **kwargs):
        self.journal = kwargs.get("journal")

    def match(self, query):
        return None

    def record_outcome(self, *args, **kwargs):
        return None

    def sync_from_markdown(self):
        return None

    def sync_to_markdown(self):
        return None

    async def generate_skill(self, *args, **kwargs):
        return None


class _FakeMetrics:
    def __init__(self):
        return None

    def log_turn(self, *args, **kwargs):
        return None

    def log_runtime(self, *args, **kwargs):
        return None


class BrainSessionRuntimeTests(unittest.TestCase):
    def _make_brain(self):
        from liagent.agent.brain import AgentBrain

        with (
            patch("liagent.agent.brain.LongTermMemory", _FakeLongTermMemory),
            patch("liagent.agent.brain.ExperienceMemory", _FakeExperience),
            patch("liagent.agent.brain.InteractionMetrics", _FakeMetrics),
        ):
            return AgentBrain(_FakeEngine())

    def test_clear_memory_is_scoped_per_session(self):
        brain = self._make_brain()

        with brain._session_runtime("sess-a"):
            brain.memory.add("user", "hello from A")
            brain.pending_confirmations["tok-a"] = {"created_at": None}
            session_a_id = brain.session_id

        with brain._session_runtime("sess-b"):
            brain.memory.add("user", "hello from B")
            brain.pending_confirmations["tok-b"] = {"created_at": None}
            session_b_id = brain.session_id

        asyncio.run(brain.clear_memory_for_session("sess-a"))

        with brain._session_runtime("sess-a"):
            self.assertEqual(brain.memory.turn_count(), 0)
            self.assertEqual(brain.pending_confirmations, {})
            self.assertNotEqual(brain.session_id, session_a_id)

        with brain._session_runtime("sess-b"):
            self.assertEqual(brain.memory.turn_count(), 1)
            self.assertEqual(brain.memory.last_user_message(), "hello from B")
            self.assertIn("tok-b", brain.pending_confirmations)
            self.assertEqual(brain.session_id, session_b_id)

    def test_record_user_feedback_uses_session_specific_session_id(self):
        brain = self._make_brain()

        with brain._session_runtime("sess-a"):
            session_a_id = brain.session_id
        with brain._session_runtime("sess-b"):
            session_b_id = brain.session_id

        asyncio.run(
            brain.record_user_feedback(
                query="q",
                answer="a",
                tool_used=None,
                feedback="positive",
                session_key="sess-b",
            )
        )

        self.assertEqual(len(brain.long_term.feedback_calls), 1)
        self.assertEqual(brain.long_term.feedback_calls[0]["session_id"], session_b_id)
        self.assertNotEqual(session_a_id, session_b_id)

    def test_resolve_confirmation_uses_session_specific_pending_bucket(self):
        brain = self._make_brain()

        state_a = brain._get_or_create_session_state("sess-a")
        state_b = brain._get_or_create_session_state("sess-b")
        state_a.pending_confirmations["tok-a"] = {"created_at": None}
        state_b.pending_confirmations["tok-b"] = {"created_at": None}

        fake_resolve = AsyncMock(return_value={"status": "rejected"})
        with patch("liagent.agent.brain._resolve_confirmation", new=fake_resolve):
            asyncio.run(brain.resolve_confirmation("tok-a", approved=False, session_key="sess-a"))

        self.assertIs(fake_resolve.await_args.kwargs["pending_confirmations"], state_a.pending_confirmations)
        self.assertIsNot(fake_resolve.await_args.kwargs["pending_confirmations"], state_b.pending_confirmations)

    def test_conversation_id_isolated_per_session_scope(self):
        brain = self._make_brain()

        with brain._session_runtime("sess-a"):
            conv_a = brain._conversation_id
        with brain._session_runtime("sess-b"):
            conv_b = brain._conversation_id
        with brain._session_runtime():
            conv_default = brain._conversation_id

        self.assertEqual(conv_a, "session:sess-a")
        self.assertEqual(conv_b, "session:sess-b")
        self.assertNotEqual(conv_a, conv_b)
        self.assertNotEqual(conv_default, conv_a)

    def test_raw_keyword_tool_call_does_not_leak_to_tokens(self):
        from liagent.agent.brain import AgentBrain

        class _RawCallEngine(_FakeEngine):
            def __init__(self):
                super().__init__()
                self._stream_responses = [
                    ["I will search now.\n", 'web_search(query="GOOG stock", timelimit="d")'],
                    ["Support is around 300."],
                ]

            async def stream_text(self, *args, **kwargs):
                for token in self._stream_responses.pop(0):
                    yield token

        with (
            patch("liagent.agent.brain.LongTermMemory", _FakeLongTermMemory),
            patch("liagent.agent.brain.ExperienceMemory", _FakeExperience),
            patch("liagent.agent.brain.InteractionMetrics", _FakeMetrics),
        ):
            brain = AgentBrain(_RawCallEngine())

        async def fake_execute_with_fallback(tool_def, tool_args, *, get_tool_fn=None):
            return "1. GOOG overview\nURL: https://example.com", False, "", "web_search", tool_args

        brain._tool_executor.execute_with_fallback = AsyncMock(side_effect=fake_execute_with_fallback)

        async def run():
            events = []
            async for event in brain.run("check GOOG support"):
                events.append(event)
            return events

        events = asyncio.run(run())
        token_text = "".join(ev[1] for ev in events if ev and ev[0] == "token")
        self.assertNotIn('web_search(query="GOOG stock", timelimit="d")', token_text)
        self.assertTrue(any(ev[0] == "tool_start" and ev[1] == "web_search" for ev in events))
        self.assertTrue(any(ev[0] == "tool_result" and ev[1] == "web_search" for ev in events))
        done_events = [ev for ev in events if ev and ev[0] == "done"]
        self.assertEqual(len(done_events), 1)
        self.assertIn("Support is around 300.", done_events[0][1])
