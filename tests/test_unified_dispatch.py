"""Tests for the unified LLM-driven dispatch path."""

import asyncio
import inspect
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from liagent.skills.router import SkillConfig, RuntimeBudget, BudgetOverride, STANDARD_CHAT, build_runtime_budget


class BrainRunSignatureTests(unittest.TestCase):
    """Verify brain.run() accepts BudgetOverride, not BudgetSlice."""

    def test_run_has_budget_override_param(self):
        from liagent.agent.brain import AgentBrain
        sig = inspect.signature(AgentBrain.run)
        params = list(sig.parameters.keys())
        self.assertIn("budget", params)

    def test_brain_no_orchestrator_budget_import(self):
        """brain.py must not import from orchestrator.budget."""
        import liagent.agent.brain as brain_mod
        with open(brain_mod.__file__) as fh:
            source = fh.read()
        self.assertNotIn("from ..orchestrator.budget", source)

    def test_brain_no_service_tier_import(self):
        """brain.py must not import from service_tier."""
        import liagent.agent.brain as brain_mod
        with open(brain_mod.__file__) as fh:
            source = fh.read()
        self.assertNotIn("from .service_tier", source)

    def test_brain_no_is_chinese_heavy(self):
        """_is_chinese_heavy removed (LLM handles multilingual queries)."""
        from liagent.agent.brain import AgentBrain
        self.assertFalse(hasattr(AgentBrain, "_is_chinese_heavy"))


class RuntimeBudgetContractTests(unittest.TestCase):
    """Verify RuntimeBudget satisfies all downstream field expectations."""

    def test_policy_gate_field(self):
        """policy_gate.py:223 reads ctx.budget.enable_policy_review"""
        budget = build_runtime_budget(STANDARD_CHAT)
        self.assertFalse(budget.enable_policy_review)

    def test_tool_orchestrator_fields(self):
        """tool_orchestrator.py:436 reads ctx.budget.llm_max_tokens, .llm_temperature"""
        budget = build_runtime_budget(STANDARD_CHAT)
        self.assertGreater(budget.llm_max_tokens, 0)
        self.assertGreater(budget.llm_temperature, 0.0)

    def test_engine_manager_tier(self):
        """engine_manager.py:644 checks tier == 'deep_task'"""
        budget = build_runtime_budget(STANDARD_CHAT)
        self.assertEqual(budget.tier, "standard_chat")

    def test_budget_override_applied(self):
        """BudgetOverride clamps budget fields."""
        budget = build_runtime_budget(STANDARD_CHAT)
        override = BudgetOverride(max_steps=4, max_tool_calls=2, timeout_ms=60_000,
                                  allowed_tools={"read_file", "write_file"})
        budget.apply_override(override)
        self.assertEqual(budget.max_steps, 4)
        self.assertEqual(budget.max_tool_calls, 2)


class ThinOrchestratorTests(unittest.TestCase):
    """Test the thin orchestrator wrapper."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_dispatch_yields_dispatch_event_first(self):
        from liagent.orchestrator.orchestrator import Orchestrator
        from liagent.orchestrator.events import AgentEvent

        mock_engine = MagicMock()
        mock_brain = MagicMock()

        async def fake_run(query, images=None, low_latency=False, session_key=None):
            yield ("done", "test answer")

        mock_brain.run = fake_run
        orch = Orchestrator(engine=mock_engine, brain=mock_brain)

        events = []
        async def collect():
            async for e in orch.dispatch("hello"):
                events.append(e)
        self._run(collect())

        self.assertGreater(len(events), 0)
        self.assertIsInstance(events[0], AgentEvent)
        self.assertEqual(events[0].type, "dispatch")

    def test_dispatch_wraps_brain_events(self):
        from liagent.orchestrator.orchestrator import Orchestrator
        from liagent.orchestrator.events import AgentEvent

        mock_engine = MagicMock()
        mock_brain = MagicMock()

        async def fake_run(query, images=None, low_latency=False, session_key=None):
            yield ("token", "hello ")
            yield ("token", "world")
            yield ("done", "hello world")

        mock_brain.run = fake_run
        orch = Orchestrator(engine=mock_engine, brain=mock_brain)

        events = []
        async def collect():
            async for e in orch.dispatch("test"):
                events.append(e)
        self._run(collect())

        # dispatch + 3 brain events + quality_gate = 5
        self.assertEqual(len(events), 5)
        for e in events:
            self.assertIsInstance(e, AgentEvent)
        legacy = events[1].to_legacy_tuple()
        self.assertEqual(legacy, ("token", "hello "))
        # Last event should be quality_gate from grounding gate
        self.assertEqual(events[-1].type, "quality_gate")

    def test_dispatch_shutdown(self):
        from liagent.orchestrator.orchestrator import Orchestrator

        mock_engine = MagicMock()
        mock_brain = MagicMock()
        mock_brain.shutdown = AsyncMock()
        orch = Orchestrator(engine=mock_engine, brain=mock_brain)
        self._run(orch.shutdown())
        mock_brain.shutdown.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
