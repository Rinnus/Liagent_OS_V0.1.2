"""Regression tests for the brain.py 4-layer split modules.

Covers: run_context, policy_gate, tool_orchestrator, response_guard.
These tests ensure event/memory consistency that the original inline code had.
"""

import asyncio
import json
import unittest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from liagent.agent.run_context import RunContext
from liagent.agent.policy_gate import PolicyDecision, evaluate_tool_policy
from liagent.agent.tool_orchestrator import ToolExecResult, execute_and_record, maybe_vision_analysis
from liagent.agent.response_guard import GuardResult, check_response


# ── Shared test helpers ────────────────────────────────────────────────


def _make_ctx(**overrides) -> RunContext:
    """Build a RunContext with sensible defaults for testing."""
    budget = MagicMock()
    budget.enable_policy_review = False
    budget.llm_max_tokens = 256
    budget.llm_temperature = 0.2
    budget.tier = "standard_chat"
    ctx = RunContext(
        start_ts=1000.0,
        user_input="test input",
        budget=budget,
        max_steps=8,
        max_tool_calls=3,
    )
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _make_handle_policy_block():
    """Return a mock handle_policy_block_fn that records calls and returns proper events."""
    calls = []

    def handler(tool_name, tool_args, blocked_reason, full_response, hint_msg):
        calls.append({
            "tool_name": tool_name,
            "blocked_reason": blocked_reason,
            "hint_msg": hint_msg,
        })
        observation = f"[policy_blocked] {blocked_reason}"
        events = [
            ("policy_blocked", tool_name, blocked_reason),
            ("tool_result", tool_name, observation),
        ]
        return observation, "clean", events

    handler.calls = calls
    return handler


def _make_tool_policy():
    """Build a mock ToolPolicy."""
    policy = MagicMock()
    policy.evaluate.return_value = (True, "allowed")
    policy.audit.return_value = None
    policy.capability_summary.return_value = ""
    return policy


def _make_planner():
    """Build a mock TaskPlanner."""
    planner = MagicMock()
    planner.allow_followup_tool = AsyncMock(return_value=(True, ""))
    planner.review_tool_action = AsyncMock()
    return planner


# ── RunContext tests ───────────────────────────────────────────────────


class RunContextTests(unittest.TestCase):
    def test_defaults(self):
        ctx = RunContext()
        self.assertEqual(ctx.tool_calls, 0)
        self.assertEqual(ctx.tool_errors, 0)
        self.assertIsInstance(ctx.quality_issues, list)
        self.assertIsInstance(ctx.tools_used, set)
        self.assertIsInstance(ctx.tool_sig_count, dict)
        self.assertIsInstance(ctx.tool_artifacts, dict)
        self.assertIsInstance(ctx.context_vars, dict)
        self.assertFalse(ctx.copout_retried)
        self.assertFalse(ctx.hallucination_retried)
        self.assertFalse(ctx.unwritten_code_retried)

    def test_mutable_fields_not_shared(self):
        """Each instance must have independent mutable containers."""
        c1 = RunContext()
        c2 = RunContext()
        c1.quality_issues.append("x")
        c1.tools_used.add("y")
        self.assertEqual(len(c2.quality_issues), 0)
        self.assertEqual(len(c2.tools_used), 0)

    def test_reasoning_chain_default_empty(self):
        ctx = RunContext()
        self.assertIsInstance(ctx.reasoning_chain, list)
        self.assertEqual(len(ctx.reasoning_chain), 0)

    def test_reasoning_chain_not_shared(self):
        c1 = RunContext()
        c2 = RunContext()
        c1.reasoning_chain.append({"step": 0})
        self.assertEqual(len(c2.reasoning_chain), 0)

    def test_execution_origin_default(self):
        ctx = RunContext()
        self.assertEqual(ctx.execution_origin, "user")

    def test_goal_id_default_none(self):
        ctx = RunContext()
        self.assertIsNone(ctx.goal_id)


# ── PolicyGate tests ──────────────────────────────────────────────────


class PolicyGateTests(unittest.IsolatedAsyncioTestCase):

    async def test_unknown_tool_emits_events_and_writes_memory(self):
        """Bug #1: unknown tool must return policy_blocked + tool_result events."""
        ctx = _make_ctx(tool_calls=1)
        handler = _make_handle_policy_block()

        with patch("liagent.agent.policy_gate.get_tool", return_value=None):
            decision = await evaluate_tool_policy(
                tool_name="nonexistent_tool",
                tool_args={},
                tool_sig="nonexistent_tool:{}",
                full_response="<tool_call>...</tool_call>",
                confirmed=False,
                ctx=ctx,
                tool_policy=_make_tool_policy(),
                planner=_make_planner(),
                handle_policy_block_fn=handler,
                pending_confirmations={},
                dup_tool_limit=2,
                tool_cache_enabled=True,
                enable_policy_review=False,
                disable_policy_review_in_voice=False,
            )

        self.assertFalse(decision.allowed)
        # Must have events (not empty!)
        self.assertGreater(len(decision.events), 0)
        # Must include tool_result so model gets feedback
        event_types = [ev[0] for ev in decision.events]
        self.assertIn("tool_result", event_types)
        self.assertIn("policy_blocked", event_types)
        # handle_policy_block_fn must have been called (writes memory)
        self.assertEqual(len(handler.calls), 1)
        self.assertIn("unknown tool", handler.calls[0]["blocked_reason"])

    async def test_static_policy_block_emits_events_and_writes_memory(self):
        """Bug #2: static policy block must go through handle_policy_block_fn."""
        ctx = _make_ctx(tool_calls=1)
        handler = _make_handle_policy_block()
        policy = _make_tool_policy()
        policy.evaluate.return_value = (False, "blocked by allowlist")

        mock_tool = MagicMock()
        mock_tool.risk_level = "low"
        mock_tool.capability = MagicMock()

        with patch("liagent.agent.policy_gate.get_tool", return_value=mock_tool):
            decision = await evaluate_tool_policy(
                tool_name="some_tool",
                tool_args={},
                tool_sig="some_tool:{}",
                full_response="<tool_call>...</tool_call>",
                confirmed=False,
                ctx=ctx,
                tool_policy=policy,
                planner=_make_planner(),
                handle_policy_block_fn=handler,
                pending_confirmations={},
                dup_tool_limit=2,
                tool_cache_enabled=True,
                enable_policy_review=False,
                disable_policy_review_in_voice=False,
            )

        self.assertFalse(decision.allowed)
        event_types = [ev[0] for ev in decision.events]
        self.assertIn("tool_result", event_types)
        self.assertIn("policy_blocked", event_types)
        self.assertEqual(len(handler.calls), 1)

    async def test_write_file_missing_content_gets_specific_hint(self):
        """Bug #2 detail: write_file content error must carry actionable hint."""
        ctx = _make_ctx(tool_calls=1)
        handler = _make_handle_policy_block()
        policy = _make_tool_policy()
        policy.evaluate.return_value = (False, "content parameter required")

        mock_tool = MagicMock()
        mock_tool.risk_level = "low"
        mock_tool.capability = MagicMock()

        with patch("liagent.agent.policy_gate.get_tool", return_value=mock_tool):
            decision = await evaluate_tool_policy(
                tool_name="write_file",
                tool_args={"path": "/tmp/test.py"},
                tool_sig="write_file:{path:/tmp/test.py}",
                full_response="<tool_call>...</tool_call>",
                confirmed=False,
                ctx=ctx,
                tool_policy=policy,
                planner=_make_planner(),
                handle_policy_block_fn=handler,
                pending_confirmations={},
                dup_tool_limit=2,
                tool_cache_enabled=True,
                enable_policy_review=False,
                disable_policy_review_in_voice=False,
            )

        self.assertFalse(decision.allowed)
        self.assertEqual(len(handler.calls), 1)
        # The hint_msg must contain write_file content guidance
        self.assertIn("content", handler.calls[0]["hint_msg"])
        self.assertIn("path", handler.calls[0]["hint_msg"])

    async def test_write_file_enters_single_step_confirmation_flow(self):
        ctx = _make_ctx(tool_calls=1)
        handler = _make_handle_policy_block()
        policy = _make_tool_policy()
        policy.evaluate.return_value = (False, "confirmation required for risk=medium")
        policy.confirmation_brief.return_value = {
            "tool": "write_file",
            "risk_level": "medium",
            "stage": 1,
            "required_stage": 1,
            "capability": "filesystem=true",
        }

        mock_tool = MagicMock()
        mock_tool.name = "write_file"
        mock_tool.risk_level = "medium"
        mock_tool.capability = MagicMock()
        mock_tool.capability.data_classification = "public"

        pending = {}
        with patch("liagent.agent.policy_gate.get_tool", return_value=mock_tool):
            decision = await evaluate_tool_policy(
                tool_name="write_file",
                tool_args={"path": "/tmp/cwork/test.py", "content": "print('x')"},
                tool_sig="write_file:{path:/tmp/cwork/test.py}",
                full_response="<tool_call>...</tool_call>",
                confirmed=False,
                ctx=ctx,
                tool_policy=policy,
                planner=_make_planner(),
                handle_policy_block_fn=handler,
                pending_confirmations=pending,
                dup_tool_limit=2,
                tool_cache_enabled=True,
                enable_policy_review=False,
                disable_policy_review_in_voice=False,
            )

        self.assertFalse(decision.allowed)
        self.assertTrue(decision.must_return)
        self.assertEqual(len(handler.calls), 0)
        self.assertEqual(len(decision.events), 1)
        self.assertEqual(decision.events[0][0], "confirmation_required")
        self.assertEqual(len(pending), 1)
        token, payload = next(iter(pending.items()))
        self.assertTrue(token)
        self.assertEqual(payload["tool_name"], "write_file")
        self.assertEqual(payload["required_stage"], 1)

    async def test_python_exec_enters_single_step_confirmation_flow(self):
        ctx = _make_ctx(tool_calls=1)
        handler = _make_handle_policy_block()
        policy = _make_tool_policy()
        policy.evaluate.return_value = (False, "confirmation required for risk=medium")
        policy.confirmation_brief.return_value = {
            "tool": "python_exec",
            "risk_level": "medium",
            "stage": 1,
            "required_stage": 1,
            "capability": "filesystem=true",
        }

        mock_tool = MagicMock()
        mock_tool.name = "python_exec"
        mock_tool.risk_level = "medium"
        mock_tool.capability = MagicMock()
        mock_tool.capability.data_classification = "public"

        pending = {}
        with patch("liagent.agent.policy_gate.get_tool", return_value=mock_tool):
            decision = await evaluate_tool_policy(
                tool_name="python_exec",
                tool_args={"code": "print(1)"},
                tool_sig="python_exec:{code:print(1)}",
                full_response="<tool_call>...</tool_call>",
                confirmed=False,
                ctx=ctx,
                tool_policy=policy,
                planner=_make_planner(),
                handle_policy_block_fn=handler,
                pending_confirmations=pending,
                dup_tool_limit=2,
                tool_cache_enabled=True,
                enable_policy_review=False,
                disable_policy_review_in_voice=False,
            )

        self.assertFalse(decision.allowed)
        self.assertTrue(decision.must_return)
        self.assertEqual(len(handler.calls), 0)
        self.assertEqual(len(decision.events), 1)
        self.assertEqual(decision.events[0][0], "confirmation_required")
        self.assertEqual(len(pending), 1)
        _, payload = next(iter(pending.items()))
        self.assertEqual(payload["tool_name"], "python_exec")
        self.assertEqual(payload["required_stage"], 1)

    async def test_skill_whitelist_blocks(self):
        ctx = _make_ctx(tool_calls=1, skill_allowed_tools={"web_search"}, active_skill_name="research")
        handler = _make_handle_policy_block()

        decision = await evaluate_tool_policy(
            tool_name="stock",
            tool_args={"symbol": "AAPL"},
            tool_sig="stock:{symbol:AAPL}",
            full_response="<tool_call>...</tool_call>",
            confirmed=False,
            ctx=ctx,
            tool_policy=_make_tool_policy(),
            planner=_make_planner(),
            handle_policy_block_fn=handler,
            pending_confirmations={},
            dup_tool_limit=2,
            tool_cache_enabled=True,
            enable_policy_review=False,
            disable_policy_review_in_voice=False,
        )

        self.assertFalse(decision.allowed)
        event_types = [ev[0] for ev in decision.events]
        self.assertIn("tool_result", event_types)
        self.assertEqual(len(handler.calls), 1)

    async def test_budget_exceeded_blocks(self):
        ctx = _make_ctx(tool_calls=4, max_tool_calls=3)
        handler = _make_handle_policy_block()

        decision = await evaluate_tool_policy(
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_sig="web_search:{query:test}",
            full_response="<tool_call>...</tool_call>",
            confirmed=False,
            ctx=ctx,
            tool_policy=_make_tool_policy(),
            planner=_make_planner(),
            handle_policy_block_fn=handler,
            pending_confirmations={},
            dup_tool_limit=2,
            tool_cache_enabled=True,
            enable_policy_review=False,
            disable_policy_review_in_voice=False,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("budget", decision.blocked_reason)

    async def test_dup_tool_blocks_on_third_call(self):
        ctx = _make_ctx(tool_calls=1, tool_sig_count={"web_search:{query:test}": 2})
        handler = _make_handle_policy_block()

        decision = await evaluate_tool_policy(
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_sig="web_search:{query:test}",
            full_response="<tool_call>...</tool_call>",
            confirmed=False,
            ctx=ctx,
            tool_policy=_make_tool_policy(),
            planner=_make_planner(),
            handle_policy_block_fn=handler,
            pending_confirmations={},
            dup_tool_limit=2,
            tool_cache_enabled=True,
            enable_policy_review=False,
            disable_policy_review_in_voice=False,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("repeated", decision.blocked_reason)

    async def test_allowed_tool_passes(self):
        ctx = _make_ctx(tool_calls=1)

        mock_tool = MagicMock()
        mock_tool.risk_level = "low"

        with patch("liagent.agent.policy_gate.get_tool", return_value=mock_tool):
            decision = await evaluate_tool_policy(
                tool_name="web_search",
                tool_args={"query": "test"},
                tool_sig="web_search:{query:test}",
                full_response="<tool_call>...</tool_call>",
                confirmed=False,
                ctx=ctx,
                tool_policy=_make_tool_policy(),
                planner=_make_planner(),
                handle_policy_block_fn=_make_handle_policy_block(),
                pending_confirmations={},
                dup_tool_limit=2,
                tool_cache_enabled=True,
                enable_policy_review=False,
                disable_policy_review_in_voice=False,
            )

        self.assertTrue(decision.allowed)
        self.assertFalse(decision.must_return)


# ── ToolOrchestrator tests ────────────────────────────────────────────


class ToolOrchestratorTests(unittest.IsolatedAsyncioTestCase):

    async def test_execute_success_yields_tool_result_and_context(self):
        ctx = _make_ctx(tool_calls=1)
        memory = MagicMock()
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=("search result data", False, ""))
        policy = _make_tool_policy()
        tool_def = MagicMock()

        result = await execute_and_record(
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_sig="web_search:{query:test}",
            tool_def=tool_def,
            full_response="<tool_call>...</tool_call>",
            ctx=ctx,
            executor=executor,
            tool_policy=policy,
            memory=memory,
            tool_cache_enabled=True,
        )

        self.assertFalse(result.is_error)
        self.assertEqual(result.observation, "search result data")
        event_types = [ev[0] for ev in result.events]
        self.assertIn("tool_result", event_types)
        self.assertIn("context_update", event_types)
        # Memory must be written
        self.assertEqual(memory.add.call_count, 2)  # assistant + tool
        # tools_used must be updated
        self.assertIn("web_search", ctx.tools_used)
        # Context var must be stored
        self.assertIn("web_search_result", ctx.context_vars)

    async def test_execute_unknown_tool_returns_error(self):
        ctx = _make_ctx(tool_calls=1)
        memory = MagicMock()
        policy = _make_tool_policy()

        result = await execute_and_record(
            tool_name="nonexistent",
            tool_args={},
            tool_sig="nonexistent:{}",
            tool_def=None,
            full_response="<tool_call>...</tool_call>",
            ctx=ctx,
            executor=MagicMock(),
            tool_policy=policy,
            memory=memory,
            tool_cache_enabled=False,
        )

        self.assertTrue(result.is_error)
        self.assertIn("unknown tool", result.observation)
        policy.audit.assert_called_once()

    async def test_execute_error_increments_tool_errors(self):
        ctx = _make_ctx(tool_calls=1)
        memory = MagicMock()
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=("timeout error", True, "timeout"))
        tool_def = MagicMock()

        result = await execute_and_record(
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_sig="web_search:{query:test}",
            tool_def=tool_def,
            full_response="<tool_call>...</tool_call>",
            ctx=ctx,
            executor=executor,
            tool_policy=_make_tool_policy(),
            memory=memory,
            tool_cache_enabled=False,
        )

        self.assertTrue(result.is_error)
        self.assertEqual(ctx.tool_errors, 1)

    async def test_cache_hit_skips_execution(self):
        ctx = _make_ctx(
            tool_calls=1,
            tool_artifacts={"web_search:{query:test}": "cached result"},
        )
        memory = MagicMock()
        executor = MagicMock()
        executor.execute = AsyncMock()
        tool_def = MagicMock()

        result = await execute_and_record(
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_sig="web_search:{query:test}",
            tool_def=tool_def,
            full_response="<tool_call>...</tool_call>",
            ctx=ctx,
            executor=executor,
            tool_policy=_make_tool_policy(),
            memory=memory,
            tool_cache_enabled=True,
        )

        self.assertEqual(result.observation, "cached result")
        executor.execute.assert_not_called()

    async def test_observation_truncated_at_2000(self):
        ctx = _make_ctx(tool_calls=1)
        memory = MagicMock()
        executor = MagicMock()
        long_output = "x" * 3000
        executor.execute = AsyncMock(return_value=(long_output, False, ""))
        tool_def = MagicMock()

        result = await execute_and_record(
            tool_name="web_search",
            tool_args={"query": "test"},
            tool_sig="web_search:{query:test}",
            tool_def=tool_def,
            full_response="<tool_call>...</tool_call>",
            ctx=ctx,
            executor=executor,
            tool_policy=_make_tool_policy(),
            memory=memory,
            tool_cache_enabled=False,
        )

        self.assertLessEqual(len(result.observation), 2020)
        self.assertIn("truncated", result.observation)


# ── ResponseGuard tests ───────────────────────────────────────────────


class ResponseGuardTests(unittest.IsolatedAsyncioTestCase):

    async def test_accept_clean_response(self):
        ctx = _make_ctx()
        guard = await check_response(
            full_response="Apple Vision Pro is priced at $3,499.",
            step=0,
            ctx=ctx,
            experience=MagicMock(),
            best_effort_answer_fn=AsyncMock(),
            build_done_events_fn=MagicMock(),
        )

        self.assertEqual(guard.action, "accept")
        self.assertIn("3,499", guard.answer)

    async def test_degenerate_output_aborts(self):
        ctx = _make_ctx()
        block = '<tool_call>{"name": "web_fetch", "args": {"url": "https://x.com"}}\n'
        degenerate = block * 5

        guard = await check_response(
            full_response=degenerate,
            step=0,
            ctx=ctx,
            experience=MagicMock(),
            best_effort_answer_fn=AsyncMock(return_value=("fallback answer", {})),
            build_done_events_fn=MagicMock(return_value=[("done", "fallback answer")]),
        )

        self.assertEqual(guard.action, "abort_degenerate")
        self.assertIn("degenerate_output", ctx.quality_issues)

    async def test_reasoning_leak_chatter_aborts(self):
        ctx = _make_ctx()
        chatter = (
            "Let me search for the organizer.\n"
            "I need to call web_search now.\n"
            "Let me search again to confirm.\n"
            "Calling tool web_search."
        )
        guard = await check_response(
            full_response=chatter,
            step=0,
            ctx=ctx,
            experience=MagicMock(),
            best_effort_answer_fn=AsyncMock(return_value=("fallback answer", {})),
            build_done_events_fn=MagicMock(return_value=[("done", "fallback answer")]),
        )

        self.assertEqual(guard.action, "abort_degenerate")
        self.assertIn("degenerate_output", ctx.quality_issues)

    async def test_copout_retry_on_first_occurrence(self):
        ctx = _make_ctx(tool_calls=1)
        guard = await check_response(
            full_response="please check directly",
            step=0,
            ctx=ctx,
            experience=MagicMock(),
            best_effort_answer_fn=AsyncMock(),
            build_done_events_fn=MagicMock(),
        )

        self.assertEqual(guard.action, "retry")
        self.assertEqual(guard.retry_reason, "copout")
        self.assertTrue(ctx.copout_retried)

    async def test_copout_not_retried_twice(self):
        ctx = _make_ctx(tool_calls=1, copout_retried=True)
        guard = await check_response(
            full_response="please check directly",
            step=0,
            ctx=ctx,
            experience=MagicMock(),
            best_effort_answer_fn=AsyncMock(),
            build_done_events_fn=MagicMock(),
        )

        self.assertEqual(guard.action, "accept")

    async def test_hallucination_retry(self):
        ctx = _make_ctx(tool_calls=1, tools_used={"web_search"})
        guard = await check_response(
            full_response="Report saved to cwork/report.txt.",
            step=0,
            ctx=ctx,
            experience=MagicMock(),
            best_effort_answer_fn=AsyncMock(),
            build_done_events_fn=MagicMock(),
        )

        self.assertEqual(guard.action, "retry")
        self.assertEqual(guard.retry_reason, "hallucination")
        self.assertTrue(ctx.hallucination_retried)

    async def test_experience_guard_intercept(self):
        ctx = _make_ctx(tool_calls=0)
        experience_match = MagicMock()
        experience_match.should_use_tool = True
        experience_match.confidence = 0.8
        experience_match.suggested_tool = "web_search"
        experience_match.pattern = "test"
        ctx.experience_match = experience_match
        experience = MagicMock()

        with patch("liagent.tools.get_tool", return_value=MagicMock()):
            guard = await check_response(
                full_response="I am not sure about the latest price.",
                step=0,
                ctx=ctx,
                experience=experience,
                best_effort_answer_fn=AsyncMock(),
                build_done_events_fn=MagicMock(),
            )

        self.assertEqual(guard.action, "experience_guard_tool")
        self.assertEqual(guard.guard_tool_name, "web_search")
        experience.record_outcome.assert_called_once()

    async def test_strip_residual_tool_call_fragments(self):
        ctx = _make_ctx()
        guard = await check_response(
            full_response="normal answer content<tool_call>residual fragment",
            step=0,
            ctx=ctx,
            experience=MagicMock(),
            best_effort_answer_fn=AsyncMock(),
            build_done_events_fn=MagicMock(),
        )

        self.assertEqual(guard.action, "accept")
        self.assertNotIn("<tool_call>", guard.answer)


class RunContextThinkTests(unittest.TestCase):
    def test_think_fields_default(self):
        ctx = RunContext()
        self.assertFalse(ctx.show_thinking)
        self.assertTrue(ctx.enable_thinking)
        self.assertEqual(ctx.reasoning_content, "")

    def test_think_fields_custom(self):
        ctx = RunContext(show_thinking=True, enable_thinking=False)
        self.assertTrue(ctx.show_thinking)
        self.assertFalse(ctx.enable_thinking)


class ResponseGuardThinkTests(unittest.IsolatedAsyncioTestCase):

    async def test_think_intent_triggers_guard_with_experience(self):
        """When think shows tool intent + experience match, should trigger guard."""
        ctx = _make_ctx(tool_calls=0)
        ctx.reasoning_content = "The user asks for real-time stock price, I need to call tools to search."
        experience_match = MagicMock()
        experience_match.should_use_tool = True
        experience_match.confidence = 0.5  # Below 0.6 threshold for normal guard
        experience_match.suggested_tool = "web_search"
        experience_match.pattern = "stock"
        ctx.experience_match = experience_match
        experience = MagicMock()

        with patch("liagent.tools.get_tool", return_value=MagicMock()):
            guard = await check_response(
                full_response="Google stock price is around ...",
                step=0,
                ctx=ctx,
                experience=experience,
                best_effort_answer_fn=AsyncMock(),
                build_done_events_fn=MagicMock(),
            )

        self.assertEqual(guard.action, "experience_guard_tool")

    async def test_think_dismissed_does_not_trigger(self):
        """When think dismisses tool need, should NOT trigger guard."""
        ctx = _make_ctx(tool_calls=0)
        ctx.reasoning_content = "Need to call tools, but known information is enough for a direct answer."
        experience_match = MagicMock()
        experience_match.should_use_tool = True
        experience_match.confidence = 0.5
        experience_match.suggested_tool = "web_search"
        experience_match.pattern = "test"
        ctx.experience_match = experience_match

        guard = await check_response(
            full_response="Python's GIL is the global interpreter lock.",
            step=0,
            ctx=ctx,
            experience=MagicMock(),
            best_effort_answer_fn=AsyncMock(),
            build_done_events_fn=MagicMock(),
        )

        self.assertEqual(guard.action, "accept")

    async def test_think_no_reasoning_content_skips(self):
        """No reasoning_content should skip think check entirely."""
        ctx = _make_ctx(tool_calls=0)
        ctx.reasoning_content = ""

        guard = await check_response(
            full_response="normal answer.",
            step=0,
            ctx=ctx,
            experience=MagicMock(),
            best_effort_answer_fn=AsyncMock(),
            build_done_events_fn=MagicMock(),
        )

        self.assertEqual(guard.action, "accept")


if __name__ == "__main__":
    unittest.main()
