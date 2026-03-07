"""Tests for E3 decision layer: idempotency, latency-aware timeout, failure-mode cooldown, telemetry."""
import asyncio
import unittest

from liagent.tools import ToolDef, ToolCapability
from liagent.agent.tool_executor import (
    ToolExecutor,
    _effective_timeout,
    _TIMEOUT_FLOOR,
    _TIMEOUT_CEIL,
    _SLOW_TIMEOUT_FLOOR,
    _LATENCY_TIMEOUT_MULTIPLIER,
)
from liagent.agent.tool_relations import ToolRelation, ToolRelationGraph


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_tool(func, name="test_tool", *, idempotent=True, latency_tier="fast",
               failure_modes=(), max_output_chars=2000, min_timeout_sec=0.0):
    return ToolDef(
        name=name, description="test", parameters={},
        func=func,
        capability=ToolCapability(
            idempotent=idempotent,
            latency_tier=latency_tier,
            failure_modes=failure_modes,
            max_output_chars=max_output_chars,
            min_timeout_sec=min_timeout_sec,
        ),
    )


class _FakePolicy:
    def sanitize_output(self, td, result):
        return str(result)


# ── E3-1: Idempotency-Aware Retry ───────────────────────────────────────

class TestIdempotencyRetry(unittest.TestCase):
    """Non-idempotent tools should not be retried in execute()."""

    def test_idempotent_tool_retries(self):
        """Idempotent tool (default) retries retry_count times."""
        call_count = 0

        async def flaky(**kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "ok"

        td = _make_tool(flaky, idempotent=True)
        executor = ToolExecutor(_FakePolicy(), retry_count=3, timeout_sec=5.0)
        obs, is_err, err_type = asyncio.run(executor.execute(td, {}))
        self.assertFalse(is_err)
        self.assertEqual(call_count, 3)

    def test_non_idempotent_tool_no_retry(self):
        """Non-idempotent tool should only execute once (no retry)."""
        call_count = 0

        async def side_effect(**kw):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        td = _make_tool(side_effect, idempotent=False)
        executor = ToolExecutor(_FakePolicy(), retry_count=3, timeout_sec=5.0)
        obs, is_err, err_type = asyncio.run(executor.execute(td, {}))
        self.assertTrue(is_err)
        self.assertEqual(call_count, 1)  # only 1 attempt, no retries

    def test_non_idempotent_blocks_same_args_self_retry_in_graph(self):
        """Non-idempotent tool blocks same-args self-retry in fallback graph."""
        call_count = 0

        async def non_idem(**kw):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        td = _make_tool(non_idem, name="send_email", idempotent=False)
        graph = ToolRelationGraph()
        graph.add(ToolRelation(
            source="send_email", target="send_email",
            relation="fallback", confidence=0.9,
            condition="on_error_retry",
            allow_same_args=True,  # would normally allow same-args retry
        ))
        executor = ToolExecutor(_FakePolicy(), retry_count=0, timeout_sec=5.0,
                                relation_graph=graph)
        obs, is_err, _, eff_tool, eff_args = asyncio.run(
            executor.execute_with_fallback(td, {"to": "x@y.com"})
        )
        self.assertTrue(is_err)
        # Only 1 call: initial execute. Self-retry blocked by idempotency guard.
        self.assertEqual(call_count, 1)

    def test_non_idempotent_allows_changed_args_self_retry(self):
        """Non-idempotent tool allows self-retry if args changed."""
        call_count = 0

        async def non_idem(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("fail")
            return "sent"

        td = _make_tool(non_idem, name="send_email", idempotent=False)
        graph = ToolRelationGraph()
        graph.add(ToolRelation(
            source="send_email", target="send_email",
            relation="fallback", confidence=0.9,
            condition="on_error_retry",
            allow_same_args=False,
            args_transform=lambda a: {**a, "subject": "retry"},
        ))
        executor = ToolExecutor(_FakePolicy(), retry_count=0, timeout_sec=5.0,
                                relation_graph=graph)
        obs, is_err, _, eff_tool, eff_args = asyncio.run(
            executor.execute_with_fallback(td, {"to": "x@y.com"})
        )
        self.assertFalse(is_err)
        self.assertEqual(call_count, 2)
        self.assertEqual(eff_args.get("subject"), "retry")


# ── E3-2: Latency-Aware Timeout ─────────────────────────────────────────

class TestLatencyTimeout(unittest.TestCase):

    def test_fast_tool_reduced_timeout(self):
        td = _make_tool(lambda: None, latency_tier="fast")
        t = _effective_timeout(td, 20.0)
        self.assertAlmostEqual(t, max(_TIMEOUT_FLOOR, 20.0 * 0.6))

    def test_medium_tool_base_timeout(self):
        td = _make_tool(lambda: None, latency_tier="medium")
        t = _effective_timeout(td, 20.0)
        self.assertAlmostEqual(t, 20.0)

    def test_slow_tool_increased_timeout(self):
        td = _make_tool(lambda: None, latency_tier="slow")
        t = _effective_timeout(td, 20.0)
        expected = max(_SLOW_TIMEOUT_FLOOR, 20.0 * 1.8)
        self.assertAlmostEqual(t, expected)

    def test_slow_tool_floor(self):
        """Slow tool should never be below _SLOW_TIMEOUT_FLOOR."""
        td = _make_tool(lambda: None, latency_tier="slow")
        t = _effective_timeout(td, 5.0)  # 5 * 1.8 = 9, but floor is 45
        self.assertGreaterEqual(t, _SLOW_TIMEOUT_FLOOR)

    def test_timeout_ceil(self):
        """Timeout should never exceed _TIMEOUT_CEIL (without min_timeout_sec)."""
        td = _make_tool(lambda: None, latency_tier="slow")
        t = _effective_timeout(td, 500.0)  # 500 * 1.8 = 900, but ceil is 180
        self.assertLessEqual(t, _TIMEOUT_CEIL)

    def test_min_timeout_sec_overrides_tier_floor(self):
        """Tool-declared min_timeout_sec raises the floor above tier default."""
        td = _make_tool(lambda: None, latency_tier="slow", min_timeout_sec=125.0)
        t = _effective_timeout(td, 20.0)  # 20 * 1.8 = 36, floor raised to 125
        self.assertGreaterEqual(t, 125.0)

    def test_min_timeout_sec_overrides_ceil(self):
        """Tool-declared min_timeout_sec > _TIMEOUT_CEIL raises the ceiling."""
        td = _make_tool(lambda: None, latency_tier="slow", min_timeout_sec=200.0)
        t = _effective_timeout(td, 20.0)  # floor=200, ceil raised to 200
        self.assertGreaterEqual(t, 200.0)

    def test_run_tests_not_truncated(self):
        """run_tests (internal 120s) should get outer timeout > 120s."""
        td = _make_tool(lambda: None, name="run_tests",
                        latency_tier="slow", min_timeout_sec=125.0)
        t = _effective_timeout(td, 20.0)  # default base_timeout
        self.assertGreaterEqual(t, 125.0)

    def test_python_exec_not_truncated(self):
        """python_exec (internal 30s) should get outer timeout > 30s."""
        td = _make_tool(lambda: None, name="python_exec",
                        latency_tier="medium", min_timeout_sec=35.0)
        t = _effective_timeout(td, 20.0)  # 20 * 1.0 = 20, but floor raised to 35
        self.assertGreaterEqual(t, 35.0)

    def test_timeout_floor_fast(self):
        """Fast tool should never be below _TIMEOUT_FLOOR."""
        td = _make_tool(lambda: None, latency_tier="fast")
        t = _effective_timeout(td, 5.0)  # 5 * 0.6 = 3, but floor is 10
        self.assertGreaterEqual(t, _TIMEOUT_FLOOR)

    def test_mcp_tool_default_fast_treated_as_medium(self):
        """MCP tool (__ in name) with default 'fast' latency treated as medium."""
        td = _make_tool(lambda: None, name="mcp__search", latency_tier="fast")
        t = _effective_timeout(td, 20.0)
        self.assertAlmostEqual(t, 20.0)  # medium: multiplier 1.0

    def test_mcp_tool_explicit_slow_unchanged(self):
        """MCP tool with explicitly set 'slow' latency is not overridden."""
        td = _make_tool(lambda: None, name="mcp__run_tests", latency_tier="slow")
        t = _effective_timeout(td, 20.0)
        expected = max(_SLOW_TIMEOUT_FLOOR, 20.0 * 1.8)
        self.assertAlmostEqual(t, expected)

    def test_timeout_error_reports_effective_timeout(self):
        """Timeout error message should report effective_timeout, not base."""
        async def slow_func(**kw):
            await asyncio.sleep(100)

        td = _make_tool(slow_func, latency_tier="fast")
        executor = ToolExecutor(_FakePolicy(), retry_count=0, timeout_sec=20.0)
        effective = _effective_timeout(td, 20.0)
        obs, is_err, err_type = asyncio.run(executor.execute(td, {}))
        self.assertTrue(is_err)
        self.assertEqual(err_type, "timeout")
        self.assertIn(f">{effective:.0f}s", obs)
        # Should NOT contain the base timeout
        self.assertNotIn(">20s", obs)


# ── E3-3: Failure-Mode Cooldown ──────────────────────────────────────────

class TestFailureModeCooldown(unittest.TestCase):

    def test_timeout_with_network_timeout_blocks_same_args_retry(self):
        """Tool with network_timeout failure mode: timeout → skip same-args self-retry."""
        call_count = 0

        async def net_tool(**kw):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(100)  # will timeout

        td = _make_tool(net_tool, name="web_fetch",
                        failure_modes=("network_timeout",),
                        latency_tier="medium")
        graph = ToolRelationGraph()
        graph.add(ToolRelation(
            source="web_fetch", target="web_fetch",
            relation="fallback", confidence=0.8,
            condition="on_error_retry",
            allow_same_args=True,
        ))
        # Very short timeout so the test is fast
        executor = ToolExecutor(_FakePolicy(), retry_count=0, timeout_sec=0.05,
                                relation_graph=graph)
        obs, is_err, err_type, _, _ = asyncio.run(
            executor.execute_with_fallback(td, {"url": "http://example.com"})
        )
        self.assertTrue(is_err)
        self.assertEqual(call_count, 1)  # no self-retry after timeout

    def test_timeout_without_network_timeout_allows_retry(self):
        """Tool without network_timeout: timeout → allow_same_args self-retry proceeds."""
        call_count = 0

        async def flaky(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(100)  # timeout on first call
            return "ok"

        td = _make_tool(flaky, name="api_call",
                        failure_modes=("exception",),
                        latency_tier="medium")
        graph = ToolRelationGraph()
        graph.add(ToolRelation(
            source="api_call", target="api_call",
            relation="fallback", confidence=0.8,
            condition="on_error_retry",
            allow_same_args=True,
        ))
        executor = ToolExecutor(_FakePolicy(), retry_count=0, timeout_sec=0.05,
                                relation_graph=graph)
        obs, is_err, _, _, _ = asyncio.run(
            executor.execute_with_fallback(td, {"endpoint": "/data"})
        )
        self.assertFalse(is_err)
        self.assertEqual(call_count, 2)


# ── E3-4: Telemetry ─────────────────────────────────────────────────────

class TestTelemetryFields(unittest.TestCase):

    def test_run_context_has_telemetry_fields(self):
        from liagent.agent.run_context import RunContext
        ctx = RunContext()
        self.assertEqual(ctx.tool_fallback_count, 0)
        self.assertEqual(ctx.tool_timeout_count, 0)

    def test_last_err_type_tracked_through_fallback(self):
        """err_type returned should reflect the last failure in the chain."""
        async def always_timeout(**kw):
            await asyncio.sleep(100)

        async def always_exception(**kw):
            raise ValueError("boom")

        td_a = _make_tool(always_timeout, name="tool_a", latency_tier="medium")
        td_b = _make_tool(always_exception, name="tool_b")

        graph = ToolRelationGraph()
        graph.add(ToolRelation(
            source="tool_a", target="tool_b",
            relation="fallback", confidence=0.7,
            condition="on_error_fallback",
        ))

        def get_tool(name):
            return {"tool_a": td_a, "tool_b": td_b}.get(name)

        executor = ToolExecutor(_FakePolicy(), retry_count=0, timeout_sec=0.05,
                                relation_graph=graph)
        obs, is_err, err_type, _, _ = asyncio.run(
            executor.execute_with_fallback(td_a, {}, get_tool_fn=get_tool)
        )
        self.assertTrue(is_err)
        # Last error came from tool_b which raises exception → classified as "provider"
        self.assertEqual(err_type, "provider")


class TestSelfSupervisionE3(unittest.TestCase):
    """Test that self_supervision accepts and stores E3 telemetry fields."""

    def test_log_turn_accepts_new_fields(self):
        import tempfile
        from pathlib import Path
        from liagent.agent.self_supervision import InteractionMetrics

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            m = InteractionMetrics(db_path=db)
            # Should not raise
            m.log_turn(
                session_id="test",
                latency_ms=100.0,
                tool_calls=2,
                tool_errors=1,
                policy_blocked=0,
                task_success=True,
                answer_revision_count=0,
                quality_issues="",
                answer_chars=50,
                tool_fallback_count=1,
                tool_timeout_count=1,
            )
            summary = m.weekly_summary()
            self.assertEqual(summary["avg_fallback_count"], 1.0)
            self.assertEqual(summary["avg_timeout_count"], 1.0)

    def test_log_turn_defaults_zero(self):
        """Without E3 fields, defaults should be 0."""
        import tempfile
        from pathlib import Path
        from liagent.agent.self_supervision import InteractionMetrics

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            m = InteractionMetrics(db_path=db)
            m.log_turn(
                session_id="test",
                latency_ms=50.0,
                tool_calls=1,
                tool_errors=0,
                policy_blocked=0,
                task_success=True,
                answer_revision_count=0,
                quality_issues="",
                answer_chars=30,
            )
            summary = m.weekly_summary()
            self.assertEqual(summary["avg_fallback_count"], 0.0)
            self.assertEqual(summary["avg_timeout_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
