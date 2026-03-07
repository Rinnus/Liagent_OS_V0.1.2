"""Tests for global retry budget mechanism."""
import unittest
from liagent.agent.run_context import RunContext


class GlobalRetryBudgetTests(unittest.TestCase):
    def test_defaults(self):
        ctx = RunContext()
        self.assertEqual(ctx.global_retry_budget, 4)
        self.assertEqual(ctx.global_retries_used, 0)
        self.assertFalse(ctx.retry_budget_exhausted)

    def test_consume_retry_success(self):
        ctx = RunContext(global_retry_budget=3)
        self.assertTrue(ctx.consume_retry("copout"))
        self.assertEqual(ctx.global_retries_used, 1)
        self.assertEqual(ctx.retry_ledger, ["copout"])

    def test_consume_retry_exhausted(self):
        ctx = RunContext(global_retry_budget=1, global_retries_used=1)
        self.assertFalse(ctx.consume_retry("hallucination"))
        self.assertEqual(ctx.global_retries_used, 1)

    def test_budget_dynamic_init(self):
        ctx = RunContext(max_steps=15)
        ctx.global_retry_budget = min(4, max(2, ctx.max_steps // 3))
        self.assertEqual(ctx.global_retry_budget, 4)

        ctx2 = RunContext(max_steps=6)
        ctx2.global_retry_budget = min(4, max(2, ctx2.max_steps // 3))
        self.assertEqual(ctx2.global_retry_budget, 2)

    def test_exhausted_property(self):
        ctx = RunContext(global_retry_budget=2, global_retries_used=2)
        self.assertTrue(ctx.retry_budget_exhausted)

    def test_ledger_tracks_reasons(self):
        ctx = RunContext(global_retry_budget=3)
        ctx.consume_retry("copout")
        ctx.consume_retry("hallucination")
        self.assertEqual(ctx.retry_ledger, ["copout", "hallucination"])
