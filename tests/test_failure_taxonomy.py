"""Tests for unified failure classification."""
import unittest
from liagent.agent.failure_taxonomy import (
    FailureKind, classify_error, RECOVERY_STRATEGY, FALLBACK_ELIGIBLE,
)


class ClassifyErrorTests(unittest.TestCase):
    def test_timeout(self):
        self.assertEqual(classify_error("timeout", "timed out"), FailureKind.TIMEOUT)

    def test_rate_limit_429(self):
        self.assertEqual(classify_error("exception", "429 Too Many Requests"), FailureKind.RATE_LIMIT)

    def test_auth_401(self):
        self.assertEqual(classify_error("exception", "401 Unauthorized"), FailureKind.AUTH)

    def test_provider_503(self):
        self.assertEqual(classify_error("exception", "503 Service Unavailable"), FailureKind.PROVIDER)

    def test_bad_args(self):
        self.assertEqual(classify_error("exception", "missing required argument: query"), FailureKind.BAD_ARGS)

    def test_unknown_defaults_provider(self):
        self.assertEqual(classify_error("exception", "something weird"), FailureKind.PROVIDER)

    def test_backward_compat_string_equality(self):
        self.assertEqual(FailureKind.TIMEOUT, "timeout")
        self.assertTrue(FailureKind.TIMEOUT == "timeout")

    def test_policy_subtypes(self):
        self.assertIsInstance(FailureKind.POLICY_BUDGET, str)
        self.assertIsInstance(FailureKind.POLICY_ALLOWLIST, str)
        self.assertIsInstance(FailureKind.POLICY_DEDUP, str)


class RecoveryStrategyTests(unittest.TestCase):
    def test_fallback_eligible(self):
        self.assertIn(FailureKind.TIMEOUT, FALLBACK_ELIGIBLE)
        self.assertIn(FailureKind.PROVIDER, FALLBACK_ELIGIBLE)
        self.assertIn(FailureKind.RATE_LIMIT, FALLBACK_ELIGIBLE)
        self.assertNotIn(FailureKind.AUTH, FALLBACK_ELIGIBLE)
        self.assertNotIn(FailureKind.BAD_ARGS, FALLBACK_ELIGIBLE)

    def test_all_kinds_have_strategy(self):
        for kind in FailureKind:
            self.assertIn(kind, RECOVERY_STRATEGY, f"Missing strategy for {kind}")

    def test_policy_budget_forces_answer(self):
        self.assertEqual(RECOVERY_STRATEGY[FailureKind.POLICY_BUDGET], "force_answer")

    def test_policy_allowlist_suggests_alt(self):
        self.assertEqual(RECOVERY_STRATEGY[FailureKind.POLICY_ALLOWLIST], "suggest_alternative")
