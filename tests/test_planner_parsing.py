import unittest

from liagent.agent.planner import _parse_policy_review


class PlannerParsingTests(unittest.TestCase):
    def test_parse_policy_review_parse_failure_no_extra_confirmation(self):
        out = _parse_policy_review("invalid response")
        self.assertTrue(out.allow)
        self.assertFalse(out.needs_confirmation)
        self.assertIn("parse failed", out.reason)

    def test_parse_policy_review_empty_response_soft_allows_without_warning_reason(self):
        out = _parse_policy_review("")
        self.assertTrue(out.allow)
        self.assertFalse(out.needs_confirmation)
        self.assertIn("unavailable", out.reason)

    def test_parse_policy_review_high_risk_heuristic(self):
        out = _parse_policy_review("allow true, but high risk operation")
        self.assertTrue(out.allow)
        self.assertEqual(out.risk, "high")
        self.assertTrue(out.needs_confirmation)


    def test_parse_policy_review_with_think_block(self):
        raw = '<think>Let me evaluate...</think>\n{"allow": true, "risk": "low", "reason": "read-only"}'
        out = _parse_policy_review(raw)
        self.assertTrue(out.allow)
        self.assertEqual(out.risk, "low")
        self.assertNotIn("parse failed", out.reason)

    def test_parse_policy_review_unclosed_think_still_parses(self):
        raw = '<think>reasoning\n{"allow": false, "risk": "high", "reason": "destructive"}'
        out = _parse_policy_review(raw)
        # Even with unclosed think, brace extraction should find the JSON
        self.assertFalse(out.allow)


if __name__ == "__main__":
    unittest.main()
