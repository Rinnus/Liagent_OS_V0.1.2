"""Tests for first-use confirmation -> auto-approve trust flow."""

import tempfile
import unittest
from pathlib import Path

from liagent.tools import ToolDef
from liagent.tools.policy import ToolPolicy


class FirstUseApprovalTests(unittest.TestCase):

    def _make_policy(self, tmp_dir: str) -> ToolPolicy:
        from liagent.tools.trust_registry import TrustRegistry
        reg = TrustRegistry(store_path=Path(tmp_dir) / "trust.json")
        return ToolPolicy(
            db_path=Path(tmp_dir) / "audit.db",
            tool_profile="full",
            trust_registry=reg,
        )

    def test_unknown_blocks_then_approve_allows(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td)
            tool = ToolDef(
                name="github__search", description="Search",
                parameters={"properties": {}}, func=lambda: None, risk_level="low",
            )
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertTrue(reason.startswith("confirmation required"))
            policy.trust_registry.set_status("github", "approved", source="first_use")
            allowed, reason = policy.evaluate(tool, {})
            self.assertTrue(allowed, f"Expected allowed, got: {reason}")

    def test_revoke_after_approve_blocks_hard(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td)
            policy.trust_registry.set_status("github", "approved", source="first_use")
            tool = ToolDef(
                name="github__search", description="Search",
                parameters={"properties": {}}, func=lambda: None, risk_level="low",
            )
            allowed, _ = policy.evaluate(tool, {})
            self.assertTrue(allowed)
            policy.trust_registry.set_status("github", "revoked", source="manual")
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertIn("revoked", reason.lower())

    def test_confirmed_flag_bypasses_unknown_trust(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td)
            tool = ToolDef(
                name="github__search", description="Search",
                parameters={"properties": {}}, func=lambda: None, risk_level="low",
            )
            allowed, reason = policy.evaluate(tool, {}, confirmed=True)
            self.assertTrue(allowed, f"Expected allowed with confirmed=True, got: {reason}")


if __name__ == "__main__":
    unittest.main()
