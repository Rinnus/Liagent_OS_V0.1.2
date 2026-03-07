"""Tests for trust-aware ToolPolicy evaluation."""

import tempfile
import unittest
from pathlib import Path

from liagent.tools import ToolDef, ToolCapability
from liagent.tools.policy import ToolPolicy


def _mcp_tool(name: str, risk: str = "low") -> ToolDef:
    return ToolDef(
        name=name, description=f"Test tool {name}",
        parameters={"properties": {}}, func=lambda: None, risk_level=risk,
    )


def _builtin_tool(name: str, risk: str = "low") -> ToolDef:
    return ToolDef(
        name=name, description=f"Test tool {name}",
        parameters={"properties": {}}, func=lambda: None, risk_level=risk,
    )


class TrustPolicyBlockTests(unittest.TestCase):

    def _make_policy(self, tmp_dir: str, profile: str = "full") -> ToolPolicy:
        from liagent.tools.trust_registry import TrustRegistry
        reg = TrustRegistry(store_path=Path(tmp_dir) / "trust.json")
        return ToolPolicy(
            db_path=Path(tmp_dir) / "audit.db",
            tool_profile=profile,
            trust_registry=reg,
        )

    def test_unknown_mcp_returns_confirmation_required(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td)
            tool = _mcp_tool("github__search_repos", "low")
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertTrue(reason.startswith("confirmation required"),
                          f"Expected 'confirmation required...' but got: {reason}")
            self.assertIn("trust", reason)
            self.assertIn("github", reason)

    def test_revoked_mcp_hard_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td)
            policy.trust_registry.set_status("github", "revoked", source="manual")
            tool = _mcp_tool("github__search_repos", "low")
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertFalse(reason.startswith("confirmation required"))
            self.assertIn("revoked", reason.lower())

    def test_approved_mcp_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td)
            policy.trust_registry.set_status("github", "approved", source="curated")
            tool = _mcp_tool("github__search_repos", "low")
            allowed, reason = policy.evaluate(tool, {})
            self.assertTrue(allowed, f"Expected allowed, got: {reason}")

    def test_builtin_tool_bypasses_trust(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td)
            tool = _builtin_tool("web_search", "low")
            allowed, reason = policy.evaluate(tool, {})
            self.assertTrue(allowed, f"Expected allowed, got: {reason}")

    def test_no_trust_registry_skips_check(self):
        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(
                db_path=Path(td) / "audit.db",
                tool_profile="full",
                trust_registry=None,
            )
            tool = _mcp_tool("github__search_repos", "low")
            allowed, reason = policy.evaluate(tool, {})
            self.assertTrue(allowed, f"Expected allowed (no registry), got: {reason}")

    def test_trust_check_runs_before_profile(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td, profile="minimal")
            tool = _mcp_tool("github__search_repos", "low")
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertIn("trust", reason.lower())


class TrustPolicyResearchProfileTests(unittest.TestCase):

    def _make_policy(self, tmp_dir: str) -> ToolPolicy:
        from liagent.tools.trust_registry import TrustRegistry
        reg = TrustRegistry(store_path=Path(tmp_dir) / "trust.json")
        reg.set_status("github", "approved", source="curated")
        return ToolPolicy(
            db_path=Path(tmp_dir) / "audit.db",
            tool_profile="research",
            trust_registry=reg,
        )

    def test_approved_medium_mcp_allowed_in_research(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td)
            tool = _mcp_tool("github__search_repos", "medium")
            allowed, reason = policy.evaluate(tool, {})
            self.assertTrue(allowed, f"Expected allowed, got: {reason}")

    def test_approved_high_mcp_blocked_by_risk(self):
        with tempfile.TemporaryDirectory() as td:
            policy = self._make_policy(td)
            tool = _mcp_tool("github__delete_repo", "high")
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            self.assertNotIn("trust", reason.lower())


if __name__ == "__main__":
    unittest.main()
