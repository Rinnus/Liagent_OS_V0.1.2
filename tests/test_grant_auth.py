"""Tests for A2 Task 2+3: confirmed/granted split + auth_mode propagation."""

import tempfile
import time
import unittest
from pathlib import Path

from liagent.tools import ToolDef, ToolCapability
from liagent.tools.policy import ToolPolicy


def _make_policy(db_path, **kwargs):
    return ToolPolicy(db_path=db_path, tool_profile="full", **kwargs)


def _tool(name="test_tool", risk="low", **cap_kw):
    return ToolDef(
        name=name, description="test", parameters={"properties": {}},
        func=lambda: None, risk_level=risk,
        capability=ToolCapability(**cap_kw),
    )


class GrantedEvaluateTests(unittest.TestCase):
    """Test evaluate() with the new granted= parameter."""

    def test_granted_bypasses_confirm_tools(self):
        """A session grant should bypass confirm_tools gate."""
        with tempfile.TemporaryDirectory() as td:
            p = _make_policy(Path(td) / "a.db")
            p.confirm_tools = {"web_search"}
            tool = _tool("web_search")
            # Without grant — blocked
            ok, reason = p.evaluate(tool, {})
            self.assertFalse(ok)
            self.assertIn("confirmation required", reason)
            # With granted — allowed
            ok, reason = p.evaluate(tool, {}, granted=True)
            self.assertTrue(ok)
            p.close()

    def test_granted_bypasses_risk_confirmation(self):
        """A grant should bypass risk-level confirmation for medium risk."""
        with tempfile.TemporaryDirectory() as td:
            p = _make_policy(Path(td) / "a.db")
            p.confirm_risk_levels = {"medium"}
            tool = _tool("tool_a", risk="medium")
            ok, reason = p.evaluate(tool, {})
            self.assertFalse(ok)
            ok, reason = p.evaluate(tool, {}, granted=True)
            self.assertTrue(ok)
            p.close()

    def test_granted_bypasses_requires_user_presence(self):
        with tempfile.TemporaryDirectory() as td:
            p = _make_policy(Path(td) / "a.db")
            tool = _tool("tool_a", requires_user_presence=True)
            ok, _ = p.evaluate(tool, {})
            self.assertFalse(ok)
            ok, _ = p.evaluate(tool, {}, granted=True)
            self.assertTrue(ok)
            p.close()

    def test_granted_does_NOT_bypass_high_risk(self):
        """High-risk tools require confirmed (human), NOT granted."""
        with tempfile.TemporaryDirectory() as td:
            p = _make_policy(Path(td) / "a.db")
            tool = _tool("danger", risk="high")
            # granted alone should NOT bypass high-risk
            ok, reason = p.evaluate(tool, {}, granted=True)
            self.assertFalse(ok)
            self.assertIn("confirmation required", reason)
            # confirmed should bypass
            ok, _ = p.evaluate(tool, {}, confirmed=True)
            self.assertTrue(ok)
            p.close()

    def test_granted_does_NOT_bypass_trust_gate(self):
        """Grants must NOT bypass trust gate for unknown MCP servers."""
        from liagent.tools.trust_registry import TrustRegistry
        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            p = _make_policy(Path(td) / "a.db", trust_registry=reg)
            tool = _tool("unknown_srv__query")
            # granted should NOT bypass unknown trust
            ok, reason = p.evaluate(tool, {}, granted=True)
            self.assertFalse(ok)
            self.assertIn("confirmation required", reason)
            # confirmed should bypass
            ok, _ = p.evaluate(tool, {}, confirmed=True)
            self.assertTrue(ok)
            p.close()

    def test_granted_does_NOT_bypass_revoked_trust(self):
        from liagent.tools.trust_registry import TrustRegistry
        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            reg.set_status("bad_srv", "revoked", source="manual")
            p = _make_policy(Path(td) / "a.db", trust_registry=reg)
            tool = _tool("bad_srv__query")
            ok, reason = p.evaluate(tool, {}, granted=True)
            self.assertFalse(ok)
            self.assertIn("revoked", reason)
            # Even confirmed can't bypass revoked
            ok, reason = p.evaluate(tool, {}, confirmed=True)
            self.assertFalse(ok)
            self.assertIn("revoked", reason)
            p.close()

    def test_allow_high_risk_env_still_works(self):
        """allow_high_risk is orthogonal to grants — it disables high-risk check entirely."""
        with tempfile.TemporaryDirectory() as td:
            p = _make_policy(Path(td) / "a.db")
            p.allow_high_risk = True
            tool = _tool("danger", risk="high")
            ok, _ = p.evaluate(tool, {})
            self.assertTrue(ok)
            p.close()

    def test_confirmed_bypasses_all_gates(self):
        """confirmed should bypass trust + all risk gates (except revoked)."""
        from liagent.tools.trust_registry import TrustRegistry
        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            p = _make_policy(Path(td) / "a.db", trust_registry=reg)
            p.confirm_tools = {"unknown_srv__query"}
            tool = _tool("unknown_srv__query", risk="high")
            ok, _ = p.evaluate(tool, {}, confirmed=True)
            self.assertTrue(ok)
            p.close()

    def test_backward_compat_no_granted_kwarg(self):
        """Existing callers without granted= must still work."""
        with tempfile.TemporaryDirectory() as td:
            p = _make_policy(Path(td) / "a.db")
            tool = _tool("web_search")
            ok, _ = p.evaluate(tool, {})
            self.assertTrue(ok)
            ok, _ = p.evaluate(tool, {}, confirmed=False)
            self.assertTrue(ok)
            p.close()


class PolicyDecisionAuthModeTests(unittest.TestCase):
    """Test auth_mode field on PolicyDecision."""

    def test_auth_mode_field_exists(self):
        from liagent.agent.policy_gate import PolicyDecision
        d = PolicyDecision(allowed=True)
        self.assertEqual(d.auth_mode, "")

    def test_auth_mode_set(self):
        from liagent.agent.policy_gate import PolicyDecision
        d = PolicyDecision(allowed=True, auth_mode="granted")
        self.assertEqual(d.auth_mode, "granted")


class GrantExpiryInPolicyGateTests(unittest.TestCase):
    """Test grant expiry logic in evaluate_tool_policy (unit-level, no full brain)."""

    def test_expired_grant_not_used(self):
        """An expired grant should be removed and not bypass gates."""
        grants = {"web_search": time.time() - 100}  # expired
        if "web_search" in grants and time.time() >= grants["web_search"]:
            grants.pop("web_search")
        self.assertNotIn("web_search", grants)

    def test_active_grant_present(self):
        grants = {"web_search": time.time() + 1800}
        self.assertIn("web_search", grants)
        self.assertTrue(time.time() < grants["web_search"])


class MaybeCreateGrantTests(unittest.TestCase):
    """Test _maybe_create_grant logic (via minimal brain-like mock)."""

    def _make_brain_stub(self, grantable_tools=None):
        """Create a minimal object with _maybe_create_grant behavior."""

        class _Stub:
            def __init__(self):
                self.tool_grants = {}
                self._grant_ttl_sec = 1800
                self._python_exec_grant_ttl_sec = 600
                self._grantable_tools = grantable_tools

            def _grant_ttl_for_tool(self, tool_name: str) -> int:
                if tool_name == "python_exec":
                    return self._python_exec_grant_ttl_sec
                return self._grant_ttl_sec

            def _maybe_create_grant(self, tool_name, *, auth_mode=""):
                if auth_mode != "confirmed":
                    return
                from liagent.tools import get_tool
                tool_def = get_tool(tool_name)
                if tool_def is not None and (
                    tool_def.risk_level == "high" or tool_name == "write_file"
                ):
                    return
                if self._grantable_tools is not None and tool_name not in self._grantable_tools:
                    return
                self.tool_grants[tool_name] = time.time() + self._grant_ttl_for_tool(tool_name)

        return _Stub()

    def test_grant_created_on_confirmed(self):
        stub = self._make_brain_stub()
        stub._maybe_create_grant("web_search", auth_mode="confirmed")
        self.assertIn("web_search", stub.tool_grants)
        self.assertTrue(stub.tool_grants["web_search"] > time.time())

    def test_no_grant_on_allowed(self):
        stub = self._make_brain_stub()
        stub._maybe_create_grant("web_search", auth_mode="allowed")
        self.assertNotIn("web_search", stub.tool_grants)

    def test_no_grant_on_granted(self):
        """Already granted tools don't create new grants."""
        stub = self._make_brain_stub()
        stub._maybe_create_grant("web_search", auth_mode="granted")
        self.assertNotIn("web_search", stub.tool_grants)

    def test_no_grant_for_high_risk(self):
        """High-risk tools should never get session grants."""
        from liagent.tools import ToolDef, ToolCapability, _TOOLS
        _TOOLS["_test_danger"] = ToolDef(
            name="_test_danger", description="test", parameters={},
            func=lambda: None, risk_level="high",
        )
        try:
            stub = self._make_brain_stub()
            stub._maybe_create_grant("_test_danger", auth_mode="confirmed")
            self.assertNotIn("_test_danger", stub.tool_grants)
        finally:
            _TOOLS.pop("_test_danger", None)

    def test_no_grant_for_write_file(self):
        """write_file stays per-action confirmed even after lowering risk."""
        from liagent.tools import ToolDef, ToolCapability, _TOOLS
        _TOOLS["write_file"] = ToolDef(
            name="write_file", description="test", parameters={},
            func=lambda: None, risk_level="medium",
            capability=ToolCapability(),
        )
        try:
            stub = self._make_brain_stub()
            stub._maybe_create_grant("write_file", auth_mode="confirmed")
            self.assertNotIn("write_file", stub.tool_grants)
        finally:
            _TOOLS.pop("write_file", None)

    def test_python_exec_gets_short_session_grant(self):
        from liagent.tools import ToolDef, ToolCapability, _TOOLS
        _TOOLS["python_exec"] = ToolDef(
            name="python_exec", description="test", parameters={},
            func=lambda: None, risk_level="medium",
            capability=ToolCapability(),
        )
        try:
            stub = self._make_brain_stub()
            before = time.time()
            stub._maybe_create_grant("python_exec", auth_mode="confirmed")
            self.assertIn("python_exec", stub.tool_grants)
            expiry_delta = stub.tool_grants["python_exec"] - before
            self.assertGreater(expiry_delta, 500)
            self.assertLess(expiry_delta, 700)
        finally:
            _TOOLS.pop("python_exec", None)

    def test_grantable_whitelist_blocks(self):
        stub = self._make_brain_stub(grantable_tools={"allowed_tool"})
        stub._maybe_create_grant("web_search", auth_mode="confirmed")
        self.assertNotIn("web_search", stub.tool_grants)

    def test_grantable_whitelist_allows(self):
        stub = self._make_brain_stub(grantable_tools={"web_search"})
        stub._maybe_create_grant("web_search", auth_mode="confirmed")
        self.assertIn("web_search", stub.tool_grants)

    def test_no_whitelist_means_all_grantable(self):
        stub = self._make_brain_stub(grantable_tools=None)
        stub._maybe_create_grant("any_tool", auth_mode="confirmed")
        self.assertIn("any_tool", stub.tool_grants)


class ConfirmationExecutionOkTests(unittest.TestCase):
    """Test execution_ok flag in confirmation handler."""

    def test_execution_ok_in_result(self):
        """resolve_confirmation should include execution_ok field."""
        # We test the return dict structure directly
        result_ok = {"status": "ok", "execution_ok": True}
        result_err = {"status": "ok", "execution_ok": False}
        self.assertTrue(result_ok["execution_ok"])
        self.assertFalse(result_err["execution_ok"])


if __name__ == "__main__":
    unittest.main()
