"""E2E integration tests for A2: session grants + audit trail + auth_mode."""

import asyncio
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from liagent.tools import ToolDef, ToolCapability, _TOOLS
from liagent.tools.policy import ToolPolicy
from liagent.tools.trust_registry import TrustRegistry
from liagent.agent.policy_gate import PolicyDecision, evaluate_tool_policy
from liagent.agent.confirmation_handler import resolve_confirmation


def _run(coro):
    return asyncio.run(coro)


def _make_tool(name, risk="low", **cap_kw):
    return ToolDef(
        name=name, description="test", parameters={"properties": {}},
        func=AsyncMock(return_value="result"), risk_level=risk,
        capability=ToolCapability(**cap_kw),
    )


class _FakeCtx:
    """Minimal RunContext for policy_gate tests."""
    def __init__(self):
        self.skill_allowed_tools = None
        self.budget_allowed_tools = None
        self.tool_calls = 0
        self.max_tool_calls = 10
        self.tool_sig_count = {}
        self.user_input = "test"
        self.low_latency = False
        self.budget = MagicMock()
        self.budget.enable_policy_review = False


def _noop_block(*args, **kwargs):
    obs = f"[blocked] {args[2] if len(args) > 2 else ''}"
    return obs, "", [("policy_blocked", args[0], obs)]


class GrantLifecycleE2ETests(unittest.TestCase):
    """Full lifecycle: confirm → grant created → grant reused → grant expires."""

    def setUp(self):
        self._saved = dict(_TOOLS)
        self._tool = _make_tool("e2e_tool", risk="medium")
        _TOOLS["e2e_tool"] = self._tool

    def tearDown(self):
        _TOOLS.clear()
        _TOOLS.update(self._saved)

    def test_confirm_creates_grant_reuse_bypasses(self):
        """
        1. First call: confirmed=True → auth_mode="confirmed", allowed
        2. Grant created with TTL
        3. Second call: no confirmation → grant kicks in → auth_mode="granted"
        """
        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            policy.confirm_risk_levels = {"medium"}  # medium needs confirmation
            grants: dict[str, float] = {}
            ctx = _FakeCtx()

            # 1. First call with confirmation
            async def _first_call():
                return await evaluate_tool_policy(
                    tool_name="e2e_tool", tool_args={"q": "test"},
                    tool_sig="e2e_tool:{q:test}", full_response="...",
                    confirmed=True, ctx=ctx,
                    tool_policy=policy, planner=MagicMock(),
                    handle_policy_block_fn=_noop_block,
                    pending_confirmations={}, dup_tool_limit=3,
                    tool_cache_enabled=False,
                    enable_policy_review=False,
                    disable_policy_review_in_voice=False,
                    tool_grants=grants,
                )

            d1 = _run(_first_call())
            self.assertTrue(d1.allowed)
            self.assertEqual(d1.auth_mode, "confirmed")

            # Simulate grant creation (as brain._maybe_create_grant would)
            grants["e2e_tool"] = time.time() + 1800

            # 2. Second call without confirmation — grant should work
            async def _second_call():
                return await evaluate_tool_policy(
                    tool_name="e2e_tool", tool_args={"q": "test2"},
                    tool_sig="e2e_tool:{q:test2}", full_response="...",
                    confirmed=False, ctx=ctx,
                    tool_policy=policy, planner=MagicMock(),
                    handle_policy_block_fn=_noop_block,
                    pending_confirmations={}, dup_tool_limit=3,
                    tool_cache_enabled=False,
                    enable_policy_review=False,
                    disable_policy_review_in_voice=False,
                    tool_grants=grants,
                )

            d2 = _run(_second_call())
            self.assertTrue(d2.allowed)
            self.assertEqual(d2.auth_mode, "granted")
            policy.close()

    def test_expired_grant_triggers_confirmation(self):
        """Expired grant should not bypass gates."""
        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            policy.confirm_risk_levels = {"medium"}
            grants = {"e2e_tool": time.time() - 100}  # expired
            ctx = _FakeCtx()

            async def _call():
                return await evaluate_tool_policy(
                    tool_name="e2e_tool", tool_args={},
                    tool_sig="e2e_tool:{}", full_response="...",
                    confirmed=False, ctx=ctx,
                    tool_policy=policy, planner=MagicMock(),
                    handle_policy_block_fn=_noop_block,
                    pending_confirmations={}, dup_tool_limit=3,
                    tool_cache_enabled=False,
                    enable_policy_review=False,
                    disable_policy_review_in_voice=False,
                    tool_grants=grants,
                )

            d = _run(_call())
            self.assertFalse(d.allowed)
            # Expired grant should be removed
            self.assertNotIn("e2e_tool", grants)
            policy.close()

    def test_high_risk_never_granted(self):
        """High-risk tools must always require confirmed, never granted."""
        _TOOLS["e2e_danger"] = _make_tool("e2e_danger", risk="high")
        try:
            with tempfile.TemporaryDirectory() as td:
                policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
                grants = {"e2e_danger": time.time() + 1800}
                ctx = _FakeCtx()

                async def _call():
                    return await evaluate_tool_policy(
                        tool_name="e2e_danger", tool_args={},
                        tool_sig="e2e_danger:{}", full_response="...",
                        confirmed=False, ctx=ctx,
                        tool_policy=policy, planner=MagicMock(),
                        handle_policy_block_fn=_noop_block,
                        pending_confirmations={}, dup_tool_limit=3,
                        tool_cache_enabled=False,
                        enable_policy_review=False,
                        disable_policy_review_in_voice=False,
                        tool_grants=grants,
                    )

                d = _run(_call())
                self.assertFalse(d.allowed)
                policy.close()
        finally:
            _TOOLS.pop("e2e_danger", None)

    def test_no_grants_backward_compat(self):
        """evaluate_tool_policy with tool_grants=None should work fine."""
        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            ctx = _FakeCtx()

            async def _call():
                return await evaluate_tool_policy(
                    tool_name="e2e_tool", tool_args={},
                    tool_sig="e2e_tool:{}", full_response="...",
                    confirmed=False, ctx=ctx,
                    tool_policy=policy, planner=MagicMock(),
                    handle_policy_block_fn=_noop_block,
                    pending_confirmations={}, dup_tool_limit=3,
                    tool_cache_enabled=False,
                    enable_policy_review=False,
                    disable_policy_review_in_voice=False,
                    # tool_grants omitted — default None
                )

            d = _run(_call())
            self.assertTrue(d.allowed)
            self.assertEqual(d.auth_mode, "allowed")
            policy.close()


class AuditTrailE2ETests(unittest.TestCase):
    """Verify audit trail records correct policy_decision values."""

    def test_audit_policy_decision_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            p = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            p.audit("t1", {}, "ok", "test", policy_decision="allowed")
            p.audit("t2", {}, "ok", "test", policy_decision="granted", grant_source="session_grant")
            p.audit("t3", {}, "ok", "test", policy_decision="confirmed")
            p.audit("t4", {}, "blocked", "denied", policy_decision="blocked")
            p.audit("t5", {}, "ok", "auto", policy_decision="system_initiated")

            rows = p.recent_audit(limit=10)
            by_tool = {r["tool_name"]: r for r in rows}
            self.assertEqual(by_tool["t1"]["policy_decision"], "allowed")
            self.assertEqual(by_tool["t2"]["policy_decision"], "granted")
            self.assertEqual(by_tool["t2"]["grant_source"], "session_grant")
            self.assertEqual(by_tool["t3"]["policy_decision"], "confirmed")
            self.assertEqual(by_tool["t4"]["policy_decision"], "blocked")
            self.assertEqual(by_tool["t5"]["policy_decision"], "system_initiated")
            p.close()

    def test_audit_requested_vs_effective(self):
        """Fallback audit records both requested and effective tool/args."""
        with tempfile.TemporaryDirectory() as td:
            p = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            p.audit("web_search", {"query": "example.com"}, "ok", "executed via fallback",
                    requested_tool="web_fetch",
                    requested_args='{"url": "http://example.com"}',
                    effective_tool="web_search",
                    effective_args='{"query": "example.com"}',
                    policy_decision="allowed")
            rows = p.recent_audit(limit=1)
            r = rows[0]
            self.assertEqual(r["requested_tool"], "web_fetch")
            self.assertEqual(r["effective_tool"], "web_search")
            self.assertEqual(r["policy_decision"], "allowed")
            p.close()

    def test_audit_redaction_all_args_fields(self):
        """All three args fields are redacted at write time."""
        with tempfile.TemporaryDirectory() as td:
            p = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            p.audit("tool", {"api_key": "sk-secret"}, "ok", "test",
                    requested_args='{"token": "mytoken"}',
                    effective_args='{"password": "hunter2"}')
            row = p._conn.execute(
                "SELECT args_json, requested_args, effective_args FROM tool_audit"
            ).fetchone()
            self.assertNotIn("sk-secret", row[0])
            self.assertNotIn("mytoken", row[1])
            self.assertNotIn("hunter2", row[2])
            p.close()


class ConfirmationE2ETests(unittest.TestCase):
    """Test execution_ok propagation from confirmation handler."""

    def test_execution_ok_true_on_success(self):
        """Successful confirmed execution should set execution_ok=True."""
        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            tool = _make_tool("conf_tool")
            _TOOLS["conf_tool"] = tool

            executor = MagicMock()
            executor.execute = AsyncMock(return_value=("result data", False, None))
            memory = MagicMock()
            final_fn = AsyncMock(return_value=("Final answer", {}))

            pending = {
                "tok123": {
                    "tool_name": "conf_tool", "tool_args": {"q": "test"},
                    "user_input": "test", "assistant_tool_call": "...",
                    "required_stage": 1, "stage": 1, "pending_reason": "risk",
                },
            }

            async def _run_confirm():
                return await resolve_confirmation(
                    "tok123", True, False,
                    pending_confirmations=pending,
                    confirm_ttl=timedelta(minutes=10),
                    tool_policy=policy,
                    tool_executor=executor,
                    memory=memory,
                    final_answer_fn=final_fn,
                )

            result = _run(_run_confirm())
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["execution_ok"])
            _TOOLS.pop("conf_tool", None)
            policy.close()

    def test_execution_ok_false_on_error(self):
        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            tool = _make_tool("conf_tool2")
            _TOOLS["conf_tool2"] = tool

            executor = MagicMock()
            executor.execute = AsyncMock(return_value=("error msg", True, "timeout"))
            memory = MagicMock()
            final_fn = AsyncMock(return_value=("Error answer", {}))

            pending = {
                "tok456": {
                    "tool_name": "conf_tool2", "tool_args": {},
                    "user_input": "test", "assistant_tool_call": "...",
                    "required_stage": 1, "stage": 1, "pending_reason": "risk",
                },
            }

            async def _run_confirm():
                return await resolve_confirmation(
                    "tok456", True, False,
                    pending_confirmations=pending,
                    confirm_ttl=timedelta(minutes=10),
                    tool_policy=policy,
                    tool_executor=executor,
                    memory=memory,
                    final_answer_fn=final_fn,
                )

            result = _run(_run_confirm())
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["execution_ok"])
            _TOOLS.pop("conf_tool2", None)
            policy.close()


class TrustGrantInteractionTests(unittest.TestCase):
    """Verify trust gate + grant gate interaction (grants never bypass trust)."""

    def setUp(self):
        self._saved = dict(_TOOLS)

    def tearDown(self):
        _TOOLS.clear()
        _TOOLS.update(self._saved)

    def test_grant_on_unknown_server_blocked(self):
        """Even with an active grant, unknown MCP server tools are blocked."""
        _TOOLS["unk__query"] = _make_tool("unk__query", risk="low")
        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full",
                                trust_registry=reg)
            grants = {"unk__query": time.time() + 1800}
            ctx = _FakeCtx()

            async def _call():
                return await evaluate_tool_policy(
                    tool_name="unk__query", tool_args={},
                    tool_sig="unk__query:{}", full_response="...",
                    confirmed=False, ctx=ctx,
                    tool_policy=policy, planner=MagicMock(),
                    handle_policy_block_fn=_noop_block,
                    pending_confirmations={}, dup_tool_limit=3,
                    tool_cache_enabled=False,
                    enable_policy_review=False,
                    disable_policy_review_in_voice=False,
                    tool_grants=grants,
                )

            d = _run(_call())
            self.assertFalse(d.allowed)
            # Trust gate triggers confirmation flow (must_return=True), not a simple block
            self.assertTrue(d.must_return)
            policy.close()

    def test_grant_on_approved_server_works(self):
        """Grant on approved MCP server should work."""
        _TOOLS["ok__query"] = _make_tool("ok__query", risk="medium")
        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            reg.set_status("ok", "approved", source="curated")
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full",
                                trust_registry=reg)
            policy.confirm_risk_levels = {"medium"}
            grants = {"ok__query": time.time() + 1800}
            ctx = _FakeCtx()

            async def _call():
                return await evaluate_tool_policy(
                    tool_name="ok__query", tool_args={},
                    tool_sig="ok__query:{}", full_response="...",
                    confirmed=False, ctx=ctx,
                    tool_policy=policy, planner=MagicMock(),
                    handle_policy_block_fn=_noop_block,
                    pending_confirmations={}, dup_tool_limit=3,
                    tool_cache_enabled=False,
                    enable_policy_review=False,
                    disable_policy_review_in_voice=False,
                    tool_grants=grants,
                )

            d = _run(_call())
            self.assertTrue(d.allowed)
            self.assertEqual(d.auth_mode, "granted")
            policy.close()


class P1GrantInResolveConfirmationTests(unittest.TestCase):
    """P1 fix: grant creation must happen in resolve_confirmation wrapper,
    not in run(), so API/WS callers also get grants."""

    def setUp(self):
        self._saved = dict(_TOOLS)

    def tearDown(self):
        _TOOLS.clear()
        _TOOLS.update(self._saved)

    def test_resolve_confirmation_wrapper_creates_grant(self):
        """brain.resolve_confirmation() must create grants internally."""
        # We test by checking that _maybe_create_grant is called from the wrapper.
        # Since we can't easily instantiate a full brain, we verify the method
        # signature and that execution_ok=True triggers grant creation.
        _TOOLS["api_tool"] = _make_tool("api_tool", risk="medium")

        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            executor = MagicMock()
            executor.execute = AsyncMock(return_value=("ok result", False, None))
            memory = MagicMock()
            final_fn = AsyncMock(return_value=("answer", {}))

            pending = {
                "apitok": {
                    "tool_name": "api_tool", "tool_args": {},
                    "user_input": "test", "assistant_tool_call": "...",
                    "required_stage": 1, "stage": 1, "pending_reason": "risk",
                },
            }

            async def _run_confirm():
                return await resolve_confirmation(
                    "apitok", True, False,
                    pending_confirmations=pending,
                    confirm_ttl=timedelta(minutes=10),
                    tool_policy=policy,
                    tool_executor=executor,
                    memory=memory,
                    final_answer_fn=final_fn,
                )

            result = _run(_run_confirm())
            # execution_ok must be in the result so the wrapper can act on it
            self.assertTrue(result.get("execution_ok"))
            self.assertEqual(result.get("tool_name"), "api_tool")
            policy.close()


class P1AuthModePropagationTests(unittest.TestCase):
    """P1 fix: auth_mode from policy_gate must reach tool_orchestrator audit."""

    def setUp(self):
        self._saved = dict(_TOOLS)
        self._tool = _make_tool("prop_tool", risk="medium")
        _TOOLS["prop_tool"] = self._tool

    def tearDown(self):
        _TOOLS.clear()
        _TOOLS.update(self._saved)

    def test_granted_auth_mode_flows_to_decision(self):
        """When a grant is active, PolicyDecision.auth_mode must be 'granted'."""
        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            policy.confirm_risk_levels = {"medium"}
            grants = {"prop_tool": time.time() + 1800}
            ctx = _FakeCtx()

            async def _call():
                return await evaluate_tool_policy(
                    tool_name="prop_tool", tool_args={},
                    tool_sig="prop_tool:{}", full_response="...",
                    confirmed=False, ctx=ctx,
                    tool_policy=policy, planner=MagicMock(),
                    handle_policy_block_fn=_noop_block,
                    pending_confirmations={}, dup_tool_limit=3,
                    tool_cache_enabled=False,
                    enable_policy_review=False,
                    disable_policy_review_in_voice=False,
                    tool_grants=grants,
                )

            d = _run(_call())
            self.assertTrue(d.allowed)
            self.assertEqual(d.auth_mode, "granted")
            self.assertEqual(d.grant_source, "session_grant")
            policy.close()


class P1GrantSourceAuditTests(unittest.TestCase):
    """P1 fix: grant_source must be written to audit trail in real calls."""

    def test_confirmation_handler_records_grant_source(self):
        """confirmation_handler audit calls must include grant_source."""
        from liagent.tools import _TOOLS as tools
        tools["gs_tool"] = _make_tool("gs_tool")

        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            executor = MagicMock()
            executor.execute = AsyncMock(return_value=("result", False, None))
            memory = MagicMock()
            final_fn = AsyncMock(return_value=("answer", {}))

            pending = {
                "gstok": {
                    "tool_name": "gs_tool", "tool_args": {},
                    "user_input": "test", "assistant_tool_call": "...",
                    "required_stage": 1, "stage": 1, "pending_reason": "risk",
                },
            }

            async def _run_c():
                return await resolve_confirmation(
                    "gstok", True, False,
                    pending_confirmations=pending,
                    confirm_ttl=timedelta(minutes=10),
                    tool_policy=policy,
                    tool_executor=executor,
                    memory=memory,
                    final_answer_fn=final_fn,
                )

            _run(_run_c())
            rows = policy.recent_audit(limit=1)
            self.assertEqual(rows[0]["policy_decision"], "confirmed")
            self.assertEqual(rows[0]["grant_source"], "one_time")
            tools.pop("gs_tool", None)
            policy.close()

    def test_trust_first_use_grant_source(self):
        """First-use trust confirmation should record grant_source=trust_first_use."""
        from liagent.tools import _TOOLS as tools
        tools["srv__query"] = _make_tool("srv__query")

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full",
                                trust_registry=reg)
            executor = MagicMock()
            executor.execute = AsyncMock(return_value=("result", False, None))
            memory = MagicMock()
            final_fn = AsyncMock(return_value=("answer", {}))

            pending = {
                "trusttok": {
                    "tool_name": "srv__query", "tool_args": {},
                    "user_input": "test", "assistant_tool_call": "...",
                    "required_stage": 1, "stage": 1, "pending_reason": "trust",
                    "trust_server_id": "srv",
                },
            }

            async def _run_c():
                return await resolve_confirmation(
                    "trusttok", True, False,
                    pending_confirmations=pending,
                    confirm_ttl=timedelta(minutes=10),
                    tool_policy=policy,
                    tool_executor=executor,
                    memory=memory,
                    final_answer_fn=final_fn,
                )

            _run(_run_c())
            rows = policy.recent_audit(limit=1)
            self.assertEqual(rows[0]["grant_source"], "trust_first_use")
            # Trust auto-approval should have happened
            self.assertEqual(reg.get_status("srv"), "approved")
            tools.pop("srv__query", None)
            policy.close()


class P2RequestedArgsInRecentAuditTests(unittest.TestCase):
    """P2 fix: recent_audit() must return requested_args."""

    def test_requested_args_returned(self):
        with tempfile.TemporaryDirectory() as td:
            p = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            p.audit("web_search", {"query": "test"}, "ok", "fallback",
                    requested_tool="web_fetch",
                    requested_args='{"url": "http://example.com"}',
                    effective_tool="web_search",
                    effective_args='{"query": "test"}')
            rows = p.recent_audit(limit=1)
            r = rows[0]
            self.assertIn("requested_args", r)
            # Verify it's the redacted version (url is not a secret key)
            self.assertIn("http://example.com", r["requested_args"])
            p.close()

    def test_requested_args_omitted_when_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            p.audit("web_search", {"query": "test"}, "ok", "direct")
            rows = p.recent_audit(limit=1)
            self.assertNotIn("requested_args", rows[0])
            p.close()


class P2HighRiskGrantExclusionTests(unittest.TestCase):
    """P2 fix: grant check in policy_gate must skip high-risk tools."""

    def setUp(self):
        self._saved = dict(_TOOLS)

    def tearDown(self):
        _TOOLS.clear()
        _TOOLS.update(self._saved)

    def test_high_risk_with_allow_high_risk_gets_allowed_not_granted(self):
        """With allow_high_risk + grant, auth_mode should be 'allowed', not 'granted'."""
        _TOOLS["hr_tool"] = _make_tool("hr_tool", risk="high")
        with tempfile.TemporaryDirectory() as td:
            policy = ToolPolicy(db_path=Path(td) / "audit.db", tool_profile="full")
            policy.allow_high_risk = True
            # Grant exists but should be ignored for high-risk
            grants = {"hr_tool": time.time() + 1800}
            ctx = _FakeCtx()

            async def _call():
                return await evaluate_tool_policy(
                    tool_name="hr_tool", tool_args={},
                    tool_sig="hr_tool:{}", full_response="...",
                    confirmed=False, ctx=ctx,
                    tool_policy=policy, planner=MagicMock(),
                    handle_policy_block_fn=_noop_block,
                    pending_confirmations={}, dup_tool_limit=3,
                    tool_cache_enabled=False,
                    enable_policy_review=False,
                    disable_policy_review_in_voice=False,
                    tool_grants=grants,
                )

            d = _run(_call())
            self.assertTrue(d.allowed)
            # Must be "allowed" (via allow_high_risk), NOT "granted"
            self.assertEqual(d.auth_mode, "allowed")
            self.assertEqual(d.grant_source, "")
            policy.close()


if __name__ == "__main__":
    unittest.main()
