# tests/test_trust_integration.py
"""End-to-end integration tests for the trust registry system."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from liagent.config import MCPServerConfig
from liagent.tools import ToolDef, ToolCapability, _TOOLS
from liagent.tools.policy import ToolPolicy
from liagent.tools.trust_registry import TrustRegistry


def _run(coro):
    return asyncio.run(coro)


class _FakeTool:
    def __init__(self, name, description="", inputSchema=None):
        self.name = name
        self.description = description or name
        self.inputSchema = inputSchema or {"properties": {}}


class _FakeListToolsResult:
    def __init__(self, tools):
        self.tools = tools


class TrustE2ETests(unittest.TestCase):
    """Full lifecycle: discover -> block -> confirm -> allow -> revoke -> unload."""

    def setUp(self):
        self._saved_tools = dict(_TOOLS)

    def tearDown(self):
        for k in list(_TOOLS):
            if "__" in k:
                del _TOOLS[k]
        _TOOLS.update(self._saved_tools)

    def test_full_trust_lifecycle(self):
        from liagent.tools.mcp_bridge import MCPBridge

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            policy = ToolPolicy(
                db_path=Path(td) / "audit.db",
                tool_profile="full",
                trust_registry=reg,
            )

            # 1. Server is unknown — pre-connection filter blocks it
            cfg = MCPServerConfig(name="newmcp", command="echo")
            reg.ensure_registered("newmcp", source="discovered")
            self.assertFalse(reg.is_connectable("newmcp"))

            # 2. Approve the server (simulates first-use confirmation)
            reg.set_status("newmcp", "approved", source="first_use")
            self.assertTrue(reg.is_connectable("newmcp"))

            # 3-6. MCPBridge register + policy + revoke + reload must run
            #    inside a single event loop because the lifecycle worker
            #    calls _shutdown_impl on CancelledError (when asyncio.run
            #    closes the loop), which clears _sessions.
            bridge = MCPBridge([cfg], trust_registry=reg)
            session = AsyncMock()
            session.list_tools = AsyncMock(
                return_value=_FakeListToolsResult([_FakeTool("query")])
            )
            bridge._sessions["newmcp"] = (session, None)

            results = {}

            async def _full_lifecycle():
                # 3. Register tools
                await bridge.register_all()
                results["registered"] = "newmcp__query" in _TOOLS

                # 4. Policy allows approved MCP tool
                tool = _TOOLS.get("newmcp__query")
                results["tool"] = tool
                if tool:
                    allowed, reason = policy.evaluate(tool, {})
                    results["approved_allowed"] = allowed
                    results["approved_reason"] = reason

                # 5. Revoke
                reg.set_status("newmcp", "revoked", source="manual")
                if tool:
                    allowed, reason = policy.evaluate(tool, {})
                    results["revoked_allowed"] = allowed
                    results["revoked_reason"] = reason

                # 6. Hot-reload unloads revoked tools
                await bridge.reload_servers([cfg])
                results["after_reload"] = "newmcp__query" in _TOOLS

            _run(_full_lifecycle())

            self.assertTrue(results["registered"], "Tool should be registered after register_all")
            self.assertIsNotNone(results["tool"], "newmcp__query should exist in _TOOLS")
            self.assertTrue(results["approved_allowed"],
                          f"Expected allowed, got: {results.get('approved_reason')}")
            self.assertFalse(results["revoked_allowed"])
            self.assertIn("revoked", results["revoked_reason"])
            self.assertFalse(results["after_reload"],
                           "newmcp__query should be removed after reload_servers")

    def test_unknown_tool_returns_confirmation_required(self):
        """Verify the exact reason string format that policy_gate.py expects."""
        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            policy = ToolPolicy(
                db_path=Path(td) / "audit.db",
                tool_profile="full",
                trust_registry=reg,
            )
            tool = ToolDef(
                name="unknown_server__tool", description="Test",
                parameters={"properties": {}}, func=lambda: None, risk_level="low",
            )
            allowed, reason = policy.evaluate(tool, {})
            self.assertFalse(allowed)
            # policy_gate.py L128: reason.startswith("confirmation required")
            self.assertTrue(reason.startswith("confirmation required"),
                          f"Expected 'confirmation required...' for policy_gate flow, got: {reason}")

    def test_curated_catalog_auto_approved(self):
        from liagent.tools.curated_catalog import bootstrap_curated_catalog

        with tempfile.TemporaryDirectory() as td:
            mcp_dir = Path(td) / "mcp.d"
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)

            policy = ToolPolicy(
                db_path=Path(td) / "audit.db",
                tool_profile="research",
                trust_registry=reg,
            )

            # Curated MCP tool passes both trust AND profile checks
            tool = ToolDef(
                name="filesystem__read_file", description="Read file",
                parameters={"properties": {}}, func=lambda: None, risk_level="low",
                capability=ToolCapability(filesystem_access=True),
            )
            allowed, reason = policy.evaluate(tool, {})
            self.assertTrue(allowed, f"Expected allowed for curated tool, got: {reason}")

    def test_persistence_survives_restart(self):
        with tempfile.TemporaryDirectory() as td:
            trust_path = Path(td) / "trust.json"

            reg1 = TrustRegistry(store_path=trust_path)
            reg1.set_status("server_a", "approved", source="first_use")
            reg1.set_status("server_b", "revoked", source="manual")

            # "Restart" — new instance
            reg2 = TrustRegistry(store_path=trust_path)
            self.assertEqual(reg2.get_status("server_a"), "approved")
            self.assertEqual(reg2.get_status("server_b"), "revoked")
            self.assertEqual(reg2.get_status("server_c"), "unknown")

    def test_pre_connection_filter_blocks_revoked_only(self):
        """Revoked servers are blocked. Unknown servers connect (gated by evaluate)."""
        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            reg.set_status("approved_srv", "approved", source="curated")
            # "unknown_srv" not in registry -> unknown
            # "revoked_srv" explicitly revoked
            reg.set_status("revoked_srv", "revoked", source="manual")

            servers = [
                MCPServerConfig(name="approved_srv", command="echo"),
                MCPServerConfig(name="unknown_srv", command="echo"),
                MCPServerConfig(name="revoked_srv", command="echo"),
            ]

            # Simulate _resolve_mcp_servers trust filter (revoked-only block)
            for s in servers:
                reg.ensure_registered(s.name, source="discovered")
            connectable = [s for s in servers if reg.get_status(s.name) != "revoked"]

            self.assertEqual(len(connectable), 2)
            names = sorted(s.name for s in connectable)
            self.assertEqual(names, ["approved_srv", "unknown_srv"])


if __name__ == "__main__":
    unittest.main()
