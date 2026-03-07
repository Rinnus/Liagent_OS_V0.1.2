"""Tests for the MCP bridge adapter layer."""

import asyncio
import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from liagent.config import MCPServerConfig
from liagent.tools import ToolDef, _TOOLS
from liagent.tools.policy import ToolPolicy


class _FakeContent:
    """Mimics mcp Content with a .text attribute."""

    def __init__(self, text: str):
        self.text = text


class _FakeCallResult:
    """Mimics mcp CallToolResult."""

    def __init__(self, content: list):
        self.content = content


class _FakeTool:
    """Mimics mcp Tool from list_tools response."""

    def __init__(self, name: str, description: str = "", inputSchema: dict | None = None):
        self.name = name
        self.description = description or name
        self.inputSchema = inputSchema or {"properties": {"query": {"type": "string"}}}


class _FakeListToolsResult:
    """Mimics mcp ListToolsResult."""

    def __init__(self, tools: list):
        self.tools = tools


def _run(coro):
    """Helper to run async in sync tests."""
    return asyncio.run(coro)


class TestMakeCallerBridgesKwargs(unittest.TestCase):
    """Verify _make_caller generates a closure that forwards kwargs to MCP call_tool."""

    def test_caller_passes_kwargs_and_joins_text(self):
        from liagent.tools.mcp_bridge import MCPBridge

        session = AsyncMock()
        session.call_tool = AsyncMock(
            return_value=_FakeCallResult([
                _FakeContent("Hello"),
                _FakeContent("World"),
            ])
        )

        caller = MCPBridge._make_caller(session, "my_tool")
        result = _run(caller(query="test", limit=5))

        session.call_tool.assert_awaited_once_with("my_tool", {"query": "test", "limit": 5})
        self.assertEqual(result, "Hello\nWorld")

    def test_caller_empty_content_returns_fallback(self):
        from liagent.tools.mcp_bridge import MCPBridge

        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=_FakeCallResult([]))

        caller = MCPBridge._make_caller(session, "empty_tool")
        result = _run(caller())

        self.assertIn("empty result", result)

    def test_caller_skips_non_text_content(self):
        from liagent.tools.mcp_bridge import MCPBridge

        # Content without .text attribute (e.g. image content)
        non_text = MagicMock(spec=[])  # no text attr
        text_content = _FakeContent("only text")

        session = AsyncMock()
        session.call_tool = AsyncMock(
            return_value=_FakeCallResult([non_text, text_content])
        )

        caller = MCPBridge._make_caller(session, "mixed_tool")
        result = _run(caller())
        self.assertEqual(result, "only text")

    def test_caller_enforces_timeout(self):
        from liagent.tools.mcp_bridge import MCPBridge

        async def _slow_call(tool_name, args):
            await asyncio.sleep(0.25)
            return _FakeCallResult([_FakeContent("slow")])

        session = AsyncMock()
        session.call_tool = AsyncMock(side_effect=_slow_call)

        caller = MCPBridge._make_caller(session, "slow_tool", timeout_sec=0.1)
        with self.assertRaises(asyncio.TimeoutError):
            _run(caller())

    def test_caller_truncates_oversized_response(self):
        from liagent.tools.mcp_bridge import MCPBridge

        session = AsyncMock()
        session.call_tool = AsyncMock(
            return_value=_FakeCallResult([_FakeContent("A" * 20000)])
        )
        caller = MCPBridge._make_caller(
            session,
            "big_tool",
            max_response_bytes=4096,
        )
        result = _run(caller())
        self.assertIn("[MCP response truncated]", result)
        self.assertLess(len(result.encode("utf-8")), 5000)


class TestRegisterCreatesToolDefs(unittest.TestCase):
    """Verify register_all() populates _TOOLS with namespaced ToolDefs."""

    def setUp(self):
        # Save and clear _TOOLS to isolate test
        self._saved_tools = dict(_TOOLS)
        for k in list(_TOOLS):
            if "__" in k:
                del _TOOLS[k]

    def tearDown(self):
        # Restore
        for k in list(_TOOLS):
            if "__" in k:
                del _TOOLS[k]
        _TOOLS.update(self._saved_tools)

    def test_register_two_tools(self):
        from liagent.tools.mcp_bridge import MCPBridge

        cfg = MCPServerConfig(
            name="testserver",
            command="echo",
            risk_level="low",
            network_access=False,
            filesystem_access=True,
        )
        bridge = MCPBridge([cfg])

        # Manually inject a mock session
        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_FakeListToolsResult([
                _FakeTool("tool_a", "Tool A desc"),
                _FakeTool("tool_b", "Tool B desc"),
            ])
        )
        bridge._sessions["testserver"] = (session, None)

        _run(bridge.register_all())

        self.assertIn("testserver__tool_a", _TOOLS)
        self.assertIn("testserver__tool_b", _TOOLS)
        self.assertEqual(len(bridge.registered_tools), 2)

    def test_namespace_isolation(self):
        from liagent.tools.mcp_bridge import MCPBridge

        cfg = MCPServerConfig(name="github", command="npx", risk_level="medium")
        bridge = MCPBridge([cfg])

        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_FakeListToolsResult([_FakeTool("create_issue")])
        )
        bridge._sessions["github"] = (session, None)
        _run(bridge.register_all())

        # Must be namespaced
        self.assertIn("github__create_issue", _TOOLS)
        self.assertNotIn("create_issue", bridge.registered_tools)


class TestRiskFromConfig(unittest.TestCase):
    """Verify ToolDef.risk_level comes from MCPServerConfig."""

    def setUp(self):
        self._saved_tools = dict(_TOOLS)

    def tearDown(self):
        for k in list(_TOOLS):
            if "__" in k:
                del _TOOLS[k]
        _TOOLS.update(self._saved_tools)

    def test_risk_level_propagated(self):
        from liagent.tools.mcp_bridge import MCPBridge

        cfg = MCPServerConfig(name="risky", command="echo", risk_level="high")
        bridge = MCPBridge([cfg])

        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_FakeListToolsResult([_FakeTool("danger_tool")])
        )
        bridge._sessions["risky"] = (session, None)
        _run(bridge.register_all())

        td = _TOOLS["risky__danger_tool"]
        self.assertEqual(td.risk_level, "high")

    def test_capability_from_config(self):
        from liagent.tools.mcp_bridge import MCPBridge

        cfg = MCPServerConfig(
            name="fs", command="echo",
            network_access=False, filesystem_access=True,
        )
        bridge = MCPBridge([cfg])

        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_FakeListToolsResult([_FakeTool("read_dir")])
        )
        bridge._sessions["fs"] = (session, None)
        _run(bridge.register_all())

        td = _TOOLS["fs__read_dir"]
        self.assertFalse(td.capability.network_access)
        self.assertTrue(td.capability.filesystem_access)


class TestShutdownCleansRegistry(unittest.TestCase):
    """Verify shutdown removes MCP tools from _TOOLS."""

    def setUp(self):
        self._saved_tools = dict(_TOOLS)

    def tearDown(self):
        for k in list(_TOOLS):
            if "__" in k:
                del _TOOLS[k]
        _TOOLS.update(self._saved_tools)

    def test_shutdown_removes_tools(self):
        from liagent.tools.mcp_bridge import MCPBridge

        cfg = MCPServerConfig(name="temp", command="echo")
        bridge = MCPBridge([cfg])

        session = AsyncMock()
        session.list_tools = AsyncMock(
            return_value=_FakeListToolsResult([_FakeTool("ephemeral")])
        )
        bridge._sessions["temp"] = (session, None)
        _run(bridge.register_all())
        self.assertIn("temp__ephemeral", _TOOLS)

        _run(bridge.shutdown())
        self.assertNotIn("temp__ephemeral", _TOOLS)
        self.assertEqual(bridge.registered_tools, [])
        self.assertEqual(bridge.connected_servers, [])


class TestMcpPreflight(unittest.TestCase):
    def test_missing_command_is_recorded_as_server_error(self):
        from liagent.tools.mcp_bridge import MCPBridge

        cfg = MCPServerConfig(name="fetch", command="uvx")
        bridge = MCPBridge([cfg])
        with patch("liagent.tools.mcp_bridge.shutil.which", return_value=None):
            _run(bridge.connect_all())
        self.assertEqual(bridge.connected_servers, [])
        self.assertIn("fetch", bridge.server_errors)
        self.assertIn("uvx", bridge.server_error("fetch"))
        msg = _run(bridge.call_tool("fetch", "browser_navigate", {}))
        self.assertIn("unavailable", msg.lower())


class TestPolicyAllowsMCPResearch(unittest.TestCase):
    """Verify research profile allows medium-risk MCP tools."""

    def test_mcp_medium_allowed_in_research(self):
        policy = ToolPolicy(tool_profile="research")
        td = ToolDef(
            name="github__search_repos",
            description="Search repos",
            parameters={"properties": {}},
            func=lambda: None,
            risk_level="medium",
        )
        allowed, reason = policy.evaluate(td, {})
        self.assertTrue(allowed, f"Expected allowed, got: {reason}")

    def test_mcp_high_blocked_in_research(self):
        policy = ToolPolicy(tool_profile="research")
        td = ToolDef(
            name="github__delete_repo",
            description="Delete repo",
            parameters={"properties": {}},
            func=lambda: None,
            risk_level="high",
        )
        allowed, reason = policy.evaluate(td, {})
        self.assertFalse(allowed)
        self.assertIn("profile", reason)

    def test_mcp_low_allowed_in_research(self):
        policy = ToolPolicy(tool_profile="research")
        td = ToolDef(
            name="fs__list_dir",
            description="List directory",
            parameters={"properties": {}},
            func=lambda: None,
            risk_level="low",
        )
        allowed, reason = policy.evaluate(td, {})
        self.assertTrue(allowed, f"Expected allowed, got: {reason}")

    def test_non_mcp_tool_blocked_in_research(self):
        """A non-MCP tool not in the allowset should still be blocked."""
        policy = ToolPolicy(tool_profile="research")
        td = ToolDef(
            name="custom_tool",
            description="Custom",
            parameters={"properties": {}},
            func=lambda: None,
            risk_level="low",
        )
        allowed, reason = policy.evaluate(td, {})
        self.assertFalse(allowed)

    def test_mcp_allowed_in_full_profile(self):
        policy = ToolPolicy(tool_profile="full")
        td = ToolDef(
            name="github__create_issue",
            description="Create issue",
            parameters={"properties": {}},
            func=lambda: None,
            risk_level="high",
        )
        # full profile allows everything (profile_allowset is None)
        # but high risk may be blocked by risk policy — test profile gate only
        # High risk is blocked by allow_high_risk policy, not profile
        allowed, reason = policy.evaluate(td, {})
        self.assertNotIn("profile", reason)


class TestDisabledServerSkipped(unittest.TestCase):
    """Verify disabled servers are not connected."""

    def test_disabled_server_filtered(self):
        from liagent.tools.mcp_bridge import MCPBridge

        enabled = MCPServerConfig(name="active", command="echo", enabled=True)
        disabled = MCPServerConfig(name="inactive", command="echo", enabled=False)
        bridge = MCPBridge([enabled, disabled])

        self.assertEqual(len(bridge._configs), 1)
        self.assertEqual(bridge._configs[0].name, "active")


class TestImportGuardNoMCP(unittest.TestCase):
    """Verify graceful fallback when mcp SDK is not installed."""

    def test_mcp_available_flag(self):
        from liagent.tools.mcp_bridge import mcp_available
        # Just verify the function exists and returns a bool
        self.assertIsInstance(mcp_available(), bool)


class TestMCPDiscovery(unittest.TestCase):
    def test_discover_local_servers_reads_mcpservers_format(self):
        from liagent.tools.mcp_bridge import MCPBridge

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "servers.json"
            p.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "github": {
                                "command": "npx",
                                "args": ["-y", "@modelcontextprotocol/server-github"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            servers = MCPBridge.discover_local_servers([td])
            self.assertEqual(len(servers), 1)
            self.assertEqual(servers[0].name, "github")
            self.assertEqual(servers[0].command, "npx")

    def test_merge_servers_prefers_configured_entries(self):
        from liagent.tools.mcp_bridge import MCPBridge

        configured = [
            MCPServerConfig(name="github", command="custom-github", args=[]),
            MCPServerConfig(name="sqlite", command="uvx", args=["mcp-server-sqlite"]),
        ]
        discovered = [
            MCPServerConfig(name="github", command="npx", args=["-y"]),
            MCPServerConfig(name="filesystem", command="npx", args=["-y"]),
        ]
        merged = MCPBridge.merge_servers(configured, discovered)
        merged_map = {s.name: s for s in merged}
        self.assertIn("github", merged_map)
        self.assertIn("sqlite", merged_map)
        self.assertIn("filesystem", merged_map)
        self.assertEqual(merged_map["github"].command, "custom-github")


class TestMCPLifecycleWorker(unittest.TestCase):
    def test_lifecycle_ops_run_in_single_worker_task(self):
        from liagent.tools.mcp_bridge import MCPBridge

        bridge = MCPBridge([MCPServerConfig(name="x", command="echo")])
        seen_tasks: list[int] = []

        async def _fake_connect_impl():
            seen_tasks.append(id(asyncio.current_task()))

        async def _fake_shutdown_impl():
            seen_tasks.append(id(asyncio.current_task()))

        bridge._connect_all_impl = _fake_connect_impl
        bridge._shutdown_impl = _fake_shutdown_impl

        async def _exercise():
            await bridge._run_lifecycle_op("connect_all")
            await bridge._run_lifecycle_op("shutdown")
            await bridge._run_lifecycle_op("_stop_worker")

        _run(_exercise())
        self.assertEqual(len(seen_tasks), 2)
        self.assertEqual(len(set(seen_tasks)), 1)


if __name__ == "__main__":
    unittest.main()
