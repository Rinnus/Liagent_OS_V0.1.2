"""Tests for browser wrapper tools."""
import asyncio
import unittest


class BrowserRiskClassifyTests(unittest.TestCase):
    """Test dynamic risk upgrade based on URL/selector."""

    def test_normal_click_is_write(self):
        from liagent.tools.browser import classify_browser_action
        self.assertEqual(classify_browser_action("browser_click", "https://example.com", "#btn"), "write")

    def test_login_url_upgrades_to_sensitive(self):
        from liagent.tools.browser import classify_browser_action
        self.assertEqual(classify_browser_action("browser_click", "https://example.com/login", "#btn"), "write_sensitive")

    def test_password_selector_upgrades(self):
        from liagent.tools.browser import classify_browser_action
        self.assertEqual(classify_browser_action("browser_fill", "https://example.com", "input[type=password]"), "write_sensitive")

    def test_payment_url_upgrades(self):
        from liagent.tools.browser import classify_browser_action
        self.assertEqual(classify_browser_action("browser_click", "https://shop.com/checkout", ".pay-btn"), "write_sensitive")

    def test_navigate_is_read(self):
        from liagent.tools.browser import classify_browser_action
        self.assertEqual(classify_browser_action("browser_navigate", "https://login.example.com", ""), "read")

    def test_screenshot_is_read(self):
        from liagent.tools.browser import classify_browser_action
        self.assertEqual(classify_browser_action("browser_screenshot", "https://example.com", ""), "read")

    def test_submit_always_sensitive(self):
        from liagent.tools.browser import classify_browser_action
        self.assertEqual(classify_browser_action("browser_submit", "https://example.com", ""), "write_sensitive")


class DynamicRiskUpgradeIntegrationTests(unittest.TestCase):
    """Test that _call_mcp actually applies dynamic risk classification."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_sensitive_click_blocked(self):
        """Clicking on a login page should be blocked by _call_mcp."""
        from liagent.tools.browser import _call_mcp
        result = self._run(_call_mcp("browser_click", {
            "element": "#login-btn",
            "url": "https://example.com/login",
        }))
        self.assertIn("Sensitive Action", result)

    def test_sensitive_fill_password_blocked(self):
        """Filling a password field should be blocked."""
        from liagent.tools.browser import _call_mcp
        result = self._run(_call_mcp("browser_fill", {
            "element": "input[type=password]",
            "value": "secret123",
        }))
        self.assertIn("Sensitive Action", result)

    def test_normal_click_not_blocked(self):
        """Normal click on non-sensitive page should not be blocked by risk check.
        (Will fail at bridge level since no MCP server, but should NOT say Sensitive.)"""
        from liagent.tools.browser import _call_mcp
        result = self._run(_call_mcp("browser_click", {
            "element": "#next",
        }))
        self.assertNotIn("Sensitive Action", result)

    def test_navigate_never_blocked_by_risk(self):
        """Navigate is a read tool — even to login URLs, risk check shouldn't block."""
        from liagent.tools.browser import _call_mcp
        result = self._run(_call_mcp("browser_navigate", {
            "url": "https://example.com/login",
        }))
        self.assertNotIn("Sensitive Action", result)

    def test_bridge_unavailable_reason_is_surface_to_user(self):
        from unittest.mock import patch

        from liagent.tools.browser import _call_mcp

        class _FakeBridge:
            def server_error(self, server_name):
                return "command not found: uvx"

        with patch("liagent.tools.mcp_bridge.get_bridge", return_value=_FakeBridge()):
            result = self._run(_call_mcp("browser_navigate", {"url": "https://example.com"}))
        self.assertIn("unavailable", result.lower())
        self.assertIn("uvx", result)


class WrapperMappingTests(unittest.TestCase):
    """Test compile-time fixed mapping."""

    def test_all_wrappers_have_mapping(self):
        from liagent.tools.browser import _WRAPPER_TO_MCP
        expected = {"browser_navigate", "browser_screenshot", "browser_extract",
                    "browser_click", "browser_fill", "browser_submit"}
        self.assertEqual(set(_WRAPPER_TO_MCP.keys()), expected)

    def test_mapping_values_are_tuples(self):
        from liagent.tools.browser import _WRAPPER_TO_MCP
        for name, mapping in _WRAPPER_TO_MCP.items():
            self.assertIsInstance(mapping, tuple)
            self.assertEqual(len(mapping), 2)
            self.assertEqual(mapping[0], "playwright")  # all map to same server


class ToolRegistrationTests(unittest.TestCase):
    """Test that browser wrapper tools are registered."""

    def test_tools_registered(self):
        from liagent.tools import get_tool
        import liagent.tools.browser  # noqa: F401
        for name in ("browser_navigate", "browser_screenshot", "browser_extract",
                      "browser_click", "browser_fill", "browser_submit"):
            td = get_tool(name)
            self.assertIsNotNone(td, f"{name} should be registered")

    def test_read_tools_medium_risk(self):
        from liagent.tools import get_tool
        import liagent.tools.browser  # noqa: F401
        for name in ("browser_navigate", "browser_screenshot", "browser_extract"):
            self.assertEqual(get_tool(name).risk_level, "medium")

    def test_write_tools_medium_with_confirmation(self):
        from liagent.tools import get_tool
        import liagent.tools.browser  # noqa: F401
        for name in ("browser_click", "browser_fill"):
            td = get_tool(name)
            self.assertEqual(td.risk_level, "medium")
            self.assertTrue(td.requires_confirmation)

    def test_submit_high_risk(self):
        from liagent.tools import get_tool
        import liagent.tools.browser  # noqa: F401
        td = get_tool("browser_submit")
        self.assertEqual(td.risk_level, "high")


class CuratedCatalogTests(unittest.TestCase):
    """Test playwright added to curated catalog."""

    def test_playwright_in_catalog(self):
        from liagent.tools.curated_catalog import CURATED_SERVERS
        names = [s["name"] for s in CURATED_SERVERS]
        self.assertIn("playwright", names)

    def test_playwright_wrapper_only(self):
        from liagent.tools.curated_catalog import CURATED_SERVERS
        pw = next(s for s in CURATED_SERVERS if s["name"] == "playwright")
        self.assertTrue(pw.get("wrapper_only", False))
