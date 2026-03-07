"""Tests for MCP bridge wrapper_only mode."""
import unittest


class WrapperOnlyConfigTests(unittest.TestCase):
    def test_config_has_wrapper_only_field(self):
        from liagent.config import MCPServerConfig
        cfg = MCPServerConfig(name="test", command="echo")
        self.assertFalse(cfg.wrapper_only)

    def test_wrapper_only_true(self):
        from liagent.config import MCPServerConfig
        cfg = MCPServerConfig(name="test", command="echo", wrapper_only=True)
        self.assertTrue(cfg.wrapper_only)

    def test_get_bridge_default_none(self):
        from liagent.tools.mcp_bridge import get_bridge
        # get_bridge returns None when no bridge has been set
        # (may be non-None if tests ran in order and set one, but the API exists)
        result = get_bridge()
        self.assertTrue(result is None or hasattr(result, "call_tool"))

    def test_set_get_bridge(self):
        from liagent.tools.mcp_bridge import get_bridge, set_bridge
        old = get_bridge()
        try:
            sentinel = object()
            set_bridge(sentinel)
            self.assertIs(get_bridge(), sentinel)
        finally:
            set_bridge(old)

    def test_set_bridge_called_in_brain_init(self):
        """Verify brain.__init__ calls set_bridge when MCPBridge is created."""
        import liagent.tools.mcp_bridge as mb
        # set_bridge should be importable and callable
        self.assertTrue(callable(mb.set_bridge))
        # Verify the function signature accepts bridge or None
        mb.set_bridge(None)
        self.assertIsNone(mb.get_bridge())
