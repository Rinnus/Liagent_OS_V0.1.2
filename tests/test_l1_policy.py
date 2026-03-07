"""Tests for L1 tool profile integration."""
import unittest


class L1ProfileTests(unittest.TestCase):
    def test_research_has_browser_read_tools(self):
        from liagent.tools.policy import _TOOL_PROFILE_MAP
        research = _TOOL_PROFILE_MAP["research"]
        for name in ("browser_navigate", "browser_screenshot", "browser_extract"):
            self.assertIn(name, research, f"{name} should be in research profile")

    def test_research_no_browser_write_tools(self):
        from liagent.tools.policy import _TOOL_PROFILE_MAP
        research = _TOOL_PROFILE_MAP["research"]
        for name in ("browser_click", "browser_fill", "browser_submit"):
            self.assertNotIn(name, research, f"{name} should NOT be in research profile")

    def test_research_has_shell_exec(self):
        from liagent.tools.policy import _TOOL_PROFILE_MAP
        research = _TOOL_PROFILE_MAP["research"]
        self.assertIn("shell_exec", research)

    def test_research_no_stateful_repl(self):
        from liagent.tools.policy import _TOOL_PROFILE_MAP
        research = _TOOL_PROFILE_MAP["research"]
        self.assertNotIn("stateful_repl", research)

    def test_minimal_no_l1_tools(self):
        from liagent.tools.policy import _TOOL_PROFILE_MAP
        minimal = _TOOL_PROFILE_MAP["minimal"]
        for name in ("shell_exec", "stateful_repl", "browser_navigate",
                      "browser_click", "browser_fill", "browser_submit"):
            self.assertNotIn(name, minimal)
