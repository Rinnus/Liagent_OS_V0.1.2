"""Tests for shell_exec tool."""
import asyncio
import unittest


class ShellExecToolTests(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_safe_echo(self):
        from liagent.tools.shell_exec import shell_exec
        result = self._run(shell_exec(command="echo hello"))
        self.assertIn("hello", result)

    def test_safe_pwd(self):
        from liagent.tools.shell_exec import shell_exec
        result = self._run(shell_exec(command="pwd"))
        self.assertIn("/", result)

    def test_denied_curl(self):
        from liagent.tools.shell_exec import shell_exec
        result = self._run(shell_exec(command="curl http://example.com"))
        self.assertIn("[Denied]", result)

    def test_denied_env(self):
        from liagent.tools.shell_exec import shell_exec
        result = self._run(shell_exec(command="env"))
        self.assertIn("[Denied]", result)

    def test_denied_find_exec(self):
        from liagent.tools.shell_exec import shell_exec
        result = self._run(shell_exec(command="find . -exec rm {} ;"))
        self.assertIn("[Denied]", result)

    def test_denied_python_c(self):
        from liagent.tools.shell_exec import shell_exec
        result = self._run(shell_exec(command="python -c 'print(1)'"))
        self.assertIn("[Denied]", result)

    def test_validator_rejects_empty(self):
        from liagent.tools.shell_exec import _validate_shell_exec
        ok, reason = _validate_shell_exec({"command": ""})
        self.assertFalse(ok)

    def test_validator_rejects_too_long(self):
        from liagent.tools.shell_exec import _validate_shell_exec
        ok, reason = _validate_shell_exec({"command": "echo " + "a" * 2001})
        self.assertFalse(ok)

    def test_rejection_message_has_alternative(self):
        from liagent.tools.shell_exec import shell_exec
        result = self._run(shell_exec(command="curl http://example.com"))
        self.assertIn("Alternative", result)

    def test_tool_registered(self):
        from liagent.tools import get_tool
        import liagent.tools.shell_exec  # noqa: F401
        td = get_tool("shell_exec")
        self.assertIsNotNone(td)
        self.assertEqual(td.risk_level, "medium")
