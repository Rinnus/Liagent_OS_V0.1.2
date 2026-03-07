"""Tests for REPL shutdown integration."""
import asyncio
import unittest
from unittest.mock import patch, AsyncMock


class ReplShutdownTests(unittest.TestCase):
    def test_shutdown_repl_callable(self):
        """shutdown_repl() should be callable and clean up sessions."""
        from liagent.tools.stateful_repl import shutdown_repl
        asyncio.run(shutdown_repl())

    def test_shutdown_repl_cleans_sessions(self):
        """After shutdown, all sessions should be cleared."""
        from liagent.tools.stateful_repl import ReplSessionManager

        async def _test():
            mgr = ReplSessionManager()
            await mgr.execute("s1", "x = 1")
            assert "s1" in mgr._sessions
            await mgr.shutdown()
            assert len(mgr._sessions) == 0

        asyncio.run(_test())

    def test_brain_shutdown_imports_repl(self):
        """brain.shutdown() code path should be able to import shutdown_repl."""
        from liagent.tools.stateful_repl import shutdown_repl
        self.assertTrue(callable(shutdown_repl))
