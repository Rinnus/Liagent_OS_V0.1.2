"""Tests for stateful_repl tool and session manager."""
import asyncio
import unittest


class ReplSessionTests(unittest.TestCase):
    def test_basic_run(self):
        async def _test():
            from liagent.tools.stateful_repl import ReplSessionManager
            mgr = ReplSessionManager()
            mgr.set_mode_sync("sandboxed")
            try:
                result = await mgr.execute("test_session", "x = 42\nprint(x)")
                assert "42" in result, result
            finally:
                await mgr.shutdown()
        asyncio.run(_test())

    def test_state_persists_across_calls(self):
        async def _test():
            from liagent.tools.stateful_repl import ReplSessionManager
            mgr = ReplSessionManager()
            mgr.set_mode_sync("sandboxed")
            try:
                await mgr.execute("s1", "counter = 100")
                result = await mgr.execute("s1", "counter += 1\nprint(counter)")
                assert "101" in result, result
            finally:
                await mgr.shutdown()
        asyncio.run(_test())

    def test_session_isolation(self):
        async def _test():
            from liagent.tools.stateful_repl import ReplSessionManager
            mgr = ReplSessionManager()
            mgr.set_mode_sync("sandboxed")
            try:
                await mgr.execute("s1", "secret = 'abc'")
                result = await mgr.execute("s2", "print(secret)")
                assert "Error" in result, result
            finally:
                await mgr.shutdown()
        asyncio.run(_test())

    def test_reset(self):
        async def _test():
            from liagent.tools.stateful_repl import ReplSessionManager
            mgr = ReplSessionManager()
            mgr.set_mode_sync("sandboxed")
            try:
                await mgr.execute("s1", "x = 1")
                await mgr.execute("s1", "", reset=True)
                result = await mgr.execute("s1", "print(x)")
                assert "Error" in result, result
            finally:
                await mgr.shutdown()
        asyncio.run(_test())

    def test_crash_recovery(self):
        async def _test():
            from liagent.tools.stateful_repl import ReplSessionManager
            mgr = ReplSessionManager()
            mgr.set_mode_sync("sandboxed")
            try:
                await mgr.execute("s1", "x = 1")
                session = mgr._sessions.get("s1")
                if session and session._proc:
                    session._proc.kill()
                result = await mgr.execute("s1", "print('recovered')")
                assert "recovered" in result, result
            finally:
                await mgr.shutdown()
        asyncio.run(_test())

    def test_tool_registered(self):
        from liagent.tools import get_tool
        import liagent.tools.stateful_repl  # noqa: F401
        td = get_tool("stateful_repl")
        self.assertIsNotNone(td)

    def test_error_paths_do_not_crash_worker(self):
        async def _test():
            from liagent.tools.stateful_repl import ReplSessionManager
            mgr = ReplSessionManager()
            mgr.set_mode_sync("sandboxed")
            try:
                r1 = await mgr.execute("s1", "1/0")
                assert "ZeroDivisionError" in r1, r1
                assert "REPL process crashed" not in r1, r1

                r2 = await mgr.execute("s1", "import os")
                assert "ImportError" in r2, r2
                assert "REPL process crashed" not in r2, r2

                r3 = await mgr.execute("s1", "import subprocess")
                assert "ImportError" in r3, r3
                assert "REPL process crashed" not in r3, r3

                r4 = await mgr.execute("s1", "print('alive')")
                assert "alive" in r4, r4
            finally:
                await mgr.shutdown()
        asyncio.run(_test())

    def test_mode_off_blocks_execution(self):
        async def _test():
            from liagent.tools.stateful_repl import ReplSessionManager
            mgr = ReplSessionManager()
            mgr.set_mode_sync("off")
            try:
                result = await mgr.execute("s1", "print('x')")
                assert "repl_mode=off" in result, result
            finally:
                await mgr.shutdown()
        asyncio.run(_test())

    def test_status_reset_and_kill(self):
        async def _test():
            from liagent.tools.stateful_repl import ReplSessionManager
            mgr = ReplSessionManager()
            mgr.set_mode_sync("sandboxed")
            try:
                await mgr.execute("s1", "x = 1")
                st = mgr.status("s1")
                assert st.get("exists") is True, st
                assert st.get("mode") == "sandboxed", st

                ok_reset = await mgr.reset_session("s1")
                assert ok_reset is True

                ok_kill = await mgr.kill_session("s1")
                assert ok_kill is True
                st2 = mgr.status("s1")
                assert st2.get("exists") is False, st2
            finally:
                await mgr.shutdown()
        asyncio.run(_test())
