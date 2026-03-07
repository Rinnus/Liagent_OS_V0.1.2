"""Tests for tool fallback strategy chain."""
import asyncio
import unittest


class FallbackMapTests(unittest.TestCase):
    """Test TOOL_FALLBACK_MAP entries produce valid args."""

    def test_web_search_refined(self):
        from liagent.agent.tool_executor import TOOL_FALLBACK_MAP
        strategies = TOOL_FALLBACK_MAP.get("web_search", [])
        self.assertTrue(len(strategies) >= 1)
        stype, *rest = strategies[0]
        self.assertEqual(stype, "retry_refined")

    def test_stock_fallback_to_web_search(self):
        from liagent.agent.tool_executor import TOOL_FALLBACK_MAP
        strategies = TOOL_FALLBACK_MAP.get("stock", [])
        fallback = [s for s in strategies if s[0] == "fallback_tool"]
        self.assertTrue(len(fallback) >= 1)
        self.assertEqual(fallback[0][1], "web_search")

    def test_stock_fallback_args(self):
        from liagent.agent.tool_executor import TOOL_FALLBACK_MAP
        strategies = TOOL_FALLBACK_MAP.get("stock", [])
        fallback = [s for s in strategies if s[0] == "fallback_tool"][0]
        args_fn = fallback[2]
        result = args_fn({"symbol": "AAPL"})
        self.assertIn("AAPL", result.get("query", ""))

    def test_unknown_tool_has_no_fallback(self):
        from liagent.agent.tool_executor import TOOL_FALLBACK_MAP
        self.assertNotIn("unknown_tool", TOOL_FALLBACK_MAP)


class ExecuteWithFallbackTests(unittest.TestCase):
    """Test ToolExecutor.execute_with_fallback()."""

    def _make_tool_def(self, func, name="test_tool"):
        from liagent.tools import ToolDef, ToolCapability
        return ToolDef(
            name=name, description="test", parameters={},
            func=func, capability=ToolCapability(),
        )

    def _make_policy(self):
        class FakePolicy:
            def sanitize_output(self, td, result):
                return str(result)
        return FakePolicy()

    def test_fallback_on_failure(self):
        from liagent.agent.tool_executor import ToolExecutor
        call_log = []

        async def failing_stock(**kwargs):
            call_log.append(("stock", kwargs))
            raise RuntimeError("API down")

        async def working_search(**kwargs):
            call_log.append(("web_search", kwargs))
            return "AAPL: $198"

        executor = ToolExecutor(self._make_policy(), retry_count=0, timeout_sec=5.0)

        from liagent.tools import ToolDef, ToolCapability
        stock_def = self._make_tool_def(failing_stock, "stock")
        search_def = self._make_tool_def(working_search, "web_search")

        def get_tool_fn(name):
            return {"stock": stock_def, "web_search": search_def}.get(name)

        obs, is_err, err_type, eff_tool, eff_args = asyncio.run(
            executor.execute_with_fallback(stock_def, {"symbol": "AAPL"}, get_tool_fn=get_tool_fn)
        )
        self.assertFalse(is_err)
        self.assertIn("198", obs)
        self.assertEqual(eff_tool, "web_search")
        self.assertIn("AAPL", eff_args.get("query", ""))


if __name__ == "__main__":
    unittest.main()
