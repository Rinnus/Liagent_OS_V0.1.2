"""Tests for TaskPlanner.decompose() plan parsing."""
import json
import unittest


class ParsePlanTests(unittest.TestCase):
    """Test _parse_plan_json() extracts valid plans from LLM output."""

    def test_valid_plan(self):
        from liagent.agent.planner import _parse_plan_json
        raw = json.dumps({
            "goal": "Compare stocks",
            "steps": [
                {"id": "s1", "title": "Query AAPL", "tool_hint": "stock",
                 "done_criteria": "Get AAPL price", "status": "pending", "evidence_ref": None},
                {"id": "s2", "title": "Summarize", "tool_hint": None,
                 "done_criteria": "Output comparison", "status": "pending", "evidence_ref": None},
            ]
        })
        plan = _parse_plan_json(raw)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["goal"], "Compare stocks")
        self.assertEqual(len(plan["steps"]), 2)
        self.assertEqual(plan["steps"][0]["id"], "s1")

    def test_plan_in_markdown_fence(self):
        from liagent.agent.planner import _parse_plan_json
        raw = '```json\n{"goal": "Test", "steps": [{"id": "s1", "title": "Do X", "tool_hint": null, "done_criteria": "X done", "status": "pending", "evidence_ref": null}]}\n```'
        plan = _parse_plan_json(raw)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan["steps"]), 1)

    def test_invalid_json_returns_none(self):
        from liagent.agent.planner import _parse_plan_json
        self.assertIsNone(_parse_plan_json("not json"))
        self.assertIsNone(_parse_plan_json(""))

    def test_missing_steps_returns_none(self):
        from liagent.agent.planner import _parse_plan_json
        raw = json.dumps({"goal": "Test"})
        self.assertIsNone(_parse_plan_json(raw))

    def test_empty_steps_returns_none(self):
        from liagent.agent.planner import _parse_plan_json
        raw = json.dumps({"goal": "Test", "steps": []})
        self.assertIsNone(_parse_plan_json(raw))

    def test_steps_capped_at_8(self):
        from liagent.agent.planner import _parse_plan_json
        steps = [{"id": f"s{i}", "title": f"Step {i}", "tool_hint": None,
                  "done_criteria": f"Done {i}", "status": "pending", "evidence_ref": None}
                 for i in range(12)]
        raw = json.dumps({"goal": "Big task", "steps": steps})
        plan = _parse_plan_json(raw)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan["steps"]), 8)


class ShouldPlanTests(unittest.TestCase):
    """Test _should_plan() heuristic gate."""

    def test_simple_greeting_no_plan(self):
        from liagent.agent.planner import _should_plan
        self.assertFalse(_should_plan("hello"))
        self.assertFalse(_should_plan("hi"))

    def test_single_entity_no_plan(self):
        from liagent.agent.planner import _should_plan
        self.assertFalse(_should_plan("AAPL stock price lookup"))  # single ticker, long enough, no signal
        self.assertFalse(_should_plan("What is the weather today?"))

    def test_comparison_triggers_plan(self):
        from liagent.agent.planner import _should_plan
        self.assertTrue(_should_plan("Check AAPL, TSLA, and NVDA and compare them"))
        self.assertTrue(_should_plan("compare AAPL and TSLA"))

    def test_multi_step_triggers_plan(self):
        from liagent.agent.planner import _should_plan
        self.assertTrue(_should_plan("Search for a Python async tutorial, then summarize the key points"))
        self.assertTrue(_should_plan("Check the news first, then analyze the market trend"))

    def test_short_query_no_plan(self):
        from liagent.agent.planner import _should_plan
        self.assertFalse(_should_plan(""))
        self.assertFalse(_should_plan("ok"))


class FormatPlanStatusTests(unittest.TestCase):
    """Test format_plan_status() output."""

    def test_format_with_progress(self):
        from liagent.agent.planner import format_plan_status
        steps = [
            {"id": "s1", "title": "Query AAPL", "status": "done", "evidence_ref": "AAPL=$198", "done_criteria": "Get price"},
            {"id": "s2", "title": "Query TSLA", "status": "pending", "evidence_ref": None, "done_criteria": "Get price"},
            {"id": "s3", "title": "Compare", "status": "pending", "evidence_ref": None, "done_criteria": "Output comparison"},
        ]
        result = format_plan_status("Compare stocks", steps, current_idx=1)
        self.assertIn("Query AAPL", result)
        self.assertIn("YOU ARE HERE", result)
        self.assertIn("Query TSLA", result)
        self.assertIn("Done criteria:", result)


class CompletionGateTests(unittest.TestCase):
    """Test plan completion gate logic."""

    def test_pending_non_final_blocks_completion(self):
        from liagent.agent.planner import should_block_completion
        steps = [
            {"id": "s1", "title": "Query AAPL", "status": "done"},
            {"id": "s2", "title": "Query TSLA", "status": "pending"},
            {"id": "s3", "title": "Summarize", "status": "pending"},
        ]
        blocked, msg = should_block_completion(steps, plan_idx=1, plan_total=3)
        self.assertTrue(blocked)
        self.assertIn("1", msg)

    def test_all_non_final_done_allows_completion(self):
        from liagent.agent.planner import should_block_completion
        steps = [
            {"id": "s1", "title": "Query AAPL", "status": "done"},
            {"id": "s2", "title": "Query TSLA", "status": "done"},
            {"id": "s3", "title": "Summarize", "status": "pending"},
        ]
        blocked, msg = should_block_completion(steps, plan_idx=2, plan_total=3)
        self.assertFalse(blocked)

    def test_all_done_allows_completion(self):
        from liagent.agent.planner import should_block_completion
        steps = [
            {"id": "s1", "title": "Query AAPL", "status": "done"},
            {"id": "s2", "title": "Query TSLA", "status": "done"},
        ]
        blocked, msg = should_block_completion(steps, plan_idx=2, plan_total=2)
        self.assertFalse(blocked)

    def test_last_step_pending_with_done_prefixes_allows(self):
        from liagent.agent.planner import should_block_completion
        steps = [
            {"id": "s1", "title": "Query AAPL", "status": "done"},
            {"id": "s2", "title": "Summarize", "status": "pending"},
        ]
        blocked, msg = should_block_completion(steps, plan_idx=1, plan_total=2)
        self.assertFalse(blocked)

    def test_skipped_steps_dont_block(self):
        from liagent.agent.planner import should_block_completion
        steps = [
            {"id": "s1", "title": "Query AAPL", "status": "done"},
            {"id": "s2", "title": "Query TSLA", "status": "skipped"},
            {"id": "s3", "title": "Summarize", "status": "done"},
        ]
        blocked, msg = should_block_completion(steps, plan_idx=3, plan_total=3)
        self.assertFalse(blocked)

    def test_false_pass_window_caught(self):
        """Regression: plan_idx at last step but earlier step still pending."""
        from liagent.agent.planner import should_block_completion
        steps = [
            {"id": "s1", "title": "Query AAPL", "status": "done"},
            {"id": "s2", "title": "Query TSLA", "status": "pending"},
            {"id": "s3", "title": "Summarize", "status": "pending"},
        ]
        blocked, msg = should_block_completion(steps, plan_idx=2, plan_total=3)
        self.assertTrue(blocked)


class PlanLifecycleTests(unittest.TestCase):
    """Test full plan lifecycle: parse → track → gate → complete."""

    def test_full_lifecycle(self):
        import json
        from liagent.agent.planner import _parse_plan_json, format_plan_status, should_block_completion

        # 1. Parse plan
        raw = json.dumps({
            "goal": "Research topic",
            "steps": [
                {"id": "s1", "title": "Search", "tool_hint": "web_search",
                 "done_criteria": "Find sources", "status": "pending", "evidence_ref": None},
                {"id": "s2", "title": "Fetch page", "tool_hint": "web_fetch",
                 "done_criteria": "Get content", "status": "pending", "evidence_ref": None},
                {"id": "s3", "title": "Summarize", "tool_hint": None,
                 "done_criteria": "Output summary", "status": "pending", "evidence_ref": None},
            ]
        })
        plan = _parse_plan_json(raw)
        self.assertEqual(len(plan["steps"]), 3)

        # 2. Simulate step 1 completion
        plan["steps"][0]["status"] = "done"
        plan["steps"][0]["evidence_ref"] = "Found 3 relevant articles"
        plan_idx = 1

        # 3. Check format
        status = format_plan_status(plan["goal"], plan["steps"], plan_idx)
        self.assertIn("YOU ARE HERE", status)
        self.assertIn("Fetch page", status)

        # 4. Gate should block (2 pending steps before final)
        blocked, msg = should_block_completion(plan["steps"], plan_idx, 3)
        self.assertTrue(blocked)

        # 5. Complete step 2
        plan["steps"][1]["status"] = "done"
        plan_idx = 2

        # 6. Gate should allow (only synthesis step left)
        blocked, msg = should_block_completion(plan["steps"], plan_idx, 3)
        self.assertFalse(blocked)


class ReplanParsingTests(unittest.TestCase):
    """Test TaskPlanner.replan() response parsing via mocked engine."""

    def _make_planner(self, engine_response: str):
        """Create a TaskPlanner with a mocked engine returning the given response."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from liagent.agent.planner import TaskPlanner
        engine = MagicMock()
        engine.generate_extraction = AsyncMock(return_value=engine_response)
        prompt_builder = MagicMock()
        return TaskPlanner(engine, prompt_builder)

    def _run(self, coro):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_replan_parses_json_array(self):
        planner = self._make_planner('[{"title": "New step A"}, {"title": "New step B"}]')
        frozen = [{"id": "s1", "title": "Done step", "status": "done"}]
        steps = frozen + [{"id": "s2", "title": "Old pending", "status": "pending"}]
        result = self._run(planner.replan("Goal", steps, "step failed", "tools"))
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "New step A")
        self.assertEqual(result[0]["id"], "s2")  # continues numbering after frozen
        self.assertEqual(result[0]["status"], "pending")

    def test_replan_parses_fenced_array(self):
        planner = self._make_planner('```json\n[{"title": "Fenced step"}]\n```')
        steps = [{"id": "s1", "title": "Done", "status": "done"},
                 {"id": "s2", "title": "Pending", "status": "pending"}]
        result = self._run(planner.replan("Goal", steps, "reason", "tools"))
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["title"], "Fenced step")

    def test_replan_caps_at_6_steps(self):
        raw = json.dumps([{"title": f"Step {i}"} for i in range(10)])
        planner = self._make_planner(raw)
        steps = [{"id": "s1", "title": "Done", "status": "done"},
                 {"id": "s2", "title": "Pending", "status": "pending"}]
        result = self._run(planner.replan("Goal", steps, "reason", "tools"))
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 6)

    def test_replan_returns_none_on_empty(self):
        planner = self._make_planner("")
        steps = [{"id": "s1", "title": "Done", "status": "done"},
                 {"id": "s2", "title": "Pending", "status": "pending"}]
        result = self._run(planner.replan("Goal", steps, "reason", "tools"))
        self.assertIsNone(result)

    def test_replan_returns_none_on_garbage(self):
        planner = self._make_planner("not valid json at all")
        steps = [{"id": "s1", "title": "Done", "status": "done"},
                 {"id": "s2", "title": "Pending", "status": "pending"}]
        result = self._run(planner.replan("Goal", steps, "reason", "tools"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
