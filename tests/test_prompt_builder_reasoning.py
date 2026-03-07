import unittest


class PriorReasoningInjectionTests(unittest.TestCase):
    def test_injects_prior_reasoning_at_step_1(self):
        from liagent.agent.run_context import RunContext
        ctx = RunContext()
        ctx.reasoning_chain = [
            {"step": 0, "think": "Need to search for AAPL price",
             "tools": ["web_search"], "evidence": [
                 {"tool": "web_search", "summary": "AAPL at $185"}
             ]},
        ]
        messages = [{"role": "system", "content": "You are an agent."}]
        from liagent.agent.prompt_builder import inject_prior_reasoning
        result = inject_prior_reasoning(messages, ctx, step=1)
        self.assertTrue(any("Prior Reasoning" in m.get("content", "") for m in result))
        self.assertIn("AAPL", str(result))

    def test_no_injection_at_step_0(self):
        from liagent.agent.run_context import RunContext
        ctx = RunContext()
        messages = [{"role": "system", "content": "test"}]
        from liagent.agent.prompt_builder import inject_prior_reasoning
        result = inject_prior_reasoning(messages, ctx, step=0)
        self.assertFalse(any("Prior Reasoning" in m.get("content", "") for m in result))

    def test_window_limited_to_3_steps(self):
        from liagent.agent.run_context import RunContext
        ctx = RunContext()
        for i in range(5):
            ctx.reasoning_chain.append({
                "step": i, "think": f"Step {i} thinking",
                "tools": [], "evidence": [],
            })
        messages = [{"role": "system", "content": "test"}]
        from liagent.agent.prompt_builder import inject_prior_reasoning
        result = inject_prior_reasoning(messages, ctx, step=5)
        injected = [m for m in result if "Prior Reasoning" in m.get("content", "")]
        self.assertEqual(len(injected), 1)
        content = injected[0]["content"]
        self.assertNotIn("[Step 0]", content)
        self.assertIn("[Step 4]", content)


if __name__ == "__main__":
    unittest.main()
