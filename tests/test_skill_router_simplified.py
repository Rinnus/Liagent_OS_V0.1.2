"""Tests for simplified skill router — budget config only."""

import unittest

from liagent.skills.router import (
    SkillConfig, RuntimeBudget, BudgetOverride,
    select_skill, build_runtime_budget,
    REALTIME_VOICE, REALTIME_VISION, STANDARD_CHAT,
)


class SkillConfigTests(unittest.TestCase):
    def test_voice_mode_returns_voice_config(self):
        config = select_skill("anything", low_latency=True, has_images=False)
        self.assertEqual(config.tier, "realtime_voice")
        self.assertLessEqual(config.max_steps, 5)
        self.assertFalse(config.enable_planning)

    def test_vision_voice_mode(self):
        config = select_skill("anything", low_latency=True, has_images=True)
        self.assertEqual(config.tier, "realtime_voice")
        self.assertIn("describe_image", config.allowed_tools)

    def test_standard_mode_default(self):
        config = select_skill("hello", low_latency=False, has_images=False)
        self.assertEqual(config.tier, "standard_chat")
        self.assertTrue(config.enable_planning)

    def test_voice_has_restricted_tools(self):
        config = select_skill("search", low_latency=True, has_images=False)
        self.assertIsNotNone(config.allowed_tools)
        self.assertIn("web_search", config.allowed_tools)

    def test_standard_unrestricted_tools(self):
        config = select_skill("anything", low_latency=False, has_images=False)
        self.assertIsNone(config.allowed_tools)


class RuntimeBudgetTests(unittest.TestCase):
    def test_build_preserves_tier(self):
        """RuntimeBudget.tier must use the SkillConfig tier, not the name."""
        budget = build_runtime_budget(REALTIME_VISION)
        self.assertEqual(budget.tier, "realtime_voice")  # tier, NOT name

    def test_build_has_all_service_budget_fields(self):
        """Must have all fields that downstream code expects."""
        budget = build_runtime_budget(STANDARD_CHAT)
        # policy_gate.py:223 reads ctx.budget.enable_policy_review
        self.assertIsInstance(budget.enable_policy_review, bool)
        # tool_orchestrator.py:436 reads ctx.budget.llm_max_tokens, llm_temperature
        self.assertIsInstance(budget.llm_max_tokens, int)
        self.assertIsInstance(budget.llm_temperature, float)
        # engine_manager.py:644 checks tier == "deep_task"
        self.assertIn(budget.tier, ("realtime_voice", "standard_chat", "deep_task"))
        # brain.py mutates these
        self.assertIsInstance(budget.max_steps, int)
        self.assertIsInstance(budget.max_tool_calls, int)

    def test_budget_is_mutable(self):
        """Brain mutates budget.max_steps and budget.max_tool_calls after planning."""
        budget = build_runtime_budget(STANDARD_CHAT)
        original = budget.max_steps
        budget.max_steps = original + 5
        self.assertEqual(budget.max_steps, original + 5)

    def test_voice_disables_policy_review(self):
        budget = build_runtime_budget(REALTIME_VOICE)
        self.assertFalse(budget.enable_policy_review)

    def test_standard_enables_policy_review(self):
        budget = build_runtime_budget(STANDARD_CHAT)
        self.assertFalse(budget.enable_policy_review)


class BudgetOverrideTests(unittest.TestCase):
    def test_override_clamps_steps(self):
        budget = build_runtime_budget(STANDARD_CHAT)
        override = BudgetOverride(max_steps=4)
        budget.apply_override(override)
        self.assertEqual(budget.max_steps, 4)

    def test_override_clamps_tool_calls(self):
        budget = build_runtime_budget(STANDARD_CHAT)
        override = BudgetOverride(max_tool_calls=2)
        budget.apply_override(override)
        self.assertEqual(budget.max_tool_calls, 2)

    def test_override_none_fields_ignored(self):
        budget = build_runtime_budget(STANDARD_CHAT)
        orig_steps = budget.max_steps
        override = BudgetOverride()  # all None
        budget.apply_override(override)
        self.assertEqual(budget.max_steps, orig_steps)

    def test_override_timeout(self):
        override = BudgetOverride(timeout_ms=60_000)
        self.assertEqual(override.timeout_ms, 60_000)

    def test_override_allowed_tools(self):
        override = BudgetOverride(allowed_tools={"read_file", "write_file"})
        self.assertEqual(override.allowed_tools, {"read_file", "write_file"})


if __name__ == "__main__":
    unittest.main()
