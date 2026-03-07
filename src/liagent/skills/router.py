"""Skill router — budget configuration for the LLM-driven pipeline.

The LLM decides what to do within the budget. No keyword-based intent
classification — just runtime context (voice vs. text, images vs. not).

Exports:
    SkillConfig       — immutable skill template
    RuntimeBudget     — mutable runtime budget (replaces ServiceBudget)
    BudgetOverride    — optional hard constraints from external callers
    select_skill()    — pick config from runtime context
    build_runtime_budget() — create mutable budget from config
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillConfig:
    """Immutable skill template — defines the budget envelope."""
    name: str
    tier: str                          # "realtime_voice" | "standard_chat" | "deep_task"
    max_steps: int
    max_tool_calls: int
    max_tokens: int                    # LLM max output tokens
    llm_temperature: float
    enable_planning: bool
    enable_policy_review: bool
    allowed_tools: set[str] | None = None  # None = unrestricted


@dataclass
class RuntimeBudget:
    """Mutable runtime budget — preserves full ServiceBudget field contract.

    Downstream consumers:
    - policy_gate.py:223    → .enable_policy_review
    - tool_orchestrator.py:436 → .llm_max_tokens, .llm_temperature
    - engine_manager.py:644 → .tier == "deep_task"
    - brain.py              → mutates .max_steps, .max_tool_calls after planning
    """
    tier: str
    llm_max_tokens: int
    llm_temperature: float
    max_steps: int
    max_tool_calls: int
    enable_planning: bool
    enable_policy_review: bool

    def apply_override(self, override: BudgetOverride) -> None:
        """Clamp budget fields to override limits."""
        if override.max_steps is not None:
            self.max_steps = min(self.max_steps, override.max_steps)
        if override.max_tool_calls is not None:
            self.max_tool_calls = min(self.max_tool_calls, override.max_tool_calls)


@dataclass
class BudgetOverride:
    """Optional hard constraints from external callers."""
    max_steps: int | None = None
    max_tool_calls: int | None = None
    timeout_ms: int | None = None
    allowed_tools: set[str] | None = None


# ── Predefined configs ────────────────────────────────────────────

REALTIME_VOICE = SkillConfig(
    name="realtime_voice",
    tier="realtime_voice",
    max_steps=5, max_tool_calls=2, max_tokens=640,
    llm_temperature=0.25,
    enable_planning=False, enable_policy_review=False,
    allowed_tools={"web_search", "web_fetch", "create_task", "delete_task",
                   "delete_all_tasks", "list_tasks"},
)

REALTIME_VISION = SkillConfig(
    name="realtime_vision",
    tier="realtime_voice",          # same tier as voice — low latency
    max_steps=5, max_tool_calls=2, max_tokens=800,
    llm_temperature=0.25,
    enable_planning=False, enable_policy_review=False,
    allowed_tools={"web_search", "web_fetch", "describe_image", "create_task",
                   "delete_task", "delete_all_tasks", "list_tasks"},
)

STANDARD_CHAT = SkillConfig(
    name="standard_chat",
    tier="standard_chat",
    max_steps=10, max_tool_calls=8, max_tokens=2048,
    llm_temperature=0.35,
    enable_planning=True, enable_policy_review=False,
    allowed_tools=None,
)


def select_skill(
    user_input: str,
    *,
    low_latency: bool,
    has_images: bool,
) -> SkillConfig:
    """Select budget config from runtime context. No keyword detection."""
    if low_latency:
        return REALTIME_VISION if has_images else REALTIME_VOICE
    return STANDARD_CHAT


def build_runtime_budget(config: SkillConfig) -> RuntimeBudget:
    """Create a mutable runtime budget from an immutable skill config."""
    return RuntimeBudget(
        tier=config.tier,
        llm_max_tokens=config.max_tokens,
        llm_temperature=config.llm_temperature,
        max_steps=config.max_steps,
        max_tool_calls=config.max_tool_calls,
        enable_planning=config.enable_planning,
        enable_policy_review=config.enable_policy_review,
    )
