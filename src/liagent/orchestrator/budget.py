"""Global and per-agent budget management."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetSlice:
    max_steps: int
    max_tool_calls: int
    timeout_ms: int
    allowed_tools: set[str] | None = None  # None = unrestricted
    max_tokens: int = 1024


@dataclass
class GlobalBudget:
    max_steps: int
    max_tool_calls: int
    timeout_ms: int
    max_workers: int = 3

    def slice(self, count: int, reserve_for_lead: int = 3) -> list[BudgetSlice]:
        """Divide budget into N worker slices, reserving steps for Lead synthesis."""
        worker_steps = max(1, (self.max_steps - reserve_for_lead) // max(1, count))
        worker_tools = max(1, self.max_tool_calls // max(1, count))
        worker_timeout = max(5000, self.timeout_ms // max(1, count + 1))
        return [
            BudgetSlice(
                max_steps=worker_steps,
                max_tool_calls=worker_tools,
                timeout_ms=worker_timeout,
            )
            for _ in range(count)
        ]

    def lead_budget(self, reserve_steps: int = 3, reserve_tokens: int = 2048) -> BudgetSlice:
        """Budget slice for Lead Agent synthesis phase."""
        return BudgetSlice(
            max_steps=reserve_steps,
            max_tool_calls=0,
            timeout_ms=max(5000, self.timeout_ms // 3),
            max_tokens=reserve_tokens,
        )
