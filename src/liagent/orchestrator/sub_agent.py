"""SubAgent context and result types for multi-agent research."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .budget import BudgetSlice


@dataclass
class SubAgentContext:
    """Execution context passed to each SubAgent worker."""
    trace_id: str
    objective: str
    budget: BudgetSlice
    allowed_tools: set[str] | None = None
    strategy: str = ""  # e.g. "web_search", "stock_lookup", "code_exec"
    agent_id: str = ""  # e.g. "sub:0", "sub:1"
    parent_query: str = ""


@dataclass
class SubAgentResult:
    """Result returned by a SubAgent after execution."""
    agent_id: str
    strategy: str
    summary: str = ""
    data_points: list[dict[str, Any]] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    raw_observations: list[str] = field(default_factory=list)
    llm_calls: int = 0
    tool_calls: int = 0
    success: bool = True
    error: str = ""
    elapsed_ms: float = 0.0
