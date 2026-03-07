"""RunContext — shared state for the ReAct loop, replacing 15+ local variables."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunContext:
    # Counters
    tool_calls: int = 0
    tool_errors: int = 0
    policy_blocked: int = 0
    revision_count: int = 0
    quality_issues: list[str] = field(default_factory=list)
    tools_used: set[str] = field(default_factory=set)

    # Retry flags
    copout_retried: bool = False
    hallucination_retried: bool = False
    unsourced_data_retried: bool = False
    ungrounded_retried: bool = False
    unwritten_code_retried: bool = False

    # Tool dedup / cache
    tool_sig_count: dict[str, int] = field(default_factory=dict)
    tool_artifacts: dict[str, str] = field(default_factory=dict)
    context_vars: dict[str, str] = field(default_factory=dict)

    # Plan execution state
    plan_goal: str = ""
    plan_steps: list[dict] = field(default_factory=list)
    plan_total_steps: int = 0
    plan_idx: int = 0
    plan_step_failures: dict[str, int] = field(default_factory=dict)  # step_id → failure count
    plan_replan_count: int = 0  # re-plan attempts used (max 2)

    # Loop state
    last_tool_name: str = ""
    last_observation: str = ""

    # Budget / config (set at startup, read-only in loop)
    budget: object = None  # RuntimeBudget
    user_input: str = ""
    low_latency: bool = False
    experience_match: object = None
    experience_constraint: str = ""
    skill_allowed_tools: set | None = None
    budget_allowed_tools: set | None = None  # Hard gate from BudgetOverride
    active_skill_name: str = ""

    # System prompts (built before loop)
    system_prompt_vlm: str = ""
    system_prompt_coder: str = ""
    native_tool_schemas: list = field(default_factory=list)
    user_images: list | None = None
    vision_hold_steps: int = 2
    max_steps: int = 8
    max_tool_calls: int = 3

    # Search orchestration
    source_urls: list[tuple[str, str]] = field(default_factory=list)  # (title, url)
    auto_fetch_enabled: bool = False

    # Think / reasoning
    show_thinking: bool = False       # UI: whether to show <think> content to user
    enable_thinking: bool = True      # Model: whether to generate <think> reasoning chain
    reasoning_content: str = ""       # Current step's <think> content (cleared each step)
    reasoning_chain: list[dict] = field(default_factory=list)
    execution_origin: str = "user"      # "user" | "system" | "goal"
    goal_id: int | None = None

    # Proactive intelligence
    behavior_signal_store: object = None
    session_id: str = ""
    repl_session_id: str = ""
    conversation_id: str = ""

    # Long-term memory (for zero-LLM fact extraction in tool_orchestrator)
    long_term_memory: object = None

    # E3: execution telemetry
    tool_fallback_count: int = 0
    tool_timeout_count: int = 0
    failure_counts: dict[str, int] = field(default_factory=dict)

    # Global retry budget
    global_retry_budget: int = 4
    global_retries_used: int = 0
    retry_ledger: list[str] = field(default_factory=list)

    # Timing
    start_ts: float = 0.0

    @property
    def retry_budget_exhausted(self) -> bool:
        return self.global_retries_used >= self.global_retry_budget

    def consume_retry(self, reason: str) -> bool:
        if self.retry_budget_exhausted:
            return False
        self.global_retries_used += 1
        self.retry_ledger.append(reason)
        return True
