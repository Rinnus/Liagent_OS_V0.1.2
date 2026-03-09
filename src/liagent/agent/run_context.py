"""RunContext — shared state for the ReAct loop, replacing 15+ local variables."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .run_control import RunCancellationScope


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
    progress_placeholder_retried: bool = False
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
    last_tool_args: dict = field(default_factory=dict)
    tool_results_history: list[dict] = field(default_factory=list)
    cancel_scope: RunCancellationScope | None = None

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

    def raise_if_cancelled(self) -> None:
        if self.cancel_scope is not None:
            self.cancel_scope.raise_if_cancelled()

    def record_tool_result(
        self,
        tool_name: str,
        observation: str,
        tool_args: dict | None = None,
        *,
        is_error: bool = False,
        system_initiated: bool = False,
    ) -> None:
        self.record_tool_event(
            requested_tool_name=tool_name,
            requested_tool_args=tool_args,
            effective_tool_name=tool_name,
            effective_tool_args=tool_args,
            observation=observation,
            status="error" if is_error else "result",
            reason="",
            system_initiated=system_initiated,
        )

    def record_tool_skip(
        self,
        tool_name: str,
        *,
        tool_args: dict | None = None,
        reason: str = "",
        system_initiated: bool = False,
    ) -> None:
        self.record_tool_event(
            requested_tool_name=tool_name,
            requested_tool_args=tool_args,
            effective_tool_name=tool_name,
            effective_tool_args=tool_args,
            observation="",
            status="skip",
            reason=reason,
            system_initiated=system_initiated,
        )

    def record_tool_event(
        self,
        *,
        requested_tool_name: str,
        requested_tool_args: dict | None = None,
        effective_tool_name: str | None = None,
        effective_tool_args: dict | None = None,
        observation: str = "",
        status: str = "result",
        reason: str = "",
        system_initiated: bool = False,
    ) -> None:
        requested_name = str(requested_tool_name or "").strip()
        effective_name = str(effective_tool_name or requested_name or "").strip()
        requested_args = dict(requested_tool_args) if isinstance(requested_tool_args, dict) else {}
        final_args = (
            dict(effective_tool_args)
            if isinstance(effective_tool_args, dict)
            else dict(requested_args)
        )
        obs = str(observation or "")
        self.last_tool_name = effective_name or requested_name
        self.last_observation = obs
        self.last_tool_args = dict(final_args)
        if obs and (effective_name or requested_name):
            self.context_vars[f"{effective_name or requested_name}_result"] = obs
        self.tool_results_history.append(
            {
                "tool_name": effective_name or requested_name,
                "requested_tool_name": requested_name,
                "requested_tool_args": requested_args,
                "effective_tool_name": effective_name or requested_name,
                "effective_tool_args": dict(final_args),
                "observation": obs,
                "tool_args": dict(final_args),
                "status": str(status or "result"),
                "reason": str(reason or ""),
                "is_error": bool(status == "error"),
                "system_initiated": bool(system_initiated),
                "recorded_at": time.time(),
            }
        )
        if len(self.tool_results_history) > 8:
            self.tool_results_history[:] = self.tool_results_history[-8:]

    def latest_tool_contexts(
        self,
        *,
        limit: int = 3,
        include_errors: bool = False,
        include_skips: bool = False,
        include_system_initiated: bool = True,
    ) -> list[dict]:
        results: list[dict] = []
        seen: set[str] = set()
        for item in reversed(self.tool_results_history):
            status = str(item.get("status") or "result")
            if status == "error" and not include_errors:
                continue
            if status == "skip" and not include_skips:
                continue
            if not include_system_initiated and item.get("system_initiated"):
                continue
            obs = str(item.get("observation") or "")
            reason = str(item.get("reason") or "")
            if status != "skip" and not obs.strip():
                continue
            if status == "skip" and not reason.strip():
                continue
            sig = json.dumps(
                {
                    "status": status,
                    "requested_tool_name": str(item.get("requested_tool_name") or item.get("tool_name") or ""),
                    "effective_tool_name": str(item.get("effective_tool_name") or item.get("tool_name") or ""),
                    "requested_tool_args": dict(item.get("requested_tool_args") or {}),
                    "effective_tool_args": dict(item.get("effective_tool_args") or item.get("tool_args") or {}),
                    "observation": obs,
                    "reason": reason,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if sig in seen:
                continue
            seen.add(sig)
            results.append(
                {
                    "tool_name": str(item.get("tool_name") or ""),
                    "requested_tool_name": str(item.get("requested_tool_name") or item.get("tool_name") or ""),
                    "requested_tool_args": dict(item.get("requested_tool_args") or item.get("tool_args") or {}),
                    "effective_tool_name": str(item.get("effective_tool_name") or item.get("tool_name") or ""),
                    "effective_tool_args": dict(item.get("effective_tool_args") or item.get("tool_args") or {}),
                    "observation": obs,
                    "tool_args": dict(item.get("tool_args") or {}),
                    "status": status,
                    "reason": reason,
                    "is_error": bool(item.get("is_error") or status == "error"),
                    "system_initiated": bool(item.get("system_initiated")),
                }
            )
            if len(results) >= max(1, int(limit or 1)):
                break
        return results

    def fallback_tool_contexts(self, *, limit: int = 4) -> list[dict]:
        primary = self.latest_tool_contexts(limit=limit, include_errors=False, include_skips=False)
        supporting = self.latest_tool_contexts(limit=limit, include_errors=True, include_skips=True)
        if not primary:
            return supporting[: max(1, int(limit or 1))]
        combined: list[dict] = list(primary)
        seen: set[str] = set()
        for item in combined:
            seen.add(
                json.dumps(
                    {
                        "requested_tool_name": item.get("requested_tool_name"),
                        "effective_tool_name": item.get("effective_tool_name"),
                        "status": item.get("status"),
                        "reason": item.get("reason"),
                        "observation": item.get("observation"),
                        "tool_args": item.get("tool_args"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        for item in supporting:
            sig = json.dumps(
                {
                    "requested_tool_name": item.get("requested_tool_name"),
                    "effective_tool_name": item.get("effective_tool_name"),
                    "status": item.get("status"),
                    "reason": item.get("reason"),
                    "observation": item.get("observation"),
                    "tool_args": item.get("tool_args"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if sig in seen:
                continue
            combined.append(item)
            seen.add(sig)
            if len(combined) >= max(1, int(limit or 1)):
                break
        return combined
