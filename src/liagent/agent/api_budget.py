"""API budget tracking — token counting, context trimming, and usage collection."""

from __future__ import annotations

import json
import re
from typing import Any

from ..logging import get_logger

_EVIDENCE_RE = re.compile(r"\[\[evidence:(\w+)\]\]")

_log = get_logger("api_budget")


class ApiBudgetTracker:
    """Mixin-style helper for API token budget management.

    Designed to be composed into AgentBrain (not subclassed).
    All methods operate on explicit state passed via parameters so that
    this module has ZERO imports from brain.py (no circular dependency).
    """

    def __init__(
        self,
        *,
        api_context_char_budget: int,
        api_input_token_budget: int,
        api_turn_token_budget: int,
        api_budget_reserve_tokens: int,
    ):
        self.api_context_char_budget = api_context_char_budget
        self.api_input_token_budget = api_input_token_budget
        self.api_turn_token_budget = api_turn_token_budget
        self.api_budget_reserve_tokens = api_budget_reserve_tokens
        self._api_budget_active = False
        self._api_turn_tokens_used = 0

    # ── Token estimation helpers ──────────────────────────────────────

    @staticmethod
    def message_char_cost(msg: dict) -> int:
        """Estimate character payload size for one chat message."""
        n = len(msg.get("content", "") or "")
        for tc in msg.get("tool_calls", []):
            n += len(str(tc))
        return n

    @staticmethod
    def estimate_tokens_from_chars(chars: int) -> int:
        return max(1, (max(0, int(chars)) + 3) // 4)

    def estimate_message_tokens(
        self, messages: list[dict], *, tools: list[dict] | None = None
    ) -> int:
        total_chars = sum(self.message_char_cost(m) for m in messages)
        if tools:
            try:
                total_chars += len(json.dumps(tools, ensure_ascii=False))
            except Exception:
                total_chars += len(str(tools))
        return self.estimate_tokens_from_chars(total_chars)

    # ── Budget accounting ─────────────────────────────────────────────

    def budget_remaining_tokens(self) -> int:
        if not self._api_budget_active:
            return 10**9
        return max(0, self.api_turn_token_budget - self._api_turn_tokens_used)

    def budget_consume(
        self,
        *,
        messages: list[dict],
        response_text: str,
        tools: list[dict] | None = None,
        engine: Any = None,
    ) -> int:
        """Record token consumption for one LLM call.

        *engine* is used to query real usage when available (via
        ``engine.get_last_llm_usage()``).  Falls back to character-based
        estimation.
        """
        if not self._api_budget_active:
            return 0
        usage: dict = {}
        if engine is not None:
            getter = getattr(engine, "get_last_llm_usage", None)
            if callable(getter):
                try:
                    usage = getter() or {}
                except Exception:
                    usage = {}
        total = int(usage.get("total_tokens", 0) or 0)
        if total <= 0:
            total = self.estimate_message_tokens(messages, tools=tools)
            total += self.estimate_tokens_from_chars(len(response_text or ""))
        self._api_turn_tokens_used += max(0, total)
        return max(0, total)

    def collect_turn_llm_usage(
        self, *, engine: Any = None
    ) -> dict[str, int | str | bool | float]:
        """Aggregate LLM usage stats for the current turn."""
        usage: dict[str, int | str | bool | float] = {}
        if engine is not None:
            getter = getattr(engine, "get_cumulative_llm_usage", None)
            if callable(getter):
                try:
                    usage = getter() or {}
                except Exception:
                    usage = {}
        total = int(usage.get("total_tokens", 0) or 0)
        if total <= 0:
            api_mode = bool(self._api_budget_active)
            turn_used = int(self._api_turn_tokens_used or 0)
            if not api_mode or turn_used <= 0:
                return {}
            llm_cfg = getattr(getattr(engine, "config", None), "llm", None) if engine else None
            usage = {
                "provider": str(getattr(llm_cfg, "api_model", "") or ""),
                "model": str(getattr(llm_cfg, "api_model", "") or ""),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": turn_used,
                "cached_prompt_tokens": 0,
                "cache_write_tokens": 0,
                "cache_hit_ratio": 0.0,
                "estimated_cost_usd": 0.0,
                "estimated": True,
            }
        out: dict[str, int | str | bool | float] = {
            "provider": str(usage.get("provider", "") or ""),
            "model": str(usage.get("model", "") or ""),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "cached_prompt_tokens": int(usage.get("cached_prompt_tokens", 0) or 0),
            "cache_write_tokens": int(usage.get("cache_write_tokens", 0) or 0),
            "cache_hit_ratio": round(float(usage.get("cache_hit_ratio", 0.0) or 0.0), 4),
            "estimated_cost_usd": round(float(usage.get("estimated_cost_usd", 0.0) or 0.0), 8),
            "estimated": bool(usage.get("estimated", False)),
        }
        api_mode = bool(self._api_budget_active)
        if api_mode:
            turn_budget = max(0, self.api_turn_token_budget)
            turn_used = max(0, self._api_turn_tokens_used)
            out["turn_budget"] = turn_budget
            out["turn_used"] = turn_used
            out["turn_remaining"] = max(0, turn_budget - turn_used)
        return out

    # ── Context trimming ──────────────────────────────────────────────

    def trim_messages_for_api(
        self, messages: list[dict], *, budget_chars: int | None = None,
        pinned_step_ids: set[str] | None = None,
        dropped_evidence_out: list[dict] | None = None,
    ) -> list[dict]:
        """Trim API-bound chat history to reduce per-call token overhead.

        Strategy:
        - Keep system message.
        - Keep most recent user/tool/assistant messages.
        - Keep latest execution-control user message if present.
        - Fill remaining budget from newest to oldest.
        """
        if not messages:
            return messages

        raw_budget = int(
            budget_chars if budget_chars is not None else self.api_context_char_budget
        )
        budget = max(80, raw_budget)
        costs = [self.message_char_cost(m) for m in messages]
        total_chars = sum(costs)
        if total_chars <= budget:
            return messages

        keep_idx: set[int] = set()
        if messages[0].get("role") == "system":
            keep_idx.add(0)

        for role in ("user", "tool", "assistant"):
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == role:
                    keep_idx.add(i)
                    break

        control_markers = (
            "[Execution Progress]",
            "All required data has been collected.",
            "Execution constraint:",
        )
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "user":
                continue
            content = messages[i].get("content", "") or ""
            if any(marker in content for marker in control_markers):
                keep_idx.add(i)
                break

        # Evidence pinning: keep messages with markers for done plan steps
        if pinned_step_ids:
            for i, msg in enumerate(messages):
                content = msg.get("content") or ""
                m = _EVIDENCE_RE.search(content)
                if m and m.group(1) in pinned_step_ids:
                    keep_idx.add(i)

        used = sum(costs[i] for i in keep_idx)
        for i in range(len(messages) - 1, -1, -1):
            if i in keep_idx:
                continue
            c = costs[i]
            if used + c > budget:
                continue
            keep_idx.add(i)
            used += c

        # Track dropped evidence messages
        if dropped_evidence_out is not None:
            for i, msg in enumerate(messages):
                if i not in keep_idx and _EVIDENCE_RE.search(msg.get("content") or ""):
                    dropped_evidence_out.append(msg)

        trimmed = [messages[i] for i in sorted(keep_idx)]
        _log.trace(
            "api_context_trim",
            before_messages=len(messages),
            after_messages=len(trimmed),
            before_chars=total_chars,
            after_chars=sum(self.message_char_cost(m) for m in trimmed),
            budget=budget,
        )
        return trimmed

    # ── Turn lifecycle ────────────────────────────────────────────────

    def reset_turn(self, *, active: bool) -> None:
        """Reset per-turn counters (call at start of each run)."""
        self._api_budget_active = active
        self._api_turn_tokens_used = 0
