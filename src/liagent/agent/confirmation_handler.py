"""Confirmation handler — tool confirmation flow with multi-stage support."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, TypeAlias

from ..tools import get_tool
from ..tools.policy import ToolPolicy
from ..logging import get_logger
from .quality import estimate_task_success, detect_hallucinated_action
from .self_supervision import InteractionMetrics
from .tool_executor import ToolExecutor, build_tool_degrade_observation
from .tool_exchange import append_tool_exchange

_log = get_logger("confirmation")

# Regex patterns for confirmation commands
_CONFIRM_RE = re.compile(r"^/confirm\s+([A-Za-z0-9_-]+)(?:\s+(--force|force))?\s*$")
_REJECT_RE = re.compile(r"^/reject\s+([A-Za-z0-9_-]+)\s*$")


def parse_confirmation_command(user_input: str) -> tuple[str, str, bool] | None:
    """Parse a /confirm or /reject command from user input.

    Returns (action, token, force) or None if not a confirmation command.
    """
    text = user_input.strip()
    m = _CONFIRM_RE.match(text)
    if m:
        force = bool(m.group(2))
        return ("confirm", m.group(1), force)
    m = _REJECT_RE.match(text)
    if m:
        return ("reject", m.group(1), False)
    return None


def cleanup_pending_confirmations(
    pending: dict[str, dict],
    ttl: timedelta,
) -> None:
    """Remove expired confirmation tokens in-place."""
    now = datetime.now(timezone.utc)
    stale = [
        token
        for token, payload in pending.items()
        if now - payload.get("created_at", now) > ttl
    ]
    for token in stale:
        pending.pop(token, None)


async def resolve_confirmation(
    token: str,
    approved: bool,
    force: bool,
    *,
    pending_confirmations: dict[str, dict],
    confirm_ttl: timedelta,
    tool_policy: ToolPolicy,
    tool_executor: ToolExecutor,
    memory: Any,
    final_answer_fn: Any,
) -> dict:
    """Process a confirmation or rejection for a pending tool call.

    Args:
        token: Confirmation token string.
        approved: True for /confirm, False for /reject.
        force: True when --force flag is present.
        pending_confirmations: Shared dict of pending tokens.
        confirm_ttl: TTL for confirmation expiry.
        tool_policy: ToolPolicy instance for auditing.
        tool_executor: ToolExecutor for running the tool.
        memory: ConversationMemory instance.
        final_answer_fn: Async callable that generates a final answer from memory.
            Signature: ``async () -> tuple[str, dict]``

    Returns:
        Result dict with 'status' key and payload.
    """
    cleanup_pending_confirmations(pending_confirmations, confirm_ttl)
    payload = pending_confirmations.get(token)
    if not payload:
        return {"status": "error", "message": "Confirmation token does not exist or has expired."}

    tool_name = payload["tool_name"]
    tool_args = payload["tool_args"]
    user_input = payload["user_input"]
    tool_def = get_tool(tool_name)
    if tool_def is None:
        pending_confirmations.pop(token, None)
        tool_policy.audit(tool_name, tool_args, "blocked", "unknown tool on confirm",
                          policy_decision="blocked")
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    if not approved:
        pending_confirmations.pop(token, None)
        tool_policy.audit(tool_name, tool_args, "rejected", "user rejected confirmation",
                          policy_decision="rejected")
        return {"status": "rejected", "message": f"Tool call rejected: {tool_name}"}

    required_stage = int(payload.get("required_stage", 1))
    current_stage = int(payload.get("stage", 1))
    if required_stage > 1 and current_stage < required_stage:
        if force and current_stage == 1:
            return {
                "status": "error",
                "message": "You must complete first-step confirmation before `--force` confirmation.",
            }
        if not force:
            payload["stage"] = current_stage + 1
            pending_confirmations[token] = payload
            brief = tool_policy.confirmation_brief(
                tool_def,
                tool_args,
                payload.get("pending_reason", "high-risk confirmation"),
                stage=payload["stage"],
                required_stage=required_stage,
            )
            tool_policy.audit(
                tool_name,
                tool_args,
                "pending",
                f"waiting second confirmation stage={payload['stage']}/{required_stage}",
                policy_decision="pending_confirmation",
            )
            return {
                "status": "need_second_confirm",
                "token": token,
                "message": f"Second confirmation is required. Run `/confirm {token} --force`.",
                "brief": brief,
            }

    pending_confirmations.pop(token, None)

    observation, is_err, err_type = await tool_executor.execute(tool_def, tool_args)
    # Determine grant_source for audit
    trust_server_id = payload.get("trust_server_id")
    _grant_src = "trust_first_use" if trust_server_id else "one_time"
    if is_err:
        observation = build_tool_degrade_observation(tool_name, tool_args, observation)
        tool_policy.audit(
            tool_name, tool_args, "error", f"confirmed but failed ({err_type})",
            policy_decision="confirmed", grant_source=_grant_src,
        )
    else:
        tool_policy.audit(tool_name, tool_args, "ok", "confirmed and executed",
                          policy_decision="confirmed", grant_source=_grant_src)
        # Auto-approve server trust on first-use confirmation
        if trust_server_id and hasattr(tool_policy, "trust_registry") and tool_policy.trust_registry:
            tool_policy.trust_registry.set_status(trust_server_id, "approved", source="first_use")
    append_tool_exchange(
        memory,
        assistant_content=payload.get("assistant_tool_call", ""),
        tool_name=tool_name,
        tool_args=tool_args,
        observation=observation,
        hint=f"This tool call was confirmed and executed by the user. Original request: {user_input}\n"
             "Do not call tools again. Answer directly from available information.",
    )
    answer, qmeta = await final_answer_fn()
    result = {
        "status": "ok",
        "tool_name": tool_name,
        "tool_args": tool_args,
        "observation": observation,
        "answer": answer,
        "quality": qmeta,
        "execution_ok": not is_err,
    }
    if "shell_grant_key" in payload:
        result["shell_grant_key"] = payload["shell_grant_key"]
    return result


def build_confirmation_run_events(
    result: dict,
    *,
    start_ts: float,
    session_id: str,
    metrics: InteractionMetrics,
) -> list[tuple]:
    """Build legacy events from a resolve_confirmation result.

    Call this from run() after resolve_confirmation returns to yield
    the appropriate events. Returns a list of LegacyEvent tuples.
    """
    status = result.get("status")
    events: list[tuple] = []

    if status == "error":
        events.append(("error", result.get("message", "Confirmation failed")))
        return events

    if status == "rejected":
        metrics.log_turn(
            session_id=session_id,
            latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            tool_calls=0, tool_errors=0, policy_blocked=0,
            task_success=True, answer_revision_count=0,
            quality_issues="", plan_completion_ratio=1.0,
            answer_chars=len(result.get("message", "")),
        )
        events.append(("done", result.get("message", "Rejected")))
        return events

    if status == "need_second_confirm":
        brief = result.get("brief", {}) if isinstance(result, dict) else {}
        events.append((
            "confirmation_required",
            result.get("token", ""),
            brief.get("tool", ""),
            result.get("message", ""),
            json.dumps(brief, ensure_ascii=False),
        ))
        events.append(("done", result.get("message", "Second confirmation required")))
        return events

    if status == "ok":
        q = result.get("quality", {}) or {}
        issues = q.get("issues", []) if isinstance(q, dict) else []
        task_success, task_reason = estimate_task_success(
            answer=result.get("answer", ""),
            tool_calls=1, tool_errors=0, policy_blocked=0,
            plan_total_steps=0, plan_completed_steps=0,
            detect_hallucinated_action_fn=detect_hallucinated_action,
        )
        metrics.log_turn(
            session_id=session_id,
            latency_ms=(time.perf_counter() - start_ts) * 1000.0,
            tool_calls=1, tool_errors=0, policy_blocked=0,
            task_success=task_success, answer_revision_count=len(issues),
            quality_issues=",".join(issues), plan_completion_ratio=1.0,
            answer_chars=len(result.get("answer", "")),
        )
        events.append((
            "task_outcome",
            json.dumps({"success": task_success, "reason": task_reason}, ensure_ascii=False),
        ))
        events.append(("tool_start", result["tool_name"], result["tool_args"]))
        events.append(("tool_result", result["tool_name"], result["observation"]))
        events.append(("done", result["answer"]))
        return events

    # Unexpected status — should not happen
    events.append(("error", f"Unexpected confirmation status: {status}"))
    return events
