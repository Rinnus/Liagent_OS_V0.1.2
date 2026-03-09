"""Policy gate — tool access control checks extracted from brain.py run()."""

from __future__ import annotations

import json
import shlex
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..logging import get_logger
from ..tools import get_tool
from .run_context import RunContext
from .tool_parsing import extract_tool_call_block, tool_call_signature

_log = get_logger("policy_gate")


@dataclass
class PolicyDecision:
    allowed: bool
    events: list[tuple] = field(default_factory=list)
    blocked_reason: str = ""
    failure_kind: str = ""  # FailureKind value for recovery routing
    confirmed: bool = False
    must_return: bool = False
    return_events: list[tuple] = field(default_factory=list)
    auth_mode: str = ""  # "allowed" | "granted" | "confirmed" — propagated to audit
    grant_source: str = ""  # "session_grant" | "" — propagated to audit


def _shell_tier_gate(
    tool_args: dict,
    tool_def,
    tool_grants: dict[str, float] | None,
    already_granted: bool,
    ctx,
    pending_confirmations: dict,
    clean_resp: str,
) -> PolicyDecision | None:
    """Check shell_exec command tier and gate dev/privileged commands.

    Returns None if the command should proceed (safe tier or already granted),
    or a PolicyDecision blocking execution for dev/privileged without grant.
    """
    from ..tools.shell_classify import classify_command, grant_key, grant_scope_label

    command_str = tool_args.get("command", "")
    try:
        argv = shlex.split(command_str)
    except ValueError:
        return None  # let shell_exec handle parse errors

    if not argv:
        return None

    tier, tier_reason = classify_command(argv)

    if tier in ("safe", "denied"):
        return None  # safe: proceed; denied: shell_exec handles it

    # Dev tier: check shell-specific grant
    if tier == "dev":
        gk = grant_key(argv)
        if already_granted:
            return None
        if tool_grants and gk in tool_grants:
            if time.time() < tool_grants[gk]:
                return None  # valid grant
            else:
                tool_grants.pop(gk, None)  # expired

    # Privileged tier: always require confirmation (2-step)
    required_stage = 2 if tier == "privileged" else 1
    gk = grant_key(argv) if tier == "dev" else f"shell_exec:priv:{argv[0]}"
    grant_scope = grant_scope_label(argv)

    token = uuid.uuid4().hex[:10]
    cmd_str = " ".join(argv)
    pending_confirmations[token] = {
        "tool_name": "shell_exec",
        "tool_args": tool_args,
        "created_at": datetime.now(timezone.utc),
        "user_input": ctx.user_input,
        "assistant_tool_call": clean_resp,
        "required_stage": required_stage,
        "stage": 1,
        "pending_reason": f"shell_exec {tier} command: {tier_reason}",
        "shell_grant_key": gk,
    }

    if tier == "dev":
        done_msg = (
            f"Command `{cmd_str}` is **dev-tier** ({tier_reason}) and requires a session grant. "
            f"Use `/confirm {token}` to approve. Subsequent `{grant_scope}` commands will auto-execute."
        )
    else:
        done_msg = (
            f"Command `{cmd_str}` is **privileged** ({tier_reason}) and requires two-step confirmation. "
            f"Use `/confirm {token}`, then `/confirm {token} --force`."
        )

    _log.trace("policy_gate", tool="shell_exec", allowed=False,
               reason=f"shell_{tier}", command=cmd_str)
    return PolicyDecision(
        allowed=False,
        events=[("confirmation_required", token, "shell_exec",
                 f"shell {tier}: {tier_reason}",
                 json.dumps({"message": f"Shell command `{cmd_str}` classified as {tier} tier."}))],
        must_return=True,
        return_events=[("done", done_msg)],
    )


async def evaluate_tool_policy(
    *,
    tool_name: str,
    tool_args: dict,
    tool_sig: str,
    full_response: str,
    confirmed: bool,
    ctx: RunContext,
    tool_policy,
    planner,
    handle_policy_block_fn: Callable,
    pending_confirmations: dict,
    dup_tool_limit: int,
    tool_cache_enabled: bool,
    enable_policy_review: bool,
    disable_policy_review_in_voice: bool,
    tool_grants: dict[str, float] | None = None,
) -> PolicyDecision:
    """Evaluate all policy checks for a tool call. Returns a PolicyDecision."""

    # 1. Skill whitelist
    if ctx.skill_allowed_tools is not None and tool_name not in ctx.skill_allowed_tools:
        obs, _, events = handle_policy_block_fn(
            tool_name, tool_args,
            f"tool not allowed by skill={ctx.active_skill_name}",
            full_response, "The active skill forbids this tool call. Answer using available information.",
        )
        _log.trace("policy_gate", tool=tool_name, allowed=False, reason="skill_whitelist")
        return PolicyDecision(
            allowed=False,
            events=events,
            blocked_reason=f"tool not allowed by skill={ctx.active_skill_name}",
            failure_kind="policy_allowlist",
        )

    # 1b. Budget-level tool allowlist (hard gate from Orchestrator)
    if ctx.budget_allowed_tools is not None and tool_name not in ctx.budget_allowed_tools:
        obs, _, events = handle_policy_block_fn(
            tool_name, tool_args,
            f"tool '{tool_name}' not in budget allowlist",
            full_response, "Tool is not in the budget allowlist. Use an allowed tool or answer directly.",
        )
        _log.trace("policy_gate", tool=tool_name, allowed=False, reason="budget_allowlist")
        return PolicyDecision(
            allowed=False,
            events=events,
            blocked_reason=f"tool '{tool_name}' not in budget allowlist",
            failure_kind="policy_allowlist",
        )

    # 2. Budget limit
    if ctx.tool_calls >= ctx.max_tool_calls:
        obs, _, events = handle_policy_block_fn(
            tool_name, tool_args,
            f"tool budget exceeded: {ctx.tool_calls}>{ctx.max_tool_calls}",
            full_response, "Tool call budget exceeded. Do not call more tools; summarize and answer directly.",
        )
        _log.trace("policy_gate", tool=tool_name, allowed=False, reason="budget_exceeded")
        return PolicyDecision(
            allowed=False,
            events=events,
            blocked_reason=f"tool budget exceeded: {ctx.tool_calls}>{ctx.max_tool_calls}",
            failure_kind="policy_budget",
        )

    # 3. Dedup
    ctx.tool_sig_count[tool_sig] = ctx.tool_sig_count.get(tool_sig, 0) + 1
    if ctx.tool_sig_count[tool_sig] > dup_tool_limit:
        obs, _, events = handle_policy_block_fn(
            tool_name, tool_args,
            f"repeated tool call loop: {tool_name} x{ctx.tool_sig_count[tool_sig]}",
            full_response, "Repeated tool-call loop detected. Provide a direct summary answer.",
        )
        _log.trace("policy_gate", tool=tool_name, allowed=False, reason="dup_tool_loop")
        return PolicyDecision(
            allowed=False,
            events=events,
            blocked_reason=f"repeated tool call loop: {tool_name} x{ctx.tool_sig_count[tool_sig]}",
            failure_kind="policy_dedup",
        )

    # 4. Tool lookup
    tool_def = get_tool(tool_name)
    clean_resp = extract_tool_call_block(full_response) or full_response
    if tool_def is None:
        obs, _, events = handle_policy_block_fn(
            tool_name, tool_args,
            f"unknown tool: {tool_name}",
            full_response, "Use a registered tool or answer directly.",
        )
        _log.trace("policy_gate", tool=tool_name, allowed=False, reason="unknown_tool")
        return PolicyDecision(
            allowed=False,
            events=events,
            blocked_reason=f"unknown tool: {tool_name}",
        )

    # 5. Session grant check — non-expired grant bypasses risk gates (not trust/high-risk)
    granted = False
    grant_source = ""
    _is_high_risk = tool_def is not None and tool_def.risk_level == "high"
    if not confirmed and not _is_high_risk and tool_grants and tool_name in tool_grants:
        expiry = tool_grants[tool_name]
        if time.time() < expiry:
            granted = True
            grant_source = "session_grant"
        else:
            # Expired — remove stale grant
            tool_grants.pop(tool_name, None)

    # 5b. Shell command tier-based gating
    #     classify_command() returns safe/dev/privileged/denied.
    #     - safe: pass through to standard evaluation
    #     - dev: requires a shell-specific grant (grant_key) or confirmation
    #     - privileged: always requires confirmation (2-step)
    #     - denied: handled inside shell_exec itself
    if tool_name == "shell_exec" and not confirmed:
        _shell_decision = _shell_tier_gate(
            tool_args, tool_def, tool_grants, granted, ctx, pending_confirmations,
            clean_resp,
        )
        if _shell_decision is not None:
            return _shell_decision

    # 6. Static policy evaluation (HARD GATE — runs before LLM review)
    #    Static rules are deterministic and reliable; they form the primary
    #    security boundary.  LLM review is advisory only.
    allowed, reason = tool_policy.evaluate(tool_def, tool_args, confirmed=confirmed, granted=granted)
    _log.trace("policy_eval", tool=tool_name, allowed=allowed, reason=reason or "",
               confirmed=confirmed, granted=granted)

    if not allowed:
        if reason.startswith("confirmation required"):
            token = uuid.uuid4().hex[:10]
            required_stage = (
                2 if tool_def.risk_level == "high"
                or getattr(tool_def.capability, "data_classification", "public") == "sensitive"
                or "repl_mode=trusted_local" in reason
                else 1
            )
            pending_confirmations[token] = {
                "tool_name": tool_name, "tool_args": tool_args,
                "created_at": datetime.now(timezone.utc),
                "user_input": ctx.user_input, "assistant_tool_call": clean_resp,
                "required_stage": required_stage, "stage": 1,
                "pending_reason": reason,
            }
            # Attach trust server ID for auto-approval on confirm
            if "trust=unknown" in reason and "__" in tool_name:
                pending_confirmations[token]["trust_server_id"] = tool_name.split("__", 1)[0]
            brief = tool_policy.confirmation_brief(
                tool_def, tool_args, reason, stage=1, required_stage=required_stage,
            )
            tool_policy.audit(tool_name, tool_args, "pending", reason,
                              policy_decision="pending_confirmation")
            confirm_event = (
                "confirmation_required", token, tool_name, reason,
                json.dumps(brief, ensure_ascii=False),
            )
            if required_stage > 1:
                done_msg = f"Tool `{tool_name}` requires two-step confirmation. First `/confirm {token}`, then `/confirm {token} --force`."
            else:
                done_msg = f"Tool `{tool_name}` requires confirmation. Use `/confirm {token}` or `/reject {token}`."
            _log.trace("policy_gate", tool=tool_name, allowed=False, reason="needs_confirmation_static")
            return PolicyDecision(
                allowed=False,
                events=[confirm_event],
                must_return=True,
                return_events=[("done", done_msg)],
            )

        # Hard block (not confirmation)
        if tool_name == "write_file" and "content" in reason:
            hint_msg = (
                "write_file is missing the `content` argument. "
                "You must provide both `path` and `content` in args. "
                "Call write_file again with the full code in `content`."
            )
        else:
            hint_msg = "Blocked by policy. Answer with available information or try an alternative approach."
        obs, _, block_events = handle_policy_block_fn(
            tool_name, tool_args, reason,
            full_response, hint_msg,
        )
        _log.trace("policy_gate", tool=tool_name, allowed=False, reason=reason)
        return PolicyDecision(
            allowed=False,
            events=block_events,
            blocked_reason=reason,
        )

    # 7. LLM policy review (ADVISORY — may escalate to confirmation)
    #    Only runs after static policy allows.  Parse failures default to
    #    needs_confirmation=True (conservative), not silent allow.
    do_policy_review = (
        ctx.budget.enable_policy_review
        and enable_policy_review
        and not (ctx.low_latency and disable_policy_review_in_voice)
        and tool_def.risk_level == "high"
    )
    events_so_far: list[tuple] = []
    if do_policy_review:
        cur_step = None
        review = await planner.review_tool_action(
            user_input=ctx.user_input, step=cur_step,
            tool_name=tool_name, tool_args=tool_args,
            capability_desc=tool_policy.capability_summary(tool_def),
        )
        review_obj = {
            "allow": review.allow, "risk": review.risk,
            "needs_confirmation": review.needs_confirmation, "reason": review.reason,
        }
        review_event = ("policy_review", tool_name, json.dumps(review_obj, ensure_ascii=False))

        if not review.allow:
            reason = f"policy review rejected: {review.reason or 'denied'}"
            obs, _, block_events = handle_policy_block_fn(
                tool_name, tool_args, reason,
                full_response, "Policy review rejected this tool call. Answer directly from available information.",
            )
            _log.trace("policy_gate", tool=tool_name, allowed=False, reason="policy_review_rejected")
            return PolicyDecision(
                allowed=False,
                events=[review_event] + block_events,
                blocked_reason=reason,
            )

        if review.needs_confirmation and not confirmed:
            reason = f"confirmation required by policy-review ({review.risk})"
            token = uuid.uuid4().hex[:10]
            required_stage = (
                2 if review.risk == "high"
                or getattr(tool_def.capability, "data_classification", "public") == "sensitive"
                else 1
            )
            pending_confirmations[token] = {
                "tool_name": tool_name, "tool_args": tool_args,
                "created_at": datetime.now(timezone.utc),
                "user_input": ctx.user_input, "assistant_tool_call": clean_resp,
                "required_stage": required_stage, "stage": 1,
                "pending_reason": reason,
            }
            brief = tool_policy.confirmation_brief(
                tool_def, tool_args, reason, stage=1, required_stage=required_stage,
            )
            tool_policy.audit(tool_name, tool_args, "pending", reason,
                              policy_decision="pending_confirmation")
            confirm_event = (
                "confirmation_required", token, tool_name, reason,
                json.dumps(brief, ensure_ascii=False),
            )
            if required_stage > 1:
                done_msg = f"Tool `{tool_name}` requires two-step confirmation. First `/confirm {token}`, then `/confirm {token} --force`."
            else:
                done_msg = f"Tool `{tool_name}` requires confirmation. Use `/confirm {token}` or `/reject {token}`."
            _log.trace("policy_gate", tool=tool_name, allowed=False, reason="needs_confirmation_review")
            return PolicyDecision(
                allowed=False,
                events=[review_event, confirm_event],
                must_return=True,
                return_events=[("done", done_msg)],
            )

        # Policy review approved → override confirmed
        confirmed = True
        events_so_far = [review_event]

    # Determine auth_mode for audit trail
    if confirmed:
        auth_mode = "confirmed"
    elif granted:
        auth_mode = "granted"
    else:
        auth_mode = "allowed"

    _log.trace("policy_gate", tool=tool_name, allowed=True, reason="allowed", auth_mode=auth_mode)
    return PolicyDecision(
        allowed=True,
        events=events_so_far,
        confirmed=confirmed,
        auth_mode=auth_mode,
        grant_source=grant_source,
    )
