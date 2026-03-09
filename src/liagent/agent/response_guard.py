"""Response guard — post-generation quality checks extracted from brain.py run()."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..logging import get_logger
from .evidence import aggregate_evidence
from .run_context import RunContext
from .quality import (
    detect_copout,
    detect_degenerate_output,
    detect_hallucinated_action,
    detect_progress_placeholder,
    detect_tool_protocol_leak,
    detect_unsourced_tool_failure,
    detect_ungrounded_numbers,
    detect_unsourced_data,
    detect_unwritten_code,
    quality_fix,
    validate_key_metrics,
)
from .text_utils import clean_output
from .tool_parsing import contains_tool_call_syntax, strip_any_tool_call

_log = get_logger("response_guard")
_RESIDUAL_TOOL_PROTOCOL_RE = re.compile(r"<(?:tool_call|function_calls|invoke)\b.*", re.DOTALL | re.IGNORECASE)

_RETRY_STATUS_TEXT = {
    "progress_placeholder": "The previous reply only described process and did not contain actual results. Retrying from the real execution state.",
    "hallucinated_tool_failure": "The previous reply described a tool failure that was not observed. Retrying from the real execution state.",
    "copout": "The previous reply was not specific or reliable enough. Retrying with a different approach.",
    "hallucination": "The previous reply claimed completion without the required execution. Completing the missing execution and retrying.",
    "unsourced_data": "The previous reply cited unverified data. Fetching real data before retrying.",
    "ungrounded_numbers": "The previous reply included numbers that do not match the tool outputs. Rechecking the raw results before retrying.",
    "unwritten_code": "The previous reply described code changes, but no file write happened. Writing the file before retrying.",
}


@dataclass
class GuardResult:
    action: str  # "accept" | "retry" | "abort_degenerate" | "experience_guard_tool"

    # accept
    answer: str = ""
    quality_meta: dict = field(default_factory=dict)

    # retry
    retry_reason: str = ""
    retry_injection: str = ""  # user message injected into memory

    # abort_degenerate
    abort_answer: str = ""
    abort_events: list[tuple] = field(default_factory=list)

    # experience_guard_tool
    guard_tool_name: str = ""
    guard_tool_args: dict = field(default_factory=dict)


def describe_retry_reason(reason: str) -> str:
    key = str(reason or "").strip()
    return _RETRY_STATUS_TEXT.get(key, "The previous reply was not stable enough. Retrying once.")


async def check_response(
    *,
    full_response: str,
    step: int,
    ctx: RunContext,
    experience,
    best_effort_answer_fn: Callable,
    build_done_events_fn: Callable,
) -> GuardResult:
    """Run all response quality checks. Returns a GuardResult describing what to do."""

    # 0. Think-based tool intent detection (positive/negative cancellation)
    if ctx.reasoning_content and ctx.tool_calls == 0 and step == 0:
        think = ctx.reasoning_content
        tool_intent = any(p in think for p in ["need tool", "should use tool", "i should search", "look up", "search"])
        tool_dismissed = any(n in think for n in ["no tool needed", "enough known info", "answer directly", "no need to search"])
        if tool_intent and not tool_dismissed:
            # Check if experience also suggests a tool — combine signals
            _em = ctx.experience_match
            if _em and _em.suggested_tool:
                from ..tools import get_tool
                guard_tool_name = _em.suggested_tool
                guard_tool_def = get_tool(guard_tool_name)
                if guard_tool_def is not None:
                    if guard_tool_name == "web_search":
                        guard_args = {"query": ctx.user_input}
                    else:
                        guard_args = {"query": ctx.user_input}
                    _log.trace("response_guard", action="think_intent_guard",
                               tool=guard_tool_name)
                    return GuardResult(
                        action="experience_guard_tool",
                        guard_tool_name=guard_tool_name,
                        guard_tool_args=guard_args,
                    )

    # 1. Experience guard — intercept when model skipped a tool it should have used
    experience_match = ctx.experience_match
    if (
        experience_match
        and experience_match.should_use_tool
        and experience_match.confidence >= 0.6
        and ctx.tool_calls == 0
        and step == 0
        and experience_match.suggested_tool
    ):
        from ..tools import get_tool
        _log.trace("response_guard",
                   action="experience_guard_intercept",
                   match_pattern=getattr(experience_match, 'pattern', ''),
                   suggested_tool=experience_match.suggested_tool)
        experience.record_outcome(ctx.user_input, None, False, "self_eval")
        guard_tool_name = experience_match.suggested_tool
        guard_tool_def = get_tool(guard_tool_name)
        if guard_tool_def is not None:
            if guard_tool_name == "web_search":
                guard_args = {"query": ctx.user_input}  # caller will refine
            else:
                guard_args = {"query": ctx.user_input}

            _log.trace("response_guard", action="experience_guard_tool",
                       step=step, tool=guard_tool_name)
            return GuardResult(
                action="experience_guard_tool",
                guard_tool_name=guard_tool_name,
                guard_tool_args=guard_args,
            )

    # 2. Degenerate output
    if detect_degenerate_output(full_response):
        stripped = strip_any_tool_call(full_response)
        stripped = _RESIDUAL_TOOL_PROTOCOL_RE.sub("", stripped).strip()
        if stripped and len(stripped) > 10:
            answer = clean_output(stripped)
            answer, qmeta = quality_fix(answer)
        else:
            answer, qmeta = await best_effort_answer_fn(
                "Model output became degenerate (repetitive generation) and was interrupted."
            )
        ctx.quality_issues.append("degenerate_output")
        ctx.revision_count += 1
        events = [("token", answer)]
        done_events = build_done_events_fn(
            answer=answer, start_ts=ctx.start_ts,
            tool_calls=ctx.tool_calls, tool_errors=ctx.tool_errors,
            policy_blocked=ctx.policy_blocked,
            plan_total_steps=ctx.plan_total_steps, plan_idx=ctx.plan_idx,
            revision_count=ctx.revision_count, quality_issues=ctx.quality_issues,
            tools_used=ctx.tools_used, force_success=False,
        )
        events.extend(done_events)
        _log.trace("response_guard", action="abort_degenerate", step=step)
        return GuardResult(
            action="abort_degenerate",
            abort_answer=answer,
            abort_events=events,
        )

    # 3. Strip residual tool_call fragments
    cleaned_response = full_response
    if contains_tool_call_syntax(cleaned_response):
        cleaned_response = strip_any_tool_call(cleaned_response)
        cleaned_response = _RESIDUAL_TOOL_PROTOCOL_RE.sub("", cleaned_response).strip()

    answer = clean_output(cleaned_response)
    answer, qmeta = quality_fix(answer)

    _log.trace("response_guard",
               action="quality_check", step=step,
               copout=detect_copout(answer),
               progress_placeholder=detect_progress_placeholder(answer),
               tool_failure_claim=detect_unsourced_tool_failure(answer),
               tool_protocol_leak=detect_tool_protocol_leak(answer),
               hallucination=detect_hallucinated_action(answer, ctx.tools_used) or "",
               unwritten_code=detect_unwritten_code(answer, ctx.user_input, ctx.tools_used)[0])

    # 3.5. Progress placeholder retry
    if (
        detect_progress_placeholder(answer)
        and not ctx.progress_placeholder_retried
        and step < ctx.max_steps - 1
        and not ctx.retry_budget_exhausted
    ):
        ctx.progress_placeholder_retried = True
        _log.trace("response_guard", action="retry", type="progress_placeholder")
        return GuardResult(
            action="retry",
            retry_reason="progress_placeholder",
            retry_injection=(
                "Do not answer with progress chatter such as 'let me check'. "
                "If you need a tool, emit only the tool call. If execution is blocked on confirmation, "
                "state that you are waiting for confirmation and do not pretend the work is still running."
            ),
            answer=answer,
        )

    # 3.6. Tool-failure hallucination retry
    if (
        detect_unsourced_tool_failure(answer)
        and ctx.tool_errors == 0
        and not ctx.hallucination_retried
        and step < ctx.max_steps - 1
        and not ctx.retry_budget_exhausted
    ):
        ctx.hallucination_retried = True
        ctx.quality_issues.append("hallucinated_tool_failure")
        ctx.revision_count += 1
        _log.trace("response_guard", action="retry", type="hallucinated_tool_failure")
        return GuardResult(
            action="retry",
            retry_reason="hallucinated_tool_failure",
            retry_injection=(
                "Do not claim that a tool failed, timed out, or returned no result unless you actually observed a tool error. "
                "If no tool ran yet, emit the correct tool call only. "
                "If execution is blocked on confirmation, state that explicitly instead of inventing a technical issue."
            ),
            answer=answer,
        )

    # 4. Copout retry
    if detect_copout(answer) and ctx.tool_calls > 0 and not ctx.copout_retried and step < ctx.max_steps - 1 and not ctx.retry_budget_exhausted:
        ctx.copout_retried = True
        _log.trace("response_guard", action="retry", type="copout")
        return GuardResult(
            action="retry",
            retry_reason="copout",
            retry_injection=(
                "Search quality is insufficient. Retry with different English keywords, "
                "using either more specific or more general terms. "
                "Do not give up and do not ask the user to search manually."
            ),
            answer=answer,
        )

    # 5. Hallucinated action retry
    missing_tool = detect_hallucinated_action(answer, ctx.tools_used)
    if missing_tool and not ctx.hallucination_retried and step < ctx.max_steps - 1 and not ctx.retry_budget_exhausted:
        ctx.hallucination_retried = True
        _log.trace("response_guard", action="retry", type="hallucination", tool=missing_tool)
        return GuardResult(
            action="retry",
            retry_reason="hallucination",
            retry_injection=(
                f"You claimed the operation is complete, but `{missing_tool}` was not called. "
                f"Call `{missing_tool}` to actually perform the operation; do not simulate completion."
            ),
            answer=answer,
        )

    # 6. Unsourced data retry — model cites specific numbers without tool backing
    if (
        detect_unsourced_data(answer, ctx.user_input, ctx.tools_used)
        and not ctx.unsourced_data_retried
        and step < ctx.max_steps - 1
        and not ctx.retry_budget_exhausted
    ):
        ctx.unsourced_data_retried = True
        _log.trace("response_guard", action="retry", type="unsourced_data")
        ctx.quality_issues.append("unsourced_data")
        ctx.revision_count += 1
        return GuardResult(
            action="retry",
            retry_reason="unsourced_data",
            retry_injection=(
                "Your answer cites concrete numbers without any data-fetch tool call. "
                "Those numbers may be incorrect. Use `web_search` to fetch fresh data first, "
                "then answer from results. Do not rely on memory for exact figures."
            ),
            answer=answer,
        )

    # 6.2. Ungrounded numbers — answer cites numbers not found in tool observations
    if (
        ctx.tools_used & {"web_search", "web_fetch", "stock", "system_status"}
        and ctx.tool_calls > 0
        and not ctx.ungrounded_retried
        and step < ctx.max_steps - 1
        and not ctx.retry_budget_exhausted
    ):
        obs_texts_ug = [
            v for k, v in ctx.context_vars.items()
            if isinstance(v, str) and k.endswith("_result")
        ]
        if obs_texts_ug:
            combined_obs_ug = "\n".join(obs_texts_ug)
            if detect_ungrounded_numbers(answer, combined_obs_ug):
                ctx.ungrounded_retried = True
                _log.trace("response_guard", action="retry", type="ungrounded")
                ctx.quality_issues.append("ungrounded_numbers")
                ctx.revision_count += 1
                return GuardResult(
                    action="retry",
                    retry_reason="ungrounded_numbers",
                    retry_injection=(
                        "Several numbers in your answer cannot be mapped to search results. "
                        "Re-check raw tool outputs and cite only numbers explicitly present there. "
                        "Do not add figures from memory."
                    ),
                    answer=answer,
                )

    # 7. Unwritten code retry
    should_write, code_lines, code_score = detect_unwritten_code(
        answer, ctx.user_input, ctx.tools_used,
    )
    if should_write and not ctx.unwritten_code_retried and step < ctx.max_steps - 1 and not ctx.retry_budget_exhausted:
        ctx.unwritten_code_retried = True
        _log.trace("response_guard", action="retry", type="unwritten_code",
                   lines=code_lines, score=code_score)
        ctx.quality_issues.append("unwritten_code")
        ctx.revision_count += 1
        return GuardResult(
            action="retry",
            retry_reason="unwritten_code",
            retry_injection=(
                f"Your response includes {code_lines} lines of code (compliance score={code_score}) "
                "but `write_file` was not called. "
                "Use `write_file` to save the full code to an appropriate path, "
                "and do not paste full code in the chat response. "
                f"Infer file name/path from the original user request: \"{ctx.user_input}\""
            ),
            answer=answer,
        )

    # 8. Key metrics cross-validation (best-effort for market/finance summaries)
    market_obs = str(ctx.context_vars.get("web_search_result", "") or "")
    if market_obs:
        answer, metric_fixes = validate_key_metrics(answer, market_obs)
        if metric_fixes:
            ctx.quality_issues.append("key_metrics_fixed")
            ctx.revision_count += 1
            _log.trace("response_guard", action="metrics_fix", fixes=metric_fixes)

    # 8.5. Chart data consistency tracking
    chart_raw = ctx.context_vars.get("chart_analysis_result", "")
    if chart_raw and ctx.tool_calls > 0:
        from .chart_analysis import parse_chart_result
        chart = parse_chart_result(chart_raw)
        if chart.chart_type != "unknown":
            ctx.quality_issues.append("chart_grounded")

    # 8.6. Evidence aggregation — append cross-source summary when multiple tools used
    if ctx.tool_calls >= 2:
        evidence = aggregate_evidence(ctx.context_vars, ctx.source_urls)
        if evidence:
            answer = answer + "\n\n" + evidence
            _log.trace("response_guard", action="evidence_appended",
                       evidence_len=len(evidence))

    # 9. Accept
    _log.trace("response_guard", action="accept", step=step)
    return GuardResult(
        action="accept",
        answer=answer,
        quality_meta=qmeta,
    )


def compute_confidence_label(
    evidence_list: list[dict],
    quality_issues: list[str],
    experience_score: float = 0.5,
) -> tuple[str, str | None]:
    """Compute user-visible confidence label from quality signals."""
    src_count = len({e.get("tool") for e in evidence_list})
    cross_validated = src_count >= 2
    has_issues = bool(quality_issues)

    if cross_validated and not has_issues and experience_score > 1.2:
        return "high", None
    elif has_issues or src_count == 0:
        if src_count <= 1:
            note = "Single source only, consider cross-validation"
        else:
            note = quality_issues[0] if quality_issues else "No data sources"
        return "low", note
    else:
        return "medium", None
