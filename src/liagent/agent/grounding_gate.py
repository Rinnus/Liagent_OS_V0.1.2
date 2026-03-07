"""Grounding Gate — unified quality gate for both brain and orchestrator paths.

Wraps the existing quality checks from quality.py behind a simple
interface that doesn't require the full RunContext machinery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .quality import (
    detect_copout,
    detect_degenerate_output,
    detect_hallucinated_action,
    detect_ungrounded_numbers,
    detect_unsourced_data,
    quality_fix,
)
from .text_utils import clean_output
from .tool_parsing import strip_any_tool_call
from ..logging import get_logger

_log = get_logger("grounding_gate")


@dataclass
class GateVerdict:
    """Result of a grounding gate check."""
    action: str  # "accept" | "retry" | "abort_degenerate"
    answer: str = ""
    quality_meta: dict = field(default_factory=dict)
    retry_reason: str = ""
    retry_hint: str = ""


class GroundingGate:
    """Unified quality gate callable from brain.py and orchestrator paths.

    Usage:
        gate = GroundingGate(experience=experience_memory)
        verdict = gate.check(answer=text, query=user_query, source_urls=[...])
    """

    def __init__(self, *, experience: Any = None):
        self._experience = experience

    def check(
        self,
        *,
        answer: str,
        query: str,
        source_urls: list[str] | None = None,
        context_vars: dict[str, str] | None = None,
        step: int = 0,
        max_steps: int = 1,
        tools_used: set[str] | None = None,
        copout_retried: bool = False,
        unsourced_retried: bool = False,
        ungrounded_retried: bool = False,
        hallucination_retried: bool = False,
    ) -> GateVerdict:
        """Run quality checks on an answer. Stateless — caller tracks retry flags."""
        text = str(answer or "").strip()
        if not text:
            return GateVerdict(action="retry", retry_reason="empty_answer")

        text = strip_any_tool_call(text)
        text = clean_output(text)

        _tools = tools_used or set()

        # 1. Degenerate output
        if detect_degenerate_output(text):
            _log.trace("grounding_gate_degenerate", answer_len=len(text))
            return GateVerdict(
                action="abort_degenerate",
                answer=self._best_effort_answer(query),
                quality_meta={"issues": ["degenerate_output"], "score": 0.0},
            )

        # 2. Copout
        if not copout_retried and detect_copout(text) and step < max_steps - 1:
            _log.trace("grounding_gate_copout")
            return GateVerdict(
                action="retry",
                retry_reason="copout",
                retry_hint=(
                    "Your previous response was a copout. "
                    "Answer the question directly using available data."
                ),
            )

        # 3. Hallucinated action
        if not hallucination_retried and step < max_steps - 1:
            missing = detect_hallucinated_action(text, _tools)
            if missing:
                _log.trace("grounding_gate_hallucinated_action", tool=missing)
                return GateVerdict(
                    action="retry",
                    retry_reason="hallucinated_action",
                    retry_hint=f"Call `{missing}` to actually perform the operation.",
                )

        # 4. Unsourced data
        if not unsourced_retried and step < max_steps - 1:
            if detect_unsourced_data(text, query, _tools):
                _log.trace("grounding_gate_unsourced")
                return GateVerdict(
                    action="retry",
                    retry_reason="unsourced_data",
                    retry_hint="Use a data-fetch tool first, then cite from results.",
                )

        # 5. Ungrounded numbers
        if not ungrounded_retried and context_vars and step < max_steps - 1:
            obs_texts = [
                v for k, v in context_vars.items()
                if isinstance(v, str) and k.endswith("_result")
            ]
            if obs_texts:
                combined = "\n".join(obs_texts)
                if detect_ungrounded_numbers(text, combined):
                    _log.trace("grounding_gate_ungrounded")
                    return GateVerdict(
                        action="retry",
                        retry_reason="ungrounded_numbers",
                        retry_hint="Verify numeric claims against source data.",
                    )

        # Accept
        fixed, qmeta = quality_fix(text)
        return GateVerdict(
            action="accept",
            answer=fixed,
            quality_meta=qmeta,
        )

    @staticmethod
    def _best_effort_answer(query: str) -> str:
        return f"Unable to generate a quality answer for: {query}"
