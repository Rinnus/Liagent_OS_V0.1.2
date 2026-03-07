"""Synthesizer — multi-source synthesis with forced citation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger
from .sub_agent import SubAgentResult

_log = get_logger("synthesizer")


@dataclass
class SynthesisResult:
    answer: str = ""
    citations: list[dict[str, str]] = field(default_factory=list)
    agent_results: list[SubAgentResult] = field(default_factory=list)
    quality_meta: dict[str, Any] = field(default_factory=dict)


_CITATION_RE = re.compile(r"\[sub:(\d+)\]")


def extract_citations(
    text: str, results: list[SubAgentResult],
) -> list[dict[str, str]]:
    """Validate [sub:N] references in text map to actual results.

    Returns list of citation dicts. Unmapped refs marked [unverified].
    """
    citations: list[dict[str, str]] = []
    seen: set[int] = set()
    for m in _CITATION_RE.finditer(text):
        idx = int(m.group(1))
        if idx in seen:
            continue
        seen.add(idx)
        if 0 <= idx < len(results) and results[idx].success:
            r = results[idx]
            citations.append({
                "ref": f"[sub:{idx}]",
                "agent_id": r.agent_id,
                "strategy": r.strategy,
                "urls": ", ".join(r.source_urls[:3]) if r.source_urls else "",
                "status": "verified",
            })
        else:
            citations.append({
                "ref": f"[sub:{idx}]",
                "agent_id": "",
                "strategy": "",
                "urls": "",
                "status": "unverified",
            })
    return citations


class Synthesizer:
    """Combine multiple SubAgentResult observations into a coherent answer.

    The actual LLM synthesis call is delegated to the caller (engine).
    This class handles citation extraction and quality metadata.
    """

    def __init__(self, *, generate_fn: Any = None):
        self._generate_fn = generate_fn

    async def synthesize(
        self,
        query: str,
        results: list[SubAgentResult],
        *,
        budget: Any = None,
        prior_partial: str | None = None,
        quality_retry_hint: str = "",
    ) -> SynthesisResult:
        """Build synthesis from SubAgent results.

        If generate_fn is provided, uses LLM to synthesize. Otherwise
        concatenates summaries as fallback.
        """
        successful = [r for r in results if r.success and r.summary]
        if not successful:
            return SynthesisResult(
                answer="No data collected from sub-agents.",
                agent_results=results,
                quality_meta={"issues": ["no_data"]},
            )

        # Build context for synthesis
        context_parts: list[str] = []
        for i, r in enumerate(successful):
            part = f"[sub:{i}] ({r.strategy}): {r.summary}"
            if r.source_urls:
                part += f"\nSources: {', '.join(r.source_urls[:3])}"
            context_parts.append(part)

        combined_context = "\n\n".join(context_parts)

        if self._generate_fn is not None:
            prompt = (
                f"Synthesize the following research results to answer: {query}\n\n"
                f"{combined_context}\n\n"
                "Cite sources using [sub:N] notation. "
                "Be concise and factual."
            )
            if quality_retry_hint:
                prompt += f"\n\nQuality note: {quality_retry_hint}"
            if prior_partial:
                prompt += f"\n\nPrevious attempt (improve upon): {prior_partial[:500]}"

            try:
                answer = await self._generate_fn(prompt)
            except Exception as e:
                _log.warning(f"synthesis_llm_failed: {e}")
                answer = combined_context
        else:
            answer = combined_context

        citations = extract_citations(answer, successful)
        return SynthesisResult(
            answer=answer,
            citations=citations,
            agent_results=results,
        )
