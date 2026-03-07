"""Policy Router — intent classification replacing keyword regex.

Two-tier design:
1. Fast regex pre-filter for unambiguous cases (greetings, code, high-risk)
2. Rule-based heuristics for remaining cases (entity count, question patterns)

LLM-based classification is reserved for P1 when we have labeled data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..logging import get_logger

_log = get_logger("policy_router")


# ── Fast-path regex patterns ──────────────────────────────────────────

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s*(?:morning|evening|afternoon))"
    r"[!.]?\s*$",
    re.IGNORECASE,
)

_META_REFLECTIVE_RE = re.compile(
    r"what do you think|do you agree|is that right|can you redo"
    r"|that seems wrong|please answer again|try again",
    re.IGNORECASE,
)

_FOLLOWUP_ANAPHORA_RE = re.compile(
    r"^(and |also |what about |how about |then |next |another )",
    re.IGNORECASE,
)

_CODE_TASK_RE = re.compile(
    r"write\s+(?:a\s+)?(?:python|script|code|function|class|program)"
    r"|implement\s+a"
    r"|refactor|debug|fix\s+(?:the|this)\s+(?:bug|error|issue)",
    re.IGNORECASE,
)

_HIGH_RISK_RE = re.compile(
    r"(?:delete|remove|drop|rm\s+-rf|kill|shutdown|destroy|purge)"
    r"(?:\s+all|\s+every|\s+/)",
    re.IGNORECASE,
)

_PRIVATE_RE = re.compile(
    r"my\s+(?:file|folder|document|photo|password|secret|key)"
    r"|~/|/Users/|/home/|localhost",
    re.IGNORECASE,
)

_RESEARCH_SIGNAL_RE = re.compile(
    r"compare|contrast|analyze|analysis|research|investigate"
    r"|summarize|earnings|revenue|market|industry|trend|outlook|report"
    r"|benchmark|review|evaluate",
    re.IGNORECASE,
)

_DEEP_RESEARCH_RE = re.compile(
    r"deep|thorough|comprehensive|source|primary|official|10-k|10q"
    r"|annual\s+report|quarterly\s+report",
    re.IGNORECASE,
)

_LOOKUP_SIGNAL_RE = re.compile(
    r"(?:price|stock\s+price|current\s+price|latest|today|quote)"
    r"(?!\s*(?:trend|compare|analy))",
    re.IGNORECASE,
)

_ENTITY_SPLIT_RE = re.compile(r"[,;]|\s+(?:and|vs\.?|versus)\s+")

_ENTITY_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "of", "and", "in", "to", "for",
    "what", "how", "why", "when", "where", "which", "who",
    "compare", "contrast", "analyze", "find", "search", "show", "tell",
})

_ENTITY_ACTION_RE = re.compile(
    r"^(compare|contrast|analyze|find|search|tell\s+me|show\s+me|look\s+up)$",
    re.IGNORECASE,
)


@dataclass
class Intent:
    """Classified intent for a user query."""
    category: str  # "chat" | "lookup" | "research" | "code" | "meta"
    task_class: str  # "chat" | "research" | "code_exec" | "high_risk_tool" | "long_context"
    confidence: float = 0.9
    entities: list[str] = field(default_factory=list)
    is_followup: bool = False
    needs_tool: bool = False
    data_sensitivity: str = "mixed"
    deep_research: bool = False
    reason: str = ""


class PolicyRouter:
    """Intent classifier for query routing."""

    def classify(
        self,
        query: str,
        *,
        images: list[str] | None = None,
        low_latency: bool = False,
    ) -> Intent:
        text = (query or "").strip()
        if not text:
            return Intent(category="chat", task_class="chat", reason="empty")

        if len(text) < 6:
            return Intent(category="chat", task_class="chat", confidence=0.95, reason="short")

        if _GREETING_RE.fullmatch(text):
            return Intent(category="chat", task_class="chat", confidence=0.99, reason="greeting")

        if _META_REFLECTIVE_RE.search(text):
            return Intent(
                category="chat", task_class="chat", confidence=0.85,
                reason="meta_reflective",
            )

        if _HIGH_RISK_RE.search(text):
            return Intent(
                category="code", task_class="high_risk_tool",
                data_sensitivity="private_local_only",
                needs_tool=True, reason="high_risk_signal",
            )

        if _CODE_TASK_RE.search(text):
            return Intent(
                category="code", task_class="code_exec",
                needs_tool=True, reason="code_signal",
            )

        has_private = bool(_PRIVATE_RE.search(text))
        is_followup = bool(_FOLLOWUP_ANAPHORA_RE.search(text))
        entities = self._extract_entities(text)

        has_research = bool(_RESEARCH_SIGNAL_RE.search(text))
        has_deep = bool(_DEEP_RESEARCH_RE.search(text))
        has_lookup = bool(_LOOKUP_SIGNAL_RE.search(text))
        has_question = "?" in text or "？" in text

        if has_lookup and not has_research and len(entities) <= 1:
            return Intent(
                category="lookup", task_class="research",
                entities=entities, is_followup=is_followup,
                needs_tool=True, deep_research=False,
                data_sensitivity="private_local_only" if has_private else "public_cloud_ok",
                reason="lookup_signal",
            )

        if has_research or len(entities) >= 2:
            return Intent(
                category="research", task_class="research",
                entities=entities, is_followup=is_followup,
                needs_tool=True, deep_research=has_deep,
                data_sensitivity="private_local_only" if has_private else "public_cloud_ok",
                reason="research_signal" if has_research else "multi_entity",
            )

        if has_question and entities:
            return Intent(
                category="lookup", task_class="research",
                entities=entities, is_followup=is_followup,
                needs_tool=True,
                data_sensitivity="private_local_only" if has_private else "mixed",
                reason="question_with_entity",
            )

        return Intent(
            category="chat", task_class="chat",
            entities=entities, is_followup=is_followup,
            data_sensitivity="private_local_only" if has_private else "mixed",
            reason="default_chat",
        )

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        parts = [p.strip() for p in _ENTITY_SPLIT_RE.split(text) if p and p.strip()]
        cleaned: list[str] = []
        seen: set[str] = set()
        for p in parts:
            tokens = p.split()
            tokens = [t for t in tokens if t.lower() not in _ENTITY_STOPWORDS]
            normalized = " ".join(tokens).strip()
            if len(normalized) < 2:
                continue
            if _ENTITY_ACTION_RE.fullmatch(normalized):
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized)
        return cleaned[:8]
