"""Checkpoint semantic matching — lexical + alias + embedding + time gate."""

from __future__ import annotations

import re
from datetime import datetime, timezone

# ── Alias table: cross-language concept pairs ──────────────────────────
_ALIASES: list[tuple[str, ...]] = [
    ("AAPL", "apple"),
    ("GOOGL", "GOOG", "google"),
    ("MSFT", "microsoft"),
    ("AMZN", "amazon"),
    ("TSLA", "tesla"),
    ("META", "facebook", "meta"),
    ("NVDA", "nvidia"),
    ("BABA", "alibaba"),
    ("JD", "jd.com"),
    ("PDD", "pinduoduo"),
    ("BIDU", "baidu"),
    ("stock", "price", "stock price"),
    ("earnings", "quarterly", "annual"),
    ("revenue", "income"),
    ("profit", "net income"),
    ("market cap", "valuation"),
    ("analysis", "review"),
    ("chart", "graph"),
    ("weather", "forecast"),
    ("news", "headlines"),
]

# Build reverse lookup: token → canonical (first item in alias group)
_TOKEN_TO_CANONICAL: dict[str, str] = {}
for _group in _ALIASES:
    canonical = _group[0].lower()
    for _alias in _group:
        _TOKEN_TO_CANONICAL[_alias.lower()] = canonical


def _tokenize_simple(text: str) -> list[str]:
    """Tokenize text — simple word split."""
    tokens = re.findall(r"[\w]+", text)
    return [t.lower().strip() for t in tokens if t.strip() and len(t.strip()) > 0]


def _normalize(tokens: list[str]) -> set[str]:
    """Map tokens to canonical forms via alias table."""
    result: set[str] = set()
    for t in tokens:
        canonical = _TOKEN_TO_CANONICAL.get(t)
        if canonical:
            result.add(canonical)
        result.add(t)
    return result


def _word_overlap_score(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard-like overlap using min denominator for tolerance."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    denom = min(len(set_a), len(set_b))
    return len(intersection) / denom if denom > 0 else 0.0


def _time_penalty(created_at: str | None) -> float:
    """Time decay multiplier: fresh=1.0, >24h=0.5, >48h=0.25."""
    if not created_at:
        return 1.0
    try:
        ts = datetime.fromisoformat(created_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        if age_hours > 48:
            return 0.25
        if age_hours > 24:
            return 0.5
        return 1.0
    except (ValueError, TypeError):
        return 1.0


def checkpoint_relevance(
    goal: str,
    user_input: str,
    *,
    embedder=None,
    created_at: str | None = None,
) -> float:
    """Score relevance between a checkpoint goal and current user input.

    Returns:
        Float 0.0-1.0 relevance score.
    """
    # Layer 1: Raw lexical overlap
    tokens_goal = _tokenize_simple(goal)
    tokens_input = _tokenize_simple(user_input)
    set_goal = set(tokens_goal)
    set_input = set(tokens_input)
    lexical = _word_overlap_score(set_goal, set_input)

    # Layer 2: Alias-normalized overlap
    norm_goal = _normalize(tokens_goal)
    norm_input = _normalize(tokens_input)
    alias_score = _word_overlap_score(norm_goal, norm_input)

    # Layer 3: Embedding cosine similarity
    semantic = 0.0
    if embedder is not None:
        try:
            vecs = embedder.encode([goal, user_input])
            if vecs is not None and len(vecs) >= 2:
                import numpy as np

                v1, v2 = vecs[0], vecs[1]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    cosine = float(np.dot(v1, v2) / (norm1 * norm2))
                    semantic = max(0.0, cosine)
        except Exception:
            semantic = 0.0

    # Combine: best of lexical/alias, with semantic weighted down slightly
    combined = max(lexical, alias_score, semantic * 0.85)

    # Apply time penalty
    penalty = _time_penalty(created_at)
    return combined * penalty
