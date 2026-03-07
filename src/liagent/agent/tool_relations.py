"""Tool relation graph: fallback, equivalence, and composition relationships.

Part of Iteration E2 — the understanding layer for tool intelligence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse


@dataclass(frozen=True)
class ToolRelation:
    """Directed edge between two tools describing a fallback or equivalence relationship."""

    source: str  # triggering tool
    target: str  # alternative tool (can be self for retry)
    relation: str  # "fallback" | "equivalent" | "superset" | "compose_with"
    confidence: float  # 0-1, used for ordering
    condition: str  # human-readable: "on_error_retry", "on_error_search_domain", etc.
    source_type: str = "static"  # "static" | "learned"
    allow_same_args: bool = False  # True = allow retry with identical args
    args_transform: Callable[[dict], dict] | None = None  # optional arg transform

    def transform_args(self, args: dict) -> dict:
        """Apply args_transform if set, otherwise return args unchanged."""
        if self.args_transform is not None:
            return self.args_transform(args)
        return args


class ToolRelationGraph:
    """Directed graph of tool relationships, indexed by source tool."""

    def __init__(self) -> None:
        self._edges: dict[str, list[ToolRelation]] = {}

    def add(self, relation: ToolRelation) -> None:
        self._edges.setdefault(relation.source, []).append(relation)

    def get_fallbacks(self, source: str) -> list[ToolRelation]:
        """Return fallback relations for *source*, sorted by confidence descending."""
        edges = self._edges.get(source, [])
        return sorted(
            [e for e in edges if e.relation == "fallback"],
            key=lambda e: e.confidence,
            reverse=True,
        )

    def get_relations(
        self, source: str, relation: str | None = None
    ) -> list[ToolRelation]:
        """Return all relations for *source*, optionally filtered by relation type."""
        edges = self._edges.get(source, [])
        if relation is not None:
            edges = [e for e in edges if e.relation == relation]
        return sorted(edges, key=lambda e: e.confidence, reverse=True)

    def has_fallback(self, source: str) -> bool:
        return any(e.relation == "fallback" for e in self._edges.get(source, []))

    @property
    def relation_count(self) -> int:
        return sum(len(v) for v in self._edges.values())


# ── Default arg transforms ──────────────────────────────────────────────────

def _simplify_query_transform(args: dict) -> dict:
    """Strip stopwords and keep max 3 keywords."""
    from .tool_executor import _simplify_query

    return {"query": _simplify_query(args.get("query", ""))}


def _extract_hostname_transform(args: dict) -> dict:
    """Extract hostname from URL for a web_search query."""
    url = args.get("url", "")
    hostname = urlparse(url).hostname or url
    return {"query": hostname}


def _stock_to_search_transform(args: dict) -> dict:
    """Convert stock args to a web_search query."""
    symbol = args.get("symbol", "")
    return {"query": f"{symbol} stock price today"}


# ── Default graph builder ────────────────────────────────────────────────────

def build_default_graph() -> ToolRelationGraph:
    """Build the default tool relation graph, migrating from TOOL_FALLBACK_MAP semantics."""
    g = ToolRelationGraph()

    # web_search: retry with simplified query
    g.add(ToolRelation(
        source="web_search",
        target="web_search",
        relation="fallback",
        confidence=0.7,
        condition="on_error_retry_refined",
        allow_same_args=False,
        args_transform=_simplify_query_transform,
    ))

    # web_fetch: retry with same args (transient errors)
    g.add(ToolRelation(
        source="web_fetch",
        target="web_fetch",
        relation="fallback",
        confidence=0.8,
        condition="on_error_retry",
        allow_same_args=True,
    ))
    # web_fetch: fall back to web_search with hostname
    g.add(ToolRelation(
        source="web_fetch",
        target="web_search",
        relation="fallback",
        confidence=0.5,
        condition="on_error_search_domain",
        allow_same_args=False,
        args_transform=_extract_hostname_transform,
    ))

    # stock: retry with same args (transient errors)
    g.add(ToolRelation(
        source="stock",
        target="stock",
        relation="fallback",
        confidence=0.8,
        condition="on_error_retry",
        allow_same_args=True,
    ))
    # stock: fall back to web_search
    g.add(ToolRelation(
        source="stock",
        target="web_search",
        relation="fallback",
        confidence=0.6,
        condition="on_error_search_price",
        allow_same_args=False,
        args_transform=_stock_to_search_transform,
    ))

    return g
