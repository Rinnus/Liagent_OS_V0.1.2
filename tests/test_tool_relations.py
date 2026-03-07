"""Tests for tool_relations — E2 understanding layer."""

import pytest

from liagent.agent.tool_relations import (
    ToolRelation,
    ToolRelationGraph,
    build_default_graph,
    _simplify_query_transform,
    _extract_hostname_transform,
    _stock_to_search_transform,
)


# ── ToolRelation ────────────────────────────────────────────────────────────


class TestToolRelation:

    def test_frozen(self):
        rel = ToolRelation("a", "b", "fallback", 0.5, "test")
        with pytest.raises(AttributeError):
            rel.source = "c"  # type: ignore

    def test_transform_args_with_fn(self):
        rel = ToolRelation("a", "b", "fallback", 0.5, "test",
                           args_transform=lambda a: {"q": a.get("x", "")})
        assert rel.transform_args({"x": "hello"}) == {"q": "hello"}

    def test_transform_args_identity(self):
        rel = ToolRelation("a", "b", "fallback", 0.5, "test")
        args = {"x": 1}
        assert rel.transform_args(args) is args

    def test_allow_same_args_default_false(self):
        rel = ToolRelation("a", "b", "fallback", 0.5, "test")
        assert rel.allow_same_args is False

    def test_source_type_default(self):
        rel = ToolRelation("a", "b", "fallback", 0.5, "test")
        assert rel.source_type == "static"


# ── ToolRelationGraph ───────────────────────────────────────────────────────


class TestToolRelationGraph:

    def test_add_and_get_fallbacks(self):
        g = ToolRelationGraph()
        r1 = ToolRelation("a", "b", "fallback", 0.5, "test1")
        r2 = ToolRelation("a", "c", "fallback", 0.8, "test2")
        g.add(r1)
        g.add(r2)
        fallbacks = g.get_fallbacks("a")
        assert len(fallbacks) == 2
        # Sorted by confidence desc
        assert fallbacks[0].confidence >= fallbacks[1].confidence
        assert fallbacks[0].target == "c"

    def test_get_fallbacks_empty(self):
        g = ToolRelationGraph()
        assert g.get_fallbacks("nonexistent") == []

    def test_has_fallback(self):
        g = ToolRelationGraph()
        g.add(ToolRelation("a", "b", "fallback", 0.5, "test"))
        assert g.has_fallback("a")
        assert not g.has_fallback("b")

    def test_get_relations_filter(self):
        g = ToolRelationGraph()
        g.add(ToolRelation("a", "b", "fallback", 0.5, "test"))
        g.add(ToolRelation("a", "c", "equivalent", 0.9, "test2"))
        assert len(g.get_relations("a")) == 2
        assert len(g.get_relations("a", relation="fallback")) == 1
        assert len(g.get_relations("a", relation="equivalent")) == 1
        assert len(g.get_relations("a", relation="compose_with")) == 0

    def test_relation_count(self):
        g = ToolRelationGraph()
        assert g.relation_count == 0
        g.add(ToolRelation("a", "b", "fallback", 0.5, "t"))
        g.add(ToolRelation("a", "c", "fallback", 0.8, "t"))
        g.add(ToolRelation("x", "y", "equivalent", 0.9, "t"))
        assert g.relation_count == 3

    def test_non_fallback_not_in_get_fallbacks(self):
        g = ToolRelationGraph()
        g.add(ToolRelation("a", "b", "equivalent", 0.9, "test"))
        assert g.get_fallbacks("a") == []
        assert g.has_fallback("a") is False


# ── Default graph ───────────────────────────────────────────────────────────


class TestBuildDefaultGraph:

    def test_graph_not_empty(self):
        g = build_default_graph()
        assert g.relation_count > 0

    def test_web_search_has_self_retry(self):
        g = build_default_graph()
        fallbacks = g.get_fallbacks("web_search")
        self_retry = [r for r in fallbacks if r.target == "web_search"]
        assert len(self_retry) == 1
        assert self_retry[0].allow_same_args is False

    def test_web_fetch_has_self_retry_and_fallback(self):
        g = build_default_graph()
        fallbacks = g.get_fallbacks("web_fetch")
        targets = {r.target for r in fallbacks}
        assert "web_fetch" in targets
        assert "web_search" in targets
        # Self-retry allows same args
        self_retry = [r for r in fallbacks if r.target == "web_fetch"][0]
        assert self_retry.allow_same_args is True

    def test_stock_has_self_retry_and_fallback(self):
        g = build_default_graph()
        fallbacks = g.get_fallbacks("stock")
        targets = {r.target for r in fallbacks}
        assert "stock" in targets
        assert "web_search" in targets
        self_retry = [r for r in fallbacks if r.target == "stock"][0]
        assert self_retry.allow_same_args is True

    def test_unknown_tool_no_fallback(self):
        g = build_default_graph()
        assert not g.has_fallback("unknown_tool")

    def test_confidence_ordering(self):
        g = build_default_graph()
        for source in ("web_fetch", "stock"):
            fallbacks = g.get_fallbacks(source)
            for i in range(len(fallbacks) - 1):
                assert fallbacks[i].confidence >= fallbacks[i + 1].confidence


# ── Arg transform functions ─────────────────────────────────────────────────


class TestArgTransforms:

    def test_simplify_query_transform(self):
        result = _simplify_query_transform({"query": "what is the stock price of AAPL today"})
        assert "query" in result
        # Should be simplified (stopwords removed, max 3 words)
        words = result["query"].split()
        assert len(words) <= 3

    def test_extract_hostname_transform(self):
        result = _extract_hostname_transform({"url": "https://finance.yahoo.com/quote/AAPL"})
        assert result == {"query": "finance.yahoo.com"}

    def test_extract_hostname_no_url(self):
        result = _extract_hostname_transform({"url": ""})
        assert "query" in result

    def test_stock_to_search_transform(self):
        result = _stock_to_search_transform({"symbol": "GOOG"})
        assert "GOOG" in result["query"]
        assert "stock" in result["query"].lower()
