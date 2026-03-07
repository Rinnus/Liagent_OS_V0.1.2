"""Tests for the interest monitoring module (InterestStore + FactorResolver)."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from liagent.agent.interest import (
    InterestStore,
    Resolution,
    ResolvedFactor,
    build_coverage_summary,
    resolve_factor,
    resolve_factors,
    _frequency_to_seconds,
)


# ── FactorResolver tests ───────────────────────────────────────────────────

class TestResolveFactorClassification:
    """Verify deterministic source_hint → resolution mapping."""

    TOOLS = {"stock", "web_search", "screenshot"}

    def test_stock_price_executable(self):
        res, tool, rel = resolve_factor("stock_price", available_tools=self.TOOLS)
        assert res == Resolution.EXECUTABLE
        assert tool == "stock"
        assert rel == 1.0

    def test_stock_quote_executable(self):
        res, tool, rel = resolve_factor("stock_quote", available_tools=self.TOOLS)
        assert res == Resolution.EXECUTABLE
        assert tool == "stock"

    def test_news_search_proxy(self):
        res, tool, rel = resolve_factor("news_search", available_tools=self.TOOLS)
        assert res == Resolution.PROXY
        assert tool == "web_search"
        assert rel == 0.7

    def test_company_news_proxy(self):
        res, tool, rel = resolve_factor("company_news", available_tools=self.TOOLS)
        assert res == Resolution.PROXY
        assert tool == "web_search"

    def test_options_flow_blind(self):
        res, tool, rel = resolve_factor("options_flow", available_tools=self.TOOLS)
        assert res == Resolution.BLIND
        assert tool is None
        assert rel == 0.0

    def test_short_interest_blind(self):
        res, tool, rel = resolve_factor("short_interest", available_tools=self.TOOLS)
        assert res == Resolution.BLIND
        assert tool is None

    def test_analyst_rating_now_proxy(self):
        """analyst_rating moved from BLIND to PROXY."""
        res, tool, rel = resolve_factor("analyst_rating", available_tools=self.TOOLS)
        assert res == Resolution.PROXY
        assert tool == "web_search"
        assert rel == 0.7

    def test_formerly_blind_hints_now_proxy(self):
        """Hints that were BLIND but are publicly web-searchable are now PROXY."""
        for hint in (
            "insider_trading", "institutional_holdings",
            "credit_rating", "sec_filing", "patent_filing",
            "regulatory_action",
        ):
            res, tool, rel = resolve_factor(hint, available_tools=self.TOOLS)
            assert res == Resolution.PROXY, f"{hint} should be PROXY"
            assert tool == "web_search"

    def test_unknown_hint_proxied(self):
        """Unknown source_hints fall back to PROXY via web_search."""
        res, tool, rel = resolve_factor("something_unknown", available_tools=self.TOOLS)
        assert res == Resolution.PROXY
        assert tool == "web_search"

    def test_no_web_search_unknown_is_blind(self):
        """Without web_search, unknown hints become BLIND."""
        res, tool, rel = resolve_factor("something_unknown", available_tools={"stock"})
        assert res == Resolution.BLIND
        assert tool is None

    def test_stock_tool_missing_falls_through(self):
        """stock_price without stock tool → PROXY via web_search."""
        res, tool, rel = resolve_factor("stock_price", available_tools={"web_search"})
        assert res == Resolution.PROXY
        assert tool == "web_search"


class TestResolveFactors:
    """Test batch resolution of LLM-generated factor dicts."""

    TOOLS = {"stock", "web_search", "screenshot"}

    def test_mixed_factors(self):
        raw = [
            {"name": "AAPL Price", "source_hint": "stock_price", "entity": "AAPL",
             "frequency": "realtime", "weight": 1.0},
            {"name": "Supply Chain News", "source_hint": "company_news", "entity": "Foxconn",
             "frequency": "hourly", "weight": 0.7},
            {"name": "Options Flow", "source_hint": "options_flow", "entity": "AAPL",
             "frequency": "daily", "weight": 0.5},
        ]
        results = resolve_factors(raw, available_tools=self.TOOLS)
        assert len(results) == 3
        assert results[0].resolution == Resolution.EXECUTABLE
        assert results[0].bound_tool == "stock"
        assert results[1].resolution == Resolution.PROXY
        assert results[2].resolution == Resolution.BLIND

    def test_preserves_metadata(self):
        raw = [{"name": "Test", "source_hint": "stock_price", "entity": "TSLA",
                "frequency": "5min", "weight": 0.9}]
        results = resolve_factors(raw, available_tools=self.TOOLS)
        f = results[0]
        assert f.name == "Test"
        assert f.entity == "TSLA"
        assert f.frequency == "5min"
        assert f.weight == 0.9

    def test_empty_list(self):
        assert resolve_factors([], available_tools=self.TOOLS) == []

    def test_missing_fields_default(self):
        raw = [{"name": "Bare", "source_hint": "stock_price"}]
        results = resolve_factors(raw, available_tools=self.TOOLS)
        assert results[0].entity == ""
        assert results[0].frequency == ""
        assert results[0].weight == 1.0


# ── Frequency conversion ───────────────────────────────────────────────────

class TestFrequencyConversion:
    def test_known_frequencies(self):
        assert _frequency_to_seconds("realtime") == 30
        assert _frequency_to_seconds("hourly") == 3600
        assert _frequency_to_seconds("daily") == 86400
        assert _frequency_to_seconds("quarterly") == 7776000

    def test_case_insensitive(self):
        assert _frequency_to_seconds("HOURLY") == 3600
        assert _frequency_to_seconds("Daily") == 86400

    def test_unknown_defaults_300(self):
        assert _frequency_to_seconds("biweekly") == 300
        assert _frequency_to_seconds("") == 300
        assert _frequency_to_seconds(None) == 300


# ── InterestStore tests ─────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """Create an InterestStore with a temp database."""
    return InterestStore(db_path=tmp_path / "test.db")


def _make_factors():
    return [
        ResolvedFactor(name="AAPL Price", source_hint="stock_price", entity="AAPL",
                       frequency="realtime", resolution=Resolution.EXECUTABLE,
                       bound_tool="stock", reliability=1.0, weight=1.0),
        ResolvedFactor(name="Supply Chain News", source_hint="company_news", entity="Foxconn",
                       frequency="hourly", resolution=Resolution.PROXY,
                       bound_tool="web_search", reliability=0.7, weight=0.7),
        ResolvedFactor(name="Options Flow", source_hint="options_flow", entity="AAPL",
                       frequency="daily", resolution=Resolution.BLIND,
                       bound_tool=None, reliability=0.0, weight=0.5),
    ]


def _make_edges():
    return [
        {"from": "Supply Chain News", "to": "AAPL Price", "relation": "supply chain impact"},
    ]


class TestInterestStoreCRUD:
    def test_create_and_get(self, store):
        interest = store.create_interest(
            query="Watch AAPL, cost basis 142",
            intent="AAPL holding safety",
            context={"cost_basis": 142},
            factors=_make_factors(),
            edges=_make_edges(),
        )
        assert interest is not None
        assert interest["query"] == "Watch AAPL, cost basis 142"
        assert interest["intent"] == "AAPL holding safety"
        assert interest["context"] == {"cost_basis": 142}
        assert interest["status"] == "active"
        assert len(interest["factors"]) == 3
        assert len(interest["edges"]) == 1

    def test_coverage_ratio(self, store):
        interest = store.create_interest(
            query="test",
            factors=_make_factors(),  # 2 active (exe+proxy), 1 blind
        )
        # 2/3 ≈ 0.667
        assert abs(interest["coverage_ratio"] - 2 / 3) < 0.01

    def test_list_interests(self, store):
        store.create_interest(query="first")
        store.create_interest(query="second")
        all_ = store.list_interests()
        assert len(all_) == 2

    def test_list_excludes_archived(self, store):
        interest = store.create_interest(query="to archive")
        store.archive_interest(interest["id"])
        active = store.list_interests()
        assert len(active) == 0
        all_ = store.list_interests(include_archived=True)
        assert len(all_) == 1

    def test_pause_and_resume(self, store):
        interest = store.create_interest(query="test")
        iid = interest["id"]

        assert store.pause_interest(iid) is True
        assert store.get_interest(iid)["status"] == "paused"

        assert store.resume_interest(iid) is True
        assert store.get_interest(iid)["status"] == "active"

    def test_pause_nonexistent(self, store):
        assert store.pause_interest("nonexistent") is False

    def test_archive(self, store):
        interest = store.create_interest(query="test")
        assert store.archive_interest(interest["id"]) is True
        assert store.get_interest(interest["id"])["status"] == "archived"

    def test_get_nonexistent(self, store):
        assert store.get_interest("nope") is None

    def test_no_factors(self, store):
        interest = store.create_interest(query="bare")
        assert interest["factors"] == []
        assert interest["edges"] == []
        assert interest["coverage_ratio"] == 0.0

    def test_update_interest(self, store):
        interest = store.create_interest(query="test")
        updated = store.update_interest(
            interest["id"], discord_thread_id="123456",
        )
        assert updated["discord_thread_id"] == "123456"

    def test_factor_poll_interval(self, store):
        factors = [
            ResolvedFactor(name="RT", source_hint="stock_price", entity="X",
                           frequency="realtime", resolution=Resolution.EXECUTABLE,
                           bound_tool="stock", reliability=1.0, weight=1.0),
        ]
        interest = store.create_interest(query="test", factors=factors)
        assert interest["factors"][0]["poll_interval"] == 30


class TestBlindBacklog:
    def test_blind_factors_tracked(self, store):
        store.create_interest(query="test1", factors=_make_factors())
        backlog = store.get_blind_backlog()
        assert len(backlog) == 1
        assert backlog[0]["source_hint"] == "options_flow"
        assert backlog[0]["requested_count"] == 1

    def test_backlog_count_increments(self, store):
        store.create_interest(query="test1", factors=_make_factors())
        store.create_interest(query="test2", factors=_make_factors())
        backlog = store.get_blind_backlog()
        assert backlog[0]["requested_count"] == 2


# ── Coverage summary tests ──────────────────────────────────────────────────

class TestCoverageSummary:
    def test_build_summary(self, store):
        interest = store.create_interest(
            query="test",
            intent="AAPL safety",
            factors=_make_factors(),
        )
        summary = build_coverage_summary(interest)
        assert summary["interest_id"] == interest["id"]
        assert summary["intent"] == "AAPL safety"
        assert len(summary["executable"]) == 1
        assert len(summary["proxy"]) == 1
        assert len(summary["blind"]) == 1
        assert summary["total"] == 3
        assert summary["active"] == 2
        assert abs(summary["coverage_ratio"] - 2 / 3) < 0.01

    def test_empty_factors(self, store):
        interest = store.create_interest(query="bare")
        summary = build_coverage_summary(interest)
        assert summary["total"] == 0
        assert summary["active"] == 0
        assert summary["coverage_ratio"] == 0.0
