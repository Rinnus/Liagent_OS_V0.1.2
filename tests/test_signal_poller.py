"""Tests for the signal polling module (Layer 0 of the attention system)."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from liagent.agent.signal_poller import (
    SignalPoller,
    compute_delta,
    fetch_stock_quote,
    fetch_web_signal,
    is_significant,
)
from liagent.agent.interest import InterestStore, Resolution, ResolvedFactor


# ── compute_delta tests ───────────────────────────────────────────────────

class TestComputeDelta:
    """Test change detection logic."""

    def test_first_poll_returns_none(self):
        assert compute_delta("stock_price", None, {"price": 150}) is None

    def test_invalid_prev_json(self):
        assert compute_delta("stock_price", "not-json{", {"price": 150}) is None

    def test_stock_price_increase(self):
        prev = json.dumps({"price": 100.0})
        delta = compute_delta("stock_price", prev, {"price": 105.0})
        assert delta is not None
        assert delta["abs_change"] == 5.0
        assert delta["pct_change"] == 5.0
        assert delta["prev_price"] == 100.0
        assert delta["new_price"] == 105.0

    def test_stock_price_decrease(self):
        prev = json.dumps({"price": 200.0})
        delta = compute_delta("stock_quote", prev, {"price": 190.0})
        assert delta["abs_change"] == -10.0
        assert delta["pct_change"] == -5.0

    def test_stock_no_change(self):
        prev = json.dumps({"price": 100.0})
        delta = compute_delta("stock_price", prev, {"price": 100.0})
        assert delta is not None
        assert delta["pct_change"] == 0.0

    def test_stock_missing_price_returns_none(self):
        prev = json.dumps({"price": 100.0})
        assert compute_delta("stock_price", prev, {}) is None

    def test_stock_prev_zero_price_returns_none(self):
        prev = json.dumps({"price": 0})
        assert compute_delta("stock_price", prev, {"price": 50}) is None

    def test_market_cap_hint_uses_stock_logic(self):
        prev = json.dumps({"price": 100.0})
        delta = compute_delta("market_cap", prev, {"price": 110.0})
        assert delta["pct_change"] == 10.0

    def test_company_profile_hint_uses_stock_logic(self):
        prev = json.dumps({"price": 50.0})
        delta = compute_delta("company_profile", prev, {"price": 55.0})
        assert delta["pct_change"] == 10.0

    def test_web_content_changed(self):
        prev = json.dumps({"content_hash": "abc123", "snippet": "old"})
        new = {"content_hash": "def456", "snippet": "new content"}
        delta = compute_delta("news_search", prev, new)
        assert delta is not None
        assert delta["type"] == "content_changed"
        assert delta["prev_hash"] == "abc123"
        assert delta["new_hash"] == "def456"
        assert delta["snippet"] == "new content"

    def test_web_content_unchanged(self):
        prev = json.dumps({"content_hash": "same123"})
        new = {"content_hash": "same123"}
        assert compute_delta("news_search", prev, new) is None

    def test_web_missing_hash(self):
        prev = json.dumps({})
        assert compute_delta("news_search", prev, {"content_hash": "abc"}) is None

    def test_stock_rounding(self):
        prev = json.dumps({"price": 142.35})
        delta = compute_delta("stock_price", prev, {"price": 145.78})
        assert delta["abs_change"] == 3.43
        assert delta["pct_change"] == 2.41

    def test_stock_hint_web_fallback_uses_hash(self):
        """Stock hint with fetch_mode='web' uses hash comparison, not price."""
        prev = json.dumps({"content_hash": "old123", "snippet": "old"})
        new = {"content_hash": "new456", "snippet": "new stuff"}
        delta = compute_delta("stock_price", prev, new, fetch_mode="web")
        assert delta is not None
        assert delta["type"] == "content_changed"

    def test_fetch_mode_api_forces_numeric(self):
        prev = json.dumps({"price": 100.0})
        delta = compute_delta("news_search", prev, {"price": 110.0}, fetch_mode="api")
        assert delta is not None
        assert delta["pct_change"] == 10.0

    def test_auto_detect_price_key(self):
        """Without fetch_mode, presence of 'price' key triggers numeric."""
        prev = json.dumps({"price": 100.0})
        delta = compute_delta("some_unknown_hint", prev, {"price": 105.0})
        assert delta is not None
        assert delta["pct_change"] == 5.0

    def test_content_changed_includes_full_content(self):
        prev = json.dumps({"content_hash": "abc"})
        new = {"content_hash": "def", "snippet": "s", "full_content": "full text here"}
        delta = compute_delta("news_search", prev, new)
        assert delta["full_content"] == "full text here"


# ── is_significant tests ─────────────────────────────────────────────────

class TestIsSignificant:
    def test_none_delta(self):
        assert is_significant("stock_price", None) is False

    def test_stock_below_threshold(self):
        assert is_significant("stock_price", {"pct_change": 1.5}) is False

    def test_stock_at_threshold(self):
        assert is_significant("stock_price", {"pct_change": 2.0}) is True

    def test_stock_above_threshold(self):
        assert is_significant("stock_price", {"pct_change": -3.5}) is True

    def test_content_changed(self):
        assert is_significant("news_search", {"type": "content_changed"}) is True

    def test_unknown_delta_type(self):
        assert is_significant("unknown", {"foo": "bar"}) is False


# ── Async tests using IsolatedAsyncioTestCase ────────────────────────────

class _TempStoreTestCase(unittest.IsolatedAsyncioTestCase):
    """Base with a temp InterestStore."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.store = InterestStore(db_path=Path(self._tmpdir) / "test.db")

    def _make_interest(self):
        factors = [
            ResolvedFactor(
                name="AAPL Price", source_hint="stock_price", entity="AAPL",
                frequency="realtime", resolution=Resolution.EXECUTABLE,
                bound_tool="stock", reliability=1.0, weight=1.0,
            ),
            ResolvedFactor(
                name="AAPL News", source_hint="company_news", entity="AAPL",
                frequency="hourly", resolution=Resolution.PROXY,
                bound_tool="web_search", reliability=0.7, weight=0.7,
            ),
            ResolvedFactor(
                name="Options Flow", source_hint="options_flow", entity="AAPL",
                frequency="daily", resolution=Resolution.BLIND,
                bound_tool=None, reliability=0.0, weight=0.5,
            ),
        ]
        return self.store.create_interest(
            query="Watch AAPL", intent="AAPL safety", factors=factors,
        )


class TestFetchStockQuote(unittest.IsolatedAsyncioTestCase):
    async def test_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await fetch_stock_quote("AAPL")
            self.assertIsNone(result)

    async def test_successful_quote(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "c": 150.25, "d": 2.5, "dp": 1.69,
            "o": 148.0, "h": 151.0, "l": 147.5, "pc": 147.75,
        }
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await fetch_stock_quote("AAPL")

        self.assertEqual(result["price"], 150.25)
        self.assertEqual(result["change"], 2.5)
        self.assertEqual(result["change_pct"], 1.69)
        self.assertEqual(result["prev_close"], 147.75)

    async def test_null_price_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"c": None}
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await fetch_stock_quote("INVALID")
        self.assertIsNone(result)

    async def test_api_error_returns_none(self):
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("timeout")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict("os.environ", {"FINNHUB_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await fetch_stock_quote("AAPL")
        self.assertIsNone(result)


class TestFetchWebSignal(unittest.IsolatedAsyncioTestCase):
    async def test_successful_search(self):
        mock_results = [
            {"title": "AAPL News", "body": "Apple stock rises today"},
            {"title": "Market Update", "body": "Tech stocks rally"},
        ]

        mock_ddgs_inst = MagicMock()
        mock_ddgs_inst.__enter__ = MagicMock(return_value=mock_ddgs_inst)
        mock_ddgs_inst.__exit__ = MagicMock(return_value=False)
        mock_ddgs_inst.text.return_value = mock_results

        mock_cls = MagicMock(return_value=mock_ddgs_inst)
        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=mock_cls)}):
            result = await fetch_web_signal("AAPL news")

        self.assertIsNotNone(result)
        self.assertIn("content_hash", result)
        self.assertEqual(len(result["content_hash"]), 16)
        self.assertEqual(result["result_count"], 2)
        self.assertIn("Apple stock", result["snippet"])

    async def test_empty_results(self):
        mock_ddgs_inst = MagicMock()
        mock_ddgs_inst.__enter__ = MagicMock(return_value=mock_ddgs_inst)
        mock_ddgs_inst.__exit__ = MagicMock(return_value=False)
        mock_ddgs_inst.text.return_value = []

        mock_cls = MagicMock(return_value=mock_ddgs_inst)
        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=mock_cls)}):
            result = await fetch_web_signal("nonexistent query")
        self.assertIsNone(result)


class TestSignalPollerLifecycle(_TempStoreTestCase):
    async def test_start_creates_tasks_for_pollable(self):
        self._make_interest()
        poller = SignalPoller(self.store)
        poller._poll_loop = AsyncMock()

        await poller.start()
        # EXECUTABLE + PROXY = 2 tasks, not BLIND
        self.assertEqual(len(poller._poll_tasks), 2)
        await poller.stop()

    async def test_stop_clears_tasks(self):
        self._make_interest()
        poller = SignalPoller(self.store)
        poller._poll_loop = AsyncMock()

        await poller.start()
        self.assertGreater(len(poller._poll_tasks), 0)

        await poller.stop()
        self.assertEqual(len(poller._poll_tasks), 0)
        self.assertTrue(poller._stopped)

    async def test_add_interest_starts_polling(self):
        poller = SignalPoller(self.store)
        interest = self._make_interest()

        with patch.object(poller, '_poll_loop', new_callable=AsyncMock):
            poller.add_interest(interest)
            self.assertEqual(len(poller._poll_tasks), 2)
        await poller.stop()

    async def test_add_interest_skips_blind(self):
        poller = SignalPoller(self.store)
        blind_only = {
            "factors": [{"id": "f1", "resolution": "blind", "name": "X"}],
        }
        poller.add_interest(blind_only)
        self.assertEqual(len(poller._poll_tasks), 0)

    async def test_duplicate_factor_not_started_twice(self):
        poller = SignalPoller(self.store)
        interest = self._make_interest()

        with patch.object(poller, '_poll_loop', new_callable=AsyncMock):
            poller.add_interest(interest)
            count = len(poller._poll_tasks)
            poller.add_interest(interest)
            self.assertEqual(len(poller._poll_tasks), count)
        await poller.stop()

    async def test_remove_interest_cancels_tasks(self):
        poller = SignalPoller(self.store)
        interest = self._make_interest()

        with patch.object(poller, '_poll_loop', new_callable=AsyncMock):
            poller.add_interest(interest)
            self.assertEqual(len(poller._poll_tasks), 2)
            poller.remove_interest(interest["id"])
            self.assertEqual(len(poller._poll_tasks), 0)
        await poller.stop()


class TestFetchRouting(_TempStoreTestCase):
    async def test_stock_hint_routes_to_stock_quote(self):
        poller = SignalPoller(self.store)
        factor = {"entity": "AAPL", "name": "Price"}

        with patch("liagent.agent.signal_poller.fetch_stock_quote", new_callable=AsyncMock) as m:
            m.return_value = {"price": 150}
            value, mode = await poller._fetch(factor, "stock_price", None)
            m.assert_called_once_with("AAPL")
            self.assertEqual(value, {"price": 150})
            self.assertEqual(mode, "api")

    async def test_stock_hint_falls_back_to_web(self):
        """Stock API failure → web search fallback with mode='web'."""
        poller = SignalPoller(self.store)
        factor = {"entity": "AAPL", "name": "Price"}

        with patch("liagent.agent.signal_poller.fetch_stock_quote", new_callable=AsyncMock) as stock_m, \
             patch("liagent.agent.signal_poller.fetch_web_signal", new_callable=AsyncMock) as web_m:
            stock_m.return_value = None  # API unavailable
            web_m.return_value = {"content_hash": "fallback123", "snippet": "AAPL news"}
            value, mode = await poller._fetch(factor, "stock_price", None)
            stock_m.assert_called_once_with("AAPL")
            web_m.assert_called_once()
            self.assertEqual(mode, "web")
            self.assertIn("content_hash", value)

    async def test_news_hint_routes_to_web_signal(self):
        poller = SignalPoller(self.store)
        factor = {"entity": "AAPL", "name": "News"}

        with patch("liagent.agent.signal_poller.fetch_web_signal", new_callable=AsyncMock) as m:
            m.return_value = {"content_hash": "abc"}
            value, mode = await poller._fetch(factor, "company_news", None)
            m.assert_called_once_with("AAPL News")
            self.assertEqual(value, {"content_hash": "abc"})
            self.assertEqual(mode, "web")

    async def test_web_search_bound_tool_routes_to_web(self):
        poller = SignalPoller(self.store)
        factor = {"entity": "Tesla", "name": "Updates"}

        with patch("liagent.agent.signal_poller.fetch_web_signal", new_callable=AsyncMock) as m:
            m.return_value = {"content_hash": "xyz"}
            value, mode = await poller._fetch(factor, "unknown_hint", "web_search")
            m.assert_called_once_with("Tesla Updates")
            self.assertEqual(mode, "web")

    async def test_unknown_hint_no_tool_returns_none(self):
        poller = SignalPoller(self.store)
        factor = {"entity": "X", "name": "Y"}
        value, mode = await poller._fetch(factor, "unknown_hint", None)
        self.assertIsNone(value)
        self.assertIsNone(mode)

    async def test_formerly_blind_hints_now_web_searchable(self):
        """Hints moved from BLIND to PROXY should route to web search."""
        poller = SignalPoller(self.store)
        for hint in ("insider_trading", "analyst_rating", "sec_filing"):
            factor = {"entity": "AAPL", "name": hint}
            with patch("liagent.agent.signal_poller.fetch_web_signal", new_callable=AsyncMock) as m:
                m.return_value = {"content_hash": "x"}
                value, mode = await poller._fetch(factor, hint, None)
                self.assertEqual(mode, "web", f"{hint} should route to web")


class TestPollLoopIntegration(_TempStoreTestCase):
    async def test_signal_callback_fires_on_significant_change(self):
        """End-to-end: significant stock change → callback fires + DB recorded."""
        # Create a real interest so factor IDs exist in DB
        interest = self._make_interest()
        stock_factor = [f for f in interest["factors"] if f["source_hint"] == "stock_price"][0]

        callback = AsyncMock()
        poller = SignalPoller(self.store, on_signal=callback)

        # Set a previous value for the factor
        prev_value = {"price": 100.0}
        prev_json = json.dumps(prev_value)
        self.store.update_factor_value(stock_factor["id"], prev_json, "2026-02-25T00:00:00Z")

        # Simulate new poll: 5% increase
        value = {"price": 105.0}
        value_json = json.dumps(value)
        delta = compute_delta("stock_price", prev_json, value)

        self.assertTrue(is_significant("stock_price", delta))

        delta_json = json.dumps(delta)
        self.store.record_signal(
            factor_id=stock_factor["id"],
            interest_id=interest["id"],
            value_json=value_json,
            delta_json=delta_json,
        )
        await poller.on_signal({
            "type": "signal",
            "interest_id": interest["id"],
            "factor_id": stock_factor["id"],
            "delta": delta,
        })

        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        self.assertEqual(call_args["type"], "signal")
        self.assertEqual(call_args["delta"]["pct_change"], 5.0)

        signals = self.store.get_recent_signals(interest["id"], limit=10)
        self.assertEqual(len(signals), 1)

    async def test_no_callback_on_insignificant_change(self):
        prev = json.dumps({"price": 100.0})
        new = {"price": 100.5}  # 0.5% — below 2% threshold
        delta = compute_delta("stock_price", prev, new)
        self.assertFalse(is_significant("stock_price", delta))

    async def test_first_poll_no_signal(self):
        delta = compute_delta("stock_price", None, {"price": 150.0})
        self.assertIsNone(delta)
        self.assertFalse(is_significant("stock_price", delta))
