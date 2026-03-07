"""Tests for the anomaly detection module (Layer 1 of the attention system)."""

import asyncio
import unittest
from unittest.mock import AsyncMock

from liagent.agent.anomaly_detector import (
    ANOMALY_THRESHOLD,
    AnomalyDetector,
    evaluate_window,
    score_signal,
)


# ── score_signal tests ───────────────────────────────────────────────────

class TestScoreSignal(unittest.TestCase):
    def test_large_stock_drop(self):
        sig = {"delta": {"pct_change": -12.0}, "source_hint": "stock_price"}
        self.assertEqual(score_signal(sig), 2.0)

    def test_medium_stock_move(self):
        sig = {"delta": {"pct_change": 5.5}, "source_hint": "stock_price"}
        self.assertEqual(score_signal(sig), 1.5)

    def test_moderate_stock_move(self):
        sig = {"delta": {"pct_change": 3.2}, "source_hint": "stock_price"}
        self.assertEqual(score_signal(sig), 1.0)

    def test_threshold_stock_move(self):
        sig = {"delta": {"pct_change": 2.0}, "source_hint": "stock_price"}
        self.assertEqual(score_signal(sig), 0.6)

    def test_small_stock_move(self):
        sig = {"delta": {"pct_change": 1.5}, "source_hint": "stock_price"}
        self.assertEqual(score_signal(sig), 0.3)

    def test_content_changed(self):
        sig = {"delta": {"type": "content_changed"}, "source_hint": "company_news"}
        self.assertEqual(score_signal(sig), 0.5)

    def test_unknown_delta(self):
        sig = {"delta": {"foo": "bar"}, "source_hint": "unknown"}
        self.assertEqual(score_signal(sig), 0.2)

    def test_no_delta(self):
        sig = {"source_hint": "stock_price"}
        self.assertEqual(score_signal(sig), 0.2)

    def test_enriched_api_signal(self):
        """Enriched API delta uses severity*confidence*2 formula."""
        sig = {"delta": {"severity": 0.5, "confidence": 1.0, "source": "api"}}
        self.assertAlmostEqual(score_signal(sig), 1.0)

    def test_enriched_web_signal(self):
        """Enriched web delta score: 0.8 * 0.7 * 2 = 1.12."""
        sig = {"delta": {"severity": 0.8, "confidence": 0.7, "source": "web_enriched"}}
        self.assertAlmostEqual(score_signal(sig), 1.12)

    def test_enriched_score_capped_at_2(self):
        sig = {"delta": {"severity": 1.0, "confidence": 1.0, "source": "api"}}
        self.assertEqual(score_signal(sig), 2.0)


# ── evaluate_window tests ────────────────────────────────────────────────

class TestEvaluateWindow(unittest.TestCase):
    def _make_signal(
        self, factor_id="f1", factor_name="Price", source_hint="stock_price",
        pct_change=None, content_changed=False, interest_id="i1",
    ):
        delta = {}
        if pct_change is not None:
            delta = {"pct_change": pct_change, "prev_price": 100, "new_price": 100 + pct_change}
        elif content_changed:
            delta = {"type": "content_changed", "snippet": "test content changed"}
        return {
            "type": "signal",
            "interest_id": interest_id,
            "factor_id": factor_id,
            "factor_name": factor_name,
            "source_hint": source_hint,
            "delta": delta,
            "intent": "test intent",
            "discord_thread_id": "thread-123",
            "value": {},
        }

    def test_empty_window(self):
        self.assertIsNone(evaluate_window([]))

    def test_single_large_executable_triggers(self):
        """Single stock signal ≥5% triggers solo anomaly."""
        sig = self._make_signal(pct_change=-6.0)
        result = evaluate_window([sig])
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "anomaly")
        self.assertGreaterEqual(result["score"], ANOMALY_THRESHOLD)
        self.assertEqual(result["signal_count"], 1)

    def test_single_small_executable_no_trigger(self):
        """Single stock signal 2.5% doesn't trigger."""
        sig = self._make_signal(pct_change=2.5)
        result = evaluate_window([sig])
        self.assertIsNone(result)

    def test_single_unenriched_proxy_no_trigger(self):
        """Single unenriched web content change doesn't trigger anomaly."""
        sig = self._make_signal(
            factor_id="f2", source_hint="company_news",
            content_changed=True,
        )
        result = evaluate_window([sig])
        self.assertIsNone(result)

    def test_single_enriched_high_severity_web_triggers(self):
        """Single enriched web signal with high severity+confidence triggers solo."""
        sig = self._make_signal(
            factor_id="f2", source_hint="company_news",
        )
        sig["delta"] = {
            "severity": 0.9,
            "confidence": 0.7,
            "event_type": "regulatory",
            "key_fact": "SEC investigation",
            "source": "web_enriched",
            "raw_delta": {"type": "content_changed", "snippet": "SEC probe"},
        }
        result = evaluate_window([sig])
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["score"], ANOMALY_THRESHOLD)

    def test_single_enriched_low_severity_web_no_trigger(self):
        """Single enriched web signal with low severity doesn't trigger."""
        sig = self._make_signal(
            factor_id="f2", source_hint="company_news",
        )
        sig["delta"] = {
            "severity": 0.3,
            "confidence": 0.7,
            "event_type": "other",
            "key_fact": "Minor update",
            "source": "web_enriched",
            "raw_delta": {"type": "content_changed"},
        }
        result = evaluate_window([sig])
        self.assertIsNone(result)

    def test_multi_factor_corroboration(self):
        """Stock drop + news change together should trigger."""
        stock_sig = self._make_signal(
            factor_id="f1", source_hint="stock_price", pct_change=-3.5,
        )
        news_sig = self._make_signal(
            factor_id="f2", factor_name="News", source_hint="company_news",
            content_changed=True,
        )
        result = evaluate_window([stock_sig, news_sig])
        self.assertIsNotNone(result)
        self.assertEqual(result["factor_count"], 2)
        self.assertGreaterEqual(result["score"], ANOMALY_THRESHOLD)

    def test_corroboration_bonus_applied(self):
        """Multiple factors get a 20% bonus per additional factor."""
        stock_sig = self._make_signal(
            factor_id="f1", pct_change=-3.0,
        )
        news_sig = self._make_signal(
            factor_id="f2", factor_name="News", source_hint="company_news",
            content_changed=True,
        )
        result = evaluate_window([stock_sig, news_sig])
        # 1.0 (stock) + 0.5 (news) = 1.5, * 1.2 (2-factor bonus) = 1.8
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["score"], 1.8, places=1)

    def test_duplicate_factor_uses_best_score(self):
        """Multiple signals from same factor: only best score counts."""
        sig1 = self._make_signal(factor_id="f1", pct_change=2.0)  # 0.6
        sig2 = self._make_signal(factor_id="f1", pct_change=3.0)  # 1.0
        result = evaluate_window([sig1, sig2])
        # Single factor, best = 1.0, no corroboration bonus, below threshold
        self.assertIsNone(result)

    def test_summary_includes_factor_names(self):
        sig = self._make_signal(factor_name="AAPL Price", pct_change=-7.0)
        result = evaluate_window([sig])
        self.assertIsNotNone(result)
        self.assertIn("AAPL Price", result["summary"])
        self.assertIn("-7.0%", result["summary"])

    def test_summary_content_change(self):
        stock_sig = self._make_signal(
            factor_id="f1", factor_name="AAPL Price", pct_change=-5.0,
        )
        news_sig = self._make_signal(
            factor_id="f2", factor_name="AAPL News",
            source_hint="company_news", content_changed=True,
        )
        result = evaluate_window([stock_sig, news_sig])
        self.assertIsNotNone(result)
        self.assertIn("AAPL Price", result["summary"])
        self.assertIn("AAPL News", result["summary"])
        self.assertIn("content changed", result["summary"])

    def test_preserves_metadata(self):
        sig = self._make_signal(pct_change=-8.0)
        result = evaluate_window([sig])
        self.assertEqual(result["interest_id"], "i1")
        self.assertEqual(result["intent"], "test intent")
        self.assertEqual(result["discord_thread_id"], "thread-123")

    def test_three_factor_bonus(self):
        """Three distinct factors get 40% bonus (2 * 20%)."""
        s1 = self._make_signal(factor_id="f1", pct_change=-2.5)  # 0.6
        s2 = self._make_signal(
            factor_id="f2", factor_name="News",
            source_hint="company_news", content_changed=True,
        )  # 0.5
        s3 = self._make_signal(
            factor_id="f3", factor_name="Macro",
            source_hint="macro_indicator", content_changed=True,
        )  # 0.5
        result = evaluate_window([s1, s2, s3])
        # (0.6 + 0.5 + 0.5) * 1.4 = 2.24
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["score"], 2.24, places=1)

    def test_enriched_summary_uses_key_fact(self):
        """Summary includes key_fact from enriched delta."""
        sig = self._make_signal(factor_name="AAPL Price", pct_change=-7.0)
        sig["delta"] = {
            "severity": 1.0,
            "confidence": 1.0,
            "key_fact": "AAPL fell 7.0% on earnings miss",
            "source": "api",
            "raw_delta": {"pct_change": -7.0},
        }
        result = evaluate_window([sig])
        self.assertIsNotNone(result)
        self.assertIn("earnings miss", result["summary"])


# ── AnomalyDetector integration tests ───────────────────────────────────

class TestAnomalyDetectorIngest(unittest.IsolatedAsyncioTestCase):
    def _make_signal(self, interest_id="i1", factor_id="f1",
                     source_hint="stock_price", pct_change=-6.0):
        delta = {"pct_change": pct_change, "prev_price": 100, "new_price": 100 + pct_change}
        return {
            "type": "signal",
            "interest_id": interest_id,
            "factor_id": factor_id,
            "factor_name": "Test Factor",
            "source_hint": source_hint,
            "delta": delta,
            "intent": "test",
            "discord_thread_id": "thread-1",
            "value": {},
        }

    async def test_urgent_signal_triggers_immediate_flush(self):
        """Large stock move (≥5%) flushes immediately without waiting."""
        on_anomaly = AsyncMock()
        detector = AnomalyDetector(on_anomaly=on_anomaly, window_seconds=300)

        sig = self._make_signal(pct_change=-7.0)
        await detector.ingest(sig)

        # Should have been flushed immediately
        on_anomaly.assert_called_once()
        anomaly = on_anomaly.call_args[0][0]
        self.assertEqual(anomaly["type"], "anomaly")
        self.assertEqual(anomaly["interest_id"], "i1")

        await detector.shutdown()

    async def test_non_urgent_signal_buffers(self):
        """Small stock move buffers and doesn't fire immediately."""
        on_anomaly = AsyncMock()
        on_passthrough = AsyncMock()
        detector = AnomalyDetector(
            on_anomaly=on_anomaly,
            on_signal_passthrough=on_passthrough,
            window_seconds=0.1,  # short window for testing
        )

        sig = self._make_signal(pct_change=2.5)  # below solo threshold
        await detector.ingest(sig)

        # Not yet flushed
        on_anomaly.assert_not_called()
        on_passthrough.assert_not_called()

        # Wait for window to close
        await asyncio.sleep(0.2)

        # Should have passed through (below anomaly threshold)
        on_passthrough.assert_called_once()
        on_anomaly.assert_not_called()

        await detector.shutdown()

    async def test_multi_signal_anomaly(self):
        """Two corroborating signals within window trigger anomaly."""
        on_anomaly = AsyncMock()
        detector = AnomalyDetector(
            on_anomaly=on_anomaly,
            window_seconds=0.2,
        )

        # Stock signal + news signal from same interest
        stock = self._make_signal(factor_id="f1", pct_change=-3.5)
        news = {
            **self._make_signal(factor_id="f2", source_hint="company_news"),
            "delta": {"type": "content_changed", "snippet": "bad news"},
            "factor_name": "News",
        }

        await detector.ingest(stock)
        await detector.ingest(news)

        # Neither is urgent alone, wait for window
        await asyncio.sleep(0.3)

        on_anomaly.assert_called_once()
        anomaly = on_anomaly.call_args[0][0]
        self.assertEqual(anomaly["factor_count"], 2)

        await detector.shutdown()

    async def test_different_interests_independent(self):
        """Signals from different interests don't mix."""
        on_anomaly = AsyncMock()
        on_passthrough = AsyncMock()
        detector = AnomalyDetector(
            on_anomaly=on_anomaly,
            on_signal_passthrough=on_passthrough,
            window_seconds=0.1,
        )

        sig1 = self._make_signal(interest_id="i1", pct_change=2.5)
        sig2 = self._make_signal(interest_id="i2", pct_change=2.5)

        await detector.ingest(sig1)
        await detector.ingest(sig2)

        await asyncio.sleep(0.2)

        # Each should pass through individually, no anomaly
        on_anomaly.assert_not_called()
        self.assertEqual(on_passthrough.call_count, 2)

        await detector.shutdown()

    async def test_enriched_urgent_signal_triggers_immediate(self):
        """Enriched signal with severity ≥ 0.85 flushes immediately."""
        on_anomaly = AsyncMock()
        detector = AnomalyDetector(on_anomaly=on_anomaly, window_seconds=300)

        sig = self._make_signal(pct_change=-3.0)  # below legacy urgent threshold
        sig["delta"] = {
            "severity": 0.9,
            "confidence": 0.7,
            "key_fact": "Major event",
            "source": "web_enriched",
            "raw_delta": {"type": "content_changed"},
        }
        await detector.ingest(sig)
        # Should flush immediately due to high severity
        on_anomaly.assert_called_once()
        await detector.shutdown()

    async def test_shutdown_flushes_buffers(self):
        """Shutdown flushes all pending buffers."""
        on_passthrough = AsyncMock()
        detector = AnomalyDetector(
            on_signal_passthrough=on_passthrough,
            window_seconds=999,  # long window
        )

        sig = self._make_signal(pct_change=2.5)
        await detector.ingest(sig)

        # Nothing yet
        on_passthrough.assert_not_called()

        await detector.shutdown()

        # Should have flushed
        on_passthrough.assert_called_once()
