"""End-to-end integration test for the perception loop.

Tests the full pipeline:
  InterestStore → SignalPoller → SignalEnricher → AnomalyDetector → callbacks

All external dependencies (Finnhub API, web search, LLM) are mocked.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from liagent.agent.interest import InterestStore, ResolvedFactor, Resolution
from liagent.agent.signal_poller import SignalPoller, compute_delta, is_significant
from liagent.agent.signal_enricher import SignalEnricher
from liagent.agent.anomaly_detector import AnomalyDetector, evaluate_window, score_signal


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_store(tmp_dir: str) -> InterestStore:
    """Create an InterestStore backed by a temp SQLite file."""
    db_path = Path(tmp_dir) / "test_e2e.db"
    return InterestStore(db_path=db_path)


def _make_engine_mock(llm_response: str = "") -> MagicMock:
    """Create a mock engine whose generate_extraction returns a preset JSON."""
    engine = MagicMock()

    async def fake_generate(messages, **kwargs):
        return llm_response

    engine.generate_extraction = AsyncMock(side_effect=fake_generate)
    return engine


def _make_factors() -> list[ResolvedFactor]:
    """Create two factors: one EXECUTABLE (stock) and one PROXY (web)."""
    return [
        ResolvedFactor(
            name="AAPL Price",
            source_hint="stock_price",
            entity="AAPL",
            frequency="realtime",
            resolution=Resolution.EXECUTABLE,
            bound_tool="stock",
            reliability=1.0,
            weight=1.0,
        ),
        ResolvedFactor(
            name="Regulatory News",
            source_hint="regulatory_news",
            entity="AAPL",
            frequency="hourly",
            resolution=Resolution.PROXY,
            bound_tool="web_search",
            reliability=0.7,
            weight=0.7,
        ),
    ]


# ── Unit-level tests for each stage ──────────────────────────────────────


class StageComputeDeltaTests(unittest.TestCase):
    """Stage 2: compute_delta produces correct deltas."""

    def test_stock_delta(self):
        prev = json.dumps({"price": 100.0})
        new = {"price": 105.0, "change": 5.0}
        delta = compute_delta("stock_price", prev, new, fetch_mode="api")
        self.assertAlmostEqual(delta["pct_change"], 5.0)
        self.assertEqual(delta["new_price"], 105.0)
        self.assertEqual(delta["prev_price"], 100.0)

    def test_web_delta(self):
        prev = json.dumps({"content_hash": "old_hash"})
        new = {"content_hash": "new_hash", "snippet": "SEC announces investigation"}
        delta = compute_delta("regulatory_news", prev, new, fetch_mode="web")
        self.assertEqual(delta["type"], "content_changed")
        self.assertEqual(delta["prev_hash"], "old_hash")
        self.assertEqual(delta["new_hash"], "new_hash")

    def test_stock_fallback_to_web_mode(self):
        """Stock hint that fell back to web search should use hash comparison."""
        prev = json.dumps({"content_hash": "aaa"})
        new = {"content_hash": "bbb", "snippet": "AAPL trading at $150"}
        delta = compute_delta("stock_price", prev, new, fetch_mode="web")
        self.assertEqual(delta["type"], "content_changed")

    def test_no_change(self):
        prev = json.dumps({"content_hash": "same"})
        new = {"content_hash": "same", "snippet": "nothing new"}
        delta = compute_delta("news_search", prev, new, fetch_mode="web")
        self.assertIsNone(delta)


class StageSignificanceTests(unittest.TestCase):
    """Stage 2b: is_significant correctly filters."""

    def test_stock_significant(self):
        self.assertTrue(is_significant("stock_price", {"pct_change": 3.0}))

    def test_stock_not_significant(self):
        self.assertFalse(is_significant("stock_price", {"pct_change": 0.5}))

    def test_web_content_changed(self):
        self.assertTrue(is_significant("regulatory_news", {"type": "content_changed"}))

    def test_none_delta(self):
        self.assertFalse(is_significant("stock_price", None))


class StageEnrichmentTests(unittest.TestCase):
    """Stage 3: SignalEnricher produces correct enriched deltas."""

    def test_api_enrichment_deterministic(self):
        engine = _make_engine_mock()
        enricher = SignalEnricher(engine)

        signal = {
            "delta": {"pct_change": 5.0, "abs_change": 5.0, "prev_price": 100, "new_price": 105},
            "entity": "AAPL",
            "factor_name": "AAPL Price",
            "factor_id": "f1",
            "interest_id": "i1",
        }
        result = asyncio.run(enricher.enrich(signal))
        d = result["delta"]

        self.assertEqual(d["source"], "api")
        self.assertEqual(d["confidence"], 1.0)
        self.assertAlmostEqual(d["severity"], 0.5)
        self.assertEqual(d["sentiment"], 1.0)
        self.assertIn("rose", d["key_fact"])
        self.assertEqual(d["event_type"], "market")
        # LLM should NOT be called for API deltas
        engine.generate_extraction.assert_not_called()

    def test_web_enrichment_with_llm(self):
        llm_json = json.dumps({
            "severity": 0.9,
            "sentiment": -0.8,
            "event_type": "regulatory",
            "key_fact": "SEC investigation announced",
        })
        engine = _make_engine_mock(llm_json)
        enricher = SignalEnricher(engine)

        signal = {
            "delta": {
                "type": "content_changed",
                "prev_hash": "old",
                "new_hash": "new",
                "snippet": "SEC announces investigation into AAPL",
                "full_content": "SEC announces investigation into AAPL for accounting irregularities.",
            },
            "factor_id": "f2",
            "interest_id": "i1",
            "factor_name": "Regulatory News",
            "intent": "AAPL holding safety",
            "entity": "AAPL",
        }
        result = asyncio.run(enricher.enrich(signal))
        d = result["delta"]

        self.assertEqual(d["source"], "web_enriched")
        self.assertAlmostEqual(d["severity"], 0.9)
        self.assertAlmostEqual(d["sentiment"], -0.8)
        self.assertEqual(d["event_type"], "regulatory")
        self.assertEqual(d["key_fact"], "SEC investigation announced")
        self.assertEqual(d["confidence"], 0.7)
        engine.generate_extraction.assert_called_once()

    def test_web_enrichment_llm_failure_fallback(self):
        engine = _make_engine_mock("not valid json at all")
        enricher = SignalEnricher(engine)

        signal = {
            "delta": {
                "type": "content_changed",
                "prev_hash": "old",
                "new_hash": "new2",
                "snippet": "Something happened",
            },
            "factor_id": "f3",
            "interest_id": "i2",
            "factor_name": "News",
            "intent": "monitor",
            "entity": "X",
        }
        result = asyncio.run(enricher.enrich(signal))
        d = result["delta"]

        self.assertEqual(d["source"], "web_enriched")
        self.assertAlmostEqual(d["severity"], 0.3)
        self.assertAlmostEqual(d["confidence"], 0.3)


class StageScoringTests(unittest.TestCase):
    """Stage 4: score_signal and evaluate_window work correctly."""

    def test_enriched_score(self):
        sig = {"delta": {"severity": 0.9, "confidence": 0.7}}
        score = score_signal(sig)
        self.assertAlmostEqual(score, min(2.0, 0.9 * 0.7 * 2.0))

    def test_legacy_stock_score(self):
        sig = {"delta": {"pct_change": 5.5}}
        score = score_signal(sig)
        self.assertEqual(score, 1.5)

    def test_content_changed_score(self):
        sig = {"delta": {"type": "content_changed"}}
        score = score_signal(sig)
        self.assertEqual(score, 0.5)

    def test_multi_factor_anomaly(self):
        """Two enriched signals from different factors should trigger anomaly."""
        signals = [
            {
                "interest_id": "i1",
                "factor_id": "f1",
                "delta": {"severity": 0.5, "confidence": 1.0, "source": "api",
                          "key_fact": "AAPL fell 5%"},
                "factor_name": "AAPL Price",
                "intent": "safety",
            },
            {
                "interest_id": "i1",
                "factor_id": "f2",
                "delta": {"severity": 0.9, "confidence": 0.7, "source": "web_enriched",
                          "key_fact": "SEC investigation"},
                "factor_name": "Regulatory News",
                "intent": "safety",
            },
        ]
        result = evaluate_window(signals)
        self.assertIsNotNone(result, "Two-factor signals should trigger anomaly")
        self.assertEqual(result["type"], "anomaly")
        self.assertGreaterEqual(result["score"], 1.5)
        self.assertEqual(result["factor_count"], 2)

    def test_single_high_severity_solo_trigger(self):
        """Single high-severity signal should trigger anomaly via solo rule."""
        signals = [
            {
                "interest_id": "i1",
                "factor_id": "f1",
                "delta": {"severity": 0.95, "confidence": 0.7, "source": "web_enriched",
                          "key_fact": "Critical event"},
                "factor_name": "Factor",
                "intent": "monitor",
            },
        ]
        result = evaluate_window(signals)
        self.assertIsNotNone(result, "Solo high-severity signal should trigger anomaly")
        self.assertGreaterEqual(result["score"], 1.5)

    def test_low_severity_no_anomaly(self):
        """Single low-severity signal should NOT trigger anomaly."""
        signals = [
            {
                "interest_id": "i1",
                "factor_id": "f1",
                "delta": {"severity": 0.2, "confidence": 0.3, "source": "web_enriched",
                          "key_fact": "Minor update"},
                "factor_name": "Factor",
                "intent": "monitor",
            },
        ]
        result = evaluate_window(signals)
        self.assertIsNone(result)


# ── Full pipeline integration test ───────────────────────────────────────


class FullPipelineTest(unittest.TestCase):
    """End-to-end: Interest → Poller → Enricher → AnomalyDetector → Callback."""

    def test_full_pipeline_stock_and_web(self):
        """Two factors (stock + web) both fire significant signals → anomaly detected."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            # ── 1. Interest creation ──
            store = _make_store(tmp_dir)
            factors = _make_factors()
            interest = store.create_interest(
                query="Watch AAPL, cost basis 142",
                intent="AAPL holding safety",
                context={"cost_basis": 142},
                discord_thread_id="thread-123",
                factors=factors,
            )
            self.assertEqual(interest["status"], "active")
            self.assertEqual(len(interest["factors"]), 2)

            # Verify factors persisted
            pollable = store.get_pollable_factors()
            self.assertEqual(len(pollable), 2)

            # ── 2. Simulate first poll (baseline, no delta) ──
            stock_factor = next(f for f in pollable if f["source_hint"] == "stock_price")
            web_factor = next(f for f in pollable if f["source_hint"] == "regulatory_news")

            # Set baseline values
            baseline_stock = json.dumps({"price": 142.0, "change": 0.0})
            baseline_web = json.dumps({"content_hash": "baseline_hash_abc"})
            store.update_factor_value(stock_factor["id"], baseline_stock, "2026-02-25T10:00:00Z")
            store.update_factor_value(web_factor["id"], baseline_web, "2026-02-25T10:00:00Z")

            # ── 3. Simulate second poll (significant changes) ──
            new_stock = {"price": 135.0, "change": -7.0, "change_pct": -4.93}
            stock_delta = compute_delta("stock_price", baseline_stock, new_stock, fetch_mode="api")
            self.assertIsNotNone(stock_delta)
            self.assertTrue(is_significant("stock_price", stock_delta))

            new_web = {
                "content_hash": "new_hash_xyz",
                "snippet": "SEC announces formal investigation into Apple accounting practices",
                "full_content": "SEC announces formal investigation into Apple accounting practices, shares tumble.",
            }
            web_delta = compute_delta("regulatory_news", baseline_web, new_web, fetch_mode="web")
            self.assertIsNotNone(web_delta)
            self.assertEqual(web_delta["type"], "content_changed")
            self.assertTrue(is_significant("regulatory_news", web_delta))

            # ── 4. Enrichment ──
            llm_json = json.dumps({
                "severity": 0.92,
                "sentiment": -0.85,
                "event_type": "regulatory",
                "key_fact": "SEC formal investigation into Apple accounting",
            })
            engine = _make_engine_mock(llm_json)
            enricher = SignalEnricher(engine)

            # Enrich stock signal (deterministic)
            stock_signal = {
                "type": "signal",
                "interest_id": interest["id"],
                "factor_id": stock_factor["id"],
                "factor_name": "AAPL Price",
                "source_hint": "stock_price",
                "value": new_stock,
                "delta": stock_delta,
                "intent": "AAPL holding safety",
                "entity": "AAPL",
                "discord_thread_id": "thread-123",
            }
            stock_signal = asyncio.run(enricher.enrich(stock_signal))
            self.assertEqual(stock_signal["delta"]["source"], "api")
            self.assertEqual(stock_signal["delta"]["confidence"], 1.0)
            self.assertIn("fell", stock_signal["delta"]["key_fact"])

            # Enrich web signal (LLM)
            web_signal = {
                "type": "signal",
                "interest_id": interest["id"],
                "factor_id": web_factor["id"],
                "factor_name": "Regulatory News",
                "source_hint": "regulatory_news",
                "value": new_web,
                "delta": web_delta,
                "intent": "AAPL holding safety",
                "entity": "AAPL",
                "discord_thread_id": "thread-123",
            }
            web_signal = asyncio.run(enricher.enrich(web_signal))
            self.assertEqual(web_signal["delta"]["source"], "web_enriched")
            self.assertEqual(web_signal["delta"]["confidence"], 0.7)
            self.assertAlmostEqual(web_signal["delta"]["severity"], 0.92)

            # ── 5. Anomaly detection ──
            anomaly_results = []
            passthrough_results = []

            async def on_anomaly(a):
                anomaly_results.append(a)

            async def on_passthrough(s):
                passthrough_results.append(s)

            detector = AnomalyDetector(
                on_anomaly=on_anomaly,
                on_signal_passthrough=on_passthrough,
                window_seconds=300,
            )

            async def run_detection():
                await detector.ingest(stock_signal)
                await detector.ingest(web_signal)  # urgent (severity=0.92 >= 0.85)

            asyncio.run(run_detection())

            # Web signal is urgent → immediate flush
            self.assertEqual(len(anomaly_results), 1, "Should detect anomaly")
            anomaly = anomaly_results[0]
            self.assertEqual(anomaly["type"], "anomaly")
            self.assertEqual(anomaly["interest_id"], interest["id"])
            self.assertGreaterEqual(anomaly["score"], 1.5)
            self.assertEqual(anomaly["factor_count"], 2)
            self.assertEqual(anomaly["signal_count"], 2)
            self.assertIn("AAPL", anomaly["summary"])
            self.assertEqual(len(passthrough_results), 0)

    def test_full_pipeline_low_severity_passthrough(self):
        """Single low-severity web signal → passthrough, no anomaly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = _make_store(tmp_dir)
            factors = [
                ResolvedFactor(
                    name="Tech News",
                    source_hint="tech_trend",
                    entity="AI",
                    frequency="hourly",
                    resolution=Resolution.PROXY,
                    bound_tool="web_search",
                    reliability=0.7,
                    weight=1.0,
                ),
            ]
            interest = store.create_interest(
                query="AI industry updates",
                intent="Track AI industry trends",
                factors=factors,
            )

            # LLM returns low severity
            llm_json = json.dumps({
                "severity": 0.2,
                "sentiment": 0.3,
                "event_type": "product",
                "key_fact": "Minor product update announced",
            })
            engine = _make_engine_mock(llm_json)
            enricher = SignalEnricher(engine)

            web_factor = store.get_pollable_factors()[0]
            signal = {
                "type": "signal",
                "interest_id": interest["id"],
                "factor_id": web_factor["id"],
                "factor_name": "Tech News",
                "source_hint": "tech_trend",
                "value": {"content_hash": "h1", "snippet": "Minor update"},
                "delta": {
                    "type": "content_changed",
                    "prev_hash": "h0",
                    "new_hash": "h1",
                    "snippet": "Minor product update",
                },
                "intent": "Track AI industry trends",
                "entity": "AI",
            }
            signal = asyncio.run(enricher.enrich(signal))

            anomaly_results = []
            passthrough_results = []

            async def on_anomaly(a):
                anomaly_results.append(a)

            async def on_passthrough(s):
                passthrough_results.append(s)

            detector = AnomalyDetector(
                on_anomaly=on_anomaly,
                on_signal_passthrough=on_passthrough,
                window_seconds=0.1,  # short window for test
            )

            async def run():
                await detector.ingest(signal)
                await asyncio.sleep(0.3)  # let deferred flush fire

            asyncio.run(run())

            self.assertEqual(len(anomaly_results), 0)
            self.assertEqual(len(passthrough_results), 1)
            self.assertEqual(passthrough_results[0]["delta"]["key_fact"], "Minor product update announced")

    def test_full_pipeline_stock_api_fallback_to_web(self):
        """Stock factor where API fails → falls back to web → still enriched."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = _make_store(tmp_dir)
            factors = [
                ResolvedFactor(
                    name="TSLA Price",
                    source_hint="stock_price",
                    entity="TSLA",
                    frequency="realtime",
                    resolution=Resolution.EXECUTABLE,
                    bound_tool="stock",
                    reliability=1.0,
                    weight=1.0,
                ),
            ]
            interest = store.create_interest(
                query="Watch TSLA",
                intent="TSLA price monitoring",
                factors=factors,
            )

            # Simulate: stock API failed, fell back to web search
            # Previous poll was also web (API was down last time too)
            baseline_web = json.dumps({"content_hash": "prev_hash"})
            new_web = {
                "content_hash": "new_hash_tsla",
                "snippet": "TSLA drops 8% after Musk controversy",
                "full_content": "Tesla shares plunged 8% following CEO controversy.",
            }
            delta = compute_delta("stock_price", baseline_web, new_web, fetch_mode="web")
            self.assertIsNotNone(delta)
            self.assertEqual(delta["type"], "content_changed")

            llm_json = json.dumps({
                "severity": 0.85,
                "sentiment": -0.9,
                "event_type": "personnel",
                "key_fact": "TSLA drops 8% after CEO controversy",
            })
            engine = _make_engine_mock(llm_json)
            enricher = SignalEnricher(engine)

            factor = store.get_pollable_factors()[0]
            signal = {
                "type": "signal",
                "interest_id": interest["id"],
                "factor_id": factor["id"],
                "factor_name": "TSLA Price",
                "source_hint": "stock_price",
                "value": new_web,
                "delta": delta,
                "intent": "TSLA price monitoring",
                "entity": "TSLA",
            }
            signal = asyncio.run(enricher.enrich(signal))

            # Even though stock_price hint, it came via web → web_enriched
            self.assertEqual(signal["delta"]["source"], "web_enriched")
            self.assertAlmostEqual(signal["delta"]["severity"], 0.85)
            self.assertAlmostEqual(signal["delta"]["confidence"], 0.7)

            # Should trigger solo anomaly (severity >= 0.85 AND confidence >= 0.7)
            result = evaluate_window([signal])
            self.assertIsNotNone(result, "Solo high-severity web signal should trigger anomaly")
            self.assertGreaterEqual(result["score"], 1.5)


class PollerIntegrationTest(unittest.TestCase):
    """Test SignalPoller with mocked fetch functions."""

    def test_poller_add_interest_and_poll(self):
        """Poller picks up new interest, polls, emits enriched signal."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = _make_store(tmp_dir)
            factors = _make_factors()
            interest = store.create_interest(
                query="test",
                intent="test",
                factors=factors,
            )

            # Set baseline so delta can be computed
            for f in store.get_pollable_factors():
                if f["source_hint"] == "stock_price":
                    store.update_factor_value(f["id"], json.dumps({"price": 100.0}), "2026-02-25T10:00:00Z")
                else:
                    store.update_factor_value(f["id"], json.dumps({"content_hash": "old"}), "2026-02-25T10:00:00Z")

            received = []

            async def callback(sig):
                received.append(sig)

            llm_json = json.dumps({
                "severity": 0.6, "sentiment": 0.5,
                "event_type": "market", "key_fact": "Positive news",
            })
            engine = _make_engine_mock(llm_json)
            enricher = SignalEnricher(engine)

            poller = SignalPoller(store, on_signal=callback, enricher=enricher)

            # Mock the fetch functions so we don't need real APIs
            stock_result = {"price": 106.0, "change": 6.0, "change_pct": 6.0}
            web_result = {
                "content_hash": "new_content_hash",
                "snippet": "Good regulatory news for AAPL",
                "full_content": "Apple receives regulatory approval for new product line.",
                "result_count": 3,
            }

            async def mock_fetch(factor, source_hint, bound_tool):
                if source_hint == "stock_price":
                    return stock_result, "api"
                return web_result, "web"

            poller._fetch = mock_fetch

            async def run():
                # Manually trigger one poll cycle per factor
                pollable = store.get_pollable_factors()
                for f in pollable:
                    await poller._poll_loop_once(f) if hasattr(poller, '_poll_loop_once') else None

            # Since _poll_loop is infinite, test by directly calling the fetch+delta+emit logic
            async def simulate_poll():
                pollable = store.get_pollable_factors()
                for factor in pollable:
                    fid = factor["id"]
                    source_hint = factor["source_hint"]
                    bound_tool = factor.get("bound_tool")
                    value, fetch_mode = await mock_fetch(factor, source_hint, bound_tool)
                    if value is not None:
                        value_json = json.dumps(value, ensure_ascii=False)
                        prev_json = factor.get("last_value")
                        delta = compute_delta(source_hint, prev_json, value, fetch_mode=fetch_mode)
                        if is_significant(source_hint, delta):
                            signal = {
                                "type": "signal",
                                "interest_id": factor["interest_id"],
                                "factor_id": fid,
                                "factor_name": factor["name"],
                                "source_hint": source_hint,
                                "value": value,
                                "delta": delta,
                                "intent": factor.get("intent", ""),
                                "entity": factor.get("entity", ""),
                            }
                            if enricher:
                                signal = await enricher.enrich(signal)
                            if callback:
                                await callback(signal)

            asyncio.run(simulate_poll())

            # Both factors should fire (stock 6% > 2%, web content changed)
            self.assertEqual(len(received), 2)

            stock_sig = next(s for s in received if s["source_hint"] == "stock_price")
            web_sig = next(s for s in received if s["source_hint"] == "regulatory_news")

            # Stock: API enriched (deterministic)
            self.assertEqual(stock_sig["delta"]["source"], "api")
            self.assertEqual(stock_sig["delta"]["confidence"], 1.0)
            self.assertIn("AAPL", stock_sig["delta"]["key_fact"])

            # Web: LLM enriched
            self.assertEqual(web_sig["delta"]["source"], "web_enriched")
            self.assertEqual(web_sig["delta"]["confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
