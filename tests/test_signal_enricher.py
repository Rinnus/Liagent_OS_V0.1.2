"""Tests for the signal enrichment module."""

import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from liagent.agent.signal_enricher import (
    SignalEnricher,
    _extract_json,
)


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self):
        raw = '{"severity": 0.8, "sentiment": -0.5}'
        obj = _extract_json(raw)
        self.assertAlmostEqual(obj["severity"], 0.8)

    def test_fenced_json(self):
        raw = '```json\n{"severity": 0.6}\n```'
        obj = _extract_json(raw)
        self.assertAlmostEqual(obj["severity"], 0.6)

    def test_embedded_json(self):
        raw = 'Here is the analysis: {"severity": 0.9, "key_fact": "test"} done.'
        obj = _extract_json(raw)
        self.assertAlmostEqual(obj["severity"], 0.9)

    def test_no_json(self):
        self.assertIsNone(_extract_json("no json here"))

    def test_empty(self):
        self.assertIsNone(_extract_json(""))

    def test_with_think_block(self):
        raw = '<think>reasoning</think>{"severity": 0.5}'
        obj = _extract_json(raw)
        self.assertAlmostEqual(obj["severity"], 0.5)


class TestEnrichApiDelta(unittest.IsolatedAsyncioTestCase):
    async def test_stock_increase(self):
        engine = MagicMock()
        enricher = SignalEnricher(engine)

        signal = {
            "factor_name": "AAPL Price",
            "entity": "AAPL",
            "delta": {
                "pct_change": 5.0,
                "abs_change": 7.5,
                "prev_price": 150.0,
                "new_price": 157.5,
            },
        }
        result = await enricher.enrich(signal)
        delta = result["delta"]

        self.assertEqual(delta["source"], "api")
        self.assertEqual(delta["confidence"], 1.0)
        self.assertAlmostEqual(delta["severity"], 0.5)
        self.assertEqual(delta["sentiment"], 1.0)
        self.assertEqual(delta["event_type"], "market")
        self.assertIn("AAPL", delta["key_fact"])
        self.assertIn("rose", delta["key_fact"])
        self.assertIn("raw_delta", delta)

    async def test_stock_decrease(self):
        engine = MagicMock()
        enricher = SignalEnricher(engine)

        signal = {
            "entity": "TSLA",
            "factor_name": "TSLA Price",
            "delta": {"pct_change": -10.0, "prev_price": 200, "new_price": 180},
        }
        result = await enricher.enrich(signal)
        delta = result["delta"]

        self.assertAlmostEqual(delta["severity"], 1.0)
        self.assertEqual(delta["sentiment"], -1.0)
        self.assertIn("fell", delta["key_fact"])

    async def test_severity_capped_at_1(self):
        engine = MagicMock()
        enricher = SignalEnricher(engine)

        signal = {
            "entity": "X",
            "delta": {"pct_change": -25.0, "prev_price": 100, "new_price": 75},
        }
        result = await enricher.enrich(signal)
        self.assertLessEqual(result["delta"]["severity"], 1.0)


class TestEnrichWebDelta(unittest.IsolatedAsyncioTestCase):
    async def test_successful_llm_enrichment(self):
        engine = MagicMock()
        engine.generate_extraction = AsyncMock(return_value=json.dumps({
            "severity": 0.8,
            "sentiment": -0.7,
            "event_type": "regulatory",
            "key_fact": "SEC investigation announced",
        }))
        enricher = SignalEnricher(engine)

        signal = {
            "interest_id": "i1",
            "factor_id": "f1",
            "factor_name": "Regulatory News",
            "intent": "compliance monitoring",
            "entity": "ACME",
            "delta": {
                "type": "content_changed",
                "prev_hash": "aaa",
                "new_hash": "bbb",
                "snippet": "SEC announces probe",
                "full_content": "SEC announces probe into ACME Corp accounting",
            },
        }
        result = await enricher.enrich(signal)
        delta = result["delta"]

        self.assertEqual(delta["source"], "web_enriched")
        self.assertAlmostEqual(delta["severity"], 0.8)
        self.assertAlmostEqual(delta["sentiment"], -0.7)
        self.assertEqual(delta["event_type"], "regulatory")
        self.assertEqual(delta["confidence"], 0.7)
        self.assertIn("raw_delta", delta)
        engine.generate_extraction.assert_called_once()

    async def test_llm_failure_uses_fallback(self):
        engine = MagicMock()
        engine.generate_extraction = AsyncMock(side_effect=Exception("LLM down"))
        enricher = SignalEnricher(engine)

        signal = {
            "interest_id": "i1",
            "factor_id": "f1",
            "factor_name": "News",
            "entity": "X",
            "delta": {
                "type": "content_changed",
                "prev_hash": "a",
                "new_hash": "b",
                "snippet": "some news",
            },
        }
        result = await enricher.enrich(signal)
        delta = result["delta"]

        self.assertEqual(delta["confidence"], 0.3)
        self.assertEqual(delta["severity"], 0.3)
        self.assertEqual(delta["source"], "web_enriched")

    async def test_llm_garbage_output_uses_fallback(self):
        engine = MagicMock()
        engine.generate_extraction = AsyncMock(return_value="not json at all")
        enricher = SignalEnricher(engine)

        signal = {
            "interest_id": "i1",
            "factor_id": "f1",
            "entity": "X",
            "delta": {
                "type": "content_changed",
                "prev_hash": "a",
                "new_hash": "b",
                "snippet": "news",
            },
        }
        result = await enricher.enrich(signal)
        self.assertEqual(result["delta"]["confidence"], 0.3)

    async def test_cache_hit(self):
        engine = MagicMock()
        engine.generate_extraction = AsyncMock(return_value=json.dumps({
            "severity": 0.6,
            "sentiment": 0.0,
            "event_type": "product",
            "key_fact": "New product launch",
        }))
        enricher = SignalEnricher(engine)

        signal = {
            "interest_id": "i1",
            "factor_id": "f1",
            "factor_name": "Product",
            "entity": "X",
            "delta": {
                "type": "content_changed",
                "prev_hash": "a",
                "new_hash": "cached_hash",
                "snippet": "product launch",
                "full_content": "product launch details",
            },
        }
        # First call — LLM invoked
        await enricher.enrich(dict(signal))
        self.assertEqual(engine.generate_extraction.call_count, 1)

        # Second call with same content_hash — cache hit
        await enricher.enrich(dict(signal))
        self.assertEqual(engine.generate_extraction.call_count, 1)  # not called again

    async def test_rate_limit(self):
        engine = MagicMock()
        engine.generate_extraction = AsyncMock(return_value=json.dumps({
            "severity": 0.5,
            "sentiment": 0.0,
            "event_type": "other",
            "key_fact": "fact",
        }))
        enricher = SignalEnricher(engine)

        signal = {
            "interest_id": "i1",
            "factor_id": "f1",
            "factor_name": "Test",
            "entity": "X",
            "delta": {
                "type": "content_changed",
                "prev_hash": "a",
                "new_hash": "hash1",
                "snippet": "first",
                "full_content": "first content",
            },
        }
        await enricher.enrich(dict(signal))
        self.assertEqual(engine.generate_extraction.call_count, 1)

        # Different content hash but same factor within rate limit window
        signal["delta"]["new_hash"] = "hash2"
        signal["delta"]["snippet"] = "second"
        result = await enricher.enrich(dict(signal))
        # Should be rate-limited → fallback
        self.assertEqual(engine.generate_extraction.call_count, 1)
        self.assertEqual(result["delta"]["confidence"], 0.3)

    async def test_different_factors_not_rate_limited(self):
        engine = MagicMock()
        engine.generate_extraction = AsyncMock(return_value=json.dumps({
            "severity": 0.5,
            "sentiment": 0.0,
            "event_type": "other",
            "key_fact": "fact",
        }))
        enricher = SignalEnricher(engine)

        signal1 = {
            "interest_id": "i1",
            "factor_id": "f1",
            "factor_name": "Factor1",
            "entity": "X",
            "delta": {
                "type": "content_changed",
                "prev_hash": "a",
                "new_hash": "h1",
                "snippet": "s1",
                "full_content": "content1",
            },
        }
        signal2 = {
            "interest_id": "i1",
            "factor_id": "f2",
            "factor_name": "Factor2",
            "entity": "Y",
            "delta": {
                "type": "content_changed",
                "prev_hash": "b",
                "new_hash": "h2",
                "snippet": "s2",
                "full_content": "content2",
            },
        }
        await enricher.enrich(signal1)
        await enricher.enrich(signal2)
        # Both should invoke LLM (different factor_ids)
        self.assertEqual(engine.generate_extraction.call_count, 2)


class TestEnrichNoOp(unittest.IsolatedAsyncioTestCase):
    async def test_no_delta(self):
        engine = MagicMock()
        enricher = SignalEnricher(engine)
        signal = {"factor_name": "X"}
        result = await enricher.enrich(signal)
        self.assertIs(result, signal)

    async def test_unknown_delta_type(self):
        engine = MagicMock()
        enricher = SignalEnricher(engine)
        signal = {"delta": {"type": "unknown_type"}}
        result = await enricher.enrich(signal)
        self.assertEqual(result["delta"]["type"], "unknown_type")  # unchanged


class TestEnrichClampValues(unittest.IsolatedAsyncioTestCase):
    async def test_severity_clamped(self):
        engine = MagicMock()
        engine.generate_extraction = AsyncMock(return_value=json.dumps({
            "severity": 5.0,
            "sentiment": -3.0,
            "event_type": "other",
            "key_fact": "extreme",
        }))
        enricher = SignalEnricher(engine)

        signal = {
            "interest_id": "i1",
            "factor_id": "f1",
            "entity": "X",
            "delta": {
                "type": "content_changed",
                "prev_hash": "a",
                "new_hash": "b",
                "snippet": "extreme event",
                "full_content": "extreme event details",
            },
        }
        result = await enricher.enrich(signal)
        delta = result["delta"]
        self.assertLessEqual(delta["severity"], 1.0)
        self.assertGreaterEqual(delta["sentiment"], -1.0)
