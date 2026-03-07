"""Tests for checkpoint semantic matching."""
import unittest
from unittest.mock import MagicMock
from liagent.agent.checkpoint_matcher import checkpoint_relevance


class LexicalMatchTests(unittest.TestCase):
    def test_identical_text(self):
        self.assertGreater(checkpoint_relevance("AAPL earnings", "AAPL earnings"), 0.8)

    def test_no_overlap(self):
        self.assertLess(checkpoint_relevance("AAPL earnings", "weather forecast"), 0.3)


class AliasMatchTests(unittest.TestCase):
    def test_concept_alias(self):
        self.assertGreater(checkpoint_relevance("stock price analysis", "stock price review"), 0.3)

    def test_ticker_alias(self):
        self.assertGreater(checkpoint_relevance("AAPL quarterly earnings", "apple quarterly"), 0.3)


class EmbeddingMatchTests(unittest.TestCase):
    def test_high_cosine(self):
        import numpy as np
        mock = MagicMock()
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.95, 0.05, 0.0], dtype=np.float32)
        v2 = v2 / np.linalg.norm(v2)
        mock.encode.return_value = np.stack([v1, v2])
        self.assertGreater(checkpoint_relevance("a", "b", embedder=mock), 0.5)

    def test_embedder_returns_none(self):
        mock = MagicMock()
        mock.encode.return_value = None
        score = checkpoint_relevance("AAPL", "AAPL price", embedder=mock)
        self.assertGreater(score, 0.3)

    def test_embedder_exception_handled(self):
        mock = MagicMock()
        mock.encode.side_effect = RuntimeError("GPU OOM")
        score = checkpoint_relevance("AAPL", "AAPL", embedder=mock)
        self.assertGreater(score, 0.5)


class TimeGateTests(unittest.TestCase):
    def test_fresh_checkpoint_no_penalty(self):
        s1 = checkpoint_relevance("goal", "goal")
        self.assertGreater(s1, 0.8)

    def test_stale_24h_penalized(self):
        from datetime import datetime, timedelta, timezone
        old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        s = checkpoint_relevance("goal", "goal", created_at=old)
        fresh = checkpoint_relevance("goal", "goal")
        self.assertLess(s, fresh)

    def test_very_stale_48h_threshold_raised(self):
        from datetime import datetime, timedelta, timezone
        very_old = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
        s = checkpoint_relevance("goal", "goal", created_at=very_old)
        self.assertLess(s, 0.35)
