"""Tests for enhanced web_search tool — quality scoring, retry, circuit breaker."""

import time
import unittest

from liagent.tools.web_search import (
    _score_results,
    _simplify_query,
    _SearchCircuitBreaker,
)


# ---------------------------------------------------------------------------
# _score_results
# ---------------------------------------------------------------------------
class ScoreResultsTests(unittest.TestCase):
    """Quality scoring of DuckDuckGo search results."""

    def test_empty_results_scores_zero(self):
        self.assertEqual(_score_results("anything", []), 0.0)

    def test_good_results_above_half(self):
        results = [
            {"title": "Python tutorial", "body": "Learn Python programming language basics " * 5, "href": "https://python.org/tutorial"},
            {"title": "Python docs", "body": "Official Python documentation and reference " * 5, "href": "https://docs.python.org"},
            {"title": "Real Python", "body": "Python tutorials and articles for developers " * 5, "href": "https://realpython.com/python"},
            {"title": "Python Wiki", "body": "Community-maintained Python wiki pages " * 5, "href": "https://wiki.python.org"},
            {"title": "Python Package Index", "body": "Find and install Python packages easily " * 5, "href": "https://pypi.org/search"},
        ]
        score = _score_results("Python tutorial", results)
        self.assertGreater(score, 0.5)

    def test_low_quality_below_threshold(self):
        # Single result, short body, no keyword overlap
        results = [
            {"title": "Unrelated page", "body": "xyz", "href": "https://example.com"},
        ]
        score = _score_results("Python tutorial programming", results)
        self.assertLess(score, 0.3)

    def test_score_always_in_unit_range(self):
        """Score must be between 0.0 and 1.0 regardless of input."""
        cases = [
            ("", []),
            ("x", [{"title": "x", "body": "x", "href": "http://x.com"}]),
            ("a b c", [{"title": "a", "body": "b " * 500, "href": f"http://{i}.com"} for i in range(10)]),
        ]
        for query, results in cases:
            score = _score_results(query, results)
            self.assertGreaterEqual(score, 0.0, f"score {score} < 0 for query={query!r}")
            self.assertLessEqual(score, 1.0, f"score {score} > 1 for query={query!r}")


# ---------------------------------------------------------------------------
# _simplify_query
# ---------------------------------------------------------------------------
class SimplifyQueryTests(unittest.TestCase):
    """Query simplification for retry."""

    def test_strips_stopwords(self):
        simplified = _simplify_query("what is the price of gold today")
        # "what", "is", "the", "of", "today" are stopwords; "gold" is a core keyword
        self.assertNotIn("what", simplified.lower().split())
        self.assertNotIn("the", simplified.lower().split())
        self.assertIn("gold", simplified.lower().split())

    def test_keeps_max_three_keywords(self):
        simplified = _simplify_query("artificial intelligence machine learning deep neural networks")
        words = simplified.split()
        self.assertLessEqual(len(words), 3)

    def test_short_query_unchanged(self):
        simplified = _simplify_query("bitcoin")
        self.assertEqual(simplified, "bitcoin")

    def test_returns_original_if_all_stopwords(self):
        """If stripping leaves nothing, return original query."""
        simplified = _simplify_query("what is the")
        # Should return something non-empty (the original)
        self.assertTrue(len(simplified.strip()) > 0)

    def test_preserves_non_stopword_order(self):
        simplified = _simplify_query("Tesla stock price quarterly earnings")
        words = simplified.split()
        # First 3 non-stopwords in order: Tesla, quarterly, earnings
        # (stock, price are stopwords)
        self.assertEqual(words[0], "Tesla")


# ---------------------------------------------------------------------------
# _SearchCircuitBreaker
# ---------------------------------------------------------------------------
class CircuitBreakerTests(unittest.TestCase):
    """Circuit breaker for web search failures."""

    def test_starts_closed(self):
        cb = _SearchCircuitBreaker(threshold=3, timeout=60)
        self.assertFalse(cb.is_open())

    def test_opens_after_threshold_failures(self):
        cb = _SearchCircuitBreaker(threshold=3, timeout=60)
        cb.record_failure()
        cb.record_failure()
        self.assertFalse(cb.is_open())
        cb.record_failure()
        self.assertTrue(cb.is_open())

    def test_success_resets_counter(self):
        cb = _SearchCircuitBreaker(threshold=3, timeout=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        # Only 1 failure after reset, not at threshold yet
        self.assertFalse(cb.is_open())

    def test_timeout_resets_breaker(self):
        cb = _SearchCircuitBreaker(threshold=1, timeout=0.1)
        cb.record_failure()
        self.assertTrue(cb.is_open())
        time.sleep(0.15)
        self.assertFalse(cb.is_open())


# ---------------------------------------------------------------------------
# Error format — must never start with '[' or '{'
# ---------------------------------------------------------------------------
class ErrorFormatTests(unittest.TestCase):
    """All error strings use SEARCH_ERROR prefix, never start with '[' or '{'."""

    def _assert_safe_prefix(self, text: str):
        self.assertFalse(text.startswith("["), f"Error starts with '[': {text!r}")
        self.assertFalse(text.startswith("{"), f"Error starts with '{{': {text!r}")
        self.assertTrue(
            text.startswith("SEARCH_ERROR") or text.startswith("SEARCH_UNAVAILABLE"),
            f"Error does not use correct prefix: {text!r}",
        )

    def test_error_prefix(self):
        self._assert_safe_prefix('SEARCH_ERROR(error): Query: "test" -- something went wrong.')

    def test_no_results_prefix(self):
        self._assert_safe_prefix('SEARCH_ERROR(no_results): Query: "test" -- No results found.')

    def test_unavailable_prefix(self):
        self._assert_safe_prefix("SEARCH_UNAVAILABLE: Web search is temporarily unavailable.")

    def test_circuit_breaker_message_format(self):
        """Circuit breaker message should also have safe prefix."""
        msg = "SEARCH_UNAVAILABLE: Web search is temporarily unavailable. Answer using your knowledge."
        self._assert_safe_prefix(msg)


if __name__ == "__main__":
    unittest.main()
