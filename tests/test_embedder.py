# tests/test_embedder.py
"""Tests for embedding provider abstraction."""

import unittest

from liagent.agent.embedder import classify_query_type


class QueryTypeTests(unittest.TestCase):
    def test_ticker_is_exact(self):
        self.assertEqual(classify_query_type("AAPL"), "exact")

    def test_error_code_is_exact(self):
        self.assertEqual(classify_query_type("ERR_CONNECTION_REFUSED"), "exact")

    def test_uuid_is_exact(self):
        self.assertEqual(classify_query_type("a1b2c3d4-e5f6"), "exact")

    def test_natural_question_is_semantic(self):
        self.assertEqual(classify_query_type("how should I configure remote access"), "semantic")

    def test_english_question_is_semantic(self):
        self.assertEqual(classify_query_type("how to configure remote access"), "semantic")

    def test_mixed_default(self):
        self.assertEqual(classify_query_type("AAPL stock analysis"), "mixed")


if __name__ == "__main__":
    unittest.main()
