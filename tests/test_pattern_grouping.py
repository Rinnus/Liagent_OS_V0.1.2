import unittest


class NormalizeTests(unittest.TestCase):
    def test_stock_query_extracts_entity_and_intent(self):
        from liagent.agent.pattern_grouping import normalize_patterns
        raw = [{"signal_type": "stock_query", "key": "AAPL", "domain": "stock",
                "confidence": 0.8, "count": 5, "distinct_days": 3, "pattern_key": "stock_query:AAPL"}]
        result = normalize_patterns(raw)
        self.assertEqual(result[0]["entities"], ["AAPL"])
        self.assertEqual(result[0]["intent"], "price_check")

    def test_web_search_intent(self):
        from liagent.agent.pattern_grouping import normalize_patterns
        raw = [{"signal_type": "web_search", "key": "tech:AI", "domain": "tech",
                "confidence": 0.6, "count": 3, "distinct_days": 2, "pattern_key": "web_search:tech:AI"}]
        result = normalize_patterns(raw)
        self.assertEqual(result[0]["intent"], "info_search")
        self.assertIn("AI", result[0]["entities"])

    def test_unknown_signal_type(self):
        from liagent.agent.pattern_grouping import normalize_patterns
        raw = [{"signal_type": "unknown_type", "key": "foo", "domain": "misc",
                "confidence": 0.5, "count": 1, "distinct_days": 1, "pattern_key": "unknown:foo"}]
        result = normalize_patterns(raw)
        self.assertEqual(result[0]["intent"], "general")


class UnionFindTests(unittest.TestCase):
    def test_basic_union(self):
        from liagent.agent.pattern_grouping import UnionFind
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        self.assertEqual(uf.find(0), uf.find(1))
        self.assertNotEqual(uf.find(0), uf.find(2))

    def test_transitive(self):
        from liagent.agent.pattern_grouping import UnionFind
        uf = UnionFind(3)
        uf.union(0, 1)
        uf.union(1, 2)
        self.assertEqual(uf.find(0), uf.find(2))


class ShouldMergeTests(unittest.TestCase):
    def test_same_domain_shared_entity(self):
        from liagent.agent.pattern_grouping import _should_merge
        a = {"domain": "stock", "entities": ["AAPL", "TSLA"], "intent": "price_check"}
        b = {"domain": "stock", "entities": ["AAPL", "GOOGL"], "intent": "info_search"}
        self.assertTrue(_should_merge(a, b))

    def test_same_domain_same_intent(self):
        from liagent.agent.pattern_grouping import _should_merge
        a = {"domain": "stock", "entities": ["AAPL"], "intent": "price_check"}
        b = {"domain": "stock", "entities": ["TSLA"], "intent": "price_check"}
        self.assertTrue(_should_merge(a, b))

    def test_different_domain(self):
        from liagent.agent.pattern_grouping import _should_merge
        a = {"domain": "stock", "entities": ["AAPL"], "intent": "price_check"}
        b = {"domain": "tech", "entities": ["AAPL"], "intent": "price_check"}
        self.assertFalse(_should_merge(a, b))

    def test_no_overlap(self):
        from liagent.agent.pattern_grouping import _should_merge
        a = {"domain": "stock", "entities": ["AAPL"], "intent": "price_check"}
        b = {"domain": "stock", "entities": ["BTC"], "intent": "info_search"}
        self.assertFalse(_should_merge(a, b))


class UpdatePatternGroupsTests(unittest.TestCase):
    def setUp(self):
        import os, tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.goal_store import GoalStore
        self.store = GoalStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_group_from_patterns(self):
        from liagent.agent.pattern_grouping import update_pattern_groups
        patterns = [
            {"domain": "stock", "entities": ["AAPL"], "intent": "price_check",
             "signal_type": "stock_query", "key": "AAPL", "confidence": 0.8},
            {"domain": "stock", "entities": ["AAPL", "TSLA"], "intent": "price_check",
             "signal_type": "stock_query", "key": "TSLA", "confidence": 0.7},
        ]
        new_ids = update_pattern_groups(patterns, self.store)
        self.assertGreater(len(new_ids), 0)

    def test_updates_existing_group(self):
        from liagent.agent.pattern_grouping import update_pattern_groups
        patterns = [
            {"domain": "stock", "entities": ["AAPL"], "intent": "price_check",
             "signal_type": "stock_query", "key": "AAPL", "confidence": 0.8},
        ]
        update_pattern_groups(patterns, self.store)
        update_pattern_groups(patterns, self.store)
        groups = self.store.get_recent_groups(limit=10)
        self.assertEqual(len(groups), 1)


if __name__ == "__main__":
    unittest.main()
