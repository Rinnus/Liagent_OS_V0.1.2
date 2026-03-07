import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone


class SignalIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        # Initialize schema via LongTermMemory
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)
        from liagent.agent.behavior import BehaviorSignalStore, BehaviorPatternDetector
        self.signal_store = BehaviorSignalStore(self.db_path)
        self.detector = BehaviorPatternDetector(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _insert_signals(self, key, domain, source_origin, count, distinct_days=2):
        """Insert signals directly into DB to bypass dedup and control dates."""
        now = datetime.now(timezone.utc)
        with sqlite3.connect(self.db_path) as conn:
            for i in range(count):
                # Spread across distinct_days days
                day_offset = i % distinct_days
                ts = (now - timedelta(days=day_offset)).isoformat()
                conn.execute(
                    "INSERT INTO behavior_signals "
                    "(signal_type, key, domain, source_origin, metadata, "
                    "session_id, hour, weekday, created_at) "
                    "VALUES (?, ?, ?, ?, '{}', ?, 10, 1, ?)",
                    ("tool_use", key, domain, source_origin, f"s{i}", ts),
                )

    def test_system_signals_excluded_from_detection(self):
        # Record signals with source_origin='system'
        self._insert_signals("stock", "stock", "system", 5, distinct_days=3)
        patterns = self.detector.detect()
        # System signals should not appear as patterns
        stock_patterns = [p for p in patterns if p.get("key") == "stock" and p.get("signal_type") == "tool_use"]
        self.assertEqual(len(stock_patterns), 0)

    def test_user_signals_included_in_detection(self):
        # Record signals with source_origin='user'
        self._insert_signals("web_search", "news", "user", 5, distinct_days=3)
        patterns = self.detector.detect()
        web_patterns = [p for p in patterns if p.get("key") == "web_search"]
        self.assertGreater(len(web_patterns), 0)

    def test_mixed_signals_only_user_detected(self):
        # Mix of user and system signals
        self._insert_signals("stock", "stock", "user", 3, distinct_days=2)
        self._insert_signals("stock", "stock", "system", 5, distinct_days=3)
        patterns = self.detector.detect()
        stock_patterns = [p for p in patterns if p.get("key") == "stock"]
        if stock_patterns:
            # Count should reflect only user signals (3), not system ones (5)
            self.assertLessEqual(stock_patterns[0].get("count", 0), 3)

    def test_record_with_source_origin_stores_correctly(self):
        """Verify that source_origin is persisted via the store API."""
        self.signal_store.record("tool_use", "stock", domain="stock",
                                 session_id="test_sys", source_origin="system")
        self.signal_store.record("tool_use", "web_search", domain="news",
                                 session_id="test_usr", source_origin="user")
        self.signal_store.flush()
        # Both signals stored
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key, source_origin FROM behavior_signals ORDER BY key"
            ).fetchall()
        origins = {r[0]: r[1] for r in rows}
        self.assertEqual(origins["stock"], "system")
        self.assertEqual(origins["web_search"], "user")

    def test_default_source_origin_is_user(self):
        """Without explicit source_origin, default is 'user'."""
        self.signal_store.record("tool_use", "lint", domain="coding",
                                 session_id="test_default")
        self.signal_store.flush()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT source_origin FROM behavior_signals WHERE key = 'lint'"
            ).fetchone()
        self.assertEqual(row[0], "user")


if __name__ == "__main__":
    unittest.main()
