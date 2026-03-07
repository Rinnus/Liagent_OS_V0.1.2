"""Tests for proactive intelligence: behavior signal storage and pattern detection."""
import sqlite3
import tempfile
import os
import unittest
from datetime import datetime, timedelta, timezone


class BehaviorSchemaTests(unittest.TestCase):
    """Verify the 4 new tables exist after LongTermMemory init."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_behavior_signals_table_exists(self):
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        self.assertIn("behavior_signals", tables)

    def test_pending_suggestions_table_exists(self):
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        self.assertIn("pending_suggestions", tables)

    def test_behavior_suppressions_table_exists(self):
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        self.assertIn("behavior_suppressions", tables)

    def test_domain_feedback_table_exists(self):
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        self.assertIn("domain_feedback", tables)

    def test_behavior_signals_json_check(self):
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO behavior_signals "
                    "(signal_type, key, metadata, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    ("topic", "test", "NOT_JSON", "2026-01-01T00:00:00Z"),
                )

    def test_pending_suggestions_partial_unique_index(self):
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        now = "2026-01-01T00:00:00Z"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO pending_suggestions "
                "(pattern_key, domain, suggestion_type, message, action_json, "
                "confidence, net_value, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                ("pk1", "stock", "watch", "msg", '{}', 0.8, 0.5, now, now),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO pending_suggestions "
                    "(pattern_key, domain, suggestion_type, message, action_json, "
                    "confidence, net_value, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                    ("pk1", "stock", "watch", "msg2", '{}', 0.9, 0.6, now, now),
                )

    def test_domain_feedback_composite_pk(self):
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        now = "2026-01-01T00:00:00Z"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO domain_feedback "
                "(domain, suggestion_type, updated_at) VALUES (?, ?, ?)",
                ("stock", "watch", now),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO domain_feedback "
                    "(domain, suggestion_type, updated_at) VALUES (?, ?, ?)",
                    ("stock", "watch", now),
                )

    def test_behavior_signals_indexes_exist(self):
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            indexes = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        self.assertIn("idx_bs_key_created", indexes)
        self.assertIn("idx_bs_created", indexes)


class BehaviorSignalStoreTests(unittest.TestCase):
    """CRUD + queue flush for behavior_signals."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_signal_basic(self):
        from liagent.agent.behavior import BehaviorSignalStore
        store = BehaviorSignalStore(self.db_path)
        store.record("topic", "AAPL", domain="stock", hour=9, weekday=0)
        store.flush()
        signals = store.get_signals("topic", "AAPL", days=30)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["domain"], "stock")

    def test_queue_batches_before_flush(self):
        from liagent.agent.behavior import BehaviorSignalStore
        store = BehaviorSignalStore(self.db_path)
        store.record("topic", "AAPL", domain="stock")
        store.record("topic", "GOOG", domain="stock")
        signals = store.get_signals("topic", "AAPL", days=30)
        self.assertEqual(len(signals), 0)
        store.flush()
        signals = store.get_signals("topic", "AAPL", days=30)
        self.assertEqual(len(signals), 1)

    def test_dedup_same_turn(self):
        from liagent.agent.behavior import BehaviorSignalStore
        store = BehaviorSignalStore(self.db_path)
        store.record("topic", "AAPL", domain="stock", session_id="s1")
        store.record("topic", "AAPL", domain="stock", session_id="s1")
        store.flush()
        signals = store.get_signals("topic", "AAPL", days=30)
        self.assertEqual(len(signals), 1)

    def test_different_sessions_not_deduped(self):
        from liagent.agent.behavior import BehaviorSignalStore
        store = BehaviorSignalStore(self.db_path)
        store.record("topic", "AAPL", domain="stock", session_id="s1")
        store.flush()
        store.record("topic", "AAPL", domain="stock", session_id="s2")
        store.flush()
        signals = store.get_signals("topic", "AAPL", days=30)
        self.assertEqual(len(signals), 2)

    def test_auto_flush_threshold(self):
        from liagent.agent.behavior import BehaviorSignalStore
        store = BehaviorSignalStore(self.db_path, flush_threshold=3)
        store.record("topic", "A", domain="stock")
        store.record("topic", "B", domain="stock")
        store.record("topic", "C", domain="stock")
        signals_a = store.get_signals("topic", "A", days=30)
        self.assertEqual(len(signals_a), 1)

    def test_source_origin_default_user(self):
        from liagent.agent.behavior import BehaviorSignalStore
        store = BehaviorSignalStore(self.db_path)
        store.record("topic", "AAPL", domain="stock")
        store.flush()
        signals = store.get_signals("topic", "AAPL", days=30)
        self.assertEqual(signals[0]["source_origin"], "user")

    def test_source_origin_system(self):
        from liagent.agent.behavior import BehaviorSignalStore
        store = BehaviorSignalStore(self.db_path)
        store.record("topic", "AAPL", domain="stock", source_origin="system")
        store.flush()
        signals = store.get_signals("topic", "AAPL", days=30)
        self.assertEqual(signals[0]["source_origin"], "system")

    def test_prune_old_signals(self):
        from liagent.agent.behavior import BehaviorSignalStore
        store = BehaviorSignalStore(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO behavior_signals "
                "(signal_type, key, domain, source_origin, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("topic", "OLD", "stock", "user", "2025-01-01T00:00:00Z"),
            )
        store.record("topic", "NEW", domain="stock")
        store.flush()
        store.prune(days=90)
        signals_old = store.get_signals("topic", "OLD", days=999)
        signals_new = store.get_signals("topic", "NEW", days=30)
        self.assertEqual(len(signals_old), 0)
        self.assertEqual(len(signals_new), 1)


class DomainFeedbackTests(unittest.TestCase):
    """accept_rate, consecutive_ignored, record methods."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_accept_rate_laplace_no_data(self):
        from liagent.agent.behavior import DomainFeedback
        df = DomainFeedback(self.db_path)
        rate = df.accept_rate("stock", "watch")
        self.assertAlmostEqual(rate, 1 / 2)

    def test_accept_rate_with_data(self):
        from liagent.agent.behavior import DomainFeedback
        df = DomainFeedback(self.db_path)
        df.record_accepted("stock", "watch")
        df.record_accepted("stock", "watch")
        df.record_rejected("stock", "watch")
        rate = df.accept_rate("stock", "watch")
        self.assertAlmostEqual(rate, 3 / 5)

    def test_consecutive_ignored_increments(self):
        from liagent.agent.behavior import DomainFeedback
        df = DomainFeedback(self.db_path)
        df.record_ignored("stock", "watch")
        df.record_ignored("stock", "watch")
        self.assertEqual(df.consecutive_ignored("stock", "watch"), 2)

    def test_consecutive_ignored_resets_on_accept(self):
        from liagent.agent.behavior import DomainFeedback
        df = DomainFeedback(self.db_path)
        df.record_ignored("stock", "watch")
        df.record_ignored("stock", "watch")
        df.record_accepted("stock", "watch")
        self.assertEqual(df.consecutive_ignored("stock", "watch"), 0)

    def test_consecutive_ignored_resets_on_reject(self):
        from liagent.agent.behavior import DomainFeedback
        df = DomainFeedback(self.db_path)
        df.record_ignored("stock", "watch")
        df.record_rejected("stock", "watch")
        self.assertEqual(df.consecutive_ignored("stock", "watch"), 0)

    def test_record_suggested_increments_total(self):
        from liagent.agent.behavior import DomainFeedback
        df = DomainFeedback(self.db_path)
        df.record_suggested("stock", "watch")
        df.record_suggested("stock", "watch")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT total_suggested FROM domain_feedback "
                "WHERE domain=? AND suggestion_type=?",
                ("stock", "watch"),
            ).fetchone()
        self.assertEqual(row[0], 2)

    def test_record_accepted_outcome_does_not_double_count_impression(self):
        from liagent.agent.behavior import DomainFeedback
        df = DomainFeedback(self.db_path)
        df.record_suggested("stock", "watch")
        df.record_accepted_outcome("stock", "watch")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT total_suggested, total_accepted FROM domain_feedback "
                "WHERE domain=? AND suggestion_type=?",
                ("stock", "watch"),
            ).fetchone()
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], 1)

    def test_different_domains_independent(self):
        from liagent.agent.behavior import DomainFeedback
        df = DomainFeedback(self.db_path)
        df.record_accepted("stock", "watch")
        df.record_rejected("news", "briefing")
        self.assertAlmostEqual(df.accept_rate("stock", "watch"), 2 / 3)
        self.assertAlmostEqual(df.accept_rate("news", "briefing"), 1 / 3)


def _insert_signals(db_path, entries):
    """Helper: insert raw behavior_signals rows for testing."""
    with sqlite3.connect(db_path) as conn:
        for e in entries:
            conn.execute(
                "INSERT INTO behavior_signals "
                "(signal_type, key, domain, source_origin, hour, weekday, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    e.get("signal_type", "topic"),
                    e["key"],
                    e.get("domain", "stock"),
                    e.get("source_origin", "user"),
                    e.get("hour", 9),
                    e.get("weekday", 0),
                    e.get("created_at", datetime.now(timezone.utc).isoformat()),
                ),
            )


class PatternDetectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_minimum_evidence_3_mentions_2_days(self):
        from liagent.agent.behavior import BehaviorPatternDetector
        now = datetime.now(timezone.utc)
        _insert_signals(self.db_path, [
            {"key": "AAPL", "created_at": now.isoformat()},
            {"key": "AAPL", "created_at": now.isoformat()},
            {"key": "AAPL", "created_at": (now - timedelta(days=1)).isoformat()},
        ])
        detector = BehaviorPatternDetector(self.db_path)
        candidates = detector.detect()
        keys = [c["pattern_key"] for c in candidates]
        self.assertIn("topic:AAPL", keys)

    def test_below_minimum_evidence_rejected(self):
        from liagent.agent.behavior import BehaviorPatternDetector
        now = datetime.now(timezone.utc)
        _insert_signals(self.db_path, [
            {"key": "AAPL", "created_at": now.isoformat()},
            {"key": "AAPL", "created_at": now.isoformat()},
        ])
        detector = BehaviorPatternDetector(self.db_path)
        candidates = detector.detect()
        self.assertEqual(len(candidates), 0)

    def test_system_origin_excluded(self):
        from liagent.agent.behavior import BehaviorPatternDetector
        now = datetime.now(timezone.utc)
        _insert_signals(self.db_path, [
            {"key": "AAPL", "source_origin": "system", "created_at": now.isoformat()},
            {"key": "AAPL", "source_origin": "system", "created_at": (now - timedelta(days=1)).isoformat()},
            {"key": "AAPL", "source_origin": "system", "created_at": (now - timedelta(days=2)).isoformat()},
        ])
        detector = BehaviorPatternDetector(self.db_path)
        candidates = detector.detect()
        self.assertEqual(len(candidates), 0)

    def test_confidence_in_01_range(self):
        from liagent.agent.behavior import BehaviorPatternDetector
        now = datetime.now(timezone.utc)
        entries = [{"key": "AAPL", "created_at": (now - timedelta(days=d)).isoformat()} for d in range(15)]
        _insert_signals(self.db_path, entries)
        detector = BehaviorPatternDetector(self.db_path)
        candidates = detector.detect()
        for c in candidates:
            self.assertGreaterEqual(c["confidence"], 0.0)
            self.assertLessEqual(c["confidence"], 1.0)

    def test_temporal_regularity_bonus(self):
        from liagent.agent.behavior import BehaviorPatternDetector
        now = datetime.now(timezone.utc)
        # Use older data (15+ days ago) so recency is low and confidence doesn't saturate at 1.0
        base_offset = 15
        regular = [{"key": "REGULAR", "hour": 9, "created_at": (now - timedelta(days=base_offset + d)).isoformat()} for d in range(5)]
        irregular = [{"key": "IRREGULAR", "hour": d * 4 % 24, "created_at": (now - timedelta(days=base_offset + d)).isoformat()} for d in range(5)]
        _insert_signals(self.db_path, regular + irregular)
        detector = BehaviorPatternDetector(self.db_path)
        candidates = detector.detect()
        cmap = {c["pattern_key"]: c["confidence"] for c in candidates}
        self.assertGreater(cmap.get("topic:REGULAR", 0), cmap.get("topic:IRREGULAR", 0))

    def test_confidence_components(self):
        from liagent.agent.behavior import BehaviorPatternDetector
        now = datetime.now(timezone.utc)
        _insert_signals(self.db_path, [
            {"key": "X", "created_at": now.isoformat()},
            {"key": "X", "created_at": (now - timedelta(days=1)).isoformat()},
            {"key": "X", "created_at": (now - timedelta(days=2)).isoformat()},
        ])
        detector = BehaviorPatternDetector(self.db_path)
        candidates = detector.detect()
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertGreater(c["confidence"], 0.3)


class NetValueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fatigue_clamped_to_01(self):
        from liagent.agent.behavior import compute_fatigue
        f = compute_fatigue(today_count=4, daily_limit=5, consecutive_ignored=20)
        self.assertLessEqual(f, 1.0)
        self.assertGreaterEqual(f, 0.0)

    def test_fatigue_zero_when_fresh(self):
        from liagent.agent.behavior import compute_fatigue
        f = compute_fatigue(today_count=0, daily_limit=5, consecutive_ignored=0)
        self.assertAlmostEqual(f, 0.0)

    def test_fatigue_consecutive_penalty(self):
        from liagent.agent.behavior import compute_fatigue
        f0 = compute_fatigue(today_count=1, daily_limit=5, consecutive_ignored=0)
        f3 = compute_fatigue(today_count=1, daily_limit=5, consecutive_ignored=3)
        self.assertGreater(f3, f0)

    def test_route_suppressed_skip(self):
        from liagent.agent.behavior import ProactiveActionRouter, RoutingContext
        router = ProactiveActionRouter(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO behavior_suppressions VALUES (?, ?, ?, ?)",
                ("topic:AAPL", "stock",
                 (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(), "user rejected"),
            )
        candidate = {"pattern_key": "topic:AAPL", "domain": "stock", "confidence": 0.9, "is_read_only": True}
        ctx = RoutingContext(in_complex_task=False, in_quiet_hours=False, today_touch_count=0, daily_limit=5)
        self.assertEqual(router.route(candidate, ctx), "skip")

    def test_route_unauthorized_suggest_only(self):
        from liagent.agent.behavior import ProactiveActionRouter, RoutingContext
        router = ProactiveActionRouter(self.db_path, authorization={"coding": "suggest"})
        candidate = {"pattern_key": "intent:refactor", "domain": "coding", "confidence": 0.9, "is_read_only": False}
        ctx = RoutingContext(in_complex_task=False, in_quiet_hours=False, today_touch_count=0, daily_limit=5)
        self.assertEqual(router.route(candidate, ctx), "suggest_only")

    def test_route_complex_task_defer(self):
        from liagent.agent.behavior import ProactiveActionRouter, RoutingContext
        router = ProactiveActionRouter(self.db_path, authorization={"stock": "auto"})
        candidate = {"pattern_key": "topic:AAPL", "domain": "stock", "confidence": 0.9, "is_read_only": True}
        ctx = RoutingContext(in_complex_task=True, in_quiet_hours=False, today_touch_count=0, daily_limit=5)
        self.assertEqual(router.route(candidate, ctx), "defer")

    def test_route_quiet_hours_defer(self):
        from liagent.agent.behavior import ProactiveActionRouter, RoutingContext
        router = ProactiveActionRouter(self.db_path, authorization={"stock": "auto"})
        candidate = {"pattern_key": "topic:AAPL", "domain": "stock", "confidence": 0.9, "is_read_only": True}
        ctx = RoutingContext(in_complex_task=False, in_quiet_hours=True, today_touch_count=0, daily_limit=5)
        self.assertEqual(router.route(candidate, ctx), "defer")

    def test_route_high_conf_read_only_auto(self):
        from liagent.agent.behavior import ProactiveActionRouter, RoutingContext
        router = ProactiveActionRouter(self.db_path, authorization={"stock": "auto"}, t_auto=0.0)
        candidate = {"pattern_key": "topic:AAPL", "domain": "stock", "confidence": 0.9, "is_read_only": True}
        ctx = RoutingContext(in_complex_task=False, in_quiet_hours=False, today_touch_count=0, daily_limit=5)
        self.assertEqual(router.route(candidate, ctx), "auto_create")

    def test_route_execution_task_never_auto(self):
        from liagent.agent.behavior import ProactiveActionRouter, RoutingContext
        router = ProactiveActionRouter(self.db_path, authorization={"stock": "auto"}, t_auto=0.0)
        candidate = {"pattern_key": "intent:exec", "domain": "stock", "confidence": 0.9, "is_read_only": False}
        ctx = RoutingContext(in_complex_task=False, in_quiet_hours=False, today_touch_count=0, daily_limit=5)
        self.assertNotEqual(router.route(candidate, ctx), "auto_create")

    def test_hysteresis_enter_exit(self):
        from liagent.agent.behavior import ProactiveActionRouter, RoutingContext
        router = ProactiveActionRouter(self.db_path, authorization={"stock": "auto"}, t_auto=-999)
        ctx = RoutingContext(in_complex_task=False, in_quiet_hours=False, today_touch_count=0, daily_limit=5)
        # conf=0.79 -> no auto
        c1 = {"pattern_key": "topic:X", "domain": "stock", "confidence": 0.79, "is_read_only": True}
        self.assertNotEqual(router.route(c1, ctx), "auto_create")
        # conf=0.80 -> auto (enters)
        c2 = {"pattern_key": "topic:X", "domain": "stock", "confidence": 0.80, "is_read_only": True}
        self.assertEqual(router.route(c2, ctx), "auto_create")
        # conf=0.71 -> still auto (hysteresis)
        c3 = {"pattern_key": "topic:X", "domain": "stock", "confidence": 0.71, "is_read_only": True}
        self.assertEqual(router.route(c3, ctx), "auto_create")
        # conf=0.69 -> exit auto
        c4 = {"pattern_key": "topic:X", "domain": "stock", "confidence": 0.69, "is_read_only": True}
        self.assertNotEqual(router.route(c4, ctx), "auto_create")

    def test_expired_suppression_not_skipped(self):
        from liagent.agent.behavior import ProactiveActionRouter, RoutingContext
        router = ProactiveActionRouter(self.db_path, authorization={"stock": "auto"}, t_auto=0.0)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO behavior_suppressions VALUES (?, ?, ?, ?)",
                ("topic:AAPL", "stock",
                 (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(), "old"),
            )
        candidate = {"pattern_key": "topic:AAPL", "domain": "stock", "confidence": 0.9, "is_read_only": True}
        ctx = RoutingContext(in_complex_task=False, in_quiet_hours=False, today_touch_count=0, daily_limit=5)
        self.assertNotEqual(router.route(candidate, ctx), "skip")


class PendingSuggestionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_and_get_pending(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(pattern_key="topic:AAPL", domain="stock", suggestion_type="watch",
                  message="Monitor AAPL?", action_json='{"type":"watch"}', confidence=0.8, net_value=0.5)
        pending = store.get_pending(max_items=5)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["pattern_key"], "topic:AAPL")

    def test_partial_unique_blocks_duplicate_active(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(pattern_key="topic:AAPL", domain="stock", suggestion_type="watch",
                  message="msg1", action_json='{}', confidence=0.8, net_value=0.5)
        added = store.add(pattern_key="topic:AAPL", domain="stock", suggestion_type="watch",
                         message="msg2", action_json='{}', confidence=0.9, net_value=0.6)
        self.assertFalse(added)

    def test_mark_shown_increments_atomically(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(pattern_key="topic:AAPL", domain="stock", suggestion_type="watch",
                  message="msg", action_json='{}', confidence=0.8, net_value=0.5)
        pending = store.get_pending(max_items=5)
        sid = pending[0]["id"]
        store.mark_shown(sid)
        store.mark_shown(sid)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT sessions_shown FROM pending_suggestions WHERE id=?", (sid,)).fetchone()
        self.assertEqual(row[0], 2)

    def test_mark_shown_sets_next_eligible_at_when_cooldown_requested(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(pattern_key="topic:AAPL", domain="stock", suggestion_type="watch",
                  message="msg", action_json='{}', confidence=0.8, net_value=0.5)
        pending = store.get_pending(max_items=5)
        sid = pending[0]["id"]
        store.mark_shown(sid, cooldown_sec=300)
        self.assertEqual(store.get_pending(max_items=5), [])
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT next_eligible_at, sessions_shown FROM pending_suggestions WHERE id=?",
                (sid,),
            ).fetchone()
        self.assertIsNotNone(row[0])
        self.assertEqual(row[1], 1)

    def test_expire_after_3_shown(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(pattern_key="topic:AAPL", domain="stock", suggestion_type="watch",
                  message="msg", action_json='{}', confidence=0.8, net_value=0.5)
        pending = store.get_pending(max_items=5)
        sid = pending[0]["id"]
        store.mark_shown(sid)
        store.mark_shown(sid)
        store.mark_shown(sid)
        store.expire_stale()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT status FROM pending_suggestions WHERE id=?", (sid,)).fetchone()
        self.assertEqual(row[0], "expired")

    def test_get_pending_respects_next_eligible_at(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        now = datetime.now(timezone.utc)
        future = (now + timedelta(hours=2)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO pending_suggestions "
                "(pattern_key, domain, suggestion_type, message, action_json, "
                "confidence, net_value, next_eligible_at, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                ("topic:FUTURE", "stock", "watch", "msg", '{}', 0.8, 0.5, future, now.isoformat(), now.isoformat()),
            )
        pending = store.get_pending(max_items=5)
        self.assertEqual(len(pending), 0)

    def test_update_status(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(pattern_key="topic:AAPL", domain="stock", suggestion_type="watch",
                  message="msg", action_json='{}', confidence=0.8, net_value=0.5)
        pending = store.get_pending(max_items=5)
        sid = pending[0]["id"]
        store.update_status(sid, "accepted")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT status FROM pending_suggestions WHERE id=?", (sid,)).fetchone()
        self.assertEqual(row[0], "accepted")

    def test_order_by_net_value_desc(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(pattern_key="topic:LOW", domain="stock", suggestion_type="watch",
                  message="low", action_json='{}', confidence=0.5, net_value=0.1)
        store.add(pattern_key="topic:HIGH", domain="stock", suggestion_type="watch",
                  message="high", action_json='{}', confidence=0.9, net_value=0.8)
        pending = store.get_pending(max_items=5)
        self.assertEqual(pending[0]["pattern_key"], "topic:HIGH")

    def test_get_pending_filters_by_target_session(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(
            pattern_key="topic:GLOBAL", domain="stock", suggestion_type="watch",
            message="global", action_json='{}', confidence=0.5, net_value=0.2,
        )
        store.add(
            pattern_key="topic:SESSION", domain="stock", suggestion_type="watch",
            message="session", action_json='{}', confidence=0.9, net_value=0.8,
            target_session_id="session-a",
        )

        pending_a = store.get_pending(max_items=5, session_id="session-a")
        pending_b = store.get_pending(max_items=5, session_id="session-b")
        pending_none = store.get_pending(max_items=5)

        self.assertEqual(
            [item["pattern_key"] for item in pending_a],
            ["topic:SESSION", "topic:GLOBAL"],
        )
        self.assertEqual([item["pattern_key"] for item in pending_b], ["topic:GLOBAL"])
        self.assertEqual([item["pattern_key"] for item in pending_none], ["topic:GLOBAL"])


class ProactiveConfigTests(unittest.TestCase):
    def test_default_config(self):
        from liagent.config import ProactiveConfig
        cfg = ProactiveConfig()
        self.assertEqual(cfg.authorization["stock"], "auto")
        self.assertEqual(cfg.authorization["coding"], "suggest")
        self.assertEqual(cfg.daily_touch_limit, 5)
        self.assertEqual(cfg.suppression_days, 30)

    def test_load_from_yaml(self):
        from liagent.config import ProactiveConfig
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "proactive.yaml")
            with open(p, "w") as f:
                f.write("authorization:\n  stock: suggest\ndaily_touch_limit: 3\n")
            cfg = ProactiveConfig.load(Path(p))
            self.assertEqual(cfg.authorization["stock"], "suggest")
            self.assertEqual(cfg.daily_touch_limit, 3)

    def test_load_missing_file_returns_defaults(self):
        from liagent.config import ProactiveConfig
        from pathlib import Path
        cfg = ProactiveConfig.load(Path("/nonexistent/proactive.yaml"))
        self.assertEqual(cfg.daily_touch_limit, 5)

    def test_quiet_hours_parsing(self):
        from liagent.config import ProactiveConfig
        cfg = ProactiveConfig(quiet_hours="23:00-07:00")
        self.assertEqual(cfg.quiet_hours, "23:00-07:00")


class L0ToolSignalTests(unittest.TestCase):
    def test_tool_domain_map_coverage(self):
        from liagent.agent.behavior import TOOL_DOMAIN_MAP
        self.assertEqual(TOOL_DOMAIN_MAP["stock"], "stock")
        self.assertEqual(TOOL_DOMAIN_MAP["web_search"], "news")
        self.assertEqual(TOOL_DOMAIN_MAP["python_exec"], "coding")

    def test_record_tool_signal(self):
        from liagent.agent.behavior import BehaviorSignalStore, TOOL_DOMAIN_MAP
        tmp = tempfile.TemporaryDirectory()
        db_path = os.path.join(tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=db_path)
        store = BehaviorSignalStore(db_path)
        tool_name = "stock"
        domain = TOOL_DOMAIN_MAP.get(tool_name, "general")
        store.record("tool_use", tool_name, domain=domain)
        store.flush()
        signals = store.get_signals("tool_use", tool_name, days=30)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["domain"], "stock")
        tmp.cleanup()

    def test_topic_extraction_stock_ticker(self):
        from liagent.agent.behavior import extract_topics_from_tool_args
        topics = extract_topics_from_tool_args("stock", {"symbol": "AAPL"})
        self.assertIn("AAPL", topics)

    def test_topic_extraction_web_search(self):
        from liagent.agent.behavior import extract_topics_from_tool_args
        topics = extract_topics_from_tool_args("web_search", {"query": "Apple stock price"})
        self.assertIsInstance(topics, list)


class L1BehaviorExtractionTests(unittest.TestCase):
    def test_behavior_extraction_prompt_has_system_msg(self):
        from liagent.agent.prompt_builder import PromptBuilder
        from liagent.agent.memory import LongTermMemory
        tmp = tempfile.TemporaryDirectory()
        db_path = os.path.join(tmp.name, "test.db")
        ltm = LongTermMemory(db_path=db_path)
        pb = PromptBuilder(ltm)
        msgs = [
            {"role": "user", "content": "What is the AAPL stock price?"},
            {"role": "assistant", "content": "AAPL is trading at $180."},
        ]
        result = pb.build_behavior_extraction_prompt(msgs)
        self.assertTrue(any(m["role"] == "system" for m in result))
        tmp.cleanup()

    def test_parse_behavior_signals_valid_json(self):
        from liagent.agent.behavior import parse_behavior_signals
        raw = '[{"signal_type":"intent","key":"check_stock","domain":"stock","confidence":0.8,"metadata":{"reason":"asked twice"}}]'
        signals = parse_behavior_signals(raw)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["key"], "check_stock")

    def test_parse_behavior_signals_empty(self):
        from liagent.agent.behavior import parse_behavior_signals
        signals = parse_behavior_signals("[]")
        self.assertEqual(len(signals), 0)

    def test_parse_behavior_signals_filters_low_confidence(self):
        from liagent.agent.behavior import parse_behavior_signals
        raw = '[{"signal_type":"intent","key":"low","domain":"general","confidence":0.3,"metadata":{}}]'
        signals = parse_behavior_signals(raw)
        self.assertEqual(len(signals), 0)

    def test_parse_behavior_signals_max_6(self):
        from liagent.agent.behavior import parse_behavior_signals
        entries = [
            f'{{"signal_type":"intent","key":"k{i}","domain":"general","confidence":0.9,"metadata":{{}}}}'
            for i in range(10)
        ]
        raw = "[" + ",".join(entries) + "]"
        signals = parse_behavior_signals(raw)
        self.assertLessEqual(len(signals), 6)

    def test_parse_behavior_signals_strips_fences(self):
        from liagent.agent.behavior import parse_behavior_signals
        raw = '```json\n[{"signal_type":"intent","key":"x","domain":"stock","confidence":0.8,"metadata":{}}]\n```'
        signals = parse_behavior_signals(raw)
        self.assertEqual(len(signals), 1)


class SuggestionInjectionTests(unittest.TestCase):
    """Verify pending suggestions store works for injection."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pending_suggestion_available(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(
            pattern_key="topic:AAPL", domain="stock",
            suggestion_type="watch", message="Would you like me to monitor AAPL?",
            action_json='{"type":"watch","entity":"AAPL"}',
            confidence=0.8, net_value=0.5,
        )
        pending = store.get_pending(max_items=2)
        self.assertEqual(len(pending), 1)
        self.assertIn("AAPL", pending[0]["message"])

    def test_mark_shown_after_injection(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(
            pattern_key="topic:AAPL", domain="stock",
            suggestion_type="watch", message="Monitor AAPL?",
            action_json='{}', confidence=0.8, net_value=0.5,
        )
        pending = store.get_pending(max_items=2)
        store.mark_shown(pending[0]["id"])
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT sessions_shown FROM pending_suggestions WHERE id=?",
                (pending[0]["id"],)
            ).fetchone()
        self.assertEqual(row[0], 1)


class HygieneTests(unittest.TestCase):
    """Verify behavior tables are pruned via store methods."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prune_old_signals(self):
        from liagent.agent.behavior import BehaviorSignalStore
        store = BehaviorSignalStore(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO behavior_signals "
                "(signal_type, key, domain, source_origin, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("topic", "OLD", "stock", "user", "2024-01-01T00:00:00Z"),
            )
        deleted = store.prune(days=90)
        self.assertGreaterEqual(deleted, 1)

    def test_prune_expired_suppressions(self):
        from liagent.agent.behavior import ProactiveActionRouter
        router = ProactiveActionRouter(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO behavior_suppressions VALUES (?, ?, ?, ?)",
                ("topic:OLD", "stock", "2024-01-01T00:00:00Z", "old"),
            )
        deleted = router.clean_expired_suppressions()
        self.assertGreaterEqual(deleted, 1)

    def test_expire_stale_suggestions(self):
        from liagent.agent.behavior import PendingSuggestionStore
        store = PendingSuggestionStore(self.db_path)
        store.add(
            pattern_key="topic:STALE", domain="stock",
            suggestion_type="watch", message="stale",
            action_json='{}', confidence=0.5, net_value=0.1,
        )
        pending = store.get_pending(max_items=5)
        sid = pending[0]["id"]
        store.mark_shown(sid)
        store.mark_shown(sid)
        store.mark_shown(sid)
        expired = store.expire_stale()
        self.assertGreaterEqual(len(expired), 1)


class EndToEndTests(unittest.TestCase):
    """Full pipeline: signals -> detect -> route -> suggest."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_pipeline_topic_to_suggestion(self):
        """Record signals -> detect pattern -> route -> create suggestion."""
        from liagent.agent.behavior import (
            BehaviorSignalStore,
            BehaviorPatternDetector,
            ProactiveActionRouter,
            PendingSuggestionStore,
            DomainFeedback,
            RoutingContext,
        )
        now = datetime.now(timezone.utc)

        # Simulate 5 days of AAPL queries
        for d in range(5):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO behavior_signals "
                    "(signal_type, key, domain, source_origin, hour, weekday, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("topic", "AAPL", "stock", "user", 9, d % 7,
                     (now - timedelta(days=d)).isoformat()),
                )

        # Detect
        detector = BehaviorPatternDetector(self.db_path)
        candidates = detector.detect()
        self.assertTrue(len(candidates) > 0)

        # Route
        router = ProactiveActionRouter(
            self.db_path,
            authorization={"stock": "auto"},
            t_auto=0.0,
        )
        ctx = RoutingContext(today_touch_count=0, daily_limit=5)
        best = candidates[0]
        best["is_read_only"] = True
        best["suggestion_type"] = "watch"
        decision = router.route(best, ctx)
        self.assertIn(decision, ("auto_create", "suggest"))

        # Create suggestion
        sug_store = PendingSuggestionStore(self.db_path)
        added = sug_store.add(
            pattern_key=best["pattern_key"], domain=best["domain"],
            suggestion_type="watch",
            message=f"You check {best['key']} often. Want me to monitor it automatically?",
            action_json='{"type":"watch"}',
            confidence=best["confidence"], net_value=0.5,
        )
        self.assertTrue(added)
        pending = sug_store.get_pending(max_items=5)
        self.assertEqual(len(pending), 1)

    def test_rejection_creates_suppression(self):
        """User rejects -> 30-day suppression + feedback recorded."""
        from liagent.agent.behavior import (
            ProactiveActionRouter,
            PendingSuggestionStore,
            DomainFeedback,
            RoutingContext,
        )
        sug_store = PendingSuggestionStore(self.db_path)
        sug_store.add(
            pattern_key="topic:AAPL", domain="stock",
            suggestion_type="watch", message="Monitor?",
            action_json='{}', confidence=0.8, net_value=0.5,
        )
        pending = sug_store.get_pending(max_items=5)
        sid = pending[0]["id"]

        # Reject
        sug_store.update_status(sid, "rejected")
        router = ProactiveActionRouter(self.db_path)
        router.add_suppression("topic:AAPL", "stock", days=30, reason="user rejected")
        df = DomainFeedback(self.db_path)
        df.record_rejected("stock", "watch")

        # Verify suppression active
        candidate = {"pattern_key": "topic:AAPL", "domain": "stock",
                    "confidence": 0.9, "is_read_only": True}
        ctx = RoutingContext(today_touch_count=0, daily_limit=5)
        self.assertEqual(router.route(candidate, ctx), "skip")

    def test_pattern_detector_separates_session_scopes(self):
        from liagent.agent.behavior import BehaviorPatternDetector

        now = datetime.now(timezone.utc)
        for session_id in ("session-a", "session-b"):
            for day in range(3):
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "INSERT INTO behavior_signals "
                        "(signal_type, key, domain, source_origin, session_id, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            "topic",
                            "AAPL",
                            "stock",
                            "user",
                            session_id,
                            (now - timedelta(days=day)).isoformat(),
                        ),
                    )

        detector = BehaviorPatternDetector(self.db_path)
        candidates = detector.detect()
        scoped = sorted(
            (
                item["target_session_id"],
                item["pattern_key"],
            )
            for item in candidates
            if item["key"] == "AAPL"
        )
        self.assertEqual(
            scoped,
            [
                ("session-a", "topic:AAPL:session-a"),
                ("session-b", "topic:AAPL:session-b"),
            ],
        )

    def test_ignore_3x_creates_short_cooldown(self):
        """3x ignored -> expired + 7-day short cooldown."""
        from liagent.agent.behavior import (
            PendingSuggestionStore,
            ProactiveActionRouter,
            DomainFeedback,
            RoutingContext,
        )
        sug_store = PendingSuggestionStore(self.db_path)
        sug_store.add(
            pattern_key="topic:TSLA", domain="stock",
            suggestion_type="watch", message="Monitor TSLA?",
            action_json='{}', confidence=0.7, net_value=0.3,
        )
        pending = sug_store.get_pending(max_items=5)
        sid = pending[0]["id"]
        sug_store.mark_shown(sid)
        sug_store.mark_shown(sid)
        sug_store.mark_shown(sid)
        sug_store.expire_stale()

        # Record ignore + short cooldown
        df = DomainFeedback(self.db_path)
        df.record_ignored("stock", "watch")
        df.record_ignored("stock", "watch")
        df.record_ignored("stock", "watch")
        self.assertEqual(df.consecutive_ignored("stock", "watch"), 3)

        router = ProactiveActionRouter(self.db_path)
        router.add_suppression("topic:TSLA", "stock", days=7, reason="3x ignored")
        candidate = {"pattern_key": "topic:TSLA", "domain": "stock",
                    "confidence": 0.9, "is_read_only": True}
        ctx = RoutingContext(today_touch_count=0, daily_limit=5)
        self.assertEqual(router.route(candidate, ctx), "skip")


class QuietHoursTests(unittest.TestCase):
    """Verify is_quiet_hours parsing and time checking."""

    def test_empty_string_returns_false(self):
        from liagent.agent.behavior import is_quiet_hours
        self.assertFalse(is_quiet_hours(""))
        self.assertFalse(is_quiet_hours(None))

    def test_invalid_format_returns_false(self):
        from liagent.agent.behavior import is_quiet_hours
        self.assertFalse(is_quiet_hours("invalid"))
        self.assertFalse(is_quiet_hours("25:00-07:00"))

    def test_wrap_around_range(self):
        """23:00-07:00 should handle midnight wrap."""
        from liagent.agent.behavior import is_quiet_hours
        from unittest.mock import patch, MagicMock

        # Mock 2am local time → should be quiet
        mock_local = MagicMock(hour=2, minute=0)
        mock_now = MagicMock()
        mock_now.astimezone.return_value = mock_local
        with patch('liagent.agent.behavior.datetime') as mock_dt:
            mock_dt.now.return_value = mock_now
            result = is_quiet_hours("23:00-07:00")
            self.assertTrue(result)

    def test_non_wrap_range(self):
        """09:00-17:00 normal daytime range."""
        from liagent.agent.behavior import is_quiet_hours
        from unittest.mock import patch, MagicMock

        # Mock 12pm → should be quiet (within 09:00-17:00)
        mock_local = MagicMock(hour=12, minute=0)
        mock_now = MagicMock()
        mock_now.astimezone.return_value = mock_local
        with patch('liagent.agent.behavior.datetime') as mock_dt:
            mock_dt.now.return_value = mock_now
            result = is_quiet_hours("09:00-17:00")
            self.assertTrue(result)

    def test_outside_quiet_hours(self):
        """20:00 is outside 23:00-07:00."""
        from liagent.agent.behavior import is_quiet_hours
        from unittest.mock import patch, MagicMock

        mock_local = MagicMock(hour=20, minute=0)
        mock_now = MagicMock()
        mock_now.astimezone.return_value = mock_local
        with patch('liagent.agent.behavior.datetime') as mock_dt:
            mock_dt.now.return_value = mock_now
            result = is_quiet_hours("23:00-07:00")
            self.assertFalse(result)


class HeartbeatPatternPipelineTests(unittest.TestCase):
    """Verify HeartbeatRunner Phase 4: detect → route → suggest."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_heartbeat_creates_suggestion_from_pattern(self):
        """Full heartbeat Phase 4: detect pattern → route → create suggestion."""
        import asyncio
        from liagent.agent.behavior import (
            BehaviorPatternDetector,
            ProactiveActionRouter,
            PendingSuggestionStore,
        )
        from liagent.agent.heartbeat import HeartbeatRunner, HeartbeatConfig, CursorStore
        from pathlib import Path
        from datetime import timedelta

        # Seed signals: 5 days of TSLA queries
        now = datetime.now(timezone.utc)
        for d in range(5):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO behavior_signals "
                    "(signal_type, key, domain, source_origin, hour, weekday, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("topic", "TSLA", "stock", "user", 10, d % 7,
                     (now - timedelta(days=d)).isoformat()),
                )

        detector = BehaviorPatternDetector(self.db_path)
        router = ProactiveActionRouter(
            self.db_path, authorization={"stock": "auto"}, t_auto=0.0,
        )
        sug_store = PendingSuggestionStore(self.db_path)
        cursor_db = os.path.join(self.tmp.name, "cursor.db")
        cursor_store = CursorStore(db_path=Path(cursor_db))

        actions_logged = []
        hb = HeartbeatRunner(
            config=HeartbeatConfig(instructions="test"),
            engine=None,
            long_term_memory=self.ltm,
            notification_router=None,
            cursor_store=cursor_store,
            pattern_detector=detector,
            proactive_router=router,
            suggestion_store=sug_store,
            on_action=lambda msg: actions_logged.append(msg),
        )

        metrics = asyncio.run(hb.run())
        # auto_create suggestions go to delivery_mode='auto' (consumed by bridge)
        pending = sug_store.get_by_delivery_mode("auto")
        self.assertGreaterEqual(len(pending), 1)
        self.assertIn("TSLA", pending[0]["message"])
        self.assertTrue(len(actions_logged) > 0)


class ExtractToolFactTests(unittest.TestCase):
    """Tests for zero-LLM fact extraction from tool calls."""

    def test_stock_extraction(self):
        from liagent.agent.tool_orchestrator import _extract_tool_fact
        result = _extract_tool_fact("stock", {"symbol": "AAPL"})
        self.assertIsNotNone(result)
        self.assertEqual(result["fact_key"], "tool:stock:AAPL")
        self.assertIn("AAPL", result["fact"])
        self.assertEqual(result["category"], "interest")

    def test_stock_empty_symbol(self):
        from liagent.agent.tool_orchestrator import _extract_tool_fact
        self.assertIsNone(_extract_tool_fact("stock", {"symbol": ""}))
        self.assertIsNone(_extract_tool_fact("stock", {}))

    def test_web_search_extraction(self):
        from liagent.agent.tool_orchestrator import _extract_tool_fact
        result = _extract_tool_fact("web_search", {"query": "Python async"})
        self.assertIsNotNone(result)
        self.assertTrue(result["fact_key"].startswith("tool:web_search:"))
        self.assertIn("Python async", result["fact"])
        self.assertEqual(result["category"], "interest")

    def test_web_search_dedup_hash(self):
        from liagent.agent.tool_orchestrator import _extract_tool_fact
        r1 = _extract_tool_fact("web_search", {"query": "Python async"})
        r2 = _extract_tool_fact("web_search", {"query": "Python async"})
        self.assertEqual(r1["fact_key"], r2["fact_key"])
        # Different query → different key
        r3 = _extract_tool_fact("web_search", {"query": "Rust async"})
        self.assertNotEqual(r1["fact_key"], r3["fact_key"])

    def test_web_fetch_extraction(self):
        from liagent.agent.tool_orchestrator import _extract_tool_fact
        result = _extract_tool_fact("web_fetch", {"url": "https://docs.python.org/3/library/"})
        self.assertIsNotNone(result)
        self.assertEqual(result["fact_key"], "tool:web_fetch:docs.python.org")
        self.assertEqual(result["category"], "reference")

    def test_web_fetch_skips_low_value(self):
        from liagent.agent.tool_orchestrator import _extract_tool_fact
        self.assertIsNone(_extract_tool_fact("web_fetch", {"url": "https://www.google.com/search?q=test"}))
        self.assertIsNone(_extract_tool_fact("web_fetch", {"url": "https://youtube.com/watch?v=abc"}))

    def test_unknown_tool_returns_none(self):
        from liagent.agent.tool_orchestrator import _extract_tool_fact
        self.assertIsNone(_extract_tool_fact("python_exec", {"code": "print(1)"}))
        self.assertIsNone(_extract_tool_fact("run_tests", {}))


class UpsertToolFactTests(unittest.TestCase):
    """Tests for LongTermMemory.upsert_tool_fact() dedup and confidence."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_insert_new_fact(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        ltm.upsert_tool_fact(
            fact="The user follows AAPL stock",
            fact_key="tool:stock:AAPL",
            category="interest",
        )
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT fact, confidence, source, category FROM key_facts WHERE fact_key = ?",
                ("tool:stock:AAPL",)
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "The user follows AAPL stock")
        self.assertAlmostEqual(row[1], 0.60, places=2)
        self.assertEqual(row[2], "tool_extract")
        self.assertEqual(row[3], "interest")

    def test_upsert_bumps_confidence(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        ltm.upsert_tool_fact(fact="The user follows AAPL", fact_key="tool:stock:AAPL")
        ltm.upsert_tool_fact(fact="The user follows AAPL", fact_key="tool:stock:AAPL")
        with sqlite3.connect(self.db_path) as conn:
            conf = conn.execute(
                "SELECT confidence FROM key_facts WHERE fact_key = ?",
                ("tool:stock:AAPL",)
            ).fetchone()[0]
        self.assertAlmostEqual(conf, 0.65, places=2)

    def test_confidence_caps_at_090(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        # Insert + 7 upserts → 0.60 + 7*0.05 = 0.95, but capped at 0.90
        for _ in range(8):
            ltm.upsert_tool_fact(fact="The user follows TSLA", fact_key="tool:stock:TSLA")
        with sqlite3.connect(self.db_path) as conn:
            conf = conn.execute(
                "SELECT confidence FROM key_facts WHERE fact_key = ?",
                ("tool:stock:TSLA",)
            ).fetchone()[0]
        self.assertAlmostEqual(conf, 0.90, places=2)

    def test_no_duplicate_rows(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        ltm.upsert_tool_fact(fact="The user follows AAPL", fact_key="tool:stock:AAPL")
        ltm.upsert_tool_fact(fact="The user follows AAPL", fact_key="tool:stock:AAPL")
        ltm.upsert_tool_fact(fact="The user follows AAPL", fact_key="tool:stock:AAPL")
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM key_facts WHERE fact_key = ?",
                ("tool:stock:AAPL",)
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_fts_index_created_for_new_fact(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        ltm.upsert_tool_fact(fact="The user searched for Python", fact_key="tool:web_search:abc123")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM key_facts WHERE fact_key = ?",
                ("tool:web_search:abc123",)
            ).fetchone()
            fts_row = conn.execute(
                "SELECT rowid FROM key_facts_fts WHERE rowid = ?",
                (row[0],)
            ).fetchone()
        self.assertIsNotNone(fts_row)


class BridgeMethodTests(unittest.TestCase):
    """Tests for get_by_delivery_mode, try_claim, add_simple."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _store(self):
        from liagent.agent.behavior import PendingSuggestionStore
        return PendingSuggestionStore(self.db_path)

    def _add_auto_suggestion(self, store, domain="stock"):
        return store.add(
            pattern_key=f"test:{domain}:1", domain=domain,
            suggestion_type="monitoring", message="Test suggestion",
            action_json='{"type":"display"}', confidence=0.8,
            net_value=0.6, delivery_mode="auto",
        )

    def test_get_by_delivery_mode_filters_correctly(self):
        store = self._store()
        self._add_auto_suggestion(store, "stock")
        store.add(pattern_key="test:tech:1", domain="tech",
            suggestion_type="monitoring", message="Session sug",
            action_json='{"type":"display"}', confidence=0.5,
            net_value=0.4, delivery_mode="session")
        results = store.get_by_delivery_mode("auto")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["domain"], "stock")

    def test_get_by_delivery_mode_empty(self):
        store = self._store()
        self.assertEqual(len(store.get_by_delivery_mode("auto")), 0)

    def test_try_claim_success(self):
        store = self._store()
        self._add_auto_suggestion(store)
        items = store.get_by_delivery_mode("auto")
        self.assertTrue(store.try_claim(items[0]["id"]))

    def test_try_claim_prevents_double_claim(self):
        store = self._store()
        self._add_auto_suggestion(store)
        items = store.get_by_delivery_mode("auto")
        sid = items[0]["id"]
        self.assertTrue(store.try_claim(sid))
        self.assertFalse(store.try_claim(sid))

    def test_try_claim_nonexistent(self):
        store = self._store()
        self.assertFalse(store.try_claim(99999))

    def test_add_simple_creates_pending_suggestion(self):
        store = self._store()
        ok = store.add_simple(message="Hello from goal", domain="finance")
        self.assertTrue(ok)
        items = store.get_pending(max_items=10)
        self.assertEqual(len(items), 1)
        self.assertIn("Hello from goal", items[0]["message"])

    def test_add_simple_auto_mode(self):
        store = self._store()
        store.add_simple(message="Auto msg", delivery_mode="auto", domain="tech")
        items = store.get_by_delivery_mode("auto")
        self.assertEqual(len(items), 1)


if __name__ == "__main__":
    unittest.main()
