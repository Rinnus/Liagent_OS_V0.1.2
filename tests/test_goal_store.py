# tests/test_goal_store.py
import json
import os
import sqlite3
import tempfile
import unittest


class GoalStoreSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_tables_created(self):
        from liagent.agent.goal_store import GoalStore
        GoalStore(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        self.assertIn("autonomous_goals", tables)
        self.assertIn("goal_events", tables)
        self.assertIn("pattern_groups", tables)
        self.assertIn("decision_outbox", tables)

    def test_indexes_created(self):
        from liagent.agent.goal_store import GoalStore
        GoalStore(self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            indexes = {r[1] for r in conn.execute(
                "SELECT * FROM sqlite_master WHERE type='index'"
            ).fetchall() if r[1]}
        self.assertIn("idx_goals_idemp_active", indexes)
        self.assertIn("idx_goals_state_review", indexes)
        self.assertIn("idx_goal_events_goal_created", indexes)
        self.assertIn("idx_pattern_groups_last_seen", indexes)
        self.assertIn("idx_outbox_pending", indexes)


class GoalCRUDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.goal_store import GoalStore
        self.store = GoalStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_goal(self):
        gid = self.store.create(
            source="reflection_insight", domain="finance",
            objective="Track AAPL", rationale="Frequent queries", confidence=0.8,
        )
        self.assertIsInstance(gid, int)
        self.assertGreater(gid, 0)

    def test_create_idempotent(self):
        g1 = self.store.create(source="test", objective="A", idempotency_key="test:key:1")
        g2 = self.store.create(source="test", objective="B", idempotency_key="test:key:1")
        self.assertIsNotNone(g1)
        self.assertIsNone(g2)

    def test_get_active(self):
        self.store.create(source="test", objective="A")
        self.store.create(source="test", objective="B")
        self.store.transition(1, "active")
        active = self.store.get_by_state("active")
        self.assertEqual(len(active), 1)

    def test_count_active(self):
        self.store.create(source="test", objective="A")
        self.store.transition(1, "active")
        self.assertEqual(self.store.count_active(), 1)

    def test_transition_to_retired(self):
        self.store.create(source="test", objective="A")
        self.store.transition(1, "active")
        self.store.transition(1, "retired", reason="No longer relevant")
        goal = self.store.get(1)
        self.assertEqual(goal["state"], "retired")
        self.assertEqual(goal["retired_reason"], "No longer relevant")

    def test_record_event(self):
        gid = self.store.create(source="test", objective="A")
        self.store.record_event(gid, "created", {"test": True}, summary="Goal created")
        events = self.store.get_events(gid)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "created")

    def test_record_global_event(self):
        self.store.record_event(None, "observation", {"text": "test"})
        events = self.store.get_events(None, event_type="observation")
        self.assertEqual(len(events), 1)

    def test_adjust_confidence(self):
        gid = self.store.create(source="test", objective="A", confidence=0.5)
        self.store.adjust_confidence(gid, delta=0.2)
        goal = self.store.get(gid)
        self.assertAlmostEqual(goal["confidence"], 0.7)

    def test_adjust_confidence_clamps(self):
        gid = self.store.create(source="test", objective="A", confidence=0.9)
        self.store.adjust_confidence(gid, delta=0.5)
        goal = self.store.get(gid)
        self.assertLessEqual(goal["confidence"], 1.0)

    def test_count_created_today(self):
        self.store.create(source="test", objective="A")
        self.store.create(source="test", objective="B")
        self.assertEqual(self.store.count_created_today(), 2)

    def test_get_due_for_review(self):
        gid = self.store.create(source="test", objective="A")
        self.store.transition(gid, "active")
        due = self.store.get_due_for_review()
        self.assertGreater(len(due), 0)


class PatternGroupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.goal_store import GoalStore
        self.store = GoalStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_group(self):
        gid = self.store.create_group(
            group_key="finance:active_investor", domain="finance",
            entities=["AAPL", "TSLA"], intents=["price_check"], support_count=5,
        )
        self.assertIsInstance(gid, int)

    def test_get_by_key(self):
        self.store.create_group(group_key="test:key", domain="test")
        g = self.store.get_group_by_key("test:key")
        self.assertIsNotNone(g)
        self.assertEqual(g["domain"], "test")

    def test_get_unlabeled(self):
        self.store.create_group(group_key="k1", domain="d1")
        self.store.create_group(group_key="k2", domain="d2")
        self.store.set_group_label(1, "Labeled group")
        unlabeled = self.store.get_unlabeled_groups(limit=10)
        self.assertEqual(len(unlabeled), 1)
        self.assertEqual(unlabeled[0]["group_key"], "k2")

    def test_update_support(self):
        self.store.create_group(group_key="k1", domain="d1", support_count=1)
        self.store.update_group_support(1, new_count=5)
        g = self.store.get_group_by_key("k1")
        self.assertEqual(g["support_count"], 5)

    def test_has_recent_updates(self):
        self.store.create_group(group_key="k1", domain="d1")
        self.assertTrue(self.store.has_recent_updates(hours=2))


class StaleGoalsAndBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.goal_store import GoalStore
        self.store = GoalStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_get_stale_goals_empty(self):
        self.assertEqual(self.store.get_stale_goals(days=14), [])

    def test_get_stale_goals_recent_not_stale(self):
        gid = self.store.create(source="test", objective="Fresh goal")
        self.store.transition(gid, "active")
        stale = self.store.get_stale_goals(days=14)
        self.assertEqual(len(stale), 0)

    def test_get_stale_goals_old_is_stale(self):
        gid = self.store.create(source="test", objective="Old goal")
        self.store.transition(gid, "active")
        # Backdate updated_at to 30 days ago
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE autonomous_goals SET updated_at = datetime('now', '-30 days') WHERE id = ?",
                (gid,),
            )
        stale = self.store.get_stale_goals(days=14)
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]["id"], gid)

    def test_remaining_daily_budget_defaults(self):
        budget = self.store.remaining_daily_budget()
        self.assertIn("goals_remaining", budget)
        self.assertIn("task_runs_remaining", budget)
        self.assertEqual(budget["goals_remaining"], 3)
        self.assertEqual(budget["task_runs_remaining"], 20)

    def test_remaining_daily_budget_decrements(self):
        self.store.create(source="test", objective="A")
        self.store.create(source="test", objective="B")
        self.store.record_event(1, "task_completed", {"task_id": 1})
        budget = self.store.remaining_daily_budget(max_goals=3, max_task_runs=5)
        self.assertEqual(budget["goals_remaining"], 1)
        self.assertEqual(budget["task_runs_remaining"], 4)

    def test_remaining_daily_budget_floors_at_zero(self):
        for i in range(5):
            self.store.create(source="test", objective=f"Goal {i}")
        budget = self.store.remaining_daily_budget(max_goals=3, max_task_runs=20)
        self.assertEqual(budget["goals_remaining"], 0)


class OutboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.goal_store import GoalStore
        self.store = GoalStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_insert_and_drain(self):
        with sqlite3.connect(self.db_path) as conn:
            self.store.insert_outbox(conn, "add_suggestion", {"message": "test"})
        pending = self.store.drain_outbox()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["action_type"], "add_suggestion")
        self.assertEqual(len(self.store.drain_outbox()), 0)


if __name__ == "__main__":
    unittest.main()
