import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class ExecuteDecisionTests(unittest.TestCase):
    """Tests for _execute_decision transactional logic."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.goal_store import GoalStore
        self.goal_store = GoalStore(self.db_path)
        from liagent.agent.task_queue import TaskStore
        self.task_store = TaskStore(db_path=self.db_path)
        self.config = MagicMock()
        self.config.proactive.max_active_goals = 5
        self.config.proactive.max_new_goals_per_day = 3
        self.config.proactive.authorization = {}

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_goal_decision(self):
        from liagent.agent.goal_loop import _execute_decision
        d = {
            "decision_id": "dec-1",
            "type": "create_goal",
            "confidence": 0.8,
            "requires_consent": False,
            "idempotency_key": "goal:test:1:2026-03-05",
            "params": {
                "objective": "Track AAPL",
                "domain": "finance",
                "rationale": "Frequent queries",
                "priority": 7,
            },
            "initial_tasks": [],
        }
        post = _execute_decision(d, self.goal_store, self.task_store, self.config)
        self.assertIsInstance(post, list)
        goals = self.goal_store.get_by_state("proposed")
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0]["objective"], "Track AAPL")

    def test_create_goal_with_initial_task(self):
        from liagent.agent.goal_loop import _execute_decision
        d = {
            "decision_id": "dec-2",
            "type": "create_goal",
            "confidence": 0.8,
            "requires_consent": False,
            "idempotency_key": "goal:test:2:2026-03-05",
            "params": {
                "objective": "Monitor portfolio",
                "domain": "finance",
                "rationale": "User interest",
            },
            "initial_tasks": [
                {
                    "prompt": "Check AAPL price",
                    "trigger": "once",
                    "idempotency_key": "task:test:1",
                }
            ],
        }
        post = _execute_decision(d, self.goal_store, self.task_store, self.config)
        self.assertTrue(any(a[0] == "register_trigger" for a in post))

    def test_create_goal_requires_consent(self):
        from liagent.agent.goal_loop import _execute_decision
        d = {
            "decision_id": "dec-3",
            "type": "create_goal",
            "confidence": 0.6,
            "requires_consent": True,
            "idempotency_key": "goal:test:3:2026-03-05",
            "params": {
                "objective": "Auto-trade stocks",
                "domain": "finance",
                "rationale": "Risky",
            },
            "initial_tasks": [],
        }
        post = _execute_decision(d, self.goal_store, self.task_store, self.config)
        # Should NOT create goal, should add to outbox instead
        goals = self.goal_store.get_by_state("proposed")
        self.assertEqual(len(goals), 0)
        outbox = self.goal_store.drain_outbox()
        self.assertEqual(len(outbox), 1)
        self.assertEqual(outbox[0]["action_type"], "add_suggestion")
        payload = json.loads(outbox[0]["payload_json"])
        self.assertTrue(payload["action_json"]["create_goal"])
        self.assertEqual(payload["action_json"]["objective"], "Auto-trade stocks")
        self.assertEqual(payload["pattern_key"], "goal:test:3:2026-03-05")

    def test_retire_goal_decision(self):
        from liagent.agent.goal_loop import _execute_decision
        gid = self.goal_store.create(source="test", objective="Old goal")
        self.goal_store.transition(gid, "active")
        d = {
            "decision_id": "dec-4",
            "type": "retire_goal",
            "goal_id": gid,
            "reason": "No longer relevant",
        }
        post = _execute_decision(d, self.goal_store, self.task_store, self.config)
        goal = self.goal_store.get(gid)
        self.assertEqual(goal["state"], "retired")

    def test_idempotent_goal_creation(self):
        from liagent.agent.goal_loop import _execute_decision
        d = {
            "decision_id": "dec-5",
            "type": "create_goal",
            "confidence": 0.8,
            "requires_consent": False,
            "idempotency_key": "goal:test:dup:2026-03-05",
            "params": {"objective": "Test", "domain": "test"},
            "initial_tasks": [],
        }
        _execute_decision(d, self.goal_store, self.task_store, self.config)
        _execute_decision(d, self.goal_store, self.task_store, self.config)
        goals = self.goal_store.get_by_state("proposed")
        self.assertEqual(len(goals), 1)


class DrainOutboxTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.goal_store import GoalStore
        self.goal_store = GoalStore(self.db_path)
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        from liagent.agent.behavior import PendingSuggestionStore
        self.suggestion_store = PendingSuggestionStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    async def test_drain_processes_add_suggestion(self):
        from liagent.agent.goal_loop import _drain_outbox
        with sqlite3.connect(self.db_path) as conn:
            self.goal_store.insert_outbox(conn, "add_suggestion", {
                "message": "Should I track AAPL?",
                "delivery_mode": "session",
                "domain": "finance",
            })
        await _drain_outbox(self.goal_store, self.suggestion_store)
        # Outbox should be drained
        remaining = self.goal_store.drain_outbox()
        self.assertEqual(len(remaining), 0)

    async def test_drain_processes_multiple_entries_without_collision(self):
        from liagent.agent.goal_loop import _drain_outbox
        with sqlite3.connect(self.db_path) as conn:
            self.goal_store.insert_outbox(conn, "add_suggestion", {
                "message": "One",
                "delivery_mode": "session",
                "domain": "finance",
            })
            self.goal_store.insert_outbox(conn, "add_suggestion", {
                "message": "Two",
                "delivery_mode": "session",
                "domain": "finance",
            })
        await _drain_outbox(self.goal_store, self.suggestion_store)
        pending = self.suggestion_store.get_pending(max_items=10)
        self.assertEqual(len(pending), 2)


class BuildReflectionContextTests(unittest.TestCase):
    def test_builds_context_dict(self):
        from liagent.agent.goal_loop import _build_reflection_context
        ctx = _build_reflection_context(
            goals=[],
            groups=[],
            recent_feedback=[],
            user_profile={},
            tool_inventory=[{"name": "web_search", "description": "Search"}],
            constraints={"active_goals": 0, "max_active_goals": 5},
            unlabeled_groups=[],
        )
        self.assertIn("goals", ctx)
        self.assertIn("tool_inventory", ctx)
        self.assertIn("constraints", ctx)


class QuietHoursInReflectionTests(unittest.TestCase):
    """Verify is_quiet_hours import and usage in goal_loop."""

    def test_quiet_hours_import(self):
        from liagent.agent.goal_loop import is_quiet_hours
        # Should be the same function from behavior
        from liagent.agent.behavior import is_quiet_hours as beh_qh
        self.assertIs(is_quiet_hours, beh_qh)

    def test_quiet_hours_blocks_during_quiet(self):
        from liagent.agent.behavior import is_quiet_hours
        # Verify the function itself works (basic contract test)
        self.assertFalse(is_quiet_hours(""))
        self.assertFalse(is_quiet_hours("invalid"))
        self.assertIsInstance(is_quiet_hours("23:00-07:00"), bool)


class AutoRetireStaleGoalsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.goal_store import GoalStore
        self.goal_store = GoalStore(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stale_goal_gets_retired(self):
        gid = self.goal_store.create(source="test", objective="Old goal")
        self.goal_store.transition(gid, "active")
        # Backdate to 30 days ago
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE autonomous_goals SET updated_at = datetime('now', '-30 days') WHERE id = ?",
                (gid,),
            )
        # Simulate what reflection_loop does
        for stale in self.goal_store.get_stale_goals(days=14):
            self.goal_store.transition(stale["id"], "retired", reason="auto-retired: no activity for 14 days")
            self.goal_store.record_event(stale["id"], "retired", {"reason": "auto-retired"})
        goal = self.goal_store.get(gid)
        self.assertEqual(goal["state"], "retired")

    def test_recent_goal_not_retired(self):
        gid = self.goal_store.create(source="test", objective="Fresh goal")
        self.goal_store.transition(gid, "active")
        stale = self.goal_store.get_stale_goals(days=14)
        self.assertEqual(len(stale), 0)


class ParseAndValidateTests(unittest.TestCase):
    def test_valid_json(self):
        from liagent.agent.goal_loop import _parse_and_validate
        raw = json.dumps({
            "schema_version": "1.0",
            "discoveries": [],
            "decisions": [],
            "self_observations": [],
            "next_review_minutes": 30,
        })
        result = _parse_and_validate(raw)
        self.assertEqual(result["schema_version"], "1.0")

    def test_invalid_json_returns_empty(self):
        from liagent.agent.goal_loop import _parse_and_validate
        result = _parse_and_validate("not json at all")
        self.assertEqual(result["decisions"], [])

    def test_missing_fields_defaults(self):
        from liagent.agent.goal_loop import _parse_and_validate
        result = _parse_and_validate("{}")
        self.assertIn("decisions", result)
        self.assertIn("discoveries", result)


if __name__ == "__main__":
    unittest.main()
