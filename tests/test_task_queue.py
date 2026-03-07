# tests/test_task_queue.py
"""Tests for TaskStore: transition_run, expires_at column, heartbeat trigger type."""

import asyncio
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from liagent.agent.task_queue import TaskStore
from liagent.agent.time_utils import _now_local_iso


class TransitionRunTests(unittest.TestCase):
    """Tests for TaskStore.transition_run atomic state transitions."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.store = TaskStore(db_path=self.db_path)
        # Create a task and a run to work with
        self.task = self.store.create_task(
            name="test-task",
            trigger_type="once",
            trigger_config={},
            prompt_template="do something",
        )
        self.run = self.store.create_run(self.task["id"], prompt="test prompt")

    def tearDown(self):
        self.tmp.close()
        self.db_path.unlink(missing_ok=True)

    def test_transition_run_success(self):
        """transition_run updates status when from_status matches."""
        ok = self.store.transition_run(self.run["id"], "pending", "running")
        self.assertTrue(ok)
        # Verify actual DB state
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT status FROM autonomous_task_runs WHERE id = ?",
                (self.run["id"],),
            ).fetchone()
        self.assertEqual(row["status"], "running")

    def test_transition_run_wrong_from_status(self):
        """transition_run returns False when from_status doesn't match current status."""
        ok = self.store.transition_run(self.run["id"], "running", "success")
        self.assertFalse(ok)
        # Status should still be 'pending'
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT status FROM autonomous_task_runs WHERE id = ?",
                (self.run["id"],),
            ).fetchone()
        self.assertEqual(row["status"], "pending")

    def test_transition_run_prevents_double_confirm(self):
        """Two concurrent transitions from same state: only one succeeds."""
        # First: pending → pending_confirm
        ok1 = self.store.transition_run(self.run["id"], "pending", "pending_confirm")
        self.assertTrue(ok1)
        # Second: pending → pending_confirm (should fail, already moved)
        ok2 = self.store.transition_run(self.run["id"], "pending", "pending_confirm")
        self.assertFalse(ok2)

    def test_transition_run_with_extra_fields(self):
        """transition_run sets extra fields (result, error, started_at, finished_at, expires_at)."""
        now = _now_local_iso()
        ok = self.store.transition_run(
            self.run["id"],
            "pending",
            "running",
            started_at=now,
            expires_at="2099-12-31T23:59:59",
        )
        self.assertTrue(ok)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT status, started_at, expires_at FROM autonomous_task_runs WHERE id = ?",
                (self.run["id"],),
            ).fetchone()
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["started_at"], now)
        self.assertEqual(row["expires_at"], "2099-12-31T23:59:59")


class HeartbeatTriggerTypeTests(unittest.TestCase):
    """Test that heartbeat trigger_type is not cleaned up on init."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.close()
        self.db_path.unlink(missing_ok=True)

    def test_heartbeat_trigger_type_not_cleaned(self):
        """Tasks with trigger_type='heartbeat' should NOT be soft-deleted by _init_db cleanup."""
        store = TaskStore(db_path=self.db_path)
        task = store.create_task(
            name="heartbeat-task",
            trigger_type="heartbeat",
            trigger_config={"schedule": "*/30 * * * *"},
            prompt_template="check things",
        )
        self.assertEqual(task["status"], "active")

        # Re-initialize store (simulates restart) — should NOT delete heartbeat tasks
        store2 = TaskStore(db_path=self.db_path)
        reloaded = store2.get_task(task["id"])
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded["status"], "active")


class ListTasksFilterTests(unittest.TestCase):
    """Test list_tasks include_system parameter."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.store = TaskStore(db_path=self.db_path)

    def tearDown(self):
        self.tmp.close()
        self.db_path.unlink(missing_ok=True)

    def test_list_tasks_exclude_system(self):
        """include_system=False filters out heartbeat tasks."""
        self.store.create_task(
            name="user-task",
            trigger_type="once",
            trigger_config={},
            prompt_template="user prompt",
        )
        self.store.create_task(
            name="heartbeat-task",
            trigger_type="heartbeat",
            trigger_config={"schedule": "*/30 * * * *"},
            prompt_template="heartbeat prompt",
        )
        # Default: include_system=True → both visible
        all_tasks = self.store.list_tasks()
        self.assertEqual(len(all_tasks), 2)

        # include_system=False → only user task
        user_tasks = self.store.list_tasks(include_system=False)
        self.assertEqual(len(user_tasks), 1)
        self.assertEqual(user_tasks[0]["trigger_type"], "once")


class ExpiresAtColumnTests(unittest.TestCase):
    """Tests for expires_at column in autonomous_task_runs."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.store = TaskStore(db_path=self.db_path)
        self.task = self.store.create_task(
            name="test-task",
            trigger_type="once",
            trigger_config={},
            prompt_template="do something",
        )

    def tearDown(self):
        self.tmp.close()
        self.db_path.unlink(missing_ok=True)

    def test_update_run_with_expires_at(self):
        """update_run should accept expires_at field."""
        run = self.store.create_run(self.task["id"], prompt="test")
        self.store.update_run(run["id"], expires_at="2099-12-31T23:59:59")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT expires_at FROM autonomous_task_runs WHERE id = ?",
                (run["id"],),
            ).fetchone()
        self.assertEqual(row["expires_at"], "2099-12-31T23:59:59")

    def test_get_expired_pending_confirms(self):
        """get_expired_pending_confirms returns only expired pending_confirm runs."""
        # Run 1: pending_confirm with expired deadline
        run1 = self.store.create_run(self.task["id"], prompt="expired")
        past = "2020-01-01T00:00:00+00:00"
        self.store.update_run(run1["id"], status="pending_confirm", expires_at=past)

        # Run 2: pending_confirm with future deadline (not expired)
        run2 = self.store.create_run(self.task["id"], prompt="future")
        future = "2099-12-31T23:59:59+00:00"
        self.store.update_run(run2["id"], status="pending_confirm", expires_at=future)

        # Run 3: pending_confirm with no expires_at (not expired)
        run3 = self.store.create_run(self.task["id"], prompt="no deadline")
        self.store.update_run(run3["id"], status="pending_confirm")

        # Run 4: different status, expired (should NOT be returned)
        run4 = self.store.create_run(self.task["id"], prompt="running")
        self.store.update_run(run4["id"], status="running", expires_at=past)

        expired = self.store.get_expired_pending_confirms()
        self.assertEqual(len(expired), 1)
        self.assertIn(run1["id"], expired)

    def test_get_recoverable_runs_includes_pending_and_confirmed(self):
        pending = self.store.create_run(self.task["id"], prompt="pending")
        confirmed = self.store.create_run(self.task["id"], prompt="confirmed")
        self.store.update_run(confirmed["id"], status="confirmed")
        running = self.store.create_run(self.task["id"], prompt="running")
        self.store.update_run(running["id"], status="running")

        recoverable = self.store.get_recoverable_runs()
        ids = {row["id"] for row in recoverable}
        self.assertIn(pending["id"], ids)
        self.assertIn(confirmed["id"], ids)
        self.assertNotIn(running["id"], ids)


class ExecutorRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.store = TaskStore(db_path=self.db_path)
        self.task = self.store.create_task(
            name="heartbeat-task",
            trigger_type="heartbeat",
            trigger_config={},
            prompt_template="heartbeat prompt",
            priority=8,
        )

    async def asyncTearDown(self):
        self.tmp.close()
        self.db_path.unlink(missing_ok=True)

    async def test_start_requeues_pending_and_confirmed_runs(self):
        pending = self.store.create_run(self.task["id"], trigger_event="manual", prompt="do work")
        confirmed = self.store.create_run(
            self.task["id"],
            trigger_event="heartbeat:test",
            prompt="Execute the following action exactly as specified.\nTool: web_search\nArguments: {}\n",
        )
        self.store.update_run(confirmed["id"], status="confirmed", origin="system")

        from liagent.agent.task_queue import TaskExecutor
        executor = TaskExecutor(SimpleNamespace(), self.store, asyncio.Lock())

        with patch.object(executor, "_consumer_loop", new=AsyncMock(return_value=None)):
            await executor.start()

        self.assertEqual(executor._queue.qsize(), 2)
        await executor.stop()


class ExpiresAtMigrationTests(unittest.TestCase):
    """Test that expires_at column is added to existing DBs via ALTER TABLE."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.close()
        self.db_path.unlink(missing_ok=True)

    def test_alter_table_adds_column_to_existing_db(self):
        """Opening an existing DB without expires_at should add the column via ALTER TABLE."""
        # Create the table WITHOUT expires_at (simulates old schema)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS autonomous_tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    trigger_config TEXT NOT NULL DEFAULT '{}',
                    prompt_template TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    priority INTEGER NOT NULL DEFAULT 10,
                    created_at TEXT NOT NULL,
                    next_run_at TEXT,
                    last_run_at TEXT,
                    result_summary TEXT,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 2
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS autonomous_task_runs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES autonomous_tasks(id),
                    trigger_event TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    prompt TEXT,
                    result TEXT,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL
                )"""
            )

        # Now open with TaskStore — should add expires_at via ALTER TABLE
        store = TaskStore(db_path=self.db_path)

        # Verify the column exists by inserting and querying
        task = store.create_task(
            name="migration-test",
            trigger_type="once",
            trigger_config={},
            prompt_template="test",
        )
        run = store.create_run(task["id"], prompt="test")
        store.update_run(run["id"], expires_at="2099-12-31T23:59:59")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT expires_at FROM autonomous_task_runs WHERE id = ?",
                (run["id"],),
            ).fetchone()
        self.assertEqual(row["expires_at"], "2099-12-31T23:59:59")


class QueueItemBudgetTests(unittest.TestCase):
    def test_queue_item_accepts_budget(self):
        from liagent.agent.task_queue import _QueueItem
        from liagent.skills.router import BudgetOverride
        budget = BudgetOverride(allowed_tools={"web_search"}, max_tool_calls=1, timeout_ms=30000)
        item = _QueueItem(priority=5, run_id="r1", task_id="t1", prompt="test", budget=budget)
        self.assertEqual(item.budget.allowed_tools, {"web_search"})
        self.assertEqual(item.budget.max_tool_calls, 1)

    def test_queue_item_budget_default_none(self):
        from liagent.agent.task_queue import _QueueItem
        item = _QueueItem(priority=5, run_id="r1", task_id="t1", prompt="test")
        self.assertIsNone(item.budget)

    def test_queue_item_ordering_unaffected_by_budget(self):
        from liagent.agent.task_queue import _QueueItem
        a = _QueueItem(priority=1, run_id="r1", task_id="t1", prompt="a")
        b = _QueueItem(priority=10, run_id="r2", task_id="t2", prompt="b")
        self.assertTrue(a < b)


class TaskSchemaExtensionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        from liagent.agent.task_queue import TaskStore
        self.store = TaskStore(db_path=self.db_path)

    def tearDown(self):
        self.tmp.close()
        import os
        os.unlink(self.db_path)

    def test_new_columns_exist(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(autonomous_tasks)").fetchall()}
        for col in ("source", "goal_id", "dedup_key", "risk_level"):
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_dedup_index_exists(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            indexes = {r[1] for r in conn.execute("SELECT * FROM sqlite_master WHERE type='index'").fetchall() if r[1]}
        self.assertIn("idx_tasks_dedup_active", indexes)

    def test_runs_new_columns_exist(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(autonomous_task_runs)").fetchall()}
        for col in ("origin", "goal_id", "confidence_label", "evidence_json"):
            self.assertIn(col, cols, f"Missing column: {col}")


class CreateTaskFromPromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        from liagent.agent.task_queue import TaskStore
        self.store = TaskStore(db_path=self.db_path)

    def tearDown(self):
        self.tmp.close()
        import os
        os.unlink(self.db_path)

    def test_creates_task_with_defaults(self):
        task = self.store.create_task_from_prompt(prompt="search AAPL price")
        self.assertIsNotNone(task)
        self.assertEqual(task["prompt_template"], "search AAPL price")

    def test_sets_extended_fields(self):
        import sqlite3
        task = self.store.create_task_from_prompt(
            prompt="test prompt", source="goal_driven",
            goal_id=42, dedup_key="test:dedup:1", risk_level="medium",
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT source, goal_id, dedup_key, risk_level FROM autonomous_tasks WHERE id = ?",
                (task["id"],),
            ).fetchone()
        self.assertEqual(row["source"], "goal_driven")
        self.assertEqual(row["goal_id"], 42)
        self.assertEqual(row["dedup_key"], "test:dedup:1")
        self.assertEqual(row["risk_level"], "medium")

    def test_idempotent_on_duplicate_dedup_key(self):
        t1 = self.store.create_task_from_prompt(prompt="first", dedup_key="dup:1")
        t2 = self.store.create_task_from_prompt(prompt="second", dedup_key="dup:1")
        self.assertIsNotNone(t1)
        self.assertIsNone(t2)

    def test_no_dedup_key_allows_duplicates(self):
        t1 = self.store.create_task_from_prompt(prompt="same")
        t2 = self.store.create_task_from_prompt(prompt="same")
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        self.assertNotEqual(t1["id"], t2["id"])


class UpdateRunExtendedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        from liagent.agent.task_queue import TaskStore
        self.store = TaskStore(db_path=self.db_path)
        self.task = self.store.create_task(name="t", trigger_type="once", trigger_config={}, prompt_template="p")
        self.run = self.store.create_run(self.task["id"], prompt="test")

    def tearDown(self):
        self.tmp.close()
        import os
        os.unlink(self.db_path)

    def test_update_run_new_fields(self):
        import sqlite3
        self.store.update_run(self.run["id"], origin="goal", goal_id=7, confidence_label="high", evidence_json='[{"tool":"web_search"}]')
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT origin, goal_id, confidence_label, evidence_json FROM autonomous_task_runs WHERE id = ?", (self.run["id"],)).fetchone()
        self.assertEqual(row["origin"], "goal")
        self.assertEqual(row["goal_id"], 7)
        self.assertEqual(row["confidence_label"], "high")
        self.assertIn("web_search", row["evidence_json"])


class GetRecentRunsForGoalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        from liagent.agent.task_queue import TaskStore
        self.store = TaskStore(db_path=self.db_path)
        self.task = self.store.create_task(name="t", trigger_type="once", trigger_config={}, prompt_template="p")

    def tearDown(self):
        self.tmp.close()
        import os
        os.unlink(self.db_path)

    def test_returns_runs_for_goal(self):
        run = self.store.create_run(self.task["id"], prompt="test")
        self.store.update_run(run["id"], goal_id=5)
        results = self.store.get_recent_runs_for_goal(5)
        self.assertEqual(len(results), 1)

    def test_excludes_other_goals(self):
        run = self.store.create_run(self.task["id"], prompt="test")
        self.store.update_run(run["id"], goal_id=5)
        results = self.store.get_recent_runs_for_goal(99)
        self.assertEqual(len(results), 0)


class GoalFeedbackTests(unittest.IsolatedAsyncioTestCase):
    """Tests for goal feedback on task completion."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        self.store = TaskStore(db_path=self.db_path)
        from liagent.agent.goal_store import GoalStore
        self.goal_store = GoalStore(str(self.db_path))

    def tearDown(self):
        self.tmp.close()
        import os
        os.unlink(self.db_path)

    def test_enqueue_with_goal_id(self):
        """enqueue() accepts goal_id and source without error."""
        task = self.store.create_task(
            name="test", trigger_type="once",
            trigger_config={}, prompt_template="test",
        )
        run = self.store.create_run(task["id"], prompt="test")
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from liagent.agent.task_queue import TaskExecutor
        brain = MagicMock()
        lock = asyncio.Lock()
        executor = TaskExecutor(brain, self.store, lock, goal_store=self.goal_store)
        executor.enqueue(
            run["id"], task["id"], "test",
            goal_id=1, source="reflection",
        )

    def test_queue_item_has_goal_fields(self):
        """_QueueItem stores goal_id and source."""
        from liagent.agent.task_queue import _QueueItem
        item = _QueueItem(
            priority=5, run_id="r1", task_id="t1", prompt="test",
            goal_id=42, source="goal_driven",
        )
        self.assertEqual(item.goal_id, 42)
        self.assertEqual(item.source, "goal_driven")

    def test_queue_item_goal_fields_default(self):
        """_QueueItem defaults: goal_id=None, source='user_manual'."""
        from liagent.agent.task_queue import _QueueItem
        item = _QueueItem(priority=5, run_id="r1", task_id="t1", prompt="test")
        self.assertIsNone(item.goal_id)
        self.assertEqual(item.source, "user_manual")


if __name__ == "__main__":
    unittest.main()
