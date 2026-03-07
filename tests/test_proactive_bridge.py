# tests/test_proactive_bridge.py
import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class BridgeLoopTests(unittest.IsolatedAsyncioTestCase):
    """Tests for bridge_loop one-shot iteration."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        LongTermMemory(db_path=self.db_path)
        from liagent.agent.behavior import PendingSuggestionStore
        self.suggestion_store = PendingSuggestionStore(self.db_path)
        from liagent.agent.task_queue import TaskStore
        self.task_store = TaskStore(db_path=self.db_path)
        self.trigger_mgr = AsyncMock()
        self.config = self._make_config()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_config(self):
        cfg = MagicMock()
        cfg.proactive.bridge_scan_interval_sec = 0  # no sleep in test
        cfg.proactive.quiet_hours = ""
        cfg.proactive.authorization = {"stock": "auto", "general": "suggest"}
        return cfg

    def _insert_auto_suggestion(self, domain="stock", action=None):
        action = action or {"prompt": "search AAPL price", "type": "execute"}
        self.suggestion_store.add(
            pattern_key=f"test:{domain}:1",
            domain=domain,
            suggestion_type="monitoring",
            message="Test",
            action_json=json.dumps(action),
            confidence=0.8,
            net_value=0.6,
            delivery_mode="auto",
        )

    async def test_bridge_iteration_creates_task(self):
        from liagent.agent.proactive_bridge import bridge_iteration
        self._insert_auto_suggestion()
        created = await bridge_iteration(
            self.suggestion_store, None, self.task_store,
            self.trigger_mgr, self.config,
        )
        self.assertEqual(created, 1)
        # Verify suggestion was claimed
        remaining = self.suggestion_store.get_by_delivery_mode("auto")
        self.assertEqual(len(remaining), 0)

    async def test_bridge_iteration_registers_trigger(self):
        from liagent.agent.proactive_bridge import bridge_iteration
        self._insert_auto_suggestion()
        await bridge_iteration(
            self.suggestion_store, None, self.task_store,
            self.trigger_mgr, self.config,
        )
        self.trigger_mgr.register_once.assert_called_once()

    async def test_bridge_iteration_skips_already_claimed(self):
        from liagent.agent.proactive_bridge import bridge_iteration
        self._insert_auto_suggestion()
        items = self.suggestion_store.get_by_delivery_mode("auto")
        self.suggestion_store.try_claim(items[0]["id"])
        created = await bridge_iteration(
            self.suggestion_store, None, self.task_store,
            self.trigger_mgr, self.config,
        )
        self.assertEqual(created, 0)

    async def test_bridge_iteration_goal_creation(self):
        from liagent.agent.proactive_bridge import bridge_iteration
        goal_store = MagicMock()
        goal_store.create.return_value = 1
        self._insert_auto_suggestion(
            action={"create_goal": True, "objective": "Track AAPL", "rationale": "Frequent queries"}
        )
        await bridge_iteration(
            self.suggestion_store, goal_store, self.task_store,
            self.trigger_mgr, self.config,
        )
        goal_store.create.assert_called_once()

    async def test_bridge_iteration_empty_queue(self):
        from liagent.agent.proactive_bridge import bridge_iteration
        created = await bridge_iteration(
            self.suggestion_store, None, self.task_store,
            self.trigger_mgr, self.config,
        )
        self.assertEqual(created, 0)

    async def test_bridge_marks_accepted_on_success(self):
        from liagent.agent.proactive_bridge import bridge_iteration
        self._insert_auto_suggestion()
        await bridge_iteration(
            self.suggestion_store, None, self.task_store,
            self.trigger_mgr, self.config,
        )
        # Verify terminal state: 'accepted' (not stuck at 'processing')
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT status FROM pending_suggestions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(row[0], "accepted")



if __name__ == "__main__":
    unittest.main()
