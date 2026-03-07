"""Tests for execution checkpoint persistence in LongTermMemory."""
import json
import os
import sqlite3
import tempfile
import unittest


class CheckpointCRUDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_upsert_creates_checkpoint(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        steps = [{"id": "s1", "title": "Query AAPL", "status": "pending"}]
        cp_id = ltm.upsert_checkpoint(
            session_id="sess1", goal="Compare stocks",
            plan_steps=steps, completed_steps=0,
            total_steps=1, evidence=[],
        )
        self.assertIsNotNone(cp_id)
        cp = ltm.get_active_checkpoint()
        self.assertIsNotNone(cp)
        self.assertEqual(cp["goal_text"], "Compare stocks")

    def test_upsert_updates_existing(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        steps = [{"id": "s1", "status": "pending"}, {"id": "s2", "status": "pending"}]
        ltm.upsert_checkpoint("sess1", "Goal", steps, 0, 2, [])
        steps[0]["status"] = "done"
        ltm.upsert_checkpoint("sess1", "Goal", steps, 1, 2, [{"step_id": "s1", "ref": "data"}])
        cp = ltm.get_active_checkpoint()
        self.assertEqual(cp["completed_steps"], 1)
        plan = json.loads(cp["plan_json"])
        self.assertEqual(plan[0]["status"], "done")

    def test_complete_checkpoint(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        steps = [{"id": "s1", "status": "done"}]
        ltm.upsert_checkpoint("sess1", "Goal", steps, 1, 1, [])
        cp = ltm.get_active_checkpoint()
        ltm.complete_checkpoint(cp["id"])
        self.assertIsNone(ltm.get_active_checkpoint())

    def test_abandon_checkpoint(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        ltm.upsert_checkpoint("sess1", "Goal", [{"id": "s1"}], 0, 1, [])
        cp = ltm.get_active_checkpoint()
        ltm.abandon_checkpoint(cp["id"])
        self.assertIsNone(ltm.get_active_checkpoint())

    def test_no_active_checkpoint_returns_none(self):
        from liagent.agent.memory import LongTermMemory
        ltm = LongTermMemory(db_path=self.db_path)
        self.assertIsNone(ltm.get_active_checkpoint())


if __name__ == "__main__":
    unittest.main()
