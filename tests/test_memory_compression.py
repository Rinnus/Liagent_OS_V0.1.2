"""Tests for conversation memory compression."""

import asyncio
import os
import tempfile
import unittest

from liagent.agent.memory import ConversationMemory


class ConversationMemoryTests(unittest.TestCase):
    def test_get_messages_includes_compressed_context(self):
        mem = ConversationMemory(max_turns=20)
        mem.compressed_context = "Previously discussed weather-related questions."
        mem.add("user", "hello")
        msgs = mem.get_messages()
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("History Summary", msgs[0]["content"])
        self.assertEqual(msgs[1]["role"], "user")

    def test_get_messages_without_compressed_context(self):
        mem = ConversationMemory(max_turns=20)
        mem.add("user", "hello")
        msgs = mem.get_messages()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")

    def test_clear_resets_compressed_context(self):
        mem = ConversationMemory(max_turns=20)
        mem.compressed_context = "old summary"
        mem.add("user", "test")
        mem.clear()
        self.assertEqual(mem.compressed_context, "")
        self.assertEqual(len(mem.messages), 0)

    def test_compress_skips_when_below_char_budget(self):
        mem = ConversationMemory(max_turns=20)
        for i in range(5):
            mem.add("user", f"msg {i}")
            mem.add("assistant", f"resp {i}")

        class FakeEngine:
            pass

        # 10 short messages are well below the default 16000 char budget
        asyncio.run(mem.compress(FakeEngine()))
        self.assertEqual(mem.compressed_context, "")
        self.assertEqual(len(mem.messages), 10)


class SlidingWindowTests(unittest.TestCase):
    def test_sliding_window_truncation(self):
        mem = ConversationMemory(max_turns=3)
        for i in range(10):
            mem.add("user", f"q{i}")
            mem.add("assistant", f"a{i}")
        # max_turns=3, so max_msgs=9 (3 messages per turn: user + assistant + tool)
        self.assertEqual(len(mem.messages), 9)

    def test_last_user_message(self):
        mem = ConversationMemory()
        mem.add("user", "first")
        mem.add("assistant", "response")
        mem.add("user", "second")
        self.assertEqual(mem.last_user_message(), "second")

    def test_turn_count(self):
        mem = ConversationMemory()
        mem.add("user", "q1")
        mem.add("assistant", "a1")
        mem.add("user", "q2")
        self.assertEqual(mem.turn_count(), 2)


class TrimOversizedMessageTests(unittest.TestCase):
    """Regression: a single message exceeding char_budget must NOT wipe all messages."""

    def test_oversized_message_preserves_newest(self):
        """Adding a message larger than char_budget should keep at least that message."""
        mem = ConversationMemory(max_turns=30, char_budget=4000)
        mem.add("user", "hello")
        mem.add("assistant", "hi there")
        # Add a message far exceeding the 4000 char budget
        huge = "X" * 10000
        mem.add("tool", huge)
        # Must NOT wipe to empty — at least the newest message survives
        self.assertGreater(len(mem.messages), 0)
        self.assertEqual(mem.messages[-1]["content"], huge)

    def test_oversized_message_evicts_older(self):
        """Oversized message should evict older messages but keep the oversized one."""
        mem = ConversationMemory(max_turns=30, char_budget=4000)
        for i in range(5):
            mem.add("user", f"msg-{i}")
        huge = "Y" * 10000
        mem.add("assistant", huge)
        # Only the oversized message should remain (or very few)
        self.assertGreaterEqual(len(mem.messages), 1)
        self.assertEqual(mem.messages[-1]["content"], huge)

    def test_normal_trim_still_works(self):
        """Normal-sized messages should still trim correctly under char budget.

        Note: char_budget floor is 4000, so we need large messages to trigger trim.
        """
        mem = ConversationMemory(max_turns=30, char_budget=4000)
        # Each 500-char message, 20 of them = 10000 chars > 4000 budget
        for i in range(20):
            mem.add("user", f"{'A' * 490} msg-{i}")
        # ~500 chars/msg, 4000 budget → ~8 messages fit
        self.assertLess(len(mem.messages), 15)
        self.assertGreater(len(mem.messages), 0)
        # Most recent message should be the last one added
        self.assertIn("msg-19", mem.messages[-1]["content"])

    def test_single_message_not_wiped(self):
        """A single message that exceeds budget should survive."""
        mem = ConversationMemory(max_turns=30, char_budget=4000)
        huge = "Z" * 20000
        mem.add("user", huge)
        self.assertEqual(len(mem.messages), 1)
        self.assertEqual(mem.messages[0]["content"], huge)


class CheckpointReasoningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reasoning_summary_column_exists(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(pending_goals)").fetchall()}
        self.assertIn("reasoning_summary_json", cols)

    def test_upsert_checkpoint_with_reasoning(self):
        summary = {"steps": [{"step": 0, "think": "test reasoning"}]}
        cid = self.ltm.upsert_checkpoint(
            session_id="test-sess", goal="test goal",
            plan_steps=[{"title": "step1"}], completed_steps=1,
            total_steps=2, evidence=[], reasoning_summary=summary,
        )
        self.assertIsInstance(cid, int)

    def test_upsert_checkpoint_backward_compatible(self):
        cid = self.ltm.upsert_checkpoint(
            session_id="test-sess-2", goal="test goal",
            plan_steps=[], completed_steps=0, total_steps=1, evidence=[],
        )
        self.assertIsInstance(cid, int)


class GetRecentFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")
        from liagent.agent.memory import LongTermMemory
        self.ltm = LongTermMemory(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_get_recent_feedback_empty(self):
        result = self.ltm.get_recent_feedback(days=7)
        self.assertEqual(result, [])

    def test_get_recent_feedback_returns_entries(self):
        self.ltm.save_feedback("sess1", 0, "what is AAPL?", "AAPL is...", "web_search", "positive")
        result = self.ltm.get_recent_feedback(days=7)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["feedback"], "positive")


if __name__ == "__main__":
    unittest.main()
