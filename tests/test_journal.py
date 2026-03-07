"""Tests for OptimizationJournal — event recording + Coder review filtering."""

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from liagent.agent.journal import OptimizationJournal


class AppendTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.journal = OptimizationJournal(base_dir=Path(self._tmpdir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _read_today(self):
        path = self.journal._today_path()
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def test_append_creates_file(self):
        self.journal._append("test_event", key="value")
        content = self._read_today()
        self.assertIn("### [", content)
        self.assertIn("test_event", content)
        self.assertIn("- **key**: value", content)

    def test_append_accumulates(self):
        self.journal._append("event_a", x="1")
        self.journal._append("event_b", y="2")
        content = self._read_today()
        self.assertEqual(content.count("---"), 2)
        self.assertIn("event_a", content)
        self.assertIn("event_b", content)

    def test_session_events_buffer(self):
        self.journal._append("ev1", a="1")
        self.journal._append("ev2", b="2")
        self.assertEqual(len(self.journal._session_events), 2)
        self.assertEqual(self.journal._session_events[0]["category"], "ev1")
        self.assertEqual(self.journal._session_events[1]["category"], "ev2")

    def test_value_truncation(self):
        long_val = "x" * 1000
        self.journal._append("trunc_test", data=long_val)
        content = self._read_today()
        # Should be truncated to 500
        self.assertNotIn("x" * 501, content)


class EventMethodTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.journal = OptimizationJournal(base_dir=Path(self._tmpdir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _read_today(self):
        path = self.journal._today_path()
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def test_skill_generated(self):
        self.journal.skill_generated(
            query="gold price", failed_answer="I don't know",
            pattern="precious metal query", tool="web_search",
            confidence=0.7, source="nemotron_gen",
        )
        content = self._read_today()
        self.assertIn("skill_generated", content)
        self.assertIn("gold price", content)
        self.assertIn("web_search", content)

    def test_outcome_recorded(self):
        self.journal.outcome_recorded(
            query="test", tool="web_search", success=True,
            source="self_eval", reward=1.5,
        )
        content = self._read_today()
        self.assertIn("outcome_recorded", content)
        self.assertIn("success", content)
        self.assertIn("1.50", content)

    def test_fact_learned_new(self):
        self.journal.fact_learned(
            fact="user likes Python", category="preference",
            confidence=0.85, source="llm_extract", is_new=True,
        )
        content = self._read_today()
        self.assertIn("fact_learned", content)
        self.assertIn("new", content)

    def test_fact_learned_update(self):
        self.journal.fact_learned(
            fact="user likes Python", category="preference",
            confidence=0.85, source="llm_extract", is_new=False,
        )
        content = self._read_today()
        self.assertIn("updated", content)

    def test_fact_conflict(self):
        self.journal.fact_conflict(
            new_fact="user likes Go", old_fact="user likes Java",
            old_confidence=0.8, demoted_confidence=0.4,
        )
        content = self._read_today()
        self.assertIn("fact_conflict", content)
        self.assertIn("0.40", content)

    def test_session_summary(self):
        self.journal.session_summary(
            session_id="abc123def456", summary="discussed weather and stocks",
            turn_count=5, facts_count=2,
        )
        content = self._read_today()
        self.assertIn("session_summary", content)
        self.assertIn("abc123def4", content)  # truncated to 10

    def test_experience_pruned_nonzero(self):
        self.journal.experience_pruned(deleted_count=3)
        content = self._read_today()
        self.assertIn("experience_pruned", content)
        self.assertIn("3", content)

    def test_experience_pruned_zero_skips(self):
        self.journal.experience_pruned(deleted_count=0)
        content = self._read_today()
        self.assertEqual(content, "")

    def test_memory_pruned_nonzero(self):
        self.journal.memory_pruned(expired=1, low_confidence=2, excess=0)
        content = self._read_today()
        self.assertIn("memory_pruned", content)

    def test_memory_pruned_all_zero_skips(self):
        self.journal.memory_pruned(expired=0, low_confidence=0, excess=0)
        content = self._read_today()
        self.assertEqual(content, "")


class ReviewFilterTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.journal = OptimizationJournal(base_dir=Path(self._tmpdir))

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _read_today(self):
        path = self.journal._today_path()
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def test_no_review_when_empty(self):
        result = asyncio.run(
            self.journal.generate_review(AsyncMock())
        )
        self.assertIsNone(result)

    def test_no_review_only_outcomes(self):
        """Only outcome_recorded events should not trigger a review."""
        self.journal.outcome_recorded(
            query="test", tool="web_search", success=True,
            source="self_eval", reward=1.0,
        )
        engine = AsyncMock()
        result = asyncio.run(
            self.journal.generate_review(engine)
        )
        self.assertIsNone(result)
        engine.generate_reasoning.assert_not_called()
        # Buffer should be cleared
        self.assertEqual(len(self.journal._session_events), 0)

    def test_no_review_short_output(self):
        """Coder output < 50 chars should be discarded."""
        self.journal.fact_learned(
            fact="test fact", category="test", confidence=0.8,
            source="llm_extract", is_new=True,
        )
        engine = AsyncMock()
        engine.generate_reasoning = AsyncMock(return_value="too short")
        result = asyncio.run(
            self.journal.generate_review(engine)
        )
        self.assertIsNone(result)
        self.assertEqual(len(self.journal._session_events), 0)

    def test_no_review_hollow_output(self):
        """Hollow phrases in Coder output should be discarded."""
        self.journal.fact_learned(
            fact="test", category="test", confidence=0.8,
            source="llm_extract", is_new=True,
        )
        engine = AsyncMock()
        engine.generate_reasoning = AsyncMock(
            return_value="No significant changes occurred in this session; all metrics stayed stable and no adjustment is necessary."
        )
        result = asyncio.run(
            self.journal.generate_review(engine)
        )
        self.assertIsNone(result)

    def test_generate_review_writes_markdown(self):
        """Valid review should be written to the Markdown file."""
        self.journal.session_summary(
            session_id="test123", summary="test session",
            turn_count=5, facts_count=2,
        )
        review_text = (
            "**Change Summary**: Added two new user facts in this session.\n"
            "**Root Cause**: Preferences were extracted from user conversation signals.\n"
            "**Impact Assessment**: Knowledge coverage improved.\n"
            "**Next Improvements**: Be more proactive in extracting implicit preferences."
        )
        engine = AsyncMock()
        engine.generate_reasoning = AsyncMock(return_value=review_text)
        result = asyncio.run(
            self.journal.generate_review(engine)
        )
        self.assertEqual(result, review_text)
        content = self._read_today()
        self.assertIn("Session Retrospective (Coder 30B)", content)
        self.assertIn("Change Summary", content)

    def test_generate_review_clears_buffer(self):
        """Buffer should be cleared after review, regardless of outcome."""
        self.journal.fact_learned(
            fact="test", category="test", confidence=0.8,
            source="llm_extract", is_new=True,
        )
        self.assertEqual(len(self.journal._session_events), 1)
        engine = AsyncMock()
        engine.generate_reasoning = AsyncMock(return_value="short")  # will be filtered
        asyncio.run(
            self.journal.generate_review(engine)
        )
        self.assertEqual(len(self.journal._session_events), 0)


if __name__ == "__main__":
    unittest.main()
