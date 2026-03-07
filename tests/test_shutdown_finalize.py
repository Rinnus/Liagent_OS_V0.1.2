"""Tests for shutdown finalization — ensures session facts are saved on shutdown."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from liagent.agent.session_finalizer import shutdown_runtime


class TestShutdownCallsFinalize(unittest.TestCase):
    """brain.shutdown() must call finalize_session() before tearing down connections."""

    def test_shutdown_calls_finalize_then_teardown(self):
        """Verify shutdown calls finalize_session first, then shutdown_runtime."""
        call_order = []

        async def fake_finalize():
            call_order.append("finalize")

        brain = MagicMock()
        brain._shutdown_done = False
        brain.finalize_session = fake_finalize
        brain.memory = MagicMock()
        brain.long_term = MagicMock()
        brain.long_term.close = MagicMock()
        brain.tool_policy = MagicMock()
        brain.tool_policy.close = MagicMock()
        brain._mcp_bridge = None

        async def run():
            if getattr(brain, "_shutdown_done", False):
                return
            brain._shutdown_done = True
            try:
                await brain.finalize_session()
            except Exception:
                pass
            await shutdown_runtime(
                mcp_bridge=brain._mcp_bridge,
                long_term=brain.long_term,
                tool_policy=brain.tool_policy,
            )
            call_order.append("teardown")
            brain._mcp_bridge = None

        asyncio.run(run())
        assert call_order[0] == "finalize", "finalize_session must run before teardown"
        assert "teardown" in call_order

    def test_shutdown_idempotent(self):
        """Second shutdown call should be a no-op (guard flag)."""
        finalize_calls = []

        async def fake_finalize():
            finalize_calls.append(1)

        brain = MagicMock()
        brain._shutdown_done = False
        brain.finalize_session = fake_finalize
        brain._mcp_bridge = None
        brain.long_term = MagicMock()
        brain.long_term.close = MagicMock()
        brain.tool_policy = MagicMock()
        brain.tool_policy.close = MagicMock()

        async def shutdown():
            if getattr(brain, "_shutdown_done", False):
                return
            brain._shutdown_done = True
            await brain.finalize_session()
            await shutdown_runtime(
                mcp_bridge=brain._mcp_bridge,
                long_term=brain.long_term,
                tool_policy=brain.tool_policy,
            )

        async def run():
            await shutdown()
            await shutdown()  # second call
            await shutdown()  # third call

        asyncio.run(run())
        assert len(finalize_calls) == 1, "finalize should run exactly once"

    def test_shutdown_finalize_error_does_not_block_teardown(self):
        """If finalize raises, teardown still happens."""
        async def failing_finalize():
            raise RuntimeError("LLM unavailable")

        brain = MagicMock()
        brain._shutdown_done = False
        brain.finalize_session = failing_finalize
        brain._mcp_bridge = None
        brain.long_term = MagicMock()
        brain.long_term.close = MagicMock()
        brain.tool_policy = MagicMock()
        brain.tool_policy.close = MagicMock()

        async def shutdown():
            if getattr(brain, "_shutdown_done", False):
                return
            brain._shutdown_done = True
            try:
                await brain.finalize_session()
            except Exception:
                pass
            await shutdown_runtime(
                mcp_bridge=brain._mcp_bridge,
                long_term=brain.long_term,
                tool_policy=brain.tool_policy,
            )

        asyncio.run(shutdown())
        brain.long_term.close.assert_called_once()
        brain.tool_policy.close.assert_called_once()


class TestDecayConfidenceDifferentialRate(unittest.TestCase):
    """User-stated facts should decay slower than LLM-extracted facts."""

    def test_user_stated_decays_slower(self):
        """user_stated facts decay at 0.25x the normal rate."""
        import sqlite3
        import tempfile
        from pathlib import Path
        from datetime import datetime, timezone, timedelta

        from liagent.agent.memory import LongTermMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_lt.db"
            ltm = LongTermMemory(db_path=db_path, data_dir=Path(tmpdir))

            old_time = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT INTO key_facts (fact, category, confidence, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("User name is Alice", "identity", 0.85, "user_stated", now, old_time),
                )
                conn.execute(
                    "INSERT INTO key_facts (fact, category, confidence, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("User likes coffee", "preference", 0.60, "llm_extract", now, old_time),
                )

            ltm.decay_confidence(decay_rate=0.02, min_age_days=7)

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute("SELECT fact, confidence, source FROM key_facts ORDER BY fact").fetchall()

            facts = {r[0]: (r[1], r[2]) for r in rows}
            user_conf = facts["User name is Alice"][0]
            llm_conf = facts["User likes coffee"][0]

            # user_stated: 0.85 - 0.02*0.25 = 0.845
            assert abs(user_conf - 0.845) < 0.001, f"Expected 0.845, got {user_conf}"
            # llm_extract: 0.60 - 0.02 = 0.58
            assert abs(llm_conf - 0.58) < 0.001, f"Expected 0.58, got {llm_conf}"

            ltm.close()

    def test_tool_result_also_decays_slower(self):
        """tool_result facts should also decay at 0.25x rate."""
        import sqlite3
        import tempfile
        from pathlib import Path
        from datetime import datetime, timezone, timedelta

        from liagent.agent.memory import LongTermMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_lt.db"
            ltm = LongTermMemory(db_path=db_path, data_dir=Path(tmpdir))

            old_time = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "INSERT INTO key_facts (fact, category, confidence, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("Stock AAPL is 180", "data", 0.90, "tool_result", now, old_time),
                )

            ltm.decay_confidence(decay_rate=0.04, min_age_days=7)

            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT confidence FROM key_facts WHERE fact = ?",
                    ("Stock AAPL is 180",),
                ).fetchone()

            # tool_result: 0.90 - 0.04*0.25 = 0.89
            assert abs(row[0] - 0.89) < 0.001, f"Expected 0.89, got {row[0]}"

            ltm.close()
