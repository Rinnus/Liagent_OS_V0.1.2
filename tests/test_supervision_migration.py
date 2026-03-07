# tests/test_supervision_migration.py
import sqlite3
import tempfile
from pathlib import Path
from liagent.agent.self_supervision import InteractionMetrics

def test_new_columns_exist_after_init():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        metrics = InteractionMetrics(db)
        with sqlite3.connect(db) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(interaction_metrics)")}
        assert "heuristic_success" in cols
        assert "verified_success" in cols
        assert "verify_source" in cols
        # Old column preserved
        assert "task_success" in cols

def test_log_turn_writes_new_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        metrics = InteractionMetrics(db)
        metrics.log_turn(
            session_id="s1", latency_ms=100, tool_calls=1, tool_errors=0,
            policy_blocked=0, task_success=True, answer_revision_count=0,
            quality_issues="", answer_chars=50,
            heuristic_success=True, verified_success=None, verify_source=None,
        )
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT heuristic_success, verified_success, verify_source FROM interaction_metrics").fetchone()
        assert row[0] == 1   # heuristic_success = True
        assert row[1] is None  # verified_success not yet known
        assert row[2] is None

def test_migration_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        InteractionMetrics(db)
        InteractionMetrics(db)  # second init should not error
        with sqlite3.connect(db) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(interaction_metrics)")}
        assert "heuristic_success" in cols

def test_log_turn_backward_compat():
    """Calling log_turn without new params should still work (defaults to None)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        metrics = InteractionMetrics(db)
        metrics.log_turn(
            session_id="s1", latency_ms=100, tool_calls=1, tool_errors=0,
            policy_blocked=0, task_success=True, answer_revision_count=0,
            quality_issues="", answer_chars=50,
        )
        with sqlite3.connect(db) as conn:
            row = conn.execute("SELECT heuristic_success, verified_success, verify_source FROM interaction_metrics").fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None
