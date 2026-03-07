"""Simple self-supervision metrics store for offline analysis."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import db_path as _db_path
from ..logging import get_logger

_log = get_logger("self_supervision")

DB_PATH = _db_path()


_DEFAULT_RETENTION_DAYS = int(os.environ.get("LIAGENT_METRICS_RETENTION_DAYS", "90"))


class InteractionMetrics:
    def __init__(self, db_path: Path = DB_PATH, *, retention_days: int = _DEFAULT_RETENTION_DAYS):
        self.db_path = db_path
        self.retention_days = max(7, retention_days)
        self._init_db()
        self._prune_old()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS interaction_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    tool_calls INTEGER NOT NULL,
                    tool_errors INTEGER NOT NULL DEFAULT 0,
                    policy_blocked INTEGER NOT NULL,
                    task_success INTEGER NOT NULL DEFAULT 1,
                    answer_revision_count INTEGER NOT NULL DEFAULT 0,
                    quality_issues TEXT NOT NULL DEFAULT '',
                    plan_completion_ratio REAL NOT NULL DEFAULT 1.0,
                    answer_chars INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS runtime_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    queued_ms REAL NOT NULL DEFAULT 0,
                    stream_ms REAL NOT NULL DEFAULT 0,
                    tts_ms REAL NOT NULL DEFAULT 0,
                    total_ms REAL NOT NULL DEFAULT 0,
                    voice_mode INTEGER NOT NULL DEFAULT 0,
                    final_state TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )"""
            )
            cols = {
                row[1] for row in conn.execute("PRAGMA table_info(interaction_metrics)").fetchall()
            }
            if "tool_errors" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN tool_errors INTEGER NOT NULL DEFAULT 0")
            if "task_success" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN task_success INTEGER NOT NULL DEFAULT 1")
            if "answer_revision_count" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN answer_revision_count INTEGER NOT NULL DEFAULT 0")
            if "quality_issues" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN quality_issues TEXT NOT NULL DEFAULT ''")
            if "plan_completion_ratio" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN plan_completion_ratio REAL NOT NULL DEFAULT 1.0")
            if "heuristic_success" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN heuristic_success INTEGER DEFAULT NULL")
            if "verified_success" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN verified_success INTEGER DEFAULT NULL")
            if "verify_source" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN verify_source TEXT DEFAULT NULL")
            if "tool_fallback_count" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN tool_fallback_count INTEGER NOT NULL DEFAULT 0")
            if "tool_timeout_count" not in cols:
                conn.execute("ALTER TABLE interaction_metrics ADD COLUMN tool_timeout_count INTEGER NOT NULL DEFAULT 0")

    def _prune_old(self):
        """Delete rows older than retention_days and reclaim space."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                r1 = conn.execute("DELETE FROM interaction_metrics WHERE created_at < ?", (cutoff,))
                r2 = conn.execute("DELETE FROM runtime_metrics WHERE created_at < ?", (cutoff,))
                if (r1.rowcount or 0) + (r2.rowcount or 0) > 0:
                    conn.execute("PRAGMA incremental_vacuum")
        except Exception as exc:
            _log.trace("metrics_prune_error", error=str(exc))

    def log_turn(
        self,
        *,
        session_id: str,
        latency_ms: float,
        tool_calls: int,
        tool_errors: int,
        policy_blocked: int,
        task_success: bool,
        answer_revision_count: int,
        quality_issues: str,
        plan_completion_ratio: float = 1.0,
        answer_chars: int,
        heuristic_success=None, verified_success=None, verify_source=None,  # NEW P0-5
        tool_fallback_count: int = 0,  # E3
        tool_timeout_count: int = 0,   # E3
    ):
        now = datetime.now(timezone.utc).isoformat()
        ratio = max(0.0, min(1.0, float(plan_completion_ratio)))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO interaction_metrics
                   (session_id, latency_ms, tool_calls, tool_errors, policy_blocked,
                    task_success, answer_revision_count, quality_issues,
                    plan_completion_ratio, answer_chars,
                    heuristic_success, verified_success, verify_source,
                    tool_fallback_count, tool_timeout_count,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    float(latency_ms),
                    int(tool_calls),
                    int(tool_errors),
                    int(policy_blocked),
                    1 if task_success else 0,
                    int(answer_revision_count),
                    str(quality_issues or ""),
                    ratio,
                    int(answer_chars),
                    1 if heuristic_success else (0 if heuristic_success is not None else None),
                    1 if verified_success else (0 if verified_success is not None else None),
                    str(verify_source) if verify_source is not None else None,
                    int(tool_fallback_count),
                    int(tool_timeout_count),
                    now,
                ),
            )

    def log_runtime(
        self,
        *,
        run_id: str,
        queued_ms: float,
        stream_ms: float,
        tts_ms: float,
        total_ms: float,
        voice_mode: bool,
        final_state: str,
    ):
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO runtime_metrics
                   (run_id, queued_ms, stream_ms, tts_ms, total_ms, voice_mode, final_state, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(run_id or ""),
                    float(max(0.0, queued_ms)),
                    float(max(0.0, stream_ms)),
                    float(max(0.0, tts_ms)),
                    float(max(0.0, total_ms)),
                    1 if voice_mode else 0,
                    str(final_state or ""),
                    now,
                ),
            )

    def weekly_summary(self, days: int = 7) -> dict:
        start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT
                    COUNT(*),
                    AVG(latency_ms),
                    AVG(tool_calls),
                    SUM(tool_calls),
                    SUM(tool_errors),
                    SUM(policy_blocked),
                    AVG(task_success),
                    AVG(answer_revision_count),
                    AVG(plan_completion_ratio),
                    SUM(CASE WHEN heuristic_success = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN heuristic_success IS NOT NULL THEN 1 ELSE 0 END),
                    SUM(CASE WHEN verified_success = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN verified_success IS NOT NULL THEN 1 ELSE 0 END),
                    AVG(tool_fallback_count),
                    AVG(tool_timeout_count)
                   FROM interaction_metrics
                   WHERE created_at >= ?""",
                (start,),
            ).fetchone()
            runtime_rows = conn.execute(
                """SELECT
                    COUNT(*),
                    AVG(queued_ms),
                    AVG(stream_ms),
                    AVG(tts_ms),
                    AVG(total_ms)
                   FROM runtime_metrics
                   WHERE created_at >= ?""",
                (start,),
            ).fetchone()
            issue_rows = conn.execute(
                """SELECT session_id, quality_issues, tool_errors, policy_blocked, created_at
                   FROM interaction_metrics
                   WHERE created_at >= ?
                     AND (task_success = 0 OR tool_errors > 0 OR answer_revision_count > 0)
                   ORDER BY id DESC
                   LIMIT 5""",
                (start,),
            ).fetchall()
        (
            count,
            avg_latency,
            avg_tools,
            total_tool_calls,
            total_tool_errors,
            blocked,
            success_rate,
            avg_revisions,
            avg_plan_completion,
            heuristic_successes,
            heuristic_total,
            verified_successes,
            verified_total,
            avg_fallback_count,
            avg_timeout_count,
        ) = rows or (0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0)
        (
            runtime_count,
            avg_queue_ms,
            avg_stream_ms,
            avg_tts_ms,
            avg_total_ms,
        ) = runtime_rows or (0, 0, 0, 0, 0)
        tool_error_rate = (
            float(total_tool_errors or 0) / float(total_tool_calls or 1)
            if (total_tool_calls or 0) > 0
            else 0.0
        )
        issue_samples = [
            {
                "session_id": r[0],
                "quality_issues": r[1],
                "tool_errors": int(r[2] or 0),
                "policy_blocked": int(r[3] or 0),
                "created_at": r[4],
            }
            for r in issue_rows
        ]
        return {
            "turns": int(count or 0),
            "avg_latency_ms": float(avg_latency or 0.0),
            "avg_tool_calls": float(avg_tools or 0.0),
            "policy_blocked_total": int(blocked or 0),
            "task_success_rate": float(success_rate or 0.0),
            "avg_answer_revision_count": float(avg_revisions or 0.0),
            "avg_plan_completion_rate": float(avg_plan_completion or 0.0),
            "tool_error_rate": float(tool_error_rate),
            "runtime_samples": int(runtime_count or 0),
            "avg_queue_ms": float(avg_queue_ms or 0.0),
            "avg_stream_ms": float(avg_stream_ms or 0.0),
            "avg_tts_ms": float(avg_tts_ms or 0.0),
            "avg_total_ms": float(avg_total_ms or 0.0),
            "top_issue_samples": issue_samples,
            "heuristic_success_rate": (
                float(heuristic_successes or 0) / float(heuristic_total)
                if (heuristic_total or 0) > 0 else None
            ),
            "verified_success_rate": (
                float(verified_successes or 0) / float(verified_total)
                if (verified_total or 0) > 0 else None
            ),
            "avg_fallback_count": float(avg_fallback_count or 0.0),
            "avg_timeout_count": float(avg_timeout_count or 0.0),
        }
