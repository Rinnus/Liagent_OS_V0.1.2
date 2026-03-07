"""GoalStore — persistence layer for autonomous goals, events, pattern groups, and outbox."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional


def _now_iso() -> str:
    """UTC timestamp in SQLite-compatible format (space separator, no TZ suffix)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple) -> dict:
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


class GoalStore:
    """SQLite-backed store for autonomous goals and related data."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS autonomous_goals (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key       TEXT,
                source                TEXT NOT NULL,
                domain                TEXT,
                objective             TEXT NOT NULL,
                rationale             TEXT,
                state                 TEXT NOT NULL DEFAULT 'proposed',
                priority              INTEGER DEFAULT 0,
                confidence            REAL DEFAULT 0.5,
                budget_json           TEXT,
                success_criteria_json TEXT,
                source_patterns_json  TEXT,
                source_group_id       INTEGER,
                source_discovery_id   INTEGER,
                user_consent          TEXT NOT NULL DEFAULT 'pending',
                last_review_at        TEXT,
                next_review_at        TEXT,
                created_at            TEXT NOT NULL,
                updated_at            TEXT NOT NULL,
                retired_reason        TEXT
            );

            CREATE TABLE IF NOT EXISTS goal_events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id      INTEGER,
                event_type   TEXT NOT NULL,
                payload_json TEXT,
                summary      TEXT,
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pattern_groups (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                group_key      TEXT NOT NULL UNIQUE,
                domain         TEXT,
                label          TEXT,
                entities_json  TEXT,
                intents_json   TEXT,
                support_count  INTEGER DEFAULT 0,
                last_seen      TEXT NOT NULL,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS decision_outbox (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type  TEXT NOT NULL,
                payload_json TEXT,
                status       TEXT NOT NULL DEFAULT 'pending',
                created_at   TEXT NOT NULL,
                updated_at   TEXT,
                attempts     INTEGER NOT NULL DEFAULT 0,
                error_text   TEXT
            );
        """)

        outbox_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(decision_outbox)").fetchall()
        }
        if "updated_at" not in outbox_cols:
            conn.execute("ALTER TABLE decision_outbox ADD COLUMN updated_at TEXT")
        if "attempts" not in outbox_cols:
            conn.execute("ALTER TABLE decision_outbox ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
        if "error_text" not in outbox_cols:
            conn.execute("ALTER TABLE decision_outbox ADD COLUMN error_text TEXT")

        # Indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_goals_idemp_active
            ON autonomous_goals(idempotency_key)
            WHERE idempotency_key IS NOT NULL
              AND state IN ('proposed', 'active', 'paused')
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_goals_state_review
            ON autonomous_goals(state, next_review_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_goal_events_goal_created
            ON goal_events(goal_id, created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_groups_last_seen
            ON pattern_groups(last_seen)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_outbox_pending
            ON decision_outbox(status, id)
            WHERE status = 'pending'
        """)

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Goal CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        source: str,
        objective: str,
        *,
        domain: str | None = None,
        rationale: str | None = None,
        confidence: float = 0.5,
        priority: int = 0,
        idempotency_key: str | None = None,
        budget_json: str | None = None,
        success_criteria_json: str | None = None,
        source_patterns_json: str | None = None,
        source_group_id: int | None = None,
        source_discovery_id: int | None = None,
    ) -> int | None:
        """Insert a new goal. Returns row id, or None if idempotency_key collides."""
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            # Check idempotency: reject if key already exists on a non-retired goal
            if idempotency_key is not None:
                existing = conn.execute(
                    """
                    SELECT id FROM autonomous_goals
                    WHERE idempotency_key = ?
                      AND state IN ('proposed', 'active', 'paused')
                    """,
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    return None
            cur = conn.execute(
                """
                INSERT INTO autonomous_goals
                    (idempotency_key, source, domain, objective, rationale,
                     confidence, priority, budget_json, success_criteria_json,
                     source_patterns_json, source_group_id, source_discovery_id,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key, source, domain, objective, rationale,
                    confidence, priority, budget_json, success_criteria_json,
                    source_patterns_json, source_group_id, source_discovery_id,
                    now, now,
                ),
            )
            return cur.lastrowid

    def get(self, goal_id: int) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = _row_to_dict
            row = conn.execute(
                "SELECT * FROM autonomous_goals WHERE id = ?", (goal_id,)
            ).fetchone()
        return row

    def get_by_state(self, state: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = _row_to_dict
            return conn.execute(
                "SELECT * FROM autonomous_goals WHERE state = ? ORDER BY priority DESC, id",
                (state,),
            ).fetchall()

    def count_active(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM autonomous_goals WHERE state IN ('active', 'proposed')"
            ).fetchone()
            return row[0]

    def count_created_today(self) -> int:
        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM autonomous_goals WHERE created_at LIKE ?",
                (today_prefix + "%",),
            ).fetchone()
            return row[0]

    def transition(self, goal_id: int, new_state: str, *, reason: str | None = None) -> None:
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE autonomous_goals
                SET state = ?, updated_at = ?, retired_reason = COALESCE(?, retired_reason)
                WHERE id = ?
                """,
                (new_state, now, reason, goal_id),
            )

    def update(self, goal_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _now_iso()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [goal_id]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE autonomous_goals SET {set_clause} WHERE id = ?", values
            )

    def adjust_confidence(self, goal_id: int, *, delta: float) -> None:
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE autonomous_goals
                SET confidence = MIN(1.0, MAX(0.0, confidence + ?)),
                    updated_at = ?
                WHERE id = ?
                """,
                (delta, now, goal_id),
            )

    def get_due_for_review(self) -> list[dict]:
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = _row_to_dict
            return conn.execute(
                """
                SELECT * FROM autonomous_goals
                WHERE state IN ('active', 'paused')
                  AND (next_review_at IS NULL OR next_review_at <= ?)
                ORDER BY next_review_at, id
                """,
                (now,),
            ).fetchall()

    def get_stale_goals(self, *, days: int = 14) -> list[dict]:
        """Goals with no activity for *days* days (candidates for auto-retire)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = _row_to_dict
            return conn.execute(
                """
                SELECT * FROM autonomous_goals
                WHERE state IN ('active', 'paused')
                  AND updated_at <= datetime('now', ?)
                ORDER BY updated_at
                """,
                (f"-{days} days",),
            ).fetchall()

    def remaining_daily_budget(self, *, max_goals: int = 3, max_task_runs: int = 20) -> dict:
        """Track remaining daily budget for autonomous goal/task creation."""
        today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        goals_created = self.count_created_today()
        with sqlite3.connect(self.db_path) as conn:
            task_events = conn.execute(
                "SELECT COUNT(*) FROM goal_events WHERE event_type IN ('task_completed', 'task_failed') AND created_at LIKE ?",
                (today_prefix + "%",),
            ).fetchone()[0]
        return {
            "goals_remaining": max(0, max_goals - goals_created),
            "task_runs_remaining": max(0, max_task_runs - task_events),
        }

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def record_event(
        self,
        goal_id: int | None,
        event_type: str,
        payload: dict | None = None,
        *,
        summary: str | None = None,
    ) -> int:
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO goal_events (goal_id, event_type, payload_json, summary, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (goal_id, event_type, json.dumps(payload) if payload else None, summary, now),
            )
            return cur.lastrowid

    def get_events(
        self,
        goal_id: int | None,
        *,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = _row_to_dict
            if goal_id is not None:
                if event_type:
                    return conn.execute(
                        "SELECT * FROM goal_events WHERE goal_id = ? AND event_type = ? ORDER BY created_at DESC LIMIT ?",
                        (goal_id, event_type, limit),
                    ).fetchall()
                return conn.execute(
                    "SELECT * FROM goal_events WHERE goal_id = ? ORDER BY created_at DESC LIMIT ?",
                    (goal_id, limit),
                ).fetchall()
            else:
                if event_type:
                    return conn.execute(
                        "SELECT * FROM goal_events WHERE goal_id IS NULL AND event_type = ? ORDER BY created_at DESC LIMIT ?",
                        (event_type, limit),
                    ).fetchall()
                return conn.execute(
                    "SELECT * FROM goal_events WHERE goal_id IS NULL ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

    # ------------------------------------------------------------------
    # Pattern Groups
    # ------------------------------------------------------------------

    def create_group(
        self,
        group_key: str,
        domain: str,
        *,
        entities: list[str] | None = None,
        intents: list[str] | None = None,
        support_count: int = 0,
    ) -> int:
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO pattern_groups
                    (group_key, domain, entities_json, intents_json, support_count,
                     last_seen, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    group_key, domain,
                    json.dumps(entities) if entities else None,
                    json.dumps(intents) if intents else None,
                    support_count, now, now, now,
                ),
            )
            return cur.lastrowid

    def get_group_by_key(self, group_key: str) -> dict | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = _row_to_dict
            return conn.execute(
                "SELECT * FROM pattern_groups WHERE group_key = ?", (group_key,)
            ).fetchone()

    def get_unlabeled_groups(self, *, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = _row_to_dict
            return conn.execute(
                "SELECT * FROM pattern_groups WHERE label IS NULL ORDER BY support_count DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def get_recent_groups(self, *, hours: int = 24, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = _row_to_dict
            return conn.execute(
                """
                SELECT * FROM pattern_groups
                WHERE last_seen >= datetime('now', ?)
                ORDER BY last_seen DESC LIMIT ?
                """,
                (f"-{hours} hours", limit),
            ).fetchall()

    def has_recent_updates(self, *, hours: int = 24) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM pattern_groups
                WHERE updated_at >= datetime('now', ?)
                """,
                (f"-{hours} hours",),
            ).fetchone()
            return row[0] > 0

    def set_group_label(self, group_id: int, label: str) -> None:
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE pattern_groups SET label = ?, updated_at = ? WHERE id = ?",
                (label, now, group_id),
            )

    def update_group_support(self, group_id: int, *, new_count: int) -> None:
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE pattern_groups SET support_count = ?, last_seen = ?, updated_at = ? WHERE id = ?",
                (new_count, now, now, group_id),
            )

    # ------------------------------------------------------------------
    # Decision Outbox
    # ------------------------------------------------------------------

    @staticmethod
    def insert_outbox(conn: sqlite3.Connection, action_type: str, payload: dict | None = None) -> int:
        now = _now_iso()
        cur = conn.execute(
            """
            INSERT INTO decision_outbox (action_type, payload_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (action_type, json.dumps(payload) if payload else None, now, now),
        )
        return cur.lastrowid

    def drain_outbox(self, *, limit: int = 50) -> list[dict]:
        """Atomically claim pending outbox rows for delivery."""
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = _row_to_dict
            rows = conn.execute(
                """
                SELECT * FROM decision_outbox
                WHERE status = 'pending'
                ORDER BY id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            if rows:
                ids = [r["id"] for r in rows]
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"""
                    UPDATE decision_outbox
                    SET status = 'processing', updated_at = ?, attempts = attempts + 1
                    WHERE id IN ({placeholders}) AND status = 'pending'
                    """,
                    [now, *ids],
                )
            return rows

    def complete_outbox(self, entry_id: int, *, error_text: str = "") -> None:
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE decision_outbox
                SET status = 'done', updated_at = ?, error_text = ?
                WHERE id = ?
                """,
                (now, error_text, entry_id),
            )

    def retry_outbox(self, entry_id: int, *, error_text: str = "") -> None:
        now = _now_iso()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE decision_outbox
                SET status = 'pending', updated_at = ?, error_text = ?
                WHERE id = ?
                """,
                (now, error_text, entry_id),
            )
