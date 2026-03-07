"""Event Log — append-only SQLite log driving all knowledge state changes.

SQLite is the source of truth.  Every fact, evidence, or task mutation goes
through the event log.  Downstream consumers replay events to project state.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..logging import get_logger

_log = get_logger("event_log")

_DEFAULT_DB = Path.home() / ".liagent" / "knowledge.db"


@dataclass
class ChangeEvent:
    """A single state mutation."""

    entity_type: str  # "fact" | "evidence" | "task" | "session"
    entity_id: str
    action: str  # "upsert" | "delete" | "vault_edit"
    payload: dict = field(default_factory=dict)
    timestamp: float = 0.0  # Filled on append
    seq: int = 0  # Filled on replay


class EventLog:
    """Append-only event log backed by SQLite."""

    def __init__(self, db_path: Path | str = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Note: single-writer pattern — all writes go through append() which holds the lock.
        # No method re-enters another locked method, so a plain Lock suffices.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        result = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if result and result[0].lower() != "wal":
            _log.trace("wal_mode_failed", actual=result[0])
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS change_events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                action TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_entity
                ON change_events(entity_type, entity_id);
            CREATE INDEX IF NOT EXISTS idx_events_seq
                ON change_events(seq);
        """)
        self._conn.commit()

    def append(self, event: ChangeEvent) -> int:
        """Append an event.  Returns the assigned sequence number."""
        ts = event.timestamp or time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO change_events "
                "(entity_type, entity_id, action, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    event.entity_type,
                    event.entity_id,
                    event.action,
                    json.dumps(event.payload, ensure_ascii=False),
                    ts,
                ),
            )
            self._conn.commit()
            seq = cur.lastrowid
        _log.trace("event_appended", seq=seq, entity=event.entity_id, action=event.action)
        return seq

    def replay(
        self, since_seq: int = 0, entity_type: str | None = None,
    ) -> list[ChangeEvent]:
        """Replay events after a given sequence number."""
        sql = (
            "SELECT seq, entity_type, entity_id, action, payload_json, created_at "
            "FROM change_events WHERE seq > ?"
        )
        params: list = [since_seq]
        if entity_type:
            sql += " AND entity_type = ?"
            params.append(entity_type)
        sql += " ORDER BY seq"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            ChangeEvent(
                entity_type=etype,
                entity_id=eid,
                action=action,
                payload=json.loads(pjson),
                timestamp=ts,
                seq=seq,
            )
            for seq, etype, eid, action, pjson, ts in rows
        ]

    def latest_seq(self) -> int:
        """Return the latest sequence number, or 0 if empty."""
        with self._lock:
            row = self._conn.execute("SELECT MAX(seq) FROM change_events").fetchone()
        return int(row[0] or 0) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
