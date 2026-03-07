"""Shared SQLite connection factory with WAL and busy_timeout."""
from __future__ import annotations
import sqlite3


def connect_db(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with WAL journal mode and 5s busy timeout."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
