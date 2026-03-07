"""Tests for audit schema migration and extended audit API."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path


class AuditMigrationTests(unittest.TestCase):
    """Test idempotent schema migration adds new columns."""

    def _make_policy(self, db_path):
        from liagent.tools.policy import ToolPolicy
        return ToolPolicy(db_path=db_path, tool_profile="full")

    def test_fresh_db_has_new_columns(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._make_policy(Path(td) / "audit.db")
            cols = {row[1] for row in p._conn.execute("PRAGMA table_info(tool_audit)").fetchall()}
            for col in ("requested_tool", "requested_args", "effective_tool",
                        "effective_args", "policy_decision", "grant_source"):
                self.assertIn(col, cols, f"Missing column: {col}")
            p.close()

    def test_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "audit.db"
            p1 = self._make_policy(db)
            p1.close()
            p2 = self._make_policy(db)
            cols = {row[1] for row in p2._conn.execute("PRAGMA table_info(tool_audit)").fetchall()}
            self.assertIn("policy_decision", cols)
            p2.close()

    def test_migration_preserves_existing_data(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "audit.db"
            conn = sqlite3.connect(db)
            conn.execute("""CREATE TABLE tool_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL, args_json TEXT NOT NULL,
                status TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            # Use recent timestamp to avoid _prune_audit() deletion
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("INSERT INTO tool_audit (tool_name, args_json, status, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                         ("web_search", '{"query":"test"}', "ok", "executed", now))
            conn.commit()
            conn.close()
            p = self._make_policy(db)
            rows = p._conn.execute("SELECT tool_name, policy_decision FROM tool_audit").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "web_search")
            self.assertEqual(rows[0][1], "")
            p.close()


class AuditExtendedFieldsTests(unittest.TestCase):
    """Test audit() accepts and stores new fields."""

    def _make_policy(self, db_path):
        from liagent.tools.policy import ToolPolicy
        return ToolPolicy(db_path=db_path, tool_profile="full")

    def test_audit_stores_new_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._make_policy(Path(td) / "audit.db")
            p.audit("web_search", {"query": "test"}, "ok", "executed",
                    requested_tool="web_search", requested_args='{"query":"test"}',
                    policy_decision="allowed", grant_source="")
            rows = p._conn.execute(
                "SELECT requested_tool, policy_decision, grant_source FROM tool_audit"
            ).fetchall()
            self.assertEqual(rows[0], ("web_search", "allowed", ""))
            p.close()

    def test_audit_stores_fallback_info(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._make_policy(Path(td) / "audit.db")
            p.audit("web_search", {"query": "test"}, "ok", "executed via fallback",
                    requested_tool="web_fetch", requested_args='{"url":"http://example.com"}',
                    effective_tool="web_search",
                    effective_args='{"query": "example.com"}',
                    policy_decision="allowed")
            rows = p._conn.execute(
                "SELECT requested_tool, effective_tool FROM tool_audit"
            ).fetchall()
            self.assertEqual(rows[0][0], "web_fetch")
            self.assertEqual(rows[0][1], "web_search")
            p.close()

    def test_args_json_redacted_on_write(self):
        """Primary args_json should be redacted at write time (P1 fix)."""
        with tempfile.TemporaryDirectory() as td:
            p = self._make_policy(Path(td) / "audit.db")
            p.audit("tool", {"api_key": "sk-secret123", "query": "hello"}, "ok", "test")
            rows = p._conn.execute("SELECT args_json FROM tool_audit").fetchall()
            self.assertIn("[REDACTED]", rows[0][0])
            self.assertNotIn("sk-secret123", rows[0][0])
            self.assertIn("hello", rows[0][0])
            p.close()

    def test_requested_args_redacted_on_write(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._make_policy(Path(td) / "audit.db")
            p.audit("tool", {}, "ok", "test",
                    requested_args='{"token": "mytoken123"}')
            rows = p._conn.execute("SELECT requested_args FROM tool_audit").fetchall()
            self.assertIn("[REDACTED]", rows[0][0])
            self.assertNotIn("mytoken123", rows[0][0])
            p.close()

    def test_effective_args_redacted_on_write(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._make_policy(Path(td) / "audit.db")
            p.audit("tool", {}, "ok", "test",
                    effective_args='{"api_key": "sk-secret123"}')
            rows = p._conn.execute("SELECT effective_args FROM tool_audit").fetchall()
            self.assertIn("[REDACTED]", rows[0][0])
            self.assertNotIn("sk-secret123", rows[0][0])
            p.close()

    def test_recent_audit_returns_new_fields(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._make_policy(Path(td) / "audit.db")
            p.audit("web_search", {"query": "test"}, "ok", "executed",
                    policy_decision="granted", grant_source="session_grant")
            rows = p.recent_audit(limit=1)
            self.assertEqual(rows[0]["policy_decision"], "granted")
            self.assertEqual(rows[0]["grant_source"], "session_grant")
            p.close()

    def test_backward_compat_old_callers(self):
        """Existing audit() calls without new kwargs must still work."""
        with tempfile.TemporaryDirectory() as td:
            p = self._make_policy(Path(td) / "audit.db")
            p.audit("web_search", {"query": "test"}, "ok", "executed")
            rows = p.recent_audit(limit=1)
            self.assertEqual(rows[0]["tool_name"], "web_search")
            self.assertNotIn("policy_decision", rows[0])  # empty → omitted
            p.close()


if __name__ == "__main__":
    unittest.main()
