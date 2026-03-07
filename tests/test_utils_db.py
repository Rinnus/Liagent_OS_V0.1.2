import os
import tempfile
import unittest


class ConnectDbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_wal_mode_enabled(self):
        from liagent.utils.db import connect_db
        conn = connect_db(self.db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(mode, "wal")

    def test_busy_timeout_set(self):
        from liagent.utils.db import connect_db
        conn = connect_db(self.db_path)
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.close()
        self.assertEqual(timeout, 5000)

    def test_returns_connection(self):
        from liagent.utils.db import connect_db
        import sqlite3
        conn = connect_db(self.db_path)
        self.assertIsInstance(conn, sqlite3.Connection)
        conn.close()


if __name__ == "__main__":
    unittest.main()
