"""Tests for Tool Trust Registry."""

import json
import os
import tempfile
import unittest
from pathlib import Path


class TrustRegistryReadWriteTests(unittest.TestCase):
    """Test TrustRegistry CRUD operations."""

    def _make_registry(self, tmp_dir: str) -> "TrustRegistry":
        from liagent.tools.trust_registry import TrustRegistry
        return TrustRegistry(store_path=Path(tmp_dir) / "tool_trust.json")

    def test_empty_registry_returns_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            self.assertEqual(reg.get_status("github"), "unknown")

    def test_set_and_get_approved(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            reg.set_status("github", "approved", source="curated")
            self.assertEqual(reg.get_status("github"), "approved")

    def test_set_and_get_revoked(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            reg.set_status("github", "revoked", source="manual")
            self.assertEqual(reg.get_status("github"), "revoked")

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as td:
            reg1 = self._make_registry(td)
            reg1.set_status("sqlite", "approved", source="curated")
            reg2 = self._make_registry(td)
            self.assertEqual(reg2.get_status("sqlite"), "approved")

    def test_get_entry_returns_full_record(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            reg.set_status("github", "approved", source="curated")
            entry = reg.get_entry("github")
            self.assertIsNotNone(entry)
            self.assertEqual(entry["status"], "approved")
            self.assertEqual(entry["source"], "curated")
            self.assertIn("updated_at", entry)

    def test_get_entry_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            self.assertIsNone(reg.get_entry("nonexistent"))

    def test_list_by_status(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            reg.set_status("a", "approved", source="curated")
            reg.set_status("b", "approved", source="first_use")
            reg.set_status("c", "revoked", source="manual")
            reg.set_status("d", "unknown", source="discovered")
            approved = reg.list_by_status("approved")
            self.assertEqual(set(approved), {"a", "b"})
            revoked = reg.list_by_status("revoked")
            self.assertEqual(set(revoked), {"c"})

    def test_invalid_status_raises(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            with self.assertRaises(ValueError):
                reg.set_status("x", "invalid_status", source="test")

    def test_ensure_registered_sets_unknown_if_new(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            reg.ensure_registered("new_server", source="discovered")
            self.assertEqual(reg.get_status("new_server"), "unknown")

    def test_ensure_registered_does_not_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            reg.set_status("github", "approved", source="curated")
            reg.ensure_registered("github", source="discovered")
            self.assertEqual(reg.get_status("github"), "approved")

    def test_is_connectable_approved_only(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td)
            self.assertFalse(reg.is_connectable("x"))          # unknown
            reg.set_status("x", "approved", source="curated")
            self.assertTrue(reg.is_connectable("x"))            # approved
            reg.set_status("x", "revoked", source="manual")
            self.assertFalse(reg.is_connectable("x"))           # revoked


class TrustRegistryAtomicWriteTests(unittest.TestCase):
    """Test that writes are atomic (tempfile + os.replace)."""

    def test_write_creates_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            from liagent.tools.trust_registry import TrustRegistry
            path = Path(td) / "trust.json"
            reg = TrustRegistry(store_path=path)
            reg.set_status("test", "approved", source="curated")
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("test", data)

    def test_corrupted_file_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as td:
            from liagent.tools.trust_registry import TrustRegistry
            path = Path(td) / "trust.json"
            path.write_text("{{not valid json", encoding="utf-8")
            reg = TrustRegistry(store_path=path)
            self.assertEqual(reg.get_status("anything"), "unknown")


class TrustRegistryMalformedEntryTests(unittest.TestCase):
    """Test robustness against structurally valid but semantically wrong JSON."""

    def _make_registry(self, tmp_dir: str, data: dict) -> "TrustRegistry":
        from liagent.tools.trust_registry import TrustRegistry
        path = Path(tmp_dir) / "trust.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return TrustRegistry(store_path=path)

    def test_string_entry_returns_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td, {"server_a": "just a string"})
            self.assertEqual(reg.get_status("server_a"), "unknown")

    def test_list_entry_returns_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td, {"server_a": [1, 2, 3]})
            self.assertEqual(reg.get_status("server_a"), "unknown")

    def test_int_entry_returns_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td, {"server_a": 42})
            self.assertEqual(reg.get_status("server_a"), "unknown")

    def test_get_entry_returns_none_for_non_dict(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td, {"server_a": "bad"})
            self.assertIsNone(reg.get_entry("server_a"))

    def test_list_by_status_skips_non_dict_entries(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td, {
                "good": {"status": "approved", "source": "curated", "updated_at": ""},
                "bad": "not a dict",
            })
            approved = reg.list_by_status("approved")
            self.assertEqual(approved, ["good"])

    def test_malformed_entries_filtered_on_load(self):
        with tempfile.TemporaryDirectory() as td:
            reg = self._make_registry(td, {
                "valid": {"status": "approved", "source": "test", "updated_at": ""},
                "string_val": "oops",
                "list_val": [1, 2],
                "int_val": 99,
            })
            # Only the valid entry should survive load
            self.assertEqual(reg.get_status("valid"), "approved")
            self.assertEqual(reg.get_status("string_val"), "unknown")
            # After set_status on a new server, the old malformed entries should be gone from file
            reg.set_status("new_one", "approved", source="test")
            raw = json.loads(Path(td, "trust.json").read_text())
            self.assertNotIn("string_val", raw)
            self.assertNotIn("list_val", raw)


if __name__ == "__main__":
    unittest.main()
