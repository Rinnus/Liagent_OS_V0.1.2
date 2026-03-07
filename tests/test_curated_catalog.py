"""Tests for curated MCP catalog bootstrap."""

import json
import tempfile
import unittest
from pathlib import Path


class CuratedCatalogTests(unittest.TestCase):

    def test_bootstrap_creates_config_files(self):
        from liagent.tools.curated_catalog import bootstrap_curated_catalog
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            mcp_dir = Path(td) / "mcp.d"
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            json_files = list(mcp_dir.glob("*.json"))
            self.assertGreater(len(json_files), 0)

    def test_bootstrap_sets_approved_trust(self):
        from liagent.tools.curated_catalog import bootstrap_curated_catalog
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            mcp_dir = Path(td) / "mcp.d"
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            self.assertEqual(reg.get_status("filesystem"), "approved")
            self.assertEqual(reg.get_status("fetch"), "approved")

    def test_bootstrap_idempotent(self):
        from liagent.tools.curated_catalog import bootstrap_curated_catalog
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            mcp_dir = Path(td) / "mcp.d"
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            count1 = len(list(mcp_dir.glob("*.json")))
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            count2 = len(list(mcp_dir.glob("*.json")))
            self.assertEqual(count1, count2)

    def test_config_format_discoverable(self):
        from liagent.tools.curated_catalog import bootstrap_curated_catalog
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            mcp_dir = Path(td) / "mcp.d"
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            for f in mcp_dir.glob("*.json"):
                data = json.loads(f.read_text())
                self.assertIn("mcpServers", data)
                for name, spec in data["mcpServers"].items():
                    self.assertIn("command", spec)

    def test_no_empty_env_values(self):
        from liagent.tools.curated_catalog import bootstrap_curated_catalog
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            mcp_dir = Path(td) / "mcp.d"
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            for f in mcp_dir.glob("*.json"):
                data = json.loads(f.read_text())
                for name, spec in data.get("mcpServers", {}).items():
                    env = spec.get("env", {})
                    for k, v in env.items():
                        self.assertNotEqual(v, "",
                            f"Empty env value for {k} in {name} would overwrite real env var")

    def test_filesystem_not_home_dir(self):
        from liagent.tools.curated_catalog import CURATED_SERVERS
        fs_servers = [s for s in CURATED_SERVERS if s["name"] == "filesystem"]
        self.assertEqual(len(fs_servers), 1)
        args = fs_servers[0]["args"]
        home = str(Path.home())
        for arg in args:
            self.assertNotEqual(arg, home,
                f"filesystem MCP should not have unrestricted access to {home}")

    def test_catalog_contains_expected_servers(self):
        from liagent.tools.curated_catalog import CURATED_SERVERS
        names = {s["name"] for s in CURATED_SERVERS}
        self.assertIn("filesystem", names)
        self.assertIn("fetch", names)

    def test_bootstrap_preserves_user_revocation(self):
        """User revoking a curated server must survive restart bootstrap."""
        from liagent.tools.curated_catalog import bootstrap_curated_catalog
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            mcp_dir = Path(td) / "mcp.d"
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            # First bootstrap — approved
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            self.assertEqual(reg.get_status("filesystem"), "approved")
            # User revokes
            reg.set_status("filesystem", "revoked", source="manual")
            self.assertEqual(reg.get_status("filesystem"), "revoked")
            # Second bootstrap (simulates restart) — must NOT overwrite
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            self.assertEqual(reg.get_status("filesystem"), "revoked")

    def test_bootstrap_does_not_re_approve_already_approved(self):
        """Already approved servers should keep their existing source metadata."""
        from liagent.tools.curated_catalog import bootstrap_curated_catalog
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            mcp_dir = Path(td) / "mcp.d"
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            entry1 = reg.get_entry("fetch")
            # Second bootstrap
            bootstrap_curated_catalog(mcp_dir=mcp_dir, trust_registry=reg)
            entry2 = reg.get_entry("fetch")
            # updated_at should NOT change (no write happened)
            self.assertEqual(entry1["updated_at"], entry2["updated_at"])


if __name__ == "__main__":
    unittest.main()
