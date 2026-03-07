"""Tests for pre-connection trust filtering.

Pre-connection filter blocks REVOKED servers only.
Unknown servers are allowed to connect — their tools are blocked at evaluate()
until the user confirms (first-use flow).
"""

import tempfile
import unittest
from pathlib import Path

from liagent.config import MCPServerConfig


def _filter_for_connection(reg, servers):
    """Replicate _resolve_mcp_servers trust filter: only revoked is blocked."""
    return [s for s in servers if reg.get_status(s.name) != "revoked"]


class PreConnectionTrustFilterTests(unittest.TestCase):
    """Verify is_connectable() (approved-only) semantics."""

    def test_unknown_server_not_connectable(self):
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            self.assertFalse(reg.is_connectable("newserver"))

    def test_approved_server_connectable(self):
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            reg.set_status("github", "approved", source="curated")
            self.assertTrue(reg.is_connectable("github"))

    def test_revoked_server_not_connectable(self):
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            reg.set_status("badserver", "revoked", source="manual")
            self.assertFalse(reg.is_connectable("badserver"))

    def test_ensure_registered_records_unknown(self):
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            servers = [MCPServerConfig(name="new_a", command="echo")]
            for s in servers:
                reg.ensure_registered(s.name, source="discovered")
            self.assertEqual(reg.get_status("new_a"), "unknown")
            self.assertFalse(reg.is_connectable("new_a"))


class ConnectionFilterTests(unittest.TestCase):
    """Verify the pre-connection filter: revoked blocked, unknown allowed."""

    def test_revoked_server_blocked_from_connection(self):
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            reg.set_status("bad", "revoked", source="manual")
            servers = [MCPServerConfig(name="bad", command="echo")]
            result = _filter_for_connection(reg, servers)
            self.assertEqual(len(result), 0)

    def test_unknown_server_allowed_to_connect(self):
        """Unknown servers connect — their tools are gated by evaluate()."""
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            servers = [MCPServerConfig(name="new_server", command="echo")]
            result = _filter_for_connection(reg, servers)
            self.assertEqual(len(result), 1)

    def test_approved_server_allowed_to_connect(self):
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            reg.set_status("good", "approved", source="curated")
            servers = [MCPServerConfig(name="good", command="echo")]
            result = _filter_for_connection(reg, servers)
            self.assertEqual(len(result), 1)

    def test_mixed_servers_only_revoked_blocked(self):
        from liagent.tools.trust_registry import TrustRegistry

        with tempfile.TemporaryDirectory() as td:
            reg = TrustRegistry(store_path=Path(td) / "trust.json")
            reg.set_status("approved_srv", "approved", source="curated")
            reg.set_status("revoked_srv", "revoked", source="manual")
            # unknown_srv not in registry → unknown
            servers = [
                MCPServerConfig(name="approved_srv", command="echo"),
                MCPServerConfig(name="revoked_srv", command="echo"),
                MCPServerConfig(name="unknown_srv", command="echo"),
            ]
            result = _filter_for_connection(reg, servers)
            names = [s.name for s in result]
            self.assertEqual(sorted(names), ["approved_srv", "unknown_srv"])


if __name__ == "__main__":
    unittest.main()
