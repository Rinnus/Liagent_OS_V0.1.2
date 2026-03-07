"""Tests for capability_inventory — E1 perception layer."""

import json
import threading

import pytest

from liagent.agent.capability_inventory import (
    CapabilityInventory,
    FieldChange,
    InventoryDiff,
    ToolFingerprint,
    _canonical_param_hash,
    _classify_param_diff,
    _migrate_inventory,
    build_capability_summary,
)
from liagent.tools import ToolCapability, ToolDef


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_tool(
    name: str = "web_search",
    description: str = "Search the web",
    parameters: dict | None = None,
    risk_level: str = "low",
) -> ToolDef:
    if parameters is None:
        parameters = {"properties": {"query": {"type": "string"}}, "required": ["query"]}
    return ToolDef(
        name=name,
        description=description,
        parameters=parameters,
        func=lambda **kw: "",
        risk_level=risk_level,
    )


def _make_mcp_tool(server: str = "brave", tool: str = "search", risk: str = "low") -> ToolDef:
    return _make_tool(
        name=f"{server}__{tool}",
        description=f"MCP {tool} from {server}",
        risk_level=risk,
    )


def _tools_dict(*tools: ToolDef) -> dict[str, ToolDef]:
    return {t.name: t for t in tools}


# ── TestToolFingerprint ──────────────────────────────────────────────────────

class TestToolFingerprint:

    def test_from_builtin_tool(self):
        td = _make_tool("web_search", "Search the web")
        fp = ToolFingerprint.from_tool_def(td)
        assert fp.name == "web_search"
        assert fp.source == "builtin"
        assert fp.risk_level == "low"
        assert fp.schema_version == 1
        assert len(fp.param_hash) == 12

    def test_from_mcp_tool(self):
        td = _make_mcp_tool("brave", "search")
        fp = ToolFingerprint.from_tool_def(td)
        assert fp.source == "mcp:brave"
        assert fp.name == "brave__search"

    def test_canonical_param_hash_key_order_stability(self):
        """Same schema with different key order → same hash."""
        a = {"b": 1, "a": 2, "c": {"z": 3, "y": 4}}
        b = {"a": 2, "c": {"y": 4, "z": 3}, "b": 1}
        assert _canonical_param_hash(a) == _canonical_param_hash(b)

    def test_canonical_param_hash_schema_ignored(self):
        """$schema is metadata — should be stripped."""
        a = {"$schema": "http://json-schema.org/draft-07/schema", "type": "object"}
        b = {"type": "object"}
        assert _canonical_param_hash(a) == _canonical_param_hash(b)

    def test_canonical_param_hash_additional_properties_preserved(self):
        """additionalProperties is semantic — should affect hash."""
        a = {"type": "object", "additionalProperties": True}
        b = {"type": "object", "additionalProperties": False}
        assert _canonical_param_hash(a) != _canonical_param_hash(b)

    def test_equality(self):
        td = _make_tool()
        fp1 = ToolFingerprint.from_tool_def(td)
        fp2 = ToolFingerprint.from_tool_def(td)
        assert fp1 == fp2

    def test_roundtrip_dict(self):
        td = _make_tool()
        fp = ToolFingerprint.from_tool_def(td)
        d = fp.to_dict()
        fp2 = ToolFingerprint.from_dict(d)
        assert fp == fp2


# ── TestFieldChange ──────────────────────────────────────────────────────────

class TestFieldChange:

    def test_desc_changed(self):
        fc = FieldChange("web_search", "desc_changed", "old desc", "new desc")
        assert fc.change_type == "desc_changed"
        assert fc.tool_name == "web_search"

    def test_param_changed(self):
        fc = FieldChange("stock", "param_changed", "abc123", "def456")
        assert fc.change_type == "param_changed"

    def test_risk_changed(self):
        fc = FieldChange("python_exec", "risk_changed", "high", "medium")
        assert fc.change_type == "risk_changed"


# ── TestInventoryDiff ────────────────────────────────────────────────────────

class TestInventoryDiff:

    def test_has_changes_empty(self):
        diff = InventoryDiff([], [], [])
        assert not diff.has_changes

    def test_has_changes_added(self):
        diff = InventoryDiff(["new_tool"], [], [])
        assert diff.has_changes

    def test_has_changes_modified(self):
        fc = FieldChange("t", "desc_changed", "a", "b")
        diff = InventoryDiff([], [], [fc])
        assert diff.has_changes

    def test_event_hash_stability(self):
        """Same changes → same hash."""
        fc = FieldChange("t", "desc_changed", "a", "b")
        d1 = InventoryDiff(["x"], ["y"], [fc])
        d2 = InventoryDiff(["x"], ["y"], [fc])
        assert d1.event_hash == d2.event_hash

    def test_event_hash_differs_for_different_changes(self):
        d1 = InventoryDiff(["x"], [], [])
        d2 = InventoryDiff(["y"], [], [])
        assert d1.event_hash != d2.event_hash

    def test_activity_lines_format(self):
        diff = InventoryDiff(
            added=["tool_a", "tool_b"],
            removed=["tool_c"],
            modified=[FieldChange("tool_d", "desc_changed", "old", "new")],
        )
        lines = diff.to_activity_lines()
        assert any("+ tool_a" in line for line in lines)
        assert any("+ tool_b" in line for line in lines)
        assert any("- tool_c" in line for line in lines)
        assert any("~ tool_d" in line for line in lines)

    def test_activity_lines_truncation(self):
        diff = InventoryDiff(
            added=[f"tool_{i}" for i in range(8)],
            removed=[],
            modified=[],
        )
        lines = diff.to_activity_lines()
        # 5 individual + 1 "... and 3 more added"
        assert len(lines) == 6
        assert "3 more added" in lines[-1]


# ── TestCapabilityInventory ──────────────────────────────────────────────────

class TestCapabilityInventory:

    def _make_inv(self, tmpdir) -> CapabilityInventory:
        return CapabilityInventory(snapshot_path=tmpdir / "cap.json")

    def test_first_run_all_added(self, tmp_path):
        inv = self._make_inv(tmp_path)
        tools = _tools_dict(_make_tool("a"), _make_tool("b"))
        diff = inv.refresh(tools)
        assert sorted(diff.added) == ["a", "b"]
        assert diff.removed == []
        assert diff.modified == []

    def test_no_change(self, tmp_path):
        inv = self._make_inv(tmp_path)
        tools = _tools_dict(_make_tool("a"))
        inv.refresh(tools)
        diff = inv.refresh(tools)
        assert not diff.has_changes

    def test_added_and_removed(self, tmp_path):
        inv = self._make_inv(tmp_path)
        inv.refresh(_tools_dict(_make_tool("a"), _make_tool("b")))
        diff = inv.refresh(_tools_dict(_make_tool("b"), _make_tool("c")))
        assert diff.added == ["c"]
        assert diff.removed == ["a"]

    def test_modified_desc(self, tmp_path):
        inv = self._make_inv(tmp_path)
        inv.refresh(_tools_dict(_make_tool("a", description="old")))
        diff = inv.refresh(_tools_dict(_make_tool("a", description="new")))
        assert len(diff.modified) == 1
        assert diff.modified[0].change_type == "desc_changed"

    def test_modified_params(self, tmp_path):
        inv = self._make_inv(tmp_path)
        p1 = {"properties": {"x": {"type": "string"}}}
        p2 = {"properties": {"x": {"type": "integer"}}}
        inv.refresh(_tools_dict(_make_tool("a", parameters=p1)))
        diff = inv.refresh(_tools_dict(_make_tool("a", parameters=p2)))
        assert any(c.change_type == "param_changed" for c in diff.modified)

    def test_modified_risk(self, tmp_path):
        inv = self._make_inv(tmp_path)
        inv.refresh(_tools_dict(_make_tool("a", risk_level="low")))
        diff = inv.refresh(_tools_dict(_make_tool("a", risk_level="high")))
        assert any(c.change_type == "risk_changed" for c in diff.modified)

    def test_multiple_field_changes(self, tmp_path):
        inv = self._make_inv(tmp_path)
        inv.refresh(_tools_dict(_make_tool("a", description="old", risk_level="low")))
        diff = inv.refresh(_tools_dict(_make_tool("a", description="new", risk_level="high")))
        types = {c.change_type for c in diff.modified}
        assert "desc_changed" in types
        assert "risk_changed" in types

    def test_cross_instance_persistence(self, tmp_path):
        path = tmp_path / "cap.json"
        inv1 = CapabilityInventory(snapshot_path=path)
        inv1.refresh(_tools_dict(_make_tool("a"), _make_tool("b")))

        inv2 = CapabilityInventory(snapshot_path=path)
        assert inv2.had_prior_session
        assert inv2.tool_names == {"a", "b"}
        diff = inv2.refresh(_tools_dict(_make_tool("a"), _make_tool("c")))
        assert diff.added == ["c"]
        assert diff.removed == ["b"]

    def test_cross_session_diff(self, tmp_path):
        """New instance detects changes from previous session's snapshot."""
        path = tmp_path / "cap.json"
        inv1 = CapabilityInventory(snapshot_path=path)
        inv1.refresh(_tools_dict(_make_tool("a")))

        inv2 = CapabilityInventory(snapshot_path=path)
        diff = inv2.refresh(_tools_dict(_make_tool("a"), _make_tool("b")))
        assert diff.added == ["b"]

    def test_corrupt_json_fallback(self, tmp_path):
        path = tmp_path / "cap.json"
        path.write_text("NOT JSON {{{", encoding="utf-8")
        inv = CapabilityInventory(snapshot_path=path)
        assert not inv.had_prior_session
        assert inv.tool_count == 0

    def test_missing_file_fallback(self, tmp_path):
        inv = CapabilityInventory(snapshot_path=tmp_path / "nonexistent.json")
        assert not inv.had_prior_session
        diff = inv.refresh(_tools_dict(_make_tool("a")))
        assert diff.added == ["a"]

    def test_tool_count_and_names(self, tmp_path):
        inv = self._make_inv(tmp_path)
        inv.refresh(_tools_dict(_make_tool("a"), _make_tool("b"), _make_tool("c")))
        assert inv.tool_count == 3
        assert inv.tool_names == {"a", "b", "c"}

    def test_had_prior_session_false_on_first_install(self, tmp_path):
        inv = self._make_inv(tmp_path)
        assert not inv.had_prior_session


# ── TestEventDedup ───────────────────────────────────────────────────────────

class TestEventDedup:

    def test_same_hash_within_window_suppressed(self, tmp_path):
        inv = CapabilityInventory(snapshot_path=tmp_path / "cap.json")
        tools_a = _tools_dict(_make_tool("a"))

        diff1 = inv.refresh(tools_a)
        assert diff1.has_changes  # first time, "a" added

        # Reset internal state to force same diff again
        inv._current = {}
        diff2 = inv.refresh(tools_a)
        # Same event_hash within 5 min → suppressed
        assert not diff2.has_changes

    def test_different_hash_within_window_not_suppressed(self, tmp_path):
        """Different diffs within 5 minutes must NOT be suppressed (P1-1 key test)."""
        inv = CapabilityInventory(snapshot_path=tmp_path / "cap.json")

        diff1 = inv.refresh(_tools_dict(_make_tool("a")))
        assert diff1.has_changes

        # Different change: add "b"
        diff2 = inv.refresh(_tools_dict(_make_tool("a"), _make_tool("b")))
        assert diff2.has_changes  # different hash → not suppressed
        assert diff2.added == ["b"]

    def test_same_hash_after_window_not_suppressed(self, tmp_path):
        inv = CapabilityInventory(snapshot_path=tmp_path / "cap.json")
        tools_a = _tools_dict(_make_tool("a"))

        diff1 = inv.refresh(tools_a)
        assert diff1.has_changes

        # Simulate window expiry
        inv._last_event_ts -= 301
        inv._current = {}  # force re-add
        diff2 = inv.refresh(tools_a)
        assert diff2.has_changes  # window expired → not suppressed


# ── TestConcurrency ──────────────────────────────────────────────────────────

class TestConcurrency:

    def test_concurrent_refresh_no_race(self, tmp_path):
        """Multiple threads calling refresh() must not crash or corrupt state."""
        inv = CapabilityInventory(snapshot_path=tmp_path / "cap.json")
        errors: list[Exception] = []

        def worker(i: int):
            try:
                tools = _tools_dict(
                    _make_tool("shared"),
                    _make_tool(f"tool_{i}"),
                )
                for _ in range(10):
                    inv.refresh(tools)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent refresh errors: {errors}"
        assert inv.tool_count > 0


# ── TestBuildCapabilitySummary ───────────────────────────────────────────────

class TestBuildCapabilitySummary:

    def test_full_toolset(self):
        names = {"web_search", "stock", "read_file", "list_dir", "write_file",
                 "python_exec", "describe_image", "create_task", "web_fetch"}
        summary = build_capability_summary(names, tool_profile="full", max_chars=300)
        assert "search" in summary.lower()
        assert "run code" in summary
        assert "write files" in summary

    def test_full_toolset_default_truncates(self):
        """Default max_chars truncates a large toolset."""
        names = {"web_search", "stock", "read_file", "list_dir", "write_file",
                 "python_exec", "describe_image", "create_task", "web_fetch"}
        summary = build_capability_summary(names, tool_profile="full")
        assert "I can" in summary
        assert len(summary) <= 130  # stays within budget

    def test_research_profile_filters(self):
        names = {"web_search", "python_exec", "write_file", "read_file"}
        summary = build_capability_summary(names, tool_profile="research")
        assert "search" in summary.lower()
        assert "read files" in summary
        # python_exec and write_file are not in research profile
        assert "run code" not in summary
        assert "write files" not in summary

    def test_mcp_grouping(self):
        names = {"web_search", "brave__search", "brave__suggest", "notion__query"}
        summary = build_capability_summary(names, tool_profile="full")
        assert "brave" in summary
        assert "notion" in summary

    def test_empty_set(self):
        summary = build_capability_summary(set(), tool_profile="full")
        assert "no tools" in summary.lower()

    def test_max_chars_truncation(self):
        # Many capabilities that would exceed max_chars
        names = {"web_search", "web_fetch", "read_file", "list_dir", "write_file",
                 "python_exec", "describe_image", "create_task", "stock",
                 "run_tests", "lint_code", "verify_syntax",
                 "mcp1__a", "mcp2__b", "mcp3__c"}
        summary = build_capability_summary(names, tool_profile="full", max_chars=60)
        assert "more capabilities" in summary

    def test_mcp_tools_pass_research_profile(self):
        """MCP tools with __ should be included in research profile."""
        names = {"web_search", "brave__search"}
        summary = build_capability_summary(names, tool_profile="research")
        assert "brave" in summary

    def test_minimal_profile_blocks_all(self):
        names = {"web_search", "python_exec", "write_file"}
        summary = build_capability_summary(names, tool_profile="minimal")
        assert "no tools" in summary.lower()


# ── TestInventoryVersionMigration ────────────────────────────────────────────

class TestInventoryVersionMigration:

    def test_v0_to_v2_adds_schema_version_and_param_signature(self):
        v0_data = {
            "tools": {
                "web_search": {
                    "name": "web_search",
                    "description": "Search",
                    "param_hash": "abc123",
                    "risk_level": "low",
                    "source": "builtin",
                }
            }
        }
        migrated = _migrate_inventory(v0_data, 0)
        assert migrated["inventory_version"] == 2
        assert migrated["tools"]["web_search"]["schema_version"] == 1
        assert migrated["tools"]["web_search"]["param_signature"] is None

    def test_v0_to_v2_preserves_existing_schema_version(self):
        v0_data = {
            "tools": {
                "web_search": {
                    "name": "web_search",
                    "description": "Search",
                    "param_hash": "abc123",
                    "risk_level": "low",
                    "source": "builtin",
                    "schema_version": 99,
                }
            }
        }
        migrated = _migrate_inventory(v0_data, 0)
        # setdefault should preserve existing value
        assert migrated["tools"]["web_search"]["schema_version"] == 99
        assert migrated["inventory_version"] == 2

    def test_loading_v0_file(self, tmp_path):
        """V0 file on disk is migrated transparently on load."""
        path = tmp_path / "cap.json"
        v0_data = {
            "tools": {
                "web_search": {
                    "name": "web_search",
                    "description": "Search",
                    "param_hash": "abc123",
                    "risk_level": "low",
                    "source": "builtin",
                }
            }
        }
        path.write_text(json.dumps(v0_data), encoding="utf-8")
        inv = CapabilityInventory(snapshot_path=path)
        assert inv.had_prior_session
        assert "web_search" in inv.tool_names

    def test_current_version_no_migration(self):
        v2_data = {
            "inventory_version": 2,
            "tools": {
                "a": {
                    "name": "a",
                    "description": "Tool A",
                    "param_hash": "xyz",
                    "risk_level": "low",
                    "source": "builtin",
                    "schema_version": 1,
                    "param_signature": None,
                }
            }
        }
        result = _migrate_inventory(v2_data, 2)
        assert result["inventory_version"] == 2

    def test_v1_to_v2_adds_param_signature(self):
        v1_data = {
            "inventory_version": 1,
            "tools": {
                "web_search": {
                    "name": "web_search",
                    "description": "Search",
                    "param_hash": "abc123",
                    "risk_level": "low",
                    "source": "builtin",
                    "schema_version": 1,
                }
            }
        }
        migrated = _migrate_inventory(v1_data, 1)
        assert migrated["inventory_version"] == 2
        assert migrated["tools"]["web_search"]["param_signature"] is None


# ── TestFieldChangeSeverity (E2) ─────────────────────────────────────────────

class TestFieldChangeSeverity:

    def test_default_severity(self):
        fc = FieldChange("t", "desc_changed", "a", "b")
        assert fc.severity == "info"
        assert fc.is_breaking is False

    def test_custom_severity(self):
        fc = FieldChange("t", "param_removed", "x", "", severity="breaking", is_breaking=True)
        assert fc.severity == "breaking"
        assert fc.is_breaking is True

    def test_event_hash_includes_severity(self):
        fc1 = FieldChange("t", "param_added", "", "x", severity="info")
        fc2 = FieldChange("t", "param_added", "", "x", severity="warning")
        d1 = InventoryDiff([], [], [fc1])
        d2 = InventoryDiff([], [], [fc2])
        assert d1.event_hash != d2.event_hash

    def test_activity_lines_severity_tag(self):
        fc_info = FieldChange("tool_a", "desc_changed", "old", "new")
        fc_warn = FieldChange("tool_b", "param_added", "", "x", severity="warning")
        fc_break = FieldChange("tool_c", "param_removed", "y", "", severity="breaking", is_breaking=True)
        diff = InventoryDiff([], [], [fc_info, fc_warn, fc_break])
        lines = diff.to_activity_lines()
        # info has no tag
        assert any("tool_a" in l and "[" not in l.split(")")[-1] for l in lines)
        # warning has [warning] tag
        assert any("[warning]" in l for l in lines)
        # breaking has [breaking] tag
        assert any("[breaking]" in l for l in lines)


# ── TestClassifyParamDiff (E2) ────────────────────────────────────────────────

class TestClassifyParamDiff:

    def test_param_added_optional(self):
        old = {"properties": {"q": "string"}, "required": ["q"]}
        new = {"properties": {"q": "string", "limit": "integer"}, "required": ["q"]}
        changes = _classify_param_diff("tool", old, new)
        assert len(changes) == 1
        assert changes[0].change_type == "param_added"
        assert changes[0].new_value == "limit"
        assert changes[0].severity == "info"
        assert changes[0].is_breaking is False

    def test_param_added_required(self):
        old = {"properties": {"q": "string"}, "required": ["q"]}
        new = {"properties": {"q": "string", "limit": "integer"}, "required": ["limit", "q"]}
        changes = _classify_param_diff("tool", old, new)
        added = [c for c in changes if c.change_type == "param_added"]
        assert len(added) == 1
        assert added[0].severity == "warning"
        assert added[0].is_breaking is True

    def test_param_removed_required(self):
        old = {"properties": {"q": "string", "limit": "integer"}, "required": ["limit", "q"]}
        new = {"properties": {"q": "string"}, "required": ["q"]}
        changes = _classify_param_diff("tool", old, new)
        removed = [c for c in changes if c.change_type == "param_removed"]
        assert len(removed) == 1
        assert removed[0].old_value == "limit"
        assert removed[0].severity == "breaking"
        assert removed[0].is_breaking is True

    def test_param_removed_optional(self):
        old = {"properties": {"q": "string", "limit": "integer"}, "required": ["q"]}
        new = {"properties": {"q": "string"}, "required": ["q"]}
        changes = _classify_param_diff("tool", old, new)
        removed = [c for c in changes if c.change_type == "param_removed"]
        assert len(removed) == 1
        assert removed[0].severity == "warning"
        assert removed[0].is_breaking is False

    def test_param_type_changed(self):
        old = {"properties": {"q": "string"}, "required": ["q"]}
        new = {"properties": {"q": "integer"}, "required": ["q"]}
        changes = _classify_param_diff("tool", old, new)
        assert len(changes) == 1
        assert changes[0].change_type == "param_type_changed"
        assert changes[0].old_value == "q:string"
        assert changes[0].new_value == "q:integer"
        assert changes[0].severity == "warning"

    def test_no_changes(self):
        sig = {"properties": {"q": "string"}, "required": ["q"]}
        changes = _classify_param_diff("tool", sig, sig)
        assert changes == []

    def test_multiple_changes(self):
        old = {"properties": {"a": "string", "b": "integer"}, "required": ["a"]}
        new = {"properties": {"a": "boolean", "c": "string"}, "required": ["a", "c"]}
        changes = _classify_param_diff("tool", old, new)
        types = {c.change_type for c in changes}
        assert "param_type_changed" in types  # a: string→boolean
        assert "param_removed" in types       # b removed
        assert "param_added" in types         # c added


# ── TestParamSignatureRoundtrip (E2) ──────────────────────────────────────────

class TestParamSignatureRoundtrip:

    def test_from_tool_def_creates_signature(self):
        td = _make_tool("test", parameters={
            "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
            "required": ["q"],
        })
        fp = ToolFingerprint.from_tool_def(td)
        assert fp.param_signature is not None
        assert fp.param_signature["properties"] == {"q": "string", "n": "integer"}
        assert fp.param_signature["required"] == ["q"]

    def test_roundtrip_dict(self):
        td = _make_tool("test", parameters={
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        })
        fp = ToolFingerprint.from_tool_def(td)
        d = fp.to_dict()
        fp2 = ToolFingerprint.from_dict(d)
        assert fp2.param_signature == {"properties": {"q": "string"}, "required": ["q"]}

    def test_empty_properties_no_signature(self):
        td = _make_tool("test", parameters={"properties": {}})
        fp = ToolFingerprint.from_tool_def(td)
        assert fp.param_signature is None

    def test_cross_session_semantic_diff(self, tmp_path):
        """Param change across sessions produces semantic FieldChange entries."""
        path = tmp_path / "cap.json"
        p1 = {"properties": {"q": {"type": "string"}}, "required": ["q"]}
        p2 = {"properties": {"q": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["q"]}

        inv1 = CapabilityInventory(snapshot_path=path)
        inv1.refresh(_tools_dict(_make_tool("a", parameters=p1)))

        inv2 = CapabilityInventory(snapshot_path=path)
        diff = inv2.refresh(_tools_dict(_make_tool("a", parameters=p2)))
        types = {c.change_type for c in diff.modified}
        assert "param_changed" in types
        assert "param_added" in types
