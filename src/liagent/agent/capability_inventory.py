"""Capability inventory: fingerprint snapshots, structured diffs, and dynamic summaries.

Part of Iteration E1 — the perception layer for tool intelligence.
"""

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..tools import ToolDef, get_all_tools

# ── Inventory file format version ────────────────────────────────────────────
_INVENTORY_VERSION = 2
_DEDUP_WINDOW_SEC = 300  # 5 minutes


# ── Canonical param hash ─────────────────────────────────────────────────────

def _canonical_param_hash(params: dict) -> str:
    """Compute a stable hash of tool parameters (JSON Schema).

    Ignores only ``$schema`` (pure metadata). Preserves all semantic fields
    including ``additionalProperties`` since loosening/tightening input
    constraints is a real capability change. Recursively sorts dict keys.
    """

    def _sort_deep(obj):
        if isinstance(obj, dict):
            return {k: _sort_deep(v) for k, v in sorted(obj.items()) if k != "$schema"}
        if isinstance(obj, list):
            return [_sort_deep(i) for i in obj]
        return obj

    canonical = json.dumps(_sort_deep(params), ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolFingerprint:
    name: str
    description: str
    param_hash: str          # canonical JSON → MD5[:12]
    risk_level: str
    source: str              # "builtin" | "mcp:<server>"
    schema_version: int = 1  # fingerprint format version
    param_signature: dict | None = None  # E2: lightweight param sig for cross-session semantic diff

    @classmethod
    def from_tool_def(cls, td: ToolDef) -> "ToolFingerprint":
        if "__" in td.name:
            server = td.name.split("__")[0]
            source = f"mcp:{server}"
        else:
            source = "builtin"
        # E2: lightweight param signature for cross-session semantic diff
        props = td.parameters.get("properties", {})
        sig = {
            "properties": {k: v.get("type", "any") for k, v in props.items()},
            "required": sorted(td.parameters.get("required", [])),
        } if props else None
        return cls(
            name=td.name,
            description=td.description,
            param_hash=_canonical_param_hash(td.parameters),
            risk_level=td.risk_level,
            source=source,
            param_signature=sig,
        )

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "description": self.description,
            "param_hash": self.param_hash,
            "risk_level": self.risk_level,
            "source": self.source,
            "schema_version": self.schema_version,
        }
        if self.param_signature is not None:
            d["param_signature"] = self.param_signature
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ToolFingerprint":
        return cls(
            name=d["name"],
            description=d["description"],
            param_hash=d["param_hash"],
            risk_level=d["risk_level"],
            source=d["source"],
            schema_version=d.get("schema_version", 1),
            param_signature=d.get("param_signature"),
        )


@dataclass(frozen=True)
class FieldChange:
    tool_name: str
    change_type: str   # "desc_changed" | "param_changed" | "risk_changed" | "source_changed"
                       # E2: + "param_added" | "param_removed" | "param_type_changed"
    old_value: str
    new_value: str
    severity: str = "info"       # "info" | "warning" | "breaking"
    is_breaking: bool = False


@dataclass
class InventoryDiff:
    added: list[str]
    removed: list[str]
    modified: list[FieldChange]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.modified)

    def to_activity_lines(self) -> list[str]:
        """Human-readable lines for logging. Truncates beyond 5 items per category."""
        lines: list[str] = []
        for name in self.added[:5]:
            lines.append(f"[capability] + {name}")
        if len(self.added) > 5:
            lines.append(f"[capability] ... and {len(self.added) - 5} more added")
        for name in self.removed[:5]:
            lines.append(f"[capability] - {name}")
        if len(self.removed) > 5:
            lines.append(f"[capability] ... and {len(self.removed) - 5} more removed")
        for fc in self.modified[:5]:
            sev_tag = f" [{fc.severity}]" if fc.severity != "info" else ""
            lines.append(f"[capability] ~ {fc.tool_name} ({fc.change_type}){sev_tag}")
        if len(self.modified) > 5:
            lines.append(f"[capability] ... and {len(self.modified) - 5} more modified")
        return lines

    @property
    def event_hash(self) -> str:
        """Deterministic hash of the change set, used for dedup."""
        parts: list[str] = []
        for n in sorted(self.added):
            parts.append(f"+{n}")
        for n in sorted(self.removed):
            parts.append(f"-{n}")
        for fc in sorted(self.modified, key=lambda c: (c.tool_name, c.change_type)):
            parts.append(f"~{fc.tool_name}:{fc.change_type}:{fc.old_value}:{fc.new_value}:{fc.severity}:{fc.is_breaking}")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


_EMPTY_DIFF = InventoryDiff([], [], [])


# ── Snapshot version migration ───────────────────────────────────────────────

def _migrate_inventory(data: dict, from_version: int) -> dict:
    """Forward-migrate inventory file format."""
    if from_version < 1:
        # v0→v1: add schema_version to each fingerprint
        for fp in data.get("tools", {}).values():
            fp.setdefault("schema_version", 1)
    if from_version < 2:
        # v1→v2: add param_signature (None = unknown from prior session)
        for fp in data.get("tools", {}).values():
            fp.setdefault("param_signature", None)
    data["inventory_version"] = _INVENTORY_VERSION
    return data


# ── Semantic param diff (E2) ─────────────────────────────────────────────────

def _classify_param_diff(
    tool_name: str, old_sig: dict, new_sig: dict
) -> list[FieldChange]:
    """Produce semantic FieldChange entries from two param_signature dicts."""
    changes: list[FieldChange] = []
    old_props = old_sig.get("properties", {})
    new_props = new_sig.get("properties", {})
    old_required = set(old_sig.get("required", []))
    new_required = set(new_sig.get("required", []))

    # Added params
    for k in sorted(set(new_props) - set(old_props)):
        is_req = k in new_required
        changes.append(FieldChange(
            tool_name, "param_added", "", k,
            severity="warning" if is_req else "info",
            is_breaking=is_req,
        ))

    # Removed params
    for k in sorted(set(old_props) - set(new_props)):
        was_req = k in old_required
        changes.append(FieldChange(
            tool_name, "param_removed", k, "",
            severity="breaking" if was_req else "warning",
            is_breaking=was_req,
        ))

    # Type changes
    for k in sorted(set(old_props) & set(new_props)):
        old_type = old_props[k]
        new_type = new_props[k]
        if old_type != new_type:
            changes.append(FieldChange(
                tool_name, "param_type_changed", f"{k}:{old_type}", f"{k}:{new_type}",
                severity="warning",
            ))

    return changes


# ── Main inventory class ─────────────────────────────────────────────────────

class CapabilityInventory:
    """Track tool fingerprints across sessions with structured diffs."""

    def __init__(self, snapshot_path: str | Path | None = None):
        if snapshot_path is None:
            snapshot_path = Path.home() / ".liagent" / "capability_inventory.json"
        self._path = Path(snapshot_path)
        self._lock = threading.RLock()
        self._current: dict[str, ToolFingerprint] = {}
        self._had_prior_session = False
        self._last_event_hash: str = ""
        self._last_event_ts: float = 0.0
        # Load persisted snapshot
        self._load_persisted()

    def _load_persisted(self):
        """Load snapshot from disk. Tolerates missing/corrupt files."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            file_version = raw.get("inventory_version", 0)
            if file_version < _INVENTORY_VERSION:
                raw = _migrate_inventory(raw, file_version)
            tools_data = raw.get("tools", {})
            for name, fp_dict in tools_data.items():
                self._current[name] = ToolFingerprint.from_dict(fp_dict)
            if tools_data:
                self._had_prior_session = True
        except Exception:
            # Corrupt file — start fresh
            self._current = {}

    def _save_persisted(self):
        """Atomic write: tempfile + os.replace."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "inventory_version": _INVENTORY_VERSION,
            "tools": {name: fp.to_dict() for name, fp in self._current.items()},
        }
        try:
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp", prefix=".cap_inv_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, str(self._path))
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            pass  # Best-effort persistence

    def refresh(self, tools: dict[str, ToolDef] | None = None) -> InventoryDiff:
        """Snapshot current tools, diff against previous, persist, and return changes.

        Thread-safe via RLock. Deduplicates identical diffs within 5 minutes.
        """
        with self._lock:
            if tools is None:
                tools = get_all_tools()

            # Build new snapshot
            new_snapshot: dict[str, ToolFingerprint] = {}
            for name, td in tools.items():
                new_snapshot[name] = ToolFingerprint.from_tool_def(td)

            old_names = set(self._current)
            new_names = set(new_snapshot)

            added = sorted(new_names - old_names)
            removed = sorted(old_names - new_names)

            # Detect field-level changes for common tools
            changes: list[FieldChange] = []
            for name in sorted(old_names & new_names):
                old = self._current[name]
                new = new_snapshot[name]
                if old.description != new.description:
                    changes.append(FieldChange(
                        name, "desc_changed",
                        old.description[:60], new.description[:60],
                    ))
                if old.param_hash != new.param_hash:
                    changes.append(FieldChange(
                        name, "param_changed",
                        old.param_hash, new.param_hash,
                    ))
                    # E2: semantic param diff from param_signature
                    if old.param_signature and new.param_signature:
                        semantic = _classify_param_diff(name, old.param_signature, new.param_signature)
                        changes.extend(semantic)
                if old.risk_level != new.risk_level:
                    changes.append(FieldChange(
                        name, "risk_changed",
                        old.risk_level, new.risk_level,
                    ))
                if old.source != new.source:
                    changes.append(FieldChange(
                        name, "source_changed",
                        old.source, new.source,
                    ))

            diff = InventoryDiff(added=added, removed=removed, modified=changes)

            # Event dedup: same change within 5 minutes → suppress
            if diff.has_changes:
                h = diff.event_hash
                now = time.time()
                if h == self._last_event_hash and (now - self._last_event_ts) < _DEDUP_WINDOW_SEC:
                    # Same change recently reported — update snapshot but suppress notification
                    self._current = new_snapshot
                    self._save_persisted()
                    return InventoryDiff([], [], [])
                self._last_event_hash = h
                self._last_event_ts = now

            # Update state and persist
            self._current = new_snapshot
            self._save_persisted()
            return diff

    @property
    def had_prior_session(self) -> bool:
        """True if a prior snapshot existed on disk (not first install)."""
        return self._had_prior_session

    @property
    def tool_names(self) -> set[str]:
        return set(self._current)

    @property
    def tool_count(self) -> int:
        return len(self._current)


# ── Dynamic capability summary ───────────────────────────────────────────────

_CAPABILITY_GROUPS: list[tuple[set[str], str]] = [
    ({"web_search"},                                    "search real-time info"),
    ({"web_fetch"},                                     "fetch web pages"),
    ({"read_file", "list_dir"},                         "read files"),
    ({"write_file"},                                    "write files"),
    ({"python_exec"},                                   "run code"),
    ({"describe_image"},                                "analyze images"),
    ({"create_task", "delete_task", "list_tasks"},      "schedule tasks"),
    ({"run_tests", "lint_code", "verify_syntax"},       "test/lint code"),
    ({"stock"},                                         "check stock prices"),
]

# Import lazily to avoid circular import at module level
_TOOL_PROFILE_MAP_CACHE: dict | None = None


def _get_profile_map() -> dict:
    global _TOOL_PROFILE_MAP_CACHE
    if _TOOL_PROFILE_MAP_CACHE is None:
        from ..tools.policy import _TOOL_PROFILE_MAP
        _TOOL_PROFILE_MAP_CACHE = _TOOL_PROFILE_MAP
    return _TOOL_PROFILE_MAP_CACHE


def _filter_by_profile(tool_names: set[str], tool_profile: str) -> set[str]:
    """Filter tool names to match what the profile actually allows."""
    profile_map = _get_profile_map()
    allowset = profile_map.get(tool_profile)
    if allowset is None:  # "full" — all allowed
        return tool_names
    result = set()
    for name in tool_names:
        if name in allowset:
            result.add(name)
        elif "__" in name:
            # MCP tools pass research profile if risk is low/medium
            # (consistent with policy.py evaluate() logic)
            result.add(name)
    return result


def build_capability_summary(
    tool_names: set[str],
    *,
    tool_profile: str = "full",
    max_chars: int = 120,
) -> str:
    """Build a concise capability summary from registered tool names.

    Groups tools by category. MCP tools grouped by server. Truncates
    to stay within token budget (max_chars).
    """
    filtered = _filter_by_profile(tool_names, tool_profile)
    capabilities: list[str] = []
    for pattern, label in _CAPABILITY_GROUPS:
        if pattern & filtered:
            capabilities.append(label)

    # MCP tools: group by server name
    mcp_servers = {n.split("__")[0] for n in filtered if "__" in n}
    if mcp_servers:
        capabilities.append(f"use {', '.join(sorted(mcp_servers))} services")

    if not capabilities:
        return "I have no tools available at the moment."

    result = "I can " + ", ".join(capabilities) + "."
    if len(result) > max_chars and len(capabilities) > 3:
        result = "I can " + ", ".join(capabilities[:3]) + f", and {len(capabilities) - 3} more capabilities."
    return result
