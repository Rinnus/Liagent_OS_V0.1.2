"""Tool policy gateway: validation, risk gating, and audit logging."""

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from . import ToolDef
from ..config import db_path as _db_path

# Pre-compiled secret-redaction patterns (used on every tool output)
_SK_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
_API_KEY_RE = re.compile(r"(?i)\b(api[_-]?key)\s*[:=]\s*[\"']?([A-Za-z0-9_\-]{8,})")
_BEARER_RE = re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9\-_\.]{12,})")

DB_PATH = _db_path()
_TOOL_PROFILE_MAP: dict[str, set[str] | None] = {
    # Security-first: no external or sensitive tool execution.
    "minimal": set(),
    # Read-mostly research tools.
    "research": {"web_search", "web_fetch", "read_file", "list_dir", "describe_image",
                  "create_task", "delete_task", "delete_all_tasks", "list_tasks",
                  "run_tests", "lint_code", "verify_syntax", "system_status",
                  "shell_exec",
                  "browser_navigate", "browser_screenshot", "browser_extract"},
    # All registered tools are allowed (still subject to risk/confirmation rules).
    "full": None,
}


def _run_tests_requires_host_confirmation(args: dict) -> bool:
    """Return True when run_tests would execute directly on the host."""
    try:
        from .sandbox_runtime import routes_tool_in_sandbox

        delegated = bool((args or {}).get("_delegated", False))
        return not routes_tool_in_sandbox("run_tests", delegated=delegated)
    except Exception:
        # Fail closed: if sandbox routing cannot be determined, require confirmation.
        return True


def _python_exec_requires_host_confirmation(args: dict) -> bool:
    """Return True when python_exec would execute directly on the host."""
    try:
        from .sandbox_runtime import routes_tool_in_sandbox

        delegated = bool((args or {}).get("_delegated", False))
        return not routes_tool_in_sandbox("python_exec", delegated=delegated)
    except Exception:
        return True


def _stateful_repl_requires_trusted_confirmation() -> bool:
    """Return True when stateful_repl is running in trusted_local mode."""
    try:
        from .stateful_repl import get_repl_mode

        return get_repl_mode() == "trusted_local"
    except Exception:
        return True


def tool_supports_session_grant(tool_name: str, args: dict | None = None) -> bool:
    """Return whether *tool_name* should receive a reusable session grant."""
    if tool_name == "write_file":
        return False
    if tool_name == "run_tests":
        return not _run_tests_requires_host_confirmation(args or {})
    if tool_name == "python_exec":
        return not _python_exec_requires_host_confirmation(args or {})
    if tool_name == "stateful_repl":
        return not _stateful_repl_requires_trusted_confirmation()
    return True


def should_create_session_grant(
    tool_name: str,
    *,
    auth_mode: str = "",
    args: dict | None = None,
    tool_def: ToolDef | None = None,
    grantable_tools: set[str] | None = None,
) -> bool:
    """Return whether a confirmed execution should mint a reusable session grant."""
    if auth_mode != "confirmed":
        return False
    if tool_def is not None and tool_def.risk_level == "high":
        return False
    if not tool_supports_session_grant(tool_name, args):
        return False
    if grantable_tools is not None and tool_name not in grantable_tools:
        return False
    return True


class ToolPolicy:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        *,
        tool_profile: Literal["minimal", "research", "full"] | str | None = None,
        trust_registry: "TrustRegistry | None" = None,
    ):
        self.db_path = db_path
        profile_raw = (
            tool_profile
            if tool_profile is not None
            else os.environ.get("LIAGENT_TOOL_PROFILE", "research")
        )
        self.tool_profile = str(profile_raw or "research").strip().lower()
        if self.tool_profile not in _TOOL_PROFILE_MAP:
            self.tool_profile = "research"
        self.profile_allowset = _TOOL_PROFILE_MAP[self.tool_profile]
        self.allow_high_risk = (
            os.environ.get("LIAGENT_ALLOW_HIGH_RISK_TOOLS", "").strip().lower()
            in {"1", "true", "yes"}
        )
        confirm_levels = os.environ.get(
            "LIAGENT_CONFIRM_RISK_LEVELS", "high"
        ).strip()
        self.confirm_risk_levels = {
            x.strip().lower() for x in confirm_levels.split(",") if x.strip()
        }
        allowlist = os.environ.get("LIAGENT_ALLOWED_TOOLS", "").strip()
        self.allowlist = {x.strip() for x in allowlist.split(",") if x.strip()} if allowlist else set()
        confirm_tools = os.environ.get("LIAGENT_CONFIRM_TOOLS", "").strip()
        self.confirm_tools = (
            {x.strip() for x in confirm_tools.split(",") if x.strip()}
            if confirm_tools
            else set()
        )
        self.allow_network_tools = (
            os.environ.get("LIAGENT_ALLOW_NETWORK_TOOLS", "true").strip().lower()
            in {"1", "true", "yes"}
        )
        self.allow_filesystem_tools = (
            os.environ.get("LIAGENT_ALLOW_FILESYSTEM_TOOLS", "true").strip().lower()
            in {"1", "true", "yes"}
        )
        self.default_max_output_chars = max(
            200, int(os.environ.get("LIAGENT_TOOL_OUTPUT_MAX_CHARS", "2000"))
        )
        self._audit_retention_days = max(
            7, int(os.environ.get("LIAGENT_AUDIT_RETENTION_DAYS", "30"))
        )
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        self._prune_audit()
        self.trust_registry = trust_registry

    def _init_db(self):
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS tool_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                args_json TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        # A2 idempotent migration: add columns if missing
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(tool_audit)").fetchall()
        }
        for col in (
            "requested_tool", "requested_args",
            "effective_tool", "effective_args",
            "policy_decision", "grant_source",
        ):
            if col not in existing:
                self._conn.execute(f"ALTER TABLE tool_audit ADD COLUMN {col} TEXT DEFAULT ''")
        self._conn.commit()

    def _prune_audit(self):
        """Delete audit rows older than retention period."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._audit_retention_days)).isoformat()
        try:
            r = self._conn.execute("DELETE FROM tool_audit WHERE created_at < ?", (cutoff,))
            if (r.rowcount or 0) > 0:
                self._conn.execute("PRAGMA incremental_vacuum")
            self._conn.commit()
        except Exception:
            pass

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def evaluate(
        self,
        tool_def: ToolDef,
        args: dict,
        *,
        confirmed: bool = False,
        granted: bool = False,
    ) -> tuple[bool, str]:
        """Evaluate tool access.

        Authorization hierarchy:
        - ``confirmed``: one-time human confirmation — bypasses trust + ALL risk gates.
        - ``granted``: session-level authorization — bypasses risk/presence/confirm_tools
          but NOT trust gate and NOT high-risk tools.
        - ``allow_high_risk``: env-var policy override for dev/testing — disables
          high-risk checks entirely (orthogonal to grants).
        """
        auth = confirmed or granted  # combined signal for non-trust risk gates

        # ── Trust gate (FIRST — hard security boundary) ──────────────
        # Trust requires confirmed (human confirmation). Grants do NOT bypass trust.
        if self.trust_registry is not None and "__" in tool_def.name:
            server_id = tool_def.name.split("__", 1)[0]
            trust_status = self.trust_registry.get_status(server_id)
            if trust_status == "revoked":
                return False, f"tool blocked: server '{server_id}' is revoked"
            if trust_status == "unknown" and not confirmed:
                return False, f"confirmation required for trust=unknown server={server_id}"

        if self.profile_allowset is not None and tool_def.name not in self.profile_allowset:
            # MCP tools (namespace__tool) pass research profile if risk is low/medium
            is_mcp = "__" in tool_def.name
            if not (is_mcp and tool_def.risk_level in ("low", "medium")):
                return False, f"tool blocked by profile={self.tool_profile}"

        if self.allowlist and tool_def.name not in self.allowlist:
            return False, "tool is not in allowlist"

        if tool_def.validator:
            ok, reason = tool_def.validator(args)
            if not ok:
                return False, reason

        risk = tool_def.risk_level.lower()
        cap = tool_def.capability

        if cap.network_access and not self.allow_network_tools:
            return False, "network tools are blocked by policy"

        if cap.filesystem_access and not self.allow_filesystem_tools:
            return False, "filesystem tools are blocked by policy"

        if tool_def.name == "run_tests" and _run_tests_requires_host_confirmation(args) and not confirmed:
            return False, "confirmation required for host_execution=run_tests"

        if tool_def.name == "python_exec" and _python_exec_requires_host_confirmation(args) and not confirmed:
            return False, "confirmation required for host_execution=python_exec"

        if tool_def.name == "stateful_repl" and _stateful_repl_requires_trusted_confirmation() and not confirmed:
            return False, "confirmation required for repl_mode=trusted_local"

        # Risk/presence/confirm_tools gates: confirmed OR granted bypasses
        if cap.requires_user_presence and not auth:
            return False, f"confirmation required for tool={tool_def.name}"

        if tool_def.name in self.confirm_tools and not auth:
            return False, f"confirmation required for tool={tool_def.name}"

        # High-risk: allow_high_risk (dev override) bypasses ALL high-risk gates.
        # Without it, high-risk requires confirmed (human) — grants are insufficient.
        if risk == "high":
            if not (self.allow_high_risk or confirmed):
                return False, "confirmation required for risk=high"
            # allow_high_risk or confirmed → skip risk confirmation for high-risk
        elif (tool_def.requires_confirmation or risk in self.confirm_risk_levels) and not auth:
            return False, f"confirmation required for risk={risk}"

        return True, "allowed"

    def sanitize_output(self, tool_def: ToolDef, output: str) -> str:
        text = str(output or "")
        text = self._redact_secrets(text)
        max_chars = min(
            max(100, int(getattr(tool_def.capability, "max_output_chars", 1200))),
            self.default_max_output_chars,
        )
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(policy truncated)"
        return text

    @staticmethod
    def _redact_secrets(text: str) -> str:
        out = text
        out = _SK_KEY_RE.sub("[REDACTED_KEY]", out)
        out = _API_KEY_RE.sub(r"\1=[REDACTED]", out)
        out = _BEARER_RE.sub(r"\1 [REDACTED]", out)
        return out

    def capability_summary(self, tool_def: ToolDef) -> str:
        cap = tool_def.capability
        parts = [
            f"classification={cap.data_classification}",
            f"network={cap.network_access}",
            f"filesystem={cap.filesystem_access}",
            f"user_presence={cap.requires_user_presence}",
            f"max_output_chars={cap.max_output_chars}",
        ]
        if cap.cost_tier != "free":
            parts.append(f"cost={cap.cost_tier}")
        if cap.latency_tier != "fast":
            parts.append(f"latency={cap.latency_tier}")
        if not cap.idempotent:
            parts.append("non-idempotent")
        if cap.failure_modes:
            parts.append(f"failure_modes={','.join(cap.failure_modes)}")
        return ", ".join(parts)

    def confirmation_brief(self, tool_def: ToolDef, args: dict, reason: str, *, stage: int, required_stage: int) -> dict:
        cap = tool_def.capability
        return {
            "tool": tool_def.name,
            "risk_level": tool_def.risk_level,
            "reason": reason,
            "stage": stage,
            "required_stage": required_stage,
            "capability": self.capability_summary(tool_def),
            "args_preview": self._redact_arg_values(args),
            "message": (
                "High-risk/sensitive operation requires a two-step confirmation."
                if required_stage > 1
                else "This tool call requires confirmation before execution."
            ),
        }

    def recent_audit(self, limit: int = 50) -> list[dict]:
        n = max(1, min(int(limit), 200))
        rows = self._conn.execute(
            """SELECT tool_name, args_json, status, reason, created_at,
                      requested_tool, requested_args, effective_tool, effective_args,
                      policy_decision, grant_source
               FROM tool_audit
               ORDER BY id DESC
               LIMIT ?""",
            (n,),
        ).fetchall()
        out = []
        for (tool_name, args_json, status, reason, created_at,
             requested_tool, requested_args, effective_tool, effective_args,
             policy_decision, grant_source) in rows:
            try:
                args = json.loads(args_json or "{}")
            except Exception:
                args = {}
            entry: dict = {
                "tool_name": tool_name,
                "args": self._redact_arg_values(args) if isinstance(args, dict) else args,
                "status": status,
                "reason": reason,
                "created_at": created_at,
            }
            if requested_tool:
                entry["requested_tool"] = requested_tool
            if requested_args:
                entry["requested_args"] = requested_args
            if effective_tool:
                entry["effective_tool"] = effective_tool
            if effective_args:
                entry["effective_args"] = effective_args
            if policy_decision:
                entry["policy_decision"] = policy_decision
            if grant_source:
                entry["grant_source"] = grant_source
            out.append(entry)
        return out

    @staticmethod
    def _redact_arg_values(args: dict) -> dict:
        redacted = {}
        for k, v in (args or {}).items():
            key = str(k).lower()
            if any(x in key for x in ("key", "token", "secret", "password")):
                redacted[k] = "[REDACTED]"
            else:
                text = str(v)
                redacted[k] = text if len(text) <= 120 else (text[:120] + "...(truncated)")
        return redacted

    def _redact_args_json(self, args) -> str:
        """Redact and serialize args for audit storage."""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return args
        if isinstance(args, dict):
            args = self._redact_arg_values(args)
        return json.dumps(args, ensure_ascii=False, sort_keys=True) if isinstance(args, dict) else str(args)

    def audit(
        self,
        tool_name: str,
        args: dict,
        status: str,
        reason: str,
        *,
        requested_tool: str = "",
        requested_args: str = "",
        effective_tool: str = "",
        effective_args: str = "",
        policy_decision: str = "",
        grant_source: str = "",
    ):
        now = datetime.now(timezone.utc).isoformat()
        args_json = self._redact_args_json(args)
        if requested_args:
            requested_args = self._redact_args_json(requested_args)
        if effective_args:
            effective_args = self._redact_args_json(effective_args)
        self._conn.execute(
            """INSERT INTO tool_audit
               (tool_name, args_json, status, reason, created_at,
                requested_tool, requested_args, effective_tool, effective_args,
                policy_decision, grant_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tool_name, args_json, status, reason, now,
             requested_tool, requested_args, effective_tool, effective_args,
             policy_decision, grant_source),
        )
        self._conn.commit()
