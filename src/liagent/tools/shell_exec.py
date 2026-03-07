"""Shell command execution — argv-only, 3-tier classification, cwork-sandboxed."""
from __future__ import annotations

import asyncio
import shlex

from . import ToolCapability, tool
from ._path_security import get_cwork_root
from .shell_classify import classify_command

_MAX_CMD_CHARS = 2000
_TIMEOUT_SAFE = 30
_TIMEOUT_PRIVILEGED = 60
_MAX_OUTPUT_CHARS = 4000

_ALTERNATIVES: dict[str, str] = {
    "curl": "Use web_fetch tool to retrieve URL content.",
    "wget": "Use web_fetch tool to download files.",
    "env": "Use 'echo $SPECIFIC_VAR' to check a single variable safely.",
    "ssh": "Direct SSH is not supported. Use git operations through shell_exec.",
    "nc": "Network tools are not available. Use web_fetch or web_search.",
}


def _validate_shell_exec(args: dict) -> tuple[bool, str]:
    command = str(args.get("command", "")).strip()
    if not command:
        return False, "command is required"
    if len(command) > _MAX_CMD_CHARS:
        return False, f"command too long (max {_MAX_CMD_CHARS} chars)"
    return True, "ok"


def _build_denial_message(argv: list[str], reason: str) -> str:
    """Three-part denial: reason + alternative + upgrade path."""
    cmd = argv[0] if argv else "unknown"
    cmd_str = " ".join(argv)
    alt = _ALTERNATIVES.get(cmd, "Try an allowed command or request access via /settings.")
    return (
        f"[Denied] Command '{cmd_str}' blocked: {reason}\n"
        f"Alternative: {alt}\n"
        f"Upgrade: Switch to 'full' profile via /settings, then approve dev-tier access when prompted."
    )


@tool(
    name="shell_exec",
    description=(
        "Run a shell command (argv-only, no shell expansion). "
        "Commands are classified into safe (read-only), dev (requires session grant), "
        "and privileged (always requires confirmation). "
        "All paths must be within the cwork workspace. "
        "Unrecognized commands are denied by default."
    ),
    risk_level="medium",
    requires_confirmation=False,
    capability=ToolCapability(
        data_classification="internal",
        filesystem_access=True,
        max_output_chars=_MAX_OUTPUT_CHARS,
        latency_tier="medium",
        idempotent=False,
        failure_modes=("invalid_input", "timeout"),
        min_timeout_sec=_TIMEOUT_PRIVILEGED + 5,
    ),
    validator=_validate_shell_exec,
    parameters={
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Command to run (space-separated, parsed as argv). "
                    "Safe: ls, cat, grep, rg, sed, git status/show, etc. "
                    "Dev: git commit/fetch, uv sync/run, ruff check/format, "
                    "pip/npm/pnpm/yarn/poetry workflows, pytest (needs session grant). "
                    "Denied: curl, env, unrecognized commands."
                ),
            },
        },
        "required": ["command"],
    },
)
async def shell_exec(command: str, **kwargs) -> str:
    """Run a shell command with 3-tier classification."""
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return f"[Denied] Invalid command syntax: {e}"

    if not argv:
        return "[Denied] Empty command"

    tier, reason = classify_command(argv)

    if tier == "denied":
        return _build_denial_message(argv, reason)

    timeout = _TIMEOUT_PRIVILEGED if tier == "privileged" else _TIMEOUT_SAFE
    return await _run_argv(argv, timeout=timeout)


async def _run_argv(argv: list[str], timeout: int) -> str:
    """Run command via subprocess — argv-only, never shell=True."""
    cwork_root = get_cwork_root()
    cwork_root.mkdir(parents=True, exist_ok=True)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwork_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except FileNotFoundError:
        return f"[Error] Command not found: {argv[0]}"
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"[Timeout] Command exceeded {timeout}s"

    output = stdout.decode("utf-8", errors="replace") if stdout else "[No output]"
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n...(truncated)"
    if proc.returncode and proc.returncode != 0:
        output += f"\n[exit code: {proc.returncode}]"
    return output
