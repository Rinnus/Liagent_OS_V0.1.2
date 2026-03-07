"""Test runner tool — runs pytest in the cwork sandbox."""

import asyncio
import sys

from . import ToolCapability, tool
from ._path_security import _validate_cwork_path
from .sandbox_runtime import maybe_execute_in_sandbox

_TIMEOUT = 120
_MAX_OUTPUT_CHARS = 3000


def _validate_run_tests(args: dict) -> tuple[bool, str]:
    path = str(args.get("path", "")).strip()
    if not path:
        return False, "path is required"
    ok, reason, _ = _validate_cwork_path(path)
    if not ok:
        return False, reason
    return True, "ok"


@tool(
    name="run_tests",
    description="Run pytest in the cwork sandbox and return a summary. Timeout: 120 seconds.",
    risk_level="low",
    requires_confirmation=False,
    capability=ToolCapability(
        data_classification="internal",
        filesystem_access=True,
        max_output_chars=_MAX_OUTPUT_CHARS,
        latency_tier="slow",
        failure_modes=("invalid_input",),
        min_timeout_sec=_TIMEOUT + 5,  # outer timeout must exceed internal 120s
    ),
    validator=_validate_run_tests,
    parameters={
        "properties": {
            "path": {
                "type": "string",
                "description": "Test file or directory path (must be inside cwork)",
            },
            "verbose": {
                "type": "boolean",
                "description": "Show verbose output (default false)",
            },
        },
        "required": ["path"],
    },
)
async def run_tests(path: str, verbose: bool = False, **kwargs) -> str:
    """Run pytest on a file or directory inside cwork."""
    ok, reason, resolved = _validate_cwork_path(path)
    if not ok:
        return f"[Error] {reason}"

    if not resolved.exists():
        return f"[Error] Path does not exist: {resolved.name}"

    delegated = bool(kwargs.get("_delegated", False))
    sandbox_attempt = await maybe_execute_in_sandbox(
        "run_tests",
        {"path": str(resolved), "verbose": bool(verbose)},
        timeout_sec=_TIMEOUT,
        delegated=delegated,
    )
    sandbox_note = ""
    if sandbox_attempt.used:
        return sandbox_attempt.output or "[No output]"
    if sandbox_attempt.error:
        sandbox_note = f"[Sandbox fallback] {sandbox_attempt.error}\n"

    cmd = [sys.executable, "-m", "pytest", str(resolved), "--tb=short", "-q"]
    if verbose:
        cmd.append("-v")

    # Using create_subprocess_exec (argument list, no shell) — safe from injection
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(resolved.parent if resolved.is_file() else resolved),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"[Timeout] pytest exceeded {_TIMEOUT} seconds"
    except FileNotFoundError:
        return "[Error] pytest is not installed. Run: pip install pytest"

    output = ""
    if stdout:
        output = stdout.decode("utf-8", errors="replace")
    if stderr:
        err = stderr.decode("utf-8", errors="replace")
        if err.strip():
            output += ("\n[stderr]\n" if output else "[stderr]\n") + err

    if not output.strip():
        output = "[No output]"

    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n...(truncated)"
    return f"{sandbox_note}{output}" if sandbox_note else output
