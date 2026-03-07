"""Lint tool — runs flake8 on Python files in the cwork sandbox."""

import asyncio
import sys

from . import ToolCapability, tool
from ._path_security import _validate_cwork_path
from .sandbox_runtime import maybe_execute_in_sandbox

_TIMEOUT = 30
_MAX_OUTPUT_CHARS = 2000


def _validate_lint(args: dict) -> tuple[bool, str]:
    path = str(args.get("path", "")).strip()
    if not path:
        return False, "path is required"
    ok, reason, _ = _validate_cwork_path(path)
    if not ok:
        return False, reason
    return True, "ok"


@tool(
    name="lint_code",
    description="Run flake8 on Python files and return issues or OK.",
    risk_level="low",
    capability=ToolCapability(
        data_classification="internal",
        filesystem_access=True,
        max_output_chars=_MAX_OUTPUT_CHARS,
    ),
    validator=_validate_lint,
    parameters={
        "properties": {
            "path": {
                "type": "string",
                "description": "Python file or directory path to lint (must be inside cwork)",
            },
        },
        "required": ["path"],
    },
)
async def lint_code(path: str, **kwargs) -> str:
    """Run flake8 on a file or directory inside cwork."""
    ok, reason, resolved = _validate_cwork_path(path)
    if not ok:
        return f"[Error] {reason}"

    if not resolved.exists():
        return f"[Error] Path does not exist: {resolved.name}"

    delegated = bool(kwargs.get("_delegated", False))
    sandbox_attempt = await maybe_execute_in_sandbox(
        "lint_code",
        {"path": str(resolved)},
        timeout_sec=_TIMEOUT,
        delegated=delegated,
    )
    sandbox_note = ""
    if sandbox_attempt.used:
        return sandbox_attempt.output or "[No output]"
    if sandbox_attempt.error:
        sandbox_note = f"[Sandbox fallback] {sandbox_attempt.error}\n"

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "flake8",
            "--max-line-length=120",
            "--count",
            "--statistics",
            str(resolved),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TIMEOUT)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"[Timeout] flake8 exceeded {_TIMEOUT} seconds"
    except FileNotFoundError:
        return "[Error] flake8 is not installed. Run: pip install flake8"

    output = ""
    if stdout:
        output = stdout.decode("utf-8", errors="replace")
    if stderr:
        err = stderr.decode("utf-8", errors="replace")
        if err.strip():
            output += ("\n[stderr]\n" if output else "[stderr]\n") + err

    if not output.strip():
        return f"OK - {resolved.name} has no flake8 issues"

    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n...(truncated)"
    return f"{sandbox_note}{output}" if sandbox_note else output
