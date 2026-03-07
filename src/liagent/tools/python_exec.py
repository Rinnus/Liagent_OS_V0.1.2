"""Python code execution tool — sandboxed with static analysis + resource limits."""

import asyncio
import os
import re
import sys

from . import ToolCapability, tool
from .sandbox_runtime import maybe_execute_in_sandbox

_SANDBOX_DIR = os.environ.get(
    "LIAGENT_SANDBOX_DIR",
    os.path.join(os.path.expanduser("~"), "Desktop", "cwork", "_sandbox"),
)
_TIMEOUT = 30
_MAX_CODE_CHARS = 5000
_MAX_OUTPUT_CHARS = 2000

# ── Static blocklists ────────────────────────────────────────────────
# These patterns catch common escape routes BEFORE the code runs.
# Not a replacement for OS-level sandboxing, but raises the bar significantly.

_BLOCKED_PATTERNS = (
    # Block git commit
    re.compile(r"\bgit\b(?:[^A-Za-z0-9]{0,40})\bcommit\b", re.IGNORECASE),
)

# Modules that provide dangerous capabilities.  We block both `import X`
# and `__import__("X")` forms.
_BLOCKED_MODULES = frozenset({
    # Network access
    "socket", "http", "urllib", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "poplib", "imaplib", "xmlrpc", "paramiko",
    # Process / shell execution
    "subprocess", "shlex", "pty", "pdb",
    # System-level introspection / manipulation
    "ctypes", "cffi", "signal", "resource",
    # Code execution / import manipulation
    "importlib", "imp", "runpy", "code", "codeop", "compileall",
    # Sensitive data access
    "webbrowser", "antigravity",
})

# Dangerous builtins / attribute access patterns
_BLOCKED_CALLS = (
    re.compile(r"\bexec\s*\("),           # exec()
    re.compile(r"\beval\s*\("),           # eval()
    re.compile(r"\bcompile\s*\("),        # compile()
    re.compile(r"\b__import__\s*\("),     # __import__()
    re.compile(r"\bgetattr\s*\("),        # getattr() — used to bypass blocklists
    re.compile(r"\bglobals\s*\("),        # globals()
    re.compile(r"\bbreakpoint\s*\("),     # breakpoint()
    re.compile(r"\bopen\s*\([^)]*['\"][/~]"),  # open() with absolute/home paths
)

# Filesystem escape: accessing paths outside sandbox
_PATH_ESCAPE_RE = re.compile(
    r"""(?:open|Path)\s*\(\s*['"](?:/|~|\.\./)"""
    r"""|os\.(?:listdir|scandir|walk|chmod|chown|remove|unlink|rmdir|rename|makedirs|symlink)\s*\("""
    r"""|shutil\.""",
    re.IGNORECASE,
)


def _check_blocked_modules(code: str) -> str | None:
    """Return the first blocked module found, or None."""
    # Match: import X, from X import ..., __import__("X")
    for m in re.finditer(r'(?:^|\s)(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', code):
        top_module = m.group(1).split(".")[0]
        if top_module in _BLOCKED_MODULES:
            return top_module
    for m in re.finditer(r'__import__\s*\(\s*["\']([a-zA-Z_][a-zA-Z0-9_.]*)', code):
        top_module = m.group(1).split(".")[0]
        if top_module in _BLOCKED_MODULES:
            return top_module
    return None


def _validate_python_exec(args: dict) -> tuple[bool, str]:
    code = str(args.get("code", "")).strip()
    if not code:
        return False, "code is required"
    if len(code) > _MAX_CODE_CHARS:
        return False, f"code too long (max {_MAX_CODE_CHARS} chars)"
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(code):
            return False, "git commit is blocked by policy"
    blocked_mod = _check_blocked_modules(code)
    if blocked_mod:
        return False, f"module '{blocked_mod}' is blocked by sandbox policy"
    for pattern in _BLOCKED_CALLS:
        if pattern.search(code):
            return False, f"blocked call pattern detected: {pattern.pattern[:40]}"
    if _PATH_ESCAPE_RE.search(code):
        return False, "filesystem access outside sandbox is blocked"
    return True, "ok"


# ── Sandbox environment ──────────────────────────────────────────────
# Minimal env: strip PATH (no shell commands), HOME, credentials.
_SANDBOX_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
    # Keep a minimal PATH for python itself but nothing else
    "PATH": os.path.dirname(sys.executable),
    # Block network via injected startup code (belt + suspenders)
    "PYTHONSTARTUP": "",
}

# Runtime guard injected before user code — blocks dangerous modules
# at the sys.modules level to catch dynamic imports that bypass static analysis.
_RUNTIME_GUARD = (
    "import sys as _sys\n"
    "class _BlockedModule:\n"
    "    def __getattr__(self, name): raise ImportError('blocked by sandbox')\n"
    "for _m in ['socket','subprocess','shlex','pty','ctypes','cffi','signal','webbrowser']:\n"
    "    _sys.modules[_m] = _BlockedModule()\n"
    "del _BlockedModule, _m\n"
)


@tool(
    name="python_exec",
    description="Execute Python code and return output. Useful for computation and data processing. Network access and filesystem escape are blocked. Timeout: 30s.",
    risk_level="medium",
    requires_confirmation=True,
    capability=ToolCapability(
        data_classification="internal",
        filesystem_access=True,
        max_output_chars=_MAX_OUTPUT_CHARS,
        latency_tier="medium",
        idempotent=False,
        failure_modes=("invalid_input",),
        min_timeout_sec=_TIMEOUT + 5,  # outer timeout must exceed internal 30s
    ),
    validator=_validate_python_exec,
    parameters={
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute (no network access, subprocess, or filesystem escape)",
            },
        },
        "required": ["code"],
    },
)
async def python_exec(code: str, **kwargs) -> str:
    """Execute Python code in a subprocess with sandbox restrictions."""
    from pathlib import Path

    delegated = bool(kwargs.get("_delegated", False))
    sandbox_attempt = await maybe_execute_in_sandbox(
        "python_exec",
        {"code": code},
        timeout_sec=_TIMEOUT,
        delegated=delegated,
    )
    sandbox_note = ""
    if sandbox_attempt.used:
        return sandbox_attempt.output or "[No output]"
    if sandbox_attempt.error:
        sandbox_note = f"[Sandbox fallback] {sandbox_attempt.error}\n"

    sandbox = Path(_SANDBOX_DIR)
    sandbox.mkdir(parents=True, exist_ok=True)

    guarded_code = _RUNTIME_GUARD + code

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            guarded_code,
            cwd=str(sandbox),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_SANDBOX_ENV,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_TIMEOUT
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return "[Timeout] Code execution exceeded 30 seconds"

    output = ""
    if stdout:
        output += stdout.decode("utf-8", errors="replace")
    if stderr:
        output += ("\n[stderr]\n" if output else "[stderr]\n") + stderr.decode(
            "utf-8", errors="replace"
        )
    if not output.strip():
        output = "[No output]"
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n...(truncated)"
    return f"{sandbox_note}{output}" if sandbox_note else output
