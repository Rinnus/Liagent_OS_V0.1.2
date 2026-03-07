"""Sandbox runner executed inside Docker for high-risk tools."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

_BLOCKED_MODULES = frozenset({
    "socket", "http", "urllib", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "poplib", "imaplib", "xmlrpc", "paramiko",
    "subprocess", "shlex", "pty",
    "ctypes", "cffi", "importlib", "imp", "runpy", "webbrowser",
})
_MAX_CODE_CHARS = 5000
_MAX_OUTPUT_CHARS = 3000
_RUNTIME_GUARD = (
    "import sys as _sys\n"
    "class _BlockedModule:\n"
    "    def __getattr__(self, name): raise ImportError('blocked by sandbox')\n"
    "for _m in ['socket','subprocess','shlex','pty','ctypes','cffi','webbrowser']:\n"
    "    _sys.modules[_m] = _BlockedModule()\n"
    "del _BlockedModule, _m\n"
)


def _workspace() -> Path:
    return Path(os.environ.get("LIAGENT_SANDBOX_WORKSPACE", "/workspace")).resolve()


def _path_in_workspace(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (_workspace() / p).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(_workspace())
    except Exception:
        raise ValueError("path must stay inside sandbox workspace")
    return p


def _clip_output(text: str) -> str:
    out = text if text.strip() else "[No output]"
    if len(out) > _MAX_OUTPUT_CHARS:
        out = out[:_MAX_OUTPUT_CHARS] + "\n...(truncated)"
    return out


def _validate_code(code: str):
    if not code.strip():
        raise ValueError("code is required")
    if len(code) > _MAX_CODE_CHARS:
        raise ValueError(f"code too long (max {_MAX_CODE_CHARS} chars)")
    for m in re.finditer(r'(?:^|\s)(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)', code):
        top_module = m.group(1).split(".")[0]
        if top_module in _BLOCKED_MODULES:
            raise ValueError(f"module '{top_module}' is blocked by sandbox policy")


async def _run_python_exec(args: dict, timeout_sec: float) -> str:
    code = str(args.get("code", "") or "")
    _validate_code(code)
    guarded_code = _RUNTIME_GUARD + code
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        guarded_code,
        cwd=str(_workspace()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PATH": os.path.dirname(sys.executable),
        },
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"[Timeout] Code execution exceeded {int(timeout_sec)} seconds"
    out = ""
    if stdout:
        out += stdout.decode("utf-8", errors="replace")
    if stderr:
        out += ("\n[stderr]\n" if out else "[stderr]\n") + stderr.decode("utf-8", errors="replace")
    return _clip_output(out)


async def _run_pytest(args: dict, timeout_sec: float) -> str:
    target = _path_in_workspace(str(args.get("path", "") or "").strip())
    if not target.exists():
        return f"[Error] Path does not exist: {target.name}"
    cmd = [sys.executable, "-m", "pytest", str(target), "--tb=short", "-q"]
    if bool(args.get("verbose", False)):
        cmd.append("-v")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(target.parent if target.is_file() else target),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"[Timeout] pytest exceeded {int(timeout_sec)} seconds"
    out = stdout.decode("utf-8", errors="replace") if stdout else ""
    if stderr:
        err = stderr.decode("utf-8", errors="replace")
        if err.strip():
            out += ("\n[stderr]\n" if out else "[stderr]\n") + err
    return _clip_output(out)


async def _run_flake8(args: dict, timeout_sec: float) -> str:
    target = _path_in_workspace(str(args.get("path", "") or "").strip())
    if not target.exists():
        return f"[Error] Path does not exist: {target.name}"
    cmd = [
        sys.executable,
        "-m",
        "flake8",
        "--max-line-length=120",
        "--count",
        "--statistics",
        str(target),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(target.parent if target.is_file() else target),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"[Timeout] flake8 exceeded {int(timeout_sec)} seconds"
    out = stdout.decode("utf-8", errors="replace") if stdout else ""
    if stderr:
        err = stderr.decode("utf-8", errors="replace")
        if err.strip():
            out += ("\n[stderr]\n" if out else "[stderr]\n") + err
    if not out.strip():
        return f"OK - {target.name} has no flake8 issues"
    return _clip_output(out)


async def _main() -> int:
    parser = argparse.ArgumentParser(description="LiAgent sandbox tool runner")
    parser.add_argument("--tool", required=True)
    parser.add_argument("--args-json", default="{}")
    parser.add_argument("--timeout", type=float, default=30.0)
    ns = parser.parse_args()
    try:
        args = json.loads(ns.args_json or "{}")
        if not isinstance(args, dict):
            args = {}
    except Exception:
        args = {}
    timeout_sec = max(2.0, float(ns.timeout or 30.0))

    try:
        if ns.tool == "python_exec":
            out = await _run_python_exec(args, timeout_sec)
        elif ns.tool == "run_tests":
            out = await _run_pytest(args, timeout_sec)
        elif ns.tool == "lint_code":
            out = await _run_flake8(args, timeout_sec)
        else:
            print(f"[Error] unsupported sandbox tool: {ns.tool}")
            return 2
    except Exception as exc:
        print(f"[Error] sandbox runner failed: {exc}")
        return 1

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

