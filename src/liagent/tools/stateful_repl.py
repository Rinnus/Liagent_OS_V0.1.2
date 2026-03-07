"""Stateful REPL — session-scoped persistent Python subprocess."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Literal

from . import ToolCapability, tool
from ._path_security import get_cwork_root
from ..logging import get_logger

_log = get_logger("stateful_repl")

_TIMEOUT = 30
_IDLE_TTL = 1800  # 30 minutes
_MAX_OUTPUT_CHARS = 10 * 1024
_VALID_REPL_MODES = {"off", "sandboxed", "trusted_local"}


def _normalize_repl_mode(mode: str | None) -> Literal["off", "sandboxed", "trusted_local"]:
    m = str(mode or "sandboxed").strip().lower()
    if m not in _VALID_REPL_MODES:
        m = "sandboxed"
    return m  # type: ignore[return-value]


class ReplSession:
    """Manages a single persistent REPL subprocess."""

    def __init__(self, session_id: str, mode: str = "sandboxed"):
        self.session_id = session_id
        self.mode = _normalize_repl_mode(mode)
        self._proc: asyncio.subprocess.Process | None = None
        self._async_reader: asyncio.StreamReader | None = None
        self._read_transport = None
        self._lock = asyncio.Lock()
        self._last_used = time.time()

    async def _spawn(self):
        r_fd, w_fd = os.pipe()
        # Ensure PYTHONPATH uses absolute paths so the subprocess
        # can find liagent even when cwd is set to cwork.
        parent_paths = [os.path.abspath(p) for p in sys.path if p]
        cwork_root = get_cwork_root()
        env = {
            **os.environ,
            "LIAGENT_REPL_FD": str(w_fd),
            "LIAGENT_CWORK_ROOT": str(cwork_root),
            "LIAGENT_CWORK_DIR": str(cwork_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": os.pathsep.join(parent_paths),
            "LIAGENT_REPL_MODE": self.mode,
        }
        cwd = cwork_root
        cwd.mkdir(parents=True, exist_ok=True)
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "liagent.tools.repl_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(cwd),
            pass_fds=(w_fd,),
        )
        os.close(w_fd)

        # Wrap the read fd as an async StreamReader
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        pipe_file = os.fdopen(r_fd, "rb", 0)  # raw binary, unbuffered
        transport, _ = await loop.connect_read_pipe(lambda: protocol, pipe_file)
        self._async_reader = reader
        self._read_transport = transport
        _log.event("repl_spawn", session=self.session_id, pid=self._proc.pid)

    def _is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _hard_reset(self, reason: str):
        if self._read_transport:
            try:
                self._read_transport.close()
            except Exception:
                pass
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.kill()
                await self._proc.wait()
            except Exception:
                pass
        self._proc = None
        self._async_reader = None
        self._read_transport = None
        _log.event("repl_hard_reset", session=self.session_id, reason=reason)

    async def _respawn(self, reason: str):
        await self._hard_reset(reason=reason)
        await self._spawn()

    async def run(self, code: str, reset: bool = False) -> dict:
        """Send code to subprocess, return result dict.

        On crash/pipe error: respawn and retry once. On timeout: respawn, no retry.
        """
        if not self._is_alive():
            await self._respawn(reason="process_dead")

        async with self._lock:
            self._last_used = time.time()
            for attempt in range(2):
                try:
                    cmd = json.dumps({"code": code, "reset": reset}) + "\n"
                    self._proc.stdin.write(cmd.encode())
                    await self._proc.stdin.drain()

                    line = await asyncio.wait_for(
                        self._async_reader.readline(),
                        timeout=_TIMEOUT,
                    )
                    if not line:
                        raise BrokenPipeError("Worker closed pipe")
                    return json.loads(line.decode())
                except asyncio.TimeoutError:
                    await self._respawn(reason="timeout")
                    return {"stdout": "", "error": f"Timed out ({_TIMEOUT}s). REPL state reset.", "vars": []}
                except (BrokenPipeError, OSError):
                    if attempt == 0:
                        await self._respawn(reason="crash_retry")
                        continue
                    return {"stdout": "", "error": "REPL process crashed. State reset.", "vars": []}

    @property
    def is_idle_expired(self) -> bool:
        return time.time() - self._last_used > _IDLE_TTL

    async def close(self):
        await self._hard_reset(reason="shutdown")


class ReplSessionManager:
    """Manages multiple REPL sessions, one per session_id."""

    def __init__(self):
        self._sessions: dict[str, ReplSession] = {}
        self._mode: Literal["off", "sandboxed", "trusted_local"] = _normalize_repl_mode(
            os.environ.get("LIAGENT_REPL_MODE", "sandboxed")
        )

    @property
    def mode(self) -> Literal["off", "sandboxed", "trusted_local"]:
        return self._mode

    def set_mode_sync(self, mode: str) -> Literal["off", "sandboxed", "trusted_local"]:
        self._mode = _normalize_repl_mode(mode)
        return self._mode

    async def set_mode(self, mode: str) -> Literal["off", "sandboxed", "trusted_local"]:
        new_mode = _normalize_repl_mode(mode)
        if new_mode == self._mode:
            return self._mode
        self._mode = new_mode
        # Existing worker subprocesses keep old env; recycle them on mode switch.
        await self.shutdown()
        return self._mode

    async def execute(self, session_id: str, code: str, reset: bool = False) -> str:
        """Run code in session, return formatted output."""
        if self._mode == "off":
            return "[Error] stateful_repl is disabled (repl_mode=off)."

        session = self._sessions.get(session_id)
        if session is None or session.mode != self._mode:
            if session is not None:
                await session.close()
            session = ReplSession(session_id, mode=self._mode)
            await session._spawn()
            self._sessions[session_id] = session

        if session.is_idle_expired:
            await session._respawn(reason="idle_ttl")

        result = await session.run(code, reset=reset)

        parts = []
        if result.get("stdout"):
            parts.append(result["stdout"])
        if result.get("error"):
            parts.append(f"[Error] {result['error']}")
        if result.get("vars"):
            parts.append(f"[Variables] {', '.join(result['vars'][:20])}")
        if not parts:
            parts.append("[No output]")

        return "\n".join(parts)

    def status(self, session_id: str | None = None) -> dict:
        now = time.time()
        if session_id:
            s = self._sessions.get(session_id)
            if s is None:
                return {
                    "mode": self._mode,
                    "session_id": session_id,
                    "exists": False,
                }
            return {
                "mode": self._mode,
                "session_id": session_id,
                "exists": True,
                "alive": s._is_alive(),
                "idle_seconds": int(max(0.0, now - s._last_used)),
            }
        sessions = []
        for sid, s in sorted(self._sessions.items(), key=lambda x: x[0]):
            sessions.append(
                {
                    "session_id": sid,
                    "alive": s._is_alive(),
                    "idle_seconds": int(max(0.0, now - s._last_used)),
                    "mode": s.mode,
                }
            )
        return {
            "mode": self._mode,
            "total_sessions": len(sessions),
            "sessions": sessions,
        }

    async def reset_session(self, session_id: str) -> bool:
        s = self._sessions.get(session_id)
        if s is None:
            return False
        await s._respawn(reason="manual_reset")
        return True

    async def kill_session(self, session_id: str) -> bool:
        s = self._sessions.get(session_id)
        if s is None:
            return False
        await s.close()
        self._sessions.pop(session_id, None)
        return True

    async def shutdown(self):
        for session in list(self._sessions.values()):
            await session.close()
        self._sessions.clear()


# Module-level singleton
_manager = ReplSessionManager()


def get_repl_mode() -> Literal["off", "sandboxed", "trusted_local"]:
    return _manager.mode


def set_repl_mode_sync(mode: str) -> Literal["off", "sandboxed", "trusted_local"]:
    return _manager.set_mode_sync(mode)


async def set_repl_mode(mode: str) -> Literal["off", "sandboxed", "trusted_local"]:
    return await _manager.set_mode(mode)


async def get_repl_status(session_id: str | None = None) -> dict:
    return _manager.status(session_id)


async def reset_repl_session(session_id: str) -> bool:
    return await _manager.reset_session(session_id)


async def kill_repl_session(session_id: str) -> bool:
    return await _manager.kill_session(session_id)


def _validate_repl(args: dict) -> tuple[bool, str]:
    code = str(args.get("code", "")).strip()
    reset = args.get("reset", False)
    if not code and not reset:
        return False, "code is required (or set reset=true to clear state)"
    if len(code) > 5000:
        return False, "code too long (max 5000 chars)"
    return True, "ok"


@tool(
    name="stateful_repl",
    description=(
        "Run Python code in a persistent REPL session. "
        "Variables and state persist across calls within the same conversation. "
        "Useful for multi-step data analysis, incremental computation, and Jupyter-like workflows. "
        "Set reset=true to clear all state. Network and dangerous modules are blocked."
    ),
    risk_level="medium",
    requires_confirmation=True,
    capability=ToolCapability(
        data_classification="internal",
        filesystem_access=True,
        max_output_chars=_MAX_OUTPUT_CHARS,
        latency_tier="medium",
        idempotent=False,
        failure_modes=("invalid_input", "timeout"),
        min_timeout_sec=_TIMEOUT + 5,
    ),
    validator=_validate_repl,
    parameters={
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to run. Variables persist across calls.",
            },
            "reset": {
                "type": "boolean",
                "description": "Set true to reset REPL state (clear all variables).",
            },
        },
        "required": ["code"],
    },
)
async def stateful_repl(code: str = "", reset: bool = False, **kwargs) -> str:
    """Run code in persistent REPL session."""
    session_id = kwargs.get("_session_id", "default")
    return await _manager.execute(session_id, code, reset=reset)


async def shutdown_repl():
    """Called by brain.shutdown() to clean up all REPL sessions."""
    await _manager.shutdown()
