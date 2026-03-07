"""Docker sandbox routing for high-risk local tools."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..logging import get_logger

_log = get_logger("sandbox_runtime")

_MANAGED_TOOLS = {"python_exec", "run_tests", "lint_code"}
_MODES = {"off", "non_main", "all"}


@dataclass
class SandboxAttempt:
    used: bool = False
    output: str = ""
    error: str = ""


@dataclass
class _SandboxState:
    enabled: bool = False
    mode: str = "off"  # off | non_main | all
    image: str = "liagent-sandbox:latest"
    network_default: str = "off"  # off | on
    docker_bin: str = "docker"
    workspace_mount: str = str(Path.home() / "Desktop" / "cwork")
    cpu_limit: float = 1.5
    memory_limit_mb: int = 1024
    pids_limit: int = 256


_STATE = _SandboxState()


def configure_from_app_config(config: AppConfig):
    """Sync global sandbox runtime state from app config."""
    sb = getattr(config, "sandbox", None)
    if sb is None:
        _STATE.enabled = False
        _STATE.mode = "off"
        return
    mode = str(getattr(sb, "mode", "off") or "off").strip().lower()
    if mode not in _MODES:
        mode = "off"
    _STATE.enabled = bool(getattr(sb, "enabled", False))
    _STATE.mode = mode
    _STATE.image = str(getattr(sb, "image", "") or "liagent-sandbox:latest").strip()
    _STATE.network_default = (
        "on" if str(getattr(sb, "network_default", "off")).strip().lower() == "on" else "off"
    )
    _STATE.docker_bin = str(getattr(sb, "docker_bin", "") or "docker").strip()
    _STATE.workspace_mount = str(getattr(sb, "workspace_mount", "") or (Path.home() / "Desktop" / "cwork")).strip()
    try:
        _STATE.cpu_limit = max(0.2, float(getattr(sb, "cpu_limit", 1.5) or 1.5))
    except (TypeError, ValueError):
        _STATE.cpu_limit = 1.5
    try:
        _STATE.memory_limit_mb = max(256, int(getattr(sb, "memory_limit_mb", 1024) or 1024))
    except (TypeError, ValueError):
        _STATE.memory_limit_mb = 1024
    try:
        _STATE.pids_limit = max(32, int(getattr(sb, "pids_limit", 256) or 256))
    except (TypeError, ValueError):
        _STATE.pids_limit = 256


def _should_route(tool_name: str, *, delegated: bool) -> bool:
    if not _STATE.enabled:
        return False
    if tool_name not in _MANAGED_TOOLS:
        return False
    if _STATE.mode == "all":
        return True
    if _STATE.mode == "non_main":
        return bool(delegated)
    return False


def _workspace_path() -> Path:
    raw = str(_STATE.workspace_mount or "").strip() or str(Path.home() / "Desktop" / "cwork")
    return Path(raw).expanduser().resolve()


def _project_root() -> Path:
    # src/liagent/tools/sandbox_runtime.py -> project root
    return Path(__file__).resolve().parents[3]


async def maybe_execute_in_sandbox(
    tool_name: str,
    args: dict[str, Any],
    *,
    timeout_sec: float,
    delegated: bool = False,
) -> SandboxAttempt:
    """Try executing a managed tool in Docker sandbox.

    Returns SandboxAttempt(used=False) if sandbox is disabled or not routed.
    Returns SandboxAttempt(used=False, error=...) if routed but sandbox failed.
    """
    if not _should_route(tool_name, delegated=delegated):
        return SandboxAttempt(used=False)
    docker_bin = _STATE.docker_bin or "docker"
    if shutil.which(docker_bin) is None:
        return SandboxAttempt(used=False, error=f"{docker_bin} not found")

    workspace = _workspace_path()
    if not workspace.exists():
        return SandboxAttempt(used=False, error=f"workspace not found: {workspace}")

    project_root = _project_root()
    src_dir = (project_root / "src").resolve()
    if not src_dir.exists():
        return SandboxAttempt(used=False, error=f"src not found: {src_dir}")

    args_json = json.dumps(args or {}, ensure_ascii=False)
    cmd = [
        docker_bin,
        "run",
        "--rm",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(_STATE.pids_limit),
        "--cpus",
        str(_STATE.cpu_limit),
        "--memory",
        f"{_STATE.memory_limit_mb}m",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "PYTHONIOENCODING=utf-8",
        "-e",
        "PYTHONPATH=/opt/liagent/src",
        "-e",
        "LIAGENT_SANDBOX_WORKSPACE=/workspace",
        "-v",
        f"{workspace}:/workspace:rw",
        "-v",
        f"{src_dir}:/opt/liagent/src:ro",
        "-w",
        "/workspace",
    ]
    if _STATE.network_default == "off":
        cmd.extend(["--network", "none"])
    image = _STATE.image or "liagent-sandbox:latest"
    cmd.extend(
        [
            image,
            "python",
            "-m",
            "liagent.tools.sandbox_runner",
            "--tool",
            tool_name,
            "--timeout",
            str(max(1.0, float(timeout_sec))),
            "--args-json",
            args_json,
        ]
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=max(2.0, float(timeout_sec) + 8.0),
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return SandboxAttempt(used=False, error=f"docker timeout for {tool_name}")
    except Exception as exc:
        return SandboxAttempt(used=False, error=f"docker exec failed: {exc}")

    out = stdout.decode("utf-8", errors="replace") if stdout else ""
    err = stderr.decode("utf-8", errors="replace") if stderr else ""
    if proc.returncode != 0:
        msg = (err or out or f"docker exited with {proc.returncode}").strip()
        _log.warning("sandbox_exec_failed", tool=tool_name, code=proc.returncode, error=msg[:300])
        return SandboxAttempt(used=False, error=msg)
    return SandboxAttempt(used=True, output=(out or "").strip())

