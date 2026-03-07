"""MCP (Model Context Protocol) adapter — registers MCP Server tools as local ToolDefs."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..logging import get_logger

_log = get_logger("mcp_bridge")

if TYPE_CHECKING:
    from ..config import MCPServerConfig
    from .trust_registry import TrustRegistry

# Guard all mcp imports — SDK is an optional dependency.
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


def mcp_available() -> bool:
    """Check whether the mcp SDK is installed."""
    return _MCP_AVAILABLE


class MCPBridge:
    """Connect to MCP Servers (stdio) and register their tools into the local registry."""

    def __init__(
        self,
        servers: list[MCPServerConfig],
        *,
        call_timeout_sec: float = 30.0,
        max_response_bytes: int = 1_000_000,
        trust_registry: "TrustRegistry | None" = None,
    ):
        self._configs = [s for s in servers if s.enabled]
        # name → (session, cleanup_callback)
        self._sessions: dict[str, tuple[ClientSession, object]] = {}
        self._registered: list[str] = []
        self._server_errors: dict[str, str] = {}
        # AsyncExitStack per server — properly manages anyio task groups
        self._exit_stacks: dict[str, AsyncExitStack] = {}
        self._call_timeout_sec = max(1.0, float(call_timeout_sec or 30.0))
        self._max_response_bytes = max(4096, int(max_response_bytes or 1_000_000))
        self._trust_registry = trust_registry
        # Lifecycle ops must run on one task to satisfy anyio cancel-scope invariants.
        self._lifecycle_queue: asyncio.Queue[tuple[str, tuple[Any, ...], asyncio.Future]] | None = None
        self._lifecycle_task: asyncio.Task | None = None
        self._lifecycle_loop: asyncio.AbstractEventLoop | None = None

    def _ensure_lifecycle_worker(self) -> None:
        loop = asyncio.get_running_loop()
        if (
            self._lifecycle_task is not None
            and not self._lifecycle_task.done()
            and self._lifecycle_loop is loop
            and self._lifecycle_queue is not None
        ):
            return
        self._lifecycle_queue = asyncio.Queue()
        self._lifecycle_loop = loop
        self._lifecycle_task = loop.create_task(self._lifecycle_worker())

    async def _run_lifecycle_op(self, op: str, *args: Any) -> Any:
        self._ensure_lifecycle_worker()
        queue = self._lifecycle_queue
        if queue is None:
            raise RuntimeError("MCP lifecycle worker queue is not initialized")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await queue.put((op, args, fut))
        return await fut

    async def _lifecycle_worker(self):
        queue = self._lifecycle_queue
        if queue is None:
            return
        try:
            while True:
                op, args, fut = await queue.get()
                try:
                    if op == "_stop_worker":
                        if not fut.done():
                            fut.set_result(None)
                        break
                    if op == "connect_all":
                        result = await self._connect_all_impl()
                    elif op == "register_all":
                        result = await self._register_all_impl()
                    elif op == "reload_servers":
                        result = await self._reload_servers_impl(*args)
                    elif op == "shutdown":
                        result = await self._shutdown_impl()
                    else:
                        raise RuntimeError(f"unknown MCP lifecycle op: {op}")
                    if not fut.done():
                        fut.set_result(result)
                except Exception as e:
                    if not fut.done():
                        fut.set_exception(e)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(self._shutdown_impl())
            except Exception as e:
                _log.error("mcp_bridge", e, action="worker_cancel_cleanup")
            raise
        finally:
            self._lifecycle_queue = None
            self._lifecycle_task = None
            self._lifecycle_loop = None

    # ── Discovery ───────────────────────────────────────────────────────

    @staticmethod
    def discover_local_servers(dirs: list[str] | None = None) -> list["MCPServerConfig"]:
        """Discover MCP server definitions from local JSON files.

        Supported JSON formats:
        1) {"servers": [{"name":"...", "command":"...", "args": [...]}]}
        2) {"mcpServers": {"name": {"command":"...", "args":[...]}}}
        3) {"name":"...", "command":"...", "args":[...]}
        """
        from ..config import MCPServerConfig

        search_dirs = [Path(d).expanduser() for d in (dirs or []) if str(d or "").strip()]
        discovered: list[MCPServerConfig] = []
        seen: set[str] = set()
        for base in search_dirs:
            if not base.exists() or not base.is_dir():
                continue
            for path in sorted(base.glob("*.json")):
                try:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                entries: list[dict] = []
                if isinstance(obj, dict) and isinstance(obj.get("servers"), list):
                    entries = [e for e in obj.get("servers", []) if isinstance(e, dict)]
                elif isinstance(obj, dict) and isinstance(obj.get("mcpServers"), dict):
                    for name, spec in obj["mcpServers"].items():
                        if isinstance(spec, dict):
                            item = dict(spec)
                            item.setdefault("name", str(name))
                            entries.append(item)
                elif isinstance(obj, dict) and obj.get("name") and obj.get("command"):
                    entries = [obj]

                for e in entries:
                    name = str(e.get("name", "")).strip()
                    command = str(e.get("command", "")).strip()
                    if not name or not command or name in seen:
                        continue
                    seen.add(name)
                    discovered.append(
                        MCPServerConfig(
                            name=name,
                            command=command,
                            args=list(e.get("args", [])) if isinstance(e.get("args"), list) else [],
                            env=dict(e.get("env", {})) if isinstance(e.get("env"), dict) else {},
                            risk_level=str(e.get("risk_level", "medium") or "medium"),
                            network_access=bool(e.get("network_access", True)),
                            filesystem_access=bool(e.get("filesystem_access", False)),
                            enabled=bool(e.get("enabled", True)),
                        )
                    )
        return discovered

    @staticmethod
    def merge_servers(
        configured: list["MCPServerConfig"],
        discovered: list["MCPServerConfig"],
    ) -> list["MCPServerConfig"]:
        """Merge configured + discovered servers by name (configured wins)."""
        merged: dict[str, MCPServerConfig] = {}
        for srv in discovered:
            merged[srv.name] = srv
        for srv in configured:
            merged[srv.name] = srv
        return list(merged.values())

    # ── Connection ──────────────────────────────────────────────────────

    async def connect_all(self):
        """Connect to every enabled MCP server via stdio transport."""
        if not _MCP_AVAILABLE:
            for cfg in self._configs:
                if cfg.name in self._sessions:
                    continue
                if not self._command_exists(cfg.command):
                    reason = f"command not found: {cfg.command}"
                else:
                    reason = "mcp SDK not installed"
                self._server_errors[cfg.name] = reason
            _log.warning("mcp SDK not installed, skipping MCP server connections")
            return
        await self._run_lifecycle_op("connect_all")

    async def _connect_all_impl(self):
        for cfg in self._configs:
            if cfg.name in self._sessions:
                continue
            try:
                if not self._command_exists(cfg.command):
                    reason = f"command not found: {cfg.command}"
                    self._server_errors[cfg.name] = reason
                    _log.warning("mcp_preflight_failed", server=cfg.name, reason=reason)
                    continue
                server_params = StdioServerParameters(
                    command=cfg.command,
                    args=cfg.args,
                    env={**os.environ, **cfg.env},
                )
                stack = AsyncExitStack()
                await stack.__aenter__()
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(server_params)
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                self._sessions[cfg.name] = (session, None)
                self._exit_stacks[cfg.name] = stack
                self._server_errors.pop(cfg.name, None)
                _log.event("mcp_connected", server=cfg.name)
            except Exception as e:
                self._server_errors[cfg.name] = str(e)
                _log.error("mcp_bridge", e, server=cfg.name, action="connect")

    # ── Tool Discovery & Registration ───────────────────────────────────

    async def register_all(self):
        """Discover tools from all connected servers and register them as ToolDefs."""
        await self._run_lifecycle_op("register_all")

    async def _register_all_impl(self):
        """Discover tools from all connected servers and register them as ToolDefs."""
        from . import ToolCapability, ToolDef, _TOOLS

        for server_name, (session, _) in self._sessions.items():
            cfg = next(c for c in self._configs if c.name == server_name)
            try:
                resp = await session.list_tools()

                # wrapper_only: discover tools (for bridge.call_tool) but do NOT
                # register them in _TOOLS, so the LLM cannot see or call them directly.
                if getattr(cfg, "wrapper_only", False):
                    _log.event(
                        "mcp_tools_wrapper_only",
                        server=server_name,
                        count=len(resp.tools),
                    )
                    if self._trust_registry is not None:
                        self._trust_registry.ensure_registered(server_name, source="discovered")
                    continue

                prefix = f"{server_name}__"
                # Replace prior tool set for this server to support hot-reload updates.
                for k in [name for name in list(_TOOLS.keys()) if name.startswith(prefix)]:
                    _TOOLS.pop(k, None)
                self._registered = [name for name in self._registered if not name.startswith(prefix)]
                for tool in resp.tools:
                    tool_name = f"{server_name}__{tool.name}"
                    tool_def = ToolDef(
                        name=tool_name,
                        description=tool.description or tool.name,
                        parameters=tool.inputSchema or {"properties": {}},
                        func=self._make_caller(
                            session,
                            tool.name,
                            timeout_sec=self._call_timeout_sec,
                            max_response_bytes=self._max_response_bytes,
                        ),
                        risk_level=cfg.risk_level,
                        capability=ToolCapability(
                            network_access=cfg.network_access,
                            filesystem_access=cfg.filesystem_access,
                            max_output_chars=2000,
                        ),
                    )
                    _TOOLS[tool_name] = tool_def
                    if tool_name not in self._registered:
                        self._registered.append(tool_name)
                _log.event(
                    "mcp_tools_registered",
                    server=server_name,
                    count=len(resp.tools),
                )
                # Record server in trust registry (no-op if already tracked)
                if self._trust_registry is not None:
                    self._trust_registry.ensure_registered(server_name, source="discovered")
            except Exception as e:
                _log.error("mcp_bridge", e, server=server_name, action="register")

    @staticmethod
    def _make_caller(
        session: ClientSession,
        tool_name: str,
        *,
        timeout_sec: float = 30.0,
        max_response_bytes: int = 1_000_000,
    ):
        """Create an async closure bridging ToolDef.func(**kwargs) → MCP call_tool."""

        async def call(**kwargs):
            result = await asyncio.wait_for(
                session.call_tool(tool_name, kwargs),
                timeout=max(0.1, float(timeout_sec or 30.0)),
            )
            # MCP returns Content[] — concatenate text parts
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
            text = "\n".join(parts) if parts else "[MCP] Tool returned empty result"
            raw = text.encode("utf-8", errors="replace")
            limit = max(4096, int(max_response_bytes or 1_000_000))
            if len(raw) > limit:
                truncated = raw[:limit].decode("utf-8", errors="ignore").rstrip()
                return truncated + "\n...[MCP response truncated]"
            return text

        return call

    async def call_tool(self, server_name: str, tool_name: str, args: dict) -> str:
        """Call an MCP tool directly by server and tool name.

        Used by wrapper tools (e.g. browser.py) to invoke MCP tools
        that are not registered in the global _TOOLS registry.
        """
        if server_name not in self._sessions:
            reason = self.server_error(server_name)
            if reason:
                return f"[Unavailable] MCP server '{server_name}' unavailable: {reason}"
            return f"[Error] MCP server '{server_name}' is not connected"
        session, _ = self._sessions[server_name]
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, args),
                timeout=max(0.1, self._call_timeout_sec),
            )
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
            return "\n".join(parts) if parts else "[MCP] Tool returned empty result"
        except asyncio.TimeoutError:
            return f"[Timeout] MCP call {server_name}/{tool_name} exceeded {self._call_timeout_sec}s"
        except Exception as e:
            return f"[Error] MCP call failed: {e}"

    # ── Shutdown ────────────────────────────────────────────────────────

    async def shutdown(self):
        """Close all MCP connections and remove registered tools from the registry."""
        from . import _TOOLS

        try:
            await self._run_lifecycle_op("shutdown")
        except Exception as e:
            _log.error("mcp_bridge", e, action="shutdown")
        try:
            await self._run_lifecycle_op("_stop_worker")
        except Exception:
            pass

        for tool_name in self._registered:
            _TOOLS.pop(tool_name, None)
        self._registered.clear()

    async def _shutdown_impl(self):
        for name in list(self._sessions):
            try:
                self._sessions.pop(name, None)
                stack = self._exit_stacks.pop(name, None)
                if stack:
                    await stack.aclose()
            except Exception as e:
                # Do not re-raise during teardown; force-close paths may cancel tasks.
                _log.error("mcp_bridge", e, server=name, action="shutdown_impl")

    async def reload_servers(self, servers: list["MCPServerConfig"]):
        """Hot-reload MCP server configuration without process restart."""
        await self._run_lifecycle_op("reload_servers", servers)

    async def _reload_servers_impl(self, servers: list["MCPServerConfig"]):
        """Hot-reload MCP server configuration without process restart."""
        from . import _TOOLS

        incoming = [s for s in servers if s.enabled]
        # Filter out revoked servers on reload (defense in depth).
        # Unknown servers are allowed — evaluate() blocks their tools until confirmed.
        if self._trust_registry is not None:
            incoming = [s for s in incoming if self._trust_registry.get_status(s.name) != "revoked"]
        target_names = {s.name for s in incoming}
        current_names = set(self._sessions.keys())

        # 1) Remove disabled/removed servers
        removed = current_names - target_names
        for name in removed:
            try:
                self._sessions.pop(name, None)
                stack = self._exit_stacks.pop(name, None)
                if stack:
                    await stack.aclose()
            except Exception as e:
                _log.error("mcp_bridge", e, server=name, action="reload_remove")
            prefix = f"{name}__"
            self._registered = [t for t in self._registered if not t.startswith(prefix)]
            for tool_name in [k for k in list(_TOOLS.keys()) if k.startswith(prefix)]:
                _TOOLS.pop(tool_name, None)

        self._configs = incoming

        # 2) Connect missing servers
        await self._connect_all_impl()

        # 3) Re-register tools for all connected servers (idempotent overwrite)
        await self._register_all_impl()

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def connected_servers(self) -> list[str]:
        return list(self._sessions.keys())

    @property
    def registered_tools(self) -> list[str]:
        return list(self._registered)

    @property
    def server_errors(self) -> dict[str, str]:
        return dict(self._server_errors)

    def server_error(self, server_name: str) -> str:
        return str(self._server_errors.get(server_name, "") or "")

    @staticmethod
    def _command_exists(command: str) -> bool:
        raw = str(command or "").strip()
        if not raw:
            return False
        if os.path.isabs(raw):
            return Path(raw).exists()
        return shutil.which(raw) is not None


# ── Module-level bridge accessor ────────────────────────────────────
_bridge_instance: MCPBridge | None = None


def set_bridge(bridge: MCPBridge | None):
    """Set the module-level bridge instance (called by engine_manager)."""
    global _bridge_instance
    _bridge_instance = bridge


def get_bridge() -> MCPBridge | None:
    """Get the module-level bridge instance for wrapper tools."""
    return _bridge_instance
