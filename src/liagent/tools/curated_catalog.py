"""Curated MCP server catalog — pre-approved low-risk servers.

SAFETY RULES:
- filesystem restricted to cwork directory, NOT $HOME
- No empty env placeholder values (would overwrite real env vars via os.environ merge)
- Servers needing API keys: env key documented in comment only
- Reuses caller's TrustRegistry instance (single writer)
"""

import json
import logging
from pathlib import Path

from ._path_security import get_cwork_root_str
from .trust_registry import TrustRegistry

_log = logging.getLogger(__name__)

CURATED_SERVERS = [
    {
        "name": "filesystem",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "{cwork_dir}"],
        "risk_level": "low",
        "network_access": False,
        "filesystem_access": True,
    },
    {
        "name": "fetch",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "risk_level": "low",
        "network_access": True,
        "filesystem_access": False,
    },
    {
        "name": "github",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "risk_level": "medium",
        "network_access": True,
        "filesystem_access": False,
        # Requires GITHUB_PERSONAL_ACCESS_TOKEN env var set by user
    },
    {
        "name": "sqlite",
        "command": "uvx",
        "args": ["mcp-server-sqlite", "--db-path", str(Path.home() / ".liagent" / "scratch.db")],
        "risk_level": "low",
        "network_access": False,
        "filesystem_access": True,
    },
    {
        "name": "brave-search",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "risk_level": "low",
        "network_access": True,
        "filesystem_access": False,
        # Requires BRAVE_API_KEY env var set by user
    },
    {
        "name": "memory",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "risk_level": "low",
        "network_access": False,
        "filesystem_access": True,
    },
    {
        "name": "playwright",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-playwright"],
        "risk_level": "high",
        "network_access": True,
        "filesystem_access": False,
        "wrapper_only": True,
    },
]


def _render_server(server: dict) -> dict:
    rendered = dict(server)
    args = []
    for arg in server.get("args", []):
        if isinstance(arg, str):
            args.append(arg.format(cwork_dir=get_cwork_root_str()))
        else:
            args.append(arg)
    rendered["args"] = args
    return rendered


def bootstrap_curated_catalog(
    *,
    mcp_dir: Path | None = None,
    trust_registry: TrustRegistry,
) -> None:
    """Write curated MCP server configs and pre-approve them in trust registry.

    Idempotent — safe to call on every startup. Only writes files that don't exist.
    Reuses the caller's TrustRegistry instance (single writer).
    """
    mcp_dir = mcp_dir or (Path.home() / ".liagent" / "mcp.d")
    mcp_dir.mkdir(parents=True, exist_ok=True)

    for raw_server in CURATED_SERVERS:
        server = _render_server(raw_server)
        name = server["name"]
        config_path = mcp_dir / f"curated-{name}.json"

        spec: dict = {"command": server["command"], "args": server["args"]}
        if "risk_level" in server:
            spec["risk_level"] = server["risk_level"]
        if "network_access" in server:
            spec["network_access"] = server["network_access"]
        if "filesystem_access" in server:
            spec["filesystem_access"] = server["filesystem_access"]
        # Only include env if present AND all values are non-empty
        if server.get("env"):
            clean_env = {k: v for k, v in server["env"].items() if v}
            if clean_env:
                spec["env"] = clean_env

        config_data = {"mcpServers": {name: spec}}

        if not config_path.exists():
            config_path.write_text(
                json.dumps(config_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _log.info("curated_catalog: wrote %s", config_path)

        # Only approve if not yet tracked or currently unknown.
        # Preserve user's explicit "revoked" or existing "approved" state.
        current = trust_registry.get_status(name)
        if current == "unknown":
            trust_registry.set_status(name, "approved", source="curated")
