"""Configuration REST endpoints — /api/config/*, /api/health, /api/metrics, /api/policy, /api/tool."""

from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..config import (
    BudgetConfig,
    LLMConfig,
    MCPDiscoveryConfig,
    MCPServerConfig,
    RoutingConfig,
    SandboxConfig,
    STTConfig,
    TTSConfig,
)
from ..engine.provider_registry import list_provider_presets
from ..tools.stateful_repl import (
    get_repl_status,
    kill_repl_session,
    reset_repl_session,
    set_repl_mode,
)
from . import shared_state as _shared
from .shared_state import (
    _allow_rate,
    _authorized_http,
    _client_id_from_request,
    _http_rate,
    _mask_secret,
)

router = APIRouter()


# ─── Config endpoints ───────────────────────────────────────────────────────

@router.get("/api/config")
async def get_config(request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    cfg = asdict(_shared._engine.config)
    cfg["llm"]["api_key_masked"] = _mask_secret(cfg["llm"].get("api_key", ""))
    cfg["tts"]["api_key_masked"] = _mask_secret(cfg["tts"].get("api_key", ""))
    cfg["stt"]["api_key_masked"] = _mask_secret(cfg["stt"].get("api_key", ""))
    cfg["llm"]["api_key"] = ""
    cfg["tts"]["api_key"] = ""
    cfg["stt"]["api_key"] = ""
    return JSONResponse({
        "llm_status": _shared._engine.llm_status(),
        "tts_status": _shared._engine.tts_status(),
        "stt_status": _shared._engine.stt_status(),
        "tts_enabled": _shared._engine.config.tts_enabled,
        "llm_provider_catalog": list_provider_presets(include_custom=True),
        "config": cfg,
    })


@router.post("/api/config/llm")
async def switch_llm(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    try:
        current = asdict(_shared._engine.config.llm)
        patch = {k: v for k, v in data.items() if v is not None}
        if patch.get("api_key", None) == "":
            patch.pop("api_key")
        current.update(patch)
        new_cfg = LLMConfig(**current)
        _shared._engine.switch_llm(new_cfg)
        return JSONResponse({"status": "ok", "llm_status": _shared._engine.llm_status()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/config/tts")
async def switch_tts(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    try:
        current = asdict(_shared._engine.config.tts)
        patch = {k: v for k, v in data.items() if v is not None}
        if patch.get("api_key", None) == "":
            patch.pop("api_key")
        current.update(patch)
        new_cfg = TTSConfig(**current)
        _shared._engine.switch_tts(new_cfg)
        return JSONResponse({"status": "ok", "tts_status": _shared._engine.tts_status()})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/config/stt")
async def switch_stt(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    try:
        current = asdict(_shared._engine.config.stt)
        patch = {k: v for k, v in data.items() if v is not None}
        if patch.get("api_key", None) == "":
            patch.pop("api_key")
        current.update(patch)
        backend = str(current.get("backend", "local") or "local").strip().lower()
        if backend not in {"local", "api"}:
            backend = "local"
        current["backend"] = backend
        current["model"] = str(current.get("model", "")).strip() or _shared._engine.config.stt.model
        current["language"] = str(current.get("language", "auto")).strip() or "auto"
        current["api_base_url"] = str(current.get("api_base_url", "")).strip()
        current["api_key"] = str(current.get("api_key", "")).strip()
        current["api_model"] = str(current.get("api_model", "")).strip()

        new_cfg = STTConfig(**current)
        _shared._engine.switch_stt(new_cfg)
        return JSONResponse(
            {
                "status": "ok",
                "stt_status": _shared._engine.stt_status(),
                "stt_backend": _shared._engine.config.stt.backend,
                "stt_model": _shared._engine.config.stt.model,
                "stt_language": _shared._engine.config.stt.language,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/config/runtime")
async def update_runtime_config(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    try:
        cfg = _shared._engine.config
        mode = data.get("runtime_mode")
        if mode is not None:
            runtime_mode = str(mode or "").strip().lower()
            if runtime_mode not in {"local_private", "hybrid_balanced", "cloud_performance"}:
                return JSONResponse(
                    {"error": "runtime_mode must be local_private|hybrid_balanced|cloud_performance"},
                    status_code=400,
                )
            cfg.runtime_mode = runtime_mode

        if isinstance(data.get("routing"), dict):
            current_routing = asdict(cfg.routing)
            patch = {
                k: v for k, v in data["routing"].items()
                if k in current_routing and v is not None
            }
            current_routing.update(patch)
            cfg.routing = RoutingConfig(**current_routing)

        if isinstance(data.get("sandbox"), dict):
            current_sandbox = asdict(cfg.sandbox)
            patch = {
                k: v for k, v in data["sandbox"].items()
                if k in current_sandbox and v is not None
            }
            current_sandbox.update(patch)
            cfg.sandbox = SandboxConfig(**current_sandbox)

        if isinstance(data.get("budget"), dict):
            current_budget = asdict(cfg.budget)
            patch = {
                k: v for k, v in data["budget"].items()
                if k in current_budget and v is not None
            }
            current_budget.update(patch)
            cfg.budget = BudgetConfig(**current_budget)

        if isinstance(data.get("mcp_discovery"), dict):
            current_discovery = asdict(cfg.mcp_discovery)
            patch = {
                k: v for k, v in data["mcp_discovery"].items()
                if k in current_discovery and v is not None
            }
            current_discovery.update(patch)
            cfg.mcp_discovery = MCPDiscoveryConfig(**current_discovery)

        repl_mode_raw = data.get("repl_mode")
        if repl_mode_raw is not None:
            repl_mode = str(repl_mode_raw or "").strip().lower()
            if repl_mode not in {"off", "sandboxed", "trusted_local"}:
                return JSONResponse(
                    {"error": "repl_mode must be off|sandboxed|trusted_local"},
                    status_code=400,
                )
            if repl_mode == "trusted_local" and not bool(data.get("confirm_trusted_local", False)):
                return JSONResponse(
                    {"error": "trusted_local requires confirm_trusted_local=true"},
                    status_code=400,
                )
            cfg.repl_mode = await set_repl_mode(repl_mode)

        cfg.save()
        _shared._engine.configure_runtime_adapters()
        return JSONResponse(
            {
                "status": "ok",
                "runtime_mode": cfg.runtime_mode,
                "routing": asdict(cfg.routing),
                "sandbox": asdict(cfg.sandbox),
                "budget": asdict(cfg.budget),
                "mcp_discovery": asdict(cfg.mcp_discovery),
                "repl_mode": cfg.repl_mode,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/config/tool_policy")
async def switch_tool_policy(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    profile = str(data.get("tool_profile", "")).strip().lower()
    if profile not in {"minimal", "research", "full"}:
        return JSONResponse(
            {"error": "tool_profile must be one of: minimal, research, full"},
            status_code=400,
        )

    _shared._engine.config.tool_profile = profile
    _shared._engine.config.save()
    _shared._brain.set_tool_profile(profile)
    return JSONResponse({"status": "ok", "tool_profile": _shared._engine.config.tool_profile})


@router.post("/api/config/repl_mode")
async def switch_repl_mode(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    mode = str(data.get("repl_mode", "")).strip().lower()
    if mode not in {"off", "sandboxed", "trusted_local"}:
        return JSONResponse(
            {"error": "repl_mode must be one of: off, sandboxed, trusted_local"},
            status_code=400,
        )
    if mode == "trusted_local" and not bool(data.get("confirm_trusted_local", False)):
        return JSONResponse(
            {"error": "trusted_local requires confirm_trusted_local=true"},
            status_code=400,
        )

    applied = await set_repl_mode(mode)
    _shared._engine.config.repl_mode = applied
    _shared._engine.config.save()
    return JSONResponse({"status": "ok", "repl_mode": applied})


@router.get("/api/repl/status")
async def repl_status(request: Request, session_id: str | None = None):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    sid = str(session_id or "").strip() or None
    status = await get_repl_status(sid)
    return JSONResponse({"status": "ok", "repl": status})


@router.post("/api/repl/reset")
async def repl_reset(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    sid = str(data.get("session_id", "")).strip()
    if not sid:
        return JSONResponse({"error": "session_id is required"}, status_code=400)
    ok = await reset_repl_session(sid)
    if not ok:
        return JSONResponse({"error": "session not found", "session_id": sid}, status_code=404)
    return JSONResponse({"status": "ok", "session_id": sid, "action": "reset"})


@router.post("/api/repl/kill")
async def repl_kill(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    sid = str(data.get("session_id", "")).strip()
    if not sid:
        return JSONResponse({"error": "session_id is required"}, status_code=400)
    ok = await kill_repl_session(sid)
    if not ok:
        return JSONResponse({"error": "session not found", "session_id": sid}, status_code=404)
    return JSONResponse({"status": "ok", "session_id": sid, "action": "kill"})


@router.post("/api/config/tts_toggle")
async def toggle_tts(request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    try:
        _shared._engine.set_tts_enabled(not _shared._engine.config.tts_enabled)
        return JSONResponse(
            {
                "tts_enabled": _shared._engine.config.tts_enabled,
                "tts_status": _shared._engine.tts_status(),
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/config/tts_voice")
async def switch_tts_voice(data: dict, request: Request):
    """Lightweight speaker/speed switch -- no model reload."""
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    try:
        speaker = data.get("speaker_name")
        speed = data.get("speed")

        if speaker and _shared._engine.tts and hasattr(_shared._engine.tts, "set_speaker"):
            _shared._engine.tts.set_speaker(speaker)
            _shared._engine.config.tts.speaker_name = _shared._engine.tts.speaker_name  # normalized
        if speed is not None and _shared._engine.tts:
            s = max(0.5, min(2.0, float(speed)))
            _shared._engine.tts.speed = s
            _shared._engine.config.tts.speed = s
        _shared._engine.config.save()
        return JSONResponse({
            "tts_status": _shared._engine.tts_status(),
            "speaker_name": _shared._engine.config.tts.speaker_name,
            "speed": _shared._engine.config.tts.speed,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/config/thinking")
async def update_thinking(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    try:
        if "show_thinking" in data:
            _shared._engine.config.show_thinking = bool(data["show_thinking"])
        if "enable_thinking" in data:
            _shared._engine.config.enable_thinking = bool(data["enable_thinking"])
        _shared._engine.config.save()
        return JSONResponse({
            "show_thinking": _shared._engine.config.show_thinking,
            "enable_thinking": _shared._engine.config.enable_thinking,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/metrics/weekly")
async def metrics_weekly(request: Request, days: int = 7):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    days = max(1, min(days, 30))
    return JSONResponse(_shared._metrics.weekly_summary(days=days))


@router.get("/api/policy/audit")
async def policy_audit(request: Request, limit: int = 50):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    n = max(1, min(limit, 200))
    rows = _shared._brain.tool_policy.recent_audit(limit=n)
    return JSONResponse({"items": rows})


@router.post("/api/tool/confirm")
async def api_tool_confirm(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    token = str(data.get("token", "")).strip()
    force = bool(data.get("force", False))
    if not token:
        return JSONResponse({"error": "missing token"}, status_code=400)
    result = await _shared._brain.resolve_confirmation(token, approved=True, force=force)
    return JSONResponse(result)


@router.get("/api/health")
async def health_check(request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    health = _shared._engine.health_check() if _shared._engine else {"error": "engine not initialized"}
    return JSONResponse(health)


# ─── MCP Server management ─────────────────────────────────────────────────

def _serialize_mcp_servers() -> list[dict]:
    """Serialize MCP servers with masked env values."""
    servers = []
    for s in _shared._engine.config.mcp_servers:
        servers.append({
            "name": s.name,
            "command": s.command,
            "args": s.args,
            "env_masked": {k: _mask_secret(v) for k, v in s.env.items()},
            "risk_level": s.risk_level,
            "network_access": s.network_access,
            "filesystem_access": s.filesystem_access,
            "enabled": s.enabled,
        })
    return servers


@router.get("/api/config/mcp")
async def get_mcp_config(request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    return JSONResponse({"servers": _serialize_mcp_servers()})


@router.post("/api/config/mcp")
async def add_mcp_server(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    name = str(data.get("name", "")).strip()
    command = str(data.get("command", "")).strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    if not command:
        return JSONResponse({"error": "command is required"}, status_code=400)
    if any(s.name == name for s in _shared._engine.config.mcp_servers):
        return JSONResponse({"error": f"server '{name}' already exists"}, status_code=400)

    new_server = MCPServerConfig(
        name=name,
        command=command,
        args=data.get("args", []),
        env=data.get("env", {}),
        risk_level=str(data.get("risk_level", "medium")),
        network_access=bool(data.get("network_access", True)),
        filesystem_access=bool(data.get("filesystem_access", False)),
        enabled=bool(data.get("enabled", True)),
    )
    _shared._engine.config.mcp_servers.append(new_server)
    _shared._engine.config.save()
    return JSONResponse({
        "status": "ok",
        "servers": _serialize_mcp_servers(),
        "message": "Server added. Restart to connect.",
    })


@router.post("/api/config/mcp/update")
async def update_mcp_server(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    name = str(data.get("name", "")).strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    target = None
    for s in _shared._engine.config.mcp_servers:
        if s.name == name:
            target = s
            break
    if target is None:
        return JSONResponse({"error": f"server '{name}' not found"}, status_code=404)

    if "command" in data:
        target.command = str(data["command"]).strip()
    if "args" in data:
        target.args = list(data["args"]) if isinstance(data["args"], list) else []
    if "env" in data and isinstance(data["env"], dict):
        target.env = data["env"]
    if "risk_level" in data:
        target.risk_level = str(data["risk_level"])
    if "network_access" in data:
        target.network_access = bool(data["network_access"])
    if "filesystem_access" in data:
        target.filesystem_access = bool(data["filesystem_access"])
    if "enabled" in data:
        target.enabled = bool(data["enabled"])

    _shared._engine.config.save()
    return JSONResponse({
        "status": "ok",
        "servers": _serialize_mcp_servers(),
        "message": "Server updated. Restart to apply changes.",
    })


@router.post("/api/config/mcp/delete")
async def delete_mcp_server(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    name = str(data.get("name", "")).strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    before = len(_shared._engine.config.mcp_servers)
    _shared._engine.config.mcp_servers = [s for s in _shared._engine.config.mcp_servers if s.name != name]
    if len(_shared._engine.config.mcp_servers) == before:
        return JSONResponse({"error": f"server '{name}' not found"}, status_code=404)

    _shared._engine.config.save()
    return JSONResponse({
        "status": "ok",
        "servers": _serialize_mcp_servers(),
        "message": "Server removed. Restart to clean up connections.",
    })


@router.post("/api/tool/reject")
async def api_tool_reject(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    token = str(data.get("token", "")).strip()
    if not token:
        return JSONResponse({"error": "missing token"}, status_code=400)
    result = await _shared._brain.resolve_confirmation(token, approved=False)
    return JSONResponse(result)
