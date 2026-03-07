"""Shared state and utility functions for web server route modules.

All module-level globals and common helpers that are used across multiple
route files live here to avoid circular imports.
"""

import asyncio
import json
import os
import time
from typing import Any

from fastapi import Request, WebSocket, WebSocketDisconnect

from ..logging import get_logger
from ..agent.self_supervision import InteractionMetrics
from ..engine.engine_manager import EngineManager
from ..agent.brain import AgentBrain

_log = get_logger("web_server")

# ─── Backward-compat globals (set by create_app) ────────────────────────────
_engine: EngineManager = None  # type: ignore[assignment]
_brain: AgentBrain = None  # type: ignore[assignment]
_orchestrator = None
_signal_poller = None  # SignalPoller instance, set by create_app
_anomaly_detector = None  # AnomalyDetector instance, set by create_app
_anomaly_precheck = None  # AnomalyPreCheck instance, set by main.py heartbeat wiring
_metrics: InteractionMetrics | None = None
_BRAIN_RUN_LOCK: asyncio.Lock | None = None

# ─── Auth / rate-limit config ───────────────────────────────────────────────
_WEB_AUTH_TOKEN = os.environ.get("LIAGENT_WEB_TOKEN", "").strip()
_MAX_AUTH_PAYLOAD = 1024
_WS_AUTH_TIMEOUT_SEC = float(os.environ.get("LIAGENT_WS_AUTH_TIMEOUT", "5"))
_WEB_RATE_LIMIT = max(
    10, int(os.environ.get("LIAGENT_WEB_RATE_LIMIT_PER_MIN", "120"))
)
_RATE_WINDOW_SECONDS = 60
_http_rate: dict[str, list[float]] = {}
_ws_rate: dict[str, list[float]] = {}

# ─── IP / Origin allowlists ─────────────────────────────────────────────────
_ALLOWED_IPS: set[str] = {
    ip.strip()
    for ip in os.environ.get("LIAGENT_ALLOWED_IPS", "127.0.0.1,::1").split(",")
    if ip.strip()
}


def _parse_allowed_origins() -> set[str]:
    raw = os.environ.get("LIAGENT_ALLOWED_ORIGINS", "").strip()
    if raw:
        return {o.strip().rstrip("/") for o in raw.split(",") if o.strip()}
    origins: set[str] = set()
    for ip in _ALLOWED_IPS:
        if ip in ("127.0.0.1", "::1", "localhost"):
            origins.update(["http://127.0.0.1", "http://localhost", "http://[::1]"])
    return origins


_ALLOWED_ORIGINS: set[str] = _parse_allowed_origins()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _mask_secret(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 6:
        return "*" * len(secret)
    return f"{secret[:3]}***{secret[-2:]}"


def _allow_rate(bucket: dict[str, list[float]], client_id: str) -> bool:
    now = time.time()
    # Periodic cleanup of stale client entries
    if len(bucket) > 200:
        stale_cutoff = now - _RATE_WINDOW_SECONDS * 2
        stale_keys = [k for k, v in bucket.items() if not v or v[-1] < stale_cutoff]
        for k in stale_keys:
            del bucket[k]
    window = bucket.setdefault(client_id, [])
    while window and now - window[0] > _RATE_WINDOW_SECONDS:
        window.pop(0)
    if len(window) >= _WEB_RATE_LIMIT:
        return False
    window.append(now)
    return True


def _client_id_from_request(request: Request) -> str:
    return (request.client.host if request.client else "local") or "local"


def _authorized_http(request: Request) -> bool:
    if not _WEB_AUTH_TOKEN:
        return True
    token = request.headers.get("x-liagent-token", "").strip()
    return token == _WEB_AUTH_TOKEN


def _ip_allowed(ws: WebSocket) -> bool:
    """Check if the WebSocket client IP is in the whitelist."""
    client_ip = ws.client.host if ws.client else ""
    return client_ip in _ALLOWED_IPS


def _origin_allowed(ws: WebSocket) -> bool:
    if not _WEB_AUTH_TOKEN:
        return True
    if not _ALLOWED_ORIGINS:
        return True
    origin = (ws.headers.get("origin") or "").strip().rstrip("/")
    if not origin:
        return True  # native WS clients may omit Origin
    for allowed in _ALLOWED_ORIGINS:
        if origin == allowed or origin.startswith(allowed + ":"):
            return True
    return False


# ─── In-band WS authentication ──────────────────────────────────────────────

async def _authenticate_ws_inband(ws: WebSocket, ws_lock: asyncio.Lock) -> bool:
    """Wait for an auth message as the first WS frame; close on failure."""
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=_WS_AUTH_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        async with ws_lock:
            await ws.send_json({"type": "error", "text": "auth timeout"})
        await ws.close(code=1008, reason="auth timeout")
        return False
    except (WebSocketDisconnect, Exception):
        return False
    if len(raw) > _MAX_AUTH_PAYLOAD:
        async with ws_lock:
            await ws.send_json({"type": "error", "text": "auth payload too large"})
        await ws.close(code=1008, reason="payload too large")
        return False
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        async with ws_lock:
            await ws.send_json({"type": "error", "text": "invalid auth payload"})
        await ws.close(code=1008, reason="invalid payload")
        return False
    if not isinstance(data, dict) or data.get("type") != "auth":
        async with ws_lock:
            await ws.send_json({"type": "error", "text": "expected auth message"})
        await ws.close(code=1008, reason="expected auth")
        return False
    token = str(data.get("token", "")).strip()
    if token == _WEB_AUTH_TOKEN:
        async with ws_lock:
            await ws.send_json({"type": "auth_ok"})
        return True
    async with ws_lock:
        await ws.send_json({"type": "error", "text": "unauthorized"})
    await ws.close(code=1008, reason="unauthorized")
    return False


# ─── Push / chat client registries ──────────────────────────────────────────
_push_clients: set[WebSocket] = set()
_push_clients_lock = asyncio.Lock()
_push_client_sessions: dict[WebSocket, set[str]] = {}  # task-push WS -> subscribed session keys
_chat_clients: dict[WebSocket, asyncio.Lock] = {}  # chat WS -> ws_lock
_chat_client_sessions: dict[WebSocket, str | None] = {}  # chat WS -> current session key
