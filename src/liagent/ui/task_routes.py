"""Autonomous task REST + WebSocket endpoints — /api/tasks/*, /ws/task-push."""

import asyncio
import json

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from . import shared_state as _shared
from .shared_state import (
    _allow_rate,
    _authenticate_ws_inband,
    _authorized_http,
    _chat_clients,
    _chat_client_sessions,
    _client_id_from_request,
    _http_rate,
    _ip_allowed,
    _log,
    _origin_allowed,
    _push_clients,
    _push_client_sessions,
    _push_clients_lock,
    _WEB_AUTH_TOKEN,
)

router = APIRouter()


def _normalize_session_subscription(data) -> set[str]:
    """Normalize a subscribe_sessions payload into a concrete session key set."""
    if not isinstance(data, dict):
        return set()
    raw_keys = data.get("session_keys")
    if not isinstance(raw_keys, list):
        return set()
    return {
        str(item or "").strip()
        for item in raw_keys
        if str(item or "").strip()
    }


def _parse_approved_flag(value) -> bool:
    """Parse approved flag strictly so string 'false' does not become truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


async def broadcast_task_result(result: dict) -> int:
    """Broadcast a push-style event to all connected push + chat WebSocket clients."""
    dead: set[WebSocket] = set()
    delivered = 0
    msg_type = result.get("type", "task_result")
    target_session_id = str(result.get("target_session_id", "") or "").strip() or None
    # Push to dedicated /ws/task-push clients
    if not (msg_type == "proactive_suggestion" and target_session_id):
        async with _push_clients_lock:
            for ws in _push_clients:
                try:
                    await ws.send_json(result)
                    delivered += 1
                except Exception:
                    dead.add(ws)
            _push_clients.difference_update(dead)
            for ws in dead:
                _push_client_sessions.pop(ws, None)
    else:
        async with _push_clients_lock:
            for ws in _push_clients:
                subscriptions = _push_client_sessions.get(ws, set())
                if target_session_id not in subscriptions:
                    continue
                try:
                    await ws.send_json(result)
                    delivered += 1
                except Exception:
                    dead.add(ws)
            _push_clients.difference_update(dead)
            for ws in dead:
                _push_client_sessions.pop(ws, None)

    # Also push to active /ws/chat clients so web UI receives results
    dead_chat: list[WebSocket] = []
    if msg_type in {"heartbeat_confirm", "proactive_suggestion"}:
        # Forward special event cards as-is so frontend can render them directly.
        chat_msg = result
    else:
        chat_msg = {
            "type": "task_result",
            "task_id": result.get("task_id", ""),
            "task_name": result.get("task_name", ""),
            "run_id": result.get("run_id", ""),
            "result": result.get("result", ""),
            "status": result.get("status", ""),
            "error": result.get("error", ""),
            "trigger_event": result.get("trigger_event", ""),
        }
    for ws, lock in list(_chat_clients.items()):
        if target_session_id:
            ws_session = str(_chat_client_sessions.get(ws, "") or "").strip() or None
            if ws_session != target_session_id:
                continue
        try:
            async with lock:
                await ws.send_json(chat_msg)
                delivered += 1
        except Exception:
            dead_chat.append(ws)
    for ws in dead_chat:
        _chat_clients.pop(ws, None)
    return delivered


@router.websocket("/ws/task-push")
async def ws_task_push(ws: WebSocket):
    """Read-only WebSocket for receiving autonomous task result pushes."""
    if not _ip_allowed(ws):
        await ws.close(code=1008, reason="forbidden")
        return
    if not _origin_allowed(ws):
        _log.warning(f"ws/task-push rejected: origin={ws.headers.get('origin', '')} not allowed")
        await ws.close(code=1008, reason="forbidden origin")
        return
    await ws.accept()
    # In-band auth: first message must be {type:"auth", token:"..."}
    ws_lock = asyncio.Lock()
    if _WEB_AUTH_TOKEN:
        if not await _authenticate_ws_inband(ws, ws_lock):
            return
    async with _push_clients_lock:
        _push_clients.add(ws)
        _push_client_sessions[ws] = set()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "subscribe_sessions":
                subscriptions = _normalize_session_subscription(payload)
                async with _push_clients_lock:
                    if ws in _push_clients:
                        _push_client_sessions[ws] = subscriptions
                await ws.send_json(
                    {
                        "type": "subscribed",
                        "session_count": len(subscriptions),
                    }
                )
    except (WebSocketDisconnect, asyncio.CancelledError, Exception):
        pass
    finally:
        async with _push_clients_lock:
            _push_clients.discard(ws)
            _push_client_sessions.pop(ws, None)


@router.get("/api/tasks")
async def list_tasks(request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    from .web_server import app
    store = getattr(app.state, "task_store", None)
    if store is None:
        return JSONResponse({"error": "task system not initialized"}, status_code=503)
    include_system = request.query_params.get("include_system", "").lower() == "true"
    tasks = store.list_tasks(include_system=include_system)
    return JSONResponse({"tasks": tasks})


@router.post("/api/tasks")
async def create_task(data: dict, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    from .web_server import app
    store = getattr(app.state, "task_store", None)
    trigger_mgr = getattr(app.state, "trigger_manager", None)
    if store is None:
        return JSONResponse({"error": "task system not initialized"}, status_code=503)

    text = str(data.get("text", "")).strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    # Parse natural language into structured task
    from ..agent.task_parser import parse_task_description
    parsed = await parse_task_description(_shared._engine, text)
    if parsed is None:
        return JSONResponse({"error": "failed to parse task description"}, status_code=400)

    task = store.create_task(
        name=parsed["name"],
        trigger_type=parsed["trigger_type"],
        trigger_config=parsed.get("trigger_config", {}),
        prompt_template=parsed["prompt_template"],
        priority=int(data.get("priority", 10)),
    )

    # Register trigger
    if trigger_mgr:
        if parsed["trigger_type"] == "cron":
            schedule = parsed.get("trigger_config", {}).get("schedule", "")
            if schedule:
                await trigger_mgr.register_cron(task["id"], schedule)
        elif parsed["trigger_type"] == "once":
            delay = parsed.get("trigger_config", {}).get("delay_seconds", 60)
            await trigger_mgr.register_once(task["id"], delay)

    return JSONResponse({"task": task}, status_code=201)


@router.post("/api/tasks/{task_id}/pause")
async def pause_task(task_id: str, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    from .web_server import app
    store = getattr(app.state, "task_store", None)
    trigger_mgr = getattr(app.state, "trigger_manager", None)
    if store is None:
        return JSONResponse({"error": "task system not initialized"}, status_code=503)
    if not store.pause_task(task_id):
        return JSONResponse({"error": "task not found or not active"}, status_code=404)
    if trigger_mgr:
        await trigger_mgr.unregister(task_id)
    return JSONResponse({"status": "paused"})


@router.post("/api/tasks/{task_id}/resume")
async def resume_task(task_id: str, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    from .web_server import app
    store = getattr(app.state, "task_store", None)
    trigger_mgr = getattr(app.state, "trigger_manager", None)
    if store is None:
        return JSONResponse({"error": "task system not initialized"}, status_code=503)
    if not store.resume_task(task_id):
        return JSONResponse({"error": "task not found or not paused"}, status_code=404)
    if trigger_mgr:
        await trigger_mgr.reload_task(task_id)
    return JSONResponse({"status": "active"})


@router.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str, request: Request):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    from .web_server import app
    store = getattr(app.state, "task_store", None)
    trigger_mgr = getattr(app.state, "trigger_manager", None)
    if store is None:
        return JSONResponse({"error": "task system not initialized"}, status_code=503)
    if not store.delete_task(task_id):
        return JSONResponse({"error": "task not found"}, status_code=404)
    if trigger_mgr:
        await trigger_mgr.unregister(task_id)
    return JSONResponse({"status": "deleted"})


@router.get("/api/tasks/{task_id}/runs")
async def get_task_runs(task_id: str, request: Request, limit: int = 10):
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)
    from .web_server import app
    store = getattr(app.state, "task_store", None)
    if store is None:
        return JSONResponse({"error": "task system not initialized"}, status_code=503)
    runs = store.get_recent_runs(task_id, limit=max(1, min(limit, 50)))
    return JSONResponse({"runs": runs})


@router.post("/api/tasks/confirm")
async def confirm_heartbeat_action(data: dict, request: Request):
    """Confirm or reject a pending heartbeat action."""
    if not _authorized_http(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not _allow_rate(_http_rate, _client_id_from_request(request)):
        return JSONResponse({"error": "rate limited"}, status_code=429)

    from .web_server import app
    store = getattr(app.state, "task_store", None)
    executor = getattr(app.state, "executor", None)
    hb_task_id = getattr(app.state, "heartbeat_task_id", None)
    if store is None or executor is None:
        return JSONResponse({"error": "task system not initialized"}, status_code=503)

    token = str(data.get("token", ""))
    approved = _parse_approved_flag(data.get("approved", False))

    if not token:
        return JSONResponse({"error": "token is required"}, status_code=400)
    if approved and not hb_task_id:
        return JSONResponse({"error": "heartbeat task not initialized"}, status_code=503)

    # Check expiry before accepting confirmation
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        run_row = conn.execute(
            "SELECT status, expires_at FROM autonomous_task_runs WHERE id = ?",
            (token,),
        ).fetchone()
    if not run_row or run_row["status"] != "pending_confirm":
        return JSONResponse({"error": "already_resolved"}, status_code=409)
    if run_row["expires_at"]:
        from ..agent.time_utils import _now_local
        if run_row["expires_at"] < _now_local().isoformat():
            store.transition_run(token, "pending_confirm", "expired")
            return JSONResponse({"error": "confirmation expired"}, status_code=410)

    to_status = "confirmed" if approved else "rejected"
    ok = store.transition_run(token, "pending_confirm", to_status)
    if not ok:
        return JSONResponse({"error": "already_resolved"}, status_code=409)

    if approved and hb_task_id:
        # Retrieve the run to get prompt and reconstruct budget
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT prompt, trigger_event FROM autonomous_task_runs WHERE id = ?",
                (token,),
            ).fetchone()
        if row:
            prompt = row["prompt"] or ""
            trigger_event = row["trigger_event"] or ""
            from ..agent.task_queue import _heartbeat_budget_from_prompt
            budget = _heartbeat_budget_from_prompt(prompt)
            executor.enqueue(
                token, hb_task_id, prompt,
                priority=8, trigger_event=trigger_event,
                budget=budget,
                source="system",
            )

    return JSONResponse({
        "status": to_status,
        "run_id": token,
    })
