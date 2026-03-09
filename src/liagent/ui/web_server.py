"""FastAPI backend — WebSocket for chat, REST for config/audio.

Route modules:
- config_routes.py    — /api/config/*, /api/health, /api/metrics, /api/policy, /api/tool
- interest_routes.py  — /api/interests/* (watch system)
- task_routes.py      — /api/tasks/*, /ws/task-push
- vision_routes.py    — image decode/normalize/cache helpers
"""

import asyncio
import base64
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

class _MissingNumpy:
    """Lazy numpy placeholder so module import works in minimal test envs."""

    ndarray = Any

    def __getattr__(self, _name: str) -> Any:
        raise ModuleNotFoundError(
            "numpy is required for audio processing paths in liagent.ui.web_server"
        )


try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CI env only
    np = _MissingNumpy()  # type: ignore[assignment]

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from ..logging import get_logger
from ..agent.self_supervision import InteractionMetrics
from ..engine.engine_manager import EngineManager
from ..agent.brain import AgentBrain
from ..tools import get_all_tools

# ─── Shared state (canonical location) ──────────────────────────────────────
from .shared_state import (
    _allow_rate,
    _authenticate_ws_inband,
    _authorized_http,
    _BRAIN_RUN_LOCK,
    _chat_clients,
    _chat_client_sessions,
    _client_id_from_request,
    _http_rate,
    _ip_allowed,
    _log,
    _mask_secret,
    _origin_allowed,
    _WEB_AUTH_TOKEN,
    _ws_rate,
)
import liagent.ui.shared_state as _shared

# ─── Vision helpers ─────────────────────────────────────────────────────────
from .vision_routes import (
    _cleanup_path,
    _decode_and_prepare_image,
    _resolve_image_paths,
)
from .event_envelope import (
    _attach_event_meta,
    _event_payload_tail,
    _extract_agent_event,
)
from .runtime_status import (
    RuntimeStatusSnapshot,
    apply_legacy_event_to_runtime_status as _apply_runtime_event,
    format_running_status_message as _shared_format_running_status_message,
    init_runtime_status as _init_runtime_status,
    mark_runtime_cancel_requested as _mark_runtime_cancel_requested,
    mark_runtime_status_pushed as _mark_runtime_status_pushed,
    should_push_runtime_status as _should_push_runtime_status,
    update_runtime_status as _update_runtime_status,
)

# ─── Route modules ──────────────────────────────────────────────────────────
from .config_routes import router as _config_router
from .interest_routes import router as _interest_router
from .task_routes import router as _task_router, broadcast_task_result, _parse_approved_flag  # noqa: F401

_TTS_BREAK_CHARS = ".!?;,\n"
_SESSION_SUGGESTION_RESHOW_SEC = 300
_RUNNING_STATUS_RE = re.compile(
    r"^(?:\?+|then\??|status|continue|what now\??|then what\??|"
    r"any update\??|still waiting\??|are you there\??|how long\??)\s*$",
    re.IGNORECASE,
)

STATIC_DIR = Path(__file__).parent / "static"


def _is_running_status_followup(user_input: str) -> bool:
    text = str(user_input or "").strip()
    if not text:
        return False
    if _RUNNING_STATUS_RE.match(text):
        return True
    return len(text) <= 24 and text.lower() in {"?", "then?", "status", "continue"}


def _format_status_tool(tool_name: str, tool_args: dict[str, Any] | None = None) -> str:
    name = str(tool_name or "").strip() or "unknown"
    args = tool_args if isinstance(tool_args, dict) else {}
    if not args:
        return f"`{name}`"
    parts: list[str] = []
    for key, value in list(args.items())[:3]:
        rendered = str(value)
        if len(rendered) > 48:
            rendered = rendered[:45] + "..."
        parts.append(f"{key}={rendered}")
    return f"`{name}({', '.join(parts)})`"


def _format_status_wait(seconds: float) -> str:
    try:
        secs = max(0, int(round(float(seconds or 0.0))))
    except Exception:
        secs = 0
    if secs <= 1:
        return "less than 1 second"
    if secs < 60:
        return f"about {secs} seconds"
    mins, rem = divmod(secs, 60)
    if mins < 60:
        if rem < 5:
            return f"about {mins} minutes"
        return f"about {mins} minutes {rem} seconds"
    hours, mins = divmod(mins, 60)
    if mins == 0:
        return f"about {hours} hours"
    return f"about {hours} hours {mins} minutes"


def _format_running_status_message(status: RuntimeStatusSnapshot | dict[str, Any] | None) -> str:
    return _shared_format_running_status_message(status)


def _build_tool_result_preview(result: Any, *, max_chars: int = 500) -> tuple[str, bool, int]:
    text = str(result or "")
    total_chars = len(text)
    if total_chars <= max_chars:
        return text, False, total_chars
    preview = text[:max_chars].rstrip()
    if preview:
        preview += "\n...(tool result preview truncated)"
    else:
        preview = "...(tool result preview truncated)"
    return preview, True, total_chars


def _in_quiet_hours(brain: AgentBrain | None) -> bool:
    if brain is None:
        return False
    try:
        from ..agent.behavior import is_quiet_hours
        quiet_hours = getattr(getattr(brain, "_proactive_config", None), "quiet_hours", "")
        return bool(quiet_hours) and is_quiet_hours(quiet_hours)
    except Exception:
        return False


async def _broadcast_pending_suggestion(brain: AgentBrain, suggestion: dict[str, Any]) -> bool:
    msg = {
        "type": "proactive_suggestion",
        "message": suggestion["message"],
        "suggestion_id": suggestion["id"],
        "domain": suggestion.get("domain", "general"),
        "suggestion_type": suggestion.get("suggestion_type", "watch"),
        "target_session_id": suggestion.get("target_session_id"),
    }
    delivered = await broadcast_task_result(msg)
    if delivered <= 0:
        return False
    brain.suggestion_store.mark_shown(
        suggestion["id"], cooldown_sec=_SESSION_SUGGESTION_RESHOW_SEC,
    )
    if brain.proactive_router and hasattr(brain.proactive_router, '_domain_feedback'):
        brain.proactive_router._domain_feedback.record_suggested(
            suggestion.get("domain", "general"), suggestion.get("suggestion_type", "watch"),
        )
    return True


async def _apply_suggestion_feedback(
    suggestion_id: int, action: str, *, session_id: str | None = None,
) -> None:
    store = getattr(_shared._brain, 'suggestion_store', None)
    router = getattr(_shared._brain, 'proactive_router', None)
    if store is None:
        return

    sug_record = store.get_suggestion(int(suggestion_id))
    target_session_id = (
        str(sug_record.get("target_session_id", "") or "").strip()
        if sug_record else ""
    )
    caller_session_id = str(session_id or "").strip()
    if target_session_id and target_session_id != caller_session_id:
        return

    if action == "dismiss":
        if not store.transition_status(int(suggestion_id), "pending", "dismissed"):
            return
        domain = sug_record["domain"] if sug_record else "general"
        stype = sug_record["suggestion_type"] if sug_record else "watch"
        if router:
            outcome = getattr(router._domain_feedback, "record_rejected_outcome", None)
            if callable(outcome):
                outcome(domain, stype)
            else:
                router._domain_feedback.record_rejected(domain, stype)
        return

    if not store.transition_status(int(suggestion_id), "pending", "processing"):
        return

    domain = sug_record["domain"] if sug_record else "general"
    stype = sug_record["suggestion_type"] if sug_record else "watch"

    materialized = False
    if sug_record:
        import json as _json
        try:
            payload = _json.loads(sug_record.get("action_json", "{}"))
        except (ValueError, TypeError):
            payload = {}
        goal_store = getattr(_shared, '_goal_store', None)
        task_store = getattr(_shared, '_task_store', None)
        trigger_mgr = getattr(_shared, '_trigger_mgr', None)
        try:
            if payload.get("create_goal") and goal_store:
                goal_id = goal_store.create(
                    idempotency_key=payload.get("idempotency_key"),
                    source="user_accepted",
                    domain=payload.get("domain", domain),
                    objective=payload.get("objective", sug_record.get("message", "")),
                    rationale=payload.get("rationale", ""),
                    confidence=sug_record.get("confidence", 0.5),
                    priority=payload.get("priority", 0),
                    budget_json=(
                        _json.dumps(payload.get("budget"), ensure_ascii=False)
                        if payload.get("budget") else None
                    ),
                    success_criteria_json=(
                        _json.dumps(payload.get("success_criteria"), ensure_ascii=False)
                        if payload.get("success_criteria") else None
                    ),
                    source_group_id=payload.get("source_group_id"),
                    source_discovery_id=payload.get("source_discovery_id"),
                )
                if goal_id is not None:
                    goal_store.transition(goal_id, "active")
                    goal_store.record_event(
                        goal_id, "created",
                        {"source": "user_accepted", "suggestion_id": int(suggestion_id)},
                    )
                materialized = True
            elif task_store:
                prompt = payload.get("prompt", sug_record.get("message", ""))
                dedup_key = payload.get("dedup_key") or f"suggestion:{suggestion_id}"
                task = task_store.create_task_from_prompt(
                    prompt=prompt,
                    source="user_accepted",
                    goal_id=payload.get("goal_id"),
                    dedup_key=dedup_key,
                    risk_level=payload.get("risk", "low"),
                )
                if task and trigger_mgr:
                    await trigger_mgr.register_once(task["id"], delay_seconds=0)
                materialized = True
        except Exception as mat_err:
            _log.error("web_server", mat_err, action="suggestion_materialize")

    store.update_status(int(suggestion_id), "accepted" if materialized else "failed")
    if materialized and router:
        outcome = getattr(router._domain_feedback, "record_accepted_outcome", None)
        if callable(outcome):
            outcome(domain, stype)
        else:
            router._domain_feedback.record_accepted(domain, stype)


async def _suggestion_push_loop(interval: float = 10.0):
    """Background poller: push new session suggestions to connected WS clients."""
    while True:
        await asyncio.sleep(interval)
        try:
            brain = _shared._brain
            if brain is None or not hasattr(brain, 'suggestion_store') or not brain.suggestion_store:
                continue
            if _in_quiet_hours(brain):
                continue
            pending = brain.suggestion_store.get_pending(max_items=2)
            if not pending:
                continue
            for sug in pending:
                await _broadcast_pending_suggestion(brain, sug)
        except Exception:
            pass  # never crash the background poller


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Startup: launch suggestion push poller
    _suggestion_task = asyncio.create_task(_suggestion_push_loop())
    # Startup: launch signal poller if configured
    if _shared._signal_poller is not None:
        try:
            await _shared._signal_poller.start()
        except Exception as e:
            _log.error("web_server", e, action="signal_poller_start")
    yield
    # Shutdown: stop suggestion push poller
    _suggestion_task.cancel()
    try:
        await _suggestion_task
    except asyncio.CancelledError:
        pass
    # Shutdown: stop signal poller, then flush anomaly detector
    if _shared._signal_poller is not None:
        try:
            await _shared._signal_poller.stop()
        except Exception as e:
            _log.error("web_server", e, action="signal_poller_stop")
    if _shared._anomaly_detector is not None:
        try:
            await _shared._anomaly_detector.shutdown()
        except Exception as e:
            _log.error("web_server", e, action="anomaly_detector_shutdown")
    # Shutdown: clean up orchestrator and brain (finalize_session extracts last facts)
    if _shared._orchestrator is not None:
        try:
            await _shared._orchestrator.shutdown()
        except Exception:
            pass
    if _shared._brain is not None:
        try:
            await _shared._brain.shutdown()
        except Exception as e:
            _log.error("web_server", e, action="brain_shutdown")
    # Shutdown: project knowledge events using a fresh EventLog handle
    if _shared._brain is not None:
        try:
            from ..knowledge.projector import project_pending_events
            project_pending_events()
        except Exception as e:
            _log.error("web_server", e, action="knowledge_projection")


app = FastAPI(title="LiAgent", docs_url=None, redoc_url=None, lifespan=_lifespan)

# Include extracted routers (no prefix — routes carry full paths)
app.include_router(_config_router)
app.include_router(_interest_router)
app.include_router(_task_router)


class AppContext:
    """Encapsulates all shared server state, replacing module-level globals."""

    def __init__(self, engine: EngineManager, brain: AgentBrain):
        self.engine = engine
        self.brain = brain
        self.metrics = InteractionMetrics()
        self.run_lock = asyncio.Lock()
        self._start_time = time.time()



# ─── Backward-compat module-level aliases ────────────────────────────────────

def create_app(engine: EngineManager, brain: AgentBrain) -> FastAPI:
    # Mutate shared_state globals so all route modules see the same objects
    _shared._engine = engine
    _shared._brain = brain
    if _shared._BRAIN_RUN_LOCK is None:
        _shared._BRAIN_RUN_LOCK = asyncio.Lock()
    # Create orchestrator for multi-agent routing
    try:
        from ..orchestrator.orchestrator import Orchestrator
        _shared._orchestrator = Orchestrator(engine=engine, brain=brain)
    except Exception:
        _shared._orchestrator = None
    # Create attention system: Layer 0 (SignalPoller) → Enricher → Layer 1 (AnomalyDetector)
    try:
        from ..agent.interest import InterestStore
        from ..agent.signal_poller import SignalPoller
        from ..agent.anomaly_detector import AnomalyDetector
        from ..agent.signal_enricher import SignalEnricher
        store = InterestStore()
        enricher = SignalEnricher(engine)
        # Layer 1: anomaly detector sits between poller and Discord
        _shared._anomaly_detector = AnomalyDetector(
            on_anomaly=_on_anomaly_event,
            on_signal_passthrough=_on_signal_passthrough,
        )
        # Layer 0: signals flow through enricher then into anomaly detector
        _shared._signal_poller = SignalPoller(
            store, on_signal=_shared._anomaly_detector.ingest,
            enricher=enricher,
        )
    except Exception as e:
        _log.error("web_server", e, action="attention_system_init")
        _shared._signal_poller = None
        _shared._anomaly_detector = None
    # Set up AppContext for new-style access
    ctx = AppContext(engine, brain)
    app.state.ctx = ctx
    _shared._metrics = ctx.metrics
    return app


async def _post_to_discord_thread(thread_id: str, msg: str):
    """Send a message to a Discord thread with retry on transient failures."""
    try:
        import discord as _discord
    except ImportError:
        _discord = None  # type: ignore[assignment]

    try:
        from .discord_bot import get_bot_instance
        bot = get_bot_instance()
        if bot is None:
            return
        thread = bot.get_channel(int(thread_id))
        if thread is None:
            return
    except Exception as e:
        _log.error("discord_post_resolve", e, thread_id=thread_id)
        return

    # Discord message limit: 2000 chars
    chunks = [msg[i : i + 2000] for i in range(0, max(len(msg), 1), 2000)]

    last_exc = None
    for attempt in range(3):
        try:
            for chunk in chunks:
                await thread.send(chunk)
            return
        except asyncio.TimeoutError as e:
            last_exc = e
        except Exception as e:
            if _discord is not None and isinstance(e, _discord.HTTPException) and e.status >= 500:
                last_exc = e
            else:
                _log.error("discord_post", e, thread_id=thread_id)
                return
        if attempt < 2:
            await asyncio.sleep(2 ** (attempt + 1))  # 2s, 4s
    if last_exc:
        _log.error("discord_post_retries_exhausted", last_exc, thread_id=thread_id)


async def _on_anomaly_event(anomaly: dict):
    """Callback from AnomalyDetector — confirmed cross-factor anomaly."""
    # Record for conversation context visibility
    if _shared._brain is not None:
        score = anomaly.get("score", 0)
        summary = anomaly.get("summary", "Anomaly detected")
        _shared._brain.record_system_activity(f"Anomaly (score {score:.1f}): {summary}")
    # Forward to heartbeat pre-check if wired
    if _shared._anomaly_precheck is not None:
        try:
            _shared._anomaly_precheck.on_anomaly(anomaly)
        except Exception:
            pass
    thread_id = anomaly.get("discord_thread_id")
    if not thread_id:
        return

    score = anomaly.get("score", 0)
    factor_count = anomaly.get("factor_count", 1)
    signal_count = anomaly.get("signal_count", 1)
    summary = anomaly.get("summary", "Anomaly detected")

    severity = "\U0001f534" if score >= 2.5 else "\U0001f7e0" if score >= 1.8 else "\u26a0\ufe0f"
    msg = (
        f"{severity} **Anomaly Detected** (score {score:.1f})\n"
        f"> {summary}\n"
        f"_{factor_count} factor(s), {signal_count} signal(s)_"
    )
    await _post_to_discord_thread(thread_id, msg)


async def _on_signal_passthrough(signal: dict):
    """Callback from AnomalyDetector — sub-threshold individual signal."""
    # Record for conversation context visibility
    if _shared._brain is not None:
        factor_name = signal.get("factor_name", "Unknown")
        delta = signal.get("delta", {})
        if "pct_change" in delta:
            _shared._brain.record_system_activity(
                f"Signal: {factor_name} {delta['pct_change']:+.2f}%"
            )
        elif delta.get("type") == "content_changed":
            _shared._brain.record_system_activity(f"Signal: {factor_name} content changed")
    thread_id = signal.get("discord_thread_id")
    if not thread_id:
        return

    factor_name = signal.get("factor_name", "Unknown")
    delta = signal.get("delta", {})

    # Enriched path: use key_fact + sentiment icon
    key_fact = delta.get("key_fact")
    if key_fact:
        sentiment = delta.get("sentiment", 0)
        if sentiment > 0.3:
            icon = "\U0001f7e2"  # green circle
        elif sentiment < -0.3:
            icon = "\U0001f534"  # red circle
        else:
            icon = "\U0001f7e1"  # yellow circle
        msg = f"{icon} **{factor_name}**: {key_fact}"
        await _post_to_discord_thread(thread_id, msg)
        return

    # Legacy path
    raw = delta.get("raw_delta", delta)
    if "pct_change" in raw:
        arrow = "\u2191" if raw["pct_change"] > 0 else "\u2193"
        msg = (
            f"{arrow} **{factor_name}**: "
            f"${raw.get('new_price', '?'):.2f} "
            f"({raw['pct_change']:+.2f}%) "
            f"from ${raw.get('prev_price', '?'):.2f}"
        )
    elif raw.get("type") == "content_changed":
        snippet = raw.get("snippet", "")[:150]
        msg = f"\U0001f4f0 **{factor_name}**: Content changed\n> {snippet}"
    else:
        msg = f"**{factor_name}**: Signal detected"

    await _post_to_discord_thread(thread_id, msg)


# ─── Pages ──────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    html_path = STATIC_DIR / "index.html"
    html = html_path.read_text(encoding="utf-8")
    # Inject auth token so the browser JS can use it for WebSocket
    if _WEB_AUTH_TOKEN:
        inject = (
            f'<script>window.__LIAGENT_TOKEN__="{_WEB_AUTH_TOKEN}";</script>'
        )
        html = html.replace("</head>", inject + "</head>", 1)
    return HTMLResponse(html)


@app.get("/static/{path:path}")
async def static_file(path: str):
    file_path = (STATIC_DIR / path).resolve()
    if not file_path.is_relative_to(STATIC_DIR.resolve()):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if file_path.exists():
        return FileResponse(file_path)
    return JSONResponse({"error": "not found"}, status_code=404)


# ─── TTS chunking helpers ──────────────────────────────────────────────────

def _find_tts_breakpoint(text: str, hard_limit: int, min_len: int) -> int:
    """Find a natural break point for low-latency TTS chunking."""
    if hard_limit <= 0:
        return -1
    head = text[:hard_limit]
    for i in range(len(head) - 1, -1, -1):
        if head[i] in _TTS_BREAK_CHARS and (i + 1) >= min_len:
            return i + 1
    ws = head.rfind(" ")
    if ws >= min_len:
        return ws + 1
    return -1


def _pop_realtime_tts_chunk(
    text: str,
    *,
    first_chunk: bool,
    first_target: int,
    next_target: int,
    force: bool,
) -> tuple[str, str]:
    """Pop one TTS chunk from pending text with natural-boundary preference."""
    if not text:
        return "", ""
    target = first_target if first_chunk else next_target
    min_len = 18 if first_chunk else max(32, target // 3)
    if not force and len(text) < min_len:
        return "", text

    hard_limit = min(len(text), target)
    cut = _find_tts_breakpoint(text, hard_limit=hard_limit, min_len=min_len)
    if cut <= 0:
        if force:
            cut = len(text)
        elif len(text) >= target:
            cut = target
        else:
            return "", text

    chunk = text[:cut].strip()
    rest = text[cut:].lstrip()
    return chunk, rest


# ─── Streaming TTS helper ──────────────────────────────────────────────────

async def _stream_events(
    ws: WebSocket,
    brain: AgentBrain,
    text: str,
    run_id: str,
    *,
    voice_triggered: bool = False,
    images: list[str] | None = None,
    session_key: str | None = None,
    cancel_scope=None,
    ws_lock: asyncio.Lock | None = None,
    status_tracker: dict[str, Any] | None = None,
):
    """Stream brain events over WebSocket with low-latency incremental TTS."""
    _engine_ref = _shared._engine
    _metrics_ref = _shared._metrics

    # Free 30B Coder when entering voice mode — only 4B VLM needed
    if voice_triggered:
        _engine_ref.unload_reasoning_llm()

    tts_enabled = (
        _engine_ref.config.tts_enabled
        and _engine_ref.tts is not None
    )
    run_started = time.perf_counter()
    queue_wait_started: float | None = None
    stream_started: float | None = None
    queued_ms = 0.0
    tts_total_ms = 0.0
    final_state = "accepted"
    full_answer = ""
    pending_tts = ""
    tts_chunks_emitted = 0
    stream_start = time.perf_counter()
    first_tts_ms: float | None = None
    tts_profile_sent = False
    tts_failed_reason = ""
    tts_cfg = _engine_ref.config.tts
    max_chars = max(80, int(tts_cfg.max_chunk_chars))
    first_target = min(max_chars, max(40, int(os.environ.get("LIAGENT_TTS_FIRST_CHUNK_CHARS", "72"))))
    next_target = min(max_chars, max(72, int(os.environ.get("LIAGENT_TTS_STREAM_CHUNK_CHARS", "150"))))
    status_push_after_sec = max(
        0.05, float(os.environ.get("LIAGENT_WS_STATUS_PUSH_AFTER_SEC", "8.0"))
    )
    status_push_repeat_sec = max(
        status_push_after_sec,
        float(os.environ.get("LIAGENT_WS_STATUS_PUSH_REPEAT_SEC", "15.0")),
    )
    queue_wait_timeout_sec = max(
        0.05, float(os.environ.get("LIAGENT_WS_QUEUE_WAIT_TIMEOUT_SEC", "120.0"))
    )
    status_idle_floor_sec = max(
        0.05, float(os.environ.get("LIAGENT_WS_STATUS_IDLE_FLOOR_SEC", "5.0"))
    )
    status_tick_sec = max(
        0.01, float(os.environ.get("LIAGENT_WS_STATUS_TICK_SEC", "1.0"))
    )
    run_lock_acquired = False
    pending_event_task: asyncio.Task | None = None
    pending_lock_task: asyncio.Task | None = None

    async def _send(msg: dict):
        payload = dict(msg)
        payload.setdefault("run_id", run_id)
        try:
            if ws_lock:
                async with ws_lock:
                    await ws.send_json(payload)
            else:
                await ws.send_json(payload)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            # Client disconnected or task cancelled; stop stream immediately
            raise asyncio.CancelledError

    async def _send_best_effort(msg: dict) -> bool:
        payload = dict(msg)
        payload.setdefault("run_id", run_id)
        try:
            await _send(payload)
            return True
        except (Exception, asyncio.CancelledError):
            try:
                if ws_lock:
                    async with ws_lock:
                        await ws.send_json(payload)
                else:
                    await ws.send_json(payload)
                return True
            except (Exception, asyncio.CancelledError):
                return False

    def _track_status(**updates: Any) -> None:
        _update_runtime_status(status_tracker, **updates)

    async def _maybe_send_auto_status(*, force: bool = False) -> None:
        if status_tracker is None:
            return
        state = str(status_tracker.get("state") or status_tracker.get("phase") or "").strip()
        if state in {"done", "error", "cancelled"}:
            return
        now = time.time()
        if not force:
            if not _should_push_runtime_status(
                status_tracker,
                now=now,
                after_sec=status_push_after_sec,
                idle_floor_sec=status_idle_floor_sec,
                repeat_sec=status_push_repeat_sec,
            ):
                return
        _mark_runtime_status_pushed(status_tracker, now=now)
        await _send(
            {
                "type": "status_message",
                "text": _format_running_status_message(status_tracker),
                "run_state": state or "streaming",
            }
        )

    async def _send_audio_chunk(audio: np.ndarray):
        """Encode and send a single audio chunk over WebSocket."""
        nonlocal tts_chunks_emitted, first_tts_ms
        if audio.size == 0:
            return
        tts_chunks_emitted += 1
        if first_tts_ms is None:
            first_tts_ms = (time.perf_counter() - stream_start) * 1000.0
        audio32 = audio if audio.dtype == np.float32 else audio.astype(np.float32, copy=False)
        audio32 = np.ascontiguousarray(audio32)
        peak = float(np.max(np.abs(audio32))) if audio32.size > 0 else 0.0
        sample_rate = int(getattr(_engine_ref.tts, "sample_rate", 24000) or 24000)
        audio_b64 = base64.b64encode(memoryview(audio32).cast("B")).decode()
        await _send({
            "type": "tts_chunk",
            "audio": audio_b64,
            "sample_rate": sample_rate,
            "samples": int(audio32.size),
            "peak": round(peak, 4),
        })

    async def _synthesize_and_send(text: str):
        """Synthesize a text chunk and send the audio over WebSocket.

        Uses streaming synthesis when available (Qwen3-TTS) for lower latency.
        """
        nonlocal tts_chunks_emitted, first_tts_ms, tts_profile_sent, tts_total_ms, tts_failed_reason
        if not text or len(text.strip()) < 2:
            return
        if tts_failed_reason:
            return
        t0 = time.perf_counter()
        try:
            if not tts_profile_sent:
                if _engine_ref.config.tts.backend == "api":
                    profile = f"api:{_engine_ref.config.tts.api_voice}"
                else:
                    speaker = str(
                        getattr(_engine_ref.config.tts, "speaker_name", "") or "default"
                    )
                    profile = f"qwen3-tts:{speaker}"
                await _send(
                    {"type": "tts_profile", "profile": profile}
                )
                tts_profile_sent = True

            # Use streaming synthesis if available (Qwen3-TTS)
            if hasattr(_engine_ref.tts, "synthesize_stream"):
                stream = _engine_ref.tts.synthesize_stream(text)
                try:
                    async for chunk in stream:
                        await _send_audio_chunk(chunk)
                finally:
                    # Ensure generator cleanup to release MLX resources
                    await stream.aclose()
            else:
                audio = await _engine_ref.tts.synthesize(text)
                await _send_audio_chunk(audio)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            tts_failed_reason = str(e) or "tts_synthesis_failed"
            _log.error("tts_synthesis_error", error=str(e))
        finally:
            tts_total_ms += (time.perf_counter() - t0) * 1000.0

    async def _drain_tts(*, force: bool):
        nonlocal pending_tts
        if not tts_enabled:
            return
        while pending_tts and not tts_failed_reason:
            chunk, rest = _pop_realtime_tts_chunk(
                pending_tts,
                first_chunk=(tts_chunks_emitted == 0),
                first_target=first_target,
                next_target=next_target,
                force=force,
            )
            if not chunk:
                break
            pending_tts = rest
            await _synthesize_and_send(chunk)
            force = False

    try:
        final_state = "accepted"
        _track_status(state="accepted", phase="accepted", confirmation_pending=False)
        await _send({"type": "run_state", "state": "accepted"})
        if images:
            await _send({"type": "vision_input", "count": len(images)})
        if _shared._BRAIN_RUN_LOCK is None:
            raise RuntimeError("run lock not initialized")
        queue_wait_started = time.perf_counter()
        final_state = "queued"
        _track_status(state="queued", phase="queued")
        await _send({"type": "run_state", "state": "queued"})
        # Preempt any running background task before acquiring lock
        executor = getattr(app.state, "executor", None)
        if executor is not None:
            await executor.preempt()
        pending_lock_task = asyncio.create_task(_shared._BRAIN_RUN_LOCK.acquire())
        while True:
            try:
                await asyncio.wait_for(
                    asyncio.shield(pending_lock_task),
                    timeout=status_tick_sec,
                )
                run_lock_acquired = True
                pending_lock_task = None
                break
            except asyncio.TimeoutError:
                if queue_wait_started is not None:
                    queued_elapsed = time.perf_counter() - queue_wait_started
                    if queued_elapsed >= queue_wait_timeout_sec:
                        raise TimeoutError(
                            f"brain queue wait timed out after {queue_wait_timeout_sec:.2f}s"
                        )
                await _maybe_send_auto_status()
                continue
        try:
            queued_ms = (
                (time.perf_counter() - queue_wait_started) * 1000.0
                if queue_wait_started is not None
                else 0.0
            )
            stream_started = time.perf_counter()
            final_state = "streaming"
            _track_status(state="streaming", phase="streaming")
            await _send({"type": "run_state", "state": "streaming"})
        # Stream brain/orchestrator events
            if _shared._orchestrator is not None:
                _event_source = _shared._orchestrator.dispatch(
                    text,
                    images=images,
                    low_latency=voice_triggered,
                    session_key=session_key,
                    cancel_scope=cancel_scope,
                )
            else:
                _event_source = brain.run(
                    text,
                    images=images,
                    low_latency=voice_triggered,
                    session_key=session_key,
                    cancel_scope=cancel_scope,
                )
            event_iter = _event_source.__aiter__()
            pending_event_task = asyncio.create_task(event_iter.__anext__())
            while True:
                try:
                    raw_event = await asyncio.wait_for(
                        asyncio.shield(pending_event_task),
                        timeout=status_tick_sec,
                    )
                except asyncio.TimeoutError:
                    await _maybe_send_auto_status()
                    continue
                except StopAsyncIteration:
                    pending_event_task = None
                    break
                pending_event_task = asyncio.create_task(event_iter.__anext__())
                event, event_meta = _extract_agent_event(raw_event)
                etype = event[0]
                if etype == "think":
                    _track_status(state="streaming", phase="thinking")
                    await _send(
                        _attach_event_meta(
                            {"type": "think", "text": event[1]},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "token":
                    _track_status(state="streaming", phase="streaming")
                    await _send(
                        _attach_event_meta(
                            {"type": "token", "text": event[1]},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                    if tts_enabled:
                        full_answer += event[1]
                        pending_tts += event[1]
                        await _drain_tts(force=False)
                elif etype == "bridge_tts":
                    # Immediate bridge phrase — synthesize and push before tool work
                    bridge_text = event[1]
                    if tts_enabled and bridge_text:
                        await _send(
                            _attach_event_meta(
                                {"type": "bridge_tts", "text": bridge_text},
                                event_type=etype,
                                event_meta=event_meta,
                            )
                        )
                        await _synthesize_and_send(bridge_text)
                elif etype == "tool_start":
                    _apply_runtime_event(status_tracker, etype, event[1], event[2])
                    await _send(
                        _attach_event_meta(
                            {
                                "type": "tool_start",
                                "name": event[1],
                                "args": event[2],
                            },
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "tool_result":
                    result_preview, result_truncated, total_chars = _build_tool_result_preview(
                        event[2], max_chars=500,
                    )
                    _apply_runtime_event(status_tracker, etype, event[1], event[2])
                    await _send(
                        _attach_event_meta(
                            {
                                "type": "tool_result",
                                "name": event[1],
                                "result": result_preview,
                                "truncated": result_truncated,
                                "total_chars": total_chars,
                                "full_result": str(event[2] or "") if result_truncated else "",
                            },
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "tool_error":
                    payload = event[2] if len(event) > 2 and isinstance(event[2], dict) else {}
                    _apply_runtime_event(status_tracker, etype, event[1], payload)
                    await _send(
                        _attach_event_meta(
                            {
                                "type": "tool_error",
                                "name": event[1],
                                "args": payload.get("tool_args", {}),
                                "error_type": payload.get("error_type", ""),
                                "error": payload.get("message", ""),
                            },
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "tool_fallback":
                    payload = event[3] if len(event) > 3 and isinstance(event[3], dict) else {}
                    _apply_runtime_event(status_tracker, etype, event[1], event[2], payload)
                    await _send(
                        _attach_event_meta(
                            {
                                "type": "tool_fallback",
                                "requested_name": event[1],
                                "effective_name": event[2],
                                "requested_args": payload.get("requested_args", {}),
                                "effective_args": payload.get("effective_args", {}),
                            },
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "tool_skip":
                    payload = event[3] if len(event) > 3 and isinstance(event[3], dict) else {}
                    _apply_runtime_event(status_tracker, etype, event[1], event[2], payload)
                    await _send(
                        _attach_event_meta(
                            {
                                "type": "tool_skip",
                                "name": event[1],
                                "reason": event[2],
                                "detail": payload,
                            },
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "done":
                    _track_status(state="done", phase="done", confirmation_pending=False)
                    final_text = event[1] or ""
                    if tts_enabled:
                        # Strip citation block from TTS — only read the main answer
                        tts_text = final_text.split("\n---\n")[0] if "\n---\n" in final_text else final_text
                        if tts_text and not full_answer:
                            full_answer = tts_text
                            pending_tts += tts_text
                        elif (
                            tts_text
                            and len(tts_text) > len(full_answer)
                            and tts_text.startswith(full_answer)
                        ):
                            pending_tts += tts_text[len(full_answer):]
                            full_answer = tts_text
                    # UI always gets the full text including citations
                    done_payload = {
                        "type": "done",
                        "text": event[1],
                        "voice_pending": bool(tts_enabled),
                    }
                    if len(event) > 2 and isinstance(event[2], dict):
                        done_payload["quality"] = event[2]
                    await _send(
                        _attach_event_meta(
                            done_payload,
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "error":
                    _track_status(state="error", phase="error")
                    await _send(
                        _attach_event_meta(
                            {"type": "error", "text": event[1]},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "policy_blocked":
                    await _send(
                        _attach_event_meta(
                            {
                                "type": "policy_blocked",
                                "name": event[1],
                                "reason": event[2],
                            },
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "policy_review":
                    await _send(
                        _attach_event_meta(
                            {
                                "type": "policy_review",
                                "tool": event[1],
                                "result": event[2],
                            },
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "task_outcome":
                    await _send(
                        _attach_event_meta(
                            {"type": "task_outcome", "result": event[1]},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "service_tier":
                    await _send(
                        _attach_event_meta(
                            {"type": "service_tier", "result": event[1]},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "llm_usage":
                    payload = event[1] if len(event) > 1 else {}
                    if isinstance(payload, str):
                        try:
                            payload = json.loads(payload)
                        except Exception:
                            payload = {"raw": payload}
                    await _send(
                        _attach_event_meta(
                            {"type": "llm_usage", "usage": payload},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "skill_selected":
                    await _send(
                        _attach_event_meta(
                            {"type": "skill_selected", "result": event[1]},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "context_update":
                    await _send(
                        _attach_event_meta(
                            {"type": "context_update", "data": event[1]},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "confirmation_required":
                    brief = event[4] if len(event) > 4 else ""
                    expires_at = event[5] if len(event) > 5 else ""
                    _apply_runtime_event(status_tracker, etype, event[1], event[2], event[3])
                    await _send(
                        _attach_event_meta(
                            {
                                "type": "confirmation_required",
                                "token": event[1],
                                "tool": event[2],
                                "reason": event[3],
                                "brief": brief,
                                "expires_at": expires_at,
                            },
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "guard_retry":
                    retry_reason = event[2] if len(event) > 2 else ""
                    retry_text = event[1] if len(event) > 1 else "The previous reply needs one retry."
                    _apply_runtime_event(status_tracker, etype, retry_text, retry_reason)
                    await _send(
                        _attach_event_meta(
                            {
                                "type": "status_message",
                                "text": retry_text,
                                "run_state": "guard_retry",
                                "reason": retry_reason,
                            },
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype in ("dispatch", "synthesis", "synthesis_partial", "sub_complete"):
                    await _send(
                        _attach_event_meta(
                            {"type": etype, "data": event[1]},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "proactive_suggestion":
                    await _send(
                        _attach_event_meta(
                            {"type": "proactive_suggestion",
                             "message": event[1],
                             "suggestion_id": event[2]},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                elif etype == "quality_gate":
                    payload = event[1] if len(event) > 1 else {}
                    await _send(
                        _attach_event_meta(
                            {"type": "quality_gate",
                             "verdict": payload.get("verdict", "accept") if isinstance(payload, dict) else "accept",
                             "data": payload},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
                else:
                    await _send(
                        _attach_event_meta(
                            {"type": etype, "data": _event_payload_tail(event)},
                            event_type=etype,
                            event_meta=event_meta,
                        )
                    )
        finally:
            if run_lock_acquired and _shared._BRAIN_RUN_LOCK.locked():
                _shared._BRAIN_RUN_LOCK.release()
                run_lock_acquired = False
        if tts_enabled:
            await _drain_tts(force=True)
            if not tts_failed_reason and tts_chunks_emitted == 0 and (full_answer.strip() or pending_tts.strip()):
                tts_failed_reason = "tts generated empty audio"
            if tts_failed_reason:
                await _send({"type": "error", "text": f"tts error: {tts_failed_reason}"})
            score = 1.0 if tts_chunks_emitted <= 1 else max(0.6, 1.0 - (tts_chunks_emitted - 1) * 0.08)
            await _send(
                {
                    "type": "tts_metrics",
                    "first_chunk_ms": round(first_tts_ms or 0.0, 1),
                    "chunks": int(tts_chunks_emitted),
                    "answer_chars": len(full_answer),
                    "tts_total_ms": round(tts_total_ms, 1),
                }
            )
            await _send({"type": "consistency_score", "score": round(score, 3)})
            await _send({"type": "tts_done"})
        stream_ms = (
            (time.perf_counter() - stream_started) * 1000.0
            if stream_started is not None
            else 0.0
        )
        total_ms = (time.perf_counter() - run_started) * 1000.0
        await _send(
            {
                "type": "run_metrics",
                "queued_ms": round(queued_ms, 1),
                "stream_ms": round(stream_ms, 1),
                "tts_ms": round(tts_total_ms, 1),
                "total_ms": round(total_ms, 1),
            }
        )
        final_state = "done"
        _track_status(state="done", phase="done", confirmation_pending=False)
        await _send({"type": "run_state", "state": "done"})
    except asyncio.CancelledError:
        final_state = "cancelled"
        _track_status(state="cancelled", phase="cancelled", confirmation_pending=False)
        try:
            await _send({"type": "run_state", "state": "cancelled"})
        except Exception:
            pass
        # Task cancelled during shutdown or client interrupt; exit quietly.
        pass
    except Exception as e:
        final_state = "error"
        _track_status(state="error", phase="error", confirmation_pending=False)
        await _send_best_effort({"type": "error", "text": f"stream error: {e}"})
        await _send_best_effort({"type": "run_state", "state": "error", "detail": str(e)})
    finally:
        if pending_lock_task is not None and not pending_lock_task.done():
            pending_lock_task.cancel()
            with suppress(asyncio.CancelledError):
                await pending_lock_task
        if pending_event_task is not None and not pending_event_task.done():
            pending_event_task.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await pending_event_task
        if run_lock_acquired and _shared._BRAIN_RUN_LOCK and _shared._BRAIN_RUN_LOCK.locked():
            _shared._BRAIN_RUN_LOCK.release()
        executor = getattr(app.state, "executor", None)
        if executor is not None:
            try:
                await executor.resume_deferred_runs()
            except Exception as e:
                _log.error("web_server", e, action="resume_deferred_runs")
        for p in images or []:
            try:
                path = Path(p)
                if path.exists():
                    path.unlink(missing_ok=True)
            except Exception:
                pass
        if stream_started is not None:
            stream_ms = (time.perf_counter() - stream_started) * 1000.0
        else:
            stream_ms = 0.0
        total_ms = (time.perf_counter() - run_started) * 1000.0
        try:
            _metrics_ref.log_runtime(
                run_id=run_id,
                queued_ms=queued_ms,
                stream_ms=stream_ms,
                tts_ms=tts_total_ms,
                total_ms=total_ms,
                voice_mode=voice_triggered,
                final_state=final_state,
            )
        except Exception as e:
            _log.error("web_server", e, action="log_runtime_metrics")


# ─── WebSocket chat ─────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    # S3: IP whitelist check (before accept)
    if not _ip_allowed(ws):
        client_ip = ws.client.host if ws.client else "unknown"
        _log.warning(f"ws rejected: ip={client_ip} not in allowlist")
        await ws.close(code=1008, reason="forbidden")
        return
    # S4: Origin validation (before accept)
    if not _origin_allowed(ws):
        _log.warning(f"ws rejected: origin={ws.headers.get('origin', '')} not allowed")
        await ws.close(code=1008, reason="forbidden origin")
        return
    await ws.accept()

    ws_lock = asyncio.Lock()
    push_enabled = bool((ws.headers.get("origin") or "").strip())

    # In-band auth: first message must be {type:"auth", token:"..."}
    if _WEB_AUTH_TOKEN:
        if not await _authenticate_ws_inband(ws, ws_lock):
            return

    current_task: asyncio.Task | None = None
    current_cancel_scope = None
    background_tasks: set[asyncio.Task] = set()
    vision_cache: dict | None = None
    current_run_status = RuntimeStatusSnapshot()
    client_id = (ws.client.host if ws.client else "local") or "local"
    chat_session_key: str | None = None
    cancel_wait_sec = max(
        0.2, float(os.environ.get("LIAGENT_WS_CANCEL_WAIT_SEC", "1.5"))
    )

    # Register for task result broadcasts
    if push_enabled:
        _chat_clients[ws] = ws_lock

    async def _watch_stream_task(task: asyncio.Task):
        nonlocal current_task
        nonlocal current_cancel_scope
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            async with ws_lock:
                await ws.send_json({"type": "error", "text": f"task error: {e}"})
        finally:
            if current_task is task:
                current_task = None
            current_cancel_scope = None

    def _launch_stream(
        prompt_text: str,
        *,
        run_id: str,
        voice_triggered: bool,
        image_paths: list[str] | None,
        session_key: str | None,
    ) -> None:
        nonlocal current_task
        nonlocal current_cancel_scope
        from ..agent.run_control import RunCancellationScope

        _init_runtime_status(
            current_run_status,
            run_id=run_id,
            query=prompt_text,
            session_key=session_key,
        )
        current_cancel_scope = RunCancellationScope()
        task = asyncio.create_task(
            _stream_events(
                ws,
                _shared._brain,
                prompt_text,
                run_id,
                voice_triggered=voice_triggered,
                images=image_paths,
                session_key=session_key,
                cancel_scope=current_cancel_scope,
                ws_lock=ws_lock,
                status_tracker=current_run_status,
            )
        )
        current_task = task
        monitor = asyncio.create_task(_watch_stream_task(task))
        background_tasks.add(monitor)
        monitor.add_done_callback(background_tasks.discard)

    async def _cancel_current_stream(*, send_cancelled: bool = False) -> None:
        nonlocal current_task
        nonlocal current_cancel_scope
        task = current_task
        if task and not task.done():
            if current_cancel_scope is not None:
                current_cancel_scope.cancel("user_cancelled")
            _mark_runtime_cancel_requested(current_run_status, "user_cancelled")
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=cancel_wait_sec)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            if current_task is task:
                current_task = None
            current_cancel_scope = None
            if send_cancelled:
                async with ws_lock:
                    await ws.send_json({"type": "run_state", "state": "cancelled"})

    def _resolve_session_key(raw_value: Any) -> str:
        nonlocal chat_session_key
        session_key = str(raw_value or "").strip()
        if not session_key:
            session_key = chat_session_key or f"web:{uuid.uuid4().hex}"
        chat_session_key = session_key
        _chat_client_sessions[ws] = session_key
        return session_key

    def _resolve_control_session_key(payload: dict[str, Any]) -> str | None:
        requested = str(payload.get("session_key", "") or "").strip()
        if requested:
            return requested
        if chat_session_key:
            return chat_session_key
        return None

    def _clear_active_chat_session() -> None:
        nonlocal chat_session_key
        chat_session_key = None
        _chat_client_sessions.pop(ws, None)

    try:
        while True:
            data = await ws.receive_json()
            now_ms = int(time.time() * 1000)
            if vision_cache and not _vision_cache_valid(vision_cache, now_ms):
                _cleanup_path(str(vision_cache.get("path", "")))
                vision_cache = None
            if not _allow_rate(_ws_rate, client_id):
                async with ws_lock:
                    await ws.send_json({"type": "error", "text": "rate limited"})
                continue

            msg_type = data.get("type", "")

            # Ignore late auth messages (already authenticated above)
            if msg_type == "auth":
                continue

            if msg_type == "text":
                user_text = data.get("text", "").strip()
                if not user_text:
                    continue
                if (
                    push_enabled
                    and current_task
                    and not current_task.done()
                    and _is_running_status_followup(user_text)
                ):
                    async with ws_lock:
                        await ws.send_json(
                            {
                                "type": "status_message",
                                "text": _format_running_status_message(current_run_status),
                                "run_id": str(current_run_status.get("run_id") or ""),
                                "run_state": str(
                                    current_run_status.get("state")
                                    or current_run_status.get("phase")
                                    or "streaming"
                                ),
                            }
                        )
                    continue
                if current_task and not current_task.done():
                    await _cancel_current_stream(send_cancelled=True)
                session_key = _resolve_session_key(data.get("session_key"))
                run_id = uuid.uuid4().hex[:10]
                image_paths, image_note, cache_update = _resolve_image_paths(
                    data, vision_cache=vision_cache
                )
                if cache_update:
                    old_path = str(vision_cache.get("path", "")) if vision_cache else ""
                    vision_cache = cache_update
                    if old_path and old_path != str(cache_update.get("path", "")):
                        _cleanup_path(old_path)
                if image_note:
                    async with ws_lock:
                        await ws.send_json({"type": "vision_note", "text": image_note})
                _launch_stream(
                    user_text,
                    run_id=run_id,
                    voice_triggered=False,
                    image_paths=image_paths,
                    session_key=session_key,
                )

            elif msg_type == "audio":
                # Receive recorded audio for STT
                if current_task and not current_task.done():
                    await _cancel_current_stream(send_cancelled=True)
                session_key = _resolve_session_key(data.get("session_key"))
                try:
                    audio_b64 = data.get("audio", "")
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

                    async with ws_lock:
                        await ws.send_json({"type": "stt_start"})
                    text = await _shared._engine.stt.transcribe(
                        audio_np, language=_shared._engine.config.stt.language
                    )
                    async with ws_lock:
                        await ws.send_json({"type": "stt_result", "text": text})
                except Exception as e:
                    async with ws_lock:
                        await ws.send_json({"type": "error", "text": f"stt error: {e}"})
                    continue

                if text and text.strip():
                    run_id = uuid.uuid4().hex[:10]
                    image_paths, image_note, cache_update = _resolve_image_paths(
                        data, vision_cache=vision_cache
                    )
                    if cache_update:
                        old_path = str(vision_cache.get("path", "")) if vision_cache else ""
                        vision_cache = cache_update
                        if old_path and old_path != str(cache_update.get("path", "")):
                            _cleanup_path(old_path)
                    if image_note:
                        async with ws_lock:
                            await ws.send_json({"type": "vision_note", "text": image_note})
                    _launch_stream(
                        text,
                        run_id=run_id,
                        voice_triggered=True,
                        image_paths=image_paths,
                        session_key=session_key,
                    )

            elif msg_type == "feedback":
                # Feedback from the frontend
                fb = str(data.get("feedback", "")).strip()
                turn_idx = int(data.get("turn_index", 0))
                query = str(data.get("query", "")).strip()
                answer = str(data.get("answer", "")).strip()
                tool_used = data.get("tool_used") or None
                feedback_session_key = _resolve_control_session_key(data)
                if fb in ("positive", "negative"):
                    try:
                        await _shared._brain.record_user_feedback(
                            query=query,
                            answer=answer,
                            tool_used=tool_used,
                            feedback=fb,
                            turn_index=turn_idx,
                            session_key=feedback_session_key,
                        )
                    except Exception as e:
                        _log.error("web_server", e, action="process_feedback")
                    async with ws_lock:
                        await ws.send_json({"type": "feedback_ack", "feedback": fb})

            elif msg_type == "clear":
                await _cancel_current_stream(send_cancelled=False)
                control_session_key = _resolve_control_session_key(data)
                repl_reset = False
                if control_session_key:
                    try:
                        from ..tools.stateful_repl import reset_repl_session
                        repl_reset = await reset_repl_session(control_session_key)
                    except Exception as e:
                        _log.error("web_server", e, action="clear_repl_session")
                await _shared._brain.clear_memory_for_session(control_session_key)
                if vision_cache:
                    _cleanup_path(str(vision_cache.get("path", "")))
                    vision_cache = None
                _clear_active_chat_session()
                async with ws_lock:
                    await ws.send_json(
                        {
                            "type": "cleared",
                            "session_key": control_session_key or "",
                            "memory_scope": "session_runtime",
                            "repl_reset": repl_reset,
                        }
                    )

            elif msg_type == "finalize":
                # Save session facts to long-term memory without clearing conversation.
                # Used by Discord bot on disconnect to persist memory.
                await _cancel_current_stream(send_cancelled=False)
                control_session_key = _resolve_control_session_key(data)
                try:
                    await _shared._brain.finalize_session(session_key=control_session_key)
                except Exception as e:
                    _log.error("web_server", e, action="ws_finalize")
                async with ws_lock:
                    await ws.send_json(
                        {
                            "type": "finalized",
                            "session_key": control_session_key or "",
                            "memory_scope": "session_runtime",
                        }
                    )

            elif msg_type == "barge_in":
                # User interrupted TTS — cancel current generation
                await _cancel_current_stream(send_cancelled=False)
                async with ws_lock:
                    await ws.send_json({"type": "barge_in_ack"})

            elif msg_type == "tool_confirm":
                token = str(data.get("token", "")).strip()
                approved = _parse_approved_flag(data.get("approved", True))
                force = bool(data.get("force", False))
                confirm_session_key = _resolve_control_session_key(data)
                if not token:
                    async with ws_lock:
                        await ws.send_json({"type": "error", "text": "missing token"})
                    continue
                result = await _shared._brain.resolve_confirmation(
                    token,
                    approved=approved,
                    force=force,
                    session_key=confirm_session_key,
                )
                async with ws_lock:
                    await ws.send_json({"type": "tool_confirm_result", "result": result})

            elif msg_type == "suggestion_feedback":
                sug_id = data.get("id")
                action = str(data.get("action", "")).strip()
                feedback_session_key = _resolve_control_session_key(data)
                if sug_id is not None and action in ("accept", "dismiss"):
                    try:
                        await _apply_suggestion_feedback(
                            int(sug_id), action, session_id=feedback_session_key,
                        )
                    except Exception as e:
                        _log.error("web_server", e, action="suggestion_feedback")

    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except Exception as e:
        try:
            async with ws_lock:
                await ws.send_json({"type": "error", "text": f"ws error: {e}"})
        except Exception:
            pass
    finally:
        _chat_clients.pop(ws, None)
        _chat_client_sessions.pop(ws, None)
        await _cancel_current_stream(send_cancelled=False)
        if background_tasks:
            await asyncio.gather(*list(background_tasks), return_exceptions=True)
        if vision_cache:
            _cleanup_path(str(vision_cache.get("path", "")))


def _vision_cache_valid(cache: dict | None, now_ms: int) -> bool:
    """Thin wrapper around vision_routes._cache_valid for backward compat."""
    from .vision_routes import _cache_valid
    return _cache_valid(cache, now_ms)
