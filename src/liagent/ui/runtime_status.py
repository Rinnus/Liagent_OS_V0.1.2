"""Shared runtime-status semantics for Web and Discord surfaces."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, fields
from typing import Any


_TERMINAL_STATES = {"done", "error", "cancelled"}


@dataclass
class RuntimeStatusSnapshot:
    run_id: str = ""
    query: str = ""
    session_key: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    last_status_push_at: float = 0.0
    state: str = "accepted"
    phase: str = "accepted"
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    effective_tool_name: str = ""
    effective_tool_args: dict[str, Any] = field(default_factory=dict)
    confirmation_pending: bool = False
    confirmation_tool: str = ""
    confirmation_reason: str = ""
    tool_skip_reason: str = ""
    guard_retry_text: str = ""
    guard_retry_reason: str = ""
    cancel_requested: bool = False
    cancel_reason: str = ""
    tool_history: list[dict[str, Any]] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def clear(self) -> None:
        now = time.time()
        self.run_id = ""
        self.query = ""
        self.session_key = ""
        self.started_at = now
        self.updated_at = now
        self.last_status_push_at = 0.0
        self.state = "accepted"
        self.phase = "accepted"
        self.tool_name = ""
        self.tool_args = {}
        self.effective_tool_name = ""
        self.effective_tool_args = {}
        self.confirmation_pending = False
        self.confirmation_tool = ""
        self.confirmation_reason = ""
        self.tool_skip_reason = ""
        self.guard_retry_text = ""
        self.guard_retry_reason = ""
        self.cancel_requested = False
        self.cancel_reason = ""
        self.tool_history = []

    def apply(self, **updates: Any) -> None:
        for key, value in updates.items():
            setattr(self, key, value)
        if "updated_at" not in updates:
            self.updated_at = time.time()

    def clone(self) -> "RuntimeStatusSnapshot":
        return RuntimeStatusSnapshot(
            run_id=self.run_id,
            query=self.query,
            session_key=self.session_key,
            started_at=self.started_at,
            updated_at=self.updated_at,
            last_status_push_at=self.last_status_push_at,
            state=self.state,
            phase=self.phase,
            tool_name=self.tool_name,
            tool_args=dict(self.tool_args),
            effective_tool_name=self.effective_tool_name,
            effective_tool_args=dict(self.effective_tool_args),
            confirmation_pending=self.confirmation_pending,
            confirmation_tool=self.confirmation_tool,
            confirmation_reason=self.confirmation_reason,
            tool_skip_reason=self.tool_skip_reason,
            guard_retry_text=self.guard_retry_text,
            guard_retry_reason=self.guard_retry_reason,
            cancel_requested=self.cancel_requested,
            cancel_reason=self.cancel_reason,
            tool_history=[dict(item) for item in self.tool_history],
        )


def ensure_runtime_status_snapshot(
    status: RuntimeStatusSnapshot | dict[str, Any] | None,
) -> RuntimeStatusSnapshot:
    if isinstance(status, RuntimeStatusSnapshot):
        return status
    snapshot = RuntimeStatusSnapshot()
    if isinstance(status, dict):
        allowed = {f.name for f in fields(RuntimeStatusSnapshot)}
        for key in allowed:
            if key not in status:
                continue
            value = status.get(key)
            if isinstance(value, dict):
                value = dict(value)
            elif isinstance(value, list):
                value = [dict(item) if isinstance(item, dict) else item for item in value]
            setattr(snapshot, key, value)
    return snapshot


def _normalize_tool_args(tool_args: dict[str, Any] | None) -> dict[str, Any]:
    return dict(tool_args) if isinstance(tool_args, dict) else {}


def _status_get(status: RuntimeStatusSnapshot | dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if isinstance(status, RuntimeStatusSnapshot):
        return getattr(status, key, default)
    if isinstance(status, dict):
        return status.get(key, default)
    return default


def _status_set(status: RuntimeStatusSnapshot | dict[str, Any] | None, key: str, value: Any) -> None:
    if isinstance(status, RuntimeStatusSnapshot):
        setattr(status, key, value)
    elif isinstance(status, dict):
        status[key] = value


def _status_update(status: RuntimeStatusSnapshot | dict[str, Any] | None, **updates: Any) -> None:
    if status is None:
        return
    if isinstance(status, RuntimeStatusSnapshot):
        status.apply(**updates)
        return
    status.setdefault("last_status_push_at", 0.0)
    status.update(updates)
    if "updated_at" not in updates:
        status["updated_at"] = time.time()


def _status_clear(status: RuntimeStatusSnapshot | dict[str, Any]) -> None:
    if isinstance(status, RuntimeStatusSnapshot):
        status.clear()
    else:
        status.clear()


def _append_tool_history(
    status: RuntimeStatusSnapshot | dict[str, Any] | None,
    *,
    tool_name: str,
    tool_args: dict[str, Any] | None,
    effective_tool_name: str | None = None,
    effective_tool_args: dict[str, Any] | None = None,
) -> None:
    if status is None:
        return
    history = _status_get(status, "tool_history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "tool_name": str(tool_name or ""),
            "tool_args": _normalize_tool_args(tool_args),
            "effective_tool_name": str(effective_tool_name or tool_name or ""),
            "effective_tool_args": _normalize_tool_args(effective_tool_args) or _normalize_tool_args(tool_args),
            "recorded_at": time.time(),
        }
    )
    if len(history) > 6:
        history = history[-6:]
    _status_set(status, "tool_history", history)


def init_runtime_status(
    status: RuntimeStatusSnapshot | dict[str, Any],
    *,
    run_id: str,
    query: str,
    session_key: str | None,
) -> None:
    now = time.time()
    _status_clear(status)
    _status_update(
        status,
        run_id=str(run_id or ""),
        query=str(query or ""),
        session_key=str(session_key or ""),
        started_at=now,
        updated_at=now,
        last_status_push_at=0.0,
        state="accepted",
        phase="accepted",
        tool_name="",
        tool_args={},
        effective_tool_name="",
        effective_tool_args={},
        confirmation_pending=False,
        confirmation_tool="",
        confirmation_reason="",
        tool_skip_reason="",
        guard_retry_text="",
        guard_retry_reason="",
        cancel_requested=False,
        cancel_reason="",
        tool_history=[],
    )


def update_runtime_status(status: RuntimeStatusSnapshot | dict[str, Any] | None, **updates: Any) -> None:
    _status_update(status, **updates)


def mark_runtime_cancel_requested(
    status: RuntimeStatusSnapshot | dict[str, Any] | None,
    reason: str = "user_cancelled",
) -> None:
    _status_update(
        status,
        cancel_requested=True,
        cancel_reason=str(reason or "user_cancelled"),
    )


def clear_runtime_cancel_requested(status: RuntimeStatusSnapshot | dict[str, Any] | None) -> None:
    _status_update(status, cancel_requested=False, cancel_reason="")


def runtime_status_state(status: RuntimeStatusSnapshot | dict[str, Any] | None) -> str:
    return str(_status_get(status, "state") or _status_get(status, "phase") or "").strip()


def runtime_status_is_terminal(status: RuntimeStatusSnapshot | dict[str, Any] | None) -> bool:
    return runtime_status_state(status) in _TERMINAL_STATES


def apply_legacy_event_to_runtime_status(
    status: RuntimeStatusSnapshot | dict[str, Any] | None,
    event_type: str,
    *payload: Any,
) -> None:
    if status is None:
        return
    if event_type == "tool_start":
        tool_name = str(payload[0] or "") if payload else ""
        tool_args = _normalize_tool_args(payload[1] if len(payload) > 1 else None)
        _append_tool_history(status, tool_name=tool_name, tool_args=tool_args)
        _status_update(
            status,
            state="tool_start",
            phase="tool_start",
            tool_name=tool_name,
            tool_args=tool_args,
            effective_tool_name=tool_name,
            effective_tool_args=dict(tool_args),
            confirmation_pending=False,
            confirmation_tool="",
            confirmation_reason="",
            tool_skip_reason="",
            guard_retry_text="",
        )
        return
    if event_type == "tool_fallback":
        requested_tool = str(payload[0] or "") if payload else ""
        effective_tool = str(payload[1] or requested_tool or "") if len(payload) > 1 else requested_tool
        details = payload[2] if len(payload) > 2 and isinstance(payload[2], dict) else {}
        requested_args = _normalize_tool_args(details.get("requested_args"))
        effective_args = _normalize_tool_args(details.get("effective_args"))
        _append_tool_history(
            status,
            tool_name=requested_tool,
            tool_args=requested_args,
            effective_tool_name=effective_tool,
            effective_tool_args=effective_args,
        )
        _status_update(
            status,
            state="tool_start",
            phase="tool_fallback",
            tool_name=requested_tool,
            tool_args=requested_args,
            effective_tool_name=effective_tool,
            effective_tool_args=effective_args,
            confirmation_pending=False,
        )
        return
    if event_type == "tool_result":
        tool_name = str(payload[0] or "") if payload else ""
        tool_args = _normalize_tool_args(_status_get(status, "effective_tool_args") or _status_get(status, "tool_args"))
        _status_update(
            status,
            state="tool_result",
            phase="tool_result",
            tool_name=tool_name,
            tool_args=tool_args,
            effective_tool_name=tool_name,
            effective_tool_args=dict(tool_args),
            confirmation_pending=False,
        )
        return
    if event_type == "tool_error":
        tool_name = str(payload[0] or "") if payload else ""
        details = payload[1] if len(payload) > 1 and isinstance(payload[1], dict) else {}
        tool_args = _normalize_tool_args(details.get("tool_args"))
        _status_update(
            status,
            state="tool_error",
            phase="tool_error",
            tool_name=tool_name,
            tool_args=tool_args,
            effective_tool_name=tool_name,
            effective_tool_args=dict(tool_args),
            confirmation_pending=False,
        )
        return
    if event_type == "tool_skip":
        tool_name = str(payload[0] or "") if payload else ""
        reason = str(payload[1] or "") if len(payload) > 1 else ""
        details = payload[2] if len(payload) > 2 and isinstance(payload[2], dict) else {}
        tool_args = _normalize_tool_args(details.get("effective_args") or details.get("tool_args") or details)
        _status_update(
            status,
            state="tool_skip",
            phase="tool_skip",
            tool_name=tool_name,
            tool_args=tool_args,
            effective_tool_name=str(details.get("effective_tool") or tool_name),
            effective_tool_args=dict(tool_args),
            tool_skip_reason=reason,
            confirmation_pending=False,
        )
        return
    if event_type == "guard_retry":
        retry_text = str(payload[0] or "").strip() if payload else ""
        retry_reason = str(payload[1] or "").strip() if len(payload) > 1 else ""
        _status_update(
            status,
            state="guard_retry",
            phase="guard_retry",
            guard_retry_text=retry_text,
            guard_retry_reason=retry_reason,
            confirmation_pending=False,
        )
        return
    if event_type == "confirmation_required":
        tool_name = str(payload[1] or "") if len(payload) > 1 else ""
        reason = str(payload[2] or "") if len(payload) > 2 else ""
        _status_update(
            status,
            state="confirmation_required",
            phase="confirmation_required",
            confirmation_pending=True,
            confirmation_tool=tool_name,
            confirmation_reason=reason,
            tool_name=tool_name,
            effective_tool_name=tool_name,
        )
        return
    if event_type == "run_state":
        state = str(payload[0] or "") if payload else ""
        _status_update(status, state=state, phase=state, confirmation_pending=False)


def apply_ws_message_to_runtime_status(
    status: RuntimeStatusSnapshot | dict[str, Any] | None,
    message: dict[str, Any],
) -> None:
    if status is None or not isinstance(message, dict):
        return
    mtype = str(message.get("type") or "").strip()
    if mtype == "run_state":
        state = str(message.get("state") or "streaming").strip()
        apply_legacy_event_to_runtime_status(status, "run_state", state)
        return
    if mtype == "tool_start":
        apply_legacy_event_to_runtime_status(
            status,
            "tool_start",
            str(message.get("name") or ""),
            message.get("args") if isinstance(message.get("args"), dict) else {},
        )
        return
    if mtype == "tool_fallback":
        apply_legacy_event_to_runtime_status(
            status,
            "tool_fallback",
            str(message.get("requested_name") or ""),
            str(message.get("effective_name") or ""),
            {
                "requested_args": message.get("requested_args") if isinstance(message.get("requested_args"), dict) else {},
                "effective_args": message.get("effective_args") if isinstance(message.get("effective_args"), dict) else {},
            },
        )
        return
    if mtype == "tool_result":
        apply_legacy_event_to_runtime_status(status, "tool_result", str(message.get("name") or ""))
        return
    if mtype == "tool_error":
        apply_legacy_event_to_runtime_status(
            status,
            "tool_error",
            str(message.get("name") or ""),
            {
                "tool_args": message.get("args") if isinstance(message.get("args"), dict) else {},
                "message": str(message.get("error") or message.get("result") or ""),
            },
        )
        return
    if mtype == "tool_skip":
        apply_legacy_event_to_runtime_status(
            status,
            "tool_skip",
            str(message.get("name") or ""),
            str(message.get("reason") or ""),
            message.get("details") if isinstance(message.get("details"), dict) else {},
        )
        return
    if mtype == "confirmation_required":
        apply_legacy_event_to_runtime_status(
            status,
            "confirmation_required",
            str(message.get("token") or ""),
            str(message.get("tool") or ""),
            str(message.get("reason") or ""),
        )
        return
    if mtype == "status_message" and str(message.get("run_state") or "") == "guard_retry":
        apply_legacy_event_to_runtime_status(
            status,
            "guard_retry",
            str(message.get("text") or ""),
            str(message.get("reason") or ""),
        )


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


def format_status_wait(seconds: float) -> str:
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


def format_running_status_message(status: RuntimeStatusSnapshot | dict[str, Any] | None) -> str:
    now_ts = time.time()
    state = runtime_status_state(status) or "streaming"
    tool_name = str(_status_get(status, "effective_tool_name") or _status_get(status, "tool_name") or "").strip()
    tool_args = _normalize_tool_args(_status_get(status, "effective_tool_args") or _status_get(status, "tool_args"))
    tool_ref = _format_status_tool(tool_name, tool_args) if tool_name else ""
    confirmation_tool = str(_status_get(status, "confirmation_tool") or tool_name or "").strip()
    confirmation_reason = str(_status_get(status, "confirmation_reason") or "").strip()
    query = str(_status_get(status, "query") or "").strip()
    query_head = f'The previous request "{query[:60]}"' if query else "The previous request"
    started_at = _status_get(status, "started_at")
    updated_at = _status_get(status, "updated_at")
    started_sec = max(0.0, now_ts - float(started_at or now_ts))
    idle_sec = max(0.0, now_ts - float(updated_at or now_ts))
    waited = format_status_wait(started_sec)
    idle_waited = format_status_wait(idle_sec)
    history = _status_get(status, "tool_history", [])
    history_count = len(history) if isinstance(history, list) else 0
    if _status_get(status, "confirmation_pending"):
        tool_ref = (
            _format_status_tool(confirmation_tool, tool_args)
            if confirmation_tool
            else "this tool call"
        )
        reason_tail = f" Reason: {confirmation_reason}." if confirmation_reason else ""
        return (
            f"{query_head} has been waiting for {waited} and is blocked on tool confirmation for {tool_ref}."
            f"{reason_tail} Confirm or reject the tool call before results can continue."
        )
    if _status_get(status, "cancel_requested"):
        reason = str(_status_get(status, "cancel_reason") or "").strip()
        tail = f" (reason: {reason})" if reason else ""
        return f"{query_head} is being cancelled{tail}."
    if state in {"accepted", "queued"}:
        return f"{query_head} has been accepted and is still queued after {waited}. No tool results have returned yet."
    if state == "guard_retry":
        retry_text = str(_status_get(status, "guard_retry_text") or "").strip()
        if retry_text:
            return retry_text
        return f"{query_head} is being retried after a response-quality retry."
    if state == "tool_error" and tool_ref:
        return f"{query_head} hit an issue while running {tool_ref}. A fallback path is being attempted."
    if state == "tool_skip":
        skip_reason = str(_status_get(status, "tool_skip_reason") or "").strip()
        if skip_reason == "auto_fetch_budget_exhausted":
            return f"{query_head} skipped an automatic follow-up fetch because the tool budget is exhausted. The response is being drafted from the current results."
        return f"{query_head} skipped one automatic step and is continuing from the available results."
    if state == "tool_result" and tool_ref:
        if history_count > 1:
            return f"{query_head} has completed {history_count} tool steps. The most recent result came from {tool_ref}, and the response is being drafted now."
        if idle_sec >= 8:
            return f"{query_head} already has a result from {tool_ref} and is drafting the response after {waited}."
        return f"{query_head} already has a result from {tool_ref} and is drafting the response."
    if state == "tool_start" and tool_ref:
        if idle_sec >= 8:
            return (
                f"{query_head} has been waiting for {waited} and is still waiting for {tool_ref} to return. "
                f"There has been no new progress for {idle_waited}."
            )
        return f"{query_head} is still running and is currently calling {tool_ref}. Results will continue after it returns."
    if state == "streaming" and tool_ref:
        if idle_sec >= 8:
            return (
                f"{query_head} is still processing after {waited}. "
                f"The most recent step is {tool_ref}, and there have been no new results for {idle_waited}."
            )
        return f"{query_head} is still processing. The most recent step is {tool_ref}; more output will follow when it returns."
    return f"{query_head} is still generating the response after {waited}. Give it a little more time."


def format_busy_message(status: RuntimeStatusSnapshot | dict[str, Any] | None) -> str:
    return (
        f"{format_running_status_message(status)}\n\n"
        'If you only want progress, reply with "status" or "then?". '
        "If you want to stop this run, use the cancel control. "
        "If this is a new request, wait for the current response to finish before sending it."
    )


def build_status_signature(status: RuntimeStatusSnapshot | dict[str, Any] | None) -> str:
    now_ts = time.time()
    started_at = float(_status_get(status, "started_at") or now_ts)
    updated_at = float(_status_get(status, "updated_at") or now_ts)
    elapsed_sec = max(0.0, now_ts - started_at)
    idle_sec = max(0.0, now_ts - updated_at)
    return json.dumps(
        {
            "state": runtime_status_state(status),
            "tool": str(_status_get(status, "effective_tool_name") or _status_get(status, "tool_name") or ""),
            "tool_args": json.dumps(_normalize_tool_args(_status_get(status, "effective_tool_args") or _status_get(status, "tool_args")), ensure_ascii=False, sort_keys=True),
            "confirmation": bool(_status_get(status, "confirmation_pending")),
            "skip": str(_status_get(status, "tool_skip_reason") or ""),
            "guard_retry": str(_status_get(status, "guard_retry_reason") or ""),
            "history_len": len(_status_get(status, "tool_history", []) or []),
            "elapsed_bucket": int(elapsed_sec // 5),
            "idle_bucket": int(idle_sec // 5),
            "cancel_requested": bool(_status_get(status, "cancel_requested")),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def should_push_runtime_status(
    status: RuntimeStatusSnapshot | dict[str, Any] | None,
    *,
    now: float | None = None,
    after_sec: float,
    idle_floor_sec: float,
    repeat_sec: float,
) -> bool:
    if status is None:
        return False
    state = runtime_status_state(status)
    if state in _TERMINAL_STATES:
        return False
    now_ts = float(now or time.time())
    updated_at = float(_status_get(status, "updated_at") or now_ts)
    started_at = float(_status_get(status, "started_at") or now_ts)
    last_push_at = float(_status_get(status, "last_status_push_at") or 0.0)
    elapsed_sec = max(0.0, now_ts - started_at)
    idle_sec = max(0.0, now_ts - updated_at)
    immediate = state in {"confirmation_required", "guard_retry"}
    if not immediate:
        if elapsed_sec < after_sec:
            return False
        if state != "queued" and idle_sec < idle_floor_sec:
            return False
        if last_push_at > 0.0 and (now_ts - last_push_at) < repeat_sec:
            return False
    return True


def mark_runtime_status_pushed(status: RuntimeStatusSnapshot | dict[str, Any] | None, *, now: float | None = None) -> None:
    _status_set(status, "last_status_push_at", float(now or time.time()))
