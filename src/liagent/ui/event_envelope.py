"""Event envelope helpers used by CLI/web stream adapters."""

from __future__ import annotations

from typing import Any


def _extract_agent_event(event: Any) -> tuple[Any, dict[str, Any]]:
    """Return a legacy event tuple plus optional Event Envelope v2 metadata."""
    if not hasattr(event, "to_legacy_tuple"):
        return event, {}
    meta: dict[str, Any] = {}
    source = getattr(event, "source", None)
    agent_id = getattr(event, "agent_id", None)
    seq = getattr(event, "seq", None)
    timestamp = getattr(event, "timestamp", None)
    run_id = getattr(event, "run_id", None)
    if source:
        meta["event_source"] = str(source)
    if agent_id:
        meta["agent_id"] = str(agent_id)
    if isinstance(seq, int):
        meta["event_seq"] = seq
    if isinstance(timestamp, (int, float)):
        meta["event_ts"] = float(timestamp)
    if run_id:
        meta["event_run_id"] = str(run_id)
    return event.to_legacy_tuple(), meta


def _event_payload_tail(event: Any) -> Any:
    if not isinstance(event, tuple):
        return event
    if len(event) <= 1:
        return None
    if len(event) == 2:
        return event[1]
    return event[1:]


def _attach_event_meta(
    payload: dict[str, Any],
    *,
    event_type: str,
    event_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(payload)
    out.setdefault("event_type", event_type)
    meta = event_meta or {}
    if isinstance(meta, dict):
        for k in ("event_source", "agent_id", "event_seq", "event_ts"):
            if k in meta and meta[k] is not None:
                out.setdefault(k, meta[k])
        # Preserve source run id when present, but do not override stream run id.
        if "event_run_id" in meta and meta["event_run_id"] is not None:
            out.setdefault("event_run_id", meta["event_run_id"])
    return out
