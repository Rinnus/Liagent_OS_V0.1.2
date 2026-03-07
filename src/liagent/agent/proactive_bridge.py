# src/liagent/agent/proactive_bridge.py
"""Bridge Loop: converts delivery_mode='auto' suggestions into tasks or goals."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .behavior import PendingSuggestionStore
    from .goal_store import GoalStore
    from .task_queue import TaskStore

_log = logging.getLogger(__name__)


async def bridge_iteration(
    suggestion_store: PendingSuggestionStore,
    goal_store: GoalStore | None,
    task_store: TaskStore,
    trigger_mgr,
    config,
) -> int:
    """Single iteration of the bridge loop. Returns count of items processed."""
    pendings = suggestion_store.get_by_delivery_mode("auto", status="pending")
    created = 0
    for sug in pendings:
        if not suggestion_store.try_claim(sug["id"]):
            continue
        try:
            action = json.loads(sug["action_json"])
        except (json.JSONDecodeError, TypeError):
            _log.warning("bridge: invalid action_json for suggestion %s", sug["id"])
            suggestion_store.update_status(sug["id"], "failed")
            continue

        dedup = f"bridge:{sug['domain']}:{sug['id']}:{date.today()}"

        try:
            if action.get("create_goal") and goal_store is not None:
                consent = _derive_consent(sug, config)
                gid = goal_store.create(
                    idempotency_key=dedup,
                    source="pattern_auto",
                    domain=sug["domain"],
                    objective=action.get("objective", sug["message"]),
                    rationale=action.get("rationale", ""),
                    confidence=sug["confidence"],
                )
                if gid is not None:
                    if consent == "auto":
                        goal_store.transition(gid, "active")
                    goal_store.record_event(gid, "created", {"source": "bridge", "consent": consent})
            else:
                prompt = action.get("prompt", sug.get("message", ""))
                task = task_store.create_task_from_prompt(
                    prompt=prompt,
                    source="pattern_auto",
                    goal_id=action.get("goal_id"),
                    dedup_key=dedup,
                    risk_level=action.get("risk", "low"),
                )
                if task:
                    await trigger_mgr.register_once(task["id"], delay_seconds=0)
            suggestion_store.update_status(sug["id"], "accepted")
            created += 1
        except Exception:
            _log.exception("bridge: failed to materialize suggestion %s", sug["id"])
            suggestion_store.update_status(sug["id"], "failed")
    return created


def _derive_consent(sug: dict, config) -> str:
    """Derive user consent level from config authorization."""
    domain = sug.get("domain", "general")
    auth = getattr(config.proactive, "authorization", {})
    return auth.get(domain, "suggest")


async def bridge_loop(
    suggestion_store: PendingSuggestionStore,
    goal_store: GoalStore | None,
    task_store: TaskStore,
    trigger_mgr,
    config,
) -> None:
    """Long-running bridge loop. Scans auto suggestions every N seconds."""
    interval = getattr(config.proactive, "bridge_scan_interval_sec", 30)
    while True:
        await asyncio.sleep(interval)
        try:
            await bridge_iteration(
                suggestion_store, goal_store, task_store, trigger_mgr, config,
            )
        except Exception:
            _log.exception("bridge_loop iteration failed")
