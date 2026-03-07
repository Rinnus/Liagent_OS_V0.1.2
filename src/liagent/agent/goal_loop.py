"""Reflection Loop: periodic LLM-driven goal review and decision execution."""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .behavior import PendingSuggestionStore
    from .goal_store import GoalStore
    from .task_queue import TaskStore

from ..tools import get_all_tools
from .behavior import is_quiet_hours
from .pattern_grouping import normalize_patterns, update_pattern_groups

_log = logging.getLogger(__name__)

REFLECTION_PROMPT = """You are a meta-cognitive agent reviewing user behavior patterns and active goals.

## Current Goals
{goals}

## Pattern Groups
{groups}

## Unlabeled Groups (need naming)
{unlabeled_groups}

## Recent User Feedback
{recent_feedback}

## User Profile
{user_profile}

## Available Tools
{tool_inventory}

## Constraints
{constraints}

Based on the above context, produce a JSON response with this exact schema:
{{
  "schema_version": "1.0",
  "discoveries": [
    {{
      "discovery_id": "disc-<uuid>",
      "hypothesis": "<what you discovered>",
      "evidence_refs": [],
      "confidence": 0.0-1.0,
      "impact": "low|medium|high",
      "risk": "low|medium|high",
      "novelty": true|false
    }}
  ],
  "decisions": [
    {{
      "decision_id": "dec-<uuid>",
      "type": "create_goal|create_task|retire_goal|adjust_goal|suggest_user|label_group",
      "confidence": 0.0-1.0,
      "requires_consent": true|false,
      "idempotency_key": "<scope:domain:semantic:time_bucket>",
      "goal_id": null,
      "params": {{}},
      "initial_tasks": [],
      "reason": ""
    }}
  ],
  "self_observations": ["<string>"],
  "next_review_minutes": 30
}}

Rules:
- Only create goals for clear, repeated patterns (3+ occurrences, 2+ days)
- Set requires_consent=true for anything high-risk or expensive
- Use idempotency keys to prevent duplicate actions
- Be conservative: prefer "suggest_user" over "create_goal" when uncertain
- Keep next_review_minutes between 15 and 120
"""


def _build_reflection_context(
    *,
    goals: list[dict],
    groups: list[dict],
    recent_feedback: list[dict],
    user_profile: dict,
    tool_inventory: list[dict],
    constraints: dict,
    unlabeled_groups: list[dict],
) -> dict:
    """Assemble context dict for reflection prompt."""
    return {
        "goals": json.dumps(goals, ensure_ascii=False, default=str) if goals else "No active goals.",
        "groups": json.dumps(groups, ensure_ascii=False, default=str) if groups else "No pattern groups.",
        "unlabeled_groups": json.dumps(unlabeled_groups, ensure_ascii=False, default=str) if unlabeled_groups else "None.",
        "recent_feedback": json.dumps(recent_feedback, ensure_ascii=False, default=str) if recent_feedback else "No recent feedback.",
        "user_profile": json.dumps(user_profile, ensure_ascii=False, default=str) if user_profile else "No profile data.",
        "tool_inventory": json.dumps(tool_inventory, ensure_ascii=False),
        "constraints": json.dumps(constraints, ensure_ascii=False),
    }


def _parse_and_validate(raw: str) -> dict:
    """Parse LLM output and validate against schema. Returns safe defaults on failure."""
    defaults = {
        "schema_version": "1.0",
        "discoveries": [],
        "decisions": [],
        "self_observations": [],
        "next_review_minutes": 30,
    }
    # Try to extract JSON from the response (may have markdown fences)
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        _log.warning("reflection: failed to parse LLM output as JSON")
        return defaults

    # Ensure required fields with defaults
    for key, default_val in defaults.items():
        if key not in parsed:
            parsed[key] = default_val

    # Validate decisions have required fields
    valid_decisions = []
    for d in parsed.get("decisions", []):
        if isinstance(d, dict) and "type" in d and "decision_id" in d:
            d.setdefault("confidence", 0.5)
            d.setdefault("requires_consent", True)
            d.setdefault("idempotency_key", f"auto:{d['decision_id']}")
            d.setdefault("params", {})
            d.setdefault("initial_tasks", [])
            d.setdefault("goal_id", None)
            d.setdefault("reason", "")
            valid_decisions.append(d)
    parsed["decisions"] = valid_decisions

    return parsed


def _execute_decision(d: dict, goal_store: GoalStore, task_store: TaskStore, config) -> list[tuple]:
    """Execute a single decision atomically.
    Returns list of post-commit actions: [("register_trigger", task_id, trigger_spec), ...]"""
    post_commit: list[tuple] = []
    dtype = d["type"]

    if dtype == "create_goal":
        if d.get("requires_consent"):
            # Write to outbox for user suggestion instead
            action_payload = {
                "create_goal": True,
                "objective": d["params"].get("objective", ""),
                "rationale": d["params"].get("rationale", ""),
                "domain": d["params"].get("domain"),
                "idempotency_key": d.get("idempotency_key"),
                "priority": d["params"].get("priority", 5),
                "budget": d["params"].get("budget"),
                "success_criteria": d["params"].get("success_criteria"),
                "source_group_id": d.get("params", {}).get("source_group_id"),
                "source_discovery_id": d.get("params", {}).get("source_discovery_id"),
            }
            with sqlite3.connect(goal_store.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                goal_store.insert_outbox(conn, "add_suggestion", {
                    "message": (
                        f"I noticed something worth tracking: "
                        f"{d['params'].get('objective', 'unknown')}.\n"
                        f"Reason: {d['params'].get('rationale', 'pattern detected')}\n"
                        f"Should I set up ongoing monitoring?"
                    ),
                    "delivery_mode": "session",
                    "domain": d["params"].get("domain", "general"),
                    "action_json": action_payload,
                    "pattern_key": d.get("idempotency_key"),
                })
            return post_commit

        goal_id = goal_store.create(
            idempotency_key=d.get("idempotency_key"),
            source="reflection",
            domain=d["params"].get("domain"),
            objective=d["params"].get("objective", ""),
            rationale=d["params"].get("rationale", ""),
            confidence=d.get("confidence", 0.5),
            priority=d["params"].get("priority", 5),
            budget_json=json.dumps(d["params"]["budget"]) if d["params"].get("budget") else None,
            success_criteria_json=(
                json.dumps(d["params"]["success_criteria"])
                if d["params"].get("success_criteria") else None
            ),
            source_group_id=d.get("params", {}).get("source_group_id"),
            source_discovery_id=d.get("params", {}).get("source_discovery_id"),
        )
        if goal_id is None:
            return post_commit  # idempotent skip

        goal_store.record_event(goal_id, "created", d)

        for t in d.get("initial_tasks", []):
            task = task_store.create_task_from_prompt(
                prompt=t.get("prompt", ""),
                source="reflection",
                goal_id=goal_id,
                dedup_key=t.get("idempotency_key"),
                risk_level=t.get("risk", "low"),
            )
            if task:
                goal_store.record_event(goal_id, "task_spawned", {"task_id": task["id"]})
                trigger = t.get("trigger", "once")
                post_commit.append(("register_trigger", task["id"], trigger))

    elif dtype == "create_task":
        goal_id = d.get("goal_id")
        task = task_store.create_task_from_prompt(
            prompt=d["params"].get("prompt", d.get("reason", "")),
            source="reflection",
            goal_id=goal_id,
            dedup_key=d.get("idempotency_key"),
            risk_level=d["params"].get("risk", "low"),
        )
        if task:
            if goal_id:
                goal_store.record_event(goal_id, "task_spawned", {"task_id": task["id"]})
            trigger = d.get("trigger", "once")
            post_commit.append(("register_trigger", task["id"], trigger))

    elif dtype == "retire_goal":
        goal_id = d.get("goal_id")
        if goal_id:
            goal_store.transition(goal_id, "retired", reason=d.get("reason", ""))
            goal_store.record_event(goal_id, "retired", {"reason": d.get("reason", "")})

    elif dtype == "adjust_goal":
        goal_id = d.get("goal_id")
        if goal_id:
            changes = d.get("params", d.get("changes", {}))
            goal_store.update(goal_id, **changes)
            goal_store.record_event(goal_id, "reviewed", {"changes": changes})

    elif dtype == "suggest_user":
        with sqlite3.connect(goal_store.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            goal_store.insert_outbox(conn, "add_suggestion", {
                "message": d.get("message", d.get("reason", "")),
                "delivery_mode": "session",
                "domain": d.get("domain", "general"),
                "pattern_key": d.get("idempotency_key"),
            })

    elif dtype == "label_group":
        group_id = d.get("group_id")
        label = d.get("label", d.get("params", {}).get("label", ""))
        if group_id and label:
            goal_store.set_group_label(group_id, label)

    return post_commit


async def _drain_outbox(goal_store: GoalStore, suggestion_store: PendingSuggestionStore):
    """Process pending outbox entries after all decisions committed."""
    entries = goal_store.drain_outbox()
    for entry in entries:
        payload: dict = {}
        try:
            if entry["action_type"] == "add_suggestion":
                payload_raw = entry.get("payload_json")
                if isinstance(payload_raw, str):
                    payload = json.loads(payload_raw)
                elif isinstance(payload_raw, dict):
                    payload = payload_raw
                ok = suggestion_store.add_simple(**payload)
                if ok:
                    goal_store.complete_outbox(entry["id"])
                else:
                    goal_store.complete_outbox(entry["id"], error_text="deduped")
            else:
                goal_store.retry_outbox(
                    entry["id"], error_text=f"unsupported action_type: {entry['action_type']}",
                )
        except Exception as exc:
            goal_store.retry_outbox(entry["id"], error_text=str(exc))


async def reflection_loop(
    goal_store: GoalStore,
    group_store: GoalStore,
    engine,
    pattern_detector,
    brain,
    task_store: TaskStore,
    trigger_mgr,
    suggestion_store: PendingSuggestionStore,
    config,
) -> None:
    """Long-running reflection loop. Periodically reviews goals and patterns."""
    interval_min = getattr(config.proactive, "reflection_interval_min", 30)

    while True:
        await asyncio.sleep(interval_min * 60)

        # Skip during quiet hours
        quiet_hours = getattr(config.proactive, "quiet_hours", "")
        if quiet_hours and is_quiet_hours(quiet_hours):
            continue

        try:
            # Flush buffered behavior signals before detection
            if hasattr(brain, 'behavior_signals') and brain.behavior_signals:
                try:
                    brain.behavior_signals.flush()
                except Exception:
                    pass

            # Pattern grouping FIRST (rule-based, no LLM) — must run before
            # the skip check so new systems can bootstrap from raw patterns
            raw_patterns = pattern_detector.detect()
            normalized = normalize_patterns(raw_patterns)
            update_pattern_groups(normalized, group_store)

            # Skip if no work to do (checked AFTER pattern grouping)
            if goal_store.count_active() == 0 and not group_store.has_recent_updates(hours=2):
                if not normalized:
                    interval_max = getattr(config.proactive, "reflection_interval_max", 120)
                    interval_min = min(interval_min * 1.5, interval_max)
                    continue

            # Auto-retire stale goals
            retire_days = getattr(config.proactive, "goal_auto_retire_days", 14)
            for stale in goal_store.get_stale_goals(days=retire_days):
                goal_store.transition(stale["id"], "retired", reason=f"auto-retired: no activity for {retire_days} days")
                goal_store.record_event(stale["id"], "retired", {"reason": f"auto-retired after {retire_days} days of inactivity"})

            # Build reflection context
            context = _build_reflection_context(
                goals=goal_store.get_due_for_review(),
                groups=group_store.get_recent_groups(limit=10),
                recent_feedback=(brain.long_term.get_recent_feedback(days=7)
                                 if hasattr(brain, 'long_term') and brain.long_term else []),
                user_profile=(brain.profile_store.get_all()
                              if hasattr(brain, 'profile_store') and brain.profile_store else {}),
                tool_inventory=[{"name": t.name, "description": t.description}
                                for t in get_all_tools().values()],
                constraints={
                    "remaining_daily_budget": goal_store.remaining_daily_budget(
                        max_goals=getattr(config.proactive, "max_new_goals_per_day", 3),
                        max_task_runs=getattr(config.proactive, "max_goal_tasks_per_day", 20),
                    ),
                    "active_goals": goal_store.count_active(),
                    "max_active_goals": getattr(config.proactive, "max_active_goals", 5),
                    "max_new_goals_today": (
                        getattr(config.proactive, "max_new_goals_per_day", 3)
                        - goal_store.count_created_today()
                    ),
                },
                unlabeled_groups=group_store.get_unlabeled_groups(limit=5),
            )

            # Try-lock LLM
            result = await engine.try_generate_reasoning(
                messages=[{"role": "user", "content": REFLECTION_PROMPT.format(**context)}],
                max_tokens=1000,
                temperature=0.3,
                enable_thinking=False,
                timeout=0.05,
            )
            if result is None:
                continue  # foreground active

            output = _parse_and_validate(result)

            # Execute decisions
            all_post_actions: list[tuple] = []
            for d in output["decisions"]:
                post_actions = _execute_decision(d, goal_store, task_store, config)
                all_post_actions.extend(post_actions)

            # Post-commit: register triggers
            for action in all_post_actions:
                if action[0] == "register_trigger":
                    task_id, trigger_spec = action[1], action[2]
                    if isinstance(trigger_spec, str) and trigger_spec.startswith("cron:"):
                        await trigger_mgr.register_cron(task_id, trigger_spec[5:])
                    else:
                        await trigger_mgr.register_once(task_id, delay_seconds=0)

            # Drain outbox
            await _drain_outbox(goal_store, suggestion_store)

            # Persist discoveries
            for disc in output.get("discoveries", []):
                goal_store.record_event(
                    goal_id=None,
                    event_type="discovery",
                    payload=disc,
                    summary=disc.get("hypothesis", ""),
                )

            # Self-observations
            for obs in output.get("self_observations", []):
                goal_store.record_event(None, "observation", {"text": obs})

            # Adaptive interval
            interval_max = getattr(config.proactive, "reflection_interval_max", 120)
            interval_min = max(15, min(
                output.get("next_review_minutes", 30),
                interval_max,
            ))

        except Exception:
            _log.exception("reflection_loop iteration failed")
