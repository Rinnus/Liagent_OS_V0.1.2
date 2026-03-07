"""Session finalizer — session cleanup, fact extraction, and shutdown."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from ..logging import get_logger
from .memory import UserProfileStore

_log = get_logger("session_finalizer")


async def finalize_session(
    *,
    memory: Any,
    long_term: Any,
    engine: Any,
    prompt_builder: Any,
    journal: Any,
    session_id: str,
) -> None:
    """Extract summary and facts from the current session, save to long-term memory.

    Args:
        memory: ConversationMemory instance.
        long_term: LongTermMemory instance.
        engine: EngineManager for LLM generation.
        prompt_builder: PromptBuilder for constructing extraction prompts.
        journal: OptimizationJournal for logging session results.
        session_id: Current session identifier.
    """
    if memory.turn_count() < 3:
        return
    try:
        messages = memory.get_messages()

        summary_msgs = prompt_builder.build_summary_prompt(messages)
        summary = await engine.generate_extraction(
            summary_msgs, max_tokens=200, temperature=0.3
        )
        summary = summary.strip()
        if summary:
            long_term.save_summary(session_id, summary, memory.turn_count())

        fact_msgs = prompt_builder.build_fact_extraction_prompt(messages)
        fact_raw = await engine.generate_extraction(
            fact_msgs, max_tokens=300, temperature=0.2
        )
        facts = parse_facts(fact_raw.strip())
        if facts:
            facts = long_term.apply_source_confidence(facts)
            for f in facts:
                long_term.detect_conflicts(
                    f["fact"], f.get("category", "other"), f.get("confidence", 0.7)
                )
            long_term.save_facts(facts)

        # Step 3: Preference extraction (implicit channel)
        try:
            pref_msgs = prompt_builder.build_preference_extraction_prompt(messages)
            pref_raw = await engine.generate_extraction(
                pref_msgs, max_tokens=300, temperature=0.2
            )
            prefs = parse_preferences(pref_raw.strip())
            if prefs:
                profile_store = UserProfileStore(long_term.db_path)
                for p in prefs:
                    profile_store.merge_implicit(
                        p["dimension"], p["value"], p["signal_strength"]
                    )
        except Exception as e:
            _log.error("session_finalizer", e, action="preference_extraction")

        # Step 4: Behavior signal extraction (L1 — proactive intelligence)
        try:
            from .behavior import BehaviorSignalStore, parse_behavior_signals
            beh_msgs = prompt_builder.build_behavior_extraction_prompt(messages)
            beh_raw = await engine.generate_extraction(
                beh_msgs, max_tokens=200, temperature=0.2
            )
            beh_signals = parse_behavior_signals(beh_raw.strip())
            if beh_signals:
                beh_store = BehaviorSignalStore(long_term.db_path)
                for sig in beh_signals:
                    beh_store.record(
                        sig["signal_type"], sig["key"],
                        domain=sig["domain"],
                        source_origin="user",
                        metadata=sig["metadata"],
                        session_id=session_id,
                    )
                beh_store.flush()
        except Exception as e:
            _log.error("session_finalizer", e, action="behavior_extraction")

        journal.session_summary(
            session_id=session_id,
            summary=summary or "",
            turn_count=memory.turn_count(),
            facts_count=len(facts) if facts else 0,
        )
        try:
            await journal.generate_review(engine)
        except Exception as e:
            _log.error("session_finalizer", e, action="journal_review")
    except Exception as e:
        _log.error("session_finalizer", e, action="clear_memory_facts")


async def clear_memory_and_reset(
    *,
    memory: Any,
    finalize_fn: Any,
) -> str:
    """Finalize session, clear conversation, return new session_id.

    Args:
        memory: ConversationMemory instance.
        finalize_fn: Async callable for ``finalize_session``.

    Returns:
        New session UUID string.
    """
    await finalize_fn()
    memory.clear()
    return str(uuid.uuid4())


async def shutdown_runtime(
    *,
    mcp_bridge: Any,
    long_term: Any,
    tool_policy: Any,
) -> None:
    """Best-effort runtime teardown.

    Args:
        mcp_bridge: MCPBridge instance (or None).
        long_term: LongTermMemory instance.
        tool_policy: ToolPolicy instance.

    Returns the mcp_bridge state (always None after shutdown).
    """
    if mcp_bridge is not None:
        try:
            await mcp_bridge.shutdown()
        except Exception as e:
            _log.error("session_finalizer", e, action="mcp_shutdown")
    # Close long-term memory (includes event_log)
    try:
        long_term.close()
    except Exception:
        pass
    # Close tool policy DB connection
    try:
        tool_policy.close()
    except Exception:
        pass


_VALID_DIMENSIONS = {"language", "response_style", "tone", "domains",
                     "expertise_level", "data_preference"}
_VALID_STRENGTHS = {"weak", "moderate", "strong"}


def parse_preferences(raw: str) -> list[dict]:
    """Parse LLM preference extraction output into validated list."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        arr = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                arr = json.loads(m.group())
            except (json.JSONDecodeError, ValueError):
                return []
        else:
            return []
    if not isinstance(arr, list):
        return []
    result = []
    for entry in arr:
        if not isinstance(entry, dict):
            continue
        dim = entry.get("dimension", "")
        val = entry.get("value", "")
        strength = entry.get("signal_strength", "")
        if dim in _VALID_DIMENSIONS and val and strength in _VALID_STRENGTHS:
            result.append({"dimension": dim, "value": val, "signal_strength": strength})
    return result


def parse_facts(raw: str) -> list[dict]:
    """Parse LLM fact extraction output into list of dicts."""
    def _normalize(f: dict) -> dict | None:
        fact = str(f.get("fact", "")).strip()
        if not fact:
            return None
        category = str(f.get("category", "other") or "other")
        source = str(f.get("source", "llm_extract") or "llm_extract")
        conf_raw = f.get("confidence", 0.7)
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = 0.7
        conf = max(0.0, min(1.0, conf))
        return {"fact": fact, "category": category, "confidence": conf, "source": source}

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            normalized = [_normalize(f) for f in result if isinstance(f, dict)]
            return [f for f in normalized if f]
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                normalized = [_normalize(f) for f in result if isinstance(f, dict)]
                return [f for f in normalized if f]
        except json.JSONDecodeError:
            pass
    return []
