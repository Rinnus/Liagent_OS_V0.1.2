"""Optimization journal — records self-improvement events to daily Markdown.

Dual-layer design:
1. Mechanical recording: each optimization event is appended as raw data to
   ~/.liagent/cwork/{YYYY-MM-DD}.md
2. Coder review: at session end, Qwen3-Coder 30B analyzes accumulated events
   and generates a comprehensive review (changes + root cause + impact + suggestions).
"""

from datetime import datetime, timezone
from pathlib import Path


REVIEW_SYSTEM_PROMPT = (
    "You are LiAgent's self-optimization analyst. Generate a concise retrospective "
    "report from optimization events in this session.\n\n"
    "The report must include these four sections, each 1-3 sentences:\n"
    "1. **Change Summary**: what was optimized (new rules, new facts, reward/penalty updates)\n"
    "2. **Root Cause**: why these changes occurred (user feedback, self-evaluation, pattern matches)\n"
    "3. **Impact Assessment**: effect on capability (success trend, coverage, routing quality)\n"
    "4. **Next Improvements**: what to improve in the next iteration\n\n"
    "Answer in English, within 300 words. Do not restate raw events verbatim; provide analysis."
)

# Events that represent structural changes (not just counter increments)
_STRUCTURAL_EVENTS = frozenset({
    "skill_generated", "fact_learned", "fact_conflict",
    "experience_pruned", "memory_pruned", "session_summary",
})

# Hollow phrases that indicate a low-value Coder review
_HOLLOW_PHRASES = ("no significant", "no obvious", "none", "did not occur", "no need", "not necessary")


class OptimizationJournal:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (Path.home() / ".liagent" / "cwork")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._session_events: list[dict] = []

    def _today_path(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.base_dir / f"{today}.md"

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    def _append(self, event_type: str, **fields):
        """Append a raw event entry to today's Markdown and buffer it in memory."""
        ts = self._ts()
        path = self._today_path()

        lines = [f"\n---\n### [{ts}] {event_type}"]
        for k, v in fields.items():
            val = str(v)[:500]
            lines.append(f"- **{k}**: {val}")
        lines.append("")

        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self._session_events.append({"ts": ts, "category": event_type, **fields})

    # ── Event recording methods ──────────────────────────────────────

    def skill_generated(self, *, query, failed_answer, pattern, tool, confidence, source):
        self._append(
            "skill_generated",
            trigger="user_feedback / auto_analysis",
            query=query[:200],
            failed_reply=failed_answer[:100],
            new_rule=f"pattern={pattern}, tool={tool}, conf={confidence}",
            source=source,
        )

    def outcome_recorded(self, *, query, tool, success, source, reward=None):
        self._append(
            "outcome_recorded",
            query=query[:200],
            tool=tool or "none",
            result="success" if success else "failure",
            source=source,
            reward=f"{reward:.2f}" if reward is not None else "N/A",
        )

    def fact_learned(self, *, fact, category, confidence, source, is_new):
        self._append(
            "fact_learned",
            fact=fact[:300],
            category=category,
            confidence=f"{confidence:.2f}",
            source=source,
            status="new" if is_new else "updated",
        )

    def fact_conflict(self, *, new_fact, old_fact, old_confidence, demoted_confidence):
        self._append(
            "fact_conflict",
            new_fact=new_fact[:300],
            old_fact=old_fact[:300],
            old_confidence=f"{old_confidence:.2f}",
            demoted_confidence=f"{demoted_confidence:.2f}",
        )

    def session_summary(self, *, session_id, summary, turn_count, facts_count):
        self._append(
            "session_summary",
            session=session_id[:10],
            summary=summary[:300],
            turns=turn_count,
            extracted_facts=facts_count,
        )

    def experience_pruned(self, *, deleted_count):
        if deleted_count > 0:
            self._append("experience_pruned", deleted_rules=deleted_count)

    def memory_pruned(self, *, expired, low_confidence, excess):
        total = expired + low_confidence + excess
        if total > 0:
            self._append(
                "memory_pruned",
                expired=expired,
                low_confidence=low_confidence,
                over_limit=excess,
            )

    # ── Coder review ─────────────────────────────────────────────────

    async def generate_review(self, engine) -> str | None:
        """Generate a comprehensive review using Qwen3-Coder 30B.

        Three-layer filtering to avoid wasted inference and hollow output:
        1. Structural gate: skip if only outcome_recorded events (no real change)
        2. Length gate: Coder output < 50 chars is discarded
        3. Hollow phrase gate: boilerplate like "no significant change" is discarded
        """
        if not self._session_events:
            return None

        has_structural = any(
            e["category"] in _STRUCTURAL_EVENTS for e in self._session_events
        )
        if not has_structural:
            self._session_events.clear()
            return None

        event_summary = "\n".join(
            f"- [{e['ts']}] {e['category']}: "
            + ", ".join(
                f"{k}={v}" for k, v in e.items() if k not in ("ts", "category")
            )
            for e in self._session_events
        )

        messages = [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": f"Optimization events from this session:\n\n{event_summary}"},
        ]
        review = await engine.generate_reasoning(
            messages, max_tokens=800, temperature=0.3,
            enable_thinking=False,
        )
        review = review.strip()

        if not review or len(review) < 50:
            self._session_events.clear()
            return None

        if any(p in review[:100] for p in _HOLLOW_PHRASES):
            self._session_events.clear()
            return None

        ts = self._ts()
        path = self._today_path()
        block = f"\n---\n## [{ts}] Session Retrospective (Coder 30B)\n\n{review}\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)

        self._session_events.clear()
        return review
