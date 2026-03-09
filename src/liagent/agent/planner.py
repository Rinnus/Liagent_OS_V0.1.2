"""Policy review — LLM-based tool action review."""

import json
import logging
import re
from dataclasses import dataclass

from ..engine.engine_manager import EngineManager
from .prompt_builder import PromptBuilder

_log = logging.getLogger(__name__)

_JSON_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

# ── Plan schema constants ────────────────────────────────────────────────
_MAX_PLAN_STEPS = 8

# Heuristic patterns that suggest a request needs multi-step planning
_MULTI_STEP_SIGNALS = re.compile(
    r"(compare|contrast|then|after(?:ward)?|next|first.+then|and|also|"
    r"each|multiple|several|steps?|search.+summarize|find.+analy(?:s|z)e|"
    r"find.+compare)",
    re.IGNORECASE,
)
# Count entities: uppercase tickers (AAPL, TSLA), URLs, numeric codes
# NO IGNORECASE — only match actual uppercase tickers to avoid false positives
_ENTITY_COUNT_RE = re.compile(r"[A-Z]{2,5}(?=\s|,|;|$)|https?://\S+|\d{5,}")


def _should_plan(query: str) -> bool:
    """Heuristic: does this query warrant multi-step planning?"""
    q = query.strip()
    if len(q) < 8:
        return False
    if _MULTI_STEP_SIGNALS.search(q):
        return True
    entities = _ENTITY_COUNT_RE.findall(q)
    if len(entities) >= 3:
        return True
    return False


def _parse_plan_json(raw: str) -> dict | None:
    """Parse LLM output into a validated plan dict. Returns None on failure."""
    text = str(raw or "").strip()
    if not text:
        return None
    # Try extracting from markdown fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidates = [text]
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())
    # Try each candidate
    for candidate in candidates:
        for blob in _iter_json_object_blobs(candidate):
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if "steps" not in obj or not isinstance(obj["steps"], list):
                continue
            if len(obj["steps"]) == 0:
                continue
            # Validate and cap steps
            steps = obj["steps"][:_MAX_PLAN_STEPS]
            for i, s in enumerate(steps):
                s.setdefault("id", f"s{i + 1}")
                s.setdefault("title", f"Step {i + 1}")
                s.setdefault("tool_hint", None)
                s.setdefault("done_criteria", "")
                s.setdefault("status", "pending")
                s.setdefault("evidence_ref", None)
            return {
                "goal": str(obj.get("goal", "")).strip() or "Task",
                "steps": steps,
            }
    return None


def format_plan_status(goal: str, steps: list[dict], current_idx: int) -> str:
    """Format plan as a status block for system prompt injection."""
    lines = [f"[Current Plan]", f"Goal: {goal}"]
    for i, s in enumerate(steps):
        status_icon = {"done": "\u2705", "skipped": "\u23ed\ufe0f"}.get(s["status"], "\u2b1c")
        marker = " \u2190 YOU ARE HERE" if i == current_idx and s["status"] == "pending" else ""
        evidence = ""
        if s.get("evidence_ref"):
            evidence = f" (evidence: {s['evidence_ref'][:80]})"
        lines.append(f"Step {i + 1}: {status_icon} {s['title']}{evidence}{marker}")
    if current_idx < len(steps):
        current = steps[current_idx]
        lines.append(f"\nFocus on Step {current_idx + 1}. Complete it before moving on.")
        lines.append(f"Done criteria: {current.get('done_criteria', '')}")
    return "\n".join(lines)


def should_block_completion(steps: list[dict], plan_idx: int, plan_total: int) -> tuple[bool, str]:
    """Check if the plan has pending non-final steps. Returns (blocked, reason).

    Logic: The last step is assumed to be synthesis/summarization.
    Block completion if ANY non-final step (steps[:-1]) is still 'pending'.
    This avoids the false-pass window where plan_idx reaches the last step
    but earlier steps are still pending due to skips or re-ordering.
    """
    if not steps:
        return False, ""
    # Check all non-final steps: they must be done or skipped
    non_final = steps[:-1] if len(steps) > 1 else []
    pending_non_final = [s for s in non_final if s.get("status") == "pending"]
    if pending_non_final:
        titles = [s["title"] for s in pending_non_final]
        return True, (
            f"There are still {len(pending_non_final)} unfinished steps: {titles}. "
            "Continue with the next step and do not summarize yet."
        )
    return False, ""


@dataclass
class ToolPolicyReview:
    allow: bool
    risk: str = "medium"
    needs_confirmation: bool = False
    reason: str = ""


class TaskPlanner:
    def __init__(self, engine: EngineManager, prompt_builder: PromptBuilder):
        self.engine = engine
        self.prompt_builder = prompt_builder

    async def review_tool_action(
        self,
        *,
        user_input: str,
        step: object | None,
        tool_name: str,
        tool_args: dict,
        capability_desc: str,
    ) -> ToolPolicyReview:
        messages = self.prompt_builder.build_tool_policy_review_prompt(
            user_input=user_input,
            step=step,
            tool_name=tool_name,
            tool_args=tool_args,
            capability_desc=capability_desc,
        )
        raw = await self.engine.generate_reasoning(
            messages, max_tokens=120, temperature=0.1, enable_thinking=False,
        )
        return _parse_policy_review(raw)

    async def decompose(self, query: str, tool_descriptions: str) -> dict | None:
        """Decompose a user request into execution steps. Returns plan dict or None."""
        prompt = [
            {"role": "system", "content": (
                "Decompose this user request into execution steps.\n"
                f"Available tools:\n{tool_descriptions}\n\n"
                'Output JSON: {"goal": string, "steps": [{"id": "s1", "title": string, '
                '"tool_hint": string|null, "done_criteria": string, '
                '"status": "pending", "evidence_ref": null}]}\n\n'
                "Rules:\n"
                "- Last step should synthesize/summarize results\n"
                "- Each step = one tool call or one reasoning action\n"
                "- 2-8 steps max\n"
                "- tool_hint = suggested tool name, null for pure reasoning"
            )},
            {"role": "user", "content": query},
        ]
        raw = await self.engine.generate_extraction(
            prompt, max_tokens=400, temperature=0.2,
        )
        return _parse_plan_json(raw)

    async def replan(self, goal: str, steps: list[dict], reason: str,
                     tool_descriptions: str) -> list[dict] | None:
        """Re-plan: adjust pending steps while keeping completed ones frozen."""
        frozen = [s for s in steps if s["status"] in ("done", "skipped")]
        pending = [s for s in steps if s["status"] == "pending"]
        prompt = [
            {"role": "system", "content": (
                "Adjust the remaining steps of this plan.\n"
                f"Goal: {goal}\n"
                f"Completed steps (FROZEN, do not change): {json.dumps(frozen, ensure_ascii=False)}\n"
                f"Pending steps (can be modified): {json.dumps(pending, ensure_ascii=False)}\n"
                f"Reason for re-plan: {reason}\n"
                f"Available tools:\n{tool_descriptions}\n\n"
                "Output JSON: only the NEW pending steps array.\n"
                "Rules: 2-6 new pending steps max. Keep same schema."
            )},
        ]
        raw = await self.engine.generate_extraction(
            prompt, max_tokens=300, temperature=0.2,
        )
        # Parse as array of steps
        text = str(raw or "").strip()
        # Strip markdown fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()
        for blob in _iter_json_object_blobs(f'{{"steps": {text}}}'):
            try:
                obj = json.loads(blob)
                if isinstance(obj.get("steps"), list) and obj["steps"]:
                    new_steps = obj["steps"][:6]
                    for i, s in enumerate(new_steps):
                        s.setdefault("id", f"s{len(frozen) + i + 1}")
                        s.setdefault("status", "pending")
                        s.setdefault("evidence_ref", None)
                    return new_steps
            except json.JSONDecodeError:
                continue
        # Try parsing raw as a JSON array directly
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and arr:
                for i, s in enumerate(arr[:6]):
                    s.setdefault("id", f"s{len(frozen) + i + 1}")
                    s.setdefault("status", "pending")
                    s.setdefault("evidence_ref", None)
                return arr[:6]
        except (json.JSONDecodeError, TypeError):
            pass
        return None


# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------

def _clean_json_text(raw: str) -> str:
    return _JSON_CTRL_RE.sub("", str(raw or ""))


def _iter_json_object_blobs(text: str) -> list[str]:
    """Extract candidate JSON object blobs with a brace-depth scan."""
    s = str(text or "")
    out: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for idx, ch in enumerate(s):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(s[start:idx + 1])
                    start = -1
    return out


def _json_object_candidates(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    # Strip <think>...</think> blocks (cloud APIs may wrap reasoning)
    text_no_think = _THINK_BLOCK_RE.sub("", text).strip()
    cleaned = _clean_json_text(text)
    candidates: list[str] = [text, cleaned]
    if text_no_think and text_no_think != text:
        candidates.append(text_no_think)
        candidates.append(_clean_json_text(text_no_think))
    for block in _JSON_FENCE_RE.findall(text):
        block_clean = _clean_json_text(block).strip()
        if block_clean:
            candidates.append(block_clean)
    for blob in _iter_json_object_blobs(text):
        cleaned_blob = _clean_json_text(blob).strip()
        if cleaned_blob:
            candidates.append(cleaned_blob)
    for blob in _iter_json_object_blobs(cleaned):
        if blob.strip():
            candidates.append(blob.strip())
    seen: set[str] = set()
    uniq: list[str] = []
    for item in candidates:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _parse_json_object(raw: str) -> dict | None:
    for candidate in _json_object_candidates(raw):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------------------
# Policy review parser
# ---------------------------------------------------------------------------

def _parse_policy_review(raw: str) -> ToolPolicyReview:
    raw = str(raw or "").strip()
    if not raw:
        # Empty advisory output is common under transient model/network load.
        # Keep static policy as the hard gate and avoid warning spam.
        _log.debug("policy_review_empty_response")
        return ToolPolicyReview(
            allow=True,
            risk="medium",
            needs_confirmation=False,
            reason="policy review unavailable (ignored)",
        )

    def _coerce(obj: dict) -> ToolPolicyReview:
        allow = bool(obj.get("allow", False))
        risk = str(obj.get("risk", "medium")).strip().lower()
        if risk not in {"low", "medium", "high"}:
            risk = "medium"
        needs_confirmation = bool(obj.get("needs_confirmation", False))
        reason = str(obj.get("reason", "")).strip()
        return ToolPolicyReview(
            allow=allow,
            risk=risk,
            needs_confirmation=needs_confirmation,
            reason=reason,
        )

    obj = _parse_json_object(raw)
    if isinstance(obj, dict):
        return _coerce(obj)

    low = _clean_json_text(raw).lower()
    if any(k in low for k in ('"allow": false', "allow false", "deny", "blocked")):
        return ToolPolicyReview(
            allow=False,
            risk="medium",
            needs_confirmation=False,
            reason="policy review denied by heuristic fallback",
        )
    if any(k in low for k in ('"risk": "high"', "risk high", "high risk")):
        return ToolPolicyReview(
            allow=True,
            risk="high",
            needs_confirmation=True,
            reason="policy review high risk heuristic fallback",
        )
    # Parse failure should not add extra confirmation friction.
    # Static policy rules remain the hard gate.
    _log.warning("policy_review_parse_failed raw_len=%d raw_head=%.120s", len(raw), raw)
    return ToolPolicyReview(
        allow=True,
        risk="medium",
        needs_confirmation=False,
        reason="policy review parse failed (ignored)",
    )
