"""Dynamic prompt builder — assembles system prompts with long-term memory context."""

import re

from ..tools import get_all_tools
from .memory import LongTermMemory, UserProfileStore

# ── Shared rules — split into CORE (all tiers) and REASONING (standard+deep) ──
_RULES_CORE = """\
## Response Rules

Return plain text and start directly with the answer.
Match response depth to question complexity:
- Greeting or simple fact -> 1-2 sentences
- Single data lookup (price/weather) -> 3-5 sentences with key numbers
- Analysis/comparison/causal question -> conclusion first, then 8-15 sentences of support
- Deep research -> structured sections with short headings

## Data Accuracy

For real-time data (price, market, weather, news), call tools before answering.
Follow-up questions about real-time data require a fresh lookup.
All concrete numbers must come from tool output.
Do not silently replace user-provided entities (for example, if user says AMD, keep AMD).
Preserve source units and conversions unless the user explicitly asks for conversion.
"""

_RULES_REASONING = """\
## Multi-turn Dialogue

Use full conversation context for follow-ups:
- "Why?" -> provide causal analysis for the previous conclusion
- "Go deeper" -> expand the previous answer without repeating
- "From another angle" -> provide an alternative analytical perspective
For real-time follow-ups, re-check with tools.
When multiple sources conflict, explicitly call out the discrepancy.
"""


def _build_shared_rules(tier: str) -> str:
    """Assemble shared rules based on service tier."""
    if tier == "realtime_voice":
        return _RULES_CORE
    return _RULES_CORE + "\n" + _RULES_REASONING


_SENSITIVE_PERSONAL_FACT_RE = re.compile(
    r"\b(?:my name is|name is|preferred name|call me|nickname|go by|address me as|refer to me as|full name)\b",
    re.IGNORECASE,
)


def _is_prompt_safe_fact(fact: dict) -> bool:
    """Allow general memory facts, but keep names and forms of address out of prompts."""
    text = str(fact.get("fact", "") or "").strip()
    if not text:
        return False
    category = str(fact.get("category", "") or "").strip().lower()
    if category != "personal":
        return True
    return _SENSITIVE_PERSONAL_FACT_RE.search(text) is None


# ── Tier-based memory injection config ─────────────────────────────────
_TIER_MEMORY_CONFIG = {
    "realtime_voice": {"min_confidence": 0.80, "max_facts": 3, "max_summaries": 2},
    "standard_chat":  {"min_confidence": 0.65, "max_facts": 6, "max_summaries": 5},
    "deep_task":      {"min_confidence": 0.50, "max_facts": 8, "max_summaries": 5},
}

# ── VLM (4B) system prompt — split into composable sections ────────────
_VLM_HEADER = """\
You are LiAgent, a local-first assistant. Current local time: {current_datetime}.

{shared_rules}"""

_SEARCH_RULES = """\
## Search Quality

The `web_search.query` must use focused English keywords (3-8 words).
For follow-ups, rebuild the query using conversation context.
User frustration or urgency is not search intent.
After search, the system may auto-fetch pages. Analyze the returned data and synthesize an answer.
"""

_VLM_SEARCH = _SEARCH_RULES + """\
Examples:
User: Hi
Assistant: Hi, what can I help you with?

User: What can you do?
Assistant: {capability_summary}

User: Check Google stock price
Assistant: <tool_call>{{"name": "web_search", "args": {{"query": "Google GOOG stock price today", "timelimit": "d"}}}}</tool_call>

User: AAPL price?
Assistant: <tool_call>{{"name": "web_search", "args": {{"query": "AAPL stock price today", "timelimit": "d"}}}}</tool_call>

User: Latest gold price
Assistant: <tool_call>{{"name": "web_search", "args": {{"query": "gold price today USD", "timelimit": "d"}}}}</tool_call>

User: Analyze Google's latest earnings
Assistant: <tool_call>{{"name": "web_search", "args": {{"query": "Alphabet Google {current_quarter} earnings revenue profit", "timelimit": "m"}}}}</tool_call>

User: How do I read a file in Python?
Assistant: Use `open()`, for example `with open("file.txt") as f: text = f.read()`.
"""

_VLM_TOOLS = """\
{long_term_context}## Tools

{tool_descriptions}

Tool selection guide:
- "Do X in N minutes" or recurring requests -> `create_task` (never execute immediately)
- List tasks -> `list_tasks`
- Cancel task(s) -> `delete_task` or `delete_all_tasks`
- List files/directories -> `list_dir`
- Read file text -> `read_file`
- Describe/analyze image -> `describe_image`
- Internet search -> `web_search` (refined English query)
- Fetch page body -> `web_fetch` (for URLs from search)
{exec_tool_hints}
Use this format when calling tools, one call at a time:
<tool_call>{{"name": "tool_name", "args": {{"arg_name": "value"}}}}</tool_call>

Tool results are stored as `$tool_name_result` and can be referenced later.
Example:
Step 1: <tool_call>{{"name": "web_search", "args": {{"query": "..."}}}}</tool_call>
{exec_tool_examples}
Operational requests are only complete after calling the required tool.
{experience_constraint}"""

_VLM_TOOLS_API = """\
{long_term_context}## Tools

Tool schemas are supplied separately via the API `tools` parameter.
Use native function/tool calling from the model API.
Do not print `<tool_call>...</tool_call>` blocks in plain text.
When calling a tool, output ONLY the tool call. Do not write analysis, reasoning, or narration before a tool call.

Tool selection guide:
- "Do X in N minutes" or recurring requests -> `create_task` (never execute immediately)
- List tasks -> `list_tasks`
- Cancel task(s) -> `delete_task` or `delete_all_tasks`
- List files/directories -> `list_dir`
- Read file text -> `read_file`
- Describe/analyze image -> `describe_image`
- Internet search -> `web_search` (refined English query)
- Fetch page body -> `web_fetch` (for URLs from search)
{exec_tool_hints}
Operational requests are only complete after calling the required tool.
{experience_constraint}"""


# ── Conditional tool hints (excluded under "research" profile) ──────────
_EXEC_TOOL_HINTS = """\
- Execute code/compute/open file -> `python_exec`
- Write file -> `write_file`"""

_EXEC_TOOL_EXAMPLES = """\
Step 2: <tool_call>{{"name": "write_file", "args": {{"path": "...", "content": "$web_search_result"}}}}</tool_call>
"""

def _build_exec_tool_hints(available_tool_names: set[str] | None) -> tuple[str, str]:
    if available_tool_names is None:
        return _EXEC_TOOL_HINTS, _EXEC_TOOL_EXAMPLES
    hints: list[str] = []
    if "python_exec" in available_tool_names:
        hints.append("- Execute code/compute/open file -> `python_exec`")
    if "write_file" in available_tool_names:
        hints.append("- Write file -> `write_file`")
    examples = _EXEC_TOOL_EXAMPLES if "write_file" in available_tool_names else ""
    return "\n".join(hints), examples


def _dynamic_capability_summary(
    tool_profile: str,
    *,
    available_tool_names: set[str] | None = None,
) -> str:
    """Build capability summary dynamically from the current tool registry."""
    from .capability_inventory import build_capability_summary
    tool_names = available_tool_names
    if tool_names is None:
        tools = get_all_tools()
        tool_names = set(tools.keys())
    return build_capability_summary(tool_names, tool_profile=tool_profile)


def _build_vlm_template(tier: str, *, api_mode: bool = False) -> str:
    """Assemble VLM template based on service tier."""
    parts = [_VLM_HEADER]
    if tier != "realtime_voice":
        parts.append(_VLM_SEARCH)
    parts.append(_VLM_TOOLS_API if api_mode else _VLM_TOOLS)
    return "\n".join(parts)

# ── Coder (30B) system prompt — split into composable sections ──────────
_CODER_HEADER = """\
You are LiAgent, a local-first assistant. Current local time: {current_datetime}.

{shared_rules}"""

_CODER_SEARCH = _SEARCH_RULES + """\
Examples:
User: "Check Google's latest stock price" -> query: "Google GOOG stock price today"
User: "This data looks wrong, search again" -> rebuild query from context, for example "Google {current_quarter} revenue net income official"
"""

_CODER_SYNTHESIS = """\
## Data Synthesis

After multiple tool results are available:
1. Extract the core conclusion first
2. Support conclusions with cross-source evidence
3. If sources conflict, report both values and the discrepancy
4. Mark data recency (for example, "as of 2025 Q4")
"""

_CODER_TOOLS = """\
{long_term_context}Tool definitions are injected by the system. Call tools directly when needed.
When calling a tool, output ONLY the tool call XML. Do not write analysis, reasoning, or narration before a tool call.

## Tool Decision Policy

Operational requests are only complete after the required tool call.
Classify requests before acting:
1. Delayed/recurring timing intent ("in N minutes", "every day") -> must call `create_task`, do not execute immediately
2. Task management -> `list_tasks` / `delete_task` / `delete_all_tasks`
3. Real-time data (price/weather/news/latest) -> `web_search` (English query), then `web_fetch` for source pages when needed
4. File/code operations -> `write_file`; code execution -> `python_exec`
5. Pure knowledge/chat -> direct answer

When the user says "do X in N minutes", create a delayed task with `create_task` and do not run X immediately.
{experience_constraint}
Examples:
User: Check Google's latest stock price
Assistant: <tool_call>web_search<arg_key>query</arg_key><arg_value>Google GOOG stock price today</arg_value><arg_key>timelimit</arg_key><arg_value>d</arg_value></tool_call>

User: Remind me to drink water in 3 minutes
Assistant: <tool_call>create_task<arg_key>description</arg_key><arg_value>Remind me to drink water in 3 minutes</arg_value></tool_call>

User: What tasks do I have?
Assistant: <tool_call>list_tasks</tool_call>

User: Cancel the water reminder task
Assistant: <tool_call>delete_task<arg_key>task_id</arg_key><arg_value>(check list_tasks first for the ID)</arg_value></tool_call>

User: Cancel all tasks
Assistant: <tool_call>delete_all_tasks</tool_call>

User: What is Python's GIL?
Assistant: The GIL is a global interpreter lock that allows only one thread to execute Python bytecode at a time in a process.

User: What can you do right now?
Assistant: {capability_summary}
"""


def _build_coder_template(tier: str) -> str:
    """Assemble Coder template based on service tier."""
    parts = [_CODER_HEADER]
    if tier != "realtime_voice":
        parts.append(_CODER_SEARCH)
    if tier == "deep_task":
        parts.append(_CODER_SYNTHESIS)
    parts.append(_CODER_TOOLS)
    return "\n".join(parts)

_TOOL_POLICY_REVIEW_SYSTEM = """\
You are a policy reviewer. You do not execute tools; you decide if a tool call is safe and necessary.
Current time: {current_datetime}.

Output JSON:
{{
  "allow": true/false,
  "risk": "low|medium|high",
  "needs_confirmation": true/false,
  "reason": "short reason"
}}

Rules:
- If the tool call is unrelated to user intent, set `allow=false`
- Preparatory steps (for example list directory before file edit) can be `allow=true`
- Screen capture or sensitive writes should set `needs_confirmation=true`
- Read-only operations (search/list/read) default to `allow=true`, `risk="low"`
- Treat the provided current date/time as authoritative
- Output JSON only
"""

_SUMMARY_SYSTEM = """\
Summarize the core content of the following conversation in 2-3 sentences.
Output summary only."""

_FACT_EXTRACTION_SYSTEM = """\
Extract key user facts from the conversation (for example location, preferences, work).

Output JSON array where each item has:
{{"fact": "fact text", "category": "category", "confidence": 0.0-1.0, "source": "llm_extract"}}
Allowed categories: location, preference, work, personal, other

If no facts are extractable, output []
Output JSON only."""

_PREFERENCE_EXTRACTION_SYSTEM = """\
Analyze this conversation and extract any user preferences or habits you can observe.
Output valid JSON array only. Each entry:
{{"dimension": "...", "value": "...", "signal_strength": "weak|moderate|strong"}}

Dimensions to look for:
- language: primary language used (zh/en/mixed)
- response_style: preferred response length (concise/detailed/balanced)
- tone: communication formality (casual/professional/neutral)
- domains: topics of interest (comma-separated, lowercase)
- expertise_level: user's technical sophistication (beginner/intermediate/expert)
- data_preference: citation preference (sources_required/trust_llm/balanced)

Note: do NOT extract timezone — it is explicit-only.
Only output dimensions with clear evidence. Do not guess. Output [] if no preferences observed.
"""

_BEHAVIOR_EXTRACTION_SYSTEM = """\
Analyze this conversation and extract behavioral signals.
Output valid JSON array only. Each entry:
{"signal_type": "...", "key": "...", "domain": "...", "confidence": 0.0,
 "metadata": {"reason": "...", "entities": [], "frequency_hint": "daily|weekly|ad-hoc",
              "evidence_turns": [2, 5]}}

Signal types:
- intent: recurring query intents (e.g., "check_stock_price", "summarize_news")
- time_habit: temporal patterns (e.g., "morning_checkin", "weekly_review")
- automation_candidate: manual tasks that could be automated

Constraints:
- Max 6 entries, deduplicate by key
- Only include signals with clear evidence (confidence >= 0.6)
- Do NOT extract timezone — explicit-only
- Output [] if no behavioral signals observed"""


# ── Professional tone override for formal queries ────────────────────
_PROFESSIONAL_KEYWORDS = re.compile(
    r"(report|investment\s*advice|evaluation|analysis\s*report|research\s*report)"
)

_PROFESSIONAL_OVERRIDE = """\

## Professional mode (for this query)

Use formal, objective language:
- Formal written style
- Attach source and recency to concrete numbers (for example "as of 2025 Q4")
- Mark confidence level for conclusions/recommendations (high/medium/low)
- Include risk notes and disclaimer
- Use structured format (headings, tables, bullet points)
"""


class PromptBuilder:
    def __init__(self, long_term: LongTermMemory):
        self.long_term = long_term
        self._profile_store = UserProfileStore(long_term.db_path)

    def _long_term_context(self, query: str = "", tier: str = "standard_chat") -> str:
        """Build context section from long-term memory, filtered by tier confidence."""
        cfg = _TIER_MEMORY_CONFIG.get(tier, _TIER_MEMORY_CONFIG["standard_chat"])
        min_conf = cfg["min_confidence"]
        max_facts = cfg["max_facts"]
        max_summaries = cfg["max_summaries"]

        if query:
            # get_relevant_facts internally filters >= 0.65; fetch extra then client-filter
            raw_facts = self.long_term.get_relevant_facts(query, limit=max_facts * 2)
            evidence_chunks = []
            get_ev = getattr(self.long_term, "get_relevant_evidence", None)
            if callable(get_ev):
                try:
                    evidence_chunks = get_ev(query, limit=max_facts * 2)
                except Exception:
                    evidence_chunks = []
            # Fallback: if BM25 found nothing, use all facts so the LLM
            # still has general memory context about the user.
            if not raw_facts:
                raw_facts = self.long_term.get_all_facts(min_confidence=min_conf)
        else:
            raw_facts = self.long_term.get_all_facts(min_confidence=min_conf)
            evidence_chunks = []

        visible_facts = [f for f in raw_facts if _is_prompt_safe_fact(f)]
        if len(visible_facts) != len(raw_facts):
            evidence_chunks = []
        raw_facts = visible_facts

        # Client-side confidence filter for tiers with different thresholds
        facts = [f for f in raw_facts if f.get("confidence", 0.7) >= min_conf][:max_facts]
        # TODO: move tier-specific logic to skills/router.py
        # For deep_task, also include lower-confidence facts with annotation
        low_conf_facts: list[dict] = []
        if tier == "deep_task" and min_conf < 0.65:
            low_conf_facts = [
                f for f in raw_facts
                if 0.50 <= f.get("confidence", 0.7) < 0.65
            ][:max_facts - len(facts)]

        summaries = self.long_term.get_recent_summaries(limit=max_summaries)

        if not facts and not low_conf_facts and not summaries:
            portrait = self._profile_store.compile_portrait()
            return portrait if portrait else ""

        parts = []
        if facts or low_conf_facts:
            parts.append("## User Facts")
            for i, f in enumerate(facts):
                conf = f.get("confidence", 0.7)
                source_ref = ""
                if i < len(evidence_chunks):
                    source_ref = getattr(evidence_chunks[i], "source_ref", "") or ""
                if not source_ref:
                    source_ref = f"memory:key_facts:{f.get('fact_key', '') or i}"
                parts.append(f"- {f['fact']} (confidence {conf:.2f}, source {source_ref})")
            for f in low_conf_facts:
                conf = f.get("confidence", 0.7)
                parts.append(f"- {f['fact']} (confidence {conf:.2f}, pending verification)")
            parts.append("")

        if summaries:
            parts.append("## Recent Conversation Summary")
            for s in summaries:
                parts.append(f"- {s}")
            parts.append("")

        portrait = self._profile_store.compile_portrait()
        context = "\n".join(parts) + "\n"
        if portrait:
            return portrait + context
        return context

    @staticmethod
    def _local_datetime_str() -> str:
        """Current local datetime as a human-readable string, e.g. '2026-02-12 Thu 10:35 PST'."""
        from datetime import datetime
        now = datetime.now().astimezone()
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        wd = weekdays[now.weekday()]
        tz_name = now.strftime("%Z") or now.strftime("%z")
        return f"{now.strftime('%Y-%m-%d')} {wd} {now.strftime('%H:%M:%S')} {tz_name}"

    @staticmethod
    def _current_quarter() -> str:
        """Most recent completed quarter, e.g. 'Q4 2025' when in Q1 2026."""
        from datetime import date
        today = date.today()
        q = (today.month - 1) // 3 + 1
        prev_q = q - 1 if q > 1 else 4
        prev_y = today.year if q > 1 else today.year - 1
        return f"Q{prev_q} {prev_y}"

    def build_system_prompt(
        self,
        query: str = "",
        experience_constraint: str = "",
        tier: str = "standard_chat",
        *,
        include_tool_descriptions: bool = True,
        api_mode: bool = False,
        tool_profile: str = "research",
        available_tool_names: set[str] | None = None,
    ) -> str:
        """Build the VLM execution system prompt with injected long-term memory and experience."""
        desc = ""
        if include_tool_descriptions:
            tools = get_all_tools()
            if not tools:
                desc = "(no tools available)"
            else:
                if available_tool_names is not None:
                    tools = {k: v for k, v in tools.items() if k in available_tool_names}
                desc = "\n".join(t.schema_text() for t in tools.values()) if tools else "(no tools available)"

        # Only include python_exec/write_file hints when the profile allows them
        include_exec = tool_profile not in {"research", "minimal"}
        exec_hints = ""
        exec_examples = ""
        if include_exec:
            exec_hints, exec_examples = _build_exec_tool_hints(available_tool_names)
        capability = _dynamic_capability_summary(
            tool_profile,
            available_tool_names=available_tool_names,
        )

        context = self._long_term_context(query=query, tier=tier)
        template = _build_vlm_template(tier, api_mode=api_mode)
        prompt = template.format(
            current_datetime=self._local_datetime_str(),
            current_quarter=self._current_quarter(),
            shared_rules=_build_shared_rules(tier),
            long_term_context=context,
            tool_descriptions=desc,
            experience_constraint=experience_constraint,
            exec_tool_hints=exec_hints,
            exec_tool_examples=exec_examples,
            capability_summary=capability,
        )
        if _PROFESSIONAL_KEYWORDS.search(query):
            prompt += _PROFESSIONAL_OVERRIDE
        return prompt

    def build_system_prompt_for_api(
        self,
        query: str = "",
        experience_constraint: str = "",
        tier: str = "standard_chat",
        *,
        tool_protocol: str = "openai_function",
        tool_profile: str = "research",
        available_tool_names: set[str] | None = None,
    ) -> str:
        """Build API-optimized VLM prompt to avoid duplicate inline tool schema tokens."""
        protocol = str(tool_protocol or "openai_function").strip().lower()
        if protocol not in {"", "auto", "openai_function"}:
            # Non-function-call protocols still need inline schema + XML guidance.
            return self.build_system_prompt(
                query=query,
                experience_constraint=experience_constraint,
                tier=tier,
                include_tool_descriptions=True,
                api_mode=False,
                tool_profile=tool_profile,
                available_tool_names=available_tool_names,
            )
        return self.build_system_prompt(
            query=query,
            experience_constraint=experience_constraint,
            tier=tier,
            include_tool_descriptions=False,
            api_mode=True,
            tool_profile=tool_profile,
            available_tool_names=available_tool_names,
        )

    def build_system_prompt_for_coder(
        self, query: str = "", experience_constraint: str = "", tier: str = "standard_chat"
    ) -> str:
        """Build system prompt for 30B Coder — no JSON tool format instructions.

        Tool definitions are injected via tokenizer's tools= parameter instead.
        """
        capability = _dynamic_capability_summary("full")
        context = self._long_term_context(query=query, tier=tier)
        template = _build_coder_template(tier)
        prompt = template.format(
            current_datetime=self._local_datetime_str(),
            current_quarter=self._current_quarter(),
            shared_rules=_build_shared_rules(tier),
            long_term_context=context,
            experience_constraint=experience_constraint,
            capability_summary=capability,
        )
        if _PROFESSIONAL_KEYWORDS.search(query):
            prompt += _PROFESSIONAL_OVERRIDE
        return prompt

    def build_tool_policy_review_prompt(
        self,
        *,
        user_input: str,
        step,
        tool_name: str,
        tool_args: dict,
        capability_desc: str,
    ) -> list[dict]:
        step_id = getattr(step, "step_id", "S?")
        objective = getattr(step, "objective", "")
        expected_output = getattr(step, "expected_output", "")
        content = (
            f"[User Question]\n{user_input}\n\n"
            f"[Current Step]\n{step_id} | {objective}\n"
            f"[Step Expectation]\n{expected_output}\n\n"
            f"[Tool Name]\n{tool_name}\n\n"
            f"[Tool Args]\n{tool_args}\n\n"
            f"[Tool Capability]\n{capability_desc}\n"
        )
        system = _TOOL_POLICY_REVIEW_SYSTEM.format(current_datetime=self._local_datetime_str())
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    def build_summary_prompt(self, messages: list[dict]) -> list[dict]:
        """Build messages for session summary generation."""
        conversation = _format_conversation(messages)
        return [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": conversation},
        ]

    def build_fact_extraction_prompt(self, messages: list[dict]) -> list[dict]:
        """Build messages for fact extraction from conversation."""
        conversation = _format_conversation(messages)
        return [
            {"role": "system", "content": _FACT_EXTRACTION_SYSTEM},
            {"role": "user", "content": conversation},
        ]

    def build_preference_extraction_prompt(self, messages: list[dict]) -> list[dict]:
        """Build messages for implicit preference extraction from conversation."""
        conversation = _format_conversation(messages)
        return [
            {"role": "system", "content": _PREFERENCE_EXTRACTION_SYSTEM},
            {"role": "user", "content": conversation},
        ]

    def build_behavior_extraction_prompt(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Build prompt for L1 behavior signal extraction."""
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
            if m.get("role") in ("user", "assistant")
        )
        return [
            {"role": "system", "content": _BEHAVIOR_EXTRACTION_SYSTEM},
            {"role": "user", "content": conversation},
        ]


def _format_conversation(messages: list[dict], max_chars: int = 2000) -> str:
    """Format conversation messages into a readable string, truncated to max_chars."""
    parts = []
    total = 0
    for m in messages:
        content = m.get("content", "")
        if m.get("role") == "tool":
            continue
        role = "User" if m["role"] == "user" else "Assistant"
        line = f"{role}: {content}"
        if total + len(line) > max_chars:
            remaining = max_chars - total
            if remaining > 20:
                parts.append(line[:remaining] + "...")
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


def inject_prior_reasoning(
    messages: list[dict], ctx, step: int,
) -> list[dict]:
    """Inject prior reasoning chain summary into messages for multi-step continuity."""
    if step == 0 or not ctx.reasoning_chain:
        return messages

    window = ctx.reasoning_chain[-3:]
    lines = []
    for r in window:
        tools_str = ", ".join(r["tools"]) or "none"
        ev_parts = [e["summary"][:150] for e in r.get("evidence", [])[:2]]
        ev_str = "; ".join(ev_parts) if ev_parts else "no evidence yet"
        lines.append(
            f"[Step {r['step']}] Reasoning: {r['think'][:300]}\n"
            f"  Tools: {tools_str} -> {ev_str}"
        )

    prior = "\n".join(lines)
    injection = {"role": "system", "content": f"## Your Prior Reasoning\n{prior}\n\nContinue from here."}

    result = list(messages)
    insert_idx = 1 if result and result[0].get("role") == "system" else 0
    result.insert(insert_idx, injection)
    return result
