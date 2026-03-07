"""Natural language → structured autonomous task using 30B Coder."""

import json
import re
from typing import TYPE_CHECKING

from ..logging import get_logger

if TYPE_CHECKING:
    from ..engine.engine_manager import EngineManager

_log = get_logger("task_parser")

_PARSE_PROMPT = """\
You are a task parser. Convert a user's natural-language task description into structured JSON.

## Key distinction: one-shot vs recurring

Distinguish these modes:
- "in X minutes" / "in X hours" / "later" -> one-shot delayed task (`once`)
- "every X minutes" / "every day" / "every week" -> recurring scheduled task (`cron`)

Examples:
- "remind me to drink water in 3 minutes" -> once, delay_seconds=180
- "remind me every 3 minutes" -> cron, */3 * * * *
- "check GOOG stock in 5 minutes and send me the result" -> once, delay_seconds=300
- "search news every day at 8am" -> cron, 0 8 * * *

## trigger_type

- "once": one-shot delay, trigger_config = {"delay_seconds": <seconds>}
  - "in 3 minutes" -> delay_seconds = 180
  - "in 1 hour" -> delay_seconds = 3600
  - "in 30 minutes" -> delay_seconds = 1800
  - "in 10 seconds" -> delay_seconds = 10
- "cron": recurring task, trigger_config = {"schedule": "<cron expression>"}
  - cron format: minute hour day month weekday (5 fields)
  - "every day at 8" -> "0 8 * * *"
  - "every hour" -> "0 * * * *"
  - "every Monday" -> "0 9 * * 1"
  - "every 5 minutes" -> "*/5 * * * *"

## prompt_template

`prompt_template` is the full prompt sent to the AI when the task executes.
It must include enough context for autonomous execution.

Examples:
- Reminder: "Remind the user to drink water. Maintaining hydration is important."
- Query: "Search the latest Google (GOOG) stock price and briefly analyze today's movement."
- News: "Search the latest AI news today and summarize the top three items within 500 words."

Output JSON only:
```json
{
  "name": "Short task name",
  "trigger_type": "once",
  "trigger_config": {"delay_seconds": 180},
  "prompt_template": "Full execution prompt"
}
```

User input:"""


async def parse_task_description(
    engine: "EngineManager",
    text: str,
) -> dict | None:
    """Parse a natural language task description into a structured task config.

    Returns a dict with keys: name, trigger_type, trigger_config, prompt_template.
    Returns None if parsing fails.
    """
    messages = [
        {"role": "system", "content": _PARSE_PROMPT},
        {"role": "user", "content": text},
    ]

    try:
        raw = await engine.generate_reasoning(
            messages, max_tokens=400, temperature=0.1,
            enable_thinking=False,
        )
    except Exception as e:
        _log.error("task_parser", e, action="generate_reasoning")
        return None

    if not raw:
        return None

    # Extract JSON from response (may be wrapped in ```json blocks)
    raw = raw.strip()
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        # Try to find bare JSON object
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            raw = raw[brace_start : brace_end + 1]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        _log.warning(f"task_parser: failed to parse JSON from: {raw[:200]}")
        return None

    # Validate required fields
    required = {"name", "trigger_type", "prompt_template"}
    if not required.issubset(result.keys()):
        _log.warning(f"task_parser: missing fields: {required - result.keys()}")
        return None

    # Whitelist trigger_type — only once and cron are executable
    _ALLOWED_TRIGGER_TYPES = {"once", "cron"}
    if result["trigger_type"] not in _ALLOWED_TRIGGER_TYPES:
        _log.warning(f"task_parser: unsupported trigger_type '{result['trigger_type']}', "
                     f"allowed: {_ALLOWED_TRIGGER_TYPES}")
        return None

    # Ensure trigger_config is a dict
    if "trigger_config" not in result or not isinstance(result["trigger_config"], dict):
        result["trigger_config"] = {}

    # Validate by trigger type
    if result["trigger_type"] == "cron":
        schedule = result.get("trigger_config", {}).get("schedule", "")
        if not schedule:
            _log.warning("task_parser: cron task missing schedule")
            return None
        parts = schedule.strip().split()
        if len(parts) != 5:
            _log.warning(f"task_parser: invalid cron schedule: {schedule}")
            return None

    elif result["trigger_type"] == "once":
        delay = result.get("trigger_config", {}).get("delay_seconds")
        if delay is None:
            _log.warning("task_parser: once task missing delay_seconds")
            return None
        try:
            delay = int(delay)
            if delay < 1:
                delay = 60
            result["trigger_config"]["delay_seconds"] = delay
        except (TypeError, ValueError):
            _log.warning(f"task_parser: invalid delay_seconds: {delay}")
            return None

    return result
