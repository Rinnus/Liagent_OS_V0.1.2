"""Tool call parsing — extract, strip, and normalize tool calls from LLM output.

Supports JSON-in-XML (VLMs), GLM-4.7-Flash arg_key/arg_value, and Qwen3-Coder
native XML format. Core parsing is delegated to engine.tool_format parsers.
"""

import json
import re

from ..engine.tool_format import get_default_composite

# Shared composite parser instance (JSON-in-XML + GLM47 + Qwen3-Coder)
_COMPOSITE = get_default_composite()

# Regex constants used by extract_tool_call_block
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
TOOL_CALL_NATIVE_STRIP_RE = re.compile(
    r"<tool_call>\s*<function=\w+>.*?</function>\s*</tool_call>",
    re.DOTALL,
)
RAW_TOOL_CALL_PREFIX_RE = re.compile(r"^\s*[A-Za-z_]\w*\s*\(\s*(\{|[A-Za-z_]\w*\s*=)", re.DOTALL)
RAW_TOOL_CALL_LINE_RE = re.compile(r"(^|\n)\s*[A-Za-z_]\w*\s*\(\s*(\{|[A-Za-z_]\w*\s*=)", re.DOTALL)

# Observation safety patterns
_UNSAFE_OBS_PATTERNS = [
    re.compile(r"</?tool_call>", re.IGNORECASE),
    re.compile(r"ignore (all|previous|system)", re.IGNORECASE),
    re.compile(r"</?observation>", re.IGNORECASE),
]


# ── Parsers (delegated to tool_format.py) ──────────────────────────────

def _parse_tool_call_lenient(text: str) -> dict | None:
    """Lenient parser: tries all format parsers with lenient mode."""
    return _COMPOSITE.parse_lenient(text)


def parse_all_tool_calls(text: str) -> list[dict]:
    """Extract all tool calls from a response in lexical order."""
    if not text:
        return []

    calls: list[dict] = []
    for m in TOOL_CALL_BLOCK_RE.finditer(text):
        block = m.group(0)
        parsed = _COMPOSITE.parse(block) or _COMPOSITE.parse_lenient(block)
        if not parsed:
            continue
        name = str(parsed.get("name", "")).strip()
        if not name:
            continue
        args = parsed.get("args")
        calls.append({
            "name": name,
            "args": args if isinstance(args, dict) else {},
        })

    if calls:
        return calls

    # Fallback for formats that may not render explicit XML blocks.
    single = _COMPOSITE.parse(text) or _COMPOSITE.parse_lenient(text)
    if not single:
        return []
    name = str(single.get("name", "")).strip()
    if not name:
        return []
    args = single.get("args")
    return [{
        "name": name,
        "args": args if isinstance(args, dict) else {},
    }]


# ── Strippers (delegated) ──────────────────────────────────────────────

def strip_any_tool_call(text: str) -> str:
    """Remove any tool_call block (all formats) from text."""
    return _COMPOSITE.strip(text)


def extract_tool_call_block(full_response: str) -> str | None:
    """Extract the raw tool_call XML block from a full response for memory storage."""
    m = TOOL_CALL_BLOCK_RE.search(full_response) or TOOL_CALL_NATIVE_STRIP_RE.search(full_response)
    if m:
        return m.group()
    stripped = full_response.strip()
    parsed = _COMPOSITE.parse(stripped)
    if parsed and RAW_TOOL_CALL_PREFIX_RE.match(stripped):
        return stripped
    return None


def contains_tool_call_syntax(text: str) -> bool:
    """Detect likely tool-call syntax, including bare raw-call fallbacks."""
    if not text:
        return False
    if "<tool_call>" in text or "<function=" in text:
        return True
    return bool(RAW_TOOL_CALL_LINE_RE.search(text))


# ── Observation helpers ─────────────────────────────────────────────────

def sanitize_observation(observation: str) -> str:
    """Remove unsafe patterns from tool observations."""
    text = observation
    for pat in _UNSAFE_OBS_PATTERNS:
        text = pat.sub("", text)
    return text[:2000]


def tool_call_signature(tool_name: str, tool_args: dict) -> str:
    """Stable signature for deduplicating repeated tool calls in one turn."""
    try:
        args = json.dumps(tool_args or {}, ensure_ascii=False, sort_keys=True)
    except Exception:
        args = str(tool_args or {})
    return f"{tool_name}:{args}"


def resolve_context_vars(args: dict, context_vars: dict[str, str]) -> dict:
    """Replace $var_name references in tool args with context variable values."""
    if not context_vars:
        return args
    resolved = {}
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("$") and v[1:] in context_vars:
            resolved[k] = context_vars[v[1:]]
        else:
            resolved[k] = v
    return resolved
