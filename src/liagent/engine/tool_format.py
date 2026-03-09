"""Pluggable tool-call format parsers — decouple model-specific formats from agent logic.

Each parser knows how to:
1. parse(text) → extract a tool call dict
2. parse_lenient(text) → tolerant parse for degenerate output
3. strip(text) → remove tool call XML from response
4. format_schemas(tools) → convert tool definitions for model consumption
"""

from __future__ import annotations

import ast
import json
import re
from typing import Protocol


class ToolCallParser(Protocol):
    """Protocol for pluggable tool-call format handling."""

    def parse(self, text: str) -> dict | None:
        """Extract tool call from response text. Returns {"name": ..., "args": ...} or None."""
        ...

    def parse_lenient(self, text: str) -> dict | None:
        """Lenient parse for malformed/incomplete output."""
        ...

    def strip(self, text: str) -> str:
        """Remove tool call block(s) from text."""
        ...

    def format_schemas(self, tools: list) -> list[dict] | str:
        """Format tool schemas for inclusion in model prompt or API call."""
        ...


# ── Qwen3-Coder native XML format ──────────────────────────────────────

_NATIVE_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_NATIVE_STRIP_RE = re.compile(
    r"<tool_call>\s*<function=\w+>.*?</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)


class Qwen3CoderFormat:
    """Qwen3-Coder native XML: <tool_call><function=name><parameter=key>val</parameter></function></tool_call>"""

    def parse(self, text: str) -> dict | None:
        m = _NATIVE_RE.search(text)
        if not m:
            return None
        name = m.group(1)
        body = m.group(2)
        args = {}
        for pm in _PARAM_RE.finditer(body):
            args[pm.group(1)] = pm.group(2).strip()
        return {"name": name, "args": args}

    def parse_lenient(self, text: str) -> dict | None:
        m = re.search(
            r"<tool_call>\s*<function=(\w+)>(.*?)(?:</function>|<tool_call>|\Z)",
            text, re.DOTALL,
        )
        if m:
            name = m.group(1)
            body = m.group(2)
            args = {}
            for pm in _PARAM_RE.finditer(body):
                args[pm.group(1)] = pm.group(2).strip()
            if name:
                return {"name": name, "args": args}
        return None

    def strip(self, text: str) -> str:
        return _NATIVE_STRIP_RE.sub("", text).strip()

    def format_schemas(self, tools: list) -> list[dict]:
        """Return native tool schemas for Qwen3-Coder's tools= parameter."""
        return tools  # Already in native format


# ── JSON-in-XML format ──────────────────────────────────────────────────

_JSON_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


class JsonInXmlFormat:
    """JSON-in-XML: <tool_call>{"name": "...", "args": {...}}</tool_call>"""

    def parse(self, text: str) -> dict | None:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return None

    def parse_lenient(self, text: str) -> dict | None:
        m = re.search(
            r"<tool_call>\s*(\{.+?)(?:\n\s*<tool_call>|\n\s*$|\Z)",
            text, re.DOTALL,
        )
        if not m:
            return None
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            for end in range(len(candidate), 0, -1):
                if candidate[end - 1] == "}":
                    try:
                        return json.loads(candidate[:end])
                    except json.JSONDecodeError:
                        continue
        return None

    def strip(self, text: str) -> str:
        return _BLOCK_RE.sub("", text).strip()

    def format_schemas(self, tools: list) -> str:
        """Return JSON schema string for embedding in system prompt."""
        return json.dumps(tools, ensure_ascii=False, indent=2)


# ── OpenAI function_call format ─────────────────────────────────────────

class OpenAIFormat:
    """OpenAI function_call JSON format (for API-based models)."""

    def parse(self, text: str) -> dict | None:
        # OpenAI responses are usually structured, not embedded in XML
        try:
            data = json.loads(text)
            if "function_call" in data:
                fc = data["function_call"]
                args = fc.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return {"name": fc.get("name", ""), "args": args}
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return None

    def parse_lenient(self, text: str) -> dict | None:
        return self.parse(text)

    def strip(self, text: str) -> str:
        return text  # No XML to strip

    def format_schemas(self, tools: list) -> list[dict]:
        """Return OpenAI-style function definitions."""
        return tools


# ── GLM-4.7-Flash format ──────────────────────────────────────────────

_GLM47_RE = re.compile(
    r"<tool_call>([a-zA-Z_]\w*)\s*((?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)</tool_call>",
    re.DOTALL,
)
_GLM47_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)
_FUNCTION_CALLS_BLOCK_RE = re.compile(
    r"<function_calls>\s*.*?</function_calls>", re.DOTALL | re.IGNORECASE
)
_INVOKE_RE = re.compile(
    r"<invoke\s+name=[\"']?([A-Za-z_]\w*)[\"']?\s*>(.*?)</invoke>",
    re.DOTALL | re.IGNORECASE,
)
_INVOKE_LENIENT_RE = re.compile(
    r"<invoke\s+name=[\"']?([A-Za-z_]\w*)[\"']?\s*>(.*?)(?:</invoke>|</function_calls>|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_NAMED_PARAMETER_RE = re.compile(
    r"<parameter\s+name=[\"']?([A-Za-z_]\w*)[\"']?\s*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_raw_json_call(text: str, *, strict: bool) -> tuple[str, str] | None:
    """Extract bare tool-call syntax like: web_search({"query": "x"})."""
    src = text.strip() if strict else text
    starts: list[int]
    if strict:
        starts = [0]
    else:
        starts = [m.start(1) for m in re.finditer(r"(^|\n)\s*([A-Za-z_]\w*)\s*\(", src)]

    for start in starts:
        segment = src[start:].lstrip() if not strict else src
        m = re.match(r"([A-Za-z_]\w*)\s*\(", segment)
        if not m:
            continue
        name = m.group(1)
        pos = m.end()
        while pos < len(segment) and segment[pos].isspace():
            pos += 1
        if pos >= len(segment) or segment[pos] != "{":
            continue

        brace_depth = 0
        in_string = False
        escape = False
        end_obj = -1
        for idx in range(pos, len(segment)):
            ch = segment[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    end_obj = idx
                    break
        if end_obj < 0:
            continue

        json_blob = segment[pos:end_obj + 1]
        tail = segment[end_obj + 1:].lstrip()
        if not tail.startswith(")"):
            continue
        tail = tail[1:].strip()
        if strict and tail:
            continue
        return name, json_blob
    return None


class GLM47Format:
    """GLM-4.7-Flash: <tool_call>func_name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"""

    def parse(self, text: str) -> dict | None:
        m = _GLM47_RE.search(text)
        if not m:
            return None
        name = m.group(1).strip()
        body = m.group(2)
        args = {}
        for am in _GLM47_ARG_RE.finditer(body):
            args[am.group(1).strip()] = am.group(2).strip()
        return {"name": name, "args": args}

    def parse_lenient(self, text: str) -> dict | None:
        # Try without closing </tool_call>
        m = re.search(
            r"<tool_call>([a-zA-Z_]\w*)\s*((?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)(?:</tool_call>|\Z)",
            text, re.DOTALL,
        )
        if m:
            name = m.group(1).strip()
            body = m.group(2)
            args = {}
            for am in _GLM47_ARG_RE.finditer(body):
                args[am.group(1).strip()] = am.group(2).strip()
            if name:
                return {"name": name, "args": args}
        return None

    def strip(self, text: str) -> str:
        return _BLOCK_RE.sub("", text).strip()

    def format_schemas(self, tools: list) -> list[dict]:
        """Return native tool schemas — GLM injects format via chat_template tools= parameter."""
        return tools


class FunctionCallsInvokeFormat:
    """Moonshot-style XML: <function_calls><invoke name=\"x\"><parameter name=\"k\">v</parameter></invoke></function_calls>."""

    @staticmethod
    def _parse_invoke(match: re.Match[str] | None) -> dict | None:
        if match is None:
            return None
        name = str(match.group(1) or "").strip()
        if not name:
            return None
        body = str(match.group(2) or "")
        args: dict[str, str] = {}
        for pm in _NAMED_PARAMETER_RE.finditer(body):
            args[str(pm.group(1)).strip()] = str(pm.group(2)).strip()
        return {"name": name, "args": args}

    def parse(self, text: str) -> dict | None:
        return self._parse_invoke(_INVOKE_RE.search(text))

    def parse_lenient(self, text: str) -> dict | None:
        parsed = self._parse_invoke(_INVOKE_RE.search(text))
        if parsed is not None:
            return parsed
        return self._parse_invoke(_INVOKE_LENIENT_RE.search(text))

    def strip(self, text: str) -> str:
        text = _FUNCTION_CALLS_BLOCK_RE.sub("", text)
        text = re.sub(
            r"\s*<invoke\s+name=[\"']?[A-Za-z_]\w*[\"']?\s*>.*?</invoke>\s*",
            "\n",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def format_schemas(self, tools: list) -> str:
        return json.dumps(tools, ensure_ascii=False, indent=2)


class RawJsonCallFormat:
    """Bare call syntax: web_search({"query": "test"})."""

    def parse(self, text: str) -> dict | None:
        extracted = _extract_raw_json_call(text, strict=True)
        if not extracted:
            return None
        name, json_blob = extracted
        try:
            args = json.loads(json_blob)
        except json.JSONDecodeError:
            return None
        if not isinstance(args, dict):
            return None
        return {"name": name, "args": args}

    def parse_lenient(self, text: str) -> dict | None:
        extracted = _extract_raw_json_call(text, strict=False)
        if not extracted:
            return None
        name, json_blob = extracted
        try:
            args = json.loads(json_blob)
        except json.JSONDecodeError:
            return None
        if not isinstance(args, dict):
            return None
        return {"name": name, "args": args}

    def strip(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            if self.parse(line.strip()) is not None:
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def format_schemas(self, tools: list) -> str:
        return json.dumps(tools, ensure_ascii=False, indent=2)


def _parse_keyword_call_expr(text: str) -> dict | None:
    """Parse bare keyword-call syntax like: web_search(query=\"x\")."""
    try:
        expr = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return None
    call = expr.body
    if not isinstance(call, ast.Call):
        return None
    if not isinstance(call.func, ast.Name):
        return None
    if call.args:
        return None
    args: dict[str, object] = {}
    for kw in call.keywords:
        if kw.arg is None:
            return None
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            return None
    return {"name": call.func.id, "args": args}


class RawKeywordCallFormat:
    """Bare keyword-call syntax: web_search(query=\"test\", timelimit=\"d\")."""

    def parse(self, text: str) -> dict | None:
        return _parse_keyword_call_expr(text)

    def parse_lenient(self, text: str) -> dict | None:
        parsed = _parse_keyword_call_expr(text)
        if parsed is not None:
            return parsed
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            parsed = _parse_keyword_call_expr(candidate)
            if parsed is not None:
                return parsed
        return None

    def strip(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            if self.parse(line.strip()) is not None:
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def format_schemas(self, tools: list) -> str:
        return json.dumps(tools, ensure_ascii=False, indent=2)


# ── Composite parser (tries multiple formats in sequence) ───────────────

class CompositeFormat:
    """Try multiple parsers in order. Used when the model format is unknown or mixed."""

    def __init__(self, parsers: list[ToolCallParser] | None = None):
        self.parsers = parsers or [
            JsonInXmlFormat(),
            GLM47Format(),
            Qwen3CoderFormat(),
            FunctionCallsInvokeFormat(),
            RawJsonCallFormat(),
            RawKeywordCallFormat(),
        ]

    def parse(self, text: str) -> dict | None:
        for p in self.parsers:
            result = p.parse(text)
            if result is not None:
                return result
        return None

    def parse_lenient(self, text: str) -> dict | None:
        for p in self.parsers:
            result = p.parse_lenient(text)
            if result is not None:
                return result
        return None

    def strip(self, text: str) -> str:
        for p in self.parsers:
            text = p.strip(text)
        return text

    def format_schemas(self, tools: list) -> list[dict] | str:
        """Use first parser's format."""
        if self.parsers:
            return self.parsers[0].format_schemas(tools)
        return tools


# ── Factory ─────────────────────────────────────────────────────────────

_MODEL_FAMILY_FORMATS: dict[str, type] = {
    "glm47": GLM47Format,
    "qwen3-coder": Qwen3CoderFormat,
    "qwen3-vl": JsonInXmlFormat,
    "openai": OpenAIFormat,
    "gemini": OpenAIFormat,
    "claude": OpenAIFormat,
    "llama": JsonInXmlFormat,       # default to JSON-in-XML
    "deepseek": JsonInXmlFormat,    # default to JSON-in-XML
}

_PROTOCOL_FORMATS: dict[str, type] = {
    "openai_function": OpenAIFormat,
    "native_xml": GLM47Format,
    "json_xml": JsonInXmlFormat,
}


def get_parser_for_family(model_family: str) -> ToolCallParser:
    """Get the appropriate parser for a model family."""
    cls = _MODEL_FAMILY_FORMATS.get(model_family)
    if cls:
        return cls()
    return CompositeFormat()


def get_parser_for_protocol(tool_protocol: str, *, model_family: str = "") -> ToolCallParser:
    """Get parser/formatter by declared tool protocol.

    Falls back to model-family mapping when protocol is empty or "auto".
    """
    proto = str(tool_protocol or "").strip().lower()
    if proto in {"", "auto"}:
        return get_parser_for_family(model_family)
    cls = _PROTOCOL_FORMATS.get(proto)
    if cls:
        return cls()
    return get_parser_for_family(model_family)


def get_default_composite() -> CompositeFormat:
    """Get the default composite parser (JSON-in-XML + GLM47 + Qwen3-Coder)."""
    return CompositeFormat()
