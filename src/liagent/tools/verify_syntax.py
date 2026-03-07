"""Syntax verification tool — checks Python files for syntax errors via ast.parse."""

import ast

from . import ToolCapability, tool
from ._path_security import _validate_cwork_path


def _validate_verify_syntax(args: dict) -> tuple[bool, str]:
    path = str(args.get("path", "")).strip()
    if not path:
        return False, "path is required"
    ok, reason, _ = _validate_cwork_path(path)
    if not ok:
        return False, reason
    return True, "ok"


@tool(
    name="verify_syntax",
    description="Check Python syntax and return OK or a concrete syntax error.",
    risk_level="low",
    capability=ToolCapability(
        data_classification="internal",
        filesystem_access=True,
        max_output_chars=800,
    ),
    validator=_validate_verify_syntax,
    parameters={
        "properties": {
            "path": {
                "type": "string",
                "description": "Python file path to verify (must be inside cwork)",
            },
        },
        "required": ["path"],
    },
)
async def verify_syntax(path: str, **kwargs) -> str:
    """Check Python file syntax using ast.parse."""
    ok, reason, resolved = _validate_cwork_path(path)
    if not ok:
        return f"[Error] {reason}"

    if not resolved.exists():
        return f"[Error] File not found: {resolved.name}"

    if not resolved.suffix == ".py":
        return f"[Error] Only .py files are supported, got: {resolved.suffix}"

    try:
        source = resolved.read_text(encoding="utf-8")
    except Exception as e:
        return f"[Error] Failed to read file: {e}"

    try:
        ast.parse(source, filename=str(resolved.name))
        return f"OK - {resolved.name} syntax is valid ({len(source)} chars, {source.count(chr(10))+1} lines)"
    except SyntaxError as e:
        return (
            f"[Syntax error] {resolved.name}:{e.lineno}:{e.offset}\n"
            f"  {e.msg}\n"
            f"  {e.text.rstrip() if e.text else ''}"
        )
