"""File writing tool — writes text files to the cwork sandbox."""

from . import ToolCapability, tool
from ._path_security import _validate_cwork_path, get_cwork_root_str

_MAX_CONTENT_CHARS = 50000


def _validate_write_file(args: dict) -> tuple[bool, str]:
    path = str(args.get("path", "")).strip()
    ok, reason, _ = _validate_cwork_path(path)
    if not ok:
        return False, reason
    content = args.get("content")
    if content is None or not str(content).strip():
        return False, "content is required"
    if len(str(content)) > _MAX_CONTENT_CHARS:
        return False, f"content too long (max {_MAX_CONTENT_CHARS} chars)"
    return True, "ok"


@tool(
    name="write_file",
    description="Write file content inside the configured cwork sandbox. User confirmation required.",
    risk_level="medium",
    requires_confirmation=True,
    capability=ToolCapability(
        filesystem_access=True,
        max_output_chars=200,
        idempotent=False,
        failure_modes=("permission_denied",),
    ),
    validator=_validate_write_file,
    parameters={
        "properties": {
            "path": {
                "type": "string",
                "description": f"File path (must be inside cwork: {get_cwork_root_str()})",
            },
            "content": {
                "type": "string",
                "description": "Text content to write",
            },
        },
        "required": ["path", "content"],
    },
)
async def write_file(path: str, content: str, **kwargs) -> str:
    """Write UTF-8 text to a file in the cwork sandbox."""
    # First validation
    ok, reason, resolved = _validate_cwork_path(path)
    if not ok or resolved is None:
        return f"[Path error] {reason}"

    try:
        # Create parent directories
        resolved.parent.mkdir(parents=True, exist_ok=True)

        # TOCTOU defense: re-resolve the already-normalized path before write
        # (using raw `path` would resolve relative to CWD, not cwork root)
        resolved2 = resolved.resolve()
        ok2, reason2, _ = _validate_cwork_path(str(resolved2))
        if not ok2:
            return f"[Path error] TOCTOU check failed: {reason2}"

        resolved2.write_text(content, encoding="utf-8")
        byte_count = len(content.encode("utf-8"))
    except Exception as e:
        return f"[Write error] {e}"

    return f"[Write success] {resolved.name} ({byte_count} bytes)"
