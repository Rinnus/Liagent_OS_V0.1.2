"""Directory listing tool — lists files in the cwork sandbox."""

from . import ToolCapability, tool
from ._path_security import _validate_cwork_path, get_cwork_root_str


def _validate_list_dir(args: dict) -> tuple[bool, str]:
    path = str(args.get("path", "")).strip()
    if not path:
        path = get_cwork_root_str()
    ok, reason, _ = _validate_cwork_path(path)
    return ok, reason


@tool(
    name="list_dir",
    description="List files and subdirectories. Leave `path` empty to list the cwork root.",
    risk_level="low",
    capability=ToolCapability(
        filesystem_access=True,
        max_output_chars=2000,
    ),
    validator=_validate_list_dir,
    parameters={
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path (defaults to cwork root)",
            },
        },
        "required": [],
    },
)
async def list_dir(path: str = "", **kwargs) -> str:
    """List files and subdirectories in the cwork sandbox."""
    if not path or not path.strip():
        path = get_cwork_root_str()

    ok, reason, resolved = _validate_cwork_path(path)
    if not ok or resolved is None:
        return f"[Path error] {reason}"

    if not resolved.exists():
        return f"[Directory not found] {resolved}"

    if not resolved.is_dir():
        return f"[Not a directory] {resolved}"

    try:
        entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except PermissionError:
        return f"[Permission error] Cannot read directory: {resolved}"

    if not entries:
        return "[Empty directory]"

    lines = []
    for entry in entries[:100]:  # Cap at 100 entries
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            lines.append(f"  {entry.name}/")
        else:
            try:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
            except OSError:
                size_str = "?"
            lines.append(f"  {entry.name}  ({size_str})")

    if not lines:
        return "[Empty directory]"

    header = str(resolved) + "/"
    return header + "\n" + "\n".join(lines)
