"""Image description tool — passes an image file to VLM for analysis."""

from . import ToolCapability, tool
from ._path_security import _validate_cwork_path

_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _validate_describe_image(args: dict) -> tuple[bool, str]:
    path = str(args.get("path", "")).strip()
    ok, reason, resolved = _validate_cwork_path(path)
    if not ok:
        return False, reason
    if resolved is not None:
        ext = resolved.suffix.lower()
        if ext not in _SUPPORTED_EXTS:
            return False, f"unsupported image format: {ext} (supported: {', '.join(sorted(_SUPPORTED_EXTS))})"
    return True, "ok"


@tool(
    name="describe_image",
    description="Describe or analyze an image file inside cwork.",
    risk_level="low",
    capability=ToolCapability(
        filesystem_access=True,
        max_output_chars=360,
    ),
    validator=_validate_describe_image,
    parameters={
        "properties": {
            "path": {
                "type": "string",
                "description": "Image file path in cwork (supports png/jpg/gif/bmp/webp)",
            },
        },
        "required": ["path"],
    },
)
async def describe_image(path: str, **kwargs) -> str:
    """Return the image path for VLM analysis (same pattern as screenshot)."""
    ok, reason, resolved = _validate_cwork_path(path)
    if not ok or resolved is None:
        return f"[Path error] {reason}"

    if not resolved.exists():
        return f"[File not found] {resolved}"

    if not resolved.is_file():
        return f"[Not a file] {resolved}"

    return f"[Image loaded] {resolved}"
