"""Screenshot tool — capture the screen and return the image path for VLM analysis."""

import subprocess
import tempfile
import time
from pathlib import Path

from . import ToolCapability, tool


def _validate_screenshot(args: dict) -> tuple[bool, str]:
    region = str(args.get("region", "full")).strip().lower()
    if region not in {"full", "window"}:
        return False, "region must be 'full' or 'window'"
    return True, "ok"


@tool(
    name="screenshot",
    description="Capture a screenshot and return the saved image path.",
    risk_level="medium",
    requires_confirmation=True,
    capability=ToolCapability(
        data_classification="sensitive",
        network_access=False,
        filesystem_access=True,
        requires_user_presence=True,
        max_output_chars=360,
        latency_tier="medium",
        failure_modes=("permission_denied",),
    ),
    validator=_validate_screenshot,
    parameters={
        "properties": {
            "region": {
                "type": "string",
                "description": "Optional capture region: 'full' or 'window'. Default: full",
            }
        }
    },
)
async def screenshot(region: str = "full", **kwargs) -> str:
    """Take a screenshot and return the file path."""
    tmp_dir = Path(tempfile.gettempdir()) / "liagent_screenshots"
    tmp_dir.mkdir(exist_ok=True)

    # Clean old screenshots (keep last 10)
    existing = sorted(tmp_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
    for old in existing[:-10]:
        old.unlink(missing_ok=True)

    path = tmp_dir / f"screen_{int(time.time())}.png"

    region = region if region in ("full", "window") else "full"
    cmd = ["screencapture", "-x"]  # -x = no sound
    if region == "window":
        cmd.append("-w")  # interactive window capture
    cmd.append(str(path))

    result = subprocess.run(cmd, capture_output=True, timeout=10)
    if result.returncode != 0 or not path.exists():
        return "[Error] Screenshot failed. Ensure screen recording permission is granted."

    return f"[Screenshot saved] {path}"
